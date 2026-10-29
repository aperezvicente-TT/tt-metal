# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK mHC phases: a HOST REPLICA of the exact kernel dataflow (CPU, torch float64) checked against the fused7a
arithmetic (tt/mhc_pre.reference_step + tt/mhc_post.reference_step) on L4's real hc weights and the golden residual
rows (goldens layer_io streams_in; random fallback). It reproduces, per member (row block b, column slab j) and per
root, what kernels/pmk/mhc_{stream,sync,compute}.cpp do with the SAME index formulas:
  * DRAM page orders: residual tile (rb, s, c) = rb * 512 + s * 128 + c; hn / y (rb, c) = rb * 128 + c; fn^T (s, c)
    = s * 128 + c; w tile c; the member's CB_X index s * CPC + j for column tile c0 + j;
  * the member's (M, S) partial over its 64 tiles, the root's sum over the 8 slabs, z = M / rms * scale + bias,
    pre / post / comb, diag(pre_s) collapse per column tile, S2, hn = h * 1/rms(h) * w;
  * the root's K_s / L_s packing (post_s | C[0][s] | C[1][s] | C[2][s] at columns 0 / 1 / 16 / 17, C[3][s] at
    column 0 of L_s) and the post pass out_s = y K[0] + x_0 K[1] + x_1 K[16] + x_2 K[17] + x_3 L[0] per column tile,
    written back in place (resident) or to the DRAM pages (streamed: group g's blocks g, g + 8, ...);
  * both sublayers chained; NRB 1 / 2 / 4 (SP 256 resident, SP 512 / 1,024 streamed) and an all-zero row block at
    the k = 1 boundary (block 9: the eps paths).
Also prints the post-coefficient range (a saturated 2 sigmoid would make a bitwise 'post' uninformative).
Run: python3 models/autoports/zai_org_glm_5_3_flash/tests/s16/pmk_mhc_replica.py [--random]"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5)))
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as M  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import reference_step as post_ref  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import reference_step as pre_ref  # noqa: E402

T32, NS, D, CT, P, CPC = 32, 4, 4096, 128, M.P, M.CPC
EPS = dict(hc_eps=1e-6, x_eps=1e-5, h_eps=1e-5, iters=20)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def tiles_of(t2d: torch.Tensor) -> torch.Tensor:
    """[R, C] -> [R/32, C/32, 32, 32] tiles (row tile, column tile)"""
    r, c = t2d.shape
    return t2d.reshape(r // T32, T32, c // T32, T32).permute(0, 2, 1, 3)


def flat_pages_residual(x: torch.Tensor) -> torch.Tensor:
    """x [SP, 4 * D] -> pages [n_rb * 512, 32, 32] in the DRAM order rb * 512 + s * 128 + c (= the TILE layout of
    [1, 1, SP, 16384]: row tile rb, column tile s * 128 + c)"""
    return tiles_of(x).reshape(-1, T32, T32)


def flat_pages_cols(t: torch.Tensor) -> torch.Tensor:
    """[SP, D] -> pages rb * 128 + c"""
    return tiles_of(t).reshape(-1, T32, T32)


def weights(which: str, random: bool, source=None, cfg=None):
    if random:
        g = torch.Generator().manual_seed(1 if which == "attn" else 2)
        fn_t = torch.randn(NS * D, 24, generator=g) * 0.01
        scale = torch.cat([torch.full((4,), 0.5), torch.full((4,), 0.5), torch.full((16,), 0.5)])
        bias = torch.zeros(24)
        bias[8:24] = torch.eye(4).reshape(-1) * 2
        w = 1 + 0.1 * torch.randn(D, generator=g)
        return fn_t, scale, bias, w
    n = cfg.hc_mult
    s = source.host(4, f"hc_{which}_scale").float().reshape(-1)
    per_mix = torch.cat([s[0].repeat(n), s[1].repeat(n), s[2].repeat(n * n)])
    norm = {"attn": "input_layernorm.weight", "ffn": "post_attention_layernorm.weight"}[which]
    return (
        source.host(4, f"hc_{which}_fn").reshape(n * D, (2 + n) * n).float(),
        per_mix,
        source.host(4, f"hc_{which}_base").float().reshape(-1),
        source.host(4, norm).float().reshape(-1),
    )


def replica_sublayer(xp: torch.Tensor, nrb_total: int, fn_t, scale, bias, w, y: torch.Tensor, resident: bool):
    """xp: residual pages [nrb_total * 512, 32, 32] float64 (modified in place as the kernels do). Returns (hn pages
    [nrb_total * 128, 32, 32], post [SP, 4], comb [SP, 4, 4], K/L check)."""
    n_groups, nrb = M.N_GROUPS, nrb_total // M.N_GROUPS
    fn_pages = tiles_of(torch.nn.functional.pad(fn_t.double(), (0, T32 - 24))).reshape(-1, T32, T32)  # (s*128+c)
    w_pages = torch.zeros(CT, T32, T32, dtype=torch.float64)
    for c in range(CT):
        w_pages[c, 0] = w.double()[c * T32 : (c + 1) * T32]  # row 0 = w[32 c : 32 c + 32]
    hn_pages = torch.zeros(nrb_total * CT, T32, T32, dtype=torch.float64)
    post_all = torch.zeros(nrb_total * T32, NS, dtype=torch.float64)
    comb_all = torch.zeros(nrb_total * T32, NS, NS, dtype=torch.float64)
    y_pages = flat_pages_cols(y.double())
    for g in range(n_groups):  # root g
        for k in range(nrb):
            rb = g + k * n_groups  # the kernels: rb0 = g, step 8
            # ---- round 1: each member j: M partial = sum_{s, jj} x_tile(rb, s, c0 + jj) @ fn_tile(s, c0 + jj)
            Mt = torch.zeros(T32, T32, dtype=torch.float64)
            S = torch.zeros(T32, dtype=torch.float64)
            slabs = {}
            for j in range(P):
                c0 = j * CPC
                xs = torch.stack([xp[rb * NS * CT + s * CT + c0 + jj] for s in range(NS) for jj in range(CPC)])  # CB_X
                slabs[j] = xs  # index s * CPC + jj
                fns = torch.stack([fn_pages[s * CT + c0 + jj] for s in range(NS) for jj in range(CPC)])
                Mt += torch.einsum("krc,kcm->rm", xs, fns)
                S += (xs * xs).sum(dim=(0, 2))  # diag(x x^T) row sums
            inv = torch.rsqrt(S / (NS * D) + EPS["x_eps"])
            z = Mt[:, :24] * inv[:, None] * scale.double() + bias.double()
            pre = torch.sigmoid(z[:, :4]) + EPS["hc_eps"]
            post = 2 * torch.sigmoid(z[:, 4:8])
            comb = torch.softmax(z[:, 8:].reshape(T32, 4, 4), dim=-1) + EPS["hc_eps"]
            comb = comb / (comb.sum(-2, keepdim=True) + EPS["hc_eps"])
            for _ in range(EPS["iters"] - 1):
                comb = comb / (comb.sum(-1, keepdim=True) + EPS["hc_eps"])
                comb = comb / (comb.sum(-2, keepdim=True) + EPS["hc_eps"])
            post_all[rb * T32 : (rb + 1) * T32] = post
            comb_all[rb * T32 : (rb + 1) * T32] = comb
            # the root's packed K_s / L_s tiles (columns 0 / 1 / 16 / 17 and 0); comb[t, i, s] = C[i][s]
            K = torch.zeros(NS, T32, T32, dtype=torch.float64)
            L = torch.zeros(NS, T32, T32, dtype=torch.float64)
            for s in range(NS):
                K[s, :, 0], K[s, :, 1], K[s, :, 16], K[s, :, 17] = (
                    post[:, s],
                    comb[:, 0, s],
                    comb[:, 1, s],
                    comb[:, 2, s],
                )
                L[s, :, 0] = comb[:, 3, s]
            # ---- collapse / finish per member, S2 over members
            S2 = torch.zeros(T32, dtype=torch.float64)
            hs = {}
            for j in range(P):
                xs = slabs[j]
                h = torch.stack([sum(torch.diag(pre[:, s]) @ xs[s * CPC + jj] for s in range(NS)) for jj in range(CPC)])
                hs[j] = h
                S2 += (h * h).sum(dim=(0, 2))
            inv_h = torch.rsqrt(S2 / D + EPS["h_eps"])
            for j in range(P):
                c0 = j * CPC
                for jj in range(CPC):
                    hn_pages[rb * CT + c0 + jj] = hs[j][jj] * inv_h[:, None] * w_pages[c0 + jj][0][None, :]
            # ---- post per member and column tile, in place (resident) / DRAM pages (streamed): same formula
            for j in range(P):
                c0 = j * CPC
                xs = slabs[j]  # (streamed: re-read from the DRAM pages = the same values at this point)
                for jj in range(CPC):
                    yt = y_pages[rb * CT + c0 + jj]
                    outs = []
                    for s in range(NS):
                        o = (
                            yt * K[s][:, 0:1]
                            + xs[0 * CPC + jj] * K[s][:, 1:2]
                            + xs[1 * CPC + jj] * K[s][:, 16:17]
                            + xs[2 * CPC + jj] * K[s][:, 17:18]
                            + xs[3 * CPC + jj] * L[s][:, 0:1]
                        )
                        outs.append(o)
                    for s in range(NS):  # write-back after all 4 reads of the column (staging)
                        xp[rb * NS * CT + s * CT + c0 + jj] = outs[s]
    return hn_pages, post_all, comb_all


def main(random: bool):
    source = cfg = None
    if not random:
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

        cfg = hf_config()
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        rows = golden(4, 2048)["streams_in"].float().reshape(2048, NS * D)
    else:
        rows = torch.randn(2048, NS * D, generator=torch.Generator().manual_seed(7))
    ws = {w: weights(w, random, source, cfg) for w in ("attn", "ffn")}
    for sp, zero_block in ((256, None), (512, None), (1024, None), (1024, 9)):  # NRB 1 / 2 / 4; a zero block
        x0 = rows[:sp].double()
        if zero_block is not None:  # block 9 = group 1's second block (the k = 1 boundary): all-zero rows -> the
            x0 = x0.clone()  # eps path of 1/rms(x) and 1/rms(h); comb = Sinkhorn(softmax(bias)) there
            x0[zero_block * T32 : (zero_block + 1) * T32] = 0
        ys = [rows[:sp, :D].double() * 0.5, rows[:sp, D : 2 * D].double() * 0.25]  # the unit's synthetic y
        xp = flat_pages_residual(x0.clone())
        xr = x0.clone()
        for s, which in enumerate(("attn", "ffn")):
            fn_t, scale, bias, w = ws[which]
            hn_p, post_k, comb_k = replica_sublayer(xp, sp // T32, fn_t, scale, bias, w, ys[s], sp == 256)
            ref = pre_ref(xr.float(), fn_t, scale, bias, w, matmul_tf32=False, **EPS)
            hn_k = tiles_of(torch.zeros(sp, D)).reshape(-1, T32, T32)  # shape helper
            hn_k = hn_p.reshape(sp // T32, CT, T32, T32).permute(0, 2, 1, 3).reshape(sp, D)
            d_hn = float((hn_k - ref["hn"]).abs().max())
            d_post = float((post_k - ref["post"]).abs().max())
            d_comb = float((comb_k - ref["comb"]).abs().max())
            sat = float((ref["post"] - 2.0).abs().max()), float(ref["post"].min()), float(ref["post"].max())
            tag = f"sp{sp}{'' if zero_block is None else f' zero-block {zero_block}'}"
            check(
                f"{tag} {which}: replica hn vs reference (max |diff| {d_hn:.2e})",
                d_hn < 1e-6 * max(1.0, float(ref["hn"].abs().max())),
            )
            check(
                f"{tag} {which}: replica post / comb vs reference ({d_post:.2e} / {d_comb:.2e})",
                d_post < 1e-7 and d_comb < 1e-7,
            )  # (the reference takes the fp32-cast residual: ~1e-8)
            print(
                f"     post range [{sat[1]:.4f}, {sat[2]:.4f}] (distance from 2.0: {sat[0]:.2e}); comb[0] = "
                f"{ref['comb'][0].reshape(-1).tolist()[:4]}"
            )
            if which == "ffn" and zero_block is None:  # the rev <= 5 kernel defect: sublayer 1's z on hc_attn's scale /
                fa, sa, ba, _ = ws["attn"]  # bias (CB_SB loaded once per program) -- must NOT match the reference
                bad = pre_ref(xr.float(), fn_t, sa, ba, w, matmul_tf32=False, **EPS)
                d_bad = float((bad["post"] - ref["post"]).abs().max()), float((bad["comb"] - ref["comb"]).abs().max())
                check(f"{tag} ffn: sublayer 1 on sublayer 0's scale / bias is WRONG (post / comb max |diff| "
                      f"{d_bad[0]:.2e} / {d_bad[1]:.2e}: the per-sublayer SB load is load-bearing)",
                      d_bad[0] > 1e-2 or d_bad[1] > 1e-2)  # fmt: skip
            xr = post_ref(ys[s].float(), xr.reshape(sp, NS, D).float(), ref["post"], ref["comb"]).reshape(sp, NS * D)
            x_k = xp.reshape(sp // T32, NS * CT, T32, T32).permute(0, 2, 1, 3).reshape(sp, NS * D)
            d_x = float((x_k - xr).abs().max())
            check(
                f"{tag} {which}: replica residual after the post vs reference ({d_x:.2e})",
                d_x < 1e-7 * max(1.0, float(xr.abs().max())),
            )
    print("RESULT", "PASS" if not fails else f"FAIL {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main("--random" in sys.argv))
