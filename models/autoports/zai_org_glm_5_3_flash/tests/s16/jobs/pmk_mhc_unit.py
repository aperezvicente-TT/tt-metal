# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""THE ONE device bench unit of the PMK mHC phases (serve.py job, host-p300 TP2 pair = --mgd pair, watched first).
NOT to be run as-is on a device from this branch (7a review: Traced API + from_torch inside the capture); Stage 16's
integrated copy of this job is the one that runs. ARGS probe (True): the root pass taps (tt/pmk_mhc.decode_taps) of
every round vs the host planes (z, the pass at 20 / 1 / 2 iterations).
Layer L4's attention + FFN sublayers (real hc_* weights and norm weights from the tile cache) on the golden residual
rows (goldens layer_io: streams_in of the 2,048-token prompt; chip c takes rows [1,024 c, 1,024 c + 1,024) at TP2 =
SP 1,024, the streamed geometry; sp=256 runs the resident geometry on the first 256 rows) with a synthetic sublayer
output y (deterministic, the stream-0 rows scaled: realistic magnitudes), as ONE traced program:
    pre(attn) -> post(attn, y_a) -> pre(ffn) -> post(ffn, y_f)
against the fused7a op path on the same inputs: tt/mhc_pre_pf.MhcPrePf (PF_FPU) + mhc_post_prefill_7a, twice.
Criterion (per sublayer and output, every chip): hn bitwise (bf16) and post / comb bitwise, the new residual bitwise
or max |diff| <= 2e-6 (the same FPU tile order and the same SFPU FMAs: bitwise is expected; 2e-6 covers a different
accumulation order of the fp32 partial sums if the JIT schedules differently); deterministic (two replays bitwise);
then timing: one trace each, `replays` replays, us per program and us per sublayer (pre + post) — expected at
SP 1,024 streamed: fused7a's 2 x (414 + 604) = ~2,040 us per sublayer pair vs the phase program's ~750 us (estimate,
tt/pmk_mhc.Plan.estimate_us) + the DRAM x / y streams; at SP 256 resident: ~166 us per sublayer + the one-time 16 MB
residual load / store (~70 us) that the layer program does not have.
<= 10 min: build (JIT ~1 min), 2 geometries x 2 traces x `replays` (200) replays.
ARGS: layer (4), sps ([1024, 256]), replays (200), record (False), tag. GLM53_MOCK=1: dry run (no asserts)."""

from __future__ import annotations

import json
import os
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
from models.autoports.zai_org_glm_5_3_flash.tests.s14b.s14common import Traced  # Traced(ttnn, mesh, fn): compile launch, ONE capture, replays
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as M
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf import MhcPrePf, mhc_post_prefill_7a
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

mesh = MESH  # noqa: F821
A = ARGS or {}  # noqa: F821
MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(A.get("layer", 4))
SPS = [int(s) for s in A.get("sps", [1024, 256])]
REPLAYS = int(A.get("replays", 200))
TAG = A.get("tag", "pmk-mhc-unit")
REC = bool(A.get("record", False))
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
NORM = {"attn": "input_layernorm.weight", "ffn": "post_attention_layernorm.weight"}
NDEV = int(mesh.get_num_devices())
rows = []


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test="pmk_mhc_unit", layer=LAYER, tag=TAG,
             mock=MOCK, ndev=NDEV, **kw)  # fmt: skip
    rows.append(r)
    print("ROW", json.dumps(r, default=str), flush=True)
    return r


def shard_rows(t: torch.Tensor):
    """[1, 1, 2048, W] host -> per-chip rows (chip c: rows 1,024 c ..) on the mesh, or the first SP rows replicated"""
    if NDEV > 1:
        return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))  # fmt: skip
    return ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def per_chip(t):
    return [ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(t) if NDEV > 1 else [t])]


def main():
    cfg = hf_config()
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    g = golden(LAYER, 2048)["streams_in"].float().reshape(1, 1, 2048, M.N_STREAMS * M.HIDDEN)
    for sp in SPS:
        plan = M.Plan(sp_rows=sp, y_lane=False, probe=bool(A.get("probe", True)))
        rows_host = g[:, :, : sp * NDEV] if sp * NDEV <= 2048 else g[:, :, :sp].repeat(1, 1, NDEV, 1)
        subs = [M.SubWeights.from_source(mesh, source, cfg, LAYER, w, NORM[w]) for w in ("attn", "ffn")]
        prog = M.PmkMhc(mesh, subs, plan, hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps,
                        iters=cfg.hc_sinkhorn_iters)  # fmt: skip
        ref = {w: MhcPrePf.from_source(mesh, source, cfg, LAYER, w, NORM[w], fpu=True) for w in ("attn", "ffn")}
        ys_host = [rows_host[..., : M.HIDDEN] * 0.5, rows_host[..., M.HIDDEN : 2 * M.HIDDEN] * 0.25]  # synthetic y

        # host -> device writes are NOT allowed inside a trace capture: the inputs are staged ONCE here (x0 pristine,
        # x_p / x_r the working residuals the programs update in place) and every traced call restores its working
        # residual from x0 with a device copy (part of the trace), so every replay starts from the same input
        x0 = shard_rows(rows_host)
        ys = [shard_rows(y) for y in ys_host]
        x_p, x_r = shard_rows(rows_host), shard_rows(rows_host)

        def phase_program():
            ttnn.copy(x0, x_p)
            outs = prog(x_p, ys)
            return (x_p,) + tuple(outs)  # (x, hns, posts, combs[, taps])

        def fused7a_path():
            ttnn.copy(x0, x_r)
            x = x_r
            outs = []
            for s, w in enumerate(("attn", "ffn")):
                hn, post, comb = ref[w](x, hn_dtype=ttnn.bfloat16)
                x = mhc_post_prefill_7a(ys[s], x, post, comb)
                outs.append((hn, post, comb))
            return x, [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs]

        # compile launches of BOTH paths before the FIRST capture (a program first compiled after a capture keeps a
        # live buffer: 'buffers alive before trace replay'), then one capture each
        phase_program()
        fused7a_path()
        ttnn.synchronize_device(mesh)
        res = dict(
            sp=sp, nrb=plan.nrb, resident=plan.resident, l1=plan.l1_bytes(), estimate=plan.estimate_us()["per_sublayer"]
        )
        # ONE live trace at a time (a second capture's buffers alive at the first trace's replay are corrupted by it,
        # TraceAllocationTracker 10:06Z): capture -> host copies -> replays -> release, then the other path
        host = lambda o: ([per_chip(o[0])] + [[per_chip(t) for t in o[i]] for i in (1, 2, 3)]  # noqa: E731
                          + ([per_chip(o[4])] if len(o) > 4 else []))

        def run_one(fn, det_check):
            tr_ = Traced(ttnn, mesh, fn, warm=False)
            if MOCK:
                return None, None, None
            out = host(tr_.out)
            det = None
            if det_check:
                tr_.replay()
                ttnn.synchronize_device(mesh)
                det = all(torch.equal(u, v) for u, v in zip(out[0], per_chip(tr_.out[0])))
            us = tr_.replays(REPLAYS)
            tr_.release()
            for grp in tr_.out[1:4]:
                for t_ in grp:
                    ttnn.deallocate(t_)
            for t_ in tr_.out[4:]:
                ttnn.deallocate(t_)
            tr_.out = None
            return out, det, us

        a, det, us_p = run_one(phase_program, True)
        b, _, us_r = run_one(fused7a_path, False)
        if not MOCK:

            def cmp(name, ha, hb, tol):  # ha / hb: per-chip host tensors (copied before each trace's release)
                pa, pb = ha, hb
                d = max(float((u - v).abs().max()) for u, v in zip(pa, pb))
                res[f"maxabs_{name}"] = d
                # per row block (32 rows) and per chip: which blocks differ (localises a group / block / ordering bug)
                blocks = []
                for u, v in zip(pa, pb):
                    rows_ = u.shape[-2]
                    blocks.append(
                        [
                            round(float((u[..., r : r + 32, :] - v[..., r : r + 32, :]).abs().max()), 6)
                            for r in range(0, rows_, 32)
                        ]
                    )
                res[f"blocks_{name}"] = blocks
                return d <= tol

            ok = cmp("residual", a[0], b[0], 2e-6)
            for s in range(2):
                ok &= cmp(f"hn{s}", a[1][s], b[1][s], 0.0)
                ok &= cmp(f"post{s}", a[2][s], b[2][s], 0.0)
                ok &= cmp(f"comb{s}", a[3][s], b[3][s], 0.0)
            if plan.probe and len(a) > 4:  # the root pass taps vs the host, per (sublayer, row block), chip 0
                from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import reference_step as post_ref
                from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import reference_step as pre_ref

                taps_h = a[4][0].reshape(-1, 32)
                xr = rows_host[0, 0, :sp].double()  # chip 0's rows
                tab = {}
                for s, w in enumerate(("attn", "ffn")):
                    h = subs[s].host
                    ref = pre_ref(xr.float(), h["fn_t"], h["scale"], h["bias"], h["w"], hc_eps=cfg.hc_eps,
                                  x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters)  # fmt: skip
                    z_host = ref["z"].double()  # [SP, 24] from the golden rows: independent of the device pass / 7a
                    per_block = []
                    for rb in range(sp // 32):
                        d = M.decode_taps(taps_h, plan, rb, s)
                        zb = z_host[rb * 32 : (rb + 1) * 32]
                        hp = M.host_planes(d["z"].T, cfg.hc_sinkhorn_iters, cfg.hc_eps)  # on the DEVICE z
                        pd = M.decode_pack_diag(taps_h, plan, rb, s)  # rev 5: the root's verified pack
                        per_block.append(
                            dict(
                                pack=pd,
                                z_vs_host=round(float((d["z"] - zb.T).abs().max()), 6),  # the root's z vs the true z
                                z_post_cols=round(float((d["z"][4:8] - zb.T[4:8]).abs().max()), 6),
                                z_comb_cols=round(float((d["z"][8:] - zb.T[8:]).abs().max()), 6),
                                pass_vs_host=round(
                                    float((d["pass"] - hp).abs().max()), 6
                                ),  # the pass on the device's z
                                dev_20_vs_1=round(float((d["pass"] - d["iters1"]).abs().max()), 6),
                                z_absmax=round(float(d["z"].abs().max()), 3),
                            )
                        )
                    tab[w] = per_block
                    xr = post_ref(
                        ys_host[s][0, 0, :sp].float(), xr.reshape(sp, 4, M.HIDDEN).float(), ref["post"], ref["comb"]
                    ).reshape(sp, -1)
                res["taps_per_block"] = tab
                res["taps"] = {w: dict(z_vs_host=max(b["z_vs_host"] for b in tab[w]),
                                       pass_vs_host=max(b["pass_vs_host"] for b in tab[w]),
                                       bad_z_blocks=[i for i, b in enumerate(tab[w]) if b["z_vs_host"] > 1e-2],
                                       pack_attempts_max=max(max(b["pack"]["attempts1"], b["pack"]["attempts2"])
                                                             for b in tab[w]),
                                       pack_marker_bad_blocks=[i for i, b in enumerate(tab[w])
                                                               if not b["pack"]["marker_ok"]],
                                       pack_lost_blocks=[i for i, b in enumerate(tab[w]) if b["pack"]["pack_lost"]],
                                       pack_diag_invalid_blocks=[i for i, b in enumerate(tab[w])
                                                                 if not b["pack"]["valid"]])
                               for w in tab}  # fmt: skip
            res.update(deterministic=det, us_phases=us_p, us_fused7a=us_r, replays=REPLAYS)
            res["status"] = "pass" if ok and det else "fail"
        row(**res)
    if REC and not MOCK:
        for name in ("pcc.jsonl", "perf.jsonl", "tests.jsonl"):
            with open(os.path.join(OUT, name), "a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")


try:
    main()
except Exception:
    traceback.print_exc()
    row(status="fail", error=traceback.format_exc()[-1200:])
fails = [r for r in rows if r.get("status") == "fail"]
RESULT = dict(status="pass" if not fails else "fail", rows=rows)
print("S16", json.dumps(RESULT, default=str)[:1500], flush=True)
