# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host sim of the EXACT prefill indexer levers (doc/ttnn/14-production-tp8-megakernel/dsa/indexer_scaling.md section
8) on REAL data: layer 3's 32,768-token golden prompt through the CPU mHC reference and the real indexer weights of the
tile cache (tests/s14/dsa_indexer_sim.py's data path) -> pooled keys, q_idx (bf16, as the device stores it) and head
weights of sampled query rows.  Then:
  (1) the DEVICE CHAIN of the default config, emulated per (query, pool): per head h the bf16 pack of relu(q_h . k) (the
      dot accumulated in a 16-bit DEST across the 4 head-dim tiles), the bf16 product with w_h in DEST, the bf16
      L1-accumulate pack -- heads in order -- vs the fp64 scores: the selected-set overlap (the silicon's default row
      shows 0.9915 vs the fp32 golden: the emulation's consistency check);
  (2) the same chain in the exact serial order of lever (1) (every head's matmul first, then the per-head accumulation,
      and the k-outer / head-inner loop nest of the kernel's band walk): torch.equal bitwise with (1), and the
      selected set identical on every row -- the loop-order independence the kernel change relies on;
  (3) the DEST-MAC reduction of the rejected configs (hg 0 without serial: the 32 products MAC'd in a 16-bit DEST, one
      pack): the overlap vs (1) < 1 -- why k128 / hg0 moved the selection (silicon: 0.9915 -> 0.9831 vs golden);
  (4) lever (2), the per-chunk causal bound: for every chunk of 2,048 and device rank at TP2 and TP8 the derived kv_len
      (ceil((start + (rank + 1) Sq) / 128) compressed tiles) covers every visible pool of every row (assert), and the
      sampled rows' top-512 over [0, kv_len) equals their top-512 over the whole 32K window under the visibility mask
      (set equality 100 %).
usage (host only, the s16 d0 env): python tests/s16/ix_exact_sim.py [--rows N] [--out p]"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests import common as C  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource  # noqa: E402

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
PROMPT = 32768
TOPK, KPOOL, HEADS, DIM, DENSE_MAX, CHUNK = 512, 4, 32, 128, 2051, 2048


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def dot_dest16(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q [H, 128] bf16-valued fp32, k [P, 128] -> [H, P]: the 4 head-dim tiles' fp32 partial dots accumulated in a
    16-bit DEST (the default's fp32_dest_acc_en = false): acc = bf16(acc + partial), partial exact in fp32 (the FPU's
    HiFi4 bf16 x bf16 products and its fp32 accumulator within one tile)"""
    acc = torch.zeros(q.shape[0], k.shape[0])
    for d in range(0, DIM, 32):
        acc = bf16(acc + (q[:, d : d + 32].double() @ k[:, d : d + 32].double().T).float())
    return acc


def chain_default(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """the default config's device chain per pool: s_h = bf16(relu(dot)), p_h = bf16(s_h w_h), acc = bf16(acc + p_h)
    in head order (the per-head MAC onto the zeroed DEST + the L1-accumulate pack); head 0 overwrites"""
    s = torch.relu(dot_dest16(q, k))  # bf16-valued already (DEST 16-bit) ; relu in the packer
    acc = None
    for h in range(HEADS):
        p = bf16(s[h] * w[h])
        acc = p if acc is None else bf16(acc + p)
    return acc


def chain_serial(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """lever (1): phase 1 = every head's s_h first (all packed), phase 2 = the per-head products + L1-acc in head
    order -- and the kernel's k-outer walk: pools processed in tiles of 32 columns, each tile's chain independent"""
    out = torch.empty(k.shape[0])
    for c0 in range(0, k.shape[0], 32):
        kt = k[c0 : c0 + 32]
        s_all = [torch.relu(dot_dest16(q[h : h + 1], kt))[0] for h in range(HEADS)]  # phase 1, per head
        acc = bf16(s_all[0] * w[0])
        for h in range(1, HEADS):  # phase 2
            acc = bf16(acc + bf16(s_all[h] * w[h]))
        out[c0 : c0 + 32] = acc
    return out


def chain_dest_mac(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """the rejected configs' reduction: the 32 products MAC'd onto one 16-bit DEST tile (acc = bf16(acc + s_h w_h),
    the product NOT rounded separately), one pack"""
    s = torch.relu(dot_dest16(q, k))
    acc = torch.zeros(k.shape[0])
    for h in range(HEADS):
        acc = bf16(acc + s[h] * w[h])
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 4))
    out = a.out or os.path.join(os.environ.get("GLM53_OUT_DIR", "/tmp"), "ix_exact_sim.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    src = TileCacheSource(C.CACHE_DIR, C.MANIFEST)
    cfg = C.hf_config()
    h = lambda n: src.host(LAYER, f"self_attn.indexer.{n}").double()  # noqa: E731
    ix = dict(wk=h("wk.weight"), gate=h("index_kpool_compress_gate"), wp=h("weights_proj.weight"), wq_b=h("wq_b.weight"),
              kn_w=h("k_norm.weight").reshape(-1), kn_b=h("k_norm.bias").reshape(-1),
              ape=h("index_kpool_compress_ape").reshape(KPOOL, DIM))  # fmt: skip
    qa = src.host(LAYER, "self_attn.q_a_proj.weight").double()
    kva = src.host(LAYER, "self_attn.kv_a_proj_with_mqa.weight").double()
    qa_norm = src.host(LAYER, "self_attn.q_a_layernorm.weight").double().reshape(-1)
    aa = mhc_args(src, cfg, LAYER)
    gd = C.golden(LAYER, PROMPT)
    x = gd["streams_in"][0].reshape(PROMPT, -1)
    hn = torch.cat([G.mhc_pre_ref(x[i : i + CHUNK], aa)["hn"] for i in range(0, PROMPT, CHUNK)])
    ir = G.indexer_rows(hn, ix)
    nq = PROMPT // KPOOL
    pools = G.pool(ir["k"].reshape(nq, KPOOL, DIM), ir["g"].reshape(nq, KPOOL, DIM), ix["ape"]).float()  # bf16-exact
    qk = hn @ torch.cat([qa, kva], dim=1)
    q_r = G.rms(qk[:, : cfg.q_lora_rank], qa_norm, cfg.rms_norm_eps)
    print(f"SIM data in {time.time() - t0:.1f} s", flush=True)
    rows = sorted(
        set(torch.linspace(DENSE_MAX + 1, PROMPT - 17, a.rows).long().tolist()) | set(range(PROMPT - 16, PROMPT))
    )
    qi_all = G.bf((q_r[rows] @ ix["wq_b"]).reshape(len(rows), HEADS, DIM)).float()  # bf16 q_idx
    # the head weights as the device packs them: w * 32^-0.5 * 128^-0.5 in bf16 (prefill_tp DsaIndexer.select)
    w_all = bf16((hn[rows] @ ix["wp"]).float() * (HEADS**-0.5) * (DIM**-0.5))
    ov_fp64, ov_mac, eq_serial, bit_serial, maxabs_mac = [], [], [], [], []
    for t, p in enumerate(rows):
        nc = (p + 1) // KPOOL
        kp = pools[:nc]
        d = chain_default(qi_all[t], w_all[t], kp)
        s_ = chain_serial(qi_all[t], w_all[t], kp)
        m = chain_dest_mac(qi_all[t], w_all[t], kp)
        ref = (w_all[t].double() @ torch.relu(qi_all[t].double() @ kp.double().T)).float()
        sel = lambda v: set(torch.topk(v, TOPK).indices.tolist())  # noqa: E731
        sd = sel(d)
        ov_fp64.append(len(sd & sel(ref)) / TOPK)
        ov_mac.append(len(sd & sel(m)) / TOPK)
        eq_serial.append(float(sd == sel(s_)))
        bit_serial.append(bool(torch.equal(d, s_)))
        maxabs_mac.append(float((d - m).abs().max()))
        if t % 16 == 0:
            print(f"SIM row {t}/{len(rows)} p={p} ({time.time() - t0:.0f} s)", flush=True)
    # (4) the per-chunk causal bound geometry on the 32K prompt
    bound = {}
    for tp in (2, 8):
        sq = CHUNK // tp
        ok_cover, eq_sel, n_sel = True, 0, 0
        for cs in range(0, PROMPT, CHUNK):
            if cs + CHUNK <= DENSE_MAX:
                continue
            for r in range(tp):
                kv_tiles = -(-(cs + (r + 1) * sq) // (32 * KPOOL))
                kv_pools = kv_tiles * 32
                for row in range(sq):
                    pos = cs + r * sq + row
                    nvis = (pos + 1) // KPOOL
                    ok_cover &= nvis <= kv_pools
                for t, p in enumerate(rows):  # the sampled rows that fall in this (chunk, rank)
                    if not (cs + r * sq <= p < cs + (r + 1) * sq):
                        continue
                    nvis = (p + 1) // KPOOL
                    full = chain_default(qi_all[t], w_all[t], pools[: min(nvis, nq)])
                    bounded = chain_default(qi_all[t], w_all[t], pools[: min(nvis, kv_pools)])
                    n_sel += 1
                    eq_sel += int(
                        set(torch.topk(full, TOPK).indices.tolist()) == set(torch.topk(bounded, TOPK).indices.tolist())
                    )
        bound[f"tp{tp}"] = dict(every_visible_pool_covered=bool(ok_cover), rows_checked=n_sel,
                                sel_equal_frac=eq_sel / max(1, n_sel))  # fmt: skip
    summ = lambda v: dict(mean=float(sum(v) / len(v)), min=float(min(v)), max=float(max(v)))  # noqa: E731
    row = dict(test="ix_exact_sim", layer=LAYER, prompt=PROMPT, rows=len(rows),
               default_chain_vs_fp64_overlap=summ(ov_fp64),
               serial_chain_bitwise_vs_default=all(bit_serial), serial_chain_sel_equal_frac=sum(eq_serial) / len(eq_serial),
               dest_mac_vs_default_overlap=summ(ov_mac), dest_mac_vs_default_maxabs=summ(maxabs_mac),
               causal_bound=bound, seconds=round(time.time() - t0, 1),
               time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))  # fmt: skip
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("SIM " + json.dumps(row), flush=True)
    print(f"IX_EXACT_SIM_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
