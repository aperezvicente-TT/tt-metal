# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; a 2x4 MOCK is enough -- the allocator runs, kernels do not): the prefill's DRAM need per trace key
(program 2026-10-22 19:39Z follow-up: the 8,192 bucket at 64K OOMs in Stage 9's image mock with 5.5 MB/bank free).
One build (layers, buckets, the served knobs in env), then per key (dense / fresh / sparse windows, ascending):
  retained  allocated per bank after the key's first eager forward - before it (its persistent buffers: inputs, row
            tables, per-layer objects built lazily on first use)
  need      the transient peak: the largest DRAM ballast (one interleaved buffer, MB per bank) that still lets the key's
            eager forward complete, by bisection; need = free per bank before the forward - that ballast (includes the
            ops' internal intermediates, which a post-op sampler cannot see)
Rows: per key base / free / retained / need MB per bank; the 8,192 keys' need minus the 2,048 keys' = the extra DRAM the
8K bucket needs over the 2,048 bucket (KV pool share: TP8 1,408 B per token per bank). ARGS: layers ([0..4]), buckets
([2048, 8192]), max_seq (8704), sparse_t (window list, default none), steps (9: bisection steps after a 1-GiB bracket, ~2 MB), env, tag, record."""
import gc
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.environ.get("TT_METAL_HOME", ""), "models/autoports/zai_org_glm_5_3_flash/tests/s16"))
import s16common as S  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
BUCKETS = tuple(int(v) for v in args.get("buckets", [2048, 8192]))
STEPS = int(args.get("steps", 9))
tag = args.get("tag", "p3-dram-census")
rec = bool(args.get("record", False))
TPN = int(mesh.get_num_devices())
MB = 1 << 20

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
os.environ["GLM53_PF_SPARSE_T"] = ",".join(str(int(v)) for v in args.get("sparse_t", []))
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=BUCKETS, max_seq=int(args.get("max_seq", max(BUCKETS) + 512)))
pf, backend = ent["pf"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf
if args.get("sparse_t"):
    pf.sparse_ts = tuple(int(v) for v in args["sparse_t"])


def view():
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return dict(banks=int(mv.num_banks), alloc=int(mv.total_bytes_allocated_per_bank),
                free=int(mv.total_bytes_free_per_bank), largest=int(mv.largest_contiguous_bytes_free_per_bank))


def run_key(c, t):
    """one eager forward of trace key (c, t), as TPPrefill.compile issues it"""
    if t and t != c:  # a sparse window key: the chunk ends at the window's last row
        pf.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=t - c, n_valid=c)
    else:
        pf.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=0, n_valid=c)
    out = pf.forward(c, head=True, t=t)
    del out
    ttnn.synchronize_device(mesh)
    gc.collect()


def fits(c, t, ballast_bank: int, banks: int) -> bool:
    bal = None
    try:
        if ballast_bank > 0:
            rows = max(32, ballast_bank * banks // (1024 * 2) // 32 * 32)  # bf16 [rows, 1024] over `banks` banks
            bal = ttnn.empty((1, 1, rows, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        run_key(c, t)
        return True
    except Exception as e:  # noqa: BLE001 -- an allocator TT_FATAL (out of memory) = does not fit
        if "Out of Memory" not in str(e) and "out of memory" not in str(e).lower():
            raise
        return False
    finally:
        if bal is not None:
            ttnn.deallocate(bal)
        gc.collect()


keys = [(c, 0) for c in BUCKETS] + list(pf.fresh_keys(BUCKETS)) + list(pf.sparse_keys(BUCKETS))
keys = sorted(set(keys), key=lambda k: (k[0], k[1]))
S.log(f"census keys {keys}; mode {[pf.mode(c) for c in BUCKETS]}; start {view()}")
rows_ = []
for c, t in keys:
    t0 = time.time()
    v0 = view()
    run_key(c, t)  # first forward: compiles, builds the key's persistent buffers
    v1 = view()
    lo, hi = 0, v1["free"]  # ballast bytes per bank: lo fits (0 always does after the first forward), hi does not
    guess = max(0, v1["free"] - (1 << 30))  # needs are expected well under 1 GiB per bank: bracket there first
    if guess and fits(c, t, guess, v1["banks"]):
        lo = guess
    elif guess:
        hi = guess
    if fits(c, t, hi, v1["banks"]):
        lo = hi
    else:
        for _ in range(STEPS):
            mid = (lo + hi) // 2
            if fits(c, t, mid, v1["banks"]):
                lo = mid
            else:
                hi = mid
    r = dict(key=[c, t], mode=pf.mode(c), base_mb=round(v0["alloc"] / MB, 2), retained_mb=round((v1["alloc"] - v0["alloc"]) / MB, 2),
             free_mb=round(v1["free"] / MB, 2), largest_mb=round(v1["largest"] / MB, 2),
             need_mb=round((v1["free"] - lo) / MB, 2), need_hi_mb=round((v1["free"] - hi) / MB, 2) if hi != lo else None,
             wall_s=round(time.time() - t0, 1))  # fmt: skip
    rows_.append(r)
    S.log(r)
by = {tuple(r["key"]): r for r in rows_}
need = {c: max(r["need_mb"] + r["retained_mb"] for r in rows_ if r["key"][0] == c) for c in BUCKETS}
row = dict(test=f"p3 DRAM census per prefill key (tp{TPN} layers {LAYERS}, buckets {list(BUCKETS)}, windows "
                f"{list(pf.sparse_ts)}): retained + transient need MB per bank", milestone="P3", status="info",
           tp=TPN, layers=LAYERS, keys=rows_, need_plus_retained_mb=need, mock=C.MOCK, tag=tag)  # fmt: skip
if len(BUCKETS) > 1:
    row["extra_mb_largest_vs_smallest"] = round(need[max(BUCKETS)] - need[min(BUCKETS)], 2)
    row["kv_tokens_equiv_tp8"] = int(row["extra_mb_largest_vs_smallest"] * MB / 1408)
S.log({k_: v_ for k_, v_ in row.items() if k_ != "keys"})
if rec and not C.MOCK:
    C.record("tests", row)
RESULT = [row]
