# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair or its mock; program 2026-10-27T00:59:56Z): the drafter's block-attention FOLD
(GLM53_DF2_OPS_FOLD, tt/dflash2_kind.py) vs today, on the in-trace DFlash2 kind STANDALONE (the real config, random
weights; the ops layer mode the twins serve; the drafter cache in bfp8 as on the served line):
  - the drafter caches filled with random K / V over every position < cap (the same bytes for every mode);
  - per context L (ARGS ctxs, default 4,096 and 32,768): the record set to c = L - 16 (c' + K = L - 8), PREP + ctx once,
    then per mode the block program once eagerly with the probe on:
      off     today: the stock paged SDPA decode over 32 users (8 block rows + 24 padding rows), one core per user
      spec    the fold at one core per batch row (the op contract: byte-identical to the unfolded rows)
      spec64  the fold with up to 64 cores per kv head (the speed form)
    every layer's attention output (rows 0..K, the chip's q heads) and the step's drafts (token rows 1..K, the record)
    compared with off: BITWISE for spec, max |diff| + drafts equal for spec64;
  - per context and mode the block program captured as its own trace and replayed `reps` times (>= 2,000) -> us per
    block (the drafter step's context-dependent part);
  - `soak_s` (30 when unwatched) of spec64 block replays at the longest context at the end.
Mock: shapes / program factories only (no numerics). ARGS: k (7), cap (33,280), ctxs ("4096,32768"), modes
("off,spec,spec64"), reps (2000), per_round (500), soak_s (0), kv ("bfp8"), tag."""
import importlib
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
K = int(args.get("k", 7))
T = K + 1
CAP = int(args.get("cap", 33280))
CTXS = [int(v) for v in str(args.get("ctxs", "4096,32768")).split(",")]
MODES = [m for m in str(args.get("modes", "off,spec,spec64")).split(",") if m]
REPS, PR = int(args.get("reps", 2000)), int(args.get("per_round", 500))
os.environ["GLM53_DF2_WEIGHTS"] = "random"
os.environ["GLM53_DF2_LAYER"] = "ops"
os.environ["GLM53_DF2_KV_DTYPE"] = str(args.get("kv", "bfp8"))
os.environ["GLM53_DF2_OPS_FOLD"] = "spec64"  # (builds the fold views; each mode then picks its form per layer)
os.environ["GLM53_DF2_FOLD_KCHUNK"] = str(args.get("kchunk", 32))  # (spec64's K chunk, tokens)
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402

DK = importlib.reload(DK)


def log(*a):
    print(f"S17-DF2FOLD-U {C.now()}", *a, flush=True)


S.M5.prune_ctx(ctx, [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "dflash2_kind"])
t0 = time.time()
ops0 = DK.TtnnOps(mesh, log=log)
tin = ops0.zeros((32, 64), torch.int64, "rm")
rec = ops0.zeros((1, 64), torch.int64, "rm")
kind = DK.build_dflash2_kind(mesh, K=K, cap=CAP, tin=tin, rec=rec, log=log)
ops = kind.ops
F64 = [Ly["fold"] for Ly in kind.L]  # the spec64 views (the build's GLM53_DF2_OPS_FOLD)
F1 = [dict(f, cfg=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ops.grid, q_chunk_size=32, k_chunk_size=32,
                                         max_cores_per_head_batch=1), cores=1, kchunk=32) for f in F64]  # fmt: skip
FORMS = dict(off=[None] * len(kind.L), spec=F1, spec64=F64)
nh = kind.S.nh_l
log(f"kind built in {time.time() - t0:.1f}s: tp {ops.tp}, nh_l {nh}, cache {tuple(kind.L[0]['Kc']['t'].shape)} "
    f"{os.environ['GLM53_DF2_KV_DTYPE']}, fold cores {F64[0]['cores']}")  # fmt: skip


def set_form(mode):
    for Ly, f in zip(kind.L, FORMS[mode]):
        Ly["fold"] = f


# the caches: random K / V at every position (the same bytes for every mode)
g = torch.Generator().manual_seed(1234)
dt = ttnn.bfloat8_b if os.environ["GLM53_DF2_KV_DTYPE"] == "bfp8" else ttnn.bfloat16
rep = ttnn.ReplicateTensorToMesh(mesh)
for i, Ly in enumerate(kind.L):
    for c_ in (Ly["Kc"], Ly["Vc"]):
        shp = tuple(int(v) for v in c_["t"].shape)
        src = ttnn.from_torch((torch.randn(*shp, generator=g) * 0.5).to(torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT,
                              device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        ttnn.copy(src, c_["t"])
        ttnn.deallocate(src)
ttnn.synchronize_device(mesh)
log("caches filled")


def chips(t):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


def set_ctx(L):
    rh = torch.zeros(1, 64, dtype=torch.int64)
    rh[0, 1] = L - 16  # c (n = 0: c' = c + 1, the bound c' + K = L - 8)
    ops.write(rec, rh)
    th = torch.zeros(32, 64, dtype=torch.int64)
    th[0, 0] = 1234
    ops.write(tin, th)
    kind.prep_program()
    kind.ctx_program()
    ttnn.synchronize_device(mesh)


rows, res = [], {}
for L in CTXS:
    set_ctx(L)
    outs = {}
    for mode in MODES:
        set_form(mode)
        kind.probe = {}
        kind.block_program()
        ttnn.synchronize_device(mesh)
        att = [chips(kind.probe[f"L{i}.att"]) for i in range(len(kind.L))]
        kind.probe = None
        outs[mode] = dict(att=[[a[0, :T, :nh].float() for a in per] for per in att], tin=chips(tin), rec=chips(rec))
        if C.MOCK:
            continue
    for mode in MODES:
        if mode == "off" or "off" not in outs or C.MOCK:
            continue
        a, b = outs[mode], outs["off"]
        bit = [all(torch.equal(x, y) for x, y in zip(pa, pb)) for pa, pb in zip(a["att"], b["att"])]
        dmax = max(float((x - y).abs().max()) for pa, pb in zip(a["att"], b["att"]) for x, y in zip(pa, pb))
        drafts_eq = all(torch.equal(x[1:T], y[1:T]) for x, y in zip(a["tin"], b["tin"]))
        res[(L, mode)] = dict(att_bitwise_layers=sum(bit), layers=len(bit), att_max_abs_diff=dmax, drafts_equal=drafts_eq)
        log(f"ctx {L} {mode} vs off: attention bitwise in {sum(bit)} / {len(bit)} layers, max |diff| {dmax:.3e}, "
            f"drafts equal {drafts_eq}")  # fmt: skip
    # timing: the block program as its own trace per mode, replayed REPS times (the mock cannot capture: none)
    for mode in ([] if C.MOCK else MODES):
        set_form(mode)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            kind.block_program()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        us, done = [], 0
        while done < REPS:
            n = min(PR, REPS - done)
            t1 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us.append((time.perf_counter() - t1) / n * 1e6)
            done += n
        ttnn.release_trace(mesh, tid)
        us_med = sorted(us)[len(us) // 2]
        r = dict(test="dflash2-fold-unit" + ("-mock" if C.MOCK else ""), milestone="DF2-CTX", ctx=L, mode=mode, K=K,
                 tp=int(mesh.get_num_devices()), nh_l=nh, kv=os.environ["GLM53_DF2_KV_DTYPE"], reps=REPS,
                 block_us=round(us_med, 1), block_us_rounds=[round(v, 1) for v in us],
                 fold_cores=(None if mode == "off" else FORMS[mode][0]["cores"]),
                 fold_kchunk=(32 if mode == "off" else FORMS[mode][0]["kchunk"]), **res.get((L, mode), {}),
                 status=("pass" if mode == "off" or (L, mode) not in res  # (no 'off' in this run: nothing compared, a timing / soak row)
                         or (mode == "spec" and res[(L, mode)]["att_bitwise_layers"] == res[(L, mode)]["layers"])
                         or (mode == "spec64" and res[(L, mode)]["drafts_equal"]) else ("diff" if mode == "spec64" else "fail")),
                 tag=args.get("tag"))  # fmt: skip
        S.record("perf" if not C.MOCK else "tests", r)
        rows.append(r)
        log(f"ctx {L} {mode}: block {us_med:.1f} us over {REPS} replays (rounds {r['block_us_rounds']})")
soak = float(args.get("soak_s", 0) or 0)
if soak > 0 and "spec64" in MODES and not C.MOCK:  # the unwatched soak: spec64 block replays at the longest context
    set_ctx(max(CTXS))
    set_form("spec64")
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        kind.block_program()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    t1, n = time.time(), 0
    while time.time() - t1 < soak:
        for _ in range(100):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        n += 100
    ttnn.release_trace(mesh, tid)
    log(f"soak: {n} spec64 block replays in {time.time() - t1:.1f}s at ctx {max(CTXS)}")
    S.record("tests", dict(test="dflash2-fold-soak", milestone="DF2-CTX", ctx=max(CTXS), replays=n, seconds=round(time.time() - t1, 1),
                           status="pass", tag=args.get("tag")))  # fmt: skip
if C.MOCK:  # the mock's row: every mode's program chain built and ran (shapes, program factories)
    S.record("tests", dict(test="dflash2-fold-unit-mock", milestone="DF2-CTX", ctxs=CTXS, modes=MODES, status="pass",
                           tp=int(mesh.get_num_devices()), tag=args.get("tag")))  # fmt: skip
RESULT = dict(ok=all(r["status"] != "fail" for r in rows), rows=len(rows))
