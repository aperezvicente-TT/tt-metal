# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single; mock or chip 3): Stage 14b L1 ledger — the CB / semaphore footprint of every program of a
decode step, per core role, read off the ProgramDescriptors the step launches (no device work needed: a mock-cluster
run builds and validates them): the dense_kda and sparse_kda stack programs (tt/mk_stack.py; real layers, the
production knobs), the head (tt/mk_head.py MkHead: final norm + lm_head + device top-K) and the embedding
(MkEmbed). A core role = the set of kernels on the core. Plus the L1 tensors alive (the shared intermediates pool, the
stacked residual, the hand-off buffers) = the floor every program's CB region must stay under.
ARGS: dense ([0, 1, 2]), sparse ([4, 5, 6]), ts ([1, 4]), topk (32), out (markdown path, optional)."""
import gc
import importlib
import math
import os
import sys
from collections import defaultdict

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402

for mod in (M5, MKS, MKH):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [4, 5, 6])]
TS = [int(v) for v in args.get("ts", [1, 4])]
TOPK = int(args.get("topk", 32))
# the static CB region starts at L1 size - worker_l1_size (serve.py --worker-l1 1264512: 308,352; checked against the
# device's clash messages "static circular buffer region ends at" = base + the core's CB bytes)
WORKER_L1 = int(args.get("worker_l1", 1_264_512))
BASE = 1_572_864 - WORKER_L1
ALLOC = WORKER_L1
cfg = hf_config()


def log(*a):
    print(f"L1LEDGER {C.now()}", *a, flush=True)


def footprint(desc) -> list:
    """[(role, n cores, CB bytes, CB indices, semaphores, kernels)] of one ProgramDescriptor"""
    kern = defaultdict(set)
    for kd in desc.kernels:
        for c in MKS._cores(kd):
            kern[c].add(MKS._relsrc(kd))
    cb = defaultdict(int)
    idx = defaultdict(set)
    for d in desc.cbs:
        for c in MKS._crs_cores(d.core_ranges):
            cb[c] += int(d.total_size)
            idx[c] |= {int(f.buffer_index) for f in d.format_descriptors}
    sem = defaultdict(int)
    for s in desc.semaphores:
        for c in MKS._crs_cores(s.core_ranges):
            sem[c] += 1
    roles = defaultdict(list)
    for c in set(kern) | set(cb):
        roles[tuple(sorted(kern.get(c, ())))].append(c)
    out = []
    for ks, cores in roles.items():
        b = max(cb.get(c, 0) for c in cores)
        out.append(dict(kernels=list(ks), cores=len(cores), cb_bytes=b, cb_bytes_min=min(cb.get(c, 0) for c in cores),
                        n_cbs=max(len(idx.get(c, ())) for c in cores), sems=max(sem.get(c, 0) for c in cores),
                        pct=round(100.0 * b / ALLOC, 1), cb_end=BASE + b))  # fmt: skip
    return sorted(out, key=lambda r: -r["cb_bytes"])


M5.prune_ctx(ctx, [])
M5.keep_decoders(ctx, DENSE + SPARSE)
gc.collect()
l1_0 = M5.l1_state(mesh)
seq = M5.build_layers(mesh, cfg, ctx, DENSE, SPARSE, M5.DENSE_KNOBS, M5.SPARSE_KNOBS)
kda0 = seq[0]["kda"]
progs = {}
l1 = {"start": l1_0, "layers built": M5.l1_state(mesh)}
for T in TS:
    pool = MKS.AllocPool()
    res = None
    for run in M5.kind_runs(seq):
        sparse = seq[run[0]]["sparse"]
        M5.set_env({**(M5.SPARSE_KNOBS if sparse else M5.DENSE_KNOBS),
                    **({"GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"} if sparse and T > 2 else {})})  # fmt: skip
        stk = MKS.MkStack(mesh, [seq[i]["mk"] for i in run], t=T, res=res, pool=pool)
        res = stk.residual()
        sts = [kda0.allocate_state(T) for _ in run]
        act = masks(mesh, torch.ones(T, dtype=torch.bool), kda0.lower_bound)
        stk([s.C for s in sts], act, [s.S for s in sts])
        name = f"{'sparse_kda' if sparse else 'dense_kda'} stack T={T} (N={len(run)})"
        progs[name] = dict(roles=footprint(stk.program[1]), info=dict(stk.info), table_bytes=math.prod(
            int(v) for v in stk.table.shape) * 4 if getattr(stk, "table", None) is not None else None)  # fmt: skip
        l1[f"after {name}"] = M5.l1_state(mesh)
    ttnn.synchronize_device(mesh)
    pool = stk = res = sts = None
    gc.collect()
hk = ("m5_head_host",)
if hk not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    ctx[hk] = dict(lm=MKH.lm_head_host(source), norm=MKH.norm_host(source))
if "emb" not in ctx[hk]:
    ctx[hk]["emb"] = MKH.embed_host(FdTileCacheSource(CACHE_DIR, MANIFEST))
for T in TS:
    head = MKH.MkHead(mesh, ctx[hk]["lm"], ctx[hk]["norm"], t=T, eps=cfg.rms_norm_eps, topk=TOPK if T > 1 else 5)
    emb = MKH.MkEmbed(mesh, ctx[hk]["emb"], t=T, place=head.place)
    res = ttnn.from_torch(torch.zeros(T, 1, 4, 4096), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                          memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
    with MKS._Capture() as cap:
        head(res)
        emb(head.topk_out, res)
    for i, (_, d) in enumerate(cap.calls):
        if isinstance(d, ttnn.ProgramDescriptor):
            progs[f"{'head' if i < len(cap.calls) - 1 else 'embed'} T={T} prog {i}"] = dict(roles=footprint(d))
    l1[f"after head T={T}"] = M5.l1_state(mesh)
    head = emb = res = None
    gc.collect()
lines = []
for name, p in progs.items():
    lines.append(f"### {name}")
    if p.get("table_bytes"):
        lines.append(f"arg table (DRAM): {p['table_bytes']:,} B; {p['info']}")
    lines.append(
        f"| role (kernels) | cores | CB bytes (max / min) | % of {ALLOC:,} | CB region end | CB indices | semaphores |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in p["roles"]:
        lines.append(f"| {', '.join(r['kernels']) or '(CBs only)'} | {r['cores']} | {r['cb_bytes']:,} / "
                     f"{r['cb_bytes_min']:,} | {r['pct']} | {r['cb_end']:,} | {r['n_cbs']} | {r['sems']} |")  # fmt: skip
    lines.append("")
lines.append(
    f"### L1 tensors (allocated per bank / lowest address; CB base {BASE:,}; {'MOCK: addresses not the device layout' if os.environ.get('GLM53_MOCK') else 'device'})"
)
lines.append(
    "| point | allocated per bank | lowest L1 tensor address = the floor every CB region end must stay under | "
    "headroom above the CB base |"
)
lines.append("|---|---:|---:|---:|")
for k, v in l1.items():
    lo = v.get("lowest")
    lines.append(f"| {k} | {v['allocated_per_bank']:,} | {lo if lo is None else f'{lo:,}'} | "
                 f"{'-' if lo is None else f'{lo - BASE:,}'} |")  # fmt: skip
md = "\n".join(lines)
log("LEDGER\n" + md)
if args.get("out"):
    with open(args["out"], "w") as f:
        f.write(md + "\n")
RESULT = dict(ok=True, programs=list(progs))
