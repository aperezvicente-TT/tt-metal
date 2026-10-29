# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK 2x4 / pair; host-side only, nothing is launched): Stage 14b — the runtime-argument layout of the
TP layers' captured programs per T (the wm6 / wm7 root cause, 2026-10-20). Stage 14's fabric hooks (tt/tp8_layer.py
_hooked, tt/tp_stack.py layer_hook) put the lane arguments at index len(first core's args) of the readers' and the
post-mix owners' kernels (MK_FAB_RT / MKX_FAB_RT / MK_POST_FAB_RT, ONE compile-time index) and append them to every
core's own vector: a kernel whose per-core argument counts differ gets them at other indices on the other cores.
Per T (ARGS ts), per kernel: the distinct per-core argument counts, and for the hooked kernels the cores whose lane
arguments do not sit at the define's index. ARGS: ts ([4, 8, 32]), dense (0), sparse (4 or None), tag."""
import gc
import importlib
import os
import sys

import torch

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for mod in (TPL, TPS):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
TS = [int(v) for v in args.get("ts", [4, 8, 32])]
DENSE = int(args.get("dense", 0))
SPARSE = args.get("sparse", 4)
cfg = hf_config()
source = FdTileCacheSource(CACHE_DIR, MANIFEST)
os.environ["GLM53_TP_NATIVE_T_UNSAFE"] = "1"


def log(*a):
    print(f"M5B-RT {C.now()}", *a, flush=True)


def lens(kd):
    return sorted({len(v) for v in TPL._rt_dict(kd).values()})


def check(desc, hooks):
    """-> {kernel: dict(lens, bad_cores)}; hooks: {kernel suffix: (define, expected lane args)}"""
    out = {}
    for kd in desc.kernels:
        name = str(kd.kernel_source).split("kernels/")[-1]
        r = dict(lens=lens(kd))
        for suf, (define, want) in hooks.items():
            if TPL._src_is(kd, suf):
                first = len(next(iter(TPL._rt_dict(kd).values())))  # the hooks' index rule
                idx = int(dict(kd.defines).get(define, first))
                rt = {xy: v + [int(a) for a in want] for xy, v in TPL._rt_dict(kd).items()}  # as _kernel appends
                r["hook_index"] = idx
                r["bad_cores"] = sum(1 for v in rt.values() if v[idx : idx + len(want)] != [int(a) for a in want])
                r["cores"] = len(rt)
        out[name] = r
    return out


rows = []
w_d = w_s = moe = None
for T in TS:
    gc.collect()
    tp = TPL.TPDenseKDALayer(mesh, cfg, DENSE, source, OptimizedDecoderConfig(), b=T, program="layer", weights=w_d)
    w_d = tp.w
    x = tp.replicated(torch.zeros(1, 1, T, 16384))
    act = tp.masks(torch.ones(T, dtype=torch.bool))
    Sm, Cm = tp.shards.shard_state(torch.zeros(T, 64, 128, 128), torch.zeros(1, 3, T, 24576, dtype=torch.bfloat16))
    with TPL._Capture() as cap:
        tp.mk(x, Cm, act, t=T, state=Sm)
    ((io, desc),) = cap.calls
    fab = tp.fabric
    res = check(desc, {"mk/reader_a.cpp": ("MK_FAB_RT", fab.producer_args()),
                       "mk/owner_dm0.cpp": ("MK_POST_FAB_RT", fab.consumer_args())})  # fmt: skip
    hooked = {k: v for k, v in res.items() if "hook_index" in v}
    nonuni = {k: v["lens"] for k, v in res.items() if len(v["lens"]) > 1}
    r = dict(test=f"rt probe dense L{DENSE} T={T}", T=T, kind="dense_kda", hooked=hooked, nonuniform=nonuni)
    log("ROW", r)
    rows.append(r)
    if SPARSE is not None:
        from models.autoports.zai_org_glm_5_3_flash.tt import ep_experts as EPX
        from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP

        if moe is None:
            moe = EPX.build_moe_ep(mesh, cfg, source, int(SPARSE), OptimizedDecoderConfig())
        sp = TPSP.TPSparseKDALayer(mesh, cfg, int(SPARSE), source, OptimizedDecoderConfig(), b=T, moe=moe, weights=w_s)
        w_s = sp.w
        with TPL._Capture() as cap:
            sp.mk(x, Cm, act, t=T, state=Sm)
        ((io, desc),) = cap.calls
        fab = sp.fabric
        res = check(desc, {"mk/reader_moe_a.cpp": ("MK_FAB_RT", fab.producer_args()),
                           "mk/owner_dm0.cpp": ("MK_POST_FAB_RT", fab.consumer_args())})  # fmt: skip
        hooked = {k: v for k, v in res.items() if "hook_index" in v}
        nonuni = {k: v["lens"] for k, v in res.items() if len(v["lens"]) > 1}
        r = dict(test=f"rt probe sparse L{SPARSE} T={T}", T=T, kind="sparse_kda", hooked=hooked, nonuniform=nonuni)
        log("ROW", r)
        rows.append(r)
        del sp
    del tp
bad = [r["test"] for r in rows if any(v.get("bad_cores") for v in r["hooked"].values())]
log("SUMMARY bad:", bad)
RESULT = dict(ok=not bad, bad=bad)
