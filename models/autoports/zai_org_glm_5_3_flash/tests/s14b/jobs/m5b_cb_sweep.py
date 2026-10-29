# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK pair / 2x4; host-side only, nothing is launched): Stage 14b — no circular-buffer overlap in any
stack / TP layer program for every (t, reader knobs, ring size) the serving and test paths use (program 07:29Z
after the wm7 / pw4 root cause: tt-metal's per-range CB placement put the stack argument CB inside a weight ring).

Per config: the TP dense layer program (Stage 14's single-layer TP launch) and a 3-layer TPStack, the TP sparse layer
program and a 2-layer TPSparseStack; every program's CB list replayed with tt/cb_guard.placement (tt-metal's exact
algorithm) -> overlapping CB pairs. MkStack.build itself refuses an overlapping stack (cb_guard.check), which is
reported as a finding here. ARGS: ts ([1, 2, 4, 8, 16, 32]), knobs (list of env dicts), dense (0), sparse ([4, 5])."""
import gc
import importlib
import os
import sys

import torch

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.cb_guard as CBG  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for mod in (CBG, MKS, TPL, TPS):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
TS = [int(v) for v in args.get("ts", [1, 2, 4, 8, 16, 32])]
KNOBS = args.get("knobs") or [{}, {"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "700"}]
DENSE = int(args.get("dense", 0))
SPARSE = [int(v) for v in args.get("sparse", [4, 5])]
KEYS = ("GLM53_MK_OPT", "GLM53_MK_READERS", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB", "GLM53_MKX_RING",
        "GLM53_MKX_INFLIGHT")  # fmt: skip
cfg = hf_config()
source = FdTileCacheSource(CACHE_DIR, MANIFEST)


def log(*a):
    print(f"CB-SWEEP {C.now()}", *a, flush=True)


def overlaps(desc):
    bad = CBG.overlaps(list(desc.cbs))
    return sorted({(i0, i1) for _, i0, i1 in bad}), len(bad)


def stack_desc(cls, mks, fl, t, x, hist, act, st):
    stk = cls(mesh, mks, fl, t=t)
    try:
        stk.build([hist] * len(mks), act, [st] * len(mks))
    except RuntimeError as e:  # cb_guard in MkStack.build
        return None, str(e)[:200]
    return stk.desc, None


rows, findings = [], []
w_d, w_s, moes = None, {}, {}
for knobs in KNOBS:
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in knobs.items()})
    for t in TS:
        gc.collect()
        tp = TPL.TPDenseKDALayer(mesh, cfg, DENSE, source, OptimizedDecoderConfig(), b=t, weights=w_d)
        w_d = tp.w
        x = tp.replicated(torch.zeros(1, 1, t, 16384))
        act = tp.masks(torch.ones(t, dtype=torch.bool))
        Sm, Cm = tp.shards.shard_state(torch.zeros(t, 64, 128, 128), torch.zeros(1, 3, t, 24576, dtype=torch.bfloat16))
        with TPL._Capture() as cap:
            tp.mk(x, Cm, act, t=t, state=Sm)
        ((_, d1),) = cap.calls
        o1 = overlaps(d1)
        d3, err = stack_desc(TPS.TPStack, [tp.mk] * 3, tp, t, x, Cm, act, Sm)
        o3 = overlaps(d3) if d3 is not None else ("refused", err)
        r = dict(kind="dense", t=t, knobs=knobs, layer=o1, stack3=o3)
        log("ROW", r)
        rows.append(r)
        if o1[0] or (d3 is None) or o3[0]:
            findings.append(r)
        if SPARSE and t <= 32:
            from models.autoports.zai_org_glm_5_3_flash.tt import ep_experts as EPX
            from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP

            sps = []
            for L in SPARSE:
                if L not in moes:
                    moes[L] = EPX.build_moe_ep(mesh, cfg, source, L, OptimizedDecoderConfig())
                sp = TPSP.TPSparseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=t, moe=moes[L],
                                           weights=w_s.get(L))  # fmt: skip
                w_s[L] = sp.w
                sps.append(sp)
            with TPL._Capture() as cap:
                sps[0].mk(x, Cm, act, t=t, state=Sm)
            ((_, s1),) = cap.calls
            os1 = overlaps(s1)
            ds, err = stack_desc(TPS.TPSparseStack, [s.mk for s in sps], sps[0], t, x, Cm, act, Sm)
            os2 = overlaps(ds) if ds is not None else ("refused", err)
            r = dict(kind="sparse", t=t, knobs=knobs, layer=os1, stack2=os2)
            log("ROW", r)
            rows.append(r)
            if os1[0] or (ds is None) or os2[0]:
                findings.append(r)
            del sps
        del tp
log("SUMMARY configs", len(rows), "findings", len(findings))
for f in findings[:12]:
    log("FINDING", f)
RESULT = dict(ok=not findings, configs=len(rows), findings=findings[:12])
