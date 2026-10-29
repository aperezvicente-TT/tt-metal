# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; mock is enough): device tensors a prefill forward creates and leaves referenced (they would sit in
a trace's memory at replay: the trace-allocation tracker's 'device buffer(s) still alive before trace replay').
Compiles the buckets, then runs one more forward per bucket and lists the new live tensors with their holders.

ARGS: layers ([0, 1, 2, 3, 4]), buckets ([128]), head (True), mhc, mhc_post, moe, moe_v2."""
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import _live_device_tensors  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
BUCKETS = tuple(int(v) for v in args.get("buckets", [128]))
os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=BUCKETS)
pf = ent["pf"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", False)))
pf.compile(BUCKETS, head=bool(args.get("head", True)))
leaks = []
for c in BUCKETS:
    pf.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=0, n_valid=c)
    gc.collect()
    before = _live_device_tensors()
    pf.forward(c, head=bool(args.get("head", True)))
    gc.collect()
    for t_ in _live_device_tensors().values():
        if id(t_) in before:
            continue
        holders = []
        for r in gc.get_referrers(t_):
            if isinstance(r, dict):
                owner = [o for o in gc.get_referrers(r) if hasattr(o, "__dict__") and o.__dict__ is r]
                keys = [k for k, v in r.items() if v is t_][:3]
                holders.append(f"{type(owner[0]).__name__ if owner else 'dict'}.{keys}")
            elif not isinstance(r, list) or len(r) < 64:
                holders.append(type(r).__name__)
        line = f"LEAK bucket {c}: {tuple(t_.shape)} {t_.dtype} {t_.layout} held by {holders[:4]}"
        S.log(line)
        leaks.append(line)
S.log(f"leak probe: {len(leaks)} new live tensors after one forward per bucket {BUCKETS}")
RESULT = leaks
