# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK; host-side only): Stage 14b — the L1 floor of a StackBackend: every L1 buffer alive after the
backend, its KV cache and decode IO are built (before the first program), lowest address first, with the Python
tensors that own them (gc walk: shape, dtype) -> which allocations set the floor every stack program's CB region must
stay under (2026-10-20: Stage 14's TP8 M3 knobs put the dense stack's reader CBs at 0x146600 > the floor 0x144500).
ARGS: b (32), layers (ids), native, knobs ("s14" / "s7a"), n (24: lowest buffers listed)."""
import gc
import importlib
import os
import sys
from collections import defaultdict

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

importlib.reload(SM)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4, 5, 6, 7])]
os.environ["GLM53_STACK_KNOBS"] = args.get("knobs", "s7a")


def log(*a):
    print(f"L1FLOOR {C.now()}", *a, flush=True)


M5.prune_ctx(ctx, [])
backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(),
                                 native_t=bool(args.get("native", False)))  # fmt: skip
kv = backend.allocate_kv_cache(max_batch=B, max_seq_len=2048)
backend.allocate_decode_io(kv)
owners = defaultdict(list)
for o in gc.get_objects():
    try:
        if isinstance(o, ttnn.Tensor) and o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
            if o.memory_config().buffer_type == ttnn.BufferType.L1:
                owners[int(o.buffer_address())].append((tuple(o.padded_shape), str(o.dtype).split(".")[-1],
                                                         str(o.memory_config().memory_layout).split(".")[-1]))  # fmt: skip
    except Exception:  # noqa: BLE001 (gc walk over arbitrary objects)
        continue
bufs = sorted({(int(b.address), int(b.max_size_per_bank)) for b in ttnn._ttnn.reports.get_buffers(mesh)
               if b.buffer_type == ttnn.BufferType.L1})  # fmt: skip
n = int(args.get("n", 24))
log("SUMMARY", dict(l1_buffers=len(bufs), lowest=hex(bufs[0][0]) if bufs else None,
                    bytes_per_core=sum(s for _, s in bufs), knobs=os.environ["GLM53_STACK_KNOBS"], B=B, layers=ids))  # fmt: skip
for a, s in bufs[:n]:
    log("BUF", hex(a), s, owners.get(a, [])[:2])
RESULT = dict(ok=True, lowest=hex(bufs[0][0]) if bufs else None, n=len(bufs))
