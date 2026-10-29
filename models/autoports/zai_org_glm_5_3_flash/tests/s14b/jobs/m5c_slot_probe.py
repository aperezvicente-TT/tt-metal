# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK; host-side only): Stage 14b — the slot ops' page ranges vs the per-chip tensors they address
(the pw4 / wm7 root-cause probe, 2026-10-20). StackBackend._state_ops sizes a user's KDA state rows with _pages(S) //
chunk: on a TP mesh S is sharded per chip, so the tensor-level padded_shape must be the per-chip one or the reset /
copy programs write past the chip's state into whatever DRAM follows. For every op: the [first, first + n) page range
of the destination (and source) vs the per-chip page count of that tensor, plus the DRAM neighbours of the state
tensors (the stack argument tables). ARGS: b (8), native (true), dense ([0, 1, 2]), sparse ([4]), dsa ([])."""
import gc
import importlib
import os
import sys

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

importlib.reload(SM)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 8))
ids = sorted([int(v) for v in args.get("dense", [0, 1, 2])] + [int(v) for v in args.get("sparse", [4])]
             + [int(v) for v in args.get("dsa", [])])  # fmt: skip


def log(*a):
    print(f"SLOT-PROBE {C.now()}", *a, flush=True)


backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(),
                                 native_t=bool(args.get("native", True)))  # fmt: skip
kv = backend.allocate_kv_cache(max_batch=B, max_seq_len=512)
backend.allocate_decode_io(kv)
rows = []
for c in range(backend.n_chunks):
    for li, e in enumerate(backend.seq):
        if e["sparse"] == "dsa":
            continue
        S, Cc = kv.states[c][li]
        dev0 = ttnn.get_device_tensors(S)[0]
        rows.append(dict(L=e["L"], chunk=c, S_shape=list(S.padded_shape), S_chip0=list(dev0.padded_shape),
                         pages=SM._pages(S), pages_chip0=SM._pages(dev0), S_addr=hex(S.buffer_address()),
                         C_shape=list(Cc.padded_shape), C_chip0=list(ttnn.get_device_tensors(Cc)[0].padded_shape)))  # fmt: skip
bad = []
for op in backend._state_ops(kv, B - 1, None) + backend._state_ops(kv, B - 1, 0):
    kind, s_addr, s_first, d_addr, d_first, n = op[:6]
    for c in range(backend.n_chunks):
        for li, e in enumerate(backend.seq):
            if e["sparse"] == "dsa":
                continue
            S, Cc = kv.states[c][li]
            for t_ in (S, Cc):
                limit = SM._pages(ttnn.get_device_tensors(t_)[0])
                if d_addr == t_.buffer_address() and kind in (0, 1) and d_first + n > limit:
                    bad.append(dict(op=kind, L=e["L"], dst_pages=[d_first, d_first + n], chip_pages=limit))
for r in rows:
    log("STATE", r)
log("BAD", bad[:8], "count", len(bad))
RESULT = dict(ok=not bad, bad=len(bad), state=rows[:2])
