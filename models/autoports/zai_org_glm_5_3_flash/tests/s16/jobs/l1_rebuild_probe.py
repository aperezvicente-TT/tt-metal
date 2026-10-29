# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, mock or device): which L1 allocations survive a backend close -- the "second backend in one process
lowers the L1 floor" limit (a later TPPrefill's static CBs clashed with an L1 buffer at 1,217,792 B, 2026-10-21
01:08Z). Builds s16common's backend (+ TPPrefill, compiled), records the L1 block table, closes / drops it, records
again, builds a second one, records again. Prints every L1 block allocated after the close that was not allocated
before the first build (the leak), with its size. ARGS: layers ([0, 1, 2, 3, 4]), b (1), buckets ([2048]),
compile (True), env ({})."""
import gc
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
BUCKETS = tuple(int(v) for v in args.get("buckets", [2048]))
for k_, v_ in (args.get("env") or {}).items():
    os.environ[k_] = str(v_)


def blocks():
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    out = {}
    for b in mv.block_table:
        if str(b.get("allocated", "")).lower() in ("yes", "true", "1"):
            out[int(b["address"])] = int(b.get("size", 0))
    return out, int(mv.total_bytes_allocated_per_bank)


def drop():
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-backend", "s16-full", "s16-tf")]:
        ent = ctx.pop(k)
        if "pf" in ent:
            ent["pf"].release()
        ent["backend"].close()
    gc.collect()


drop()
b0, a0 = blocks()
S.log(f"L1 before: {len(b0)} blocks, {a0} B per bank, lowest {min(b0) if b0 else None}")
states = []
for i in range(2):
    ent = S.build(mesh, ctx, LAYERS, b=int(args.get("b", 1)), buckets=BUCKETS)
    if args.get("compile", True):
        ent["pf"].compile(BUCKETS)
    b1, a1 = blocks()
    S.log(f"L1 after build {i}: {len(b1)} blocks, {a1} B per bank, lowest {min(b1) if b1 else None}")
    drop()
    del ent
    gc.collect()
    b2, a2 = blocks()
    leak = {a: s for a, s in b2.items() if a not in b0}
    S.log(f"L1 after close {i}: {len(b2)} blocks, {a2} B per bank, lowest {min(b2) if b2 else None}; "
          f"left behind {len(leak)}: {sorted(leak.items())[:40]}")  # fmt: skip
    states.append(dict(built=dict(n=len(b1), per_bank=a1, lowest=min(b1) if b1 else None),
                       closed=dict(n=len(b2), per_bank=a2, lowest=min(b2) if b2 else None),
                       leaked=sorted(leak.items())))  # fmt: skip
RESULT = states
