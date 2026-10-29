# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job: the L1 allocator state of the long-lived process (the lowest allocated L1 address bounds every
program's static CB region), optionally after dropping the cached decoders (ARGS["clear_ctx"]). Sets RESULT."""

import gc

import ttnn


def state():
    mv = ttnn.get_memory_view(MESH, ttnn.BufferType.L1)  # noqa: F821 -- serve.py global
    blocks = [b for b in mv.block_table if str(b.get("allocated", "")).lower() in ("yes", "true", "1")]
    lowest = min((int(b["address"]) for b in blocks), default=None)
    return dict(allocated_per_bank=int(mv.total_bytes_allocated_per_bank), lowest_allocated=lowest,
                n_blocks=len(blocks), largest_free=int(mv.largest_contiguous_bytes_free_per_bank))  # fmt: skip


before = state()
dropped = []
if ARGS.get("clear_ctx"):  # noqa: F821
    # the resident program objects first: they hold the dropped decoders' weights (dev_mk_layer's set)
    for k in [k for k in CTX if isinstance(k, str) and k.endswith("_objs")]:  # noqa: F821
        dropped.append(k)
        del CTX[k]  # noqa: F821
    for k in [k for k in CTX if isinstance(k, tuple) and k and k[0] == "dec"]:  # noqa: F821
        dropped.append(str(k))
        del CTX[k]  # noqa: F821
    gc.collect()
after = state()
RESULT = None
print(f"L1_STATE before={before} dropped={dropped} after={after}", flush=True)
