# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, mock or device): every DSLinear the batched prefill runs through pf_mm on the layers `layers` -- its
path, K, N (padded), weight dtype, fidelity, and the 2D multicast program pf_mm builds for an n-row chunk (the input
of mm_bench.py's per-TP shapes). No kernel runs.

ARGS: layers ([0, 3, 4]), n (2048)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear, prefill_program  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 3, 4])]
N = int(args.get("n", 2048))
TPN = int(mesh.get_num_devices())
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf = ent["pf"]


def walk(obj, path, out, seen, depth=0):
    if id(obj) in seen or depth > 3:
        return
    seen.add(id(obj))
    if isinstance(obj, DSLinear):
        out.append((path, obj))
        return
    d = getattr(obj, "__dict__", None)
    if not d:
        return
    for k, v in d.items():
        if k.startswith("_") or k in ("tpp", "b", "backend", "mesh", "device", "kv"):
            continue
        walk(v, f"{path}.{k}", out, seen, depth + 1)


rows = []
for li, pl in enumerate(pf.layers):
    found = []
    walk(pl, f"L{LAYERS[li]}", found, set())
    for path, ds in found:
        prog = prefill_program(ds.grid, N // 32, ds.k // 32, ds.n_pad // 32,
                               weight_tile_bytes={"DataType.BFLOAT8_B": 1088, "DataType.BFLOAT4_B": 576}.get(
                                   str(ds.weight.dtype), 2048), m_round=ds.ocfg.prefill_m_block_round,
                               max_block_w=ds.ocfg.prefill_k_block, l1_budget=PF.PF_MM_EST)  # fmt: skip
        r = dict(path=path, kind=pl.kind, role=ds.role, k=ds.k, n=ds.n, n_pad=ds.n_pad, wdtype=str(ds.weight.dtype),
                 fid=str(ds.ckc.math_fidelity) if hasattr(ds.ckc, "math_fidelity") else str(ds.ckc),
                 per_core_m=prog.per_core_M, per_core_n=prog.per_core_N, out_h=prog.out_block_h,
                 in0_bw=prog.in0_block_w, sub=(prog.out_subblock_h, prog.out_subblock_w),
                 grid=(ds.grid.x, ds.grid.y), tp=TPN)  # fmt: skip
        S.log(r)
        rows.append(r)
RESULT = rows
