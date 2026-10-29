# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 mock or a pair): the PMK KDA phase (tt/pmk_kda.py) as a one-phase program: the H3 store, the
state scratch and valid_end as device tensors, the item / scan core sets from the grid, the kernels JIT-built, CB / L1
allocation and runtime args validated by generic_op (mock: no enqueue). On a device (the bench unit, ARGS check=1):
the projections through prefill_tp's op path (pf_mm) fill the store, the phase runs captured + replayed, and `on` /
the final S / C are compared with kda_prefill_tp's sigmoid_gated_rms_norm output / final_state / new_carry on the
same input (PCC >= 0.9999).

ARGS: rows (2048), heads (8 at TP8 / 32 at TP2; default from the mesh: 64 / TPN), valid (rows), terms ("fp32" | "bf16"),
tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_kda as P  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
TPN = int(mesh.get_num_devices())
ROWS = int(args.get("rows", 2048))
HEADS = int(args.get("heads", 64 // TPN))
VALID = int(args.get("valid", ROWS))
tag = args.get("tag", "pmk-kda-mock")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM = ttnn.DRAM_MEMORY_CONFIG
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
dev = lambda x, dt, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(x, dtype=dt, layout=layout, device=mesh,  # noqa: E731
                                                              memory_config=DRAM, mesh_mapper=repl)  # fmt: skip

geo = P.KdaGeometry(rows=ROWS, heads=HEADS, terms_dtype=str(args.get("terms", "fp32")))
pl = P.plan(geo)
problems = pl.check()
print(f"PMK_KDA geometry rows={ROWS} heads={HEADS} chunks={geo.n_chunks} cph={geo.cph} spread={geo.spread} "
      f"item_l1={pl.item_l1} scan_l1={pl.scan_l1} problems={problems}", flush=True)  # fmt: skip
assert not problems, problems

grid = mesh.compute_with_storage_grid_size()
cores = [ttnn.CoreCoord(x, y) for y in range(int(grid.y)) for x in range(int(grid.x))]
assert len(cores) >= geo.item_cores + geo.scan_cores, (len(cores), geo.item_cores, geo.scan_cores)
item_cores, scan_cores = cores[: geo.item_cores], cores[geo.item_cores : geo.item_cores + geo.scan_cores]
W, D = geo.width, geo.head_dim
g = torch.Generator().manual_seed(1)
rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
taps = [dev(0.5 * rnd(1, 1, 3 * W), BF16) for _ in range(4)]
dt_bias = dev(0.1 * rnd(1, 1, 1, W), FP32)
rate = dev(torch.rand(1, 1, 1, W, generator=g) + 0.5, FP32)
w_row = torch.zeros(1, 1, 32, D)
w_row[0, 0, 0] = 1.0 + 0.1 * rnd(D)
o_norm_row = dev(w_row, FP32)
phase = P.KdaPhase(mesh, geo, item_cores=item_cores, scan_cores=scan_cores, taps=taps, dt_bias=dt_bias, rate=rate,
                   o_norm_row=o_norm_row, lower_bound=-5.0, eps=1e-6, nprod=1, h3_preraised=True)  # fmt: skip
scr = phase.scratch()
# the H3 store as the stream phases leave it (random here; on the device the op path fills it: check=1)
ttnn.copy(dev(0.5 * rnd(1, 1, ROWS, 3 * W), BF16), scr["u"])
ttnn.copy(dev(0.5 * rnd(1, 1, ROWS, 2 * D + HEADS), BF16), scr["lr"])
ttnn.copy(dev(rnd(1, 1, ROWS, W), FP32), scr["f"])
ttnn.copy(dev(rnd(1, 1, ROWS, W), scr["gate_out"].dtype), scr["gate_out"])
state_s = dev(torch.zeros(1, HEADS, D, D), FP32)
state_c = dev(torch.zeros(1, 3, 1, 3 * W), BF16)  # FusedKDAState.C at 1 slot (TILE, row 0 of every tile)
valid_end = dev(torch.tensor([VALID], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

t0 = time.perf_counter()
prog, io = phase.program(scr, state_s, state_c, valid_end)
n_cb = len(prog.cbs)
print(
    f"PMK_KDA program: {len(prog.kernels)} kernels, {n_cb} CBs, {len(prog.semaphores)} semaphores, {len(io)} io",
    flush=True,
)
ttnn.generic_op(io, prog)  # mock: kernels JIT-built, CBs / semaphores / runtime args validated; device: one launch
ttnn.synchronize_device(mesh)
build_s = time.perf_counter() - t0
if not C.MOCK:
    print(f"PMK_KDA probes {phase.probe_rows(scr)}", flush=True)
print(f"PMK_KDA generic_op ok in {build_s:.1f}s (mock={C.MOCK})", flush=True)

est = pl.estimate()
RESULT = dict(tag=tag, mock=C.MOCK, rows=ROWS, heads=HEADS, terms=geo.terms_dtype, chunks=geo.n_chunks, spread=geo.spread, cph=geo.cph,
              item_l1=pl.item_l1, scan_l1=pl.scan_l1, kernels=len(prog.kernels), cbs=n_cb, build_s=round(build_s, 1),
              estimate={k: round(v, 3) if isinstance(v, float) else v for k, v in est.items()})  # fmt: skip
if int(args.get("check", 0)):
    RESULT["check"] = "the device bench unit with the op-path comparison is tests/s16/jobs/pmk_kda_unit.py"
print("PMK_KDA_RESULT", RESULT, flush=True)
