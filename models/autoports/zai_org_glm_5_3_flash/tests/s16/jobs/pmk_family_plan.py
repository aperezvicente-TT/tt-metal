# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK or device; opens nothing itself): the PMK family plan of one sparse_kda layer (tt/pmk_layer.py)
at the mesh's TP -- the phase table (contract §3), the hand-off map (§4b), the L1 census per core role from real
tile sizes (§4) and the sub-chunking policy, printed and returned; with ARGS compile=1 the ProgramDescriptor with the
STUB phase bodies is built on the mesh and launched through generic_op (mock: the composite kernels JIT-compile, the CBs
allocate, the enqueue is a no-op -- the "stub bodies compile as kernels" proof; a chip would run the empty phases).

ARGS: layer (4: a sparse_kda layer; 0..2: dense_kda, Stage 17's phase set), c (2048), compile (0), grid ("": the
device's compute grid, else "11x10"),
weights (0: geometry from the HF config; 1: S.build([layer]) and the layer's real DSLinears -- slow, loads weights),
tp_override (0: the mesh's TP; 2 / 8: plan that geometry of §1b on any mesh; compile only at the mesh's TP),
chunk_store ("auto" | "l1" | "dram"), kb (8)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.config import SPARSE_KDA  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYER = int(args.get("layer", 4))
C = int(args.get("c", 2048))
TPN = int(args.get("tp_override", 0)) or int(
    mesh.get_num_devices()
)  # tp_override: plan another geometry (§1b) on any mesh
t0 = time.time()

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
KIND = "dense_kda" if cfg.layer_kind(LAYER) == "kda_dense" else "sparse_kda"
assert cfg.layer_kind(LAYER) in ("kda_moe", "kda_dense") or SPARSE_KDA, cfg.layer_kind(LAYER)
geom = PMK.PmkGeom.from_config(cfg, TPN, C, kind=KIND)
if int(args.get("weights", 0)):  # the layer's real DSLinears (shapes / dtypes / readers as served)
    ent = S.build(mesh, ctx, [LAYER], b=1, buckets=(C,))
    pl = ent["pf"].layers[0]
    kda = pl.kda
    geom.w_qkv = PMK.GeomDS.from_dslinear(kda.ds_qkv).w_dtype
    geom.w_o = PMK.GeomDS.from_dslinear(kda.ds_o).w_dtype
    S.log(
        f"real DSLinears: qkv {kda.ds_qkv.k}x{kda.ds_qkv.n} (pad {kda.ds_qkv.n_pad}) {geom.w_qkv}; o {kda.ds_o.k}x{kda.ds_o.n}"
    )
grid_s = args.get("grid", "")
if grid_s:
    gx, gy = (int(v) for v in grid_s.split("x"))
else:
    g = mesh.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
# the device's bank-adjacent streamer placement (3 per bank) keeps the members off the streamer cores (mock: yes)
try:
    from models.autoports.zai_org_glm_5_3_flash.tt.mk_readers import reader_assignments

    streamers = [(int(c_.x), int(c_.y)) for c_, _, _ in reader_assignments(mesh, 3, "MESH_ROW_PLUS_X")]
except Exception as e:  # noqa: BLE001
    streamers = []
    S.log(f"device DS streamer placement unavailable: {e!r}")
place = PMK.Placement(gx=gx, gy=gy, avoid=tuple(streamers))
fam = PMK.PmkSparseKdaPlan(geom, place, kb=int(args.get("kb", 8)), chunk_store=args.get("chunk_store", "auto")).plan()
S.log(f"PMK family plan: L{LAYER} {KIND}, C={C}, TP{TPN} -> SP rows {geom.sp_rows}, heads/chip {geom.heads_r}, "
      f"XSL slice {geom.xsl_w}, shared slice {geom.shared_r}; grid {gx}x{gy} = {place.total} cores: "
      f"members {len(place.members)} roots {len(place.roots)} mm_only {len(place.mm_only)} lanes {len(place.lanes)} "
      f"tail {len(place.tail)} streamers {len(place.streamers)} kda_scan {len(place.kda_scan)}")  # fmt: skip
print("\n=== PHASE TABLE (contract §3) ===\n" + fam.phase_table(), flush=True)
print("\n=== HAND-OFF MAP (contract §4b) ===\n" + fam.handoff_table(), flush=True)
print("\n=== L1 CENSUS (contract §4) ===\n" + fam.census_table(), flush=True)
try:
    fam.check_l1()
    l1_ok = True
    S.log("L1 census: every role under the served worker_l1 contract")
except ValueError as e:
    l1_ok = False
    S.log(f"L1 census FAIL: {e}")
clash = sorted(set(streamers) & set(place.members))
S.log(
    f"device DS streamer placement (3 per bank): {len(streamers)} cores {streamers[:6]}..; on member cores: {clash or 'none'}; "
    f"plan streamers = {place.streamers[:6]}.."
)
compiled = None
if int(args.get("compile", 0)) and TPN == int(mesh.get_num_devices()):
    import ttnn

    io = [ttnn.empty((1, 1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
    t1 = time.time()
    R, C_ = int(mesh.shape[0]), int(mesh.shape[1])
    mpd = ttnn.MeshProgramDescriptor()  # one program per chip: the XS-FREE writer lists depend on the rank
    desc = None
    for r in range(R):
        for c_ in range(C_):
            m = ttnn.MeshCoordinate(r, c_)
            desc = fam.descriptor(mesh, io, rank=r * C_ + c_)
            mpd[ttnn.MeshCoordinateRange(m, m)] = desc
    ttnn.generic_op(io, mpd)  # mock: JIT + CB allocation; enqueue no-op
    ttnn.synchronize_device(mesh)
    compiled = dict(chips=R * C_, kernels_per_chip=len(desc.kernels), cbs=len(desc.cbs), semaphores=len(desc.semaphores),
                    xs_writers_rank0=len(fam.xs_writers(0)), s=round(time.time() - t1, 1))  # fmt: skip
    S.log(f"stub program built + launched on the mesh: {compiled}")
    for t_ in io:
        ttnn.deallocate(t_)
RESULT = dict(layer=LAYER, c=C, tp=TPN, grid=f"{gx}x{gy}", cores=place.total, sp_rows=geom.sp_rows,
              census={k: {kk: vv for kk, vv in v.items() if kk != "slots"} for k, v in fam.census.items()},
              policy=fam.policy, l1_ok=l1_ok, phases=[(p.pid, p.name, p.owner, p.roles, p.body.status) for p in fam.phases],
              handoffs={h.hid: (h.counter, h.expected) for h in fam.handoffs.values()},
              rings={f"{k[0]}x{k[1]}": v["block_bytes"] for k, v in fam.rings.items()},
              streamer_clash=clash, compiled=compiled, s=round(time.time() - t0, 1))  # fmt: skip
