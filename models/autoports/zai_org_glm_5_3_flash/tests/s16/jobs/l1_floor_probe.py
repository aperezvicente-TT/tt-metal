# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK or device, TP mesh): Stage 16 L1 floor probe (program ruling 2026-10-21 20:23Z: prefill in
service needs every TPPrefill program's static CB region to end <= 1,330,000 B, below the decode stack's lowest L1
buffer). An L1 tensor height-sharded over the whole compute grid puts the lowest occupied L1 address of every core at
`floor`; then every key's eager forward runs (the single-request buckets, the batched key, and the sparse keys when
dsa is on). A program whose CB region ends above the floor throws at enqueue ("Statically allocated dataflow buffers
in program N clash with L1 buffers ... ends at X"). The row records the first offender: its end address, the program,
and the innermost model frame. Nothing is captured and no trace is replayed.

ARGS: layers ([0..4]), b (4: the decode batch of the build), floor (1330000), climb (False: True = start at `floor`,
and after each offender move the floor up to its CB end and rerun, until every program fits -> the final floor is the
maximum CB end over every program, each step's offender recorded), buckets ([2048]), nb (4: the batched key, 0 = none), the P2 / P3 knobs (moe, moe_v2, kda_scan, s16common.set_attn: mla_form, mm_nq, mm_kb, env ...), tag,
record (False). Open the server with the service's --worker-l1 (1264512: the CB region starts at 1,572,864 - worker_l1)."""
import os
import re
import sys
import traceback

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
FLOOR = int(args.get("floor", 1330000))
BUCKETS = tuple(int(v) for v in args.get("buckets", [2048]))
NB = int(args.get("nb", 4))
tag = args.get("tag", "l1-floor-probe")
TPN = int(mesh.get_num_devices())

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_DSA"] = "1" if args.get("dsa", False) else "0"
os.environ["GLM53_PF_BATCH"] = str(NB)
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=int(args.get("b", 4)), buckets=BUCKETS, max_seq=max(BUCKETS) + 512)
pf = ent["pf"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "grouped")

grid = mesh.compute_with_storage_grid_size()
ncores = grid.x * grid.y
crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
CLIMB = bool(args.get("climb", False))


def sharded(rows):
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [rows, 16], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
    return ttnn.from_torch(torch.zeros(ncores * rows, 16, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=repl)  # fmt: skip


probe = sharded(1)  # 32 B per core: the top of L1
top = int(probe.buffer_address()) + 32
ttnn.deallocate(probe)


def attempt(floor: int) -> dict:
    """one pass of every key's eager forward under an L1 floor at `floor` (rounded down to 32 B)"""
    fl = sharded(max(1, -(-(top - floor) // 32)))
    at = int(fl.buffer_address())
    # programs validate their CB region only when first built: a cached program would pass unchecked
    mesh.clear_program_cache()
    res = dict(floor=at)
    try:
        pf.compile(BUCKETS)  # every key's eager forward (single buckets + the batched key): every program enqueues once
        ttnn.synchronize_device(mesh)
        res.update(status="pass")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        m = re.search(r"in program (\d+) clash with L1 buffers on core range (\S+ - \S+)\. L1 buffer allocated at (\d+) "
                      r"and static dataflow buffer region ends at (\d+)", msg)  # fmt: skip
        m2 = re.search(r"grow to (\d+) B which is beyond max L1 size of (\d+)", msg)
        frames = [f for f in traceback.extract_tb(e.__traceback__) if "zai_org_glm_5_3_flash/tt/" in f.filename]
        res.update(status="fail", error=msg.splitlines()[0][:300] if msg else repr(e),
                   program=int(m.group(1)) if m else None, core_range=m.group(2) if m else None,
                   l1_lowest=int(m.group(3)) if m else None,
                   cb_end=int(m.group(4)) if m else (int(m2.group(1)) if m2 else None),
                   frames=[f"{os.path.basename(f.filename)}:{f.lineno} {f.name}: {f.line}" for f in frames[-3:]])  # fmt: skip
    ttnn.deallocate(fl)
    S.log(res)
    return res


row = dict(test=f"L1 floor probe: every TPPrefill key's programs under an L1 floor (tp{TPN}, L{LAYERS[0]}-L{LAYERS[-1]}, "
                f"buckets {list(BUCKETS)}, batched nb {NB}{', climb' if CLIMB else ''})", milestone="P2", tp=TPN,
           layers=LAYERS, l1_top=top, buckets=list(BUCKETS), nb=NB, mla_form=pf.mla_form, kda_scan=pf.kda_scan,
           moe=pf.moe_impl, worker_l1=ctx.get("worker_l1"), tag=tag)  # fmt: skip
steps = [attempt(FLOOR)]
while CLIMB and steps[-1]["status"] == "fail" and steps[-1].get("cb_end") and len(steps) < 12:
    # the clash may be with a buffer the forward itself allocated under the floor (a transient L1 tensor lands just
    # below the lowest buffer): the floor must clear the CB end by that much too
    st_ = steps[-1]
    gap = max(0, st_["floor"] - (st_.get("l1_lowest") or st_["floor"]))
    steps.append(attempt(int(st_["cb_end"]) + gap))
last = steps[-1]
row.update(floor=steps[0]["floor"], steps=steps, status=last["status"] if not CLIMB else ("pass" if last["status"] == "pass" else "fail"),
           offender=None if steps[0]["status"] == "pass" else {k: steps[0].get(k) for k in ("program", "core_range", "cb_end", "frames")})  # fmt: skip
if CLIMB and last["status"] == "pass":
    row["max_cb_end"] = last["floor"]  # every program fits under this floor; the previous step's offender ends here
    row["max_cb_end_program"] = (
        {k: steps[-2].get(k) for k in ("program", "core_range", "cb_end", "frames")} if len(steps) > 1 else None
    )
S.log(row)
RESULT = row
