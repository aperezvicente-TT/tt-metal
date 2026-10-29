# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 HOST MOCK / pair mock; also runs on a device as the compile step): PROGRAM A of the sparse_kda
layer ASSEMBLED from the pieces (tt/pmk_assembly.PmkProgramA): the plan (phase table, per-program census, wait
schedule), the compile launch (every composite kernel JIT-builds, every CB / semaphore / L1 allocation is validated,
the runtime-arg audit runs per rank), then a SECOND launch = the stub-replay mock (the mock has no traces: the
replay path is the same descriptor launched again, nothing re-allocated). Criterion: 0 fatals, audit 0 errors,
launch census under 1,330,000 B on every role. ARGS: layer (4), rows (2048), probe (True), mhc_probe (True), scopes
(None = all five programs; [["P1"]] = the A1 probe unit's composite), pfs_park, lowrank_idle, tag."""
import gc
import os
import sys
import time
import traceback

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
PROBE = bool(args.get("probe", True))
TPN = int(mesh.get_num_devices())
SCOPES = args.get("scopes")  # the unit's program list, e.g. [["P1"]] = A1 alone (u-pmk-asm-a1p-L4-tp2's composite)
tag = args.get("tag", f"pmk-asm-a-mock-L{L}-tp{TPN}")
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
# host-only compile check: the member census of program A (mhc + kda items + PfStream on one core) is over the
# ceiling at TP2 until the kda item CBs alias the mhc CBs dead during P4 (README conflict #1); a DEVICE launch
# refuses the over-ceiling census (PmkSparseKdaLayer.__call__), the mock JIT-checks the composites
os.environ["PMK_L1_CHECK"] = "0" if bool(args.get("l1_relax", False)) else "1"
S.set_attn(args)
t0 = time.time()
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
kda = pf.layers[0].kda
X = pf.streams_to_device(torch.zeros(c, 4, 4096), sp)
state = kda.allocate_state(1)
ttnn.copy(ttnn.ones(tuple(state.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state.slot_mask)
S.log(f"{tag}: pf built in {time.time() - t0:.1f} s (tp{TPN}, c={c}, sp={sp})")
row = dict(test=f"PMK assembly program A mock: L{L} sparse_kda, tp{TPN}, {c // TPN} SP rows per chip, compile launch + "
                f"stub replay under the {'mock' if C.MOCK else 'device'}", milestone="PMK-A", layer=L, tp=TPN, bucket=c,
           mock=C.MOCK, tag=tag)  # fmt: skip
err = None
try:
    t1 = time.time()
    knobs = {k: bool(args[k]) for k in ("tp8_resident", "a3_merge", "a_inplace") if k in args}  # else the env / defaults (PMK_TP8_RESIDENT 1, PMK_A3_MERGE 0)
    prog = ASM.PmkProgramA(pf, 0, c, sp, X, state, ci.valid_end, probe=PROBE, mhc_probe=bool(args.get("mhc_probe", True)),
                           pfs_park=int(args.get("pfs_park", 4)), lowrank_idle_streamers=bool(args.get("lowrank_idle", False)),
                           **({"scopes": [tuple(v) for v in SCOPES]} if SCOPES else {}), **knobs)
    build_s = round(time.time() - t1, 1)
    plan = prog.plan()
    S.log(f"{tag}: assembled in {build_s} s; streams {plan['assembly']['streams']}")
    S.log(f"{tag}: scopes {plan['scopes']}; wait schedule {plan['assembly']['wait_schedule']}")
    S.log(f"{tag}: fabric header audit {plan['assembly'].get('fabric_headers')}")
    S.log(f"{tag}: dram weights {plan['assembly'].get('dram_weights')}")
    S.log(plan["census_table"])
    t2 = time.time()
    prog(X, ci)  # the compile launch (mock: JIT + validation; the audit raises before any launch)
    ttnn.synchronize_device(mesh)
    compile_s = round(time.time() - t2, 1)
    t3 = time.time()
    prog(X, ci)  # the stub-replay mock: the same program launched again (no capture on the mock)
    ttnn.synchronize_device(mesh)
    replay_s = round(time.time() - t3, 1)
    census = prog.layer.launch_census
    audits = prog.engine.arg_audits
    flat = {f"{tag_}:{k}": v for tag_, c_ in census.items() for k, v in (c_.items() if isinstance(c_, dict) else [(tag_, c_)])} \
        if any(isinstance(v, dict) for v in census.values()) else dict(census)
    over = {k: v for k, v in flat.items() if v > 1330000}
    row.update(build_s=build_s, compile_s=compile_s, replay_s=replay_s, launch_census=census, arg_audit=audits,
               knobs=prog.report.get("knobs"), rs8=prog.report.get("rs8"), fabric_headers=prog.report.get("fabric_headers"),
               dram_weights=prog.report.get("dram_weights"),
               scopes=plan["scopes"], streams=plan["assembly"]["streams"], streamers=plan["assembly"].get("streamers"),
               kda_map=prog.engine.kda_map, census_over=over, calls=prog.counters["calls"],
               status="pass" if not over and all(a["errors"] == 0 for a in audits.values()) else "fail")  # fmt: skip
    S.log(f"{tag}: launch census {census}; audits {audits}; status {row['status']}")
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-4000:]
    S.log(f"{tag}: FAILED\n{err}")
    row.update(status="fail", error=err)
gc.collect()
if bool(args.get("record", True)):
    C.record("tests", row)
RESULT = row
print("PMK_ASM_A_MOCK", {k: v for k, v in row.items() if k not in ("error",)}, flush=True)
