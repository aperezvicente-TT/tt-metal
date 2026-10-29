# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 HOST MOCK -- no-tt-devices + TT_METAL_MOCK_CLUSTER_DESC_PATH; also runs on a device): Stage 16
PMK plan / L1 census of ONE layer's persistent program at the binding geometry (2,048 rows sequence-parallel in both:
"tp8" = 256 rows per chip on 2x4, "pair" = 1,024 per chip on TP2): builds the layer's TPPrefill pieces, constructs the builder of its kind
(tt/pmk_layer.py; STUB = the op-graph path when absent), reads prog.plan() (phases, L1 bytes per core, CB hand-offs,
l1_max_cb_end), runs the compile launch under the mock (kernels JIT-build, every CB / L1 allocation is validated,
enqueues are no-ops) and takes the L1 census: every live L1 device tensor (address, bytes, cores) + the lowest L1
address, as Stage 14's m5b_l1_census. Criterion: l1_max_cb_end <= floor (1,330,000 B: prefill_megakernel.md §4,
below the decode stack's lowest L1 buffer) and the compile launch raises nothing. No capture, no replay, no numbers.

ARGS: layer (4), geometry ("pair" | "tp8"), rows (2048), floor (1330000), mhc_post ("fused"), moe ("ep"), moe_v2 (True), kda_scan ("op"),
dsa (False), the P3 knobs (s16common.set_attn), tag, record (True)."""
import collections
import gc
import os
import sys
import time
import traceback

import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
L = int(args.get("layer", 4))
GEOM = args.get("geometry", {2: "pair", 8: "tp8"}.get(TPN, f"tp{TPN}"))  # binding geometry: 2,048 rows in both
assert {"pair": 2, "tp8": 8}.get(GEOM, TPN) == TPN, f"geometry {GEOM} on a {TPN}-chip mesh"
N = int(args.get("rows", 2048))
FLOOR = int(args.get("floor", 1330000))
tag = args.get("tag", f"pmk-mock-L{L}-{GEOM}")
rec = bool(args.get("record", True))

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_DSA"] = "1" if args.get("dsa", False) else "0"
os.environ["GLM53_PF_BATCH"] = "0"
os.environ["GLM53_PF_PACKED"] = ""
S.set_attn(args)
kind = layer_kind(hf_config(), L)
t0 = time.time()
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
pf, kv = ent["pf"], ent["kv"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
li, c = 0, pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
factory, impl = H.load_builder(kind)
prog = factory(pf, li, c, sp, probe=True)
plan = prog.plan() if hasattr(prog, "plan") else {}
H.log(f"L{L} {kind} built in {time.time() - t0:.1f} s; builder {impl}; c={c} sp={sp} tp={TPN} mock={C.MOCK}")
H.log(f"plan: {len(plan.get('phases', []))} phases, l1_max_cb_end={plan.get('l1_max_cb_end')}")
for ph in plan.get("phases", []):
    H.log(f"  phase {ph}")

X = pf.streams_to_device(torch.zeros(c, 4, 4096), sp)
err = None
t1 = time.time()
try:  # the compile launch: under the mock every kernel JIT-builds and every CB / L1 allocation is validated
    y = H.run_pmk(pf, li, X, ci, sp, prog)  # through TPPrefill.pmk_layers
    ttnn.synchronize_device(mesh)
    if y is not X:
        ttnn.deallocate(y)
except Exception:  # noqa: BLE001 -- recorded as the row's failure, the server stays up
    err = traceback.format_exc()[-3000:]
    H.log(f"compile launch FAILED:\n{err}")
compile_s = round(time.time() - t1, 1)

# ------------------------------------------------------------------------------------------------ L1 census
gc.collect()
l1 = {}
for o in gc.get_objects():
    if not isinstance(o, ttnn.Tensor):
        continue
    try:
        if not o.is_allocated():
            continue
        mc = o.memory_config()
        if mc.buffer_type != ttnn.BufferType.L1:
            continue
        addr = o.buffer_address()
    except Exception:  # noqa: BLE001 -- host / deallocated tensors
        continue
    if addr in l1:
        continue
    ss = mc.shard_spec
    l1[addr] = dict(addr=addr, shape=tuple(int(v) for v in o.shape), dtype=str(o.dtype).split(".")[-1],
                    cores=(str(ss.grid) if ss is not None else "interleaved")[:100],
                    shard=[int(v) for v in ss.shape] if ss is not None else None)  # fmt: skip
rows = sorted(l1.values(), key=lambda r: r["addr"])
lowest = rows[0]["addr"] if rows else None
by_dtype = collections.Counter(r["dtype"] for r in rows)
H.log(f"L1 census: {len(rows)} live L1 tensors, lowest address {lowest}, by dtype {dict(by_dtype)}")
for r in rows[:12]:
    H.log(f"  {r}")
cb_end = plan.get("l1_max_cb_end")
fits = cb_end is None or int(cb_end) <= FLOOR
ok = err is None and fits and impl != "stub"
status = "fail" if (err is not None or not fits) else ("info" if impl == "stub" else "pass")
row = dict(test=f"pmk layer mock census L{L} {kind}: builder plan + compile launch under the {'mock' if C.MOCK else 'device'} "
                f"(geometry {GEOM}: c={c}, {c // TPN} rows per chip, tp{TPN}) + L1 census; criterion l1_max_cb_end <= {FLOOR}",
           milestone="P4", status=status, layer=L, kind=kind, geometry=GEOM, rows_per_chip=c // TPN, impl=impl, n=N, bucket=c, tp=TPN, mode="sp" if sp else "rep",
           phases=len(plan.get("phases", [])), l1_max_cb_end=cb_end, floor=FLOOR, fits=fits, compile_ok=err is None,
           compile_s=compile_s, l1_live=len(rows), l1_lowest=lowest, l1_rows=rows[:24], plan=plan if len(str(plan)) < 6000 else str(plan)[:6000],
           error=err, mock=C.MOCK, tag=tag)  # fmt: skip
H.log({k: v for k, v in row.items() if k not in ("test", "l1_rows", "plan", "error")})
if rec:
    C.record("tests", row)
ttnn.deallocate(X)
RESULT = row
