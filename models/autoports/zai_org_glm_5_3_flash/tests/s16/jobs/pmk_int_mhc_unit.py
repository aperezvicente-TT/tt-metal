# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair; or the pair mock): the FIRST INTEGRATION unit of the prefill megakernel -- the family's
sparse_kda layer program (tt/pmk_layer.py) with the mhc piece's REAL P1 / P10 / P16 bodies registered
(tt/pmk_sparse_kda.py), every other phase a stub. The engine composes one kernel per (core role, RISC) out of the
phases (namespaced bodies, hoisted guarded headers, merged compute config) and emits waits only on hand-offs with a
registered producer, so this unit runs P1 -> (stubs) -> P10 -> (stubs) -> P16 in ONE program.
Inputs: layer L's golden residual rows (sequence-parallel: chip c takes rows [SP c, SP c + SP)), the sublayer outputs
y_attn / y_ffn staged in DRAM (F_Y_DRAM; the RS lanes are not registered yet) = the stream-0 rows scaled, x-hat written
to DRAM (F_XH_DRAM; the AG lanes are not registered yet).
Reference: the fused7a chain (MhcPrePf PF_FPU + mhc_post_prefill_7a, twice) on the same rows and y, traced.
Compared: hn of both sublayers, post / comb of both, the final residual. Criterion: hn / post bitwise, comb and the
residual max |diff| <= 2e-3 (the mhc piece's own unit tolerances are tighter; this unit's job is the COMPOSITION:
the parts inside one program, the census CBs, the engine's waits). Traced; one live trace at a time; every program
compiled before the first capture. ARGS: layer (4), rows (2048), replays (50), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as MHC  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_sparse_kda as INT  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf import MhcPrePf, mhc_post_prefill_7a  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
# layers: the backend's layer list (default [L]); [0..4] ADOPTS the moe bench's L0-4 build in the same d0 process (a
# second, different backend build in one process clashes static CBs: one build per d0 session)
LAYERS = [int(v) for v in args.get("layers", [L])]
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 50))
tag = args.get("tag", f"pmk-int-mhc-L{L}")
NDEV = int(mesh.get_num_devices())
NORM = {"attn": "input_layernorm.weight", "ffn": "post_attention_layernorm.weight"}
cfg = hf_config()
# the L1 check is never relaxed here (11:25Z): the program allocates only the registered bodies' CBs (active-phase
# census) and PmkSparseKdaLayer refuses an over-ceiling launch on a device; the runtime-arg audit runs at descriptor
# build (mock and device). l1_check=False is honoured on the MOCK only (compile checks).
os.environ["PMK_L1_CHECK"] = "0" if (C.MOCK and not bool(args.get("l1_check", True))) else "1"

# the layer's TPPrefill pieces (weights, scratch) -- the family builder needs pf (mesh, geometry, the layer's scratch)
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
t0 = time.time()
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
prog = PMK.PmkSparseKdaLayer(pf, LAYERS.index(L), c, sp, probe=False)
g = prog.geom
SP = g.sp_rows
S.log(f"{tag}: pf + family program ready in {time.time() - t0:.1f} s (tp{NDEV}, SP {SP}, bucket {c})")

# the mhc piece on the FAMILY's member / root cores
source = FdTileCacheSource(CACHE_DIR, MANIFEST)
mp = MHC.Plan(sp_rows=SP, y_lane=False)
subs = [MHC.SubWeights.from_source(mesh, source, cfg, L, w, NORM[w]) for w in ("attn", "ffn")]
mhc = MHC.PmkMhc(mesh, subs, mp, hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps,
                 iters=cfg.hc_sinkhorn_iters, placement=INT.family_mhc_placement(prog))  # fmt: skip

rows = golden(L, 2048)["streams_in"].float().reshape(1, 1, 2048, MHC.N_STREAMS * MHC.HIDDEN)[:, :, : SP * NDEV]
mapper = ttnn.ShardTensorToMesh(mesh, dim=2) if NDEV > 1 else None
dev = lambda t_, dt: ttnn.from_torch(t_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)  # fmt: skip
ydt = ttnn.float32 if mp.y_fp32 else ttnn.bfloat16
ys_host = [rows[..., : MHC.HIDDEN] * 0.5, rows[..., MHC.HIDDEN : 2 * MHC.HIDDEN] * 0.25]
x0 = dev(rows, ttnn.float32)  # pristine input (staged once, outside every trace)
x_p, x_r = dev(rows, ttnn.float32), dev(rows, ttnn.float32)  # the working residuals (restored by a device copy)
ys = [dev(y, ydt) for y in ys_host]
empty = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
hns = [empty((1, 1, SP, MHC.HIDDEN), ttnn.bfloat16) for _ in range(2)]
posts = [empty((1, 1, SP, MHC.N_STREAMS), ttnn.float32) for _ in range(2)]
combs = [empty((1, 1, SP, MHC.N_STREAMS * MHC.N_STREAMS), ttnn.float32) for _ in range(2)]
flags = MHC.F_XH_DRAM | MHC.F_Y_DRAM | MHC.F_PC_DRAM | MHC.F_INIT
flags |= (MHC.F_X_LOAD | MHC.F_X_STORE) if mp.resident else 0
INT.register_mhc(prog, INT.mhc_bodies(mhc, x_p, hns, ys, posts, combs, flags=flags, layer=0, fn_gate=None))
S.log(f"{tag}: mhc P1 / P10 / P16 registered; census member CB end "
      f"{prog.engine.census['member']['cb_end']:,} root {prog.engine.census['root']['cb_end']:,}")

refs = {w: MhcPrePf.from_source(mesh, source, cfg, L, w, NORM[w], fpu=True) for w in ("attn", "ffn")}


def where(xy):  # the core of the 11:25Z watcher assert: its role / block in this placement (diagnostic log)
    P = prog.engine.place
    roles = [r for r, cs in P.role_cores().items() if tuple(xy) in [tuple(c) for c in cs]]
    G = [i for i, m in enumerate(P.members) if tuple(m) == tuple(xy)]
    return dict(core=tuple(xy), roles=roles, member_G=G[0] if G else None)


def fn_pmk():
    ttnn.copy(x0, x_p)
    prog(x_p, ci)
    return (x_p, hns, posts, combs)


def fn_ref():
    ttnn.copy(x0, x_r)
    x = x_r
    outs = []
    for s, w in enumerate(("attn", "ffn")):
        hn, post, comb = refs[w](x, hn_dtype=ttnn.bfloat16)
        x = mhc_post_prefill_7a(ys[s], x, post, comb)
        outs.append((hn, post, comb))
    return (x, [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs])


per_chip = lambda t_: [ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731
host = lambda o: [per_chip(o[0])] + [[per_chip(t) for t in o[i]] for i in (1, 2, 3)]  # noqa: E731

# every program compiled before the FIRST capture (the first call also runs the runtime-arg audit, per rank)
fn_pmk()
S.log(f"{tag}: runtime-arg audit per rank {prog.engine.arg_audits}; launch census {prog.launch_census}; {where((4, 0))}")
fn_ref()
ttnn.synchronize_device(mesh)
row = dict(test=f"PMK INTEGRATION: family sparse_kda layer program with the mhc piece's P1 / P10 / P16 bodies (other "
                f"phases stubs) vs the fused7a chain, layer {L}, tp{NDEV}, SP {SP}, traced", milestone="PMK", layer=L,
           tp=NDEV, sp=SP, census=prog.launch_census, arg_audit=prog.engine.arg_audits, mock=C.MOCK, tag=tag)  # fmt: skip
if C.MOCK:
    row["status"] = "mock"
else:
    res = {}
    for name, fn in (("pmk", fn_pmk), ("ref", fn_ref)):  # one live trace at a time
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            out = fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        res[name] = host(out)
        us = []
        for _ in range(REPLAYS):
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            us.append((time.perf_counter() - t1) * 1e6)
        res[name + "_us"] = sorted(us)[len(us) // 2]
        ttnn.release_trace(mesh, tid)
    a, b = res["pmk"], res["ref"]
    mx = lambda u, v: max(float((p - q).abs().max()) for p, q in zip(u, v))  # noqa: E731
    row.update(maxabs_residual=mx(a[0], b[0]),
               maxabs_hn=[mx(a[1][s], b[1][s]) for s in range(2)], maxabs_post=[mx(a[2][s], b[2][s]) for s in range(2)],
               maxabs_comb=[mx(a[3][s], b[3][s]) for s in range(2)], us_pmk=res["pmk_us"], us_ref=res["ref_us"],
               traced=True)  # fmt: skip
    ok = (max(row["maxabs_hn"]) == 0.0 and max(row["maxabs_post"]) == 0.0 and max(row["maxabs_comb"]) <= 2e-3
          and row["maxabs_residual"] <= 2e-3)  # fmt: skip
    row["status"] = "pass" if ok else "fail"
S.log(row)
if bool(args.get("record", True)) and not C.MOCK:
    C.record("pcc", row)
RESULT = row
