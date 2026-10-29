# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the TP2 pair, a d0 card; <= 15 min): STAGE 2 of the per-layer kernel (program 2026-10-24 00:13:51Z):
program A of layer L with A2 + A3 + A4 = {P2, P3, P4, P6, P7, P8} in ONE program (PMK_A2_FUSE: three member phase windows
-- the projections | the kda items | o_proj, PfStream's rings re-placed per window --, the in-program H3 barrier, the part
regime (P3 / P7 at HiFi2 under the items' HiFi4 full-sync), C1 then RSR on the lanes with a serial open / close) vs the
five-program chain A1 | A2 | A3 | A4 | A5 of a SECOND PmkProgramA on the same residual / KDA state (the fused object's
items wait the in-program H3, so its A3 cannot run alone). Both TRACED (the traces reset x and the state); one live trace
at a time; every program compiled before the first capture.
Criterion: bitwise per chip on the A2 stores (u, lr, f, gate_out), y, on, S, C, x1, x-hat_f, post_f, comb_f; then the fused
trace replayed `det` times, read at det / 2 and det: bitwise to the first.
Also: the A2 + A3 + A4 spans (the control's, each traced alone) vs the fused program alone; the chain vs the fused chain;
the windows. ARGS: layer (4), rows (2048), replays (20), det (200), tag."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402,F401
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 20))
DET = int(args.get("det", 200))
tag = args.get("tag", f"pmk-fuse2-unit-L{L}")
NDEV = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
t0 = time.time()
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
pl = pf.layers[0]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
assert sp and c == N, (sp, c, N)
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
rows, _ = S.golden_layer(L, N)
rows = rows.reshape(N, 4, 4096).float()
x0 = pf.streams_to_device(rows, sp)  # pristine (staged once, outside every trace)
x_p = pf.streams_to_device(rows, sp)  # the residual both program-A objects are built on (its address is in their args)
kda = pl.kda
state_p = kda.allocate_state(1)
ttnn.copy(ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask)
S.log(f"{tag}: pf ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c})")
t1 = time.time()
kw = dict(probe=False, pfs_park=int(args.get("pfs_park", 4)))
# variant: "a2f" = stage 2 (A2 + A3 + A4); "a34r" = stage 1 + the o_proj row groups under the scan (PMK_O_RECTS 4)
VARIANT = str(args.get("variant", "a2f"))
ctl = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a34_merge=False, a2_fuse=False, o_rects=0, **kw)
if VARIANT == "a34r":
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a34_merge=True, a2_fuse=False, o_rects=4, **kw)
    FSCOPE = tuple(ASM.A34_SCOPE)
else:
    # REUSE (fuse8 TP8 hang 02:36Z, A/B): the fused object shares the control's collectives (one C1X / RS8 instance
    # and sync shard on the lanes, as consecutive layers of the served stack do) instead of a second instance
    rk = dict(reuse=ctl) if bool(args.get("reuse", False)) else {}
    fus = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, a2_fuse=True, **kw, **rk)
    FSCOPE = tuple(ASM.A2F_SCOPE)
assert FSCOPE in [tuple(s_) for s_ in fus.scopes], fus.scopes
S.log(f"{tag}: program A x2 assembled in {time.time() - t1:.1f} s; control {ctl.scopes}; fused {fus.scopes}; "
      f"knobs {fus.report.get('knobs')}")
_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(_zeros[key], t_)


def use(prog, scopes):
    prog.layer.engine.programs = [set(s_) for s_ in scopes]
    prog.layer.order = [set(s_) for s_ in scopes]


def fn(prog, scopes=None, reset=True):
    use(prog, scopes or prog.scopes)
    if reset:
        ttnn.copy(x0, x_p)
        zero(state_p.S)
        zero(state_p.C)
    prog(x_p, ci)


per_chip = lambda t_: [ttnn.to_torch(d).float().clone() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731


def outputs(prog):
    out = {k_: per_chip(t_) for k_, t_ in prog.stores().items()}
    out.update(y=per_chip(prog.ys[0]), on=per_chip(prog.scr["on"]), S=per_chip(state_p.S), C=per_chip(state_p.C),
               x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]), post_f=per_chip(prog.posts[1]), comb_f=per_chip(prog.combs[1]))  # fmt: skip
    return out


def compare(a, b):
    out = {}
    for k in a:
        cs = []
        for x_, y_ in zip(a[k], b[k]):
            x2, y2 = x_.reshape(-1, x_.shape[-1]), y_.reshape(-1, y_.shape[-1])
            cs.append(dict(bitwise=bool(torch.equal(x2, y2)), maxabs=float((x2 - y2).abs().max()),
                           pcc=round(C.pcc(x2, y2), 7), finite=bool(torch.isfinite(x2).all())))  # fmt: skip
        out[k] = cs
    return out


def capture(prog, scopes=None, reset=True):
    t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn(prog, scopes, reset)
    finally:
        ttnn.end_trace_capture(mesh, t_, cq_id=0)
    return t_


row = dict(variant=VARIANT, test=f"PMK per-layer kernel ({VARIANT}) stage 2 / o_proj row groups: A2 + A3 + A4 = {{P2 .. P8}} in ONE program (three phase windows, in-program "
                f"H3, C1 then RSR on the lanes) vs the A1 | A2 | A3 | A4 | A5 chain, program A L{L} tp{NDEV} bucket {c}, "
                f"traced; criterion bitwise per chip (the A2 stores, y, on, S, C, x1, x-hat_f, post_f, comb_f) + {DET} replays",
           milestone="PMK-FUSE-2", layer=L, tp=NDEV, bucket=c, control=[list(s_) for s_ in ctl.scopes],
           fused=[list(s_) for s_ in fus.scopes], knobs=fus.report.get("knobs"), mock=C.MOCK, tag=tag)  # fmt: skip
try:
    # FUSED_FIRST (A/B): the fused chain's compile launch before the control's (no earlier C1X / RS8 user on the lanes)
    order = [("fused", fus), ("control", ctl)] if bool(args.get("fused_first", False)) else [("control", ctl), ("fused", fus)]
    for nm_, pr_ in order:
        S.log(f"{tag}: {nm_} compile launch")
        fn(pr_)
        ttnn.synchronize_device(mesh)
        row["census_" + nm_] = dict(pr_.layer.launch_census)
    row["windows"] = getattr(fus.engine, "window_report", None)
    S.log(f"{tag}: census control {row['census_control']}; fused {row['census_fused']}; windows {row['windows']}")
    if C.MOCK:
        row["status"] = "mock"
    else:
        tid = capture(ctl)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ref = outputs(ctl)
        perf_ctl = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} control")
        ttnn.release_trace(mesh, tid)
        tid = capture(fus)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got = outputs(fus)
        cmp = compare(got, ref)
        ok_bw = all(ch["bitwise"] for v in cmp.values() for ch in v)
        S.log(f"{tag}: fused vs control {cmp}")
        perf_fus = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} fused")
        det, n_done = {}, 1 + (perf_fus.get("replays", 0) if perf_fus else 0)
        for mark in sorted({max(1, DET // 2), DET}):
            while n_done < mark:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                n_done += 1
            ttnn.synchronize_device(mesh)
            d_ = compare(outputs(fus), got)
            det[str(n_done)] = all(ch["bitwise"] for v in d_.values() for ch in v)
            if not det[str(n_done)]:
                det[f"{n_done}_diff"] = {k: [ch["maxabs"] for ch in v] for k, v in d_.items()}
        ttnn.release_trace(mesh, tid)
        S.log(f"{tag}: determinism {det}")
        spans = {}
        for name, prog, scs in (("A2", ctl, [("P2", "P3")]), ("A3", ctl, [("P4", "P6")]), ("A4", ctl, [("P7", "P8")]),
                                ("FUSED", fus, [FSCOPE])):  # fmt: skip
            t_ = capture(prog, scs, reset=False)
            spans[name] = S.replay_stats(mesh, t_, replays=REPLAYS, label=f"{tag} {name}").get("median_us")
            ttnn.release_trace(mesh, t_)
        use(ctl, ctl.scopes)
        use(fus, fus.scopes)
        sum3 = sum(spans[k] for k in ("A2", "A3", "A4")) if all(spans.get(k) for k in ("A2", "A3", "A4")) else None
        row.update(compare=cmp, bitwise=ok_bw, determinism=det, us_chain=perf_ctl.get("median_us"),
                   us_fused_chain=perf_fus.get("median_us"), us_spans=spans, us_a2_a3_a4=round(sum3, 1) if sum3 else None,
                   status="pass" if ok_bw and all(v for k, v in det.items() if not k.endswith("_diff")) else "fail")  # fmt: skip
        S.log(f"{tag}: chain {row['us_chain']} us, fused chain {row['us_fused_chain']} us; spans {spans}; status {row['status']}")
        if bool(args.get("record", True)):
            C.record("pcc", dict(test=row["test"], milestone="PMK-FUSE-2", layer=L, tp=NDEV, bucket=c, bitwise=ok_bw,
                                 determinism=det, status=row["status"], tag=tag))  # fmt: skip
            C.record("perf", dict(test=f"PMK fused A2 + A3 + A4 L{L} tp{NDEV}: traced spans A2 / A3 / A4 / A2F and the chain vs the fused chain",
                                  milestone="PMK-FUSE-2", layer=L, tp=NDEV, bucket=c, us_spans=spans, us_chain=row["us_chain"],
                                  us_fused_chain=row["us_fused_chain"], replays=REPLAYS, status=row["status"], tag=tag))  # fmt: skip
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:1500])
RESULT = row
print("PMK_FUSE2_UNIT", {k: v for k, v in row.items() if k not in ("compare",)}, flush=True)
