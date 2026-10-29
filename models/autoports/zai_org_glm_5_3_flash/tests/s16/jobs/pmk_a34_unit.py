# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the TP2 pair, d0 card0; <= 15 min): THE PER-LAYER KERNEL'S FIRST FUSED PROGRAM (Stage 14, program
2026-10-23 23:35:19Z: pieces 1-2 tonight). Program A of layer L with A3 + A4 = {P4, P6, P7, P8} in ONE program
(PMK_A34_MERGE: the engine's PHASE WINDOWS -- the members' kda item CBs in place, o_proj's CBs re-laid out into their dead
region after a core barrier; the member TRISC at the items' regime with o_proj's matmuls at HiFi2 through the part
regime; P7 waits the scan leader's SEM_ON in-program) vs the five-program chain of the SAME PmkProgramA object
(A1 | A2 | A3 | A4 | A5) on the same inputs. Both TRACED (the trace resets x and the KDA state, so every replay recomputes
the same layer); one live trace at a time; every program compiled before the first capture.
Criterion: bitwise per chip on y (A4's output), on, S, C, x1, x-hat_f, post_f, comb_f; then the fused trace replayed
`det` times (200) with the outputs read at 1 / det / 2 and det: bitwise to the first (the determinism rule).
Also: the A3 + A4 span (each traced ALONE on the persistent inputs) vs the fused program alone; the chain vs the fused
chain; the window layout (members: window 1 base / span, window 2 table) and the launch census.
ARGS: layer (4), rows (2048), replays (20), det (200), tag."""
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
tag = args.get("tag", f"pmk-a34-unit-L{L}")
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
x_p = pf.streams_to_device(rows, sp)  # program A's residual (its address is baked into the program)
kda = pl.kda
state_p = kda.allocate_state(1)
ttnn.copy(ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask)
S.log(f"{tag}: pf ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c})")
t1 = time.time()
prog = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, probe=False, a34_merge=True,
                       pfs_park=int(args.get("pfs_park", 4)))  # fmt: skip
FUSED = [tuple(s_) for s_ in prog.scopes]
assert tuple(ASM.A34_SCOPE) in FUSED, FUSED
CHAIN = []
for s_ in FUSED:  # the same programs with the fused one split back into A3 | A4
    CHAIN += [("P4", "P6"), ("P7", "P8")] if s_ == tuple(ASM.A34_SCOPE) else [s_]
S.log(f"{tag}: program A assembled in {time.time() - t1:.1f} s; fused {FUSED}; chain {CHAIN}; knobs {prog.report.get('knobs')}")
_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(_zeros[key], t_)


def use(scopes):
    prog.layer.engine.programs = [set(s_) for s_ in scopes]
    prog.layer.order = [set(s_) for s_ in scopes]


def fn(scopes, reset=True):
    use(scopes)
    if reset:
        ttnn.copy(x0, x_p)
        zero(state_p.S)
        zero(state_p.C)
    prog(x_p, ci)


per_chip = lambda t_: [ttnn.to_torch(d).float().clone() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731


def outputs():
    return dict(y=per_chip(prog.ys[0]), on=per_chip(prog.scr["on"]), S=per_chip(state_p.S), C=per_chip(state_p.C),
                x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]), post_f=per_chip(prog.posts[1]), comb_f=per_chip(prog.combs[1]))  # fmt: skip


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


def capture(scopes, reset=True):
    t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn(scopes, reset)
    finally:
        ttnn.end_trace_capture(mesh, t_, cq_id=0)
    return t_


row = dict(test=f"PMK per-layer kernel, first fused program: A3 + A4 = {{P4, P6, P7, P8}} in ONE program (phase windows + "
                f"part regime) vs the A3 | A4 chain, program A L{L} tp{NDEV} bucket {c}, traced; criterion bitwise per chip "
                f"(y, on, S, C, x1, x-hat_f, post_f, comb_f) + {DET} replays bitwise",
           milestone="PMK-FUSE-1", layer=L, tp=NDEV, bucket=c, fused=[list(s_) for s_ in FUSED], chain=[list(s_) for s_ in CHAIN],
           knobs=prog.report.get("knobs"), mock=C.MOCK, tag=tag)  # fmt: skip
try:
    # every program compiled before the first capture: the chain's five, then the fused chain's four
    S.log(f"{tag}: chain compile launch")
    fn(CHAIN)
    ttnn.synchronize_device(mesh)
    row["census_chain"] = dict(prog.layer.launch_census)
    S.log(f"{tag}: fused compile launch")
    fn(FUSED)
    ttnn.synchronize_device(mesh)
    row["census_fused"] = dict(prog.layer.launch_census)
    row["windows"] = getattr(prog.engine, "window_report", None)
    S.log(f"{tag}: census chain {row['census_chain']}; fused {row['census_fused']}; windows {row['windows']}")
    if args.get("dump_src"):  # the fused program's member composites (GLM53_PMK_KEEP_SRC=1 in the job env)
        d_ = str(args["dump_src"])
        os.makedirs(d_, exist_ok=True)
        for (tg_, role_, risc_, rk_), src_ in sorted(getattr(prog.engine, "sources", {}).items()):
            if tg_ == "+".join(ASM.A34_SCOPE) and rk_ == 0 and role_ in ("member", "mm_only", "root"):
                open(os.path.join(d_, f"{tg_}_{role_}_{risc_}.cpp"), "w").write(src_)
        row["dump_src"] = sorted(os.listdir(d_))
    if C.MOCK:
        row["status"] = "mock"
    else:
        tid = capture(CHAIN)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ref = outputs()
        perf_chain = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} chain")
        ttnn.release_trace(mesh, tid)
        tid = capture(FUSED)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        got = outputs()
        cmp = compare(got, ref)
        ok_bw = all(ch["bitwise"] for v in cmp.values() for ch in v)
        S.log(f"{tag}: fused vs chain {cmp}")
        perf_fused = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} fused")
        det, n_done = {}, 1 + (perf_fused.get("replays", 0) if perf_fused else 0)
        for mark in sorted({max(1, DET // 2), DET}):
            while n_done < mark:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                n_done += 1
            ttnn.synchronize_device(mesh)
            d_ = compare(outputs(), got)
            det[str(n_done)] = all(ch["bitwise"] for v in d_.values() for ch in v)
            if not det[str(n_done)]:
                det[f"{n_done}_diff"] = {k: [ch["maxabs"] for ch in v] for k, v in d_.items()}
        ttnn.release_trace(mesh, tid)
        S.log(f"{tag}: determinism {det}")
        # the A3 + A4 span vs the fused program alone (inputs: the persistent DRAM tensors the chain just produced)
        spans = {}
        for name, scs in (("A3", [("P4", "P6")]), ("A4", [("P7", "P8")]), ("A34", [tuple(ASM.A34_SCOPE)])):
            t_ = capture(scs, reset=False)
            spans[name] = S.replay_stats(mesh, t_, replays=REPLAYS, label=f"{tag} {name}").get("median_us")
            ttnn.release_trace(mesh, t_)
        use(FUSED)
        row.update(compare=cmp, bitwise=ok_bw, determinism=det, us_chain=perf_chain.get("median_us"),
                   us_fused_chain=perf_fused.get("median_us"), us_spans=spans,
                   us_a3_plus_a4=(round(spans["A3"] + spans["A4"], 1) if spans.get("A3") and spans.get("A4") else None),
                   status="pass" if ok_bw and all(v for k, v in det.items() if not k.endswith("_diff")) else "fail")  # fmt: skip
        S.log(f"{tag}: chain {row['us_chain']} us, fused chain {row['us_fused_chain']} us; spans {spans}; status {row['status']}")
        if bool(args.get("record", True)):
            C.record("pcc", dict(test=row["test"], milestone="PMK-FUSE-1", layer=L, tp=NDEV, bucket=c, bitwise=ok_bw,
                                 determinism=det, status=row["status"], tag=tag))  # fmt: skip
            C.record("perf", dict(test=f"PMK fused A3 + A4 L{L} tp{NDEV}: traced spans A3 / A4 / A34 and the chain vs the fused chain",
                                  milestone="PMK-FUSE-1", layer=L, tp=NDEV, bucket=c, us_spans=spans, us_chain=row["us_chain"],
                                  us_fused_chain=row["us_fused_chain"], replays=REPLAYS, status=row["status"], tag=tag))  # fmt: skip
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:1500])
RESULT = row
print("PMK_A34_UNIT", {k: v for k, v in row.items() if k not in ("compare",)}, flush=True)
