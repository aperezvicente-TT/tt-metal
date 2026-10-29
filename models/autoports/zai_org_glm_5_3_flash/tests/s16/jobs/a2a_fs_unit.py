"""Job (serve.py, the pair or the 2x4): Stage 16 DSA option (i) units (doc pmk/merge/a2a_fs_design.md section 4 steps
2-3) -- the SPA q / o all-to-all as FabricStreamA2A2D (tt/fabric_a2a.py) vs the op (ttnn.experimental.
all_to_all_async_generic per axis, prefill_tp.MlaPrefill._a2a's order) on the SAME seeded q / o.

Per form: q = [1, Hr, rows, 512] bf16 TILE per chip (heads -> rows) and o = [1, n Hr, rows / n, 512] (rows -> heads),
seeded N(0, 1). Four paths, each: the compile launch (all four before ANY capture), then ONE capture, one replay, the
output read per MESH COORDINATE (tensor.device_coords(), never device ids), S.replay_stats (>= `replays` replays, the
medians), the output read again (determinism across the soak), release. Checks per coordinate: FabricStream vs op
BITWISE (a copy), both vs the host reference (fabric_a2a.ref_op_a2a on the inputs read back by coordinate), and the
program cache flat across the captures / replays.
Forms: "2d" (the 2x4: stage A axis 0 + stage B axis 1 = TP8 unit, step 3), "line" (the pair's one axis = pair unit,
step 2), "a0" / "a1" (one axis of the 2x4: every column / row its own line; the op = one all_to_all with that axis).
GLM53_MOCK=1 (the host mock cannot run fabric programs): the compile launches only (programs built + JIT), the host
analysis on zero tensors, status mock.
ARGS: rows (2048), heads (0 = 64 / the a2a group), forms (default "2d" on the 2x4, "line" on the pair), replays
(S.SOAK_REPLAYS = 2,000), seed (7), nb (16), slots (4), order ("rr" | "dst"), b_lanes, a_lanes, tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import fabric_a2a as FA  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
MR, MC = int(mesh.shape[0]), int(mesh.shape[1])
N = MR * MC
ROWS = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", S.SOAK_REPLAYS))
SEED = int(args.get("seed", 7))
D = 512
DRAM = ttnn.DRAM_MEMORY_CONFIG
FORMS = list(args.get("forms") or (["2d"] if (MR, MC) == (2, 4) else ["line"]))
TAG = args.get("tag", f"a2afs-{MR}x{MC}")
KW = dict(nb=int(args.get("nb", 16)), slots=int(args.get("slots", 4)), order=args.get("order", "rr"),
          b_lanes=[tuple(v) for v in args.get("b_lanes", FA.B_LANES)], a_lanes=[tuple(v) for v in args.get("a_lanes", FA.A_LANES)])  # fmt: skip
pc = lambda: int(mesh.num_program_cache_entries()) if hasattr(mesh, "num_program_cache_entries") else -1  # noqa: E731


def coords():
    return [(r, c) for r in range(MR) for c in range(MC)]


def by_coord(t, shape):
    """{(r, c): host tensor} by mesh coordinate (zeros of the shape on the mock: no device bytes)"""
    if C.MOCK:
        return {rc: torch.zeros(shape, dtype=torch.bfloat16) for rc in coords()}
    return {(int(m[0]), int(m[1])): ttnn.to_torch(d_).reshape(shape)
            for m, d_ in zip(t.device_coords(), ttnn.get_device_tensors(t))}  # fmt: skip


def form_args(form: str) -> tuple:
    """(FabricStreamA2A2D form, axis, op axes in q order, the a2a group size)"""
    if form == "2d":
        assert (MR, MC) == (2, 4), ("the 2-D form runs on the 2x4", MR, MC)
        return "2d", None, [0, 1], N
    if form == "line":
        assert MR == 1 or MC == 1, ("the line form: a 1 x n / n x 1 mesh (a0 / a1 on a 2-D mesh)", MR, MC)
        a = 1 if MC > 1 else 0
        return "line", a, [a], N
    a = {"a0": 0, "a1": 1}[form]
    return "line", a, [a], (MR, MC)[a]


def run_form(form: str) -> dict:
    kind, axis, axes, n = form_args(form)
    hr = int(args.get("heads", 0) or 64 // n)
    rl = ROWS // n
    g = torch.Generator().manual_seed(SEED)
    q_host = torch.randn(N, hr, ROWS, D, generator=g).to(torch.bfloat16)
    o_host = torch.randn(N, hr * n, rl, D, generator=g).to(torch.bfloat16)
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,  # noqa: E731
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
    d_q, d_o = put(q_host), put(o_host)
    t0 = time.time()
    fs = FA.FabricStreamA2A2D(mesh, hr, ROWS, form=kind, axis=axis, **KW)  # persistent: sync words + relay store
    try:
        return _run(form, fs, d_q, d_o, q_host, o_host, axes, n, hr, rl, t0)
    finally:  # the standing server keeps running: free this form's buffers whatever happened
        ttnn.deallocate(d_q)
        ttnn.deallocate(d_o)
        fs.release()


def _run(form, fs, d_q, d_o, q_host, o_host, axes, n, hr, rl, t0) -> dict:
    plan_q = fs.plans[True]
    S.log(f"{TAG} {form}: heads {hr} rows {ROWS} ({rl} per chip) run {plan_q.rpb} pages, relay {plan_q.relay_pages} pages, "
          f"lanes {fs.lane_cores}, headers per BRISC {[fs.headers(*rc) for rc in coords()][:2]}, barrier {plan_q.barrier}")  # fmt: skip

    def op(x, q):
        y = x
        for i, a in enumerate(axes if q else axes[::-1]):
            z = ttnn.experimental.all_to_all_async_generic(y, in_dim=1 if q else 2, out_dim=2 if q else 1, cluster_axis=a,
                                                           memory_config=DRAM)  # fmt: skip
            if i:
                ttnn.deallocate(y)
            y = z
        return y

    paths = {"op_q": lambda: op(d_q, True), "fs_q": lambda: fs(d_q, True),
             "op_o": lambda: op(d_o, False), "fs_o": lambda: fs(d_o, False)}  # fmt: skip
    yshape = {"q": plan_q.y_shape()[1:], "o": fs.plans[False].y_shape()[1:]}
    for fn in paths.values():  # every compile launch before any capture
        ttnn.deallocate(fn())
    if not C.MOCK:
        ttnn.synchronize_device(mesh)
    pc0 = pc()
    compile_s = round(time.time() - t0, 1)
    out, again, perf = {}, {}, {}
    for name, fn in paths.items():
        shape = yshape[name[-1]]
        if C.MOCK:  # no capture on the mock: the path once more (its allocations), host analysis on zeros
            ttnn.deallocate(fn())
            out[name] = again[name] = by_coord(None, shape)
            continue
        tid, y = ttnn.begin_trace_capture(mesh, cq_id=0), None
        try:
            try:
                y = fn()
            finally:  # never leave the queue in capture mode
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            out[name] = by_coord(y, shape)
            perf[name] = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{TAG} {form} {name}")
            again[name] = by_coord(y, shape)
        finally:  # one live trace at a time: release before the next capture
            ttnn.release_trace(mesh, tid)
            if y is not None:
                ttnn.deallocate(y)
    pc1 = pc()
    # host references from the inputs as each coordinate holds them
    qin, oin = by_coord(d_q, (hr, ROWS, D)), by_coord(d_o, (hr * n, rl, D))
    if C.MOCK:
        qin = {rc: q_host[k] for k, rc in enumerate(coords())}
        oin = {rc: o_host[k] for k, rc in enumerate(coords())}
    ref = {"q": FA.ref_op_a2a(qin, MR, MC, True, axes=axes), "o": FA.ref_op_a2a(oin, MR, MC, False, axes=axes)}
    eq = lambda a, b: {f"{rc[0]},{rc[1]}": bool(torch.equal(a[rc], b[rc])) for rc in coords()}  # noqa: E731
    chk = {}
    for d_ in ("q", "o"):
        chk[f"fs_vs_op_{d_}"] = eq(out[f"fs_{d_}"], out[f"op_{d_}"])
        chk[f"fs_vs_host_{d_}"] = eq(out[f"fs_{d_}"], ref[d_])
        chk[f"op_vs_host_{d_}"] = eq(out[f"op_{d_}"], ref[d_])
        chk[f"fs_after_soak_{d_}"] = eq(again[f"fs_{d_}"], out[f"fs_{d_}"])
    ok = {k: all(v.values()) for k, v in chk.items()}
    mib = hr * (ROWS - rl) * D * 2 / 2**20  # sent per chip per a2a
    us = {k: v.get("median_us") for k, v in perf.items()}
    row = dict(test=f"SPA a2a: FabricStreamA2A2D vs all_to_all_async_generic ({form}, bitwise + replay medians)",
               milestone="P3", form=form, mesh=[MR, MC], heads_local=hr, rows=ROWS, rows_per_chip=rl, run_pages=plan_q.rpb,
               relay_pages=plan_q.relay_pages, mib_sent_per_chip=round(mib, 2), lanes={f"{k[0]}{k[1]}": v for k, v in fs.lane_cores.items()},
               nb=fs.nb, slots=fs.slots, order=KW["order"], barrier=plan_q.barrier, compile_s=compile_s,
               program_cache=[pc0, pc1], post_capture_compiles=pc1 - pc0, checks=ok, per_coord=chk, us=us,
               p90_us={k: v.get("p90_us") for k, v in perf.items()}, replays={k: v.get("replays") for k, v in perf.items()},
               gb_s={k: round(mib * 2**20 / v / 1e3, 1) for k, v in us.items() if v}, fs_calls=fs.calls, tag=f"{TAG}-{form}")  # fmt: skip
    if C.MOCK:
        row.update(status="mock", mock=True)
    else:
        row["status"] = "pass" if all(ok.values()) and pc1 == pc0 else "fail"
    S.log(f"{TAG} {form}: {row}")
    return row


rows = [run_form(f) for f in FORMS]
if args.get("record", False):
    for r_ in rows:
        C.record("pcc", r_)
RESULT = rows
print("A2A_FS_UNIT", [(r_["form"], r_["status"], r_["checks"], r_["us"]) for r_ in rows], flush=True)
