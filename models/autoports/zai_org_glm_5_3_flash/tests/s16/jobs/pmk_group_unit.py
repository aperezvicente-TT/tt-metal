# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the host-p300 TP2 pair or the host-8chip 2x4 mesh): DESIGN (b), the PMK served chain over a GROUP of n
consecutive 2,048-row chunks LAYER-MAJOR (pmk_stack.GroupStack: program A per chunk, program B ONCE per layer at
n x 2,048 rows) vs the SAME chain as n sequential single-chunk forwards on the same tokens (program 2026-10-23
19:39:53Z / 19:43:49Z). Gate: the last chunk's output residual at every probed layer, the logits row, the first token
and every KDA layer's state S / C after the group -- BITWISE. Timing: the group's traced replay vs the n sequential
replays (each chunk its own key's trace, TPPrefill.plan), per 2,048-token equivalent.

Every program (both forms, every key) compiles before the first capture; one live trace at a time; host inputs written
outside the captures; mesh.quiesce_devices() after every replay, before the next eager device work (the TP8 rule: 14b's
discriminator 2026-10-23 19:58:51Z, 2cbd310a4c4). On the mock: eager forwards, the allocations / CB checks only.
ARGS: layers ([0..4]), rows (2048), n (2), last_valid (rows: the last chunk's valid rows), start (0: the group's first token position), windows (GLM53_PF_SPARSE_T, the
served list), reps (5), probe_layers (= layers), b (2), dump ("": an npz path for a cross-run bitwise compare), tag,
record (True)."""
import gc
import math
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

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = int(args.get("rows", 2048))
NG = int(args.get("n", 2))
# last_valid (the W1 group pad-mask gate, program 2026-10-26 08:14:56Z: 14b wires GLM53_PF_MOE_PADMASK into GroupStack,
# this unit gates it): the group's LAST chunk carries last_valid < rows valid rows (GroupStack.write: only the last chunk may
# be partial); the sequential form writes the same n_valid per chunk; the probes compare the last chunk's VALID rows (pad
# rows recorded, not gated), the KDA state S / C, the logits row and the token stay bitwise
LV = int(args.get("last_valid", N))
assert 0 < LV <= N, (LV, N)
START = int(args.get("start", 0))
WINDOWS = str(args.get("windows", "65536,131072,262144,524288,1048576"))
REPS = int(args.get("reps", 5))
PROBE = [int(v) for v in args.get("probe_layers", LAYERS)]
B_BUILD = int(args.get("b", 2))
# W1 (operator 2026-10-24 21:08:55Z, the whole-prompt MoE group): b_span_layer = a program-B layer whose B is ALSO traced
# alone at n x rows on the group's own inputs (the eager compile forward's) -> us_b (B time per n chunks); dram = the DRAM
# allocated per chip after the build / the group / the compile (the memory cap of N)
BSPAN = args.get("b_span_layer")
# plain_ref (W1's op arm, the cf3h-g gate): the SAME chunks through the plain TPPrefill op forward (the PMK stack
# detached: TPPrefill.layer) as a third form, traced per key; the row then carries the stack's sequential form vs the plain
# op path (bitwise expected for the op arm, GLM53_PMK_SLOTS=none: the ServedLayer op steps = the op layer)
PLAIN = bool(args.get("plain_ref", False))
# op_diag (the op arm's non-bitwise L4, d0 22:2xZ): the routed experts on the group's OWN kept inputs, ONE pass over the
# n chunks vs n per-chunk calls on their slices, in ONE traced replay (both compiled before) -> the rows that differ
OP_DIAG = bool(args.get("op_diag", False))
tag = args.get("tag", f"pmk-group-n{NG}-L{LAYERS[0]}-{LAYERS[-1]}-tp{TPN}")
rec = bool(args.get("record", True))
assert START % N == 0, (START, N)
import pmk45_alias as A45  # noqa: E402

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(v) for v in PROBE)
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PF_SPARSE_T"] = WINDOWS
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
t0 = time.time()
info, row = {}, {}


def dram(label):
    gc.collect()
    try:
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        per_bank, banks = int(mv.total_bytes_allocated_per_bank), int(mv.num_banks)
        info.setdefault("dram_gib_per_chip", {})[label] = dict(
            allocated=round(per_bank * banks / 2**30, 3), total=round(int(mv.total_bytes_per_bank) * banks / 2**30, 3))
    except Exception as e:  # noqa: BLE001
        info.setdefault("dram_gib_per_chip", {})[label] = repr(e)[:120]


try:
    max_seq = max(int(w) for w in WINDOWS.split(",")) if WINDOWS else START + NG * N
    max_seq = max(max_seq, START + NG * N) + 512
    ent = S.build(mesh, ctx, LAYERS, b=B_BUILD, buckets=(N,), max_seq=max_seq)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    dram("built")
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    pf.probe_layers = set(PROBE)
    info["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=B_BUILD)
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    assert sp and c == N, (sp, c, N)
    g32 = S.golden_prompt(32768).reshape(-1).long()
    corpus = g32.repeat(math.ceil((START + NG * N) / g32.numel()))[: START + NG * N]
    toks = corpus[START : START + NG * N]
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
    opts.update({k: v for k, v in args.items() if k.startswith("b_") or k.startswith("group_")})
    os.environ["GLM53_PF_MODE"] = "pmk"
    stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    stack.attach()
    tg = time.time()
    group = PS.GroupStack(stack, NG, c)
    info["group_build_s"] = round(time.time() - tg, 1)
    if BSPAN is not None:
        kinds_ = {int(L_): kd_ for _, L_, kd_ in stack.layers}
        assert int(BSPAN) in group.ffn or (group.op_ffn and kinds_.get(int(BSPAN)) in group.op_kinds), \
            ("b_span_layer must be a program-B layer or an op-group FFN layer", BSPAN, sorted(group.ffn))
        group.keep_b = int(BSPAN)
    dram("group")
    info["group_plan"] = group.plan()
    info["wired"] = {f"L{L_}": v for (_, L_), v in stack.wired().items()}
    nv = [N] * (NG - 1) + [LV]
    keys = [pf.plan(START + k * N, nv[k]) for k in range(NG)]
    ts = group.windows(START, nv)
    info["keys"] = [list(k_) for k_ in keys]
    info["group_windows"] = list(ts)
    H.log(f"{tag}: group n {NG} B rows {group.cb}, keys {keys}, windows {ts}; plan {info['group_plan']}")

    def pc():
        try:
            return int(mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001
            return -1

    nq = [0]

    def qz():
        """after trace replays, before the next eager device work (import_kda_state's copies, the probe / state reads): a
        replay re-arms the host's expected worker count while a TP8 chip's count can sit off it (14b 19:58:51Z: go never
        sent / early go); quiesce zeroes both sides. Not on the mock (its dispatch reset segfaults); GLM53_PF_QUIESCE=0 off"""
        if not C.MOCK and os.environ.get("GLM53_PF_QUIESCE", "1") == "1":
            mesh.quiesce_devices()
            nq[0] += 1

    qz()
    snap0 = backend.export_kda_state(kv, 0)

    def write_seq(k: int):
        pf.write_inputs(c, toks[k * N : k * N + nv[k]], slot=0, chunk_start=START + k * N, n_valid=nv[k])

    def run_seq_eager():
        for k in range(NG):
            write_seq(k)
            pf.forward(c, t=keys[k][1])

    def run_group_eager():
        group.write(toks, slot=0, chunk_start=START, n_valid=nv)
        group.forward(ts)

    # ---- every program before the first capture: the sequential keys (TPPrefill.compile) + the group (eager once)
    t1 = time.time()
    if PLAIN:  # the plain op forward's programs, compiled before any capture too
        stack.detach()
        backend.import_kda_state(kv, 0, snap0)
        run_seq_eager()
        stack.attach()
    pf.compile((c,))
    backend.import_kda_state(kv, 0, snap0)
    run_seq_eager()
    backend.import_kda_state(kv, 0, snap0)
    run_group_eager()
    backend.import_kda_state(kv, 0, snap0)
    ttnn.synchronize_device(mesh)
    info["compile_s"] = round(time.time() - t1, 1)
    dram("compiled")
    progs0 = pc()
    H.log(f"{tag}: compiled in {info['compile_s']} s; program cache {progs0}")

    def outputs():
        probes = {L_: pf.probe(c, L_).float().clone() for L_ in PROBE}
        tok = int(backend.read(io.tokens)[0]) if not C.MOCK else -1
        lg = backend.read(io.logits)[0:1].float().clone() if not C.MOCK else torch.zeros(1, 8)
        st = backend.export_kda_state(kv, 0)
        return probes, tok, lg, {int(d["L"]): (d["S"].float(), d["C"].float()) for d in st}

    if C.MOCK:
        backend.import_kda_state(kv, 0, snap0)
        run_seq_eager()
        ttnn.synchronize_device(mesh)
        o_seq = outputs()
        backend.import_kda_state(kv, 0, snap0)
        run_group_eager()
        ttnn.synchronize_device(mesh)
        o_grp = outputs()
        us_seq = us_grp = None
    else:
        def run_seq_traced(reps):
            """each chunk its own key's trace (one live at a time), replayed in order -> (us per rep, outputs)"""
            tids = {}
            us_ = []
            for rep in range(reps + 1):
                backend.import_kda_state(kv, 0, snap0)
                ttnn.synchronize_device(mesh)
                tot = 0.0
                for k in range(NG):
                    key = keys[k]
                    if key not in tids:  # capture on first use (the eager compile above ran every program)
                        for kk, tid in list(tids.items()):
                            ttnn.release_trace(mesh, tid)
                            pf.traces.pop(kk, None)
                        tids.clear()
                        n0 = pc()
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        try:
                            pf.forward(c, t=key[1])
                        finally:
                            ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        assert pc() == n0, f"program cache grew inside the capture of {key}: {n0} -> {pc()}"
                        tids[key] = tid
                        pf.traces[key] = tid
                    write_seq(k)
                    ta = time.perf_counter()
                    ttnn.execute_trace(mesh, tids[key], cq_id=0, blocking=True)
                    tot += (time.perf_counter() - ta) * 1e6
                    qz()
                if rep:  # rep 0 may carry a capture: timed reps only after it
                    us_.append(tot)
            o_ = outputs()
            for kk, tid in list(tids.items()):
                ttnn.release_trace(mesh, tid)
                pf.traces.pop(kk, None)
            tids.clear()
            return us_, o_

        # ---- the sequential form
        seq_us, o_seq = run_seq_traced(REPS)
        if PLAIN:  # ---- the plain op forward (stack detached), the same chunks and keys
            stack.detach()
            plain_us, o_plain = run_seq_traced(max(1, REPS // 2))
            stack.attach()
            pp_, tp_, lp_, sp_ = o_plain
            ps_, ts_, ls_, ss_ = o_seq
            info["plain"] = dict(
                per_layer={f"L{L_}": H.cmp(ps_[L_].reshape(N, -1), pp_[L_].reshape(N, -1)) for L_ in PROBE},
                logits=H.cmp(ls_, lp_), tokens=[ts_, tp_],
                kda_state={f"L{L_}.{n_}": H.cmp(ss_[L_][i_], sp_[L_][i_]) for L_ in sorted(ss_) for i_, n_ in enumerate("SC")},
                us=C.stats(plain_us).get("median_us"))  # fmt: skip
            d_ = info["plain"]
            d_["bitwise"] = all(v.get("bitwise") for v in list(d_["per_layer"].values()) + list(d_["kda_state"].values())) \
                and bool(d_["logits"].get("bitwise")) and ts_ == tp_  # fmt: skip
            H.log(f"{tag}: stack sequential vs the plain op forward: bitwise {d_['bitwise']}; plain us {d_['us']}")
        # ---- the group form: one trace
        backend.import_kda_state(kv, 0, snap0)
        group.write(toks, slot=0, chunk_start=START, n_valid=nv)
        n0 = pc()

        def trace_bytes():
            try:
                v_ = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
                return int(v_.total_bytes_allocated_per_bank) * int(v_.num_banks)
            except Exception:  # noqa: BLE001
                return -1

        tb0 = trace_bytes()
        group.capture(ts)
        info["group_trace_mib_per_chip"] = round((trace_bytes() - tb0) / 2**20, 2) if tb0 >= 0 else None
        assert pc() == n0, f"program cache grew inside the group capture: {n0} -> {pc()}"
        grp_us = []
        for rep in range(REPS):
            backend.import_kda_state(kv, 0, snap0)
            group.write(toks, slot=0, chunk_start=START, n_valid=nv)
            ttnn.synchronize_device(mesh)
            ta = time.perf_counter()
            group.replay(ts, blocking=True)
            grp_us.append((time.perf_counter() - ta) * 1e6)
            qz()
        o_grp = outputs()
        group.release()
        if group.kept_b is not None:  # program B alone at n x c rows on the kept inputs: ONE traced replay per timing
            kb = group.kept_b
            n0 = pc()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outs_ = []
            try:
                if kb.get("op"):  # the op arm: the routed experts (EpMoe, xffn) alone at n x c rows
                    outs_.append(group.inst[(int(BSPAN), 0)].pl.epmoe(kb["x"], kb["idx"], kb["sel"]))
                else:
                    group.ffn[int(BSPAN)](kb["x_ext"], kb["ids_rm"], kb["wt"], group.cb, x_tile=kb["x_tile"])
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            assert pc() == n0, f"program cache grew inside the B capture: {n0} -> {pc()}"
            info["b_span"] = S.replay_stats(mesh, tid, replays=max(REPS, 5), label=f"{tag} B L{BSPAN} {group.cb} rows")
            ttnn.release_trace(mesh, tid)
            qz()
            if OP_DIAG and kb.get("op"):
                ep_ = group.inst[(int(BSPAN), 0)].pl.epmoe
                sl_ = lambda t_, k_: ttnn.slice(t_, (0, 0, k_ * N, 0), (1, 1, (k_ + 1) * N, int(t_.shape[-1])))  # noqa: E731

                def diag_fn():
                    y1 = ep_(kb["x"], kb["idx"], kb["sel"])
                    yk = []
                    for k_ in range(NG):
                        parts = [sl_(t_, k_) for t_ in (kb["x"], kb["idx"], kb["sel"])]
                        yk.append(ep_(*parts))
                        for t_ in parts:
                            ttnn.deallocate(t_)
                    return y1, yk

                y1, yk = diag_fn()  # the compile launch of the slices (the EpMoe programs exist for both row counts)
                for t_ in [y1] + yk:
                    ttnn.deallocate(t_)
                ttnn.synchronize_device(mesh)
                n1 = pc()
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    y1, yk = diag_fn()
                finally:
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                rd = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0]).float().reshape(-1, int(t_.shape[-1]))  # noqa: E731
                a_ = rd(y1)
                b_ = torch.cat([rd(t_) for t_ in yk], 0)
                ids_ = rd(kb["idx"])[:, :8].long()
                dd = (a_ - b_).abs().max(-1).values
                rows_ = torch.nonzero(dd > 0).reshape(-1)
                info["op_diag"] = dict(cache_growth=pc() - n1, rows=int(a_.shape[0]), rows_differing=int(rows_.numel()),
                                       by_chunk=[int(((rows_ >= k_ * N) & (rows_ < (k_ + 1) * N)).sum()) for k_ in range(NG)],
                                       first_rows=rows_[:16].tolist(), first_ids=ids_[rows_[:4]].tolist() if rows_.numel() else [],
                                       maxabs=float(dd.max()))  # fmt: skip
                H.log(f"{tag}: OP DIAG one pass vs per chunk: {info['op_diag']}")
                ttnn.release_trace(mesh, tid)
                for t_ in [y1] + yk:
                    ttnn.deallocate(t_)
                qz()
            for t_ in outs_ + list(v for v in kb.values() if not isinstance(v, bool)):
                ttnn.deallocate(t_)
            group.kept_b = None
        us_seq, us_grp = C.stats(seq_us).get("median_us"), C.stats(grp_us).get("median_us")
    p_s, t_s, l_s, s_s = o_seq
    p_g, t_g, l_g, s_g = o_grp
    if args.get("dump") and not C.MOCK:  # the cross-run bitwise gate (e.g. a kernel fix vs today): every output to npz
        import numpy as np  # noqa: PLC0415

        arrs = {f"seq_probe_L{L_}": p_s[L_].numpy() for L_ in PROBE} | {f"grp_probe_L{L_}": p_g[L_].numpy() for L_ in PROBE}
        arrs |= {"seq_logits": l_s.numpy(), "grp_logits": l_g.numpy(), "tokens": np.array([t_s, t_g])}
        arrs |= {f"seq_kda_L{L_}_{k}": v.numpy() for L_, sv in s_s.items() for k, v in zip("SC", sv)}
        arrs |= {f"grp_kda_L{L_}_{k}": v.numpy() for L_, sv in s_g.items() for k, v in zip("SC", sv)}
        np.savez(str(args["dump"]), **arrs)
        info["dump"] = str(args["dump"])
    per_layer = {f"L{L_}": H.cmp(p_g[L_].reshape(N, -1)[:LV], p_s[L_].reshape(N, -1)[:LV]) for L_ in PROBE}
    pad_rows = ({f"L{L_}": bool(torch.equal(p_g[L_].reshape(N, -1)[LV:], p_s[L_].reshape(N, -1)[LV:])) for L_ in PROBE}
                if LV < N else None)  # (informational: pad rows are not gated)
    state = {}
    for L_ in sorted(s_s):
        state[f"L{L_}.S"] = H.cmp(s_g[L_][0], s_s[L_][0])
        state[f"L{L_}.C"] = H.cmp(s_g[L_][1], s_s[L_][1])
    logits = H.cmp(l_g, l_s)
    bitwise = all(d.get("bitwise") for d in list(per_layer.values()) + list(state.values())) and bool(logits.get("bitwise")) \
        and t_g == t_s  # fmt: skip
    status = "mock" if C.MOCK else ("pass" if bitwise else "fail")
    row = dict(test=(f"PMK design (b) GROUP: L{LAYERS[0]}-L{LAYERS[-1]} n {NG} chunks x {N} rows layer-major (program B at {group.cb} "
                     f"rows) vs {NG} sequential single-chunk forwards, tp{TPN}"),
               milestone="PMK-B6", status=status, tp=TPN, layers=LAYERS, bucket=c, n=NG, start=START, b_rows=group.cb,
               last_valid=LV, pad_rows_equal=pad_rows,
               padmask={k_: os.environ.get(k_, "") for k_ in ("GLM53_PF_MOE_PADMASK", "GLM53_PF_KDA_PADMASK")},
               windows=list(ts), keys=info["keys"], bitwise=bitwise, token_group=t_g, token_seq=t_s, logits=logits,
               per_layer=per_layer, kda_state=state, us_group=us_grp, us_seq=us_seq,
               us_per_2048_group=(round(us_grp / NG, 1) if us_grp else None), us_per_2048_seq=(round(us_seq / NG, 1) if us_seq else None),
               speedup=(round(us_seq / us_grp, 3) if us_grp and us_seq else None), group_plan=info["group_plan"],
               compile_s=info["compile_s"], group_build_s=info["group_build_s"], quiesces=nq[0], wall_s=round(time.time() - t0, 1), tag=tag,
               b_span_layer=BSPAN, us_b=(info.get("b_span") or {}).get("median_us"), b_span=info.get("b_span"),
               dram_gib_per_chip=info.get("dram_gib_per_chip"), plan_win=os.environ.get("GLM53_PMK_PLAN_WIN", ""),
               plain_ref=info.get("plain"), op_diag=info.get("op_diag"), group_trace_mib_per_chip=info.get("group_trace_mib_per_chip"),
               arm=("op" if group.op_ffn and not group.ffn else "pmk"), group_op_ffn=group.op_ffn, counters=dict(group.counters),
               pf_plan_win=os.environ.get("GLM53_PF_MOE_PLAN_WIN", ""), pf_group_rows=os.environ.get("GLM53_PF_MOE_GROUP_ROWS", ""))  # fmt: skip
    H.log(f"{tag}: status {status} bitwise {bitwise} us group {us_grp} seq {us_seq}; logits {logits}; tokens {t_g} / {t_s}")
    if rec and not C.MOCK:
        C.record("pcc", row)
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    row = dict(test="PMK design (b) group", milestone="PMK-B6", status="fail", error=f"{type(e).__name__}: {e}"[:2000],
               tp=TPN, layers=LAYERS, n=NG, tag=tag, info={k: v for k, v in info.items() if k != "page_table"})  # fmt: skip
    if rec and not C.MOCK:
        C.record("pcc", row)
# several group jobs in ONE serve process (grpmax: the arms / n values share the cached build): this job's group traces,
# PMK stack hooks and its persistent tensors are released, so the next job's stack builds on a clean TPPrefill
for o_ in ("group", "stack"):
    try:
        if o_ in globals():
            globals()[o_].release()
    except Exception as e:  # noqa: BLE001
        H.log(f"{tag}: {o_} release: {e!r}")
gc.collect()
RESULT = row
