# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 TP2 pair, watched once then unwatched; <= 10 min): THE device unit of PROGRAM A
(tt/pmk_assembly.PmkProgramA: mHC pre -> AG -> projections -> KDA -> o_proj -> RS -> mHC post + pre -> AG, one
persistent program) on layer L alone vs the op path up to the FFN input, same golden rows, both TRACED.
Compared per chip: x-hat_f (the FFN pre's normalised input, program A's hn[1] vs rms(mhc_pre(ffn, x1))): PCC >= 0.999
(the criterion); x-hat (H1), y (H6 = the reduced attention output), the residual after P10 (= x1), the KDA state S /
C (PCC + the bitwise flag: the projections are PfStream HiFi4 programs, not pf_mm's, so bitwise S / C vs the KDA-only
unit is reported, not required); per-phase probe us (PfStream streamer / member probes per phase, the kda probe rows)
and the traced replay time of program A vs the op-path chain. One live trace at a time; every program compiled before
the first capture; no host writes inside a capture. ARGS: layer (4), rows (2048), replays (50), tag, record (True).
A1 / A2 scopes (the bisect): A2's four stores are scored per chip vs host references (u / lr from x-hat, f / gate_out
from the device lr; finite + PCC >= store_pcc_bar (0.999)); a2_heads (8) runs f_b / g_b in TP8's per-chip geometry."""
import hashlib
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 50))
SCOPES = args.get("scopes")  # e.g. [["P1"]] = A1 alone (the first silicon row); [["P1"], ["P2", "P3"]] = the A2 bisect; None = all five
P3 = args.get("p3_phases")  # the A2 bisect: ["qkv"], then ["qkv", "lowrank"], ... (default: all four)
MHC_PROBE = bool(args.get("mhc_probe", False))  # the mhc root pass taps (z / pass per block) decoded against the reference
tag = args.get("tag", f"pmk-asm-a-unit-L{L}")
NDEV = int(mesh.get_num_devices())
PCC_BAR = float(args.get("pcc_bar", 0.9999))
US_BAR = args.get("us_bar")  # the traced replay us of program A must be <= this (TP8: 4,000 us per chip); None = no timing bar
KNOBS = {k: bool(args[k]) for k in ("tp8_resident", "a3_merge", "a_inplace", "park_align") if k in args}
if "c_trim" in args:  # PMK_PFS_C_TRIM for this job's process (pmk_assembly._pfs_parts reads it when it builds the parts)
    os.environ["PMK_PFS_C_TRIM"] = "1" if args["c_trim"] else "0"
if "park2_align" in args:  # 16 1863a072ff knob name -> the merged PARK alignment (14 c7f39d2b1d)
    KNOBS["park_align"] = bool(args["park2_align"])
# SPLIT TRACES (the TP8 A3 bisect, 04:1xZ): the program list is cut into consecutive groups after the scopes that
# contain these phase ids (e.g. ["P3", "P6"]: A1+A2 | A3 | A4+A5), each group its own trace executed in order with
# host TAPS in between (no host read inside a capture): after the group with P3 the A2 stores u / lr / f / gate_out,
# after P6 S / C / on, after the last group everything again -> S after A3 vs after A5 (bitwise expected: else a later
# program corrupts it), the stores after A2 vs after A5 (bitwise expected), each tap vs the op-path reference
SPLIT = args.get("split_after")  # else env / defaults (PMK_TP8_RESIDENT 1, PMK_A3_MERGE 0)
# the TP8 A2 bisect on the pair (program 15:45:26Z): f_b / g_b on heads 0..A2_HEADS-1 (8 = TP8's per-chip geometry)
A2_HEADS = args.get("a2_heads")
STORE_BAR = float(args.get("store_pcc_bar", 0.999))  # every A2 store vs its host reference (finite elements), per chip
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
# the op reference's mHC pre = the SERVED one (fused7a + FPU sums + post7a; numaudit/mhc64, 2026-10-24 19:0xZ): with the
# composite mhc_coeffs the reference's comb was ~13x further from fp64 than the PMK's and dominated every PMK-vs-op row
for k_, v_ in (("GLM53_PF_MHC_PRE", "fused7a"), ("GLM53_PF_MHC_FPU", "1"), ("GLM53_PF_MHC_POST7A", "1")):
    os.environ.setdefault(k_, v_)
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
t0 = time.time()
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
pf, kv = ent["pf"], ent["kv"]
pl = pf.layers[0]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
assert sp and c == N, (sp, c, N)
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
if args.get("synth_seed") is not None:  # (the Galaxy: no goldens) a seeded synthetic layer input -- timing / bitwise A/B
    rows = torch.randn(N, 4, 4096, generator=torch.Generator().manual_seed(int(args["synth_seed"])))
else:
    try:
        rows, _ = S.golden_layer(L, N)
    except FileNotFoundError:  # no N-row golden (6,144 / 8,192: TTFT item a, 18:56:47Z): the model is causal, so the first N
        big, _ = S.golden_layer(L, 32768)  # rows of the 32K golden ARE the N-token golden's rows
        rows = big.reshape(32768, -1)[:N].clone()
        del big
if args.get("input_pt"):  # another fp32 layer input [N, 4, 4096] (e.g. the hf_ref chain's L{L}/streams_in: router.pt's rows)
    rows = torch.load(args["input_pt"], map_location="cpu", weights_only=True)[:N]
rows = rows.reshape(N, 4, 4096).float()
x0 = pf.streams_to_device(rows, sp)  # pristine (staged once, outside every trace)
x_p = pf.streams_to_device(rows, sp)  # program A's residual (its address is baked into the program)
x_r = pf.streams_to_device(rows, sp)  # the reference's
kda = pl.kda
state_p = kda.allocate_state(1)
ttnn.copy(ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask)
S.log(f"{tag}: pf ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c})")
t1 = time.time()
prog = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, probe=True, **({"scopes": [tuple(v) for v in SCOPES]} if SCOPES else {}),
                       **({"p3_phases": tuple(P3)} if P3 else {}), mhc_probe=MHC_PROBE,
                       pfs_park=int(args.get("pfs_park", 4)), lowrank_idle_streamers=bool(args.get("lowrank_idle", False)),
                       **({"a2_heads": int(A2_HEADS)} if A2_HEADS else {}), **KNOBS)  # fmt: skip
S.log(f"{tag}: program A assembled in {time.time() - t1:.1f} s; streamers {prog.streamers}; streams {prog.report['streams']}; "
      f"knobs {prog.report.get('knobs')}; rs8 {prog.report.get('rs8')}")
_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(_zeros[key], t_)


taps = {}


def tap(name, t):
    if name not in taps:
        taps[name] = ttnn.zeros_like(t)
    ttnn.copy(t, taps[name])


def fn_pmk():
    ttnn.copy(x0, x_p)
    zero(state_p.S)
    zero(state_p.C)
    prog(x_p, ci)


# op_taps (the GLM53_PMK_A_LR_EXACT pair check, program 2026-10-24 05:21:11Z): the op path's low-rank / f_b / g_b
# outputs copied out INSIDE the reference trace (prefill_tp.pf_mm(kda.ds_lowrank, x) and kda._pf("ds_fb" / "ds_gb")),
# compared per chip with program A's stores lr / f / gate_out (bitwise + max |d|)
OP_TAPS = bool(args.get("op_taps", False))
if OP_TAPS:
    from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as _PT

    _kda = pl.kda
    _pf_mm0 = _PT.pf_mm

    def _pf_mm_tap(ds, x, *a, **k):
        y_ = _pf_mm0(ds, x, *a, **k)
        if ds is getattr(_kda, "ds_lowrank", None):
            tap("op.lr", y_)
        return y_

    _PT.pf_mm = _pf_mm_tap
    _pf0 = _kda._pf

    def _pf_tap(name, x, w, ckc, dtype):
        y_ = _pf0(name, x, w, ckc, dtype)
        if name in ("ds_fb", "ds_gb"):
            tap("op." + name, y_)
        return y_

    _kda._pf = _pf_tap


def fn_ref():
    """the op path up to the FFN input (pmk_harness.ref_layer's chain, stopped after the FFN pre)"""
    ttnn.copy(x0, x_r)
    zero(pl.scratch.S)
    zero(pl.scratch.C)
    h, post, comb = pf._mhc_pre(pl, "attn", x_r)
    hn = pl.rms(h, "attn", BF16)
    if hn is not h:
        ttnn.deallocate(h)
    tap("H1.xhat", hn)
    tap("post0", post)
    tap("comb0", comb)
    hb = pf.gather(hn, sp)
    if hb is not hn:
        ttnn.deallocate(hn)
    a = pf.reduce(pf.attention(pl, hb, ci, 0, hn if hb is hn else None, None), sp)
    ttnn.deallocate(hb)
    tap("H6.y", a)
    x1 = pf._post(a, x_r, post, comb)
    for t_ in (a, post, comb):
        ttnn.deallocate(t_)
    tap("x1", x1)
    h2, post2, comb2 = pf._mhc_pre(pl, "ffn", x1)
    hn2 = pl.rms(h2, "ffn", BF16)
    tap("xhat_f", hn2)
    tap("post_f", post2)
    tap("comb_f", comb2)
    for t_ in (h2, hn2, x1, post2, comb2):
        if t_ is not None:
            ttnn.deallocate(t_)


per_chip = lambda t_: [ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731


def compare(a_t, b_t, name, rows_only=None):
    out = []
    for a, b in zip(per_chip(a_t), per_chip(b_t)):
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        if rows_only:
            a, b = a[:rows_only], b[:rows_only]
        out.append(dict(pcc=round(C.pcc(a, b), 7), maxabs=float((a - b).abs().max()), bitwise=bool(torch.equal(a, b)),
                        std=(round(float(a.std()), 6), round(float(b.std()), 6))))  # fmt: skip
    return out


def store_checks(st, xhat):
    """A2's stores per chip vs host references from the host weights the streams were built from (prog.host_w):
    f = lr @ W_fb and gate_out = lr @ W_gb from the DEVICE lr (the K = 512 phases alone); u = x-hat @ W_qkv and
    lr = x-hat @ W_lowrank when the chip's x-hat holds the store's rows. Per store: finite, the non-finite row / column
    tiles, absmax, and PCC / max |d| / rel over the finite elements. Stores of phases outside p3_phases are not scored."""
    hw, out = prog.host_w, {}
    ran = set(prog.p3_phases)
    for k_, ts in st.items():
        if dict(u="qkv", lr="lowrank", f="fb", gate_out="gb")[k_] not in ran:
            out[k_] = "not run (p3_phases)"
            continue
        res = []
        for r, t in enumerate(ts):
            t = t.reshape(-1, t.shape[-1]).float()
            fin = torch.isfinite(t)
            d = dict(shape=list(t.shape), finite=bool(fin.all()), n_nonfinite=int((~fin).sum()),
                     absmax=(float(t[fin].abs().max()) if fin.any() else None))  # fmt: skip
            if not d["finite"]:
                br = ((~fin).any(dim=1).nonzero().flatten() // 32).unique().tolist()
                bc = ((~fin).any(dim=0).nonzero().flatten() // 32).unique().tolist()
                d.update(bad_row_tiles=len(br), row_tiles=-(-t.shape[0] // 32), bad_row_tiles_head=br[:12],
                         bad_col_tiles=len(bc), col_tiles=-(-t.shape[1] // 32), bad_col_tiles_head=bc[:12])  # fmt: skip
            ref, why = None, None
            if k_ in ("f", "gate_out"):
                lr_ = st["lr"][r].reshape(-1, st["lr"][r].shape[-1]).float()
                w_ = hw["fb" if k_ == "f" else "gb"][r]
                ref = lr_[:, : w_.shape[0]] @ w_
            else:
                # the projections read every row: the SP chips' x-hat rows in rank order (A1 leaves each chip its own);
                # every 8th row (the reference matmul is ~200 GFLOP per chip at full size)
                xs_ = [v.reshape(-1, v.shape[-1]).float() for v in xhat]
                x_ = xs_[r] if xs_[r].shape[0] == t.shape[0] else torch.cat(xs_)
                w_ = hw["qkv" if k_ == "u" else "lowrank"][r]
                if x_.shape[0] == t.shape[0] and x_.shape[1] == w_.shape[0]:
                    ref = x_[::8] @ w_[:, : t.shape[1]]
                    t, fin = t[::8], fin[::8]
                else:
                    why = f"x-hat {list(x_.shape)} vs store {list(t.shape)} / W {list(w_.shape)}"
            if ref is not None and ref.shape == t.shape:
                m = fin & torch.isfinite(ref)
                a, b = t[m], ref[m]
                d.update(pcc=(round(C.pcc(a, b), 7) if a.numel() > 1 else None), maxabs=float((a - b).abs().max()) if a.numel() else None,
                         rel=round(float((a - b).norm()) / (float(b.norm()) or 1.0), 6), ref_absmax=float(ref.abs().max()),
                         ref_finite=bool(torch.isfinite(ref).all()))  # fmt: skip
            else:
                d["ref"] = why or f"ref {None if ref is None else list(ref.shape)} vs store {list(t.shape)}"
            res.append(d)
        out[k_] = res
    ok = all(d["finite"] and (d.get("pcc") is None or d["pcc"] >= STORE_BAR)
             for v in out.values() if isinstance(v, list) for d in v)  # fmt: skip
    return out, ok


# every program compiled before the FIRST capture (program A first: its runtime-arg audit + launch census run here)
S.log(f"{tag}: program A compile launch")
fn_pmk()
ttnn.synchronize_device(mesh)
S.log(f"{tag}: audit {prog.engine.arg_audits}; launch census {prog.layer.launch_census}")
S.log(f"{tag}: op path compile launch")
fn_ref()
ttnn.synchronize_device(mesh)
A1_ONLY = bool(SCOPES) and all(set(v) <= {"P1"} for v in SCOPES)
BISECT = bool(SCOPES) and not A1_ONLY and all(set(v) <= {"P1", "P2", "P3"} for v in SCOPES)  # A1 + A2: no FFN input yet
CRIT = "xhat" if (A1_ONLY or BISECT) else "xhat_f"
HAS_P3 = any("P3" in sc for sc in prog.scopes)  # A2 runs: its stores are scored (status needs them)
row = dict(test=(f"PMK assembly A1 alone (mhc pre) L{L} tp{NDEV} vs rms(mhc_pre(attn, x)) (bucket {c}), traced; criterion x-hat (H1) PCC >= {PCC_BAR} per chip"
                if A1_ONLY else
                f"PMK assembly program A (A1..A5: mhc / 7a C1 + proj PfStream / kda / RSR) L{L} at tp{NDEV} vs the op path up to the FFN input (bucket {c}), traced; criterion x-hat_f PCC >= {PCC_BAR} per chip"),
           milestone="PMK-A", layer=L, tp=NDEV, bucket=c, census=prog.layer.launch_census, arg_audit=prog.engine.arg_audits,
           streams=prog.report["streams"], knobs=prog.report.get("knobs"), rs8=prog.report.get("rs8"), qbase=prog.report.get("qbase"), us_bar=US_BAR,
           mock=C.MOCK, tag=tag)  # fmt: skip
if C.MOCK:
    row["status"] = "mock"
    if HAS_P3:  # the store checks' host code on synthetic stores of the real shapes (no device read under the mock)
        g_ = torch.Generator().manual_seed(0)
        dts = lambda t_: ttnn.get_device_tensors(t_) if NDEV > 1 else [t_]  # noqa: E731
        st_h = {k_: [torch.randn(*list(d.shape), generator=g_) for d in dts(t_)] for k_, t_ in prog.stores().items()}
        st_h["f"][0].reshape(-1, st_h["f"][0].shape[-1])[64:96, 40:72] = float("inf")  # the non-finite tile report
        xh = [torch.randn(*list(d.shape), generator=g_) for d in dts(prog.hns[0])]
        row["stores"], row["stores_ok"] = store_checks(st_h, xh)
        S.log(f"{tag}: mock store checks (synthetic) {row['stores']} ok {row['stores_ok']}")
    if SPLIT:  # the mock exercises the split's group partition and the per-group descriptor builds / launches
        groups, cur = [], []
        for sc in prog.scopes:
            cur.append(sc)
            if any(pid in sc for pid in SPLIT):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        for g_ in groups:
            prog.layer.engine.programs = [set(sc) for sc in g_]
            prog.layer.order = [set(sc) for sc in g_]
            prog(x_p, ci)
            ttnn.synchronize_device(mesh)
        prog.layer.engine.programs = [set(sc) for sc in prog.scopes]
        prog.layer.order = [set(sc) for sc in prog.scopes]
        row["split"] = dict(groups=[["+".join(sc) for sc in g_] for g_ in groups], mock_launched=True)
        S.log(f"{tag}: SPLIT groups {row['split']['groups']} launched under the mock")
else:
    res, us = {}, {}
    # ONE live trace at a time: program A first (its outputs are persistent: hns / ys / x_p / state), then the reference
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn_pmk()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    # the trace bytes of program A alone (the only live trace) per chip: the trace region's allocation (Stage 14, the
    # trace-light PMK A/Bs are judged on this number)
    try:
        tv_ = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        row["trace_bytes_per_chip"] = int(tv_.total_bytes_allocated_per_bank) * int(tv_.num_banks)
    except Exception as e_:  # noqa: BLE001
        row["trace_bytes_per_chip"] = f"n/a: {str(e_)[:80]}"
    row["c_trim"] = os.environ.get("PMK_PFS_C_TRIM", "1") == "1"
    S.log(f"{tag}: program A trace bytes per chip {row['trace_bytes_per_chip']} (PMK_PFS_C_TRIM {int(row['c_trim'])})")
    calls = prog.counters["calls"]
    t2 = time.perf_counter()
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    first_us = (time.perf_counter() - t2) * 1e6
    ttnn.synchronize_device(mesh)
    assert prog.counters["calls"] == calls, ("eager guard: the capture is the only launch", prog.counters)
    pm = dict(xhat=per_chip(prog.hns[0]), post0=per_chip(prog.posts[0]), comb0=per_chip(prog.combs[0]))
    ok_st = True
    if HAS_P3:  # A2's stores (u / lr / f / gate_out) vs their host references on every chip, whenever A2 runs (15:45:26Z / 16:05:22Z)
        st_h = {k_: per_chip(t_) for k_, t_ in prog.stores().items()}
        PMK_STORES = st_h  # (op_taps: compared with the op path's own outputs after the reference trace)
        row["stores"], ok_st = store_checks(st_h, pm["xhat"])
        row["stores_ok"] = ok_st
        # the A/B arms' bitwise check (16's L1, GLM53_FS_C1X_ROWFIRST): per chip sha of every A2 store and of x-hat
        _sha = lambda t_: hashlib.sha1(t_.contiguous().numpy().tobytes()).hexdigest()[:16]  # noqa: E731
        row["store_sha"] = {k_: [_sha(t_) for t_ in v_] for k_, v_ in st_h.items()}
        row["xhat_sha"] = [_sha(t_) for t_ in pm["xhat"]]
        row["c1x_rowfirst"] = os.environ.get("GLM53_FS_C1X_ROWFIRST", "0")
        row["p3_phases"] = prog.report.get("p3_phases")
        row["a2_heads"] = prog.a2_heads
        S.log(f"{tag}: A2 stores vs host references {row['stores']}")
    if not (A1_ONLY or BISECT):
        pm.update(y=per_chip(prog.ys[0]), x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]), post_f=per_chip(prog.posts[1]),
                  comb_f=per_chip(prog.combs[1]), S=per_chip(state_p.S), C=per_chip(state_p.C))  # fmt: skip
    probes = prog.probes()
    perf_a = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} program A")
    ttnn.release_trace(mesh, tid)
    if SPLIT:  # the split-trace taps (the single-trace results above stay the row's main compare)
        groups, cur = [], []
        for sc in prog.scopes:
            cur.append(sc)
            if any(pid in sc for pid in SPLIT):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        split = dict(groups=[["+".join(sc) for sc in g_] for g_ in groups], taps={})
        host = lambda t_: [h_.clone() for h_ in per_chip(t_)]  # noqa: E731
        tids = []
        try:
            for gi, g_ in enumerate(groups):
                prog.layer.engine.programs = [set(sc) for sc in g_]
                prog.layer.order = [set(sc) for sc in g_]
                t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    if gi == 0:
                        ttnn.copy(x0, x_p)
                        zero(state_p.S)
                        zero(state_p.C)
                    prog(x_p, ci)
                finally:
                    ttnn.end_trace_capture(mesh, t_, cq_id=0)
                tids.append(t_)
                ttnn.execute_trace(mesh, t_, cq_id=0, blocking=True)
                ttnn.synchronize_device(mesh)
                pids = {pid for sc in g_ for pid in sc}
                tap = {}
                if "P3" in pids or gi == len(groups) - 1:
                    tap["stores"] = {k_: host(t_) for k_, t_ in prog.stores().items()}
                if "P6" in pids or gi == len(groups) - 1:
                    tap.update(S=host(state_p.S), C=host(state_p.C), on=host(prog.scr["on"]))
                split["taps"][gi] = tap
                split.setdefault("us", {})["+".join("+".join(sc) for sc in g_)] = S.replay_stats(
                    mesh, t_, replays=max(10, REPLAYS // 5), label=f"{tag} split {gi}")["median_us"]
        finally:
            for t_ in tids:
                ttnn.release_trace(mesh, t_)
            prog.layer.engine.programs = [set(sc) for sc in prog.scopes]
            prog.layer.order = [set(sc) for sc in prog.scopes]
        row["split"] = split
    # per-program us: each of the A-programs captured ALONE (its inputs are the persistent DRAM tensors the full
    # trace just produced), replayed; the sum is reported next to the full trace (one live trace at a time)
    per_program = {}
    all_scopes = list(prog.scopes)
    for sc in all_scopes:
        prog.layer.engine.programs = [set(sc)]
        prog.layer.order = [set(sc)]
        t_ = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            prog(x_p, ci)
        finally:
            ttnn.end_trace_capture(mesh, t_, cq_id=0)
        st_ = S.replay_stats(mesh, t_, replays=max(10, REPLAYS // 5), label=f"{tag} {'+'.join(sc)}")
        per_program["+".join(sc)] = st_["median_us"]
        ttnn.release_trace(mesh, t_)
    prog.layer.engine.programs = [set(sc) for sc in all_scopes]
    prog.layer.order = [set(sc) for sc in all_scopes]
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn_ref()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    if OP_TAPS and "PMK_STORES" in globals():  # program A's lr / f / gate_out vs the op path's, per chip (common rows / cols)
        def _vs_op(pm_list, op_t, round_bf16=False):
            out_ = []
            for a_, b_ in zip(pm_list, per_chip(op_t)):
                a_, b_ = a_.reshape(-1, a_.shape[-1]).float(), b_.reshape(-1, b_.shape[-1]).float()
                r_, c_ = min(a_.shape[0], b_.shape[0]), min(a_.shape[1], b_.shape[1])
                a_, b_ = a_[:r_, :c_], b_[:r_, :c_]
                if round_bf16:  # the PMK store is fp32 raw, the op's is bf16: compare on the op's grid
                    a_ = a_.to(torch.bfloat16).float()
                out_.append(dict(shape=[r_, c_], bitwise=bool(torch.equal(a_, b_)), maxabs=float((a_ - b_).abs().max()),
                                 pcc=round(C.pcc(a_, b_), 9), rel=round(float((a_ - b_).norm()) / (float(b_.norm()) or 1.0), 9)))  # fmt: skip
            return out_
        row["op_taps"] = dict(lr=_vs_op(PMK_STORES["lr"], taps["op.lr"]), f=_vs_op(PMK_STORES["f"], taps["op.ds_fb"]),
                              gate_out=_vs_op(PMK_STORES["gate_out"], taps["op.ds_gb"], round_bf16=True),
                              lr_exact=bool(prog.report.get("lr_exact")))  # fmt: skip
        S.log(f"{tag}: A2 stores vs the OP path's own outputs {row['op_taps']}")
    rf = dict(xhat=per_chip(taps["H1.xhat"]), post0=per_chip(taps["post0"]), comb0=per_chip(taps["comb0"]), y=per_chip(taps["H6.y"]),
              x1=per_chip(taps["x1"]), xhat_f=per_chip(taps["xhat_f"]), post_f=per_chip(taps["post_f"]), comb_f=per_chip(taps["comb_f"]),
              S=per_chip(pl.scratch.S), C=per_chip(pl.scratch.C))  # fmt: skip
    perf_op = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} op path")
    ttnn.release_trace(mesh, tid)
    cmp = {}
    for name in pm:
        cs = []
        for a, b in zip(pm[name], rf[name]):
            a, b = a.reshape(-1, a.shape[-1]).float(), b.reshape(-1, b.shape[-1]).float()
            cs.append(dict(pcc=round(C.pcc(a, b), 7), maxabs=float((a - b).abs().max()), bitwise=bool(torch.equal(a, b)),
                           std=(round(float(a.std()), 6), round(float(b.std()), 6))))  # fmt: skip
        cmp[name] = cs
    if SPLIT:
        sp_ = row["split"]
        taps, last = sp_["taps"], max(sp_["taps"])
        first_s = min(g for g, t_ in taps.items() if "S" in t_)
        first_st = min(g for g, t_ in taps.items() if "stores" in t_)
        bw = lambda a, b: all(torch.equal(x_, y_) for x_, y_ in zip(a, b))  # noqa: E731
        vs_ref = lambda a, ref: [round(C.pcc(x_.reshape(-1, x_.shape[-1]).float(), y_.reshape(-1, y_.shape[-1]).float()), 7)
                                 for x_, y_ in zip(a, ref)]  # noqa: E731
        sp_["verdict"] = dict(
            s_after_a3_vs_ref=vs_ref(taps[first_s]["S"], rf["S"]), s_after_a5_vs_ref=vs_ref(taps[last]["S"], rf["S"]),
            s_a3_eq_a5_bitwise=bw(taps[first_s]["S"], taps[last]["S"]),
            c_after_a3_vs_ref=vs_ref(taps[first_s]["C"], rf["C"]),
            on_after_a3_finite=[bool(torch.isfinite(t_).all()) for t_ in taps[first_s]["on"]],
            on_a3_eq_a5_bitwise=bw(taps[first_s]["on"], taps[last]["on"]),
            stores_a2_eq_a5_bitwise={k_: bw(taps[first_st]["stores"][k_], taps[last]["stores"][k_]) for k_ in ("u", "lr", "f", "gate_out")},
            stores_after_a2_finite={k_: [bool(torch.isfinite(t_).all()) for t_ in taps[first_st]["stores"][k_]] for k_ in ("u", "lr", "f", "gate_out")},
            us=sp_.get("us"))  # fmt: skip
        for g_ in list(taps):  # the host tensors do not go into the row
            taps[g_] = sorted(taps[g_].keys())
        S.log(f"{tag}: SPLIT verdict {sp_['verdict']}")
    if bool(args.get("mhc_fp64", True)):  # each side's mHC coefficients vs a host fp64 reference of ITS OWN input
        # (program 18:48:42Z, the realgate drift: is the PMK-vs-op comb / post gap the PMK's or the op's?). attn: both
        # from the golden rows; ffn: PMK from its own x1, the op from its own x1 -- the mHC arithmetic alone, not its input
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as MHC64
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR as _CD, MANIFEST as _MF, hf_config as _hfc
        from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource as _TCS
        cfg64, src64 = _hfc(), _TCS(_CD, _MF)  # the checkpoint's mHC weights (the unit's op path is the composite mHC)

        def _planes64(sub, x2):  # x2 [T, 4 x 4096] host -> [T, 24] fp64 (pre | post | comb)
            n_, H_ = cfg64.hc_mult, cfg64.hidden_size
            fn_ = src64.host(L, f"hc_{sub}_fn").reshape(n_ * H_, (2 + n_) * n_).double()
            sc_ = src64.host(L, f"hc_{sub}_scale").double().reshape(-1)
            sc_ = torch.cat([sc_[0].repeat(n_), sc_[1].repeat(n_), sc_[2].repeat(n_ * n_)])
            b_ = src64.host(L, f"hc_{sub}_base").double().reshape(-1)
            x2 = x2.double()
            z = (x2 @ fn_) * torch.rsqrt((x2 * x2).sum(-1) / x2.shape[-1] + cfg64.rms_norm_eps)[:, None] * sc_ + b_
            return MHC64.host_planes(z, cfg64.hc_sinkhorn_iters, cfg64.hc_eps).T

        rows2 = rows.reshape(rows.shape[0], -1)
        m64 = {}
        try:
            for side, src in (("pmk", pm), ("op", rf)):
                per = []
                for i in range(len(src["post0"])):
                    r_ = src["post0"][i].numel() // 4
                    xa = rows2[i * r_:(i + 1) * r_] if r_ < rows2.shape[0] else rows2
                    pa = _planes64("attn", xa)
                    pf_ = _planes64("ffn", src["x1"][i].reshape(r_, -1))
                    d_ = lambda a_, b_: float((a_.reshape(b_.shape).double() - b_).abs().max())  # noqa: E731
                    per.append(dict(post0=d_(src["post0"][i], pa[:, 4:8]), comb0=d_(src["comb0"][i], pa[:, 8:]),
                                    post_f=d_(src["post_f"][i], pf_[:, 4:8]), comb_f=d_(src["comb_f"][i], pf_[:, 8:])))
                m64[side] = per
        except Exception as e_:  # the analysis must not void the device row
            m64 = dict(error=repr(e_))
        row["mhc_vs_fp64"] = m64
        S.log(f"{tag}: mHC coefficients vs host fp64 of each side's own input (maxabs per chip) {m64}")
    if args.get("dump_x1"):  # each path's x1 (the residual the served FFN pre / router read) for the routing analysis
        torch.save(dict(pmk=[t.cpu() for t in pm.get("x1", [])], op=[t.cpu() for t in rf.get("x1", [])], sp=sp, n=N, layer=L,
                        input_pt=args.get("input_pt")), args["dump_x1"])  # fmt: skip
        S.log(f"{tag}: x1 of both paths -> {args['dump_x1']}")
    ok = all(ch["pcc"] >= PCC_BAR for ch in cmp[CRIT])
    ok_us = US_BAR is None or perf_a["median_us"] <= float(US_BAR)
    ok = ok and ok_us and ok_st
    if MHC_PROBE and prog.taps is not None:  # the root pass tile by tile (chip 0; blocks 0, 1, last): WHICH stage breaks
        from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as MHC
        from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
        cfg_ = hf_config()
        taps_h = per_chip(prog.taps)[0].reshape(-1, 32)
        post_ref = rf["post0"][0].reshape(-1, 4)
        comb_ref = rf["comb0"][0].reshape(-1, 16)
        probe_rows = {}
        for rb in sorted({0, 1, prog.mp.n_blocks - 1}):
            d = MHC.decode_taps(taps_h, prog.mp, rb, 0)  # z / pass / iters1 / iters2: [24, 32] (plane, token)
            z_k = d["z"].double()
            pr = post_ref[rb * 32:(rb + 1) * 32].double()
            z_ref_post = torch.logit((pr / 2).clamp(1e-6, 1 - 1e-6)).T  # the reference's own z[4:8] from its post
            host_pass = MHC.host_planes(z_k.T, cfg_.hc_sinkhorn_iters, cfg_.hc_eps)  # the pass the kernel SHOULD produce from its z
            kp = d["pass"].double()
            probe_rows[f"rb{rb}"] = dict(
                z_post_vs_ref_maxabs=float((z_k[4:8] - z_ref_post).abs().max()), z_absmax=float(z_k.abs().max()),
                z_finite=bool(torch.isfinite(z_k).all()),
                pass_vs_host_of_own_z_maxabs=float((kp - host_pass).abs().max()),
                post_from_pass_vs_ref_maxabs=float((kp[4:8].T - pr).abs().max()),
                comb_from_pass_vs_ref_maxabs=float((kp[8:].T - comb_ref[rb * 32:(rb + 1) * 32].double()).abs().max()),
                iters1_vs_host1=float((d["iters1"].double() - MHC.host_planes(z_k.T, 1, cfg_.hc_eps)).abs().max()),
                pack=MHC.decode_pack_diag(taps_h, prog.mp, rb, 0))  # rev 6: the sync RISC's tap-3 record (DIAG_MAGIC, r = k in part P1)
        # the verified pack's diagnostics over EVERY block of chip 0 (mhc README §9e / 9f): valid = the record's magic;
        # attempts > 1 with marker_ok = the pack raced the pass and the retry fixed it; marker_bad / lost = the marker
        # never reached the packed page (an address / bank problem, not timing) -> the next step is a math-side probe
        pds = [MHC.decode_pack_diag(taps_h, prog.mp, rb, 0) for rb in range(prog.mp.n_blocks)]
        row["mhc_pack"] = dict(pack_attempts_max=max(max(p_["attempts1"], p_["attempts2"]) for p_ in pds),
                               pack_marker_bad_blocks=[i for i, p_ in enumerate(pds) if not p_["marker_ok"]],
                               pack_lost_blocks=[i for i, p_ in enumerate(pds) if p_["pack_lost"]],
                               pack_diag_invalid_blocks=[i for i, p_ in enumerate(pds) if not p_["valid"]],
                               pack_marker_unexpected_blocks=[i for i, p_ in enumerate(pds) if not p_["marker_expected"]])  # fmt: skip
        row["mhc_probe"] = probe_rows
        S.log(f"{tag}: mhc root-pass probe {probe_rows}")
        S.log(f"{tag}: mhc verified pack {row['mhc_pack']}")
    pmin = lambda k: (min(ch["pcc"] for ch in cmp[k]) if k in cmp else None)  # noqa: E731
    row.update(compare=cmp, probes=probes, first_replay_us=round(first_us, 1), us_program_a=perf_a["median_us"], us_per_program=per_program,
               us_op_path=perf_op["median_us"], perf_a=perf_a, perf_op=perf_op, criterion=CRIT, pcc_crit=pmin(CRIT), pcc_xhat=pmin("xhat"),
               pcc_xhat_f=pmin("xhat_f"), pcc_s=pmin("S"), pcc_c=pmin("C"),
               bitwise_s=(all(ch["bitwise"] for ch in cmp["S"]) if "S" in cmp else None), us_ok=ok_us,
               status="pass" if ok else "fail")  # fmt: skip
    S.log(f"{tag}: compare {cmp}")
    S.log(f"{tag}: probes {probes}")
    S.log(f"{tag}: program A {perf_a['median_us']:.0f} us vs op path {perf_op['median_us']:.0f} us; status {row['status']}")
    if bool(args.get("record", True)):
        C.record("pcc", dict(test=row["test"], milestone="PMK-A", layer=L, tp=NDEV, bucket=c, criterion=CRIT, pcc=row["pcc_crit"],
                             pcc_xhat=row["pcc_xhat"], pcc_xhat_f=row["pcc_xhat_f"], pcc_s=row["pcc_s"], pcc_c=row["pcc_c"],
                             bitwise_s=row["bitwise_s"], status=row["status"], tag=tag))  # fmt: skip
        C.record("perf", dict(test=f"PMK assembly program A L{L} tp{NDEV} traced replay vs the op path up to the FFN input",
                              milestone="PMK-A", layer=L, tp=NDEV, bucket=c, us=perf_a["median_us"], op_us=perf_op["median_us"],
                              replays=perf_a["replays"], probes=probes, status=row["status"], tag=tag))  # fmt: skip
RESULT = row
print("PMK_ASM_A_UNIT", {k: v for k, v in row.items() if k not in ("probes", "perf_a", "perf_op")}, flush=True)
