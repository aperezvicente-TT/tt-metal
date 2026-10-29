# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 TP2 pair / the TP8 2x4; watched once then unwatched): the device unit of AD-1-LITE (the
attention-half merge's first piece, program 2026-10-27 00:2xZ; doc/ttnn/16-prefill-production/pmk/merge/README.md):
the sparse_dsa layer's A1 + the MLA input projections in ONE program -- P1 (mHC pre) + P2 (the C1 AG of x-hat to all rows)
+ P3 (PfStream q_a | kv_a bf16 out at HiFi2, PfStream kg = x-hat [W_k | W_gate] fp32 out at HiFi4) -- vs the SAME phases
as separate programs ((P1) | (P2, P3): the unmerged form), both TRACED, the stores poisoned (NaN) before every replay.

Criteria per chip: merged vs unmerged BITWISE on qk [C, 2048] bf16, kg_pre [C, 256] fp32 and x-hat (hn[0]), every compared
replay. Reported (not gated): PCC vs the op path's linears -- pf_mm(ds_qa_kva, gather(x-hat)) / ttnn.linear(gather(x-hat),
w_kg, fp32, _ckc_fp32) -- on the PMK's own x-hat (the linears alone) and on the op mHC pre's x-hat (A1 numerics included);
the traced replay us of the merged and the unmerged programs. The layer-output PCC vs the CPU golden (32K, every chunk)
is p3_long_check's `ad1` arm. One trace alive at a time for the op reference; every program compiled before the first
capture; no host write inside a capture (the poison sources are allocated before the first capture, copied on device).
ARGS: layer (3), rows (2048), det (5: compared poisoned replays), replays (50: timing), tag, record (True), inplace
("1" default: k4b's PMK_AD1_INPLACE -- the streams read the op's resident weights, kg from idx.w_kg; "wx": kg from the
decode w_x; "0": AD-1's own bf16 copies; "ab": the in-place programs AND a merged copies program (PMK_AD1_INPLACE=0) in
the same process, in place vs copies bitwise on every compared replay besides merged vs unmerged)."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 3))
N = int(args.get("rows", 2048))
DET = int(args.get("det", 5))
MODE = str(args.get("mode", "lite"))  # "lite" (P1 + P2 + P3) | "full" (AD-1A: + D2) | "full+qidx" (AD-3a: + P18 q_idx)
assert MODE in ("lite", "full", "full+qidx"), MODE
FULL = MODE in ("full", "full+qidx")
QIDX = MODE == "full+qidx"
REPLAYS = int(args.get("replays", 50))
INPLACE = str(args.get("inplace", "1")).lower()  # k4b: PMK_AD1_INPLACE of the programs ("ab": "1" + a copies arm)
assert INPLACE in ASM.AD1_INPLACE_MODES + ("ab",), INPLACE
AB = INPLACE == "ab"
IP = "1" if AB else INPLACE
tag = args.get("tag", f"pmk-ad1-unit-L{L}")
NDEV = int(mesh.get_num_devices())
os.environ["GLM53_PF_DSA"] = "1"
os.environ.setdefault("GLM53_PF_MHC", "comp")
os.environ.setdefault("GLM53_PF_MHC_POST", "fused")
for k_, v_ in (("GLM53_PF_MHC_PRE", "fused7a"), ("GLM53_PF_MHC_FPU", "1"), ("GLM53_PF_MHC_POST7A", "1")):
    os.environ.setdefault(k_, v_)
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
t0 = time.time()
# the stack backend needs one non-DSA layer in the set (stack_model._build_tp: its first non-dsa layer's tp object): the
# built layers are [2, L] as p3_long_check's (L2 = dense_kda), the unit runs L's programs only
LAYERS = [int(v) for v in args.get("layers", [2, L])]
assert L in LAYERS, (L, LAYERS)
LI = LAYERS.index(L)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
assert pf.dsa, "TPPrefill built without GLM53_PF_DSA"
pl = pf.layers[LI]
assert pl.kind == PT.SPARSE_DSA, (L, pl.kind, "AD-1-lite is the sparse_dsa layers' program")
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
assert sp and c == N, (sp, c, N)
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
try:
    rows, _ = S.golden_layer(L, N)
except FileNotFoundError:  # the model is causal: the first N rows of the 32K golden are the N-token golden's rows
    big, _ = S.golden_layer(L, 32768)
    rows = big.reshape(32768, -1)[:N].clone()
    del big
rows = rows.reshape(N, 4, 4096).float()
x0 = pf.streams_to_device(rows, sp)  # pristine (staged once, outside every trace)
x_m = pf.streams_to_device(rows, sp)  # the merged program's residual (its address is baked into the program)
x_u = pf.streams_to_device(rows, sp)  # the unmerged programs'
x_r = pf.streams_to_device(rows, sp)  # the op reference's
x_c = pf.streams_to_device(rows, sp) if AB else None  # the copies arm's merged program (inplace "ab")
S.log(f"{tag}: pf ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c})")
t1 = time.time()
kw = dict(probe=False, attn=False, dsa_ad1=MODE, pfs_park=int(args.get("pfs_park", 4)), ad1_inplace=IP)
# the first-compute audit (after the compile launches) reads the generated member composites; restored there
_keep_src = os.environ.get("GLM53_PMK_KEEP_SRC")
os.environ["GLM53_PMK_KEEP_SRC"] = "1"
SC_M = [ASM.AD1Q_SCOPE] if QIDX else [ASM.AD1F_SCOPE] if FULL else [ASM.AD1_SCOPE]
SC_U = [("P1",), ("P2", "P3")] + ([("P17",)] if FULL else []) + ([("P18",)] if QIDX else [])
prog_m = ASM.PmkProgramA(pf, LI, c, sp, x_m, None, ci.valid_end, scopes=SC_M, **kw)
prog_u = ASM.PmkProgramA(pf, LI, c, sp, x_u, None, ci.valid_end, scopes=SC_U, **kw)
# "ab": the same merged program over AD-1's own bf16 copies (PMK_AD1_INPLACE=0), its own residual and stores
prog_c = ASM.PmkProgramA(pf, LI, c, sp, x_c, None, ci.valid_end, scopes=SC_M, **dict(kw, ad1_inplace="0")) if AB else None
PROGS = [(prog_m, "merged"), (prog_u, "unmerged")] + ([(prog_c, "copies")] if AB else [])
for p_, nm_ in PROGS:
    assert p_.ad1_inplace == ("0" if nm_ == "copies" else IP), (nm_, p_.ad1_inplace, IP)
    assert (
        p_.dsa_ad1
        and p_.dsa_ad1_full == FULL
        and [tuple(s_) for s_ in p_.scopes] == [tuple(v) for v in (SC_U if nm_ == "unmerged" else SC_M)]
    ), (nm_, p_.scopes)
    # the hand-off tensors the op remainder reads: shapes / dtypes / layouts (the write extents = the stores' own)
    assert (
        tuple(p_.qk.shape) == (1, 1, N, 2048) and p_.qk.dtype == ttnn.bfloat16 and p_.qk.layout == ttnn.TILE_LAYOUT
    ), p_.qk
    assert tuple(p_.kg_pre.shape) == (1, 1, N, 256) and p_.kg_pre.dtype == ttnn.float32, p_.kg_pre
    assert p_.mm_qa.mt * 32 == N and p_.mm_kg.mt * 32 == N, (p_.mm_qa.mt, p_.mm_kg.mt)
    if QIDX:  # AD-3a: q_idx [C, 4096] bf16 TILE (this rank's local rows first)
        assert (
            tuple(p_.q_idx.shape) == (1, 1, N, 4096)
            and p_.q_idx.dtype == ttnn.bfloat16
            and p_.q_idx.layout == ttnn.TILE_LAYOUT
        ), p_.q_idx
        assert p_.dsa_ad1_qidx, nm_
    if FULL:  # the stores the op remainder reads in the full form
        assert (
            tuple(p_.q_r.shape) == (1, 1, N, 1536)
            and p_.q_r.dtype == ttnn.bfloat16
            and p_.q_r.layout == ttnn.TILE_LAYOUT
        ), p_.q_r
        assert tuple(p_.latent.shape) == (1, 1, N, 512) and p_.latent.dtype == ttnn.bfloat16, p_.latent
        assert (
            tuple(p_.kg_rm.shape) == (1, 1, N, 256)
            and p_.kg_rm.dtype == ttnn.float32
            and p_.kg_rm.layout == ttnn.ROW_MAJOR_LAYOUT
        ), p_.kg_rm
S.log(
    f"{tag}: programs assembled in {time.time() - t1:.1f} s; streams {prog_m.report['streams']}; streamers {prog_m.streamers}; "
    f"windows {prog_m.engine.window_plan}"
)
DRAMW = {nm_: dict(inplace=p_.ad1_inplace, dram_weights=p_.report.get("dram_weights"), sources=p_.report.get("ad1_sources"))
         for p_, nm_ in PROGS}  # fmt: skip
S.log(f"{tag}: AD-1 weight sources (PMK_AD1_INPLACE {INPLACE}) {DRAMW}")
if IP != "0":
    assert all(v_ == 0 for v_ in prog_m.report["dram_weights"].values()), ("in place: no AD-1 weight copy", DRAMW)

# poison (traced-unit-harness-rules 6): NaN SOURCES allocated once here, copied over the stores on device before every replay
_pz = {}


def _nan_like(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _pz:
        _pz[key] = ttnn.from_torch(torch.full(tuple(t_.shape), float("nan")), dtype=t_.dtype, layout=t_.layout, device=mesh,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
    return _pz[key]


STORES = ("qk", "kg_pre", "hn") + (("q_r", "latent", "kg_rm") if FULL else ()) + (("q_idx",) if QIDX else ())
store = lambda p_, k_: p_.hns[0] if k_ == "hn" else getattr(p_, k_)  # noqa: E731
for p_, _nm in PROGS:
    for k_ in STORES:
        _nan_like(store(p_, k_))


def poison(p_):
    for k_ in STORES:
        ttnn.copy(_nan_like(store(p_, k_)), store(p_, k_))


def fn_m():
    ttnn.copy(x0, x_m)
    poison(prog_m)
    prog_m(x_m, ci)


def fn_u():
    ttnn.copy(x0, x_u)
    poison(prog_u)
    prog_u(x_u, ci)


def fn_c():
    ttnn.copy(x0, x_c)
    poison(prog_c)
    prog_c(x_c, ci)


taps = {}


def tap(name, t_):
    if name not in taps:
        taps[name] = ttnn.zeros_like(t_)
    ttnn.copy(t_, taps[name])


mla = pl.mla


def fn_r():
    """the op path's two linears: on the op mHC pre's x-hat (A1 numerics included) and on the merged program's own
    x-hat (prog_m.hns[0]: the linears alone)"""
    ttnn.copy(x0, x_r)
    h, post, comb = pf._mhc_pre(pl, "attn", x_r)
    hn = pl.rms(h, "attn", BF16)
    if hn is not h:
        ttnn.deallocate(h)
    for t_ in (post, comb):
        ttnn.deallocate(t_)
    tap("op.xhat", hn)
    for name, src in (("op", hn), ("pmkx", prog_m.hns[0])):
        hb = pf.gather(src, sp)
        qk = PT.pf_mm(mla.attn.ds_qa_kva, hb)
        tap(f"{name}.qk", qk)
        if FULL:
            op_norms(name, qk)
        ttnn.deallocate(qk)
        kg = ttnn.linear(
            hb, mla.idx.w_kg, dtype=FP32, compute_kernel_config=mla.idx.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tap(f"{name}.kg", kg)
        if FULL:
            rm = mla.idx._kg_rm(None, kg_pre=kg)
            tap(f"{name}.kg_rm", rm)
            ttnn.deallocate(rm)
        ttnn.deallocate(kg)
        if hb is not src:
            ttnn.deallocate(hb)
        if QIDX:  # the op indexer's q_idx of the LOCAL rows (DsaIndexer.select's own calls, on src = the chip's rows)
            qk_l = PT.pf_mm(mla.attn.ds_qa_kva, src)
            cl_ = int(src.shape[-2])
            q_l = ttnn.slice(qk_l, (0, 0, 0, 0), (1, 1, cl_, mla.attn.q_lora))
            q_rl = ttnn.rms_norm(q_l, weight=mla.attn.qa_norm, epsilon=mla.attn.eps, compute_kernel_config=mla.idx.ckc)
            qi = ttnn.linear(
                q_rl, mla.idx.wq_b, dtype=BF16, compute_kernel_config=mla.idx.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tap(f"{name}.q_idx", qi)
            for t_ in (qk_l, q_l, q_rl, qi):
                ttnn.deallocate(t_)
    if QIDX:  # P18 alone: W_qb on the merged program's own q_r, EVERY row (the stores' ground truth, both programs)
        qa_ = ttnn.linear(prog_m.q_r, mla.idx.wq_b, dtype=BF16, compute_kernel_config=mla.idx.ckc,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        tap("lin.q_idx", qa_)
        ttnn.deallocate(qa_)
    if FULL:  # D2 alone: the op's norms on the PMK's OWN linears (qk / kg_pre of the merged program)
        op_norms("d2x", prog_m.qk)
        rm = mla.idx._kg_rm(None, kg_pre=prog_m.kg_pre)
        tap("d2x.kg_rm", rm)
        ttnn.deallocate(rm)
    ttnn.deallocate(hn)


def op_norms(name, qk):
    """the op path's two RMSNorms of a [C, >= 2048] qk (MlaPrefill.__call__'s own calls)"""
    t_ = int(qk.shape[-2])
    q_ = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t_, mla.q_lora))
    c_ = ttnn.slice(qk, (0, 0, 0, mla.q_lora), (1, 1, t_, mla.q_lora + mla.kv_lora))
    q_r = ttnn.rms_norm(q_, weight=mla.attn.qa_norm, epsilon=mla.eps, compute_kernel_config=mla.ckc_fp32)
    lat = ttnn.rms_norm(c_, weight=mla.attn.kva_norm, epsilon=mla.eps, compute_kernel_config=mla.ckc_fp32)
    tap(f"{name}.q_r", q_r)
    tap(f"{name}.latent", lat)
    for x_ in (q_, c_, q_r, lat):
        ttnn.deallocate(x_)


per_chip = lambda t_: [
    ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])
]  # noqa: E731


def cmp(a_list, b_list, ncols=None):
    out = []
    for a, b in zip(a_list, b_list):
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        if ncols:
            a, b = a[:, :ncols], b[:, :ncols]
        fin = torch.isfinite(a) & torch.isfinite(b)
        out.append(dict(bitwise=bool(torch.equal(a, b)), finite=bool(torch.isfinite(a).all()),
                        pcc=round(C.pcc(a[fin], b[fin]), 7) if fin.sum() > 1 else None,
                        maxabs=float((a - b)[fin].abs().max()) if fin.any() else None))  # fmt: skip
    return out


def where(a_list, b_list):
    """where two stores differ, per chip: element count, max |d|, the 32-row and 32-column tiles (first 16 of each)"""
    out = []
    for a, b in zip(a_list, b_list):
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        ne = ~((a == b) | (torch.isnan(a) & torch.isnan(b)))
        r_, c_ = ne.nonzero(as_tuple=True)
        rt_, ct_ = sorted(set((r_ // 32).tolist())), sorted(set((c_ // 32).tolist()))
        out.append(dict(n=int(ne.sum()), maxabs=float((a - b)[ne].abs().max()) if bool(ne.any()) else 0.0,
                        row_tiles=rt_[:16], n_row_tiles=len(rt_), col_tiles=ct_[:16], n_col_tiles=len(ct_)))  # fmt: skip
    return out


# every program compiled before the FIRST capture (both PMK forms, the poison copies, the op reference)
S.log(f"{tag}: compile launches (merged, unmerged{', copies' if AB else ''}, op reference)")
fn_m()
fn_u()
if AB:
    fn_c()
ttnn.synchronize_device(mesh)
fn_r()
ttnn.synchronize_device(mesh)
if _keep_src is None:
    os.environ.pop("GLM53_PMK_KEEP_SRC", None)
else:
    os.environ["GLM53_PMK_KEEP_SRC"] = _keep_src
S.log(
    f"{tag}: audits merged {prog_m.engine.arg_audits}; census merged {prog_m.layer.launch_census}; unmerged {prog_u.layer.launch_census}"
    + (f"; copies {prog_c.layer.launch_census}" if AB else "")
)


def first_compute(p_):
    """{program tag: its rank-0 member compute composite starts with a PMK_FIRST_COMPUTE body (the full start-up)}"""
    srcs = getattr(p_.engine, "sources", None) or {}
    return {
        t_: "#define PMK_FIRST_COMPUTE 1" in s_
        for (t_, r_, k_, n_), s_ in srcs.items()
        if (r_, k_, n_) == ("member", "c", 0)
    }


# AD-3a d0 pair 01:21Z: the per-scope P18 program ran its PfStream compute first with a reconfigure-only start (stale
# unpacker config context from the D2 program): every per-scope program after P1 must start its compute with the full
# start-up, and the merged program (P1's mhc first, its own start-up) must stay as it was
FC = dict(merged=first_compute(prog_m), unmerged=first_compute(prog_u))
if AB:
    FC["copies"] = first_compute(prog_c)
    assert FC["copies"] and not any(FC["copies"].values()), ("copies: P1's mhc is the first member compute", FC)
S.log(f"{tag}: first-compute start-up {FC}")
assert FC["merged"] and not any(FC["merged"].values()), ("merged: P1's mhc is the first member compute", FC)
assert FC["unmerged"].get("P1") is False and all(v_ for t_, v_ in FC["unmerged"].items() if t_ != "P1"), (
    "every per-scope program after P1 starts its PfStream / D2 compute fully",
    FC,
)
row = dict(test=(f"PMK AD-3a (P1 + P2 + P3 + D2 + P18: the MLA input linears + norms + the indexer q_idx) L{L} tp{NDEV} (bucket {c}): merged vs (P1) | (P2, P3) | (D2) | (P18) bitwise on q_idx / q_r / latent / kg_rm / qk / kg_pre / x-hat over {DET} poisoned traced replays; PCC vs the op" if QIDX else f"PMK AD-1A (P1 + P2 + P3 + D2: the MLA input linears + norms) L{L} tp{NDEV} (bucket {c}): merged vs (P1) | (P2, P3) | (D2) "
                 f"bitwise on q_r / latent / kg_rm / qk / kg_pre / x-hat over {DET} poisoned traced replays; PCC vs the op norms" if FULL else
                 f"PMK AD-1-lite (P1 + P2 + P3 q_a|kv_a + kg) L{L} tp{NDEV} (bucket {c}): merged vs (P1) | (P2, P3) bitwise "
                 f"on qk / kg_pre / x-hat over {DET} poisoned traced replays; PCC vs the op linears"), mode=MODE,
           milestone="PMK-AD1", layer=L, tp=NDEV, bucket=c, census=dict(merged=prog_m.layer.launch_census, unmerged=prog_u.layer.launch_census),
           arg_audit=prog_m.engine.arg_audits, streams=prog_m.report["streams"], qbase=prog_m.report.get("qbase"), inplace=INPLACE,
           dram_weights=DRAMW, census_copies=prog_c.layer.launch_census if AB else None,
           windows={str(sorted(k_)): v_ for k_, v_ in (prog_m.engine.window_plan or {}).items()}, mock=C.MOCK, tag=tag)  # fmt: skip
if C.MOCK:
    row["status"] = "mock"
else:
    tids = {}
    for nm_, fn_ in [("m", fn_m), ("u", fn_u)] + ([("c", fn_c)] if AB else []):  # the PMK traces allocate nothing: all stay live
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn_()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tids[nm_] = tid
    calls = tuple(p_.counters["calls"] for p_, _nm in PROGS)
    det = []
    for i in range(DET):
        for nm_ in tids:
            ttnn.execute_trace(mesh, tids[nm_], cq_id=0, blocking=True)
        hm, hu = {k_: per_chip(store(prog_m, k_)) for k_ in STORES}, {k_: per_chip(store(prog_u, k_)) for k_ in STORES}
        d = {k_: cmp(hm[k_], hu[k_]) for k_ in STORES}
        bad = {k_: where(hm[k_], hu[k_]) for k_, v in d.items() if not all(ch["bitwise"] for ch in v)}
        ab = {}
        if AB:  # in place (merged) vs AD-1's own copies (merged): the same values streamed from the op tensors
            hc = {k_: per_chip(store(prog_c, k_)) for k_ in STORES}
            dc = {k_: cmp(hm[k_], hc[k_]) for k_ in STORES}
            ab = dict(ab_bitwise={k_: all(ch["bitwise"] for ch in v) for k_, v in dc.items()},
                      ab_where={k_: where(hm[k_], hc[k_]) for k_, v in dc.items() if not all(ch["bitwise"] for ch in v)},
                      **({"ab_detail": dc} if i == 0 else {}))  # fmt: skip
            del hc
        det.append(dict(replay=i, bitwise={k_: all(ch["bitwise"] for ch in v) for k_, v in d.items()},
                        finite={k_: all(ch["finite"] for ch in v) for k_, v in d.items()}, where=bad, **ab,
                        **({"detail": d} if i == 0 else {})))  # fmt: skip
        S.log(
            f"{tag}: replay {i}: bitwise {det[-1]['bitwise']} finite {det[-1]['finite']}"
            + (f" differ {bad}" if bad else "")
            + (f"; in place vs copies {ab['ab_bitwise']}" + (f" differ {ab['ab_where']}" if ab["ab_where"] else "") if AB else "")
        )
        del hm, hu
    assert tuple(p_.counters["calls"] for p_, _nm in PROGS) == calls, "eager guard: the captures are the only launches"
    perf_m = S.replay_stats(mesh, tids["m"], replays=REPLAYS, label=f"{tag} merged AD-1")
    perf_u = S.replay_stats(mesh, tids["u"], replays=REPLAYS, label=f"{tag} unmerged (P1) | (P2, P3)")
    perf_c = S.replay_stats(mesh, tids["c"], replays=REPLAYS, label=f"{tag} merged, AD-1 copies") if AB else None
    ttnn.execute_trace(
        mesh, tids["m"], cq_id=0, blocking=True
    )  # the merged stores fresh for the reference's 'pmkx' arm
    for tid in tids.values():
        ttnn.release_trace(mesh, tid)
    pm = dict(qk=per_chip(prog_m.qk), kg=per_chip(prog_m.kg_pre), xhat=per_chip(prog_m.hns[0]))
    if FULL:
        pm.update(q_r=per_chip(prog_m.q_r), latent=per_chip(prog_m.latent), kg_rm=per_chip(prog_m.kg_rm))
    if QIDX:  # this chip's LOCAL rows = the store's first C_l rows (P18's per-rank row permutation)
        cl_ = int(prog_m.hns[0].shape[-2])
        pm["q_idx"] = per_chip(prog_m.q_idx)
        pm["q_idx_local"] = [t_.reshape(-1, t_.shape[-1])[:cl_] for t_ in pm["q_idx"]]
        pu_qidx = per_chip(prog_u.q_idx)  # the unmerged store of its last replay (the reference trace leaves it)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)  # the op reference: the only live trace now
    try:
        fn_r()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    vs_op = dict(qk_linear_only=cmp(pm["qk"], per_chip(taps["pmkx.qk"]), ncols=2048),
                 kg_linear_only=cmp(pm["kg"], per_chip(taps["pmkx.kg"])),
                 qk_vs_op_chain=cmp(pm["qk"], per_chip(taps["op.qk"]), ncols=2048),
                 kg_vs_op_chain=cmp(pm["kg"], per_chip(taps["op.kg"])),
                 xhat_vs_op=cmp(pm["xhat"], per_chip(taps["op.xhat"])))  # fmt: skip
    if FULL:  # D2 alone (the op norms of the PMK's own linears), then the whole chain vs the op on its own x-hat / A1
        for k_ in ("q_r", "latent", "kg_rm"):
            vs_op[f"{k_}_d2_only"] = cmp(pm[k_], per_chip(taps[f"d2x.{k_}"]))
            vs_op[f"{k_}_vs_op_linears"] = cmp(pm[k_], per_chip(taps[f"pmkx.{k_}"]))
            vs_op[f"{k_}_vs_op_chain"] = cmp(pm[k_], per_chip(taps[f"op.{k_}"]))
    if (
        QIDX
    ):  # q_idx: on the PMK's own local x-hat (the op's qk / rms / W_qb chain) and on the op mHC pre's x-hat (A1 incl.)
        vs_op["q_idx_vs_op_linears"] = cmp(pm["q_idx_local"], per_chip(taps["pmkx.q_idx"]))
        vs_op["q_idx_vs_op_chain"] = cmp(pm["q_idx_local"], per_chip(taps["op.q_idx"]))
        ref = []  # W_qb(q_r) rows in each rank's store order (P18's permutation: the rank's local blocks first)
        for rank_, t_ in enumerate(per_chip(taps["lin.q_idx"])):
            p_ = prog_m.c1._pos(*divmod(rank_, prog_m.C))
            perm = torch.empty(len(prog_m.members), dtype=torch.long)
            for j_ in range(len(prog_m.members)):
                perm[prog_m._qidx_row(j_, p_)] = j_
            t_ = t_.reshape(-1, t_.shape[-1])
            ref.append(t_.reshape(len(perm), 32, -1)[perm].reshape(t_.shape))
        vs_op["q_idx_m_vs_lin"] = cmp(pm["q_idx"], ref)
        vs_op["q_idx_u_vs_lin"] = cmp(pu_qidx, ref)
        row["q_idx_m_vs_lin_where"] = where(pm["q_idx"], ref)
        row["q_idx_u_vs_lin_where"] = where(pu_qidx, ref)
    ttnn.release_trace(mesh, tid)
    ok_bit = all(all(r_["bitwise"].values()) and all(r_["finite"].values()) for r_ in det)
    ab_bit = all(all(r_["ab_bitwise"].values()) for r_ in det) if AB else None
    pmin = lambda k_: min(ch["pcc"] for ch in vs_op[k_] if ch["pcc"] is not None)  # noqa: E731
    row.update(det=det, bitwise_all=ok_bit, vs_op=vs_op, pcc_qk_linear=pmin("qk_linear_only"), pcc_kg_linear=pmin("kg_linear_only"),
               pcc_qk_op=pmin("qk_vs_op_chain"), pcc_kg_op=pmin("kg_vs_op_chain"), pcc_xhat_op=pmin("xhat_vs_op"),
               **({f"pcc_{k_}_{v_}": pmin(f"{k_}_{v_}") for k_ in ("q_r", "latent", "kg_rm") for v_ in ("d2_only", "vs_op_linears", "vs_op_chain")}
                  if FULL else {}),
               **({k_: pmin(k_[4:]) for k_ in ("pcc_q_idx_vs_op_linears", "pcc_q_idx_vs_op_chain", "pcc_q_idx_m_vs_lin",
                                               "pcc_q_idx_u_vs_lin")} if QIDX else {}),
               us_merged=perf_m["median_us"], us_unmerged=perf_u["median_us"], perf_m=perf_m, perf_u=perf_u,
               ab_bitwise_all=ab_bit, us_copies=perf_c["median_us"] if AB else None, perf_c=perf_c,
               status="pass" if (ok_bit and ab_bit is not False) else "fail")  # fmt: skip
    S.log(f"{tag}: vs op {dict((k_, [ (ch['pcc'], ch['maxabs']) for ch in v]) for k_, v in vs_op.items())}")
    S.log(
        f"{tag}: merged {perf_m['median_us']:.0f} us vs unmerged {perf_u['median_us']:.0f} us"
        + (f" vs copies {perf_c['median_us']:.0f} us; in place vs copies bitwise {ab_bit}" if AB else "")
        + f"; bitwise {ok_bit}; status {row['status']}"
    )
    if bool(args.get("record", True)):
        C.record("pcc", dict(test=row["test"], milestone="PMK-AD1", layer=L, tp=NDEV, bucket=c, bitwise=ok_bit,
                             pcc_qk_linear=row["pcc_qk_linear"], pcc_kg_linear=row["pcc_kg_linear"], pcc_qk_op=row["pcc_qk_op"],
                             pcc_kg_op=row["pcc_kg_op"], pcc_xhat_op=row["pcc_xhat_op"], mode=MODE, inplace=INPLACE,
                             ab_bitwise=ab_bit,
                             **{k_: v_ for k_, v_ in row.items() if k_.startswith(("pcc_q_r", "pcc_latent", "pcc_kg_rm", "pcc_q_idx"))},
                             status=row["status"], tag=tag))  # fmt: skip
        C.record("perf", dict(test=f"PMK AD-1 ({MODE}) L{L} tp{NDEV} traced replay: merged vs {' | '.join('(' + ', '.join(v) + ')' for v in SC_U)}", milestone="PMK-AD1",
                              layer=L, tp=NDEV, bucket=c, us=perf_m["median_us"], us_unmerged=perf_u["median_us"], inplace=INPLACE,
                              us_copies=perf_c["median_us"] if AB else None,
                              replays=perf_m["replays"], status=row["status"], tag=tag))  # fmt: skip
RESULT = row
print("PMK_AD1_UNIT", {k_: v_ for k_, v_ in row.items() if k_ not in ("perf_m", "perf_u", "perf_c", "det")}, flush=True)
