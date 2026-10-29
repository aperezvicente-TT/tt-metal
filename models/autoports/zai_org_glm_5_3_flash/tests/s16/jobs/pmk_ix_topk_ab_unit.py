# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair; program 2026-10-23 23:29:42Z, the indexer visible-pool bound STAGE 1 gate):
GLM53_PF_IX_TOPK_BOUND (topk_large_indices straight on the score op's ROW_MAJOR plane with valid_length_tensor = the
chip's vis_cols word; no to_layout / lt / typecast / where / to_layout) vs today's select, both arms with the served
GLM53_PF_IX_BOUND=1 (the same score plane). Layer L (3, the first sparse_dsa layer) in the served 1M DSA env after a real
2,048-token prefix, at every key (chunk_start in `chunk_starts`, window t in `windows`), arm A = ix.topk_bound False
(today), arm B = True (per DsaIndexer instance):
  * the selection select leaves in the window's SparseCtx after the TRACED replays of select alone on the recorded
    inputs (the layer's own eager pass, h cloned): sp.counts[..., 0] bitwise per chip; sp.ids bitwise within each row's
    valid prefix (counts[r][0], >= 1: one dummy key) and as a sorted SET per row (the top-k order among ties may legally
    differ); the buffers past a row's count are stale (one SparseCtx per window across the arms), reported only;
  * select alone traced (`replays` blocking replays per arm, median us) and the layer (embed + L - 1, L, no head);
  * the per-chip word as the device holds it (ci.vis_cols_u read back) vs ceil((chunk_start + (rank + 1) C_l) / 128) 32;
  * arm B took the bounded call (a topk_large_indices wrapper counts calls with valid_length_tensor).
Every program compiles before the first capture; one live trace at a time; host writes only outside captures.
ARGS: layer (3), chunk_starts ([0, 2048]), windows ([65536, 1048576]), replays (10), layer_timing (True), env, tag,
record (True)."""
import os
import statistics
import sys
import time
import traceback

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk45_alias as A45  # noqa: E402
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
L = int(args.get("layer", 3))
BUILD = [int(v) for v in args.get("build_layers", [L - 1, L])]
N = 2048
P0S = [int(v) for v in args.get("chunk_starts", [0, 2048])]
TS = sorted(int(v) for v in args.get("windows", [65536, 1048576]))
REPLAYS = int(args.get("replays", 10))
LAYER_T = bool(args.get("layer_timing", True))
POOL_TAP = bool(args.get("pool_tap", True))  # the in-trace probes (DsaIndexer.pool_taps / taps): the tie-vs-mask diagnosis
TAP_T = int(args.get("tap_t", 65536))
# gather_ab (False): arm B = the bounded top-k AND the bounded pooled-key gather (GLM53_PF_IX_GATHER_BOUND, stage 2:
# SparseCtx.gather_bound per window), arm A = today's; pool_buf compared bitwise on each chip's visible pool rows
GATHER_AB = bool(args.get("gather_ab", False))  # the arm-A score plane tap only at windows <= TAP_T (the 1M plane = 512 MiB per chip)
tag = args.get("tag", f"pmk-ix-topk-ab-L{L}-tp{TPN}")
rec_on = bool(args.get("record", True))
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
os.environ["GLM53_PF_SPARSE_T"] = ",".join(str(t) for t in TS)
os.environ["GLM53_PF_MODE"] = "op"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
t_job = time.time()
row, err = dict(layer=L, build_layers=BUILD, chunk_starts=P0S, windows=TS, tp=TPN,
                ix_bound_env=os.environ.get("GLM53_PF_IX_BOUND", ""), ix_cfg=os.environ.get("GLM53_PF_IX_CFG", "")), None  # fmt: skip
rec = {"arm": None}
ARMS = (False, True)  # ix.topk_bound: A = today's select, B = the bounded top-k
SENT = 0xFFFFFFFF

_tk = ttnn.experimental.topk_large_indices
TK_CALLS = {"bounded": 0, "plain": 0}


def _topk(*a, **k):
    TK_CALLS["bounded" if k.get("valid_length_tensor") is not None else "plain"] += 1
    return _tk(*a, **k)


def make_hook(ix):
    """an INSTANCE-level wrapper of ix.select: records the eager pass's select inputs at the armed key (h cloned)"""
    bound = ix.select

    def hook(attn, h_local, ci, sp):
        rec["calls"] = rec.get("calls", 0) + 1
        if rec["arm"] is not None and sp is not None:
            rec[rec["arm"]] = dict(ix=ix, attn=attn, h=ttnn.clone(h_local), ci=ci, sp=sp, p=int(getattr(sp, "p", -1)), fn=bound)
            rec["arm"] = None
        return bound(attn, h_local, ci, sp)

    return hook


def select_alone(r):
    """begin(ci) + select + end, as forward() brackets it"""
    r["sp"].begin(r["ci"])
    try:
        r["fn"](r["attn"], r["h"], r["ci"], r["sp"])
    finally:
        r["sp"].end()


def capture_replay(fn, label):
    """one capture of fn, `REPLAYS` timed blocking replays, release -> [us]"""
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
    ts = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append(round((time.perf_counter() - t0) * 1e6, 1))
    ttnn.release_trace(mesh, tid)
    H.log(f"{tag}: {label}: {ts}")
    return ts


def chips(t_):
    return [ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]


def sel_out(r):
    """the selection select left in the SparseCtx: ids / counts per chip (host copies)"""
    return {nm: chips(getattr(r["sp"], nm)) for nm in ("ids", "counts") if getattr(r["sp"], nm, None) is not None}


def tap_out(r, m):
    """the probes of the key's last select replay: the arm's local top-512 pool ids, arm A's masked score plane"""
    ix, cl, p = r["ix"], int(r["h"].shape[-2]), int(r["p"])
    o = {}
    pt = getattr(ix, "pool_taps", None)
    if pt and (cl, p, bool(m)) in pt:
        o["pools"] = chips(pt[(cl, p, bool(m))])
    st = getattr(ix, "taps", None)
    if st and not m and (cl, p) in st:
        o["sm"] = chips(st[(cl, p)])
    if st and m and ("b", cl, p) in st:  # arm B's top-k input plane
        o["s_b"] = chips(st[("b", cl, p)])
    return o


def plane_cmp(sm_chips, sb_chips, p0, words):
    """arm A's masked plane vs arm B's top-k input on each row's VISIBLE pools [0, n_closed) (bitwise + max |diff|), and
    B on [n_closed, word) (must be -inf for the bound to be right)"""
    out = []
    for k_chip, (a_, b_) in enumerate(zip(sm_chips, sb_chips)):
        a2 = a_.reshape(a_.shape[-2], a_.shape[-1])
        b2 = b_.reshape(b_.shape[-2], b_.shape[-1])
        cl, w = int(a2.shape[0]), int(words[k_chip])
        eq_rows, band_ok, mad, ex = 0, 0, 0.0, []
        for r in range(cl):
            nc = (p0 + k_chip * cl + r + 1) // 4
            va, vb = a2[r, :nc], b2[r, :nc]
            e_ = bool(torch.equal(va.view(torch.int16), vb.view(torch.int16)))
            eq_rows += int(e_)
            band = b2[r, nc:w].float()
            band_ok += int(bool(torch.all(torch.isneginf(band))) if band.numel() else True)
            if not e_:
                d_ = (va.float() - vb.float()).abs()
                fin = torch.isfinite(d_)
                m_ = float(d_[fin].max()) if bool(fin.any()) else float("nan")
                mad = max(mad, m_) if m_ == m_ else mad
                if len(ex) < 3:
                    j = int(torch.nonzero(va.view(torch.int16) != vb.view(torch.int16))[0])
                    ex.append(dict(row=r, n_closed=nc, first_col=j, a=float(va[j]), b=float(vb[j]), n_diff=int((va.view(torch.int16) != vb.view(torch.int16)).sum()),
                                   b_row_match=[r2 for r2 in range(max(0, r - 40), min(cl, r + 40)) if r2 != r and bool(torch.equal(a2[r2, : min(nc, 64)].view(torch.int16), vb[: min(nc, 64)].view(torch.int16)))][:3]))  # fmt: skip
        out.append(dict(rows=cl, visible_bitwise_rows=eq_rows, band_neginf_rows=band_ok, visible_maxabs=mad, examples=ex))
    return out


def compare(a, b):
    """arm A vs arm B selections, per chip: counts word 0 (the keys emitted: raw uint32 bits in the FP32 tensor, written
    by expand_rows) + word 1 (the valid flag, fp32 bits) bitwise; ids per row ordered and as a SET of the non-sentinel
    entries (expand_rows rewrites the 0xFFFFFFFF tail of every row: the whole buffer is comparable)"""
    out = []
    for k_chip in range(len(a["ids"])):
        ia = a["ids"][k_chip].to(torch.int64).reshape(-1, a["ids"][k_chip].shape[-1]) & SENT
        ib = b["ids"][k_chip].to(torch.int64).reshape(-1, b["ids"][k_chip].shape[-1]) & SENT
        d_ = dict(rows=int(ia.shape[0]), ids_buffer_bitwise=bool(torch.equal(ia, ib)))
        if "counts" in a:
            ca = a["counts"][k_chip].contiguous().view(torch.int32).reshape(-1, a["counts"][k_chip].shape[-1])
            cb = b["counts"][k_chip].contiguous().view(torch.int32).reshape(-1, b["counts"][k_chip].shape[-1])
            d_.update(counts_bitwise=bool(torch.equal(ca[:, :2], cb[:, :2])), counts_rows_differ=int((ca[:, 0] != cb[:, 0]).sum()),
                      count_sum_a=int(ca[:, 0].sum()), count_sum_b=int(cb[:, 0].sum()), count_max_a=int(ca[:, 0].max()))  # fmt: skip
        ord_eq = set_eq = 0
        ex = []
        for r in range(int(ia.shape[0])):
            pa, pb = ia[r][ia[r] != SENT], ib[r][ib[r] != SENT]
            e_ = bool(torch.equal(pa, pb))
            s_ = e_ or (pa.numel() == pb.numel() and bool(torch.equal(torch.sort(pa).values, torch.sort(pb).values)))
            ord_eq += int(e_)
            set_eq += int(s_)
            if not s_ and len(ex) < 3:
                ex.append(dict(row=r, n_a=int(pa.numel()), n_b=int(pb.numel()), only_a=sorted(set(pa.tolist()) - set(pb.tolist()))[:8],
                               only_b=sorted(set(pb.tolist()) - set(pa.tolist()))[:8]))  # fmt: skip
        d_.update(ids_order_equal_rows=ord_eq, ids_set_equal_rows=set_eq, set_diff_examples=ex)
        out.append(d_)
    return out


def pool_diff(pa_chips, pb_chips, p0, planes=None):
    """the two arms' LOCAL top-512 pool ids (the in-trace tap), per chip, per row p = p0 + rank C_l + r: the first
    nsel = min(n_closed, 512) entries (what expand_rows reads) as sets; a differing row is classed
      open   an only-B (or only-A) pool >= n_closed (the open / future pool, or 0xFFFFFFFF): a MASK difference;
      tie    (planes: the arm-A masked score plane of the same chunk_start) every only-A / only-B pool scores exactly the
             row's rank-nsel value: a threshold tie (a legal set swap);
      value  otherwise (scores differ beyond a tie)"""
    out = []
    for k_chip, (xa, xb) in enumerate(zip(pa_chips, pb_chips)):
        xa = xa.to(torch.int64).reshape(-1, xa.shape[-1]) & SENT
        xb = xb.to(torch.int64).reshape(-1, xb.shape[-1]) & SENT
        cl = int(xa.shape[0])
        pl = planes[k_chip].float().reshape(cl, -1) if planes is not None else None
        cls = dict(open=0, tie=0, value=0, unclassed=0)
        ord_eq = set_eq = 0
        ex = []
        for r in range(cl):
            p = p0 + k_chip * cl + r
            nc = (p + 1) // 4
            ns = min(nc, 512)
            sa, sb = xa[r, :ns], xb[r, :ns]
            if torch.equal(sa, sb):
                ord_eq += 1
                set_eq += 1
                continue
            oa, ob = set(sa.tolist()) - set(sb.tolist()), set(sb.tolist()) - set(sa.tolist())
            if not oa and not ob:
                set_eq += 1
                continue
            e_ = dict(row=r, pos=p, n_closed=nc, only_a=sorted(oa)[:6], only_b=sorted(ob)[:6], n_only=len(oa))
            if any(v >= nc for v in oa | ob):
                c_ = "open"
            elif pl is not None:
                vis = pl[r, :nc]
                thr = float(torch.sort(vis, descending=True).values[ns - 1])
                va = [float(pl[r, v]) for v in sorted(oa)]
                vb = [float(pl[r, v]) for v in sorted(ob)]
                e_.update(thr=thr, v_a=va[:6], v_b=vb[:6])
                c_ = "tie" if all(v == thr for v in va + vb) else "value"
            else:
                c_ = "unclassed"
            cls[c_] += 1
            if len(ex) < 4:
                ex.append(dict(e_, cls=c_))
        out.append(dict(rows=cl, order_equal_rows=ord_eq, set_equal_rows=set_eq, classes=cls, examples=ex))
    return out


try:
    max_seq = max(TS) + 512
    ent = S.build(mesh, ctx, BUILD, b=2, buckets=(N,), max_seq=max_seq)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    row["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=2)
    assert pf.has_dsa(), "GLM53_PF_DSA=1 needed (the sparse path)"
    c = pf.bucket_for(N)
    ixs, hooked = [], []
    for pl_ in pf.layers:
        ix_ = getattr(getattr(pl_, "mla", None), "idx", None)
        if ix_ is not None:
            ix_.select = make_hook(ix_)
            ixs.append(ix_)
            hooked.append(int(pl_.L))
    assert hooked, "no DSA indexer instance to hook"
    assert all(x_.bound for x_ in ixs), "the gate runs with the served GLM53_PF_IX_BOUND=1 in both arms"
    ci0 = pf.chunk_inputs(c)
    assert getattr(ci0, "vis_cols_u", None) is not None, "this tree has no ChunkInputs.vis_cols_u (the stage-1 edit)"
    row["sparse_ctx"] = {t: pf.sparse_ctx(c, t) is not None for t in TS}
    g32 = S.golden_prompt(32768).reshape(-1).long()
    k0 = pf.plan(0, N)
    pf.write_inputs(c, g32[:N], slot=0, chunk_start=0, n_valid=N)  # the real prefix [0, 2,048), its own key
    pf.forward(c, head=False, t=k0[1])
    ttnn.synchronize_device(mesh)

    PTAPS = {id(x_): {} for x_ in ixs} if POOL_TAP else {}
    STAPS = {id(x_): {} for x_ in ixs} if POOL_TAP else {}

    def set_arm(m, t=None):
        if GATHER_AB:
            for t_ in TS:
                sc_ = pf.sparse_ctx(c, t_)
                if sc_ is not None:
                    sc_.gather_bound = bool(m)
        for x_ in ixs:
            x_.bound = True
            x_.topk_bound = bool(m)  # read per select call (prefill_tp DsaIndexer.select)
            if POOL_TAP:  # the in-trace probes: pool ids (both arms, every key), arm A's masked plane (t <= TAP_T only)
                x_.pool_taps = PTAPS[id(x_)]
                x_.taps = STAPS[id(x_)] if (t is not None and t <= TAP_T) else None

    def write(p0):
        pf.write_inputs(c, g32[p0:p0 + N], slot=0, chunk_start=p0, n_valid=N)

    ttnn.experimental.topk_large_indices = _topk
    try:
        keys = [(p0, t, m) for p0 in P0S for t in TS for m in ARMS]
        words = {}
        for p0, t, m in keys:  # eager per key: compiles the layer per arm and records select's inputs
            set_arm(m, t)
            write(p0)
            if p0 not in words:  # the per-chip word the device holds
                words[p0] = dict(device=[int(v.reshape(-1)[0]) & SENT for v in chips(pf.chunk_inputs(c).vis_cols_u)],
                                 expect=[-(-(p0 + (r + 1) * (N // TPN)) // 128) * 32 for r in range(TPN)])  # fmt: skip
            rec["arm"] = (p0, t, m)
            pf.forward(c, head=False, t=t)
            ttnn.synchronize_device(mesh)
            assert (p0, t, m) in rec, f"select was not recorded at {(p0, t, m)} (calls {rec.get('calls', 0)}, hooked {hooked})"
        tk_fwd = dict(TK_CALLS)
        # ONE recorded input set per (cs, t) for BOTH arms: L2 is a KDA layer whose slot state advances with every
        # forward at cs > 0, so each key's own eager pass hands select a different h (and rewrites the chunk's pooled
        # keys); the arms must rank the same plane (ixtk3 23:58Z: the arms' input planes differed at cs 2,048)
        for p0, t, m in keys:  # select alone, eager (its compile pass)
            set_arm(m, t)
            write(p0)
            b0 = TK_CALLS["bounded"]
            select_alone(rec[(p0, t, False)])
            assert (TK_CALLS["bounded"] > b0) == bool(m), f"arm {m} at {(p0, t)}: bounded top-k calls {TK_CALLS['bounded'] - b0}"
        ttnn.synchronize_device(mesh)
        row.update(words=words, words_ok=all(w_["device"] == w_["expect"] for w_ in words.values()),
                   pools_per_window={t: rec[(P0S[0], t, False)]["p"] for t in TS}, topk_calls_forward=tk_fwd)  # fmt: skip
        H.log(f"{tag}: compiled; keys {keys}; words {words}; pools {row['pools_per_window']}; topk calls {TK_CALLS}")
        if C.MOCK:  # the readback + compare path on the mock's planes too (mocks must run the host readback)
            p0, t = P0S[0], TS[0]
            set_arm(False, t)
            write(p0)
            select_alone(rec[(p0, t, False)])
            a_, ta_ = sel_out(rec[(p0, t, False)]), tap_out(rec[(p0, t, False)], False)
            set_arm(True, t)
            select_alone(rec[(p0, t, False)])
            b_, tb_ = sel_out(rec[(p0, t, False)]), tap_out(rec[(p0, t, False)], True)
            row["mock_compare"] = compare(a_, b_)
            if POOL_TAP:
                row["mock_pool_diff"] = pool_diff(ta_["pools"], tb_["pools"], p0, ta_.get("sm"))
                if tb_.get("s_b") is not None and ta_.get("sm") is not None:
                    row["mock_plane_cmp"] = plane_cmp(ta_["sm"], tb_["s_b"], p0, row["words"][p0]["expect"])
                row["s_meta"] = getattr(ixs[0], "s_meta", None)
                row["mock_tap_shapes"] = {nm: [list(v.shape) for v in ta_[nm]] for nm in ta_}
            row["mock_selection_shapes"] = {nm: [list(v.shape) for v in vs] for nm, vs in a_.items()}
            H.log(f"{tag}: mock compare {row['mock_compare']} shapes {row['mock_selection_shapes']}")
        else:
            n_pc = int(mesh.num_program_cache_entries())
            layer_us, select_us, sels = {}, {}, {}
            taps = {}
            for p0 in P0S:
                for t in TS:
                    if LAYER_T:  # the layer per arm first (its replays advance the KDA state + rewrite the chunk's pools)
                        for m in ARMS:
                            set_arm(m, t)
                            write(p0)
                            layer_us[(p0, t, m)] = capture_replay(lambda: pf.forward(c, head=False, t=t),
                                                                  f"layer L{L} cs={p0} t={t} topk_bound={int(m)}")  # fmt: skip
                    r = rec[(p0, t, False)]  # then select alone per arm on the SAME recorded inputs and store contents
                    for m in ARMS:
                        set_arm(m, t)
                        write(p0)
                        select_us[(p0, t, m)] = capture_replay(lambda: select_alone(r), f"select cs={p0} t={t} topk_bound={int(m)}")
                        ttnn.synchronize_device(mesh)
                        sels[(p0, t, m)] = sel_out(r)  # after the last select replay of this arm
                        if GATHER_AB:
                            sels[(p0, t, m)]["pool_buf"] = chips(r["sp"].pool_buf)
                        if POOL_TAP:
                            taps[(p0, t, m)] = tap_out(r, m)
            assert int(mesh.num_program_cache_entries()) == n_pc, "a program compiled inside a capture"
            med = lambda v: round(statistics.median(v), 1)  # noqa: E731
            kname = lambda k_: f"cs{k_[0]}_t{k_[1]}_topk{int(k_[2])}"  # noqa: E731
            row.update(select_us={kname(k_): med(v) for k_, v in select_us.items()},
                       layer_us={kname(k_): med(v) for k_, v in layer_us.items()},
                       select_us_all={kname(k_): v for k_, v in select_us.items()},
                       layer_us_all={kname(k_): v for k_, v in layer_us.items()})  # fmt: skip
            cmp_, saved, pdiff = {}, {}, {}
            for p0 in P0S:
                for t in TS:
                    kk = f"cs{p0}_t{t}"
                    cmp_[kk] = compare(sels[(p0, t, False)], sels[(p0, t, True)])
                    if POOL_TAP:  # the scores: arm A's plane of the same chunk_start at the tapped window (visible
                        # columns are window-independent under IX_BOUND; flagged by plane_window)
                        tw = max([w for w in TS if w <= TAP_T], default=None)
                        pl_ = taps.get((p0, tw, False), {}).get("sm") if tw is not None else None
                        pdiff.setdefault(kk, dict(plane_window=tw if pl_ is not None else None,
                                                  chips=pool_diff(taps[(p0, t, False)]["pools"], taps[(p0, t, True)]["pools"], p0, pl_)))  # fmt: skip
                        sb_ = taps.get((p0, t, True), {}).get("s_b")
                        sa_ = taps.get((p0, t, False), {}).get("sm")
                        if sb_ is not None and sa_ is not None:
                            pdiff[kk]["input_planes"] = plane_cmp(sa_, sb_, p0, row["words"][p0]["expect"])
                    saved[kk] = dict(select_a=med(select_us[(p0, t, False)]), select_b=med(select_us[(p0, t, True)]),
                                     select_saved=round(med(select_us[(p0, t, False)]) - med(select_us[(p0, t, True)]), 1))  # fmt: skip
                    if LAYER_T:
                        saved[kk].update(layer_a=med(layer_us[(p0, t, False)]), layer_b=med(layer_us[(p0, t, True)]),
                                         layer_saved=round(med(layer_us[(p0, t, False)]) - med(layer_us[(p0, t, True)]), 1))  # fmt: skip
            if GATHER_AB:  # pool_buf on every chip's visible pool rows [0, word)
                pbuf = {}
                for p0 in P0S:
                    for t in TS:
                        a_, b_ = sels[(p0, t, False)]["pool_buf"], sels[(p0, t, True)]["pool_buf"]
                        ws = row["words"][p0]["expect"]
                        pbuf[f"cs{p0}_t{t}"] = [bool(torch.equal(x_.reshape(-1, x_.shape[-1])[: ws[k]].view(torch.int16),
                                                                 y_.reshape(-1, y_.shape[-1])[: ws[k]].view(torch.int16)))
                                                for k, (x_, y_) in enumerate(zip(a_, b_))]  # fmt: skip
                row["pool_buf_visible_bitwise"] = pbuf
                H.log(f"{tag}: POOL_BUF visible bitwise {pbuf}")
            gate = all(d_.get("counts_bitwise") is not False and d_["ids_order_equal_rows"] == d_["rows"]
                       for v in cmp_.values() for d_ in v)  # fmt: skip
            gate_set = all(d_.get("counts_bitwise") is not False and d_["ids_set_equal_rows"] == d_["rows"]
                           for v in cmp_.values() for d_ in v)  # fmt: skip
            row.update(compare=cmp_, saved_us=saved, gate_ids_counts_bitwise=gate, gate_ids_set_counts_bitwise=gate_set,
                       pool_diff=pdiff)  # fmt: skip
            H.log(f"{tag}: GATE bitwise {gate} set {gate_set} words_ok {row['words_ok']}; saved {saved}; compare {cmp_}")
            row["s_meta"] = getattr(ixs[0], "s_meta", None)
            H.log(f"{tag}: S META {row['s_meta']}")
            H.log(f"{tag}: POOL DIFF {pdiff}")
    finally:
        ttnn.experimental.topk_large_indices = _tk
    pb_ok = (not GATHER_AB) or all(all(v) for v in (row.get("pool_buf_visible_bitwise") or {"x": [False]}).values())
    row["status"] = "mock" if C.MOCK else ("pass" if (row.get("gate_ids_set_counts_bitwise") and row.get("words_ok") and pb_ok) else "fail")
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-5000:]
    H.log(f"FAILED:\n{err}")
    row.update(status="fail", error=err)
res = dict(test=f"indexer stage 1 (GLM53_PF_IX_TOPK_BOUND): the bounded top-k vs today's select, L{L}, keys cs {P0S} x t {TS}, tp{TPN}",
           milestone="PMK-ix-bound-stage1", mock=C.MOCK, wall_s=round(time.time() - t_job, 1), tag=tag, **row)  # fmt: skip
H.log({k: v for k, v in res.items() if k not in ("test", "layer_us_all", "select_us_all", "error")})
if rec_on and not C.MOCK:
    C.record("tests", {k: v for k, v in res.items() if k not in ("layer_us_all", "select_us_all")})
RESULT = res
