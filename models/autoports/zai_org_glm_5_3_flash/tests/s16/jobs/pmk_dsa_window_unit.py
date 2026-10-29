# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair; program 2026-10-23 19:18:59Z, the context-bound attention FIRST UNIT): which part
of the op sparse_dsa layer pays for the trace key's WINDOW t instead of the valid context. 9's 1M curve: per 2,048 rows
the device cost follows the key window (736 ms at t = 65,536, 1,168 at 1,048,576: 38 ms per DSA layer per 2,048 for the
same tokens). Here: layer L (3, the first sparse_dsa layer) alone, the served 1M DSA env, a real 2,048-token prefix
(op chunk [0, 2,048)), then the chunk at position `chunk_start` (2,048: 2,048 valid context tokens) at every window t
in `windows` (default 65,536 and 1,048,576):
  * layer: the traced forward of the chunk at key (C, t) (embed + layers L - 1, L, no head), `replays` replays;
  * select: SparseCtx.begin + DsaIndexer.select + end alone (q_idx projections, the paged pooled keys of the window's P = t / 4 pools,
    indexer_score_dsa over P, the visibility mask, the top-512 over P, the gather / expand), traced on the inputs the
    layer's own eager pass handed it (cloned there), `replays` replays.
Every program compiled before the first capture (the layer at every t, select alone at every t); one live trace at a
time; no host writes inside a capture. The difference layer(t1) - layer(t0) vs select(t1) - select(t0) splits the
window cost between the indexer and the rest (the op sparse attention + the masks).
bound_ab (False): every window with the indexers' GLM53_PF_IX_BOUND off and on (ix.bound per instance), the selection
(the SparseCtx ids / counts after select) compared per chip, bitwise (program 19:29:26Z).
split (False): select's pieces (proj / gather_pools / score / mask / topk) each traced alone at every window; ref
(False, with split): the fp64 host reference score (sum_h w relu(q_h . k)) of every ref_stride-th row over its visible
pools vs that mode's masked plane: top-512 set overlap + max |diff| per chip.
ARGS: layer (3), chunk_start (2048), windows ([65536, 1048576]), replays (10), bound_ab (False), split (False), env, tag,
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
# the backend needs a non-DSA layer beside it (stack_model: self.fl = the first non-sparse_dsa layer): L - 1 (dense_kda
# L2 for L3) is built too; its cost is the same at every window t, so the deltas are L's alone
BUILD = [int(v) for v in args.get("build_layers", [L - 1, L])]
N = 2048
P0 = int(args.get("chunk_start", 2048))
TS = sorted(int(v) for v in args.get("windows", [65536, 1048576]))
BOUND_AB = bool(args.get("bound_ab", False))
SPLIT = bool(args.get("split", False))  # time select's pieces separately at every window (program 19:35:49Z)
REF = bool(args.get("ref", False))  # with split: the fp64 host reference score of sampled rows vs each mode's masked plane
REF_STRIDE = int(args.get("ref_stride", 16))  # GLM53_PF_IX_BOUND off vs on per window, selections compared bitwise
REPLAYS = int(args.get("replays", 10))
tag = args.get("tag", f"pmk-dsa-window-L{L}-tp{TPN}")
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
row, err = dict(layer=L, build_layers=BUILD, chunk_start=P0, windows=TS, tp=TPN), None
rec = {"arm": None}


def make_hook(ix):
    """an INSTANCE-level wrapper of ix.select (a class-level patch did not intercept on d0, 19:24Z): records the eager
    pass's select inputs at the armed window (h_local cloned: the layer frees its own)"""
    bound = ix.select

    def hook(attn, h_local, ci, sp):
        rec["calls"] = rec.get("calls", 0) + 1
        if rec["arm"] is not None and sp is not None:
            rec[rec["arm"]] = dict(ix=ix, attn=attn, h=ttnn.clone(h_local), ci=ci, sp=sp, p=int(getattr(sp, "p", -1)), fn=bound)
            rec["arm"] = None
        return bound(attn, h_local, ci, sp)

    return hook


def select_alone(r):
    """select with the window's SparseCtx bracketed as forward() does (begin(ci) sets the per-chunk visibility state
    nvis / masks that select reads; end() clears it): begin + select + end = the indexer's window-dependent work"""
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
    row["layers_built"] = [dict(L=int(pl_.L), kind=str(pl_.kind), mla=type(getattr(pl_, "mla", None)).__name__,
                                idx=getattr(getattr(pl_, "mla", None), "idx", None) is not None) for pl_ in pf.layers]  # fmt: skip
    row["sparse_ctx"] = {t: pf.sparse_ctx(c, t) is not None for t in TS}
    hooked = []
    for pl_ in pf.layers:
        ix_ = getattr(getattr(pl_, "mla", None), "idx", None)
        if ix_ is not None:
            ix_.select = make_hook(ix_)
            hooked.append(int(pl_.L))
    assert hooked, "no DSA indexer instance to hook"
    H.log(f"{tag}: layers {row['layers_built']}; sparse ctx {row['sparse_ctx']}; plan(P0) {pf.plan(P0, N)}")
    g32 = S.golden_prompt(32768).reshape(-1).long()
    # the real 2,048-token prefix through the op chain (its own key), eager: the pools / latent rows of [0, 2,048)
    k0 = pf.plan(0, N)
    pf.write_inputs(c, g32[:N], slot=0, chunk_start=0, n_valid=N)
    pf.forward(c, head=False, t=k0[1])
    ttnn.synchronize_device(mesh)
    MODES = [False, True] if BOUND_AB else [None]  # None = the indexers' bound as built (GLM53_PF_IX_BOUND)
    ixs = [getattr(getattr(pl_, "mla", None), "idx", None) for pl_ in pf.layers]
    ixs = [x_ for x_ in ixs if x_ is not None]

    def set_bound(m):
        if m is not None:
            for x_ in ixs:
                x_.bound = bool(m)  # read per select call (prefill_tp DsaIndexer.select)

    keys = [(t, m) for t in TS for m in MODES]
    for t, m in keys:  # the chunk at P0, eager at every (window, bound): compiles the layer and records select's inputs
        set_bound(m)
        pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)
        rec["arm"] = (t, m)
        pf.forward(c, head=False, t=t)
        ttnn.synchronize_device(mesh)
        assert (t, m) in rec, f"select was not recorded at {(t, m)} (calls so far {rec.get('calls', 0)}, hooked {hooked})"
    for t, m in keys:  # select alone, eager
        set_bound(m)
        select_alone(rec[(t, m)])
    ttnn.synchronize_device(mesh)
    row["pools_per_window"] = {t: rec[(t, MODES[0])]["p"] for t in TS}
    row["bound_modes"] = MODES
    H.log(f"{tag}: compiled; key windows {TS}, bound modes {MODES}, pools per window {row['pools_per_window']}, prefix key {k0}")

    def sel_out(r):
        """the selection select leaves in the window's SparseCtx (sdpa / gather path: the expanded ids + the per-row
        counts of the BQ shard), per chip, host copies"""
        out = {}
        taps = getattr(r["ix"], "taps", None)  # GLM53_PF_IX_TAP=1: the masked score plane where(vis, s, NEG) top-k reads
        if taps:
            t_ = taps.get((int(r["h"].shape[-2]), int(r["p"])))
            if t_ is not None:
                out["sm"] = [ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]
        for nm in ("ids", "counts"):
            t_ = getattr(r["sp"], nm, None)
            if t_ is not None:
                out[nm] = [ttnn.to_torch(d_).clone() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]
        return out

    if C.MOCK:  # the readback path on the mock's zero planes too (the 10-23 lesson: mocks must run the host readback)
        mo = sel_out(rec[keys[0]])
        row["mock_selection_shapes"] = {nm: [list(v.shape) for v in vs] for nm, vs in mo.items()}
        H.log(f"{tag}: mock selection readback {row['mock_selection_shapes']}")
    if not C.MOCK:
        n_pc = int(mesh.num_program_cache_entries())
        layer_us, select_us, sels = {}, {}, {}
        for t, m in keys:
            set_bound(m)
            lab = f"t={t}" + ("" if m is None else f" bound={int(m)}")
            pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)  # host writes outside the capture
            layer_us[(t, m)] = capture_replay(lambda: pf.forward(c, head=False, t=t), f"layer L{L} {lab}")
            r = rec[(t, m)]
            select_us[(t, m)] = capture_replay(lambda: select_alone(r), f"select {lab}")
            ttnn.synchronize_device(mesh)
            sels[(t, m)] = sel_out(r)  # after the last select replay of this key
        assert int(mesh.num_program_cache_entries()) == n_pc, "a program compiled inside a capture"
        med = lambda v: round(statistics.median(v), 1)  # noqa: E731
        kname = lambda k_: f"t{k_[0]}" + ("" if k_[1] is None else f"_bound{int(k_[1])}")  # noqa: E731
        row.update(layer_us={kname(k_): med(v) for k_, v in layer_us.items()}, select_us={kname(k_): med(v) for k_, v in select_us.items()},
                   layer_us_all={kname(k_): v for k_, v in layer_us.items()}, select_us_all={kname(k_): v for k_, v in select_us.items()})  # fmt: skip
        m0 = MODES[0]
        dl = med(layer_us[(TS[-1], m0)]) - med(layer_us[(TS[0], m0)])
        ds = med(select_us[(TS[-1], m0)]) - med(select_us[(TS[0], m0)])
        row.update(window_cost_us=dict(layer=round(dl, 1), select=round(ds, 1), rest=round(dl - ds, 1),
                                       select_share=round(ds / dl, 3) if dl else None))  # fmt: skip
        if BOUND_AB:  # the selection with the bound vs without, per window, per chip, bitwise (the served-env gate)
            eq = {}
            for t in TS:
                a_, b_ = sels[(t, False)], sels[(t, True)]
                eq[f"t{t}"] = {nm: [bool(torch.equal(x_, y_)) for x_, y_ in zip(a_[nm], b_[nm])] for nm in a_}
            cross = {}  # the bounded selection across windows (the same visible pools: expected equal too)
            for nm in sels[(TS[0], True)]:
                cross[nm] = [bool(torch.equal(x_, y_)) for x_, y_ in zip(sels[(TS[0], True)][nm], sels[(TS[-1], True)][nm])]
            # the verdict: the masked score plane (sm) per window -- the raw ids / counts buffers keep stale slots past
            # each row's count (shared SparseCtx across the modes), so they are reported, not judged
            sm_eq = {t_k: d_.get("sm") for t_k, d_ in eq.items()}
            sel_diff = {}  # per window: the plane's max |diff| and the rows whose top-512 pool SET differs (host top-k)
            for t in TS:
                pa, pb = sels[(t, False)].get("sm"), sels[(t, True)].get("sm")
                if pa is None:
                    continue
                d_t = []
                for xa, xb in zip(pa, pb):
                    xa2, xb2 = xa.float().reshape(-1, xa.shape[-1]), xb.float().reshape(-1, xb.shape[-1])
                    fin = torch.isfinite(xa2) & torch.isfinite(xb2) & (xa2.abs() < 1e29) & (xb2.abs() < 1e29)
                    dd = torch.where(fin, (xa2 - xb2).abs(), torch.zeros_like(xa2))
                    mad = float(dd.max()) if bool(fin.any()) else None
                    arg = int(dd.reshape(-1).argmax())
                    mad_at = [arg // dd.shape[-1], arg % dd.shape[-1]]  # (row, pool) of the largest |diff|
                    k_ = min(512, xa2.shape[-1])
                    ia = torch.sort(torch.topk(xa2, k_, dim=-1).indices, dim=-1).values
                    ib = torch.sort(torch.topk(xb2, k_, dim=-1).indices, dim=-1).values
                    rows_diff = int((ia != ib).any(-1).sum())
                    d_t.append(dict(max_abs=mad, max_abs_at=mad_at, a_b_at=[float(xa2[mad_at[0], mad_at[1]]), float(xb2[mad_at[0], mad_at[1]])],
                                    nonfinite_a=int((~torch.isfinite(xa2)).sum()), nonfinite_b=int((~torch.isfinite(xb2)).sum()),
                                    rows=int(xa2.shape[0]), rows_top512_set_differ=rows_diff))  # fmt: skip
                sel_diff[f"t{t}"] = d_t
            row["selection_diff"] = sel_diff
            row.update(selection_equal_bound_vs_unbound=eq, selection_equal_across_windows_bound=cross,
                       score_plane_bitwise=(all(all(v) for v in sm_eq.values() if v is not None) if any(v is not None for v in sm_eq.values()) else None),
                       selection_bitwise=all(all(v) for d_ in eq.values() for v in d_.values()))  # fmt: skip
    if SPLIT:  # select's pieces, each its own traced replay, on the inputs the recorded select used (bound as built)
        from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: PLC0415

        split = {}
        for t, m in keys:  # every (window, bound) key: the score piece follows ix.bound
            set_bound(m)
            r = rec[(t, m)]
            ix, attn, h, ci_, sp_ = r["ix"], r["attn"], r["h"], r["ci"], r["sp"]
            cl = int(h.shape[-2])
            sp_.begin(ci_)  # the window's visibility state (nvis / c_add) stays live for every piece below
            st = {}

            def proj():  # q_idx + head weights (window-independent)
                qk = PT.pf_mm(attn.ds_qa_kva, h)
                q_r = ttnn.rms_norm(ttnn.slice(qk, (0, 0, 0, 0), (1, 1, cl, attn.q_lora)), weight=attn.qa_norm,
                                    epsilon=attn.eps, compute_kernel_config=ix.ckc)  # fmt: skip
                ttnn.deallocate(qk)
                qi = ttnn.linear(q_r, ix.wq_b, dtype=PT.BF16, compute_kernel_config=ix.ckc, memory_config=PT.DRAM)
                ttnn.deallocate(q_r)
                qi, _, _ = ttnn.experimental.nlp_create_qkv_heads(qi, num_heads=PT.INDEX_HEADS, num_kv_heads=0,
                                                                  transpose_k_heads=False, memory_config=PT.DRAM)  # fmt: skip
                w = ttnn.linear(h, ix.w_p, dtype=PT.FP32, compute_kernel_config=ix.ckc, memory_config=PT.DRAM)
                w2 = ttnn.typecast(ttnn.multiply(w, ix.w_scale), PT.BF16)
                ttnn.deallocate(w)
                return qi, w2

            def gather_pools():  # the window's pooled keys -> pool_buf (P = t / 4 pools)
                if sp_.shard is not None and PT.PAGED_POOLS:
                    sp_.paged_pools(ix.store, ci_, sp_.pool_buf)
                elif sp_.shard is not None:
                    sp_._gather_local(ix.store, ci_.pool_user, sp_.p_local // PT.POOL_PAGE, sp_.p // PT.POOL_PAGE, PT.INDEX_DIM // PT.TILE,
                                      sp_.pool_g2l, (1, 1, sp_.p_local, PT.INDEX_DIM), sp_.pool_buf)  # fmt: skip
                else:
                    PT.page_gather(ix.store, ci_.pool_user, sp_.pool_buf, sp_.p // PT.POOL_PAGE, PT.INDEX_DIM // PT.TILE)

            def score(qi, w):
                s_ = ttnn.experimental.indexer_score_dsa(
                    qi, sp_.pool_buf, w, key_compression_ratio=PT.KPOOL, compute_kernel_config=ix.ckc_score,
                    **({"chunk_start_idx_tensor": ci_.chunk_start_u} if ix.bound else {"chunk_start_idx": sp_.cs_fixed}),
                    **({"program_config": ix.score_cfg} if getattr(ix, "score_cfg", None) is not None else {}))  # fmt: skip
                if s_.layout != ttnn.TILE_LAYOUT:
                    s_ = ttnn.to_layout(s_, ttnn.TILE_LAYOUT)
                if s_.dtype != PT.BF16:
                    s_ = ttnn.typecast(s_, PT.BF16)
                return s_

            def mask(s_):  # visibility over [C_l, P] and the masked plane
                vis = ttnn.typecast(ttnn.lt(sp_.pool_iota, sp_.nvis), PT.BF16)
                sm = ttnn.where(vis, s_, PT.NEG)
                ttnn.deallocate(vis)
                return sm

            def topk(sm):  # top-512 pool ids per row over P (row-major in)
                rm = ttnn.to_layout(sm, ttnn.ROW_MAJOR_LAYOUT)
                pools = ttnn.experimental.topk_large_indices(rm, k=PT.TOPK_POOLS)
                ttnn.deallocate(rm)
                return pools

            # inputs of every piece, eager (also the compile pass of each piece before any capture)
            qi, w = proj()
            gather_pools()
            s0 = score(qi, w)
            sm0 = mask(s0)
            pools0 = topk(sm0)
            ttnn.synchronize_device(mesh)
            if REF:  # which mode is right: the host fp64 score of sampled rows over each row's VISIBLE pools
                chips = lambda t_: [ttnn.to_torch(d_).float() for d_ in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]  # noqa: E731
                qcs, wcs, kcs, smcs = chips(qi), chips(w), chips(sp_.pool_buf), chips(sm0)
                refm = []
                for k_chip, (qc, wc, kc, smc) in enumerate(zip(qcs, wcs, kcs, smcs)):
                    qc = qc.reshape(PT.INDEX_HEADS, cl, PT.INDEX_DIM) if qc.shape[1] == PT.INDEX_HEADS else qc.reshape(cl, PT.INDEX_HEADS, PT.INDEX_DIM).transpose(0, 1)
                    wc = wc.reshape(cl, -1)[:, : PT.INDEX_HEADS]
                    kc = kc.reshape(-1, kc.shape[-1])[:, : PT.INDEX_DIM]
                    smc = smc.reshape(cl, -1)
                    ov, mad = [], []
                    for r_ in range(0, cl, REF_STRIDE):
                        pos = P0 + k_chip * cl + r_
                        nv = (pos + 1) // PT.KPOOL
                        kp = kc[:nv].double()
                        ref = (wc[r_].double() @ torch.relu(qc[:, r_, :].double() @ kp.T)).float()  # [nv]
                        dev = smc[r_, :nv]
                        k_ = min(PT.TOPK_POOLS, nv)
                        sr = set(torch.topk(ref, k_).indices.tolist())
                        sd = set(torch.topk(dev, k_).indices.tolist())
                        ov.append(len(sr & sd) / k_)
                        mad.append(float((dev - ref).abs().max()))
                    ent_ = dict(rows=len(ov), top512_overlap_min=round(min(ov), 4), top512_overlap_mean=round(sum(ov) / len(ov), 4),
                                maxabs_max=round(max(mad), 4), maxabs_mean=round(sum(mad) / len(mad), 4))  # fmt: skip
                    # the TRACED replay's plane (the IX_TAP readback after this key's select replays, bound A/B section)
                    # vs the same host reference: separates a replay-vs-eager difference from a readback defect
                    tap_ = (sels.get((t, m)) or {}).get("sm") if "sels" in globals() else None
                    if tap_ is not None:
                        tp_ = tap_[k_chip].float().reshape(cl, -1)
                        tov, tmad, worst = [], [], (0.0, -1, -1)
                        for r_ in range(0, cl, REF_STRIDE):
                            pos = P0 + k_chip * cl + r_
                            nv = (pos + 1) // PT.KPOOL
                            ref = (wc[r_].double() @ torch.relu(qc[:, r_, :].double() @ kc[:nv].double().T)).float()
                            dv = tp_[r_, :nv]
                            k_ = min(PT.TOPK_POOLS, nv)
                            tov.append(len(set(torch.topk(ref, k_).indices.tolist()) & set(torch.topk(dv, k_).indices.tolist())) / k_)
                            dd_ = (dv - ref).abs()
                            mx_ = float(dd_.max())
                            tmad.append(mx_)
                            if mx_ > worst[0]:
                                worst = (mx_, r_, int(dd_.argmax()))
                        ent_.update(tap_top512_overlap_min=round(min(tov), 4), tap_maxabs_max=round(max(tmad), 4),
                                    tap_worst=dict(maxabs=round(worst[0], 4), row=worst[1], pool=worst[2],
                                                   tap=float(tp_[worst[1], worst[2]]), eager=float(smc[worst[1], worst[2]]),
                                                   host=float((wc[worst[1]].double() @ torch.relu(qc[:, worst[1], :].double() @ kc[worst[2]].double())).item())))  # fmt: skip
                    refm.append(ent_)
                row.setdefault("ref_vs_host", {})[f"t{t}" + ("" if m is None else f"_bound{int(m)}")] = refm
                H.log(f"{tag}: host reference t={t} bound={m}: {refm}")
            if not C.MOCK:
                piece = {}
                n_pc = int(mesh.num_program_cache_entries())

                def timed(name, fn_):
                    outs_ = []
                    piece[name] = capture_replay(lambda: outs_.append(fn_()), f"split t={t} bound={m} {name}")
                    for o_ in outs_:
                        for x_ in (o_ if isinstance(o_, tuple) else (o_,)):
                            if x_ is not None:
                                ttnn.deallocate(x_)

                timed("proj", proj)
                timed("gather_pools", gather_pools)
                timed("score", lambda: score(qi, w))
                timed("mask", lambda: mask(s0))
                timed("topk", lambda: topk(sm0))
                assert int(mesh.num_program_cache_entries()) == n_pc, "split: a program compiled inside a capture"
                split[f"t{t}" + ("" if m is None else f"_bound{int(m)}")] = {k_: round(statistics.median(v), 1) for k_, v in piece.items()}
            for x_ in (qi, w, s0, sm0, pools0):
                ttnn.deallocate(x_)
            sp_.end()
        row["split_us"] = split
        H.log(f"{tag}: select split (median us per piece) {split}")
    row["status"] = "mock" if C.MOCK else "pass"
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-5000:]
    H.log(f"FAILED:\n{err}")
    row.update(status="fail", error=err)
res = dict(test=f"op sparse_dsa layer L{L} at {N} valid context tokens: device time vs the trace key window t {TS} (layer, select alone), "
                f"tp{TPN}", milestone="PMK-attn-window", mock=C.MOCK, wall_s=round(time.time() - t_job, 1), tag=tag, **row)  # fmt: skip
H.log({k: v for k, v in res.items() if k not in ("test", "layer_us_all", "select_us_all", "error")})
if rec_on and not C.MOCK:
    C.record("tests", {k: v for k, v in res.items() if k not in ("layer_us_all", "select_us_all")})
RESULT = res
