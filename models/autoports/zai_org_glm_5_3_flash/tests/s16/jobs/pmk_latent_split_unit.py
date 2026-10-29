# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair; 2026-10-24 00:35Z): the BQ attention's window-sized key store
(SparseCtx.local_latents_rm: this chip's WHOLE local latent window -> bf16 ROW_MAJOR, every forward) piece by piece,
each traced alone (`replays` blocking replays, median us), at every window t -- lat_gather (page_gather of the local
latent pages), lat_typecast (if the cache is not bf16), lat_rm (to_layout ROW_MAJOR), lat_total (local_latents_rm) --
then the pooled-key gather pieces as before:
  local    page_gather of this chip's nb_l local block pages (4 tiles each) -> loc
  ag       the all_gather of loc (tpp.gather: [C x nb_l x 32, 128])
  g2l      page_gather g2l un-permute -> pad [nb x 32, 128] (global block order)
  rm       to_layout(pad, ROW_MAJOR)
  slice    rows 0-15 of each block
  tile     to_layout TILE of the [nb x 16, 128] rows
  copy     into pool_buf
  total    paged_pools itself
plus select alone (begin + select + end) with the served knobs (GLM53_PF_IX_BOUND + GLM53_PF_IX_TOPK_BOUND from env).
Build / prefix as pmk_ix_topk_ab_unit (L2 + L3, a real 2,048-token prefix, the chunk at chunk_start).
ARGS: layer (3), chunk_start (2048), windows ([65536, 1048576]), replays (10), env, tag."""
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

from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
L = int(args.get("layer", 3))
BUILD = [int(v) for v in args.get("build_layers", [L - 1, L])]
N = 2048
P0 = int(args.get("chunk_start", 2048))
TS = sorted(int(v) for v in args.get("windows", [65536, 1048576]))
REPLAYS = int(args.get("replays", 10))
# lat_ab (False): GLM53_PF_LAT_BOUND A/B per window -- the key store local_latents_rm with sp.lat_bound False vs True
# (each captured, copied into a persistent tensor), compared bitwise on each chip's VISIBLE local rows (the local units
# of ceil((cs + c) / 128) x 128), plus the layer timed per arm
LAT_AB = bool(args.get("lat_ab", False))
tag = args.get("tag", f"pmk-pool-gather-split-L{L}-tp{TPN}")
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
row, err = dict(layer=L, chunk_start=P0, windows=TS, tp=TPN, ix_bound=os.environ.get("GLM53_PF_IX_BOUND", ""),
                topk_bound=os.environ.get("GLM53_PF_IX_TOPK_BOUND", "")), None  # fmt: skip
rec = {"arm": None}
_mla_call = PT.MlaPrefill.__call__


def _mla_rec(self, h, cache, ci, *a, **k):  # the sparse layer's latent cache (an argument of MlaPrefill.__call__)
    if getattr(self, "idx", None) is not None:
        rec["cache"] = cache
    return _mla_call(self, h, cache, ci, *a, **k)


PT.MlaPrefill.__call__ = _mla_rec


def make_hook(ix):
    bound = ix.select

    def hook(attn, h_local, ci, sp):
        if rec["arm"] is not None and sp is not None:
            rec[rec["arm"]] = dict(ix=ix, attn=attn, h=ttnn.clone(h_local), ci=ci, sp=sp, fn=bound)
            rec["arm"] = None
        return bound(attn, h_local, ci, sp)

    return hook


def capture_replay(fn, label):
    if C.MOCK:  # the mock cannot capture: the eager call only (its host path)
        fn()
        return [0.0]
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ts = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append(round((time.perf_counter() - t0) * 1e6, 1))
    ttnn.release_trace(mesh, tid)
    H.log(f"{tag}: {label}: {ts}")
    return ts


try:
    ent = S.build(mesh, ctx, BUILD, b=2, buckets=(N,), max_seq=max(TS) + 512)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    row["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=2)
    c = pf.bucket_for(N)
    for pl_ in pf.layers:
        ix_ = getattr(getattr(pl_, "mla", None), "idx", None)
        if ix_ is not None:
            ix_.select = make_hook(ix_)
    g32 = S.golden_prompt(32768).reshape(-1).long()
    k0 = pf.plan(0, N)
    pf.write_inputs(c, g32[:N], slot=0, chunk_start=0, n_valid=N)
    pf.forward(c, head=False, t=k0[1])
    for t in TS:  # one eager pass per window (its compile pass) + the recorded select inputs
        pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)
        rec["arm"] = t
        pf.forward(c, head=False, t=t)
        ttnn.synchronize_device(mesh)
        assert t in rec, f"select not recorded at t={t}"
    out = {}
    pieces_all = {}
    for t in TS:
        r = rec[t]
        ix, ci, sp = r["ix"], r["ci"], r["sp"]
        pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)
        sp.begin(ci)
        nb_l, nb = sp.t_local // PT.BLOCK, sp.t // PT.BLOCK
        tpp_ = PT.INDEX_DIM // PT.TILE
        mk = lambda shape: ttnn.empty(shape, dtype=PT.BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PT.DRAM)  # noqa: E731
        loc = mk((1, 1, nb_l * PT.TILE, PT.INDEX_DIM))
        pad = mk((1, 1, nb * PT.TILE, PT.INDEX_DIM))
        st = {}

        def p_local():
            PT.page_gather(ix.store, ci.user_table, loc, nb_l, tpp_)

        def p_ag():
            g_ = pf.gather(loc, True) if TPN > 1 else None
            st["g"] = g_

        def p_g2l():
            PT.page_gather(st["g"], sp.lat_g2l, pad, nb, tpp_)

        def p_rm():
            st["rm"] = ttnn.to_layout(pad, ttnn.ROW_MAJOR_LAYOUT, memory_config=PT.DRAM)

        def p_slice():
            st["top"] = ttnn.slice(ttnn.reshape(st["rm"], (1, nb, PT.TILE, PT.INDEX_DIM)), (0, 0, 0, 0), (1, nb, PT.POOL_BLOCK_ROWS, PT.INDEX_DIM))

        def p_tile():
            st["flat"] = ttnn.to_layout(ttnn.reshape(st["top"], (1, 1, nb * PT.POOL_BLOCK_ROWS, PT.INDEX_DIM)), ttnn.TILE_LAYOUT,
                                        memory_config=PT.DRAM)  # fmt: skip

        def p_copy():
            ttnn.copy(st["flat"], sp.pool_buf)

        def p_total():
            sp.paged_pools(ix.store, ci, sp.pool_buf)

        cache = rec.get("cache")
        if cache is None:  # the layer's latent cache: TPPrefill.kv.extra["latent"][li] (prefill_tp layer(): pl.mla(h, ...))
            li = next(i for i, pl_ in enumerate(pf.layers) if int(pl_.L) == L)
            cache = pf.kv.extra["latent"][li]
        row.setdefault("cache_attr", type(cache).__name__ if cache is not None else None)
        tpl = (PT.BLOCK // PT.TILE) * (512 // PT.TILE)
        lat = mk_lat = None
        if cache is not None:
            mk_lat = lambda: ttnn.empty((1, 1, sp.t_local, 512), dtype=cache.dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PT.DRAM)  # noqa: E731
            lat = mk_lat()
            row["latent_dtype"] = str(cache.dtype)

        def l_gather():
            PT.page_gather(cache, ci.user_table, lat, sp.t_local // PT.BLOCK, tpl)

        def l_typecast():
            st["lt"] = ttnn.typecast(lat, PT.BF16) if lat.dtype != PT.BF16 else None

        def l_rm():
            st["lrm"] = ttnn.to_layout(st.get("lt") or lat, ttnn.ROW_MAJOR_LAYOUT, memory_config=PT.DRAM)

        def l_total():
            st["ltot"] = sp.local_latents_rm(cache, ci)

        order = ([("lat_gather", l_gather), ("lat_typecast", l_typecast), ("lat_rm", l_rm), ("lat_total", l_total)] if cache is not None else []) + \
                [("local", p_local), ("ag", p_ag), ("g2l", p_g2l), ("rm", p_rm), ("slice", p_slice), ("tile", p_tile), ("copy", p_copy), ("total", p_total)]
        for nm_, fn in order:  # the compile pass of every piece (eager, in order: each piece's input exists)
            fn()
        ttnn.synchronize_device(mesh)
        sp.end()
        sel_pre = dict(r)  # select alone's compile pass too, before this window's first capture
        sel_pre["sp"].begin(sel_pre["ci"])
        try:
            sel_pre["fn"](sel_pre["attn"], sel_pre["h"], sel_pre["ci"], sel_pre["sp"])
        finally:
            sel_pre["sp"].end()
        ttnn.synchronize_device(mesh)
        n_pc = None if C.MOCK else int(mesh.num_program_cache_entries())
        pieces = {}
        sp.begin(ci)
        for nm_, fn in order:  # each piece traced alone on the inputs the compile pass left (outputs freed after)
            keep = dict(st)

            def body(fn=fn, nm_=nm_):
                fn()
                for k_ in ("g", "rm", "top", "flat", "lt", "lrm", "ltot"):  # the piece's fresh output: freed inside the capture (its input stays)
                    if st.get(k_) is not None and st[k_] is not keep.get(k_):
                        ttnn.deallocate(st[k_])
                        st[k_] = keep.get(k_)

            pieces[nm_] = capture_replay(body, f"t={t} {nm_}")
        sp.end()
        for k_ in ("g", "rm", "top", "flat", "lt", "lrm", "ltot"):
            if st.get(k_) is not None:
                ttnn.deallocate(st[k_])
                st[k_] = None
        if lat is not None:
            ttnn.deallocate(lat)
        ttnn.deallocate(loc)
        ttnn.deallocate(pad)
        if LAT_AB and cache is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH  # noqa: PLC0415

            um = -(-(P0 + N) // DSH.UNIT)
            vis_rows = [max(0, -(-(um - r_) // TPN)) * DSH.UNIT for r_ in range(TPN)]
            kvp = {}
            ab = {}
            # both arms on ONE cache state: compile both, replay store A and store B back to back (no forward in
            # between: a layer replay re-advances L2's KDA state and rewrites the chunk's latent rows -- latabu 00:57Z:
            # the prefix rows matched, the chunk's own rows did not), then time the layer per arm
            arms = {}
            for m in (False, True):
                sp.lat_bound = m
                pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)
                kv0 = sp.local_latents_rm(cache, ci)  # the compile pass of this arm
                kvp[m] = ttnn.empty(tuple(kv0.shape), dtype=kv0.dtype, layout=kv0.layout, device=mesh, memory_config=PT.DRAM)
                ttnn.deallocate(kv0)

                def arm(m=m):
                    kv_ = sp.local_latents_rm(cache, ci)
                    ttnn.copy(kv_, kvp[m])
                    ttnn.deallocate(kv_)

                arm()  # the copy's compile pass
                arms[m] = arm
            sp.lat_bound = False
            pf.forward(c, head=False, t=t)  # the layer's compile pass with each arm (lat_bound True is new)
            sp.lat_bound = True
            pf.forward(c, head=False, t=t)
            ttnn.synchronize_device(mesh)
            n_arm = None if C.MOCK else int(mesh.num_program_cache_entries())
            pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)
            for m in (False, True):  # the two stores back to back
                sp.lat_bound = m
                ab[f"store_us_lat{int(m)}"] = round(statistics.median(capture_replay(arms[m], f"t={t} key store lat_bound={int(m)}")), 1)
            ttnn.synchronize_device(mesh)
            for m in (False, True):  # then the layer per arm (timing only)
                sp.lat_bound = m
                pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)
                ab[f"layer_us_lat{int(m)}"] = round(statistics.median(capture_replay(lambda: pf.forward(c, head=False, t=t),
                                                                                     f"t={t} layer lat_bound={int(m)}")), 1)  # fmt: skip
            if n_arm is not None:
                assert int(mesh.num_program_cache_entries()) == n_arm, f"t={t}: a program compiled inside a capture"
            if not C.MOCK:
                ga = [ttnn.to_torch(d_) for d_ in ttnn.get_device_tensors(kvp[False])]
                gb = [ttnn.to_torch(d_) for d_ in ttnn.get_device_tensors(kvp[True])]
                ab["visible_rows"] = vis_rows
                ab["store_visible_bitwise"] = [bool(torch.equal(a_.reshape(-1, 512)[:v].view(torch.int16), b_.reshape(-1, 512)[:v].view(torch.int16)))
                                               for a_, b_, v in zip(ga, gb, vis_rows)]  # fmt: skip
                diag = []
                for a_, b_, v in zip(ga, gb, vis_rows):  # where they differ: rows, first (row, col), zeros, a row permutation?
                    A_, B_ = a_.reshape(-1, 512)[:v].view(torch.int16), b_.reshape(-1, 512)[:v].view(torch.int16)
                    bad = (A_ != B_).any(-1)
                    nb = int(bad.sum())
                    d_ = dict(rows_differ=nb, rows=v, b_zero_rows=int((B_ == 0).all(-1).sum()), a_zero_rows=int((A_ == 0).all(-1).sum()))
                    if nb:
                        r0 = int(torch.nonzero(bad)[0])
                        c_bad = torch.nonzero(A_[r0] != B_[r0]).flatten()
                        d_.update(first_row=r0, first_cols=c_bad[:8].tolist(), cols_differ_first_row=int(c_bad.numel()),
                                  bad_rows_head=torch.nonzero(bad).flatten()[:12].tolist())  # fmt: skip
                        hits = []  # B's bad row = which A row (first 64 rows of A searched) / column-half swaps
                        for r_ in torch.nonzero(bad).flatten()[:4].tolist():
                            m_ = [int(q) for q in torch.nonzero((A_ == B_[r_]).all(-1)).flatten()[:3].tolist()]
                            half = [bool(torch.equal(A_[r_, :256], B_[r_, 256:])), bool(torch.equal(A_[r_, 256:], B_[r_, :256]))]
                            hits.append(dict(row=r_, equals_a_rows=m_, half_swapped=half))
                        d_["b_row_matches"] = hits
                    diag.append(d_)
                ab["store_diff"] = diag
                del ga, gb
            for m in (False, True):
                ttnn.deallocate(kvp[m])
            sp.lat_bound = False
            n_pc = None if C.MOCK else int(mesh.num_program_cache_entries())  # the lat A/B compiled outside captures
            row.setdefault("lat_ab", {})[f"t{t}"] = ab
            H.log(f"{tag}: t={t} LAT AB {ab}")
        # select alone with the served knobs
        pf.write_inputs(c, g32[P0:P0 + N], slot=0, chunk_start=P0, n_valid=N)

        def sel():
            r["sp"].begin(r["ci"])
            try:
                r["fn"](r["attn"], r["h"], r["ci"], r["sp"])
            finally:
                r["sp"].end()

        pieces["select"] = capture_replay(sel, f"t={t} select")
        if n_pc is not None:
            assert int(mesh.num_program_cache_entries()) == n_pc, f"t={t}: a program compiled inside a capture"
        pieces_all[t] = pieces
        out[f"t{t}"] = {k_: round(statistics.median(v), 1) for k_, v in pieces.items()}
        out[f"t{t}"].update(nb=nb, nb_l=nb_l)
        H.log(f"{tag}: t={t} medians {out[f't{t}']}")
    row.update(split_us=out, status="mock" if C.MOCK else "pass")
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-5000:]
    H.log(f"FAILED:\n{err}")
    row.update(status="fail", error=err)
PT.MlaPrefill.__call__ = _mla_call
res = dict(test=f"the pooled-key gather (paged_pools) piece by piece vs the key window {TS}, L{L}, cs {P0}, tp{TPN}", tag=tag,
           wall_s=round(time.time() - t_job, 1), **row)  # fmt: skip
H.log({k: v for k, v in res.items() if k not in ("test", "error")})
RESULT = res
