# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 / pair HOST MOCK, or a device): the PMK attention phase A6 alone (tt/pmk_attn.py
PmkAttention.standalone) on synthetic inputs, dense and sparse geometries, both TP geometries. Under the mock the
kernels JIT-build and the CBs / L1 are allocated (the ledger); on a device the output is compared to the host reference
(attention_host) -- the ONE bench unit (README): TP2 pair, L3-shaped (32 heads), chunk 2,048 at a 1,024-row context ...
ARGS: geos (["tp8-dense", "tp8-sparse", "tp2-dense", "tp2-sparse"]), ctx (context rows of the sparse cases: 4096),
max_seq (8192), check (device only: compare vs the host reference), nk (2), qg (4), lat_l1 (true: the chunk latent
store L1-sharded over 64 cores; false: DRAM), probe (false: the in-program cycle counters, decoded on a device),
replays (device only: traced replays timed, 0 = none), qg_sparse (the sparse geometries' qg), rm_rows (1: the row-major latent store as the op path's
[n, 512] (one row per page); 4: [n / 4, 2048] (one pool per page)), obf16 (None: fp32 O partials when they fit),
dual_noc (true: the dense chunk reads split over both NoCs, rev 7), kslots (2: CB_AK slots), bm_pages (None: 2 when
they fit; the sparse bitmap double buffer); rev 8: the sparse selection goes to the device as per-row BITMAPS
(select_bitmap_host of the synthetic ids; the ids stay host-side for the reference), fid_qk ("" = the kernel's config
fidelity; "HiFi2": the two bf16 Q K / P V products at HiFi2 -- the sparse unit row), exp_approx / max_lazy (rev 9
softmax knobs: the exp_21f SFPU exp; rescale the partial only when a row's max grows by more than T),
tag, record. Rev 4: the compared output is the TRACED replay's; the row carries the dm0 / dm1 / compute probe shares,
the fan-in model's prediction (estimate.span_sim_us) and, on a failure, the coordinates of the non-finite values.
Rev 12: geos "tp8-sp" / "tp2-sp" = the token-major EXACT mode (README §14): the chip's SP rows (256 / 1,024, from
chunk-relative row `row0`, default 0) x all 64 heads; inputs q [1, 1, SP x 64, 512] token-major (tile row t 2 + j =
heads 32 j .. of token t), the selection as ids [1, 1, SP, 512] uint32 (sorted, NONE padded), the row-major latent
store over the whole slot; the output o [1, 1, SP x 64, 512] token-major, compared per (head, token) to
attention_host."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_attn as A  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
MOCK = os.environ.get("GLM53_MOCK") == "1"
GEOS = list(args.get("geos", ["tp8-dense", "tp8-sparse", "tp2-dense", "tp2-sparse"]))
CTX = int(args.get("ctx", 4096))
MAX_SEQ = int(args.get("max_seq", 8192))
CHECK = bool(args.get("check", not MOCK))
NK, QG = args.get("nk", None), args.get("qg", None)  # None: the geometry defaults (nk 4 / 2, qg by auto_qg)
QG = int(QG) if QG is not None else None
QG_SPARSE = args.get("qg_sparse", None)  # the sparse geometries' qg (the unit pins 2: the TP8 sparse choice, so the
QG_SPARSE = int(QG_SPARSE) if QG_SPARSE is not None else QG  # in-place partial region j = 1 is exercised)
NK = int(NK) if NK is not None else None
LAT_L1 = bool(args.get("lat_l1", True))
PROBE = bool(args.get("probe", False))
RM_ROWS = int(args.get("rm_rows", 1))
OBF16 = args.get("obf16", None)
DUAL_NOC = bool(args.get("dual_noc", True))
KSLOTS = int(args.get("kslots", 2))
BM_PAGES = args.get("bm_pages", None)
BM_PAGES = int(BM_PAGES) if BM_PAGES is not None else None
FID_QK = str(args.get("fid_qk", "")) or None
FID_PV = str(args.get("fid_pv", "")) or None  # rev 12.1: the P V product's fidelity (default: fid_qk)
O_COPY = bool(args.get("o_copy", False))  # rev 12.1: the rev 12.0 copy-in O path instead of the packer L1 accumulation
EXP_APPROX = bool(args.get("exp_approx", False))  # rev 9
MAX_LAZY = args.get("max_lazy", None)  # rev 9: the lazy-rescale threshold T (e.g. 32.0); None = every batch
MAX_LAZY = float(MAX_LAZY) if MAX_LAZY is not None else None
EXP_21F = bool(args.get("exp_21f", False))  # rev 10
CTX_L1 = args.get("ctx_l1", None)  # rev 10: the context L1 store (None: when its per-owner share fits)
CTX_L1 = bool(CTX_L1) if CTX_L1 is not None else None
REPLAYS = int(args.get("replays", 0))
ROW0 = int(args.get("row0", 0))  # rev 12 SP: the chip's first chunk-relative row
tag = args.get("tag", "pmk-attn-mock")
rec = bool(args.get("record", False))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
DRAM = ttnn.DRAM_MEMORY_CONFIG
g = torch.Generator().manual_seed(int(args.get("seed", 0)))


def dev(t_, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t_, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=repl)


def dev0(t_):
    return ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()


rows = []
grid = A.attention_grid(mesh, 104)
for name in GEOS:
    tp, kind = name.split("-")
    cs = CTX if kind in ("sparse", "sp") else 0
    if kind == "sp":  # rev 12: the token-major exact mode (all 64 heads x the chip's SP rows)
        geo = (A.AttnGeo.tp8_sp if tp == "tp8" else A.AttnGeo.tp2_sp)(cs, max_seq=MAX_SEQ, nk_=NK, obf16=OBF16,
                                                                       dual_noc=DUAL_NOC, bm_pages_=BM_PAGES,
                                                                       exp_approx=EXP_APPROX, max_lazy=MAX_LAZY,
                                                                       exp_21f=EXP_21F, row0=ROW0)  # fmt: skip
    else:
        geo = (A.AttnGeo.tp8 if tp == "tp8" else A.AttnGeo.tp2)(cs, max_seq=MAX_SEQ, nk_=NK, lat_l1=LAT_L1, obf16=OBF16,
                                                                 qg_=QG_SPARSE if kind == "sparse" else QG,
                                                                 dual_noc=DUAL_NOC, kslots=KSLOTS, bm_pages_=BM_PAGES,
                                                                 exp_approx=EXP_APPROX, max_lazy=MAX_LAZY, exp_21f=EXP_21F,
                                                                 ctx_l1_=CTX_L1)  # fmt: skip
    h, c = geo.heads, geo.rows
    if geo.sp:  # the chunk is still 2,048 rows: the row-major store (and the reference's keys) span cs + row0 + SP
        chunk_rows = 2048
        assert ROW0 + c <= chunk_rows, (ROW0, c)
        n = cs + chunk_rows
    else:
        n = cs + c
    # inputs: q_lat [1, H_r, C, 512], the chunk latent [1, 1, C, 512], the context in the paged cache (BLOCK = 64 rows
    # per page, identity page table over the slot) and in the row-major store, the sparse ids
    q_h = (torch.randn(1, h, c, 512, generator=g) * 0.3).to(torch.bfloat16)
    lat_h = (torch.randn(n, 512, generator=g) * 0.3).to(torch.bfloat16)
    pages = MAX_SEQ // A.BLOCK
    cache_h = torch.zeros(pages, 1, A.BLOCK, 512, dtype=torch.bfloat16)
    cache_h.reshape(pages * A.BLOCK, 512)[:cs] = lat_h[:cs]
    pt_h = torch.arange(pages, dtype=torch.int32).reshape(1, pages)
    ids_h = torch.full((c, A.TOPK), A.NONE_ID, dtype=torch.int64)
    cs_rows = cs + (ROW0 if geo.sp else 0)  # the position of row 0 of q_h
    if geo.sparse:
        scores = torch.randn(c, n // A.KPOOL, generator=g)
        ids_h = A.select_rows_host(scores, cs_rows + torch.arange(c))
    if geo.sp:  # token-major q [1, 1, SP x H, 512]: tile row t qg + j = heads 32 j .. 32 j + 31 of token t
        q = dev(q_h[0].permute(1, 0, 2).reshape(1, 1, c * h, 512).contiguous(), ttnn.bfloat16)
    else:
        q = dev(q_h, ttnn.bfloat16)
    chunk_lat = lat_h[cs : cs + (2048 if geo.sp else c)]
    if LAT_L1 and not geo.sp:  # the chunk latent store L1-sharded over the first 64 attention cores (replicated)
        if TPN > 1:
            lat_all = ttnn.from_torch(chunk_lat.reshape(1, 1, -1, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=mesh, mesh_mapper=repl,
                                      memory_config=A.latent_store(mesh, c, grid).memory_config())  # fmt: skip
        else:
            lat_all = A.latent_store(mesh, c, grid, chunk_lat.reshape(1, 1, -1, 512))
    else:
        lat_all = dev(chunk_lat.reshape(1, 1, -1, 512), ttnn.bfloat16)
    cache = dev(cache_h, ttnn.bfloat16)
    pt = dev(pt_h, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    lat_rm = dev(lat_h.reshape(1, 1, n // RM_ROWS, 512 * RM_ROWS), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    if geo.sp:  # rev 12: the selection as sorted ids, one 2-KB page per token
        sel = dev(ids_h.to(torch.int32).reshape(1, 1, c, A.TOPK), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    else:
        sel_h = A.select_bitmap_host(ids_h, geo.ct0 * A.POOLS_PER_TILE)  # rev 8: the per-row selection bitmaps
        sel = dev(sel_h.to(torch.int32).reshape(1, 1, c, sel_h.shape[1]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    probe = A.probe_tensor(mesh, geo) if PROBE else None
    ctx_store = A.context_store(mesh, geo, grid, lat_h[:cs]) if geo.ctx_l1 else None  # rev 10
    att = A.PmkAttention(mesh, geo, grid, scale=0.0625, probe=probe, fid_qk=FID_QK, fid_pv=FID_PV, o_copy=O_COPY)
    t0 = time.perf_counter()
    o = att.standalone(q, cache, pt, lat_all, sel, lat_rm, ctx_store=ctx_store)  # the compile launch (the one eager launch, never compared)
    ttnn.synchronize_device(mesh)
    build_s = time.perf_counter() - t0
    replay_us = None
    if (
        not MOCK
    ):  # operator rule 09:11Z: the compared output comes from a TRACED replay (one capture, its replays timed)
        ttnn.deallocate(o)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            o = att.standalone(q, cache, pt, lat_all, sel, lat_rm, ctx_store=ctx_store)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        us = []
        for _ in range(max(REPLAYS, 1)):
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            us.append((time.perf_counter() - t1) * 1e6)
        ttnn.synchronize_device(mesh)
        replay_us = sorted(us)[len(us) // 2]
    knobs = (f"fid_qk {att.fid_qk or 'config'}, fid_pv {att.fid_pv or 'config'}, o_l1acc {not att.o_copy}, exp_approx {geo.exp_approx}, exp_21f {geo.exp_21f}, max_lazy {geo.max_lazy}, "
             f"ctx_l1 {geo.ctx_l1}, dual_noc {geo.dual_noc}, " + (f"dual_noc {geo.dual_noc}, kslots {geo.kslots}" if not geo.sparse else
             f"SP row0 {geo.row0}, tiles/token {geo.sp_tiles}, bm_pages {geo.bm_pages}" if geo.sp else
             f"bm_pages {geo.bm_pages}, group tiles/item max {max(A.group_tiles(ids_h, geo.ct0 * A.POOLS_PER_TILE, q0, A.item_qg(geo, q0)) for q0 in range(0, geo.ct, geo.qg))}"))  # fmt: skip
    row = dict(test=f"pmk attn A6 rev{A.ATT_REV} {name}: {h} heads x {c} rows at context {cs}, {geo.cores} cores, NK "
                    f"{geo.nk}, QG {geo.qg}, lat_l1 {LAT_L1}, O {'bf16' if geo.o_bf16 else 'fp32'}, rm_rows {RM_ROWS}, "
                    f"{knobs}",
               milestone="PMK", geo=name, items=geo.items, balance=A.balance(geo), l1=A.l1_plan(geo),
               estimate=A.math_estimate(geo, sim=not geo.sparse), build_s=round(build_s, 2),
               traced=not MOCK, replay_us=replay_us, replays=REPLAYS, mock=MOCK, tag=tag)  # fmt: skip
    if PROBE and not MOCK:  # the probe of the LAST traced replay
        row["probe"] = A.decode_probe(dev0(probe).to(torch.int64), geo)
    if CHECK:
        ref = A.attention_host(q_h[0].float(), lat_h.float(), cs_rows, ids_h if geo.sparse else None)
        if geo.sp:  # token-major [SP x H, 512] -> [H, SP, 512]
            got = dev0(o)[0, 0].reshape(c, h, 512).permute(1, 0, 2).contiguous()
        else:
            got = dev0(o)[0, :h]
        bad = ~torch.isfinite(got)
        row.update(pcc=round(C.pcc(got, ref), 6), max_abs=float("%.3g" % float((got - ref).abs().max())),
                   nonfinite=int(bad.sum()))  # fmt: skip
        if bool(bad.any()):  # where: (head, row, col) of the first 16, the rows / heads / cols touched, the finite PCC
            w = bad.nonzero()[:16].tolist()
            fin = ~bad
            row.update(nonfinite_where=w, nonfinite_rows=sorted({r for _, r, _ in bad.nonzero().tolist()})[:32],
                       nonfinite_heads=sorted({hh for hh, _, _ in bad.nonzero().tolist()}),
                       nonfinite_cols=sorted({cc for _, _, cc in bad.nonzero().tolist()})[:32],
                       pcc_finite=round(C.pcc(got[fin], ref[fin]), 6))  # fmt: skip
        if row["pcc"] < 0.999:  # where the (finite or not) errors sit: per head, per 32-row tile, per 32-col tile
            err = (got - ref).abs()
            err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
            thr = max(1e-2, 10.0 * float(ref.abs().mean()))
            big = err > thr
            hh, rr, cc = (big.nonzero()[:, i] for i in range(3)) if bool(big.any()) else ([], [], [])
            row["err"] = dict(thr=round(thr, 4), n_big=int(big.sum()), of=int(big.numel()),
                              heads=sorted({int(v) for v in hh}), row_tiles=sorted({int(v) // 32 for v in rr})[:64],
                              col_tiles=sorted({int(v) // 32 for v in cc}),
                              rows_first=sorted({int(v) for v in rr})[:32],
                              per_head_pcc=[round(C.pcc(got[i], ref[i]), 4) for i in range(int(got.shape[0]))],
                              zero_rows=int((got.abs().sum(dim=(0, 2)) == 0).sum()))  # fmt: skip
        row["status"] = "pass" if row["pcc"] >= 0.999 and row["nonfinite"] == 0 else "fail"
        if not (row["pcc"] >= 0.9999):  # rev 12.2: the program bar (NaN included) -- localise (per head / token /
            # key-tile hypotheses) + dump the compared tensors
            row["bar_0_9999"] = "fail"
            d = (got - ref)
            tok_pcc = torch.tensor([C.pcc(got[:, t], ref[:, t]) for t in range(int(got.shape[1]))])
            head_pcc = torch.tensor([C.pcc(got[i], ref[i]) for i in range(int(got.shape[0]))])
            worst = tok_pcc.argsort()[:8].tolist()
            row["loc"] = dict(head_pcc_min=round(float(head_pcc.min()), 6), head_pcc_max=round(float(head_pcc.max()), 6),
                              tok_pcc_min=round(float(tok_pcc.min()), 6), tok_pcc_median=round(float(tok_pcc.median()), 6),
                              tok_pcc_lt_0_9999=int((tok_pcc < 0.9999).sum()), worst_tokens=worst,
                              worst_tok_pos_mod4=[int((cs_rows + t) % 4) for t in worst],
                              err_rms=float("%.3g" % float(d.pow(2).mean().sqrt())), ref_rms=float("%.3g" % float(ref.pow(2).mean().sqrt())),
                              row_err_rms_by_pos_mod4=[float("%.3g" % float(d[:, (cs_rows + torch.arange(int(got.shape[1]))) % 4 == m].pow(2).mean().sqrt())) for m in range(4)])  # fmt: skip
            if geo.sp:  # which key tile explains a token's error: least squares of d[:, t] on the keys' V of each tile
                fits = []
                for t in worst[:4]:
                    sel_t = ids_h[t][ids_h[t] != A.NONE_ID]
                    tiles = [sel_t[j : j + 8] for j in range(0, int(sel_t.numel()), 8)] + [torch.tensor([int((cs_rows + t) // 4)])]
                    dt = d[:, t]  # [H, 512]
                    scores_t = []
                    for ti, pools in enumerate(tiles):
                        rows_k = torch.cat([torch.arange(int(pp) * 4, int(pp) * 4 + 4) for pp in pools])
                        v = lat_h[rows_k].float()  # [k, 512]
                        coef = torch.linalg.lstsq(v.T, dt.T).solution  # [k, H]
                        resid = dt.T - v.T @ coef
                        scores_t.append(1.0 - float(resid.pow(2).sum() / dt.pow(2).sum()))
                    best = max(range(len(scores_t)), key=lambda i: scores_t[i])
                    fits.append(dict(tok=t, best_tile=best if best < len(tiles) - 1 else "tail", explained=round(scores_t[best], 3),
                                     tail_explained=round(scores_t[-1], 3), n_tiles=len(tiles)))
                row["loc"]["tile_fit"] = fits
            try:
                out_dir = os.environ.get("GLM53_OUT_DIR", ".")
                fn = os.path.join(out_dir, f"attn_dump_{tag}_{name}.pt")
                torch.save(dict(got=got.to(torch.bfloat16), ref=ref.to(torch.bfloat16), ids=ids_h, cs_rows=cs_rows, geo=str(geo)), fn)
                row["dump"] = fn
            except Exception as e:  # noqa: BLE001
                row["dump"] = f"failed: {e}"
    else:
        row["status"] = "mock-built"
    S.log(row)
    rows.append(row)
    if rec:
        C.record("pcc" if CHECK else "tests", row)
    if not MOCK:
        ttnn.release_trace(mesh, tid)
    for t_ in (q, lat_all, cache, pt, lat_rm, sel, o, att.mask) + ((probe,) if probe is not None else ()) + (
        (ctx_store,) if ctx_store is not None else ()
    ):
        ttnn.deallocate(t_)
RESULT = rows
