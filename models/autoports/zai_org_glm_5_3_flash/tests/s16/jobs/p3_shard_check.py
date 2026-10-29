# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh; 2026-10-24 01:05Z, the 32K first-token miss of the SHARDED 1M construction, program
01:01:32Z): p3_long_check + (a) under GLM53_DSA_SHARD the owner-aware page table published after the build (without it
every block maps to the null page), (b) a sha1 of every chunk's layer output (out_tap: write them to a JSON; out_ref: a
previous run's JSON -> per-chunk bitwise equality = the A vs A determinism control), (c) the golden-ids overlap skipped
under shard (BQ ids are this chip's LOCAL owned rows, not global token ids). The original docstring follows.
Stage 16 P3 long-context rung -- ONE layer of the TP prefill (any kind) over a long golden
prompt (Stage 00c layer_io/L<ii>/<n>: the 32,768-token prompt's streams_in -> streams_out, fp32 CPU), chunk by chunk
through the traced layer: the dense key (bucket, 0 / FRESH) for chunk 0 and ONE sparse key (bucket, T = n) replayed
for every later chunk with that chunk's inputs (the production flow: one trace per key, replays per chunk). Both keys
compile (eager) before the first capture. Per chunk: the replayed layer output vs the golden streams_out (PCC, bar
0.995), the replay time, and -- sparse chunks with GLM53_PF_SPARSE_ATTN=sdpa -- the selected ids of every row vs the
golden selection (layer_io/L<ii>/<n>/topk.pt [1, n, 2051] int32 token ids, -1 = none): mean / min set overlap.
KDA layers: one key (bucket, 0) for every chunk, the state carried chunk to chunk in the layer's scratch (zeroed before
chunk 0), and the final state vs the golden (kda_state/<n>/L<ii>.pt).

ARGS: layers ([2, 3]: the backend's layers; it needs a non-DSA one), check ([3]: the layers run, each over the whole
prompt; every key of every checked layer compiles before the first capture), n (32768), chunk (2048), t (n: the sparse
key's window), rows (the chunks to check: all), the P3 knobs (s16common.set_attn: sparse_attn, mla_form, ...), moe
("ep"), moe_v2, mhc, mhc_post, replays (s16common.SOAK_REPLAYS: the per-key timing soak on the key's last chunk, after
the PCC rows), soak_s (0: the watched unit's unwatched soak of every trace), last_valid (CH: the LAST chunk's n_valid --
a serving request's partial final chunk; its rows past last_valid are zero input (pad "zero") or the golden's ("golden"),
and only its valid rows are compared; every row reports NaN / Inf counts of the compared rows), pad, tag, record.
INDEXER A/B (the exact-lever bars, doc/ttnn/14-production-tp8-megakernel/dsa/indexer_scaling.md section 8): ix_tap (a
directory): after every sparse chunk's replay, dump this run's indexer taps -- the masked score plane of every chip
(GLM53_PF_IX_TAP=1: DsaIndexer.taps, [C_l, P] bf16 as raw uint16) and the selected ids [C, W] -- to
<ix_tap>/L<L>_c<k>_d<r>.pt; ix_ref (a directory of such dumps, the DEFAULT config's): compare chunk by chunk -> row fields
ix_score_bitwise (every chip's plane equal bit for bit), ix_score_maxabs, ix_sel_equal_frac (rows whose selected SET is
the reference's), summary ix_score_bitwise_all / ix_sel_equal_min.  Both runs need GLM53_PF_IX_TAP=1 in env."""
import gc
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402
import pmk45_alias as A45  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [2, 3])]
CHECK = [int(v) for v in args.get("check", [args.get("layer", 3)])]
N = int(args.get("n", 32768))
CH = int(args.get("chunk", 2048))
T = int(args.get("t", N))
LAST_NV = int(args.get("last_valid", 0)) or None
EAGER = bool(args.get("eager", False))  # diagnosis only: every chunk an eager forward (no capture / replay)
PAD = args.get("pad", "zero")
tag = args.get("tag", "p3-long")
rec = bool(args.get("record", True))
IX_TAP = args.get("ix_tap") or None  # indexer A/B: dump dir
IX_REF = args.get("ix_ref") or None  # indexer A/B: the reference dumps (the default config's run)
if IX_TAP:
    os.makedirs(IX_TAP, exist_ok=True)
TPN = int(mesh.get_num_devices())

os.environ["GLM53_PF_DSA"] = "1"
os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_SPARSE_T"] = str(T)
S.set_attn(args)
if args.get("paged_pools_prefill_only"):  # a PREFILL-only unit of GLM53_DSA_PAGED_POOLS before the decode addresses the
    # paged store: let 12's allocate_kv_cache build it (mk_mla.DSA_PAGED_POOLS gates the decode; no decode runs here)
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as _MKM

    assert os.environ.get("GLM53_DSA_PAGED_POOLS") == "1", "paged_pools_prefill_only needs GLM53_DSA_PAGED_POOLS=1"
    _MKM.DSA_PAGED_POOLS = True
ent = S.build(mesh, ctx, LAYERS, b=1, max_seq=max(T, N), buckets=(CH,))
pf = ent["pf"]
assert pf.dsa, "TPPrefill built without GLM53_PF_DSA"
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
SHARD = os.environ.get("GLM53_DSA_SHARD", "0") == "1"
PT_ROW = None
if SHARD:  # the owner-aware page table (pmk_dsa_window_unit's form)
    ent["backend"].tpp = pf
    PT_ROW = A45.publish_page_table(ent["backend"], ent["kv"], ent["io"], pf, rows=1)
OUT_TAP, OUT_REF = args.get("out_tap") or None, args.get("out_ref") or None
REF_SHA = json.load(open(OUT_REF)) if OUT_REF and os.path.exists(OUT_REF) else None
MY_SHA = {}
c = pf.bucket_for(CH, sparse=True)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
chunks = list(range(0, N, CH))
rows_checked = [int(v) for v in args.get("rows", range(len(chunks)))]
X = pf.streams_to_device(torch.zeros(CH, 4, 4096), sp)  # the traces' persistent residual input
# ... and output: every trace copies its layer output into Y and frees its own (a capture's output allocated after an
# earlier capture would sit in that trace's memory: 'device buffer still alive before trace replay', 03:23Z)
Y = pf.streams_to_device(torch.zeros(CH, 4, 4096), sp)


def goldens(L: int):
    g = os.path.join(S.GOLDENS, "layer_io", f"L{L:02d}", str(N))
    x_in = torch.load(os.path.join(g, "streams_in.pt"), map_location="cpu", weights_only=True, mmap=True)
    x_out = torch.load(os.path.join(g, "streams_out.pt"), map_location="cpu", weights_only=True, mmap=True)
    tp_ = os.path.join(g, "topk.pt")
    topk = torch.load(tp_, map_location="cpu", weights_only=True, mmap=True) if os.path.exists(tp_) else None
    return x_in.reshape(N, 4, 4096), x_out.reshape(N, 4 * 4096), topk


def n_valid(k: int) -> int:
    return LAST_NV if LAST_NV and k == len(chunks) - 1 else CH


def load(x_in, k: int):
    cs, nv = chunks[k], n_valid(k)
    pf.write_inputs(c, torch.zeros(CH, dtype=torch.int64), slot=0, chunk_start=cs, n_valid=nv)
    xs = x_in[cs : cs + CH].float()
    if nv < CH and PAD == "zero":
        xs = xs.clone()
        xs[nv:] = 0.0
    host = pf.streams_to_device(xs, sp)
    ttnn.copy(host, X)
    ttnn.deallocate(host)


def zero_state(pl):
    if pl.kind != PF.SPARSE_DSA:
        for t_ in (pl.scratch.S, pl.scratch.C):
            ttnn.copy(ttnn.zeros_like(t_), t_)


def key_of(pl, k: int):
    cs = chunks[k]
    if pl.kind != PF.SPARSE_DSA:
        return (c, 0)
    if cs + CH > PF.DENSE_EXACT:
        return (c, T)
    return (c, PF.FRESH) if cs == 0 and (c, PF.FRESH) in pf.fresh_keys((c,)) else (c, 0)


def run(li: int, key):
    sctx = pf.sparse_ctx(c, key[1]) if key[1] > 0 else None
    pf._fresh_now = key[1] == PF.FRESH
    pf._sparse_now = sctx
    if sctx is not None:
        sctx.begin(ci)
    try:
        y = pf.layer(li, X, ci, sp, keep_input=True)
        ttnn.copy(y, Y)
        ttnn.deallocate(y)
        return Y
    finally:
        if sctx is not None:
            sctx.end()
        pf._sparse_now = None
        pf._fresh_now = False


# every key of every checked layer compiles before ANY capture (a program first compiled after a capture keeps a buffer
# alive in that capture's memory); the eager launches run on the key's first chunk of the layer's own golden input
plan = {}  # L -> {key: first chunk}
t0 = time.time()
for L in CHECK:
    li, x_in = LAYERS.index(L), goldens(L)[0]
    plan[L] = {}
    for k in range(len(chunks)):
        plan[L].setdefault(key_of(pf.layers[li], k), k)
    for key, k in plan[L].items():
        zero_state(pf.layers[li])
        load(x_in, k)
        run(li, key)
ttnn.synchronize_device(mesh)
gc.collect()
t_compile = time.time() - t0
traces = {}
if not C.MOCK and not EAGER:
    for L in CHECK:
        li, x_in = LAYERS.index(L), goldens(L)[0]
        for key, k in plan[L].items():
            load(x_in, k)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                run(li, key)
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traces[(L, key)] = tid
    ttnn.synchronize_device(mesh)
    gc.collect()

RESULT = []
for L in CHECK:
    li = LAYERS.index(L)
    pl = pf.layers[li]
    x_in, x_out, g_topk = goldens(L)
    results, overlaps = [], []
    ix_bits, ix_sel = [], []  # indexer A/B per chunk
    zero_state(pl)  # the compile / capture launches advanced a KDA layer's state
    for k in range(len(chunks)):
        cs, key = chunks[k], key_of(pl, k)
        load(x_in, k)
        if C.MOCK or EAGER:
            out = run(li, key)
        else:
            # per-chunk replay wall time (a blocking replay = the device time + the host's submit / wait): the per-chunk
            # curve of the indexer's causal bound (the timing rows below replay the LAST chunk only)
            t_rep = time.perf_counter()
            ttnn.execute_trace(mesh, traces[(L, key)], cq_id=0, blocking=True)
            replay_wall_ms = (time.perf_counter() - t_rep) * 1e3
            out = Y
        if k not in rows_checked:
            continue
        nv = n_valid(k)
        y = pf.streams_from_device(out, sp).reshape(CH, 4 * 4096)[:nv]
        ref = x_out[cs : cs + nv].float()
        bad = ~torch.isfinite(y)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        p = C.pcc(y, ref)
        sha = hashlib.sha1(y.contiguous().numpy().tobytes()).hexdigest()[:20]
        MY_SHA.setdefault(str(L), {})[str(k)] = sha
        row = dict(test=f"p3 long-context layer L{L} chunk {k} [{cs}, {cs + CH}) of {N}"
                        f"{f' (n_valid {nv})' if nv < CH else ''} vs CPU fp32 golden",
                   milestone="P3", layer=L, kind=pl.kind, chunk_start=cs, chunk=CH, window=key[1], n=N, tp=TPN,
                   mode="sp" if sp else "rep", sparse_attn=pf.sparse_attn, mla_form=pf.mla_form, pcc=round(p, 6),
                   rel_l2=round(float((y - ref).norm() / ref.norm()), 6), status="pass" if p >= 0.995 else "fail",
                   traced=not (C.MOCK or EAGER), replays_read=1, n_valid=nv, nonfinite=int(bad.sum()),
                   nonfinite_rows=int(bad.any(-1).sum()), tag=tag,
                   replay_wall_ms=None if (C.MOCK or EAGER) else round(replay_wall_ms, 2))  # fmt: skip
        row.update(out_sha1=sha, shard=SHARD)
        if REF_SHA is not None:
            row["out_equal_ref"] = REF_SHA.get(str(L), {}).get(str(k)) == sha
        if bad.any():
            row.update(status="fail", nonfinite_first_rows=torch.nonzero(bad.any(-1)).reshape(-1)[:8].tolist())
        # rows grouped by (position + 1) % KPOOL (= the DSA tail length of a sparse row): PCC and max |y| per phase
        ph = (torch.arange(cs, cs + nv) + 1) % PF.KPOOL
        row.update(pcc_by_phase=[round(C.pcc(y[ph == j], ref[ph == j]), 6) for j in range(PF.KPOOL)],
                   maxabs_by_phase=[float("%.3g" % float(y[ph == j].abs().max())) for j in range(PF.KPOOL)],
                   rows_over_1e3=int((y.abs().amax(-1) > 1e3).sum()))  # fmt: skip
        sctx = pf.sparse.get((c, key[1])) if key[1] > 0 else None
        if sctx is not None and sctx.ids is not None and g_topk is not None and not C.MOCK and not SHARD:
            ids = ttnn.to_torch(ttnn.get_device_tensors(sctx.ids)[0] if TPN > 1 else sctx.ids).reshape(CH, -1)
            ids = ids.to(torch.int64) & 0xFFFFFFFF
            gref = g_topk.reshape(N, -1)[cs : cs + CH].to(torch.int64)
            ov = []
            for r in sorted(set(range(0, nv, max(1, CH // 256))) | {nv - 1}):  # 256 rows per chunk + the last valid
                a = set(ids[r][ids[r] != 0xFFFFFFFF].tolist())
                b = set(gref[r][gref[r] >= 0].tolist())
                ov.append(len(a & b) / max(1, len(b)))
            row.update(sel_overlap_mean=round(sum(ov) / len(ov), 6), sel_overlap_min=round(min(ov), 6),
                       sel_rows=len(ov))  # fmt: skip
            overlaps += ov
            # ids integrity (sparse_sdpa binary-searches the first sentinel and trusts a sentinel SUFFIX): per valid row
            # the first-sentinel index vs expand_rows' count 4 min(n_closed, 512) + (p + 1) % 4, any non-sentinel after
            # it, any id past the row's position
            Wd = int(ids.shape[-1])
            is_s = ids[:nv] == 0xFFFFFFFF
            first = torch.where(is_s.any(-1), is_s.int().argmax(-1), torch.full((nv,), Wd))
            pos = torch.arange(cs, cs + nv)
            expect = PF.KPOOL * torch.clamp((pos + 1) // PF.KPOOL, max=PF.TOPK_POOLS) + (pos + 1) % PF.KPOOL
            after = (~is_s) & (torch.arange(Wd).reshape(1, -1) >= first.reshape(-1, 1))
            over = ((ids[:nv] * (~is_s)) > pos.reshape(-1, 1)).any(-1)
            row.update(ids_nv_mismatch_rows=int((first != expect).sum()), ids_suffix_bad_rows=int(after.any(-1).sum()),
                       ids_past_pos_rows=int(over.sum()),
                       ids_first_sentinel_by_phase=[sorted(set(first[((pos + 1) % PF.KPOOL) == j].tolist()))[:4]
                                                    for j in range(PF.KPOOL)])  # fmt: skip
        # the layer's DsaIndexer lives on its MLA piece (PrefillLayer.mla = MlaPrefill, MlaPrefill.idx); evaluated before the
        # MOCK guard so a mock with ix_tap set exercises the accessor
        ix = getattr(getattr(pl, "mla", None), "idx", None)
        if (IX_TAP or IX_REF) and sctx is not None and ix is not None and not C.MOCK:
            taps = ix.taps or {}
            planes = []
            for t_ in taps.values():  # one plane per (rows, pools) key; raw bf16 bits per chip
                parts = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
                planes.append([ttnn.to_torch(p_).reshape(p_.shape[-2], p_.shape[-1]).view(torch.int16) for p_ in parts])
            ids_all = None
            if sctx.ids is not None:
                ids_all = ttnn.to_torch(ttnn.get_device_tensors(sctx.ids)[0] if TPN > 1 else sctx.ids).reshape(CH, -1)
            dump = dict(chunk=k, chunk_start=cs, window=key[1], layer=L, tp=TPN, planes=planes, ids=ids_all,
                        cfg=os.environ.get("GLM53_PF_IX_CFG", ""), bound=os.environ.get("GLM53_PF_IX_BOUND", ""))  # fmt: skip
            if IX_TAP:
                torch.save(dump, os.path.join(IX_TAP, f"L{L}_c{k}_d.pt"))
            if IX_REF:
                refp = os.path.join(IX_REF, f"L{L}_c{k}_d.pt")
                if os.path.exists(refp):
                    ref = torch.load(refp, map_location="cpu", weights_only=False)
                    bitwise, maxabs = True, 0.0
                    for pa, pb in zip(planes, ref["planes"]):
                        for a_, b_ in zip(pa, pb):
                            if a_.shape != b_.shape or not torch.equal(a_, b_):
                                bitwise = False
                                if a_.shape == b_.shape:
                                    fa = (a_.to(torch.int32) << 16).view(torch.float32)
                                    fb = (b_.to(torch.int32) << 16).view(torch.float32)
                                    fin = torch.isfinite(fa) & torch.isfinite(fb)
                                    maxabs = max(maxabs, float((fa[fin] - fb[fin]).abs().max()) if fin.any() else 0.0)
                                else:
                                    maxabs = float("inf")
                    eq_rows, n_rows = 0, 0
                    if ids_all is not None and ref.get("ids") is not None:
                        ia = ids_all.to(torch.int64) & 0xFFFFFFFF
                        ib = ref["ids"].to(torch.int64) & 0xFFFFFFFF
                        for r in range(nv):
                            sa = set(ia[r][ia[r] != 0xFFFFFFFF].tolist())
                            sb = set(ib[r][ib[r] != 0xFFFFFFFF].tolist())
                            eq_rows += int(sa == sb)
                            n_rows += 1
                    row.update(ix_score_bitwise=bitwise, ix_score_maxabs=maxabs, ix_planes=sum(len(p_) for p_ in planes),
                               ix_sel_equal_frac=round(eq_rows / n_rows, 6) if n_rows else None, ix_ref=refp)  # fmt: skip
                    ix_bits.append(bitwise)
                    if n_rows:
                        ix_sel.append(eq_rows / n_rows)
                else:
                    row.update(ix_ref_missing=refp)
        S.log(row)
        if rec:
            C.record("pcc", row)
        results.append(row)
    state = {}
    if (
        pl.kind != PF.SPARSE_DSA and not C.MOCK and not LAST_NV
    ):  # the KDA state after the whole prompt vs the golden (chips' heads)
        Sg = S.golden_kda_state(L, N)["recurrent"].float()
        parts = ttnn.get_device_tensors(pl.scratch.S) if TPN > 1 else [pl.scratch.S]
        Sd = torch.cat([ttnn.to_torch(t_).float() for t_ in parts], dim=1)
        state = dict(state_pcc=round(C.pcc(Sd, Sg[:, : Sd.shape[1]]), 6), state_heads=int(Sd.shape[1]))
    # timing: each key's trace on its LAST chunk's inputs, after the PCC and state reads
    for key in plan[L] if not C.MOCK and not EAGER and args.get("replays") != 0 else []:
        k_last = max(k for k in range(len(chunks)) if key_of(pl, k) == key)
        load(x_in, k_last)
        st = S.replay_stats(mesh, traces[(L, key)], replays=args.get("replays"))
        trow = dict(test=f"p3 long-context layer L{L} {pl.kind} key {key} replay (chunk {k_last} of {N}, window "
                         f"{key[1]})", milestone="P3", layer=L, kind=pl.kind, n=N, chunk=CH, chunk_start=chunks[k_last],
                    window=key[1], tp=TPN, sparse_attn=pf.sparse_attn, traced=True, tag=tag, **st)  # fmt: skip
        S.log(trow)
        if rec:
            C.record("perf", trow)
        results.append(trow)
    pcc = [r["pcc"] for r in results if "pcc" in r]
    eq_ref = [r.get("out_equal_ref") for r in results if "out_equal_ref" in r]
    nonfinite = sum(r.get("nonfinite", 0) for r in results)
    ok = bool(pcc) and min(pcc) >= 0.995 and state.get("state_pcc", 1.0) >= 0.995 and nonfinite == 0
    summary = dict(test=f"p3 long-context layer L{L} over {N} tokens in {len(chunks)} chunks of {CH} (summary)",
                   milestone="P3", layer=L, kind=pl.kind, n=N, chunk=CH, tp=TPN, sparse_attn=pf.sparse_attn,
                   compile_s=round(t_compile, 1), **state, pcc_min=min(pcc) if pcc else None,
                   pcc_mean=round(sum(pcc) / len(pcc), 6) if pcc else None,
                   sel_overlap_mean=round(sum(overlaps) / len(overlaps), 6) if overlaps else None,
                   sel_overlap_min=round(min(overlaps), 6) if overlaps else None,
                   nonfinite=nonfinite, last_valid=LAST_NV, pad=PAD if LAST_NV else None, shard=SHARD,
                   pcc_by_chunk=[r["pcc"] for r in results if "pcc" in r],
                   out_equal_ref_all=(all(eq_ref) if eq_ref else None), out_equal_ref_chunks=eq_ref or None,
                   status="pass" if ok else "fail", traced=not (C.MOCK or EAGER), tag=tag)  # fmt: skip
    if IX_REF:  # the exact-lever bars: every chunk's score plane bitwise + every row's selected set equal
        ix_ok = bool(ix_bits) and all(ix_bits) and bool(ix_sel) and min(ix_sel) == 1.0
        summary.update(ix_score_bitwise_all=bool(ix_bits) and all(ix_bits), ix_chunks_compared=len(ix_bits),
                       ix_sel_equal_min=min(ix_sel) if ix_sel else None, ix_cfg=os.environ.get("GLM53_PF_IX_CFG", ""),
                       ix_bound=os.environ.get("GLM53_PF_IX_BOUND", ""), ix_ref_dir=IX_REF,
                       ix_status="pass" if ix_ok else "fail")  # fmt: skip
        if not ix_ok:
            summary["status"] = "fail"
    S.log(summary)
    if rec:
        C.record("pcc", summary)
    RESULT += results + [summary]

if float(args.get("soak_s", 0)) > 0:  # the watched unit's closing soak (unwatched server: same program)
    RESULT.append(dict(test="p3 long-context closing soak of every trace", tag=tag,
                       **S.soak(mesh, list(traces.values()), float(args["soak_s"]))))  # fmt: skip
    S.log(RESULT[-1])
if OUT_TAP and not C.MOCK:
    json.dump(MY_SHA, open(OUT_TAP, "w"))
for tid in traces.values():
    ttnn.release_trace(mesh, tid)
ttnn.deallocate(X)
ttnn.deallocate(Y)
