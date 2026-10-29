# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P3 — DSA sparse prefill of one sparse_dsa layer (tt/prefill_tp.py, GLM53_PF_DSA=1)
over the 4,096-token golden prompt as two 2,048-token chunks of slot 0: chunk 0 (dense trace: every query selects all
visible keys) then chunk 1 at position 2,048 (sparse trace, key window t: indexer top-512 pools + tail, flash SDPA with
the selection mask over the slot's latents). Checked per chunk:
  attention: the attention input x of the P3 CPU reference (work-p3/ref/L<ii>_4096.pt: the vendored MLAAttention +
      Indexer in fp32) -> pl.mla on the mesh -> o_proj partials summed over the chips vs the reference `out`;
  layer: golden streams_in -> pf.layer -> vs golden streams_out (Stage 00c layer_io/L<ii>/4096).
Every checked output is a traced replay's (the eager launch before the capture is the compile); the attention replays
are timed (reps). MoE: moe (default "sorted": the watched fused_experts path).

ARGS: layers ([2, 3]), check ([3]), n (4096), chunk (2048), t (4096), reps (3), the P3 attention knobs
(s16common.set_attn: mla_form "mha" = the non-absorbed form for chunk 0 and the sparse chunk), tag, record (True)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [2, 3])]  # the stack needs a non-DSA layer
CHECK = [int(v) for v in args.get("check", [3])]
N = int(args.get("n", 4096))
CH = int(args.get("chunk", 2048))
T = int(args.get("t", 4096))
REPS = int(args.get("reps", 3))
tag = args.get("tag", "p3-dsa")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
REF = os.environ.get("GLM53_P3_REF_DIR", "/opt/glm53/work-p3/ref")

os.environ["GLM53_PF_DSA"] = "1"
os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=1, max_seq=max(T, N), buckets=(CH,))
pf, kv = ent["pf"], ent["kv"]
assert pf.dsa, "TPPrefill built without GLM53_PF_DSA"
pf.moe_impl = args.get("moe", "sorted")
S.set_moe_v2(pf, bool(args.get("moe_v2", False)))
c = pf.bucket_for(CH, sparse=True)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
results = []


def mesh_rows(x, shard, dtype):
    return S.to_mesh_rows(mesh, x, shard and TPN > 1, dtype=dtype)


def summed(t_):
    parts = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
    return sum(ttnn.to_torch(p).float() for p in parts).reshape(-1, 4096)


def traced(fn, read, reps=0):
    """compile launch, capture, one replay read, then the steady-state soak (s16common.replay_stats: >= 2,000 replays,
    program 03:20Z device duty) -> (read(output of the first replay), compile s, timing fields, traced)"""
    t0 = time.time()
    out = fn()
    ttnn.synchronize_device(mesh)
    t_compile = time.time() - t0
    if C.MOCK:
        v = read(out)
        ttnn.deallocate(out)
        return v, t_compile, {}, False
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        out = fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    v = read(out)
    times = S.replay_stats(mesh, tid, replays=args.get("replays"))
    ttnn.deallocate(out)
    ttnn.release_trace(mesh, tid)
    return v, t_compile, times, True


for L in CHECK:
    li = LAYERS.index(L)
    pl = pf.layers[li]
    ref = torch.load(os.path.join(REF, f"L{L:02d}_4096.pt"), map_location="cpu", weights_only=False)
    x_in, x_out = S.golden_layer(L, N)
    x_in = x_in.reshape(N, 4, 4096).float()
    x_out = x_out.reshape(N, 4 * 4096).float()
    cache = kv.extra["latent"][li]
    for k, cs in enumerate(range(0, N, CH)):
        pf.write_inputs(c, torch.zeros(CH, dtype=torch.int64), slot=0, chunk_start=cs, n_valid=CH)
        sctx = pf.sparse_ctx(c, T) if cs + CH > PF.DENSE_EXACT else None
        fresh = sctx is None and cs == 0 and (c, PF.FRESH) in pf.fresh_keys((c,))  # MHA form at position 0
        # attention vs the P3 reference (attention input x -> o_proj output)
        xr = ref["x"][cs : cs + CH].float()
        hb = mesh_rows(xr, False, ttnn.bfloat16)
        hl = mesh_rows(xr, True, ttnn.bfloat16) if sp else hb

        def attn():
            if sctx is not None:
                sctx.begin(ci)
            a = pl.mla(hb, cache, ci, hl, sctx, fresh=fresh)
            if sctx is not None:
                sctx.end()
            return a

        # every launch rewrites the chunk's latent pages + pooled keys (same values)
        a_host, t_compile, times, was_traced = traced(attn, summed, REPS)
        o_ref = ref["out"][cs : cs + CH].float()
        p_attn = C.pcc(a_host, o_ref)
        row = dict(test=f"p3 dsa prefill attention L{L} chunk {k} [{cs}, {cs + CH}) {'sparse' if sctx else 'dense'} "
                        f"vs CPU fp32 reference", milestone="P3", layer=L, kind=pl.kind, chunk_start=cs, chunk=CH,
                   window=T if sctx else 0, tp=TPN, mode="sp" if sp else "rep", pcc=round(p_attn, 6),
                   mla_form=pf.mla_form, mha_sdpa=os.environ.get("GLM53_PF_MHA_SDPA", "") if pf.mla_form == "mha" else "",
                   rel_l2=round(float((a_host - o_ref).norm() / o_ref.norm()), 6),
                   status="pass" if p_attn >= 0.995 else "fail", traced=was_traced,
                   compile_s=round(t_compile, 1), tag=tag, **times)  # fmt: skip
        S.log(row)
        if rec:
            C.record("pcc", row)
        results.append(row)
        for t_ in {id(hb): hb, id(hl): hl}.values():
            ttnn.deallocate(t_)
        # the whole layer vs the golden streams (re-runs the attention: same pages / pooled keys)
        X = pf.streams_to_device(x_in[cs : cs + CH], sp)

        def layer():
            if sctx is not None:
                pf._sparse_now = sctx
                sctx.begin(ci)
            pf._fresh_now = fresh
            y = pf.layer(li, X, ci, sp, keep_input=True)
            pf._fresh_now = False
            if sctx is not None:
                sctx.end()
                pf._sparse_now = None
            return y

        yh, _, ltimes, was_traced = traced(layer, lambda y: pf.streams_from_device(y, sp).reshape(CH, -1))
        ttnn.deallocate(X)
        p_layer = C.pcc(yh, x_out[cs : cs + CH])
        row = dict(test=f"p3 dsa prefill layer L{L} chunk {k} [{cs}, {cs + CH}) {'sparse' if sctx else 'dense'} vs "
                        f"CPU fp32 golden", milestone="P3", layer=L, kind=pl.kind, chunk_start=cs, chunk=CH,
                   window=T if sctx else 0, tp=TPN, pcc=round(p_layer, 6),
                   rel_l2=round(float((yh - x_out[cs : cs + CH]).norm() / x_out[cs : cs + CH].norm()), 6),
                   status="pass" if p_layer >= 0.995 else "fail", traced=was_traced, tag=tag,
                   moe=pf.moe_impl, **ltimes)  # fmt: skip
        S.log(row)
        if rec:
            C.record("pcc", row)
        results.append(row)
    # the decode indexer's state after the prompt (tests/s7b/golden_mla.dsa_state at p = N): the slot's pooled keys
    # (rows < N // 4) and ring rows 0..3, vs the P3 reference's k / g (bf16 stored, the reference pooling in float64)
    ix = pl.mla.idx
    if ix is not None:
        apes = torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(a)[0] if TPN > 1 else a).reshape(-1).double() for a in ix.ape]
        )  # fmt: skip  # [4, 128]
        k_ref = ref["k_idx"][:N].double()
        g_ref = ref["gate"][:N].double()
        nq = N // 4
        e = torch.softmax(g_ref[: 4 * nq].reshape(nq, 4, 128) + apes, dim=-2)
        pools_ref = (e * k_ref[: 4 * nq].reshape(nq, 4, 128)).sum(-2).bfloat16().double()
        store = ttnn.get_device_tensors(ix.store)[0] if TPN > 1 else ix.store
        pools_dev = ttnn.to_torch(store).reshape(-1, 128)[: ix.pages_per_slot * 32][:nq].double()  # slot 0
        ring = ttnn.get_device_tensors(ix.ring)[0] if TPN > 1 else ix.ring
        ring_dev = ttnn.to_torch(ring).reshape(-1, 32, 256)[0][:4].double()
        base = 4 * nq
        ring_ref = (
            torch.stack(
                [
                    torch.cat([k_ref[pos], g_ref[pos]])
                    for pos in [(base + j if base + j < N else base - 4 + j) for j in range(4)]
                ]
            )
            .bfloat16()
            .double()
        )
        row = dict(
            test=f"p3 dsa decode state after the {N}-token prefill L{L} (pooled keys + ring vs "
            f"golden_mla.dsa_state semantics)",
            milestone="P3",
            layer=L,
            tp=TPN,
            pools_pcc=round(C.pcc(pools_dev, pools_ref), 6),
            pools_max_abs=float((pools_dev - pools_ref).abs().max()),
            ring_pcc=round(C.pcc(ring_dev, ring_ref), 6),
            ring_max_abs=float((ring_dev - ring_ref).abs().max()),
            traced=not C.MOCK,
            tag=tag,
        )  # fmt: skip  # the state the last chunk's traced replay wrote
        row["status"] = "pass" if (C.MOCK or (row["pools_pcc"] >= 0.999 and row["ring_pcc"] >= 0.999)) else "fail"
        if C.MOCK:
            row["status"] = "info"
        S.log(row)
        if rec and not C.MOCK:
            C.record("pcc", row)
        results.append(row)
RESULT = results
