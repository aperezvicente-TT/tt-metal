# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P3 -- the MLA prefill SDPA variants on one sparse_dsa layer's REAL q_lat /
latents (the P3 reference attention input of L03, chunk 0 [0, n)), device vs device: every variant traced (compile
launch, capture of k calls, replays; the checked output is the last captured call's after the replays), checked vs a
host fp32 SDPA of chip 0's heads on the same bf16 q_lat / latents (pcc, max abs, rel l2).

Variant "path,q,k,fidelity[,exp_approx]":
  chunked     the P2 path: chunked SDPA over the paged latent cache, chunk start a device tensor (any chunk)
  causal      contiguous causal SDPA over the chunk's own latents (chunk start 0 only)
  mla         flash_mla_prefill (K = V = the latents, head_dim_v 512; chunk start 0 only)
  abs         the P2 sublayer core: q W_uk, the chunked SDPA, o_lat W_uv (output per head 256: vs o_lat_ref W_uv)
  mha         non-absorbed: K / V = latents [W_uk^T | W_uv] per head, causal SDPA at head dim 256 (chunk start 0)
  mhamask     the P3 sparse shape in the MHA form (K / V of the 2 n keys, the all-visible additive mask)
  masked      the P3 sparse shape: 2 n keys (the latents twice) with an all-visible additive mask, is_causal False
              (reference: the non-causal SDPA over the latents -- duplicated keys keep the softmax weights' ratios)
Then the whole MLA attention (pl.mla, dense chunk 0) traced with GLM53_PF_SDPA = each `e2e` spec ("" = P2; "mha:<spec>"
= the MHA form with that SDPA spec) vs the P3 fp32 reference output (o_proj partials summed over the chips).

ARGS: layers ([2, 3]), layer (3), n (2048), variants (list), e2e ([""]), heads_ref (4), k (4), reps (8), tag,
record (True)."""
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
LAYERS = [int(v) for v in args.get("layers", [2, 3])]
L = int(args.get("layer", 3))
N = int(args.get("n", 2048))
VARIANTS = list(args.get("variants", ["chunked,64,64,HiFi4"]))
E2E = list(args.get("e2e", [""]))
HREF = int(args.get("heads_ref", 4))
K = int(args.get("k", 4))
REPS = int(args.get("reps", 8))
tag = args.get("tag", "p3-sdpa")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
REF = os.environ.get("GLM53_P3_REF_DIR", "/opt/glm53/work-p3/ref")
DRAM = PF.DRAM

os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_SDPA"] = ""
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf, kv = ent["pf"], ent["kv"]
c = pf.bucket_for(N)
assert c == N, f"n {N} must be a bucket (got {c})"
ci = pf.chunk_inputs(c)
li = LAYERS.index(L)
pl = pf.layers[li]
mla = pl.mla
cache = kv.extra["latent"][li]
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
ref = torch.load(os.path.join(REF, f"L{L:02d}_4096.pt"), map_location="cpu", weights_only=False)
hb = S.to_mesh_rows(mesh, ref["x"][:N].float(), False, dtype=ttnn.bfloat16)

# the MLA's own projections (MlaPrefill.__call__ up to the SDPA): q_lat [1, H_r, n, 512], the latents [1, 1, n, 512]
qk = PF.pf_mm(mla.attn.ds_qa_kva, hb)
q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, N, mla.q_lora))
cc = ttnn.slice(qk, (0, 0, 0, mla.q_lora), (1, 1, N, mla.q_lora + mla.kv_lora))
q_r = ttnn.rms_norm(q_r, weight=mla.attn.qa_norm, epsilon=mla.eps, compute_kernel_config=mla.ckc_fp32)
lat = ttnn.rms_norm(cc, weight=mla.attn.kva_norm, epsilon=mla.eps, compute_kernel_config=mla.ckc_fp32)
q = PF.pf_mm(mla.shard.ds_qb, q_r)
q, _, _ = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=mla.h, num_kv_heads=0, transpose_k_heads=False,
                                                memory_config=DRAM)  # fmt: skip
q_lat = ttnn.matmul(q, mla.w_uk, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=DRAM)
ttnn.experimental.paged_fill_cache(cache, ttnn.typecast(lat, cache.dtype), ci.chunk_table, batch_idx=0)
lat2 = ttnn.concat([lat, lat], dim=2)
mask0 = ttnn.zeros((1, 1, N, 2 * N), dtype=PF.BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
# the non-absorbed (MHA) form: K_h = latents W_uk_h^T, V_h = latents W_uv_h as two [512, H_r 256] matrices
H_ = int(mla.w_uk.shape[1])
w_k = ttnn.reshape(ttnn.permute(ttnn.transpose(mla.w_uk, -2, -1), (0, 2, 1, 3)), (1, 1, 512, H_ * 256))
w_v = ttnn.reshape(ttnn.permute(mla.w_uv, (0, 2, 1, 3)), (1, 1, 512, H_ * 256))
ttnn.synchronize_device(mesh)


def chip0(t_):
    return ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()


H = int(q_lat.shape[1])
heads = sorted({int(round(i * (H - 1) / max(1, HREF - 1))) for i in range(HREF)})
qh = chip0(q_lat)[0, heads]  # [h, n, 512]
lh = chip0(lat)[0, 0]  # [n, 512]
scores = torch.einsum("hqd,kd->hqk", qh, lh) * mla.scale
causal = torch.ones(N, N, dtype=torch.bool).tril()
ref_out = {
    "causal": torch.einsum("hqk,kd->hqd", torch.softmax(scores.masked_fill(~causal, float("-inf")), -1), lh),
    "full": torch.einsum("hqk,kd->hqd", torch.softmax(scores, -1), lh),
}
del scores
w_uv_h = chip0(mla.w_uv)[0, heads]  # [h, 512, 256]
ref_out["heads"] = torch.einsum("hqd,hde->hqe", ref_out["causal"], w_uv_h)  # o_h = o_lat_h W_uv_h [h, n, 256]
ref_out["heads_full"] = torch.einsum("hqd,hde->hqe", ref_out["full"], w_uv_h)  # non-causal (the masked shape's reference)
S.log(f"L{L} tp{TPN}: q_lat {list(q_lat.shape)}, ref heads {heads}")


def make(spec: str):
    path, qc, kc, fid, *rest = spec.split(",")
    prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
                                  q_chunk_size=int(qc), k_chunk_size=int(kc), exp_approx_mode=rest[:1] == ["1"])  # fmt: skip
    ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                 math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)  # fmt: skip
    kw = dict(scale=mla.scale, program_config=prog, compute_kernel_config=ckc, memory_config=DRAM)
    if path == "chunked":
        return "causal", lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            q_lat, cache, cache, ci.user_table, chunk_start_idx_tensor=ci.chunk_start, **kw)  # fmt: skip
    if path == "causal":
        return "causal", lambda: ttnn.transformer.scaled_dot_product_attention(q_lat, lat, lat, is_causal=True, **kw)
    if path == "mla":
        return "causal", lambda: ttnn.transformer.flash_mla_prefill(q_lat, lat, 512, is_causal=True, **kw)
    if path == "abs":  # the P2 sublayer core: q W_uk -> chunked SDPA over the paged latents -> o_lat W_uv
        def absorbed():
            ql = ttnn.matmul(q, mla.w_uk, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=DRAM)
            o_lat = ttnn.transformer.chunked_scaled_dot_product_attention(
                ql, cache, cache, ci.user_table, chunk_start_idx_tensor=ci.chunk_start, **kw)  # fmt: skip
            ttnn.deallocate(ql)
            o = ttnn.matmul(o_lat, mla.w_uv, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=DRAM)
            ttnn.deallocate(o_lat)
            return o

        return "heads", absorbed
    if path == "mha":  # non-absorbed: K / V per head from the latents, causal SDPA at head dim 256
        def mha():
            hs = []
            for w_ in (w_k, w_v):
                f = ttnn.matmul(lat, w_, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=DRAM)
                t_, _, _ = ttnn.experimental.nlp_create_qkv_heads(f, num_heads=H_, num_kv_heads=0,
                                                                  transpose_k_heads=False, memory_config=DRAM)  # fmt: skip
                ttnn.deallocate(f)
                hs.append(t_)
            o = ttnn.transformer.scaled_dot_product_attention(q, hs[0], hs[1], is_causal=True, **kw)
            for t_ in hs:
                ttnn.deallocate(t_)
            return o

        return "heads", mha
    if path == "mhamask":  # the DSA sparse chunk's shape in the MHA form: K / V over 2 n keys, additive mask
        def mhamask():
            hs = []
            for w_ in (w_k, w_v):
                f = ttnn.matmul(lat2, w_, compute_kernel_config=mla.ckc_head, dtype=PF.BF16, memory_config=DRAM)
                t_, _, _ = ttnn.experimental.nlp_create_qkv_heads(f, num_heads=H_, num_kv_heads=0,
                                                                  transpose_k_heads=False, memory_config=DRAM)  # fmt: skip
                ttnn.deallocate(f)
                hs.append(t_)
            o = ttnn.transformer.scaled_dot_product_attention(q, hs[0], hs[1], attn_mask=mask0, is_causal=False, **kw)
            for t_ in hs:
                ttnn.deallocate(t_)
            return o

        return "heads_full", mhamask
    if path == "masked":
        return "full", lambda: ttnn.transformer.scaled_dot_product_attention(q_lat, lat2, lat2, attn_mask=mask0,
                                                                             is_causal=False, **kw)  # fmt: skip
    raise ValueError(path)


def traced(fn, read):
    """compile launch, capture of K calls, REPS replays -> (read(last captured output), compile s, us per call list)"""
    t0 = time.time()
    out = fn()
    ttnn.synchronize_device(mesh)
    t_compile = time.time() - t0
    if C.MOCK:
        v = read(out)
        ttnn.deallocate(out)
        return v, t_compile, []
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for i in range(K):
            out = fn()
            if i < K - 1:
                ttnn.deallocate(out)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    try:  # >= 2,000 steady-state replays (program 03:20Z device duty); per call = per replay / K
        st = S.replay_stats(mesh, tid, replays=args.get("replays"))
        times = [st["median_us"] / K]
        v = read(out)
    finally:
        ttnn.deallocate(out)
        ttnn.release_trace(mesh, tid)
    return v, t_compile, times


results = []
flops = {"causal": 2 * 2 * H * N * N * 512 / 2, "full": 2 * 2 * H * N * 2 * N * 512, "heads": None, "heads_full": None}
for spec in VARIANTS:
    kind, fn = make(spec)
    try:
        o, t_compile, times = traced(fn, lambda t_: chip0(t_)[0, heads])
    except Exception as e:  # noqa: BLE001  (host-side program errors: L1 / config limits; the device is untouched)
        S.log(f"{spec}: not run: {str(e)[:200]}")
        results.append(dict(spec=spec, error=str(e)[:200]))
        continue
    r = ref_out[kind]
    st = S.timing(times) if times else {}
    row = dict(test=f"p3 mla prefill sdpa {spec} L{L} n={N} (real q_lat / latents, chip 0 heads {heads} vs host fp32)",
               milestone="P3", status="info", traced=bool(times), layer=L, n=N, tp=TPN, heads_local=H, spec=spec,
               pcc=round(C.pcc(o, r), 6), max_abs=round(float((o - r).abs().max()), 5),
               rel_l2=round(float((o - r).norm() / r.norm()), 6), compile_s=round(t_compile, 1), tag=tag, **st)  # fmt: skip
    if times:
        row["tflops"] = round(flops[kind] / (st["median_us"] * 1e-6) / 1e12, 1) if flops[kind] else None
    S.log(row)
    if rec and not C.MOCK:
        C.record("perf", row)
    results.append(row)
    gc.collect()

o_ref = ref["out"][:N].float()
for spec in E2E:
    fresh = spec.startswith("mha:")  # the MHA form of a chunk at position 0 (GLM53_PF_MLA_FORM=mha), its SDPA spec
    if fresh:
        mla.enable_mha(spec[4:])
    else:
        os.environ["GLM53_PF_SDPA"] = spec
        mla.sdpa_prog, mla.sdpa_ckc = PF.sdpa_config(mesh, mla.attn)

    def summed(t_):
        parts = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
        return sum(ttnn.to_torch(p).float() for p in parts).reshape(-1, 4096)

    a, t_compile, times = traced(lambda: mla(hb, cache, ci, fresh=fresh), summed)
    p = C.pcc(a, o_ref)
    row = dict(test=f"p3 mla attention L{L} dense chunk 0 n={N} "
                    f"{'MHA form SDPA ' + spec[4:] if fresh else 'GLM53_PF_SDPA=' + repr(spec)} vs CPU fp32 reference",
               milestone="P3", status="pass" if p >= 0.995 else "fail", traced=bool(times), layer=L, kind=pl.kind,
               n=N, tp=TPN, sdpa=spec or "P2 (64,64,HiFi4)", pcc=round(p, 6),
               rel_l2=round(float((a - o_ref).norm() / o_ref.norm()), 6), compile_s=round(t_compile, 1), tag=tag,
               **(S.timing(times) if times else {}))  # fmt: skip
    S.log(row)
    if rec and not C.MOCK:
        C.record("pcc", row)
    results.append(row)
os.environ["GLM53_PF_SDPA"] = ""
mla.sdpa_prog, mla.sdpa_ckc = PF.sdpa_config(mesh, mla.attn)
for t_ in (qk, q_r, cc, lat, q, q_lat, lat2, mask0, w_k, w_v):
    if t_.is_allocated():
        ttnn.deallocate(t_)
RESULT = results
