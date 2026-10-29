# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b CPU fp32 oracle of the sparse_dsa attention half in the dense-exact regime (kv_len <= 2051), the
arithmetic of modeling_glm5_next (doc/architecture.md §4, tt/mla.py docstring) in float64 on the layer's cached
weights (the values the device reads: bfloat8_b projections dequantised, bf16 W_uk / W_uv, fp32 norms):

    mHC pre (tt/mhc_pre.py reference_step) -> hn
    qk = hn W_qa_kva; q_r = RMSNorm_1536(qk[:1536]) w_qa; latent = RMSNorm_512(qk[1536:]) w_kva
    q_h = (q_r W_qb)[h]; q_lat_h = q_h W_uk[h]; s = q_lat_h . K^T / 16 over rows 0..p (row p = latent); o_lat_h =
    softmax(s) K; o = concat_h(o_lat_h W_uv[h]); a = o W_o
    mHC post (reference hc_expand) -> s1

Used once per program at acceptance (the device-vs-device chain checks are the inner loop)."""

from __future__ import annotations

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.reference.modules import hc_expand
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M


def host_weights(attn) -> dict:
    """the MLA weights of a decoder's OptimizedMLA as float64 host tensors"""
    h = lambda t: ttnn.to_torch(t).double()
    return dict(
        qa_kva=h(attn.ds_qa_kva.weight).reshape(-1, attn.ds_qa_kva.n_pad)[:, : attn.ds_qa_kva.n],
        qa_norm=h(attn.qa_norm).reshape(-1)[: attn.q_lora],
        kva_norm=h(attn.kva_norm).reshape(-1)[: attn.kv_lora],
        qb=h(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n],
        w_uk=h(attn.w_uk).reshape(attn.h, attn.dqk, attn.kv_lora),
        w_uv=h(attn.w_uv).reshape(attn.h, attn.kv_lora, attn.dv),
        o=h(attn.ds_o.weight).reshape(attn.h * attn.dv, -1)[:, : attn.ds_o.n],
    )


def rms(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def mhc_pre_ref(x, a_args: dict) -> dict:
    """x [T, 16384] -> reference_step's dict (hn, post, comb, ...) in float64"""
    return M.reference_step(x.double(), a_args["fn_t"].double(), a_args["scale"].double(), a_args["bias"].double(),
                            a_args["norm_w"].double(), hc_eps=a_args["hc_eps"], x_eps=a_args["x_eps"],
                            h_eps=a_args["h_eps"], iters=a_args["iters"], matmul_tf32=False)  # fmt: skip


def latents(hn, W: dict, eps: float, q_lora: int):
    qk = hn @ W["qa_kva"]
    return rms(qk[:, q_lora:], W["kva_norm"], eps)


def attention_half(x, a_args: dict, W: dict, past: list, *, eps: float, scale: float, heads: int, q_lora: int) -> dict:
    """x [T, 16384] (the layer input rows), past[t] = [p_t, 512] float64 latent rows 0..p_t - 1 of token t's
    sequence -> dict(hn, latent, q_lat [T, H, 512], o_lat, a [T, 4096], s1 [T, 4, 4096]) float64"""
    ref = mhc_pre_ref(x, a_args)
    hn = ref["hn"]
    qk = hn @ W["qa_kva"]
    q_r = rms(qk[:, :q_lora], W["qa_norm"], eps)
    lat = rms(qk[:, q_lora:], W["kva_norm"], eps)
    q = (q_r @ W["qb"]).reshape(x.shape[0], heads, -1)
    q_lat = torch.einsum("thd,hdk->thk", q, W["w_uk"])
    o_lat = torch.empty_like(q_lat)
    for t in range(x.shape[0]):
        kv = torch.cat([past[t], lat[t : t + 1]], dim=0)  # rows 0..p
        s = (q_lat[t] @ kv.T) * scale
        o_lat[t] = torch.softmax(s, dim=-1) @ kv
    o = torch.einsum("thk,hkd->thd", o_lat, W["w_uv"]).reshape(x.shape[0], -1)
    a = o @ W["o"]
    streams = x.double().reshape(x.shape[0], 4, -1)
    s1 = hc_expand(a, streams, ref["post"], ref["comb"])
    return dict(hn=hn, latent=lat, q_lat=q_lat, o_lat=o_lat, a=a, s1=s1)


# ------------------------------------------------------------------------------------------------- DSA (sparse_dsa)
def bf(x):
    """bf16 round trip in float64 (the device's stored ring / pooled-key values)"""
    return x.to(torch.bfloat16).double()


def indexer_rows(hn, ix: dict, eps: float = 1.0e-6) -> dict:
    """hn [P, 4096] -> k = LayerNorm_128(hn wk) and g = hn kpool_compress_gate as stored (bf16), the head weights w = hn
    weights_proj 32^-0.5 128^-0.5 (float64); ix = DsaIndexer.host"""
    k = hn @ ix["wk"].double()
    mu = k.mean(-1, keepdim=True)
    var = ((k - mu) ** 2).mean(-1, keepdim=True)
    kn = (k - mu) / torch.sqrt(var + eps) * ix["kn_w"].double() + ix["kn_b"].double()
    return dict(k=bf(kn), g=bf(hn @ ix["gate"].double()), w=(hn @ ix["wp"].double()) * (32**-0.5 * 128**-0.5))


def pool(k4, g4, ape):
    """[n, 4, 128] ring rows (k, g) -> the pooled keys [n, 128] (softmax over the 4 tokens per channel; bf16 stored)"""
    e = torch.softmax(g4 + ape.double(), dim=-2)
    return bf((e * k4).sum(-2))


def dsa_state(p: int, k, g, ape, tp: int):
    """a slot's DSA state before the decode step at position p (positions 0..p-1 processed): pools [tp, 128] (rows <
    p // 4 complete), ring [32, 256] (row j = [k | g] of position 4 (p // 4) + j if < p, else of the previous pool's)"""
    pools = torch.zeros(tp, 128, dtype=torch.float64)
    nq = p // 4
    if nq:
        pools[:nq] = pool(k[: 4 * nq].reshape(nq, 4, 128), g[: 4 * nq].reshape(nq, 4, 128), ape)
    ring = torch.zeros(32, 256, dtype=torch.float64)
    base = 4 * nq
    for j in range(4):
        pos = base + j if base + j < p else base - 4 + j
        if pos >= 0:
            ring[j] = torch.cat([k[pos], g[pos]])
    return pools, ring


def dsa_step(x, p: int, a_args: dict, W: dict, ix: dict, lat_past, pools, ring, *, eps: float, scale: float,
             heads: int, q_lora: int, dense_max: int = 2051, topk: int = 512) -> dict:  # fmt: skip
    """one decode token at position p with the DSA indexer (float64; stored state bf16): x [1, 16384], lat_past [p, 512]
    (latent rows 0..p-1), pools / ring = dsa_state(p, ...) -> dict(hn, latent, q_lat [H, 512], o_lat, a, s1, ring_row,
    pooled (row p // 4), scores (sparse), sel (selected pools, sparse), rows (attended latent rows))"""
    ref = mhc_pre_ref(x, a_args)
    hn = ref["hn"]
    qk = hn @ W["qa_kva"]
    q_r = rms(qk[:, :q_lora], W["qa_norm"], eps)
    lat = rms(qk[:, q_lora:], W["kva_norm"], eps)
    q = (q_r @ W["qb"]).reshape(heads, -1)
    q_lat = torch.einsum("hd,hdk->hk", q, W["w_uk"])
    ir = indexer_rows(hn, ix)
    ring = ring.clone()
    ring[p % 4] = torch.cat([ir["k"][0], ir["g"][0]])
    pooled = pool(ring[:4, :128].unsqueeze(0), ring[:4, 128:].unsqueeze(0), ix["ape"])[0]
    pools = pools.clone()
    pools[p // 4] = pooled
    kv_all = torch.cat([lat_past, lat], dim=0)  # rows 0..p
    out = dict(hn=hn, latent=lat, q_lat=q_lat, ring_row=ring[p % 4], pooled=pooled)
    if p + 1 <= dense_max:
        rows = torch.arange(p + 1)
    else:
        n_closed = (p + 1) // 4
        qi = (q_r @ ix["wq_b"].double()).reshape(32, 128)
        s = torch.relu(qi @ pools[:n_closed].T)  # [32, n_closed]
        scores = (ir["w"][0] @ s).reshape(-1)
        sel = torch.topk(scores, topk).indices.sort().values
        rows = torch.cat([(4 * sel[:, None] + torch.arange(4)).reshape(-1), torch.arange(4 * n_closed, p + 1)])
        out.update(scores=scores, sel=sel)
    kv = kv_all[rows]
    sc = (q_lat @ kv.T) * scale
    o_lat = torch.softmax(sc, dim=-1) @ kv
    o = torch.einsum("hk,hkd->hd", o_lat, W["w_uv"]).reshape(1, -1)
    a = o @ W["o"]
    s1 = hc_expand(a, x.double().reshape(1, 4, -1), ref["post"], ref["comb"])
    out.update(o_lat=o_lat, a=a, s1=s1, rows=rows)
    return out
