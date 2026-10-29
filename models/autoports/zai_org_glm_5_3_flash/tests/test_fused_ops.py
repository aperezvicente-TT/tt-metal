# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-contract tests of the dedicated ops the fused decoder integrates (Stage 2 directives):

P1 ttnn.experimental.kda.fused_recurrent_kda (review-deferred items + the Stage 2 options)
  * bf16 / ROW_MAJOR wrapper path: bf16 TILE q/k/v and a ROW_MAJOR beta, vs the fp32 golden on the same bf16 values
  * K/V < 128: (K, V) in {(32, 32), (64, 96), (96, 64)}, final state and per-token states (T = 3)
  * inplace_state + [1,1,B,H*D] decode rows: bitwise equal to the original call, the state returned in place
  * program cache: repeated calls with freshly allocated inputs hit one program; each attribute adds one
P2a ttnn.experimental.deepseek.{fused_hyperconnection, mix_streams} (fp32 mode, GLM shapes, layer-0 real hc weights,
  golden streams): repeated invocations with moving buffers stay at the first call's program count, outputs bitwise
  equal across cache hits and close to the Stage 1 composite mHC
P2b ttnn.experimental.deepseek.moe.fused_experts (layer 4 real bfp4 experts, P2b's prepared golden MoE inputs/outputs):
  windows of B = 8 (bound 64) and B = 32 (bound 256): union <= bound asserted, output vs the same-dtype (bfp4-weight)
  reference with the device's own routing PCC >= 0.995, vs the fp8-weight reference reported, and the runtime hit count
  (bound > union) bitwise equal to an exact-union call.
Rows -> $GLM53_STAGE_EVIDENCE/fused_ops.jsonl.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    CACHE_DIR,
    MANIFEST,
    check,
    golden,
    hf_config,
    pcc,
    record,
)
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder, open_mesh

P2B_PREP = os.environ.get("GLM53_P2B_PREP", "/opt/glm53/work-p2b")


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh()
    yield m
    ttnn.close_mesh_device(m)


def _tt(x, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)


def _th(x):
    return ttnn.to_torch(x).float()


def _golden_kda(q, k, v, g, beta, s0, per_token=False):
    from ttnn.operations.transformer_golden import fused_recurrent_kda_golden

    return fused_recurrent_kda_golden(
        q, k, v, g, beta, initial_state=s0, output_final_state=not per_token, output_per_token_state=per_token
    )


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------- P1
def test_kda_bf16_rowmajor_wrapper(mesh):
    torch.manual_seed(0)
    B, H, D = 4, 64, 128
    q, k, v = (torch.randn(B, 1, H * D).bfloat16() for _ in range(3))
    g = -torch.rand(B, 1, H * D) * 0.5
    beta = torch.rand(B, 1, H).bfloat16()
    s0 = torch.randn(B, H, D, D) * 0.1
    o, s = ttnn.experimental.kda.fused_recurrent_kda(
        _tt(q, mesh, ttnn.bfloat16),
        _tt(k, mesh, ttnn.bfloat16),
        _tt(v, mesh, ttnn.bfloat16),
        _tt(g, mesh),
        _tt(beta, mesh, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        initial_state=_tt(s0, mesh),
        output_final_state=True,
    )
    o_ref, s_ref = _golden_kda(q.float(), k.float(), v.float(), g, beta.float(), s0)
    row = dict(
        test="kda_bf16_rowmajor_wrapper",
        o_pcc=pcc(_th(o), o_ref),
        state_pcc=pcc(_th(s), s_ref),
        o_max_rel=_rel(_th(o).reshape(o_ref.shape), o_ref),
        state_max_rel=_rel(_th(s), s_ref),
        tolerance_class="ULP on the bf16 values (the wrapper converts exactly: bf16 -> fp32, ROW_MAJOR -> TILE)",
    )
    record(row, name="fused_ops.jsonl")
    check([row] if row["o_max_rel"] > 1e-5 or row["state_max_rel"] > 1e-5 else [], "bf16 / ROW_MAJOR wrapper path")


@pytest.mark.parametrize("kdim,vdim", [(32, 32), (64, 96), (96, 64)])
@pytest.mark.parametrize("tokens", [1, 3])
def test_kda_small_kv(mesh, kdim, vdim, tokens):
    torch.manual_seed(kdim * 7 + vdim + tokens)
    B, H = 2, 4
    q, k = (torch.randn(B, tokens, H, kdim) for _ in range(2))
    v = torch.randn(B, tokens, H, vdim)
    g = -torch.rand(B, tokens, H, kdim) * 0.5
    beta = torch.rand(B, tokens, H)
    s0 = torch.randn(B, H, kdim, vdim) * 0.1
    per_token = tokens > 1
    o, s = ttnn.experimental.kda.fused_recurrent_kda(
        _tt(q, mesh),
        _tt(k, mesh),
        _tt(v, mesh),
        _tt(g, mesh),
        _tt(beta, mesh),
        initial_state=_tt(s0, mesh),
        output_final_state=not per_token,
        output_per_token_state=per_token,
    )
    o_ref, s_ref = _golden_kda(q, k, v, g, beta, s0, per_token=per_token)
    row = dict(
        test="kda_small_kv",
        K=kdim,
        V=vdim,
        T=tokens,
        o_max_rel=_rel(_th(o).reshape(o_ref.shape), o_ref),
        state_max_rel=_rel(_th(s).reshape(s_ref.shape), s_ref),
    )
    record(row, name="fused_ops.jsonl")
    check([row] if row["o_max_rel"] > 1e-5 or row["state_max_rel"] > 1e-5 else [], "K/V < 128")


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_kda_inplace_decode_rows(mesh, batch):
    torch.manual_seed(batch)
    H, D = 64, 128
    q, k, v = (torch.randn(1, 1, batch, H * D) for _ in range(3))
    g = -torch.rand(1, 1, batch, H * D) * 0.5
    beta = torch.rand(1, 1, batch, H)
    s0 = torch.randn(batch, H, D, D) * 0.1
    o_ref, s_ref = ttnn.experimental.kda.fused_recurrent_kda(
        _tt(q.reshape(batch, 1, -1), mesh),
        _tt(k.reshape(batch, 1, -1), mesh),
        _tt(v.reshape(batch, 1, -1), mesh),
        _tt(g.reshape(batch, 1, -1), mesh),
        _tt(beta.reshape(batch, 1, H), mesh),
        initial_state=_tt(s0, mesh),
        output_final_state=True,
    )
    st = _tt(s0, mesh)
    o, s = ttnn.experimental.kda.fused_recurrent_kda(
        _tt(q, mesh),
        _tt(k, mesh),
        _tt(v, mesh),
        _tt(g, mesh),
        _tt(beta, mesh),
        initial_state=st,
        output_final_state=True,
        inplace_state=True,
    )
    row = dict(
        test="kda_inplace_decode_rows",
        batch=batch,
        o_bitwise=bool(torch.equal(_th(o), _th(o_ref))),
        state_bitwise=bool(torch.equal(_th(st), _th(s_ref))),
        state_returned_in_place=s.buffer_address() == st.buffer_address(),
    )
    record(row, name="fused_ops.jsonl")
    check([] if all(v for k_, v in row.items() if k_ not in ("test", "batch")) else [row], "in-place / decode rows")


def test_kda_program_cache(mesh):
    torch.manual_seed(5)
    B, H, D = 4, 64, 128
    keep = []

    def call(scale=None, inplace=True):
        q, k, v = (torch.randn(1, 1, B, H * D) for _ in range(3))
        g, beta, s0 = -torch.rand(1, 1, B, H * D) * 0.5, torch.rand(1, 1, B, H), torch.randn(B, H, D, D) * 0.1
        ins = [_tt(x, mesh) for x in (q, k, v, g, beta, s0)]
        keep.append(ins)  # previous inputs stay alive -> the next call's buffers move
        o, s = ttnn.experimental.kda.fused_recurrent_kda(
            *ins[:5], initial_state=ins[5], output_final_state=True, inplace_state=inplace, scale=scale
        )
        ref_o, ref_s = (
            _golden_kda(
                q.reshape(B, 1, -1),
                k.reshape(B, 1, -1),
                v.reshape(B, 1, -1),
                g.reshape(B, 1, -1),
                beta.reshape(B, 1, H),
                s0,
            )
            if scale is None
            else (None, None)
        )
        keep.append((o, s))
        err = None if ref_o is None else _rel(_th(o).reshape(ref_o.shape), ref_o)
        return mesh.num_program_cache_entries(), err

    n0, e0 = call()
    n1, e1 = call()
    n2, e2 = call()
    n3, _ = call(scale=0.05)
    n4, _ = call(inplace=False)
    row = dict(
        test="kda_program_cache",
        entries=[n0, n1, n2, n3, n4],
        max_rel_each_call=[e0, e1, e2],
        cache_hits_reuse_one_program=n1 == n0 and n2 == n0,
        scale_adds_one=n3 == n0 + 1,
        inplace_flag_adds_one=n4 == n3 + 1,
    )
    record(row, name="fused_ops.jsonl")
    bad = not (row["cache_hits_reuse_one_program"] and row["scale_adds_one"] and row["inplace_flag_adds_one"])
    bad |= any(e > 1e-5 for e in (e0, e1, e2))
    check([row] if bad else [], "fused_recurrent_kda program cache")


# ------------------------------------------------------------------------------------------------------------ P2a
@pytest.mark.parametrize("batch", [1, 8])
def test_mhc_program_cache_hits(mesh, batch):
    """fused_hyperconnection + mix_streams (fp32) on layer 0's attn_hc: 3 calls with freshly allocated inputs (previous
    ones kept alive, so addresses move) -> no new programs after the first, outputs bitwise equal across cache hits and
    PCC >= 0.9999 vs the Stage 1 composite mHC on the same inputs."""
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedHyperConnection, to_stacked
    from models.autoports.zai_org_glm_5_3_flash.tt.hyperconnection import HyperConnection
    from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource

    cfg = hf_config()
    src = TileCacheSource(CACHE_DIR, MANIFEST)
    kw = dict(
        hc_mult=4, hidden_size=4096, sinkhorn_iters=cfg.hc_sinkhorn_iters, hc_eps=cfg.hc_eps, norm_eps=cfg.rms_norm_eps
    )
    fn, base, scale = src.device(0, "hc_attn_fn", mesh), src.host(0, "hc_attn_base"), src.host(0, "hc_attn_scale")
    fused = FusedHyperConnection(mesh, fn, base, scale, ksplit=16, **kw)
    comp = HyperConnection(mesh, fn, base, scale, **kw)
    g = golden(0, 128)
    x_host = g["streams_in"][:, :batch].reshape(1, 1, batch, -1).float()
    y_host = torch.randn(1, 1, batch, 4096).bfloat16() * 0.05
    keep, outs, entries = [], [], []
    for _ in range(3):
        x = ttnn.from_torch(x_host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        y = ttnn.from_torch(y_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        xs = to_stacked(x, 4)
        h, post, comb = fused.pre(x, xs)
        new = fused.post(y, xs, post, comb)
        keep += [x, y, xs, h, post, comb, new]
        outs.append((_th(h), _th(new)))
        entries.append(mesh.num_program_cache_entries())
    hc, pc, cc = comp.pre(x)
    new_c = comp.post(y, x, pc, cc)
    row = dict(
        test="mhc_program_cache_hits",
        batch=batch,
        entries=entries,
        no_new_program_on_cache_hits=entries[1] == entries[0] and entries[2] == entries[0],
        outputs_bitwise_equal_across_hits=all(torch.equal(a, b) for o in outs[1:] for a, b in zip(o, outs[0])),
        collapsed_pcc_vs_composite=pcc(outs[0][0].reshape(batch, -1), _th(hc).reshape(batch, -1)),
        new_streams_pcc_vs_composite=pcc(outs[0][1].reshape(batch, -1), _th(new_c).reshape(batch, -1)),
    )
    record(row, name="fused_ops.jsonl")
    bad = not (row["no_new_program_on_cache_hits"] and row["outputs_bitwise_equal_across_hits"])
    bad |= row["collapsed_pcc_vs_composite"] < 0.9999 or row["new_streams_pcc_vs_composite"] < 0.9999
    check([row] if bad else [], "mHC fused ops program cache")


# ------------------------------------------------------------------------------------------------------------ P2b
@pytest.fixture(scope="module")
def moe_l4(mesh):
    need = [os.path.join(P2B_PREP, "L04_T128", f) for f in ("inputs.pt", "Y_bfp4_xbf16.npy", "Y_fp8.npy")]
    if not all(os.path.exists(p) for p in need):
        pytest.skip(
            f"P2b prepared inputs missing under {P2B_PREP} (doc/prework/p2b_fused_experts/scripts/prep_moe_inputs.py)"
        )
    dec = load_fused_decoder(hf_config(), 4, mesh)
    inp = torch.load(need[0])
    ys = {k: np.load(os.path.join(P2B_PREP, "L04_T128", f"{k}.npy"), mmap_mode="r") for k in ("Y_bfp4_xbf16", "Y_fp8")}
    return dec.layer.mlp, inp, ys


def _combine(y, idx, w, rows):
    out = torch.zeros(len(rows), y.shape[-1], dtype=torch.float64)
    for b, r in enumerate(rows):
        for j in range(idx.shape[1]):
            out[b] += float(w[b, j]) * torch.from_numpy(np.asarray(y[int(idx[b, j]), r], dtype=np.float64))
    return out


@pytest.mark.parametrize("batch,bound", [(8, 64), (32, 256)])
def test_fused_experts_gate(mesh, moe_l4, batch, bound):
    mo, inp, ys = moe_l4
    hf = inp["hf"].float()  # [128, 4096] golden MoE input of layer 4 (prompt 128)
    rows_all, bad = [], []
    for w0 in range(0, hf.shape[0], batch):
        rows = list(range(w0, w0 + batch))
        x = ttnn.from_torch(
            hf[rows].reshape(1, 1, batch, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        idx_tt, sc_tt = mo.router(x)
        idx = ttnn.to_torch(idx_tt).reshape(-1, 8)[:batch].long()
        sc = ttnn.to_torch(sc_tt).float().reshape(-1, 288)[:batch]
        union = len(set(idx.flatten().tolist()))
        assert union <= bound, f"union {union} exceeds the static bound {bound}"  # the P2b union assertion
        call = lambda n: ttnn.experimental.deepseek.moe.fused_experts(
            x,
            routing_indices=idx_tt,
            routing_scores=sc_tt,
            gate_up_weights=mo.gate_up,
            down_weights=mo.down,
            num_experts=n,
            intermediate_size=mo.intermediate,
            swiglu_limit=mo.limit,
            top_k=8,
            routed_scaling_factor=mo.scaling,
            routing_eps=1e-20,
            experts_block_size=1,
        )
        out_b = ttnn.to_torch(call(bound)).float().reshape(-1, 4096)[:batch]
        out_u = ttnn.to_torch(call(union)).float().reshape(-1, 4096)[:batch]
        sel = sc.gather(1, idx)
        w = mo.scaling * sel / (sel.sum(-1, keepdim=True) + 1e-20)
        ref_same = _combine(ys["Y_bfp4_xbf16"], idx, w, rows)  # bfp4 weights, the device's own routing, bf16 input
        ref_fp8 = _combine(ys["Y_fp8"], inp["idx"][rows], inp["w"][rows], rows)  # fp8 weights, fp32 reference routing
        r = dict(
            test="fused_experts_gate",
            batch=batch,
            bound=bound,
            window=w0,
            union=union,
            pcc_vs_same_dtype_ref=pcc(out_b, ref_same),
            pcc_vs_fp8_ref=pcc(out_b, ref_fp8),
            runtime_count_bitwise_equal_exact_union=bool(torch.equal(out_b, out_u)),
        )
        rows_all.append(r)
        record(r, name="fused_ops.jsonl")
        if r["pcc_vs_same_dtype_ref"] < 0.995 or not r["runtime_count_bitwise_equal_exact_union"]:
            bad.append(r)
    check(bad, "fused_experts gate (same-dtype PCC >= 0.995, runtime count == exact union)")
