# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.fused_recurrent_kda against its torch golden (the GLM-5.3-Flash `recurrent_kda`).

Real geometry: H = 64 heads, K = V = 128, B in {1, 8, 32}. Inputs follow the model's ranges: g = -5 sigmoid(z)
(per key channel), beta = sigmoid(.), q/k/v raw (the op L2-normalises q and k). Set GLM53_KDA_EVIDENCE to a
.jsonl path to append one row per check.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
import torch

import ttnn
from ttnn.operations.transformer_golden import fused_recurrent_kda_golden, recurrent_kda

H, K, V = 64, 128, 128
PCC_BAR = 0.9999
# fp32 everywhere; the remaining differences are the summation order of the 128-term reductions and the SFPU
# exp/sqrt/reciprocal ulps (measured max |diff| <= 3e-7 on O(1) values, rel. RMSE <= 2e-7; see the evidence rows).
RTOL, ATOL = 1e-5, 1e-6

OP = ttnn.experimental.kda.fused_recurrent_kda
EVIDENCE = os.environ.get("GLM53_KDA_EVIDENCE", "")
REFERENCE_DIR = os.path.join(
    os.environ.get("GLM53_WS", "/opt/glm53/kit"), "reference"
)
REAL_CAPTURE = os.environ.get("GLM53_KDA_REAL_CAPTURE", "/opt/glm53/work-p1/real/kda_decode0_L00_P2048.pt")


def record(row: dict):
    if not EVIDENCE:
        return
    row = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    with open(EVIDENCE, "a") as f:
        f.write(json.dumps(row) + "\n")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    if torch.equal(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, **meta) -> dict:
    assert actual.shape == expected.shape, f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"
    diff = (actual.double() - expected.double()).abs()
    row = {
        "check": name,
        **meta,
        "pcc": pcc(actual, expected),
        "max_abs": float(diff.max()),
        "rel_rmse": float(diff.pow(2).mean().sqrt() / expected.double().pow(2).mean().sqrt().clamp_min(1e-30)),
        "max_abs_ref": float(expected.abs().max()),
        "allclose": bool(torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "rtol": RTOL,
        "atol": ATOL,
    }
    record(row)
    print(row)
    assert row["pcc"] >= PCC_BAR, row
    assert row["allclose"], row
    return row


def make_inputs(B: int, T: int, seed: int, *, state_scale: float = 0.5, with_state: bool = True):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=gen)
    k = torch.randn(B, T, H, K, generator=gen)
    v = torch.randn(B, T, H, V, generator=gen)
    g = -5.0 * torch.sigmoid(2.0 * torch.randn(B, T, H, K, generator=gen))
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen))
    s0 = state_scale * torch.randn(B, H, K, V, generator=gen) if with_state else None
    return q, k, v, g, beta, s0


def to_dev(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(x.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def run_op(device, q, k, v, g, beta, s0, **kw):
    o, st = OP(
        to_dev(q, device),
        to_dev(k, device),
        to_dev(v, device),
        to_dev(g, device),
        to_dev(beta, device),
        initial_state=None if s0 is None else to_dev(s0, device),
        **kw,
    )
    return ttnn.to_torch(o), None if st is None else ttnn.to_torch(st)


# ---------------------------------------------------------------------------------------------------------------
# golden (host only)
# ---------------------------------------------------------------------------------------------------------------


def _vendored_reference():
    if not os.path.isdir(os.path.join(REFERENCE_DIR, "glm5_next_text")):
        pytest.skip(f"GLM-5.3-Flash reference not found under {REFERENCE_DIR}")
    if REFERENCE_DIR not in sys.path:
        sys.path.insert(0, REFERENCE_DIR)
    from glm5_next_text import modules

    return modules


@pytest.mark.parametrize("B, T", [(1, 1), (2, 5)])
def test_golden_is_the_glm53_reference(B, T):
    """The registered golden is the vendored reference recurrence (modules.recurrent_kda), bit for bit."""
    modules = _vendored_reference()
    q, k, v, g, beta, s0 = make_inputs(B, T, seed=1)
    o_ref, s_ref = modules.recurrent_kda(q, k, v, g, beta, initial_state=s0)
    o, s = ttnn.get_golden_function(OP)(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)


def test_golden_per_token_states_are_chained_steps():
    q, k, v, g, beta, s0 = make_inputs(2, 4, seed=2)
    o, states = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, output_per_token_state=True)
    s = s0
    for t in range(4):
        sl = slice(t, t + 1)
        o_t, s = fused_recurrent_kda_golden(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=s, output_final_state=True
        )
        assert torch.equal(o_t, o[:, sl]) and torch.equal(s, states[:, t])


# ---------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("B", [1, 8, 32])
def test_decode_step(device, B):
    q, k, v, g, beta, s0 = make_inputs(B, 1, seed=10 + B)
    o, s = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
    o_ref, s_ref = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    check("decode_o", o, o_ref, B=B, T=1)
    check("decode_state", s, s_ref, B=B, T=1)


def test_decode_zero_initial_state(device):
    q, k, v, g, beta, _ = make_inputs(8, 1, seed=3, with_state=False)
    o, s = run_op(device, q, k, v, g, beta, None, output_final_state=True)
    o_ref, s_ref = fused_recurrent_kda_golden(q, k, v, g, beta, output_final_state=True)
    check("zero_state_o", o, o_ref, B=8, T=1)
    check("zero_state_state", s, s_ref, B=8, T=1)


def test_output_only(device):
    """Without a state request the op returns (o, None) and o is unchanged."""
    q, k, v, g, beta, s0 = make_inputs(1, 1, seed=4)
    o, s = run_op(device, q, k, v, g, beta, s0)
    assert s is None
    o2, _ = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
    assert torch.equal(o, o2)


@pytest.mark.parametrize("B", [1, 8])
def test_state_carried_64_steps(device, B):
    """64 decode calls, the device state fed back each step, against the reference recurrence over T = 64."""
    steps = 64
    q, k, v, g, beta, s0 = make_inputs(B, steps, seed=20 + B)
    o_ref, s_ref = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    state = to_dev(s0, device)
    outs = []
    for t in range(steps):
        sl = slice(t, t + 1)
        o_t, state = OP(
            to_dev(q[:, sl], device),
            to_dev(k[:, sl], device),
            to_dev(v[:, sl], device),
            to_dev(g[:, sl], device),
            to_dev(beta[:, sl], device),
            initial_state=state,
            output_final_state=True,
        )
        outs.append(ttnn.to_torch(o_t))
    check("carry64_o", torch.cat(outs, dim=1), o_ref, B=B, T=1, steps=steps)
    check("carry64_state", ttnn.to_torch(state), s_ref, B=B, T=1, steps=steps)


@pytest.mark.parametrize("B, T", [(1, 4), (1, 8), (8, 4)])
def test_verify_per_token_states(device, B, T):
    """T = K + 1 speculative verify: o and the state after every token in one call; equal to T chained decode
    calls on device, bit for bit (the state stays fp32 in L1 between tokens)."""
    q, k, v, g, beta, s0 = make_inputs(B, T, seed=30 + T + B)
    o, states = run_op(device, q, k, v, g, beta, s0, output_per_token_state=True)
    o_ref, states_ref = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, output_per_token_state=True)
    check("verify_o", o, o_ref, B=B, T=T)
    check("verify_states", states, states_ref, B=B, T=T)

    o_f, s_f = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
    assert torch.equal(o_f, o) and torch.equal(s_f, states[:, -1])

    state = to_dev(s0, device)
    for t in range(T):
        sl = slice(t, t + 1)
        o_t, state = OP(
            to_dev(q[:, sl], device),
            to_dev(k[:, sl], device),
            to_dev(v[:, sl], device),
            to_dev(g[:, sl], device),
            to_dev(beta[:, sl], device),
            initial_state=state,
            output_final_state=True,
        )
        assert torch.equal(ttnn.to_torch(o_t), o[:, sl]), f"token {t}: o differs from the T={T} call"
        assert torch.equal(ttnn.to_torch(state), states[:, t]), f"token {t}: state differs from the T={T} call"
    record({"check": "verify_equals_chained_decode", "B": B, "T": T, "bitwise": True})


def test_bitwise_on_exact_inputs(device):
    """Dyadic inputs that make every fp32 operation of the step exact (no L2 norm, scale 1, g = 0, beta = 1/2,
    small integers / 16): the device must equal the golden bit for bit, whatever the summation order -- a check of
    the data movement and of every per-element formula."""
    B, T = 8, 3  # values stay on a 2^-(4+t) grid well inside 24 bits for 3 tokens
    gen = torch.Generator().manual_seed(9)
    q = torch.randint(-1, 2, (B, T, H, K), generator=gen).float()
    k = torch.randint(-1, 2, (B, T, H, K), generator=gen).float()
    v = torch.randint(-8, 9, (B, T, H, V), generator=gen).float() / 16
    g = torch.zeros(B, T, H, K)
    beta = torch.full((B, T, H), 0.5)
    s0 = torch.randint(-8, 9, (B, H, K, V), generator=gen).float() / 16
    kw = dict(scale=1.0, use_qk_l2norm=False, output_per_token_state=True)
    o, states = run_op(device, q, k, v, g, beta, s0, **kw)
    o_ref, states_ref = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, **kw)
    assert torch.equal(o, o_ref), f"o differs: max |diff| {float((o - o_ref).abs().max())}"
    assert torch.equal(states, states_ref), f"states differ: max |diff| {float((states - states_ref).abs().max())}"
    record(
        {"check": "bitwise_exact_inputs", "B": B, "T": T, "bitwise": True, "max_abs_o_ref": float(o_ref.abs().max())}
    )


def test_flat_head_views(device):
    """[B, T, H*D] views give the same bits as [B, T, H, D]."""
    B, T = 8, 2
    q, k, v, g, beta, s0 = make_inputs(B, T, seed=5)
    o4, s4 = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
    flat = lambda x: x.reshape(B, T, -1)
    o3, s3 = run_op(device, flat(q), flat(k), flat(v), flat(g), beta, s0, output_final_state=True)
    assert torch.equal(o4, o3) and torch.equal(s4, s3)


def test_without_in_kernel_l2norm(device):
    q, k, v, g, beta, s0 = make_inputs(8, 1, seed=6)
    from ttnn.operations.transformer_golden import kda_l2norm

    qn, kn = kda_l2norm(q), kda_l2norm(k)
    o, s = run_op(device, qn, kn, v, g, beta, s0, output_final_state=True, use_qk_l2norm=False)
    o_ref, s_ref = fused_recurrent_kda_golden(
        qn, kn, v, g, beta, initial_state=s0, output_final_state=True, use_qk_l2norm=False
    )
    check("no_l2norm_o", o, o_ref, B=8, T=1)
    check("no_l2norm_state", s, s_ref, B=8, T=1)


def test_custom_scale(device):
    q, k, v, g, beta, s0 = make_inputs(1, 2, seed=7)
    o, s = run_op(device, q, k, v, g, beta, s0, output_final_state=True, scale=0.3)
    o_ref, s_ref = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, output_final_state=True, scale=0.3)
    check("scale_o", o, o_ref, B=1, T=2)
    check("scale_state", s, s_ref, B=1, T=2)


def test_program_cache(device):
    """One program per configuration; a second call with new data (new buffers) hits the cache and is correct."""
    B = 8
    device.clear_program_cache()
    base = device.num_program_cache_entries()
    for i in range(3):
        q, k, v, g, beta, s0 = make_inputs(B, 1, seed=40 + i)
        o, s = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
        o_ref, s_ref = fused_recurrent_kda_golden(q, k, v, g, beta, initial_state=s0, output_final_state=True)
        check("program_cache_o", o, o_ref, B=B, T=1, call=i)
        check("program_cache_state", s, s_ref, B=B, T=1, call=i)
        assert device.num_program_cache_entries() == base + 1, "fused_recurrent_kda recompiled on a cache hit"
    q, k, v, g, beta, s0 = make_inputs(B, 2, seed=44)
    run_op(device, q, k, v, g, beta, s0, output_per_token_state=True)
    assert device.num_program_cache_entries() == base + 2
    record({"check": "program_cache", "entries_after_3_calls": 1, "entries_after_new_T": 2})


def test_determinism(device):
    B = 32
    q, k, v, g, beta, s0 = make_inputs(B, 1, seed=8)
    first = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
    for _ in range(2):
        again = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
        assert torch.equal(first[0].view(torch.int32), again[0].view(torch.int32))
        assert torch.equal(first[1].view(torch.int32), again[1].view(torch.int32))
    record({"check": "determinism", "B": B, "runs": 3, "bitwise": True})


def test_gate_range_and_in_kernel_exp(device):
    """The gate path: g = -5 sigmoid(z) in fp32 ttnn ops over z in [-20, 20] against torch, then the op's in-kernel
    exp(g): with k = 0 the update is S * exp(g)[:, None], so state[r, c] / S0[r, c] reads exp(g[r]) back."""
    B, T = 1, 1
    n = B * T * H * K
    z = torch.linspace(-20.0, 20.0, n).reshape(B, T, H, K)
    z_dev = to_dev(z, device)
    g_dev = ttnn.multiply(ttnn.sigmoid(z_dev), -5.0)
    g_tt = ttnn.to_torch(g_dev)
    g_ref = -5.0 * torch.sigmoid(z)
    gate_err = (g_tt - g_ref).abs()
    row = {
        "check": "gate_-5sigmoid_ttnn_fp32",
        "z_range": [-20.0, 20.0],
        "max_abs": float(gate_err.max()),
        "max_abs_at_z": float(z.reshape(-1)[gate_err.reshape(-1).argmax()]),
        "pcc": pcc(g_tt, g_ref),
        "flushes_to_zero": int(((g_tt == 0) & (g_ref != 0)).sum()),
    }
    record(row)
    print(row)
    assert row["flushes_to_zero"] == 0 and row["max_abs"] < 1e-5, row

    # Stage 1's gate chain (tt/kda.py _decay): z = (f + dt_bias) * exp(A_log[h]), g = -5 sigmoid(z), fp32 ttnn ops.
    gen = torch.Generator().manual_seed(11)
    f = 4.0 * torch.randn(1, 1, 8, H * K, generator=gen)
    dt_bias = torch.randn(1, 1, 1, H * K, generator=gen)
    rate = torch.exp(torch.empty(H).uniform_(-3.0, 1.0, generator=gen)).repeat_interleave(K).reshape(1, 1, 1, H * K)
    z_chain = (f + dt_bias) * rate
    g_chain = ttnn.to_torch(
        ttnn.multiply(
            ttnn.sigmoid(ttnn.multiply(ttnn.add(to_dev(f, device), to_dev(dt_bias, device)), to_dev(rate, device))),
            -5.0,
        )
    )
    g_chain_ref = -5.0 * torch.sigmoid(z_chain)
    row = {
        "check": "gate_stage1_chain_fp32",
        "z_min": float(z_chain.min()),
        "z_max": float(z_chain.max()),
        "max_abs": float((g_chain - g_chain_ref).abs().max()),
        "flushes_to_zero": int(((g_chain == 0) & (g_chain_ref != 0)).sum()),
    }
    record(row)
    print(row)
    assert row["flushes_to_zero"] == 0 and row["max_abs"] < 1e-5, row

    # A fused-activation spelling of the same gate, recorded for reference only (the op takes g from the path above).
    fused = ttnn.to_torch(
        ttnn.multiply(z_dev, -5.0, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)])
    )
    fused_err = (fused - g_ref).abs()
    row = {
        "check": "gate_fused_sigmoid_activation_fp32_reference_only",
        "z_range": [-20.0, 20.0],
        "max_abs": float(fused_err.max()),
        "max_abs_at_z": float(z.reshape(-1)[fused_err.reshape(-1).argmax()]),
        "flushes_to_zero": int(((fused == 0) & (g_ref != 0)).sum()),
    }
    record(row)
    print(row)

    q = torch.randn(B, T, H, K)
    k = torch.zeros(B, T, H, K)
    v = torch.randn(B, T, H, V)
    beta = torch.rand(B, T, H)
    s0 = torch.ones(B, H, K, V)
    for name, g in (
        ("exp_of_ttnn_gate", g_tt),
        ("exp_g_in_[-20,0]", torch.linspace(-20.0, 0.0, n).reshape(B, T, H, K)),
    ):
        _, s = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
        decay = s[0, :, :, 0]  # [H, K]
        ref = g[0, 0].exp()
        rel = ((decay.double() - ref.double()).abs() / ref.double()).max()
        assert torch.equal(s, s[..., :1].expand_as(s)), "decay is not constant along each state row"
        row = {"check": f"in_kernel_{name}", "max_rel": float(rel), "pcc": pcc(decay, ref)}
        record(row)
        print(row)
        assert rel < 1e-6, row


@pytest.mark.skipif(not os.path.exists(REAL_CAPTURE), reason=f"real-weight capture {REAL_CAPTURE} not present")
def test_real_weights_golden_step(device):
    """Layer-0 KDA inputs of the first greedy decode step after the 2048-token golden prompt (captured on the CPU
    from goldens/layer_io/L00/2048/decode0_streams_in.pt + goldens/kda_state/2048/L00.pt through the reference
    layer); the device step against the CPU golden step."""
    cap = torch.load(REAL_CAPTURE)
    q, k, v, g, beta, s0 = (cap[n] for n in ("q", "k", "v", "g", "beta", "initial_state"))
    o, s = run_op(device, q, k, v, g, beta, s0, output_final_state=True)
    meta = {"layer": cap["layer"], "prompt": cap["prompt"], "B": q.shape[0], "T": q.shape[1]}
    check("real_L00_P2048_o", o, cap["o"], **meta)
    check("real_L00_P2048_state", s, cap["state"], **meta)
    out = os.environ.get("GLM53_KDA_REAL_DEVICE_OUT", "")
    if out:
        torch.save({"o": o, "state": s}, out)
