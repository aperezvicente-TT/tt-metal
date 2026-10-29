# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash shaped tests for ``ttnn.experimental.deepseek.{fused_hyperconnection, mix_streams}``.

H = 4 streams, hidden 4096, B in {1, 8, 32} decode rows, Sinkhorn 20 iterations, real ``hc_{attn,ffn}_{fn,base,scale}``
weights from the GLM-5.3-Flash tile cache, real residual streams (golden ``layer_io`` inputs and the layer-0
capture of ``doc/prework/p2a_mhc_ops/scripts/capture_mhc_io.py``, which also provides the real sublayer outputs
fed to ``mix_streams``). References are the pure-torch fp32 ``HyperConnection`` / ``hc_expand`` of the bring-up
workspace (``reference/glm5_next_text/modules.py``).

Environment: ``GLM53_WS`` (workspace), ``GLM53_CACHE_DIR`` (tile cache), ``GLM53_P2A_CAPTURE`` (layer-0 capture),
``GLM53_P2A_PCC_JSONL`` (optional: append one JSON row per checked output).
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import time

import pytest
import torch

import ttnn

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
CACHE = os.environ.get("GLM53_CACHE_DIR", "/opt/glm53/tt-cache/glm53-flash")
CAPTURE = os.environ.get("GLM53_P2A_CAPTURE", "/opt/glm53/work-p2a/mhc_io_L00_128.pt")
PCC_JSONL = os.environ.get("GLM53_P2A_PCC_JSONL")

PCC_BAR = 0.999
HC = 4
HIDDEN = 4096
ITERS = 20
PROMPT = 128

if not (os.path.isdir(os.path.join(WS, "reference")) and os.path.isdir(CACHE)):
    pytest.skip("GLM-5.3-Flash workspace / weight cache not available", allow_module_level=True)

sys.path.insert(0, os.path.join(WS, "reference"))
sys.path.insert(0, os.path.join(WS, "weights"))

import tile_codec  # noqa: E402
from glm5_next_text.config import Glm5NextTextConfig  # noqa: E402
from glm5_next_text.modules import HyperConnection, hc_expand, unweighted_rms_norm  # noqa: E402

CFG = Glm5NextTextConfig()
TT_DTYPES = {"bf16": (ttnn.bfloat16, torch.bfloat16), "fp32": (ttnn.float32, torch.float32)}


@functools.lru_cache(maxsize=None)
def _manifest():
    with open(os.path.join(WS, "weights", "manifest.json")) as f:
        return json.load(f)["tensors"]


def _cache_tensor(key: str) -> torch.Tensor:
    e = _manifest()[key]
    return torch.from_numpy(tile_codec.decode(os.path.join(CACHE, e["path"]), e["shape"], e["dtype"], e["layout"]))


@functools.lru_cache(maxsize=None)
def _hc_weights(layer: int, sub: str):
    """fn [24, 16384] (bf16 values, fp32 storage), base [24] fp32, scale [3] fp32 from the tile cache."""
    p = f"layers.{layer}.hc_{sub}_"
    fn = _cache_tensor(p + "fn").T.contiguous()
    return fn, _cache_tensor(p + "base"), _cache_tensor(p + "scale")


@functools.lru_cache(maxsize=None)
def _ref_hc(layer: int, sub: str) -> HyperConnection:
    fn, base, scale = _hc_weights(layer, sub)
    hc = HyperConnection(CFG, dtype=torch.float32, param_dtype=torch.float32)
    hc.load_state_dict({"fn": fn, "base": base, "scale": scale})
    return hc.eval()


@functools.lru_cache(maxsize=None)
def _capture():
    if not os.path.exists(CAPTURE):
        pytest.skip(f"layer-0 capture {CAPTURE} missing (run capture_mhc_io.py)")
    return torch.load(CAPTURE, weights_only=False)


def _token_ids(b: int) -> torch.Tensor:
    return torch.tensor([PROMPT - 1]) if b == 1 else torch.linspace(0, PROMPT - 1, b).round().long()


def _streams(layer: int, sub: str, b: int) -> torch.Tensor:
    """Real fp32 residual streams entering `layer`'s `sub` hyper-connection, B prompt tokens as [B, 1, 4, 4096]."""
    if layer == 0:
        s = _capture()["records"][sub]["streams"]
    else:
        assert sub == "attn", "only layer 0 has captured ffn-side streams"
        s = torch.load(
            os.path.join(WS, "goldens", "layer_io", f"L{layer:02d}", str(PROMPT), "streams_in.pt"), weights_only=True
        )
    return s[0, _token_ids(b)].unsqueeze(1).contiguous().float()


def _sublayer_out(sub: str, b: int) -> torch.Tensor:
    """Real fp32 layer-0 sublayer output (KDA attention / dense MLP) for the same tokens, [B, 1, 1, 4096]."""
    return _capture()["records"][sub]["sublayer_out"][0, _token_ids(b)].reshape(b, 1, 1, HIDDEN).float()


def _mixes(hc: HyperConnection, streams: torch.Tensor) -> torch.Tensor:
    """fp32 m = unweighted_rmsnorm(flatten(streams)) @ fn^T (modeling:300-303), [1, 1, B, 24]."""
    b = streams.shape[0]
    flat = unweighted_rms_norm(streams.reshape(b, 1, -1).float(), hc.norm_eps)
    return torch.nn.functional.linear(flat, hc.fn.float()).reshape(1, 1, b, -1)


def device_fn_t(device, layer: int, sub: str):
    """fp32 fn^T [1, 1, 16384, 24] on `device`, loaded from the tile cache."""
    e = _manifest()[f"layers.{layer}.hc_{sub}_fn"]
    fn_t = ttnn.load_tensor(os.path.join(CACHE, e["path"]), device=device)
    return ttnn.reshape(ttnn.typecast(fn_t, ttnn.float32), (1, 1, 4 * HIDDEN, 24))


def _device_mixes(device, streams_tt, layer: int, sub: str, tt_dtype, fn_t=None):
    """On-device m: fp32 HiFi4 matmul against fn^T, scaled by rsqrt(mean(x^2) + eps)."""
    b = streams_tt.shape[0]
    fn_t = device_fn_t(device, layer, sub) if fn_t is None else fn_t
    x = ttnn.reshape(streams_tt, (1, 1, b, 4 * HIDDEN))
    if x.dtype != ttnn.float32:
        x = ttnn.typecast(x, ttnn.float32)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    m = ttnn.matmul(x, fn_t, compute_kernel_config=ckc, dtype=ttnn.float32)
    ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
    inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / (4 * HIDDEN)), CFG.rms_norm_eps))
    m = ttnn.multiply(m, inv)
    return m if tt_dtype == ttnn.float32 else ttnn.typecast(m, tt_dtype)


def _to_tt(x: torch.Tensor, device, tt_dtype):
    return ttnn.from_torch(x, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    den = (a.norm() * b.norm()).item()
    return 1.0 if den == 0 and torch.equal(a, b) else (a @ b).item() / den


MANTISSA_BITS = {torch.bfloat16: 7, torch.float32: 23}


def _ulp_err(got: torch.Tensor, want: torch.Tensor, torch_dtype, term_scale: torch.Tensor | None) -> torch.Tensor:
    """|got - want| in units of the output dtype's ulp at |want|. For accumulated outputs, |want| is floored at the
    element's sum of |terms| (`term_scale`): an fp32 sum is only defined to the ulp of its terms, so cancellation
    to a near-zero result is not counted as many ulps of that result."""
    w = want.double().abs()
    if term_scale is not None:
        w = torch.maximum(w, term_scale.double().reshape(w.shape))
    w = w.clamp_min(torch.finfo(torch.float32).tiny)
    ulp = torch.exp2(torch.floor(torch.log2(w)) - MANTISSA_BITS[torch_dtype])
    return (got.double() - want.double()).abs() / ulp


@functools.lru_cache(maxsize=None)
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=11", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _record(row: dict):
    row.update(
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tt_metal_sha=_git_sha(),
        host=os.uname()[1],
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES"),
    )
    if PCC_JSONL:
        with open(PCC_JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")


def _check(
    rows: list[dict],
    name: str,
    got: torch.Tensor,
    ref: torch.Tensor,
    faithful_ref: torch.Tensor,
    out_dtype,
    base: dict,
    term_scale: torch.Tensor | None = None,
):
    """PCC / max-abs against the fp32 reference, plus ulp error against `faithful_ref` = fp32 math on exactly the
    op's inputs (rounded to the output dtype for bf16 outputs)."""
    got = got.float().reshape(ref.shape)
    want = faithful_ref.float().reshape(ref.shape)
    if out_dtype == torch.bfloat16:
        want = want.to(torch.bfloat16).float()
    ulp = _ulp_err(got, want, out_dtype, term_scale)
    row = dict(
        base,
        output=name,
        pcc=_pcc(got, ref),
        max_abs=(got - ref).abs().max().item(),
        ref_absmax=ref.abs().max().item(),
        max_rel=((got - ref).abs() / ref.abs().max()).max().item(),
        faithful_exact_frac=(got == want).float().mean().item(),
        faithful_max_ulp=ulp.max().item(),
        faithful_ulp_le1_frac=(ulp <= 1).float().mean().item(),
        faithful_term_floor=term_scale is not None,
    )
    rows.append(row)


def _coeff_math(hc: HyperConnection, mixes_in: torch.Tensor, op_dtype, math_dtype=torch.float32):
    """pre / post / comb from exactly the mixes, biases (rounded to the op dtype) and scales the op receives,
    evaluated in `math_dtype` (fp32 = the reference's arithmetic, fp64 = the exact value to fp32 precision)."""
    base = hc.base.detach().to(op_dtype).to(math_dtype)
    s0, s1, s2 = hc.scale.detach().to(math_dtype).unbind(0)
    eps = hc.hc_eps
    pre_w, post_w, comb_w = mixes_in.to(math_dtype).reshape(-1, 24).split([HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = base.split([HC, HC, HC * HC])
    pre = torch.sigmoid(pre_w * s0 + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * s1 + post_b)
    comb = torch.softmax(comb_w.view(-1, HC, HC) * s2 + comb_b.view(HC, HC), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _fp64_faithfulness(row: dict, got: torch.Tensor, ref32: torch.Tensor, ref64: torch.Tensor, term_scale=None):
    """fp32 ulp distance to the fp64 evaluation for the op and for the fp32 reference arithmetic itself."""
    e_tt = _ulp_err(got.reshape(ref64.shape), ref64, torch.float32, term_scale)
    e_32 = _ulp_err(ref32.reshape(ref64.shape), ref64, torch.float32, term_scale)
    row.update(
        f64_tt_max_ulp=e_tt.max().item(),
        f64_tt_mean_ulp=e_tt.mean().item(),
        f64_ref32_max_ulp=e_32.max().item(),
        f64_ref32_mean_ulp=e_32.mean().item(),
        tt_vs_ref32_max_ulp=_ulp_err(
            got.reshape(ref64.shape), ref32.reshape(ref64.shape).double(), torch.float32, term_scale
        )
        .max()
        .item(),
    )


def _run_fused_hc(device, hc: HyperConnection, streams: torch.Tensor, layer, sub, mixes_src, tt_dtype, torch_dtype):
    b = streams.shape[0]
    streams_tt = _to_tt(streams, device, tt_dtype)
    if mixes_src == "host_fp32":
        mixes_tt = _to_tt(_mixes(hc, streams), device, tt_dtype)
    else:
        mixes_tt = _device_mixes(device, streams_tt, layer, sub, tt_dtype)
    base = hc.base.detach()
    scale = [float(s) for s in hc.scale.detach()]
    post_tt, comb_tt, collapsed_tt = ttnn.experimental.deepseek.fused_hyperconnection(
        streams_tt,
        fused_w=mixes_tt,
        pre_bias=_to_tt(base[:HC].reshape(1, 1, 1, HC), device, tt_dtype),
        post_bias=_to_tt(base[HC : 2 * HC].reshape(1, 1, 1, HC), device, tt_dtype),
        comb_bias=_to_tt(base[2 * HC :].reshape(1, 1, 1, HC * HC), device, tt_dtype),
        num_streams=HC,
        sinkhorn_iters=ITERS,
        pre_scale=scale[0],
        post_scale=scale[1],
        comb_scale=scale[2],
        eps=CFG.hc_eps,
    )
    assert list(post_tt.shape) == [b, 1, HC, 1]
    assert list(comb_tt.shape) == [b, 1, HC, HC]
    assert list(collapsed_tt.shape) == [b, 1, 1, HIDDEN]
    mixes_in = ttnn.to_torch(mixes_tt).float().reshape(1, 1, b, 24)
    return post_tt, comb_tt, collapsed_tt, mixes_in


FUSED_HC_CASES = [(0, "attn"), (0, "ffn"), (3, "attn"), (43, "attn")]


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("mixes_src", ["host_fp32", "device"])
@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("layer, sub", FUSED_HC_CASES, ids=[f"L{l:02d}-{s}" for l, s in FUSED_HC_CASES])
def test_glm53_fused_hyperconnection(device, layer, sub, batch, mixes_src, dtype):
    tt_dtype, torch_dtype = TT_DTYPES[dtype]
    hc = _ref_hc(layer, sub)
    streams = _streams(layer, sub, batch)
    with torch.no_grad():
        _, ref_post, ref_comb = hc.weights(streams)
        _, _, ref_collapsed = hc(streams)

    post_tt, comb_tt, collapsed_tt, mixes_in = _run_fused_hc(
        device, hc, streams, layer, sub, mixes_src, tt_dtype, torch_dtype
    )
    got_post = ttnn.to_torch(post_tt).float()
    got_comb = ttnn.to_torch(comb_tt).float()
    got_collapsed = ttnn.to_torch(collapsed_tt).float()

    with torch.no_grad():
        emu_pre, emu_post, emu_comb = _coeff_math(hc, mixes_in, torch_dtype)
        _, f64_post, f64_comb = _coeff_math(hc, mixes_in, torch_dtype, torch.float64)
        streams_in = streams.to(torch_dtype).float()
        emu_collapsed = (emu_pre.reshape(batch, 1, HC, 1) * streams_in).sum(dim=2, keepdim=True)
        collapsed_terms = (emu_pre.reshape(batch, 1, HC, 1) * streams_in).abs().sum(dim=2, keepdim=True)

    base = dict(
        op="fused_hyperconnection",
        layer=layer,
        sublayer=sub,
        B=batch,
        dtype=dtype,
        mixes_src=mixes_src,
        hc=HC,
        hidden=HIDDEN,
        sinkhorn_iters=ITERS,
    )
    rows = []
    _check(rows, "post", got_post, ref_post.reshape(batch, 1, HC, 1), emu_post, torch_dtype, base)
    _check(rows, "comb", got_comb, ref_comb.reshape(batch, 1, HC, HC), emu_comb, torch_dtype, base)
    if dtype == "fp32":
        _fp64_faithfulness(rows[0], got_post, emu_post, f64_post)
        _fp64_faithfulness(rows[1], got_comb, emu_comb, f64_comb)
    _check(
        rows,
        "collapsed",
        got_collapsed,
        ref_collapsed.reshape(batch, 1, 1, HIDDEN),
        emu_collapsed,
        torch_dtype,
        base,
        term_scale=collapsed_terms,
    )
    for r in rows:
        _record(r)
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in ("output", "pcc", "max_abs", "faithful_max_ulp", "f64_tt_max_ulp", "f64_ref32_max_ulp")
                }
            )
        )
    bad = [(r["output"], r["pcc"]) for r in rows if not r["pcc"] >= PCC_BAR]
    assert not bad, f"PCC < {PCC_BAR}: {bad}"
    if dtype == "fp32":
        # fp32 coefficient path: as close to the exact (fp64) value as the fp32 reference arithmetic itself.
        for r in rows[:2]:
            assert (
                r["f64_tt_max_ulp"] <= 2 * r["f64_ref32_max_ulp"] + 4
            ), f"{r['output']}: {r['f64_tt_max_ulp']:.1f} ulp from fp64 vs reference fp32 {r['f64_ref32_max_ulp']:.1f}"


MIX_CASES = [(0, "attn"), (0, "ffn")]


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("coeff_src", ["reference", "device_chain"])
@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("layer, sub", MIX_CASES, ids=[f"L{l:02d}-{s}" for l, s in MIX_CASES])
def test_glm53_mix_streams(device, layer, sub, batch, coeff_src, dtype):
    tt_dtype, torch_dtype = TT_DTYPES[dtype]
    hc = _ref_hc(layer, sub)
    streams = _streams(layer, sub, batch)
    y = _sublayer_out(sub, batch)
    with torch.no_grad():
        _, ref_post, ref_comb = hc.weights(streams)
        ref = hc_expand(y.reshape(batch, 1, HIDDEN), streams, ref_post, ref_comb)  # [B, 1, 4, 4096]

    if coeff_src == "reference":
        post_tt = _to_tt(ref_post.reshape(batch, 1, HC, 1), device, tt_dtype)
        comb_tt = _to_tt(ref_comb.reshape(batch, 1, HC, HC), device, tt_dtype)
    else:
        post_tt, comb_tt, _, _ = _run_fused_hc(device, hc, streams, layer, sub, "host_fp32", tt_dtype, torch_dtype)
    streams_tt = _to_tt(streams, device, tt_dtype)
    # GLM's sublayer outputs are bf16 in both modes (Stage-1 precision policy); the fp32 kernel takes them as is.
    y_tt = _to_tt(y, device, ttnn.bfloat16)
    out_tt = ttnn.experimental.deepseek.mix_streams(post_tt, comb_tt, y_tt, streams_tt)
    assert list(out_tt.shape) == [batch, 1, HC, HIDDEN]
    got = ttnn.to_torch(out_tt).float()

    with torch.no_grad():
        p_in = ttnn.to_torch(post_tt).float().reshape(batch, 1, HC)
        c_in = ttnn.to_torch(comb_tt).float().reshape(batch, 1, HC, HC)
        y_in = y.to(torch.bfloat16).float().reshape(batch, 1, HIDDEN)
        s_in = streams.to(torch_dtype).float()
        emu = hc_expand(y_in, s_in, p_in, c_in)
        emu64 = hc_expand(y_in.double(), s_in.double(), p_in.double(), c_in.double())
        terms = hc_expand(y_in.abs(), s_in.abs(), p_in.abs(), c_in.abs())

    base = dict(
        op="mix_streams",
        layer=layer,
        sublayer=sub,
        B=batch,
        dtype=dtype,
        coeff_src=coeff_src,
        hc=HC,
        hidden=HIDDEN,
        sinkhorn_iters=ITERS,
    )
    rows = []
    _check(rows, "new_streams", got, ref, emu, torch_dtype, base, term_scale=terms)
    r = rows[0]
    if dtype == "fp32":
        _fp64_faithfulness(r, got, emu, emu64, term_scale=terms)
    _record(r)
    print(json.dumps({k: r[k] for k in ("output", "pcc", "max_abs", "faithful_max_ulp", "faithful_exact_frac")}))
    assert r["pcc"] >= PCC_BAR, f"PCC {r['pcc']} < {PCC_BAR}"
    if dtype == "bf16":
        # bf16 kernel: fp32 DST accumulation of the bf16 inputs, rounded once to bf16 -> within 1 bf16 ulp (at the scale
        # of the summed terms) of fp32 math on the same inputs.
        assert r["faithful_max_ulp"] <= 1, f"{r['faithful_max_ulp']} bf16 ulp from fp32 math"
    else:
        # fp32 kernel: as close to the exact (fp64) result as the fp32 reference arithmetic itself, at the scale of the
        # summed terms (the kernel accumulates more, smaller partial products in another order).
        assert (
            r["f64_tt_max_ulp"] <= 2 * r["f64_ref32_max_ulp"] + 4
        ), f"{r['f64_tt_max_ulp']:.1f} ulp from fp64 vs reference fp32 {r['f64_ref32_max_ulp']:.1f}"
