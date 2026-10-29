# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CI-style test with synthetic weights: no checkpoint weights, no weight cache and no goldens. Inputs are committed
files only: tests/weight_stats.json, tests/data/config.json (the model's HF config.json, vendored) and the fp32
reference package `reference/` (the program-layout link to the shared reference implementation).

Weights: deterministic normal(mean, std) per tensor with the real names/shapes/dtypes recorded from the real
checkpoint in tests/weight_stats.json (tests/weight_stats.py). They go through the same `WeightSource` boundary and dtype
map as the real weights (`TorchStateDictSource` semantics: setup-time host->device conversion); routed experts are
generated and packed per expert. The CPU reference is the vendored fp32 `reference.DecoderLayer` loaded with the *exact
device values* (bfp4/bfp8 quantized), so the comparison isolates the TTNN implementation from weight quantization.
Activations: Gaussian streams with the per-layer scale of the real hidden states (`streams_std` below).

Per layer kind (real config shapes): paged prefill of 200 tokens (non-aligned, padded tail) + one traced decode step at
position 200, PCC >= 0.995 vs the reference.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.reference.modules import KDAState, MLAState
from models.autoports.zai_org_glm_5_3_flash.reference.weights import build_layer
from models.autoports.zai_org_glm_5_3_flash.tests.common import PCC_BAR, check, int_tensor, open_mesh, pcc, record
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV, load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import FunctionalDecoder
from models.autoports.zai_org_glm_5_3_flash.tt.weights import WeightSource, cache_dtype

STATS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight_stats.json")))
# typical RMS of the 4 x 4096 residual streams entering these layers (real activations: 0.01-1)
STREAMS_STD = {"dense_kda": 0.02, "sparse_kda": 0.2, "sparse_dsa": 0.1}
T = 200


def _gen(name: str, shape, mean: float, std: float) -> torch.Tensor:
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * std + mean


class SyntheticSource(WeightSource):
    """Deterministic synthetic tensors in the tile-cache layout; remembers the device tensors it created."""

    def __init__(self, kind: str):
        self.stats = STATS["layers"][kind]["tensors"]
        self.made: dict[str, ttnn.Tensor] = {}
        self.exact: dict[str, torch.Tensor] = {}  # device values read back at creation (bf16 holds bfp8/bfp4 exactly)

    def _host(self, name):
        key = name if name in self.stats else "mlp.experts.*." + name.split(".")[-1]
        s = self.stats[key]
        t = _gen(name, s["shape"], s["mean"], s["std"])
        if name.endswith("A_log") or name.endswith("e_score_correction_bias") or "_base" in name or "_scale" in name:
            return t.float()
        return t.to(torch.bfloat16).float()

    def device(self, layer_idx, name, mesh_device):
        dt = cache_dtype(name)
        t = ttnn.from_torch(
            self._host(name).to(torch.float32 if dt == ttnn.float32 else torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.made[name] = t
        if ".experts." not in name:
            self.exact[name] = ttnn.to_torch(t).to(torch.float32 if dt == ttnn.float32 else torch.bfloat16)
        return t

    def host(self, layer_idx, name):
        return self._host(name)

    def experts(self, layer_idx, name, num_experts, mesh_device):
        parts = []
        for e in range(num_experts):
            t = self.device(layer_idx, f"mlp.experts.{e}.{name}", mesh_device)
            parts.append(ttnn.reshape(t, (1, 1, t.shape[-2], t.shape[-1])))
        stacked = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for e in range(num_experts):
            ttnn.deallocate(self.made.pop(f"mlp.experts.{e}.{name}"))
        self.made[f"mlp.experts.{name}"] = stacked
        return stacked


class DeviceExpertProvider:
    """Reference expert weights read back per expert from the stacked device tensors (exact bfp4 values)."""

    def __init__(self, gate_up: ttnn.Tensor, down: ttnn.Tensor):
        self.gate_up, self.down = gate_up, down

    def get(self, e):
        gu = ttnn.to_torch(ttnn.slice(self.gate_up, (0, e, 0, 0), (1, e + 1, 4096, 4096))).float()[0, 0]
        dn = ttnn.to_torch(ttnn.slice(self.down, (0, e, 0, 0), (1, e + 1, 2048, 4096))).float()[0, 0]
        return gu.t().contiguous(), dn.t().contiguous()  # [2I, H], [H, I]


def _config():
    """The model's HF config (vendored copy): real shapes and layer kinds without the HF snapshot."""
    return load_hf_text_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "config.json"))


def reference_state_dict(src: SyntheticSource, cfg, kind: str) -> dict:
    """Device values (cache layout) -> reference/HF names (inverse of weights.reference_to_cache_layout)."""
    sd = {}
    for name, s in src.stats.items():
        if name.startswith("mlp.experts."):
            continue
        v = (src.exact[name] if name in src.exact else src._host(name)).float().reshape(s["shape"])
        if name.startswith("hc_"):
            _, which, leaf = name.split("_")
            sd[f"{which}_hc.{leaf}"] = v.t().contiguous() if leaf == "fn" else v
        elif name in ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"):
            continue
        elif name in ("self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight"):
            sd["self_attn.forget_gate." + name[len("self_attn.") :]] = v.t().contiguous()
        elif name in ("self_attn.dt_bias", "self_attn.A_log"):
            sd["self_attn.forget_gate." + name[len("self_attn.") :]] = v
        elif name == "self_attn.kv_b_proj.W_uk":
            uv = src.exact["self_attn.kv_b_proj.W_uv"].float().reshape(src.stats["self_attn.kv_b_proj.W_uv"]["shape"])
            sd["self_attn.kv_b_proj.weight"] = torch.cat([v, uv.mT], dim=1).reshape(-1, cfg.kv_lora_rank)
        elif name == "self_attn.kv_b_proj.W_uv":
            continue
        elif len(s["shape"]) == 2 and name.endswith(".weight"):
            sd[name] = v.t().contiguous()
        else:
            sd[name] = v
    if "self_attn.q_conv1d.weight" in src.stats:
        taps = [src.exact[f"self_attn.{p}_conv1d.weight"].float().reshape(4, -1) for p in "qkv"]
        sd["self_attn.conv1d.weight"] = torch.cat(taps, dim=1).t().unsqueeze(1).contiguous()
    if kind == "sparse_dsa":  # indexer weights do not affect the output for kv_len <= 2051 (selection = all tokens)
        g = torch.Generator().manual_seed(0)
        sd.update(
            {
                "self_attn.indexer.wq_b.weight": torch.randn(32 * 128, cfg.q_lora_rank, generator=g) * 0.02,
                "self_attn.indexer.wk.weight": torch.randn(128, cfg.hidden_size, generator=g) * 0.02,
                "self_attn.indexer.k_norm.weight": torch.ones(128),
                "self_attn.indexer.k_norm.bias": torch.zeros(128),
                "self_attn.indexer.weights_proj.weight": torch.randn(32, cfg.hidden_size, generator=g) * 0.02,
                "self_attn.indexer.index_kpool_compress_gate": torch.randn(128, cfg.hidden_size, generator=g) * 0.02,
                "self_attn.indexer.index_kpool_compress_ape": torch.randn(4, 128, generator=g) * 0.02,
            }
        )
    return sd


@pytest.mark.parametrize("kind", ["dense_kda", "sparse_kda", "sparse_dsa"])
def test_synthetic_weights(kind):
    torch.set_num_threads(16)
    cfg = _config()
    layer = STATS["layers"][kind]["layer"]
    mesh = open_mesh()
    try:
        src = SyntheticSource(kind)
        dec = FunctionalDecoder.from_weight_cache(hf_config=cfg, layer_idx=layer, mesh_device=mesh, source=src)
        sd = reference_state_dict(src, cfg, kind)
        provider = None
        if dec.spec.mlp == "moe":
            provider = DeviceExpertProvider(src.made["mlp.experts.gate_up_proj"], src.made["mlp.experts.down_proj"])
        ref = build_layer(cfg, layer, sd, provider, dtype=torch.float32, strict=False)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1, T + 1, 4, 4096, generator=g) * STREAMS_STD[kind]
        state = KDAState() if dec.is_kda else MLAState()
        with torch.no_grad():
            ref_pre, _ = ref(x[:, :T], state=state)
            ref_dec, _ = ref(x[:, T : T + 1], state=state)
        cache = dec.allocate_cache(max_batch=1, max_seq_len=T + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
        pt = None if dec.is_kda else dec.make_page_table(cache)
        y = dec.prefill_forward(x[:, :T], cache=cache, user_slot=0).reshape(ref_pre.shape)
        streams = ttnn.from_torch(
            x[:, T : T + 1].reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        cur = int_tensor([T], mesh)
        snap = dec.snapshot_state(cache)
        dec.decode_forward(streams, cache=cache, page_table=pt, cur_pos=cur)
        dec.restore_state(cache, snap)
        trace = dec.capture_decode_trace(streams, cache=cache, page_table=pt, cur_pos=cur)
        yd = ttnn.to_torch(trace.replay(blocking=True)).float().reshape(ref_dec.shape)
        trace.release()
        if kind == "dense_kda":  # the HF-state-dict boundary builds the same device weights -> identical output
            dec2 = FunctionalDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
            cache2 = dec2.allocate_cache(max_batch=1, max_seq_len=T + 64)
            y2 = dec2.prefill_forward(x[:, :T], cache=cache2, user_slot=0).reshape(ref_pre.shape)
            same = bool(torch.equal(y, y2))
            record(dict(layer_kind=kind, layer=layer, from_state_dict_bitwise_equal=same), name="pcc_synthetic.jsonl")
            check([] if same else [kind], "from_state_dict output differs from the WeightSource path")
        rows = [
            dict(layer_kind=kind, layer=layer, mode="prefill", seq_len=T, pcc=pcc(y, ref_pre), weights="synthetic"),
            dict(
                layer_kind=kind,
                layer=layer,
                mode="decode",
                seq_len=T + 1,
                pcc=pcc(yd, ref_dec),
                weights="synthetic",
                trace=True,
            ),
        ]
        for r in rows:
            record(r, name="pcc_synthetic.jsonl")
        check([r for r in rows if not r["pcc"] >= PCC_BAR], "synthetic-weight PCC below the bar")
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.parametrize("kind", ["dense_kda", "sparse_kda", "sparse_dsa"])
def test_state_dict_layout_roundtrip(kind):
    """Host-only: HF/reference naming -> tile-cache layout (weights.reference_to_cache_layout) is the exact inverse of
    the cache -> reference mapping used above, for every tensor of the layer kind (routed experts on a 2-expert stack).
    """
    from models.autoports.zai_org_glm_5_3_flash.tt.weights import reference_to_cache_layout

    cfg = _config()
    src = SyntheticSource(kind)
    src.exact = {n: src._host(n) for n in src.stats if not n.startswith("mlp.experts.")}
    sd = reference_state_dict(src, cfg, kind)
    if "mlp.experts.*.gate_up_proj" in src.stats:
        gu = torch.stack([src._host(f"mlp.experts.{e}.gate_up_proj") for e in range(2)])  # cache [E, H, 2I]
        dn = torch.stack([src._host(f"mlp.experts.{e}.down_proj") for e in range(2)])  # cache [E, I, H]
        sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = gu.mT, dn.mT  # HF [E, 2I, H], [E, H, I]
    back = reference_to_cache_layout(sd, cfg)
    bad = []
    for name in src.stats:
        if name.startswith("mlp.experts."):
            leaf = name.split(".")[-1]
            want = torch.stack([src._host(f"mlp.experts.{e}.{leaf}") for e in range(2)])
            got = back[f"mlp.experts.{leaf}"]
        else:
            want, got = src.exact[name].float(), back[name].float()
        if tuple(got.shape) != tuple(want.shape) or not torch.equal(got, want.reshape(got.shape)):
            bad.append(name)
    check(bad, "state-dict layout round trip mismatch")
