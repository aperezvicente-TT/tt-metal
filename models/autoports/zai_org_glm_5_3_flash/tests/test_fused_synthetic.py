# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CI-style synthetic-weight test of the FUSED decoder (no checkpoint weights, no weight cache, no goldens): Stage 1's
`test_synthetic.py` protocol (deterministic per-tensor normal weights with the real names/shapes/dtypes from
`tests/weight_stats.json`, fp32 reference `reference.DecoderLayer` loaded with the exact device values) run through
`FusedDecoder`: paged prefill of 200 tokens (non-aligned) + one traced decode step, PCC >= 0.995 per layer kind. The
routed experts go through the fused_experts weight contract (per-expert DRAM ND-sharded, gate_up tile-permuted); the
reference reads them back per expert and undoes the permutation. Also checks `FusedDecoder.from_state_dict` against the
WeightSource path bitwise (dense_kda)."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.reference.modules import KDAState, MLAState
from models.autoports.zai_org_glm_5_3_flash.reference.weights import build_layer
from models.autoports.zai_org_glm_5_3_flash.tests import test_synthetic as s1
from models.autoports.zai_org_glm_5_3_flash.tests.common import PCC_BAR, check, int_tensor, pcc, record
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder, gate_up_column_perm

T = s1.T


class FusedExpertProvider:
    """Reference expert weights read back per expert from the fused_experts device tensors (exact bfp4 values)."""

    def __init__(self, gate_up: list, down: list):
        self.gate_up, self.down = gate_up, down
        two_i = gate_up[0].shape[-1]
        self.inv = torch.argsort(gate_up_column_perm(two_i))

    def get(self, e):
        gu = ttnn.to_torch(self.gate_up[e]).float()[..., self.inv]  # [H, 2I] = [gate | up]
        dn = ttnn.to_torch(self.down[e]).float()  # [I, H]
        return gu.reshape(gu.shape[-2], -1).t().contiguous(), dn.reshape(dn.shape[-2], -1).t().contiguous()


@pytest.mark.parametrize("kind", ["dense_kda", "sparse_kda", "sparse_dsa"])
def test_fused_synthetic_weights(kind):
    torch.set_num_threads(16)
    cfg = s1._config()
    layer = s1.STATS["layers"][kind]["layer"]
    mesh = open_mesh(trace_region_size=200_000_000)
    try:
        src = s1.SyntheticSource(kind)
        dec = FusedDecoder.from_weight_cache(hf_config=cfg, layer_idx=layer, mesh_device=mesh, source=src)
        sd = s1.reference_state_dict(src, cfg, kind)
        provider = FusedExpertProvider(dec.layer.mlp.gate_up, dec.layer.mlp.down) if dec.spec.mlp == "moe" else None
        ref = build_layer(cfg, layer, sd, provider, dtype=torch.float32, strict=False)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(1, T + 1, 4, 4096, generator=g) * s1.STREAMS_STD[kind]
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
            dec2 = FusedDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
            cache2 = dec2.allocate_cache(max_batch=1, max_seq_len=T + 64)
            y2 = dec2.prefill_forward(x[:, :T], cache=cache2, user_slot=0).reshape(ref_pre.shape)
            same = bool(torch.equal(y, y2))
            record(
                dict(impl="fused", layer_kind=kind, layer=layer, from_state_dict_bitwise_equal=same),
                name="pcc_synthetic.jsonl",
            )
            check([] if same else [kind], "from_state_dict output differs from the WeightSource path")
        rows = [
            dict(
                impl="fused",
                layer_kind=kind,
                layer=layer,
                mode="prefill",
                seq_len=T,
                pcc=pcc(y, ref_pre),
                weights="synthetic",
            ),
            dict(
                impl="fused",
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
