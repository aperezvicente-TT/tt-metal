# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`from_state_dict` with real checkpoint tensors (dense_kda layer 0: no routed experts, so the host state dict fits in
a few GB). The layer is built twice, from the HF snapshot's state dict (fp8 block-scale weights dequantised to fp32 by
the reference loader, converted to the cache layout and dtype map by `TorchStateDictSource`) and from the pre-packed tile
cache; both run the golden prompt-512 prefill and a traced decode step. Checks: each path vs the fp32 golden
(PCC >= 0.995) and the two paths against each other (they may differ in the last bit of the bfp8 packing: fp32 ->
bf16 -> bfp8 here, fp32 -> bfp8 in the cache writer).
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    PCC_BAR,
    SNAPSHOT,
    check,
    golden,
    hf_config,
    int_tensor,
    load_decoder,
    open_mesh,
    pcc,
    record,
    streams4,
    to_dev,
    to_host,
)
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import FunctionalDecoder


def _run(dec, g, mesh, n):
    cache = dec.allocate_cache(max_batch=1, max_seq_len=n + 64)
    y = streams4(dec.prefill_forward(g["streams_in"][:, :n], cache=cache, user_slot=0))
    streams, cur = to_dev(g["streams_in"][:, n : n + 1], mesh), int_tensor([n], mesh)
    snap = dec.snapshot_state(cache)
    dec.decode_forward(streams, cache=cache, cur_pos=cur)
    dec.restore_state(cache, snap)
    trace = dec.capture_decode_trace(streams, cache=cache, cur_pos=cur)
    out = to_host(trace.replay(blocking=True))
    trace.release()
    return y, out


@pytest.mark.parametrize("layer", [0])
def test_from_state_dict_real_weights(layer):
    from models.autoports.zai_org_glm_5_3_flash.reference.weights import ShardReader, load_layer

    cfg = hf_config()
    sd, _ = load_layer(ShardReader(SNAPSHOT), layer, dtype=torch.float32, experts="skip", cfg=cfg)
    mesh = open_mesh()
    try:
        g, n = golden(layer, 512), 511
        y_sd, d_sd = _run(
            FunctionalDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh), g, mesh, n
        )
        y_c, d_c = _run(load_decoder(cfg, layer, mesh), g, mesh, n)
        row = dict(
            layer=layer,
            prompt=512,
            prefill_tokens=n,
            snapshot=os.path.basename(SNAPSHOT.rstrip("/")),
            prefill_pcc_state_dict_vs_golden=pcc(y_sd, g["streams_out"][:, :n]),
            prefill_pcc_cache_vs_golden=pcc(y_c, g["streams_out"][:, :n]),
            prefill_pcc_state_dict_vs_cache=pcc(y_sd, y_c),
            prefill_bitwise_equal=bool(torch.equal(y_sd, y_c)),
            decode_pcc_state_dict_vs_golden=pcc(d_sd, g["streams_out"][:, n : n + 1]),
            decode_pcc_state_dict_vs_cache=pcc(d_sd, d_c),
            decode_bitwise_equal=bool(torch.equal(d_sd, d_c)),
            weights="real (HF snapshot state dict)",
            trace=True,
        )
        record(row, name="pcc_state_dict_real.jsonl")
        bad = [k for k in ("prefill_pcc_state_dict_vs_golden", "decode_pcc_state_dict_vs_golden") if row[k] < PCC_BAR]
        check([row] if bad else [], "from_state_dict with real weights")
    finally:
        ttnn.close_mesh_device(mesh)
