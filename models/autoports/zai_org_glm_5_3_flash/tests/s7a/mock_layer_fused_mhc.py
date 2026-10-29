# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mock-cluster check of a whole optimized decoder layer with the Stage 7a fused programs on (synthetic weights,
dense_kda layer 0 by default): decode at every row-count path (1, 2, 4 fused / stacked; 8, 32 stream-major) builds,
allocates (CB vs L1 tensors) and runs; the multichip half-layer forward_unit path too. Readback is garbage (mock)."""

import os
import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_synthetic as s1
from models.autoports.zai_org_glm_5_3_flash.tests.common import int_tensor
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoder, OptimizedDecoderConfig

KIND = os.environ.get("GLM53_MOCK_KIND", "dense_kda")
OCFG = dict(
    fused_mhc_pre=True,
    fused_kda_pre=os.environ.get("GLM53_MOCK_KDA", "1") == "1",
    fused_router_tail=True,
    fused_moe_post=True,
    fused_kda_rec=os.environ.get("GLM53_MOCK_REC", "1") == "1",
)


def main():
    cfg = s1._config()
    layer = s1.STATS["layers"][KIND]["layer"]
    mesh = ttnn.open_device(device_id=0)
    try:
        src = s1.SyntheticSource(KIND)
        ocfg = OptimizedDecoderConfig(**OCFG)
        dec = OptimizedDecoder.from_weight_cache(
            hf_config=cfg, layer_idx=layer, mesh_device=mesh, source=src, optimized_config=ocfg
        )
        print("loaded", KIND, layer, flush=True)
        for b in (1, 2, 4, 8, 32):
            t0 = time.time()
            cache = dec.allocate_cache(max_batch=b, max_seq_len=256 if dec.is_kda else 2051)
            pt = None if dec.is_kda else dec.make_page_table(cache)
            x = ttnn.from_torch(
                torch.randn(1, 1, b, 16384) * 0.5, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            cur = int_tensor(list(range(100, 100 + b)), mesh)
            y = dec.decode_forward(x, cache=cache, page_table=pt, cur_pos=cur)
            ttnn.synchronize_device(mesh)
            print(f"OK decode B={b} out={tuple(y.shape)} {time.time() - t0:.1f}s", flush=True)
            if os.environ.get("GLM53_MOCK_TRACE", "0") == "1":
                tr = dec.capture_decode_trace(x, cache=cache, page_table=pt, cur_pos=cur)
                tr.replay(blocking=True)
                tr.release()
                print(f"OK trace B={b}", flush=True)
            # the multichip half-layer units (tt/multichip_decoder.forward_unit: attn unit, then ffn unit) on the same
            # layer: the hyperconnection pre / post hooks (FusedMhcBinding.run / post_hook) instead of the layer forward
            from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import decode_attention, forward_unit

            mp = getattr(dec.layer, "mhc_post", None)
            c0 = mp.calls if mp is not None else 0
            u1 = forward_unit(dec, x, decode_attention(dec, cache, pt, cur), decode=True, part="attn")
            y2 = forward_unit(dec, u1, None, decode=True, part="ffn")
            ttnn.synchronize_device(mesh)
            hooked = (mp.calls - c0) if mp is not None else 0
            print(
                f"OK units B={b} s1={tuple(u1.shape)} out={tuple(y2.shape)} mhc_post via post_hook: {hooked}",
                flush=True,
            )
            del y, cache
        # every enabled fused program actually ran (the flags route through the layer's hooks)
        lay = dec.layer
        calls = lambda o, n: getattr(getattr(o, n, None), "calls", None)
        got = dict(mhc_post=calls(lay, "mhc_post"), kda_pre=calls(lay.attn, "kda_pre"),
                   router_tail=calls(getattr(lay.mlp, "router", None), "tail"),
                   kda_rec=calls(lay.attn, "kda_rec"))  # fmt: skip
        print("CALLS", got, flush=True)
        assert not OCFG["fused_moe_post"] or got["mhc_post"], got
        assert not (OCFG["fused_kda_pre"] and dec.is_kda) or got["kda_pre"], got
        assert not (OCFG["fused_router_tail"] and "sparse" in KIND) or got["router_tail"], got
        assert not (OCFG["fused_kda_rec"] and dec.is_kda) or got["kda_rec"], got
        print("MOCK_LAYER_FUSED_MHC_OK")
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
