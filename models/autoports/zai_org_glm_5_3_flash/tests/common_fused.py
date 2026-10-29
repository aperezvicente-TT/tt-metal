# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the fused-decoder tests (Stage 2). Reuses the Stage 1 test helpers (`tests/common.py`); evidence rows
go to $GLM53_STAGE_EVIDENCE (the Stage 2 evidence dir when run through doc/ttnn/02-fused-decoder/scripts/dev.sh).

`GLM53_DECODER_IMPL` selects the implementation a shared harness loads: `fused` (default) or `functional` (Stage 1
baseline, for like-for-like before/after rows)."""

from __future__ import annotations

import os

from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST

IMPL = os.environ.get("GLM53_DECODER_IMPL", "fused")


def fcfg_from_env(fused_config=None):
    """FusedDecoderConfig with `GLM53_FCFG="k=v,k=v"` overrides (variant runs of the shared harnesses, e.g. the
    composite-mHC control of the Tracy comparison)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoderConfig

    fcfg = fused_config or FusedDecoderConfig()
    for item in filter(None, os.environ.get("GLM53_FCFG", "").split(",")):
        k, v = item.split("=")
        assert hasattr(fcfg, k), k
        setattr(fcfg, k, {"True": True, "False": False}.get(v, int(v) if v.lstrip("-").isdigit() else v))
    return fcfg


def load_fused_decoder(cfg, layer: int, mesh, fused_config=None):
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder

    fused_config = fcfg_from_env(fused_config)

    return FusedDecoder.from_weight_cache(
        hf_config=cfg,
        layer_idx=layer,
        mesh_device=mesh,
        cache_dir=CACHE_DIR,
        manifest_path=MANIFEST,
        fused_config=fused_config,
    )


def load_impl(cfg, layer: int, mesh, impl: str | None = None, fused_config=None):
    impl = impl or IMPL
    if impl == "fused":
        return load_fused_decoder(cfg, layer, mesh, fused_config)
    if impl == "functional":
        from models.autoports.zai_org_glm_5_3_flash.tests.common import load_decoder

        return load_decoder(cfg, layer, mesh)
    raise ValueError(f"unknown GLM53_DECODER_IMPL {impl!r}")


def kda_state_of(dec, cache, slot: int = 0):
    """Recurrent state [1,64,128,128] of one slot as a device tensor (fused: batched state; functional: per slot)."""
    rec = cache.kda.recurrent
    return rec[slot] if isinstance(rec, list) else cache.kda.recurrent_of(slot)


def kda_conv_rows_of(dec, cache, slot: int = 0):
    """[3, 24576] conv history rows t-3..t-1 of one slot (host torch)."""
    import ttnn

    conv = cache.kda.conv
    if isinstance(conv, list):
        return ttnn.to_torch(conv[slot]).float()[0]
    return cache.kda.conv_rows_of(slot)[0]


def open_mesh(trace_region_size: int = 0):
    """1x1 mesh on device 0 of the visible chips (card2: chip 4). The explicit device id also works against the mock
    cluster used to compile-check configs without hardware (GLM53_MOCK=1, doc/ttnn/02-fused-decoder/scripts/mock.sh)."""
    import ttnn

    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=trace_region_size)
