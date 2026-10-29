# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the optimized-decoder tests (Stage 3). Reuses the Stage 1 / Stage 2 test helpers; evidence rows go to
$GLM53_STAGE_EVIDENCE (the Stage 3 evidence dir when run through doc/ttnn/03-optimized-decoder/scripts/dev.sh).

`GLM53_DECODER_IMPL` selects the implementation a shared harness loads: `optimized` (default here), `fused` (Stage 2
baseline) or `functional` (Stage 1). `GLM53_OCFG="k=v,k=v"` overrides OptimizedDecoderConfig fields (variant runs of the
shared harnesses: A/B candidates, Tracy controls)."""

from __future__ import annotations

import os

from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST

IMPL = os.environ.get("GLM53_DECODER_IMPL", "optimized")


def _parse(v: str):
    if v in ("True", "False"):
        return v == "True"
    if v == "None":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def ocfg_from_env(config=None, env: str = "GLM53_OCFG"):
    """OptimizedDecoderConfig with `GLM53_OCFG="k=v,k=v"` overrides (';' separates list items inside a value)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig

    cfg = config or OptimizedDecoderConfig()
    for item in filter(None, os.environ.get(env, "").split(",")):
        k, v = item.split("=", 1)
        assert hasattr(cfg, k), f"OptimizedDecoderConfig has no field {k!r}"
        setattr(cfg, k, _parse(v))
    return cfg


def load_optimized_decoder(cfg, layer: int, mesh, config=None):
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoder

    return OptimizedDecoder.from_weight_cache(
        hf_config=cfg,
        layer_idx=layer,
        mesh_device=mesh,
        cache_dir=CACHE_DIR,
        manifest_path=MANIFEST,
        optimized_config=ocfg_from_env(config),
    )


def load_impl(cfg, layer: int, mesh, impl: str | None = None, config=None):
    impl = impl or IMPL
    if impl == "optimized":
        return load_optimized_decoder(cfg, layer, mesh, config)
    from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_impl as load_stage2

    return load_stage2(cfg, layer, mesh, impl=impl)


def open_mesh(trace_region_size: int = 0):
    """1x1 mesh on device 0 of the visible chips (card3 under tt-lock run card3: chip 6)."""
    import ttnn

    return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=trace_region_size)
