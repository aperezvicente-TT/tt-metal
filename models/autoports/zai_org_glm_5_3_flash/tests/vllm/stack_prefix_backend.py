# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM53_BACKEND_MODULE for serving rehearsals of the Stage 14b stack on a layer prefix (GLM53_STACK_LAYERS, e.g.
"0,1,2,3,4" = dense_kda L0-L2 + sparse_dsa L3 + sparse_kda L4): the same tt/stack_model.py backend with `layers=`, so a
2x4 mock-cluster vLLM server builds in minutes instead of loading all 45 layers. The tokens of a prefix are not the
model's; this module is for plumbing checks only, never for served numbers."""

import os

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import hostloop_probe, r1_probe
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model

LAYERS_ENV = "GLM53_STACK_LAYERS"
# KV-connector rehearsals only: stand-ins for the LMCache join's open gaps (doc/ttnn/09-vllm/lmcache_a1_review.md),
# applied in the EngineCore process when this module is imported (model build, before the connector starts)
JOIN_SHIMS_ENV = "GLM53_JOIN_SHIMS"


def _apply_join_shims() -> None:
    """J6: TTWorker.get_kv_connector_handshake_metadata (vLLM's EngineCore calls it with any kv_transfer_config; the
    GPU worker's single-rank body). J5: LMCache sizes the TT latent from the KV geometry (NUM_DSA_LAYERS, kv_dim 1,
    1 head x LMCACHE_HIDDEN_ELEMS, use_mla) instead of glm5_next's model_config (use_mla False, 45 layers, 64 x 64)."""
    from vllm_tt_plugin.worker import TTWorker

    if not hasattr(TTWorker, "get_kv_connector_handshake_metadata"):

        def get_kv_connector_handshake_metadata(self):
            from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

            if not has_kv_transfer_group():
                return None
            md = get_kv_transfer_group().get_handshake_metadata()
            return None if md is None else {(0, 0): md}

        TTWorker.get_kv_connector_handshake_metadata = get_kv_connector_handshake_metadata
    try:
        from lmcache.integration.vllm import utils as lm_utils
        from lmcache.integration.vllm import vllm_service_factory as lm_factory
        from vllm_tt_plugin.lmcache import page_contract as pc
    except ImportError:
        return
    orig = lm_factory.VllmServiceFactory.get_or_create_metadata
    if getattr(orig, "_glm53_shim", False):
        return

    def get_or_create_metadata(self):
        if self.metadata is not None:
            return self.metadata
        mc = self.vllm_config.model_config
        over = {
            "get_num_layers": lambda *a, **k: pc.NUM_DSA_LAYERS,
            "get_num_kv_heads": lambda *a, **k: 1,
            "get_head_size": lambda *a, **k: pc.LMCACHE_HIDDEN_ELEMS,
        }
        mla = lm_utils.mla_enabled
        for k, v in over.items():
            setattr(mc, k, v)
        lm_utils.mla_enabled = lambda model_config: True
        try:
            return orig(self)
        finally:
            lm_utils.mla_enabled = mla
            for k in over:
                mc.__dict__.pop(k, None)

    get_or_create_metadata._glm53_shim = True
    lm_factory.VllmServiceFactory.get_or_create_metadata = get_or_create_metadata

    # J9: LMCache's adapter keeps vllm_config.device_config.device as its torch device (slot_mapping.to(self.device)
    # in wait_for_save / start_load_kv); on TT that is the opened ttnn MeshDevice -> TypeError at the first save. The
    # TT connector moves pages through host buffers: its torch side is the CPU.
    import torch
    from lmcache.integration.vllm import vllm_v1_adapter as lm_adapter

    init = lm_adapter.LMCacheConnectorV1Impl.__init__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if not isinstance(self.device, (torch.device, str)):
            self.device = torch.device("cpu")

    lm_adapter.LMCacheConnectorV1Impl.__init__ = __init__


if os.environ.get(JOIN_SHIMS_ENV) == "1":
    _apply_join_shims()


def serving_fabric_config() -> dict:
    return stack_model.serving_fabric_config()


def build_model_backend(model_dir, mesh_device, **kw):
    spec = os.environ.get(LAYERS_ENV, "")
    if spec:
        kw["layers"] = [int(x) for x in spec.split(",") if x.strip()]
    r1_probe.install_from_env()  # GLM53_R1_PROBE=1: the R1 page-path timings, rehearsed
    hostloop_probe.install_from_env()  # GLM53_HOSTLOOP_PROBE=1: the per-step host-loop split
    be = stack_model.build_model_backend(model_dir, mesh_device, **kw)
    if spec:
        # KV-connector rehearsals: vLLM's KV spec names all 11 DSA layers; those outside the prefix alias the prefix's
        # first DSA layer's latent pages (plumbing only, like the rest of this module)
        real = be.latent_pages
        dsa = [L for L in range(be.cfg.num_hidden_layers) if stack_model.layer_kind(be.cfg, L) == "sparse_dsa"]

        def latent_pages(kv):
            pages = real(kv)
            first = pages[min(pages)]
            return {L: pages.get(L, first) for L in dsa}

        be.latent_pages = latent_pages
    return be
