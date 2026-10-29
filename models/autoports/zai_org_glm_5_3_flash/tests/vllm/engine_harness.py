# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-process vLLM engine over the TT plugin with a fake mesh and the CPU mock generator (09a, host-only).

The real vLLM V1 engine core, TTScheduler, TTWorker, TTModelRunner and async-decode controller run in this
process (VLLM_ENABLE_V1_MULTIPROCESSING=0); only the worker's mesh open/close are replaced by a fake 8-device
mesh, and the adapter builds 06a's Glm53Generator over the CPU MockModel through GLM53_VLLM_GENERATOR_FACTORY
(glm53_mock_stack.py). No TT device is opened.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

MOCK_FACTORY = "models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack:build_mock_generator"
SNAPSHOT = os.environ.get(
    "GLM53_HF_SNAPSHOT",
    "/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/"
    "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
)


class FakeMesh:
    """What TTWorker / TTModelRunner read off the mesh: the device count and the submesh list."""

    def __init__(self, num_devices: int = 8):
        self._n = num_devices
        self.closed = False

    def get_num_devices(self) -> int:
        return self._n

    def get_submeshes(self):
        return []


@contextmanager
def fake_tt_mesh(num_devices: int = 8):
    import vllm_tt_plugin.worker as worker

    opened: list[FakeMesh] = []

    def open_mesh_device(tt_config, trace_mode, local_dp_rank=0, *, model_fabric_config=None):
        mesh = FakeMesh(num_devices)
        opened.append(mesh)
        return mesh

    def close_mesh_device(mesh_device, tt_config):
        mesh_device.closed = True
        marker = os.environ.get("GLM53_MOCK_MARKER")
        if marker:  # out-of-process tests check that shutdown closed the mesh in the engine process
            with open(marker, "a") as f:
                f.write(json.dumps({"event": "mesh_closed", "pid": os.getpid()}) + "\n")

    saved = worker.open_mesh_device, worker.close_mesh_device
    worker.open_mesh_device, worker.close_mesh_device = open_mesh_device, close_mesh_device
    try:
        yield opened
    finally:
        worker.open_mesh_device, worker.close_mesh_device = saved


def engine_env():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["GLM53_VLLM_GENERATOR_FACTORY"] = MOCK_FACTORY
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def make_llm(
    *, max_num_seqs=8, max_num_batched_tokens=512, max_model_len=None, async_scheduling=True, tt_config=None,
    enable_prefix_caching=False, **kwargs
):  # fmt: skip
    from vllm import LLM

    tt = {"sample_on_device_mode": "all", "trace_mode": "all"}
    tt.update(tt_config or {})
    return LLM(
        model=SNAPSHOT,
        tokenizer=SNAPSHOT,
        served_model_name="zai-org/GLM-5.3-Flash",
        max_model_len=max_model_len,
        block_size=64,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        async_scheduling=async_scheduling,
        additional_config={"tt": tt},
        **kwargs,
    )


def live_generator():
    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import REGISTRY

    return REGISTRY[-1]


def patch_connector_join(monkeypatch, hf_config):
    """the two KV-connector join shims of the engine mocks (test_lmcache_engine_mock): the CPU mock's one DSA layer
    stands in for vLLM's 11 MLA layers in the connector views; the TT worker's connector handshake metadata (join
    finding J6: vLLM's EngineCore asks every worker for it whenever a kv_transfer_config is set)"""
    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import RecordingMockModel
    from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import layer_kinds, text_config_of

    mla = [i for i, k in enumerate(layer_kinds(text_config_of(hf_config))) if k == "mla"]

    def latent_pages(self, kv_cache):
        lat = next(c.latent for l, c in zip(self.layers, kv_cache.caches) if not l.is_kda)
        return {i: lat for i in mla}

    monkeypatch.setattr(RecordingMockModel, "latent_pages", latent_pages)
    from vllm_tt_plugin.worker import TTWorker

    if not hasattr(TTWorker, "get_kv_connector_handshake_metadata"):

        def get_kv_connector_handshake_metadata(self):
            from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

            if not has_kv_transfer_group():
                return None
            md = get_kv_transfer_group().get_handshake_metadata()
            return None if md is None else {(0, 0): md}

        monkeypatch.setattr(TTWorker, "get_kv_connector_handshake_metadata", get_kv_connector_handshake_metadata,
                            raising=False)  # fmt: skip


def shared_connector_dirs(tmp_path_factory):
    """(pages dir, KDA checkpoint dir) shared by every engine of the test process: vLLM keeps KV-connector state
    process-wide, so a later engine's connector can still point at the first engine's directories; one shared pair
    keeps every engine consistent (tests use distinct prompts per case)"""
    root = tmp_path_factory.getbasetemp() / "kv_connector_shared"
    pages, kda = root / "pages", root / "kda"
    pages.mkdir(parents=True, exist_ok=True)
    kda.mkdir(parents=True, exist_ok=True)
    return pages, kda
