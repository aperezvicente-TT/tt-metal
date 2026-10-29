# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KV-connector prefix hits end to end on the host (LMCache ladder R1's plumbing before the device rung): the real
vLLM 0.26 V1 engine (in-process) with the vllm-tt-plugin LMCache join branch (KV-connector host, scheduler-side
TTPageAlignedConnector, KDA checkpoint store) driving vLLM's ExampleConnector (page files; any latent width), this
adapter (kv_cache_connector_views / export_kda_state / import_kda_state) and Glm53Generator over the 06a CPU
MockModel. Request A stores its pages and, at its prompt's last page boundary (the policy splits the prefill there),
its slot's KDA checkpoint;
request B with the same prompt is a connector hit truncated to that boundary: pages loaded into B's blocks, the
checkpoint imported (Glm53Generator prefix_states), only the tail prefilled; B's tokens equal the reference greedy ones.

LMCache itself is not in this loop: on the real GLM-5.3-Flash config vLLM reports use_mla False / 45 layers / 64 x 64
heads, so LMCache's kv_shape is not the TT latent geometry (11, 1, chunk, 1, 544) and A2's connector refuses at engine
start (doc/ttnn/09-vllm/work_log.md, join finding J5); the CPU mock's latent is not 512 wide either.

Runs only with the join plugin (venv-lmcache or venv-vllm, PYTHONPATH=<vllm-tt-plugin-lmcache-join>/src first)."""

from __future__ import annotations

import os

import pytest

kv_connector = pytest.importorskip("vllm_tt_plugin.kv_connector")
if not hasattr(kv_connector, "KDACheckpointStore"):
    pytest.skip("vllm-tt-plugin without the LMCache join (KDACheckpointStore)", allow_module_level=True)

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (  # noqa: E402
    RecordingMockModel,
    prompt_ids,
    reference_greedy,
)

H.engine_env()

from vllm import SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import layer_kinds, text_config_of  # noqa: E402


def _run_one(llm, rid, prompt, n):
    eng = llm.llm_engine
    params = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), params)
    out = None
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                out = o
    return list(out.outputs[0].token_ids), out


def test_connector_prefix_hit_restores_pages_and_kda_state(hf_config, tmp_path, monkeypatch, evidence):
    # the 3-layer mock has one sparse_dsa layer: it stands in for all 11 DSA layers of vLLM's KV spec (same tensor)
    mla = [i for i, k in enumerate(layer_kinds(text_config_of(hf_config))) if k == "mla"]

    def latent_pages(self, kv_cache):
        lat = next(c.latent for l, c in zip(self.layers, kv_cache.caches) if not l.is_kda)
        return {i: lat for i in mla}

    monkeypatch.setattr(RecordingMockModel, "latent_pages", latent_pages)
    from vllm_tt_plugin.worker import TTWorker

    if not hasattr(TTWorker, "get_kv_connector_handshake_metadata"):
        # join finding J6: vLLM's EngineCore asks every worker for connector handshake metadata whenever a
        # kv_transfer_config is set (v1/engine/core.py:181); the TT worker lacks the method, so the engine dies at
        # init. The GPU worker's implementation (gpu_worker.py:680), single rank:
        def get_kv_connector_handshake_metadata(self):
            from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

            if not has_kv_transfer_group():
                return None
            md = get_kv_transfer_group().get_handshake_metadata()
            return None if md is None else {(0, 0): md}

        monkeypatch.setattr(TTWorker, "get_kv_connector_handshake_metadata", get_kv_connector_handshake_metadata,
                            raising=False)  # fmt: skip
    kv = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "shared_storage_path": str(tmp_path / "pages"),
            "tt_kda_checkpoint_dir": str(tmp_path / "kda"),
        },
    )
    with H.fake_tt_mesh():
        # the join scheduler's KDA policy ends the 200-token prompt's prefill on its last page boundary first:
        # [0, 192) + [192, 200); the slot's state is checkpointed at 192 = the page-aligned len - 1 hit of B.
        # prefill_chunk_alignment 64 keeps [0, 192) one chunk: ExampleConnector stores a request's FIRST chunk only
        # (at the served 128 alignment it would hold 128 rows for a 192-token hit; LMCache stores every chunk)
        llm = H.make_llm(max_num_seqs=4, max_num_batched_tokens=512, async_scheduling=False, kv_transfer_config=kv,
                         tt_config={"prefill_chunk_alignment": 64})  # fmt: skip
        try:
            core = llm.llm_engine.engine_core.engine_core
            model = core.model_executor.driver_worker.worker.model_runner.model
            gen = H.live_generator()
            prompt = prompt_ids(51, 200)
            a, _ = _run_one(llm, "a", prompt, 8)
            exports = model.counters["kda_exports"]
            b, out_b = _run_one(llm, "b", prompt, 8)
            ref = reference_greedy(gen.backend, prompt, 8)
            row = dict(a=a, b=b, ref=ref, kda_exports=exports, kda_imports=model.counters["kda_imports"],
                       prefix_state_imports=gen.counters["prefix_state_imports"],
                       num_cached_tokens=getattr(out_b, "num_cached_tokens", None),
                       kda_files=len(os.listdir(tmp_path / "kda")) if (tmp_path / "kda").exists() else 0)
            evidence(**row)
            assert a == ref and b == ref, row
            assert exports >= 1 and model.counters["kda_imports"] == 1, row
            assert gen.counters["prefix_state_imports"] == 1, row
        finally:
            del llm
