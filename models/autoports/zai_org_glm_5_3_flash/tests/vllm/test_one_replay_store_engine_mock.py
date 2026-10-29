# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-replay store A through the real runner path (program 2026-10-25T07:25:33Z: the cf3h-g r4 mock died on the
first storing prefill -- FrozenInstanceError on kda_snapshot_at -- and neither the pair capture test nor the 22
kda_snapshot_export tests built the runner's model input). The real vLLM V1 engine (in-process), the vllm-tt-plugin
KV-connector join with GLM53_KDA_CKPT_IN_REPLAY=1 (P1: no tail split for storing prompts, kda_snapshot_at on the
prefill row, export at the cut from the snapshot) and this adapter over the 06a CPU mock taught the snapshot API
(glm53_mock_stack: GLM53_PF_KDA_SNAPSHOT=1, prefill_chunk(snap_at=), export_kda_state(source="snapshot")), with
vLLM's ExampleConnector for the pages (as test_lmcache_engine_mock).

Request A (200 tokens, 64-token pages and grid) is stored: ONE prefill chunk [0, 200) carrying snap_at 192, its
checkpoint exported from the in-replay snapshot at 192. Request B (the same prompt) hits it: pages + the 192 checkpoint
imported, only [192, 200) prefilled. A, B and the oracle give the same greedy tokens.

Run (host-8chip, host only): no-tt-devices venv-lmcache/bin/python -m pytest <this file> -q -p no:cacheprovider, with
PYTHONPATH=<the tt-metal tree>:<vllm-tt-plugin src> (a plugin with P1, e.g. 9's s9h8g line)."""

from __future__ import annotations

import os

import pytest

kv_connector = pytest.importorskip("vllm_tt_plugin.kv_connector")
if not hasattr(kv_connector.TTKVConnectorHost, "_mark_kda_snapshots"):
    pytest.skip("vllm-tt-plugin without the one-replay store (P1, GLM53_KDA_CKPT_IN_REPLAY)", allow_module_level=True)

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import prompt_ids, reference_greedy  # noqa: E402

H.engine_env()

from vllm import SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def _run_one(llm, rid, prompt, n):
    eng = llm.llm_engine
    eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), SamplingParams(max_tokens=n, temperature=0.0,
                                                                                ignore_eos=True))  # fmt: skip
    out = None
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                out = o
    return list(out.outputs[0].token_ids), out


@pytest.mark.timeout(900)
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_one_replay_store_through_the_runner(hf_config, tmp_path, tmp_path_factory, monkeypatch, evidence, async_scheduling):
    H.patch_connector_join(monkeypatch, hf_config)
    monkeypatch.setenv("GLM53_PF_KDA_SNAPSHOT", "1")
    monkeypatch.setenv("GLM53_KDA_CKPT_IN_REPLAY", "1")
    pages_dir, kda_dir = H.shared_connector_dirs(tmp_path_factory)
    kv = KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both", kv_connector_extra_config={
        "shared_storage_path": str(pages_dir), "tt_kda_checkpoint_dir": str(kda_dir)})  # fmt: skip
    with H.fake_tt_mesh():
        llm = H.make_llm(max_num_seqs=4, max_num_batched_tokens=512, async_scheduling=async_scheduling,
                         kv_transfer_config=kv, tt_config={"prefill_chunk_alignment": 64})  # fmt: skip
        try:
            core = llm.llm_engine.engine_core.engine_core
            model = core.model_executor.driver_worker.worker.model_runner.model
            assert model.kda_checkpoint_in_replay, "the adapter must report the in-replay snapshot"
            gen = H.live_generator()
            events = gen.backend.events
            prompt = prompt_ids(71 + int(async_scheduling), 200)  # distinct per case (process-wide connector state)
            # counters as deltas over this test's requests (engines of earlier tests in the process leave theirs)
            snap = lambda: (dict(model.counters), dict(gen.counters))  # noqa: E731
            m0, g0 = snap()
            n0 = len(events)
            a, _ = _run_one(llm, "a", prompt, 8)
            n1 = len(events)
            b, out_b = _run_one(llm, "b", prompt, 8)
            m1, g1 = snap()
            dm = lambda k: m1.get(k, 0) - m0.get(k, 0)  # noqa: E731
            dg = lambda k: g1.get(k, 0) - g0.get(k, 0)  # noqa: E731
            ref = reference_greedy(gen.backend, prompt, 8)
            chunks = lambda lo, hi: [(e["start"], e["n"], e.get("snap_at")) for e in events[lo:hi]  # noqa: E731
                                     if e["event"] == "prefill_chunk"]  # fmt: skip
            row = dict(async_scheduling=async_scheduling, a=a, b=b, ref=ref, a_chunks=chunks(n0, n1),
                       b_chunks=chunks(n1, len(events)), kda_exports=dm("kda_exports"),
                       kda_exports_snapshot=dm("kda_exports_snapshot"), kda_imports=dm("kda_imports"),
                       kda_snapshot_chunks=dg("kda_snapshot_chunks"),
                       num_cached_tokens=getattr(out_b, "num_cached_tokens", None),
                       kda_files=len(os.listdir(kda_dir)))  # fmt: skip
            evidence(**row)
            assert a == ref and b == ref, row
            assert row["a_chunks"] == [(0, 200, 192)], row  # ONE replay, the cut written inside it (no tail split)
            assert row["kda_exports_snapshot"] >= 1 and row["kda_snapshot_chunks"] == 1, row
            assert row["kda_imports"] == 1 and row["b_chunks"] == [(192, 8, None)], row  # B resumed at the cut
        finally:
            del llm
