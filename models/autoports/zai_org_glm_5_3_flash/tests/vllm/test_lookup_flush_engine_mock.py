# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 12 W4 F2b under concurrent requests (program 2026-10-25T07:11:24Z): the real vLLM V1 engine (in-process)
with the vllm-tt-plugin KV-connector host (held page stores, GLM53_KV_STORE_HOLD=1 + HOLD_MAX=256 + FT_GRACE=1, the
cf3h-g / cf3h-h served set) over the 06a CPU mock and vLLM's ExampleConnector (page files; as test_lmcache_engine_mock).

A long prompt L (600 tokens) prefills one 64-token chunk per step (the whole token budget), so its page stores are
held. A second prompt arriving mid-prefill waits; vLLM admits it -- and runs its connector lookup -- in the step that
carries L's 24-token LAST chunk, BEFORE that step executes, i.e. before L's first token (the served order too: a long
prompt's chunk takes the whole 2,048-token budget). Under the old rule (GLM53_KV_LOOKUP_FLUSH=all) that lookup flushes
every held step of L right there, on L's first-token path. Under =prefix it does so only when the arrival shares L's
first LMCache chunk (M: it could hit those stores); an arrival with another first chunk (S) leaves them held until
after L's first token. In every case each greedy output equals the mock oracle and every held step of L is stored by
the end. (A repeat of L is not a usable hit check here: ExampleConnector stores a chunked request's FIRST chunk under
the whole prompt's key, so its hit on a chunked prompt overruns the file.)

Run (host-8chip, host only): no-tt-devices venv-lmcache/bin/python -m pytest <this file> -q -p no:cacheprovider, with
PYTHONPATH=<the tt-metal tree>:<vllm-tt-plugin src> (plugin glm53/glm53-s12-ftgrace 2c547ca or later)."""

from __future__ import annotations

from pathlib import Path

import pytest

kv_connector = pytest.importorskip("vllm_tt_plugin.kv_connector")
if not hasattr(kv_connector.TTKVConnectorHost, "serves_prefix"):
    pytest.skip("vllm-tt-plugin without GLM53_KV_LOOKUP_FLUSH (plugin 2c547ca)", allow_module_level=True)

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import prompt_ids, reference_greedy  # noqa: E402

H.engine_env()

from vllm import SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

GREEDY = dict(temperature=0.0, ignore_eos=True)


def _drive(llm, schedule, log, clock):
    """schedule [(arrival_step, rid, prompt, max_tokens)] -> {rid: record}; clock[0] = the engine step index"""
    eng = llm.llm_engine
    pending = sorted(schedule, key=lambda x: x[0])
    rec, step = {}, 0
    while pending or eng.has_unfinished_requests():
        clock[0] = step
        while pending and pending[0][0] <= step:
            _, rid, prompt, n = pending.pop(0)
            eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), SamplingParams(max_tokens=n, **GREEDY))
            rec[rid] = dict(prompt=prompt, arrival=step)
            log.append(("arrive", rid, step))
        for out in eng.step():
            r = rec[out.request_id]
            if out.outputs[0].token_ids and "first" not in r:
                r["first"] = step
                log.append(("first", out.request_id, step))
            if out.finished:
                r.update(tokens=list(out.outputs[0].token_ids), cached=getattr(out, "num_cached_tokens", None))
        step += 1
        assert step < 5000, "engine did not drain"
    return rec


@pytest.mark.timeout(900)
@pytest.mark.parametrize("other", ["S", "M"])
@pytest.mark.parametrize("mode", ["prefix", "all"])
def test_lookup_flush_with_a_long_prompt_mid_prefill(hf_config, tmp_path, tmp_path_factory, monkeypatch, evidence, mode, other):
    H.patch_connector_join(monkeypatch, hf_config)
    monkeypatch.setenv("GLM53_KV_STORE_HOLD", "1")
    monkeypatch.setenv("GLM53_KV_STORE_HOLD_MAX", "256")
    monkeypatch.setenv("GLM53_KV_STORE_FT_GRACE", "1")
    monkeypatch.setenv("GLM53_KV_LOOKUP_FLUSH", mode)
    pages_dir, kda_dir = H.shared_connector_dirs(tmp_path_factory)
    kv = KVTransferConfig(kv_connector="ExampleConnector", kv_role="kv_both", kv_connector_extra_config={
        "shared_storage_path": str(pages_dir), "tt_kda_checkpoint_dir": str(kda_dir)})  # fmt: skip
    seed = 60 + 4 * ["prefix", "all"].index(mode) + 2 * ["S", "M"].index(other)  # distinct prompts per case
    L = prompt_ids(seed, 600)  # 9 full 64-token chunks + a 24-token last chunk (leaves budget for an admission)
    X = prompt_ids(seed + 1, 100) if other == "S" else L[:64] + prompt_ids(seed + 1, 90)  # M shares L's first chunk
    log: list = []
    clock = [0]
    with H.fake_tt_mesh():
        llm = H.make_llm(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=1024, async_scheduling=True,
                         kv_transfer_config=kv, tt_config={"prefill_chunk_alignment": 64})  # fmt: skip
        try:
            core = llm.llm_engine.engine_core.engine_core
            runner = core.model_executor.driver_worker.worker.model_runner
            host = runner.kv_connector_host
            orig = host._save_pages

            def spy(so, mi):
                rids = {str(r).split("-")[0] for r in (getattr(mi, "row_req_ids", None) or []) if r}  # vLLM suffixes ids
                log.append(("save", tuple(sorted(rids)), clock[0]))
                return orig(so, mi)

            host._save_pages = spy
            conn = host.connector
            orig_skl = conn.save_kv_layer

            def skl_spy(name, *a, **k):
                md = conn._get_connector_metadata()
                stores = [len(r.token_ids) for r in getattr(md, "requests", []) if getattr(r, "is_store", False)]
                log.append(("skl", tuple(stores), clock[0]))
                return orig_skl(name, *a, **k)

            conn.save_kv_layer = skl_spy
            # vLLM keeps process-wide KV-transfer state: a later test's engine may write into an earlier test's storage
            # path, so files are counted over the whole pytest session directory, before and after this test's requests
            store_dir = Path(tmp_path).parent.parent
            n_files = lambda: sum(1 for _ in store_dir.rglob("*.safetensors")) if store_dir.exists() else 0  # noqa: E731
            files0 = n_files()
            orig_flush = kv_connector.flush_deferred_saves

            def flush_spy(request=None):
                log.append(("lookup", str(getattr(request, "request_id", None)).split("-")[0], clock[0]))
                return orig_flush(request)

            monkeypatch.setattr(kv_connector, "flush_deferred_saves", flush_spy)
            gen = H.live_generator()
            rec = _drive(llm, [(0, "L", L, 8), (3, other, X, 8)], log, clock)
            held_left = [tuple(getattr(mi, "row_req_ids", None) or []) for _so, mi in (host._held_pages or [])]
            pages = n_files() - files0
            refs = {"L": reference_greedy(gen.backend, L, 8), other: reference_greedy(gen.backend, X, 8)}
            l_saves = [s for kind, rids, s in log if kind == "save" and "L" in rids]
            x_lookup = [st for kind, r, st in log if kind == "lookup" and r == other]
            l_first = rec["L"]["first"]
            before = [s for s in l_saves if s < l_first]  # L's page stores ahead of L's first token
            row = dict(mode=mode, other=other, l_saves_steps=l_saves, x_arrive=rec[other]["arrival"],
                       x_lookups=x_lookup, l_first=l_first, saves_before_first=len(before),
                       skipped=getattr(host, "lookup_flushes_skipped", 0), held_left=held_left, page_files=pages,
                       stores=sorted({(st, x) for kind, x, st in log if kind == "skl" and x}),
                       ok={rid: rec[rid]["tokens"] == refs[rid] for rid in refs})  # fmt: skip
            evidence(**row)
            assert all(row["ok"].values()), row
            assert x_lookup and x_lookup[0] < l_first, row  # the arrival is looked up before L's first token
            if mode == "prefix" and other == "S":
                assert not before and row["skipped"] >= 1, row  # L's held stores stay off its first-token path
            else:  # the old rule, or an arrival that could hit L's stores: flushed at that lookup
                assert x_lookup[0] in before and len(before) >= len(L) // 64, row
            assert len(l_saves) >= len(L) // 64 and not held_left and pages > 0, row  # every held step stored
        finally:
            del llm
