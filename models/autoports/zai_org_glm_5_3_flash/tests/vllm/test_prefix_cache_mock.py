# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefix cache through the real vLLM engine on the CPU mock (srv/prefixcache): vLLM automatic prefix caching ON, the
TT scheduler's prefix oracle, the adapter's PrefixStore, the generator's attach / snapshot rows and the mock
backend's snapshot pool. The mock cannot check device numerics; it checks the CALLS (which chunk resumed where, from
which snapshot entry) and that every greedy output equals the unpaged oracle of the full sequence.

Run (host-8chip, host only):
  no-tt-devices venv-vllm/bin/python -m pytest models/autoports/zai_org_glm_5_3_flash/tests/vllm/test_prefix_cache_mock.py \\
      -q -p no:cacheprovider --confcutdir models/autoports/zai_org_glm_5_3_flash/tests/vllm
"""

from __future__ import annotations

import os

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import prompt_ids
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_vllm_engine_mock import GREEDY, Engine

H.engine_env()

from vllm import SamplingParams  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import prefix_cache as PC  # noqa: E402

B = PC.BOUNDARY
TT = {"prefill_chunk_alignment": B}


def _engine(**kw):
    return Engine(max_num_seqs=4, max_num_batched_tokens=512, max_model_len=1024, enable_prefix_caching=True,
                  tt_config=TT, **kw)  # fmt: skip


def _prefills(events, since=0):
    return [e for e in events[since:] if e["event"] == "prefill_chunk"]


@pytest.mark.timeout(900)
def test_two_turn_chat_resumes_from_the_generation_snapshot():
    """Turn 1 (500-token prompt, 40 greedy tokens) leaves a P snapshot at 384 (its final chunk is split on the last
    128 boundary) and a G snapshot at 512 (the decode crossed it; keyed at release). Turn 2 = turn 1's prompt +
    answer + 30 new tokens (570): vLLM's page hit is 512, the oracle keeps 512, the row attaches to the G entry
    and prefills [512, 570) only; its tokens equal the oracle of the whole sequence."""
    e = _engine()
    try:
        model = e.runner.model
        assert model.prefix is not None and model.prefix.device_entries > 0
        p1 = prompt_ids(1, 500)
        rec = e.run([(0, "t1", p1, SamplingParams(max_tokens=40, **GREEDY))])
        assert not e.check_greedy(rec)
        out1 = rec["t1"]["tokens"]
        pf1 = _prefills(e.events)
        assert [(x["start"], x["n"]) for x in pf1] == [(0, 384), (384, 116)], pf1
        assert pf1[0]["state_snap"] is not None and pf1[1]["state_snap"] is None  # 384 is a boundary, 500 is not
        rep = model.prefix.report()
        assert rep["registered"] == 2 and rep["trailing_committed"] == 1, rep  # P@384 + G@512
        assert any(x["event"] == "snapshot_slot" for x in e.events)  # the decode crossing (plain DMA)
        if os.environ.get("GLM53_DECODE_TEE") == "1":  # ... or, with the tee, inside the crossing decode step
            assert model.generator.counters.get("snapshot_tees", 0) >= 1, model.generator.counters
        n1 = len(e.events)
        p2 = p1 + out1 + prompt_ids(2, 30)
        assert len(p2) == 570
        rec2 = e.run([(0, "t2", p2, SamplingParams(max_tokens=20, **GREEDY))])
        assert not e.check_greedy(rec2), "turn 2 attached to the cached prefix must equal the uncached oracle"
        pf2 = _prefills(e.events, n1)
        assert [(x["start"], x["n"]) for x in pf2] == [(512, 58)], pf2
        # the trailing (G) snapshot = the LAST snapshot_slot of turn 1 (the first one is the P snapshot at 384, which the
        # mock also records as a snapshot_slot inside its prefill_chunk)
        snaps = [x for x in e.events[:n1] if x["event"] == "snapshot_slot"]
        g_entry = snaps[-1]["entry"]
        assert snaps[0]["entry"] == pf1[0]["state_snap"] and g_entry != snaps[0]["entry"], snaps
        assert pf2[0]["state_src"] == g_entry, (pf2, g_entry)
        assert any(x["event"] == "restore_slot" and x["entry"] == g_entry for x in e.events[n1:])
        c = model.counters
        assert c["prefix_attach_rows"] == 1 and c["prefix_tokens_saved"] == 512 and c.get("prefix_restarts", 0) == 0
        assert model.prefix.stats["hits"] == 1
        assert getattr(e.llm.llm_engine.engine_core.engine_core.scheduler, "_prefix_hits_truncated", 0) == 0
    finally:
        e.close()


@pytest.mark.timeout(900)
def test_shared_system_prompt_across_users_attaches_the_longest_boundary():
    """Two users share a 300-token system prompt. User A (350 tokens) prefills [0, 256) + [256, 350) and snapshots at
    256; user B (380 tokens, arriving later) gets vLLM's 256-token page hit, attaches to A's P snapshot (shared
    read-only pages, the state copied into B's slot inside the prefill) and prefills [256, 380). A third user whose
    prompt diverges inside the first block gets no hit."""
    e = _engine()
    try:
        model = e.runner.model
        sys_p = prompt_ids(3, 300)
        a = sys_p + prompt_ids(4, 50)
        b = sys_p + prompt_ids(5, 80)
        c = [t + 1 for t in sys_p[:10]] + sys_p[10:] + prompt_ids(6, 40)
        rec = e.run([(0, "A", a, SamplingParams(max_tokens=8, **GREEDY)),
                     (6, "B", b, SamplingParams(max_tokens=8, **GREEDY)),
                     (12, "C", c, SamplingParams(max_tokens=8, **GREEDY))])  # fmt: skip
        assert not e.check_greedy(rec)
        pf = _prefills(e.events)
        starts = [(x["start"], x["n"], x["state_src"] is not None) for x in pf]
        assert starts[:2] == [(0, 256, False), (256, 94, False)], starts
        assert (256, 124, True) in starts, starts  # B attached at 256
        assert (0, 256, False) in starts[2:] or (0, 350 - 350, False) not in starts  # C restarted from 0
        assert sum(1 for s in starts if s[0] == 0) == 2, starts  # A and C from 0, B never
        cnt = model.counters
        assert cnt["prefix_attach_rows"] == 1 and cnt["prefix_tokens_saved"] == 256
        rep = model.prefix.report()
        assert rep["hits"] == 1 and rep["attached_slots"] == 0 and rep["entries"] >= 1
    finally:
        e.close()
