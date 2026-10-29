# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 9 k4: GLM53_RESUME_AS_DECODE (doc/ttnn/09-vllm/design/resume_as_decode.md), the model side of Stage 12's
no-recompute preemption (plugin 3db211c, GLM53_PREEMPT_SAVE). A request preempted mid-decode keeps its KV [0, n) and an
EXACT KDA checkpoint at n, usually off the 64-token page grid. On resume vLLM computes one token: the last sampled one,
at position n. With the flag, the adapter exports the off-grid live checkpoint, and prefill_forward accepts the one-token
row at that start and runs it as ONE decode step of its slot. The resumed stream must equal the unpreempted run's, and
the other requests' streams must not move. The KV pages stay in their physical blocks across the preemption (the
role LMCache's bitwise restore plays in serving)."""

from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import prompt_ids
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_generator_vllm import Pages, _adapter, _sampling
from models.autoports.zai_org_glm_5_3_flash.tt import generator as G


def _prefill(model, pages, req, seq, vslot, start=0, blob=None):
    """one prefill row of `req` (tokens seq[req][start:]) into vLLM slot `vslot`; blob: a connector checkpoint imported
    first (a hit continuing at `start`) -> the sampled token, appended to seq[req]"""
    if blob is not None:
        model.import_kda_state(model.kv_cache, vslot, blob)
    end = len(seq[req])
    pages.ensure(req, end + 1)
    o = model.prefill_forward(tokens=torch.tensor([seq[req]], dtype=torch.int32), page_table=pages.table([req]),
                              kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([end]),
                              start_pos=torch.tensor([start]), empty_slots=[vslot], sampling_params=_sampling(1))  # fmt: skip
    seq[req].append(int(torch.as_tensor(o).reshape(-1)[0]))


def _decode(model, pages, live, seq):
    """one plain decode step of the live requests {req: vLLM slot}"""
    S = model.max_batch_size
    tokens = torch.full((S, 1), -1, dtype=torch.int64)
    start = torch.full((S,), -1, dtype=torch.int64)
    for r, v in live.items():
        pages.ensure(r, len(seq[r]) + 1)
        tokens[v, 0] = seq[r][-1]
        start[v] = len(seq[r]) - 1
    tbl = pages.table([next((r for r, v in live.items() if v == i), None) for i in range(S)])
    o = torch.as_tensor(model.decode_forward(tokens, start, tbl, model.kv_cache, sampling_params=_sampling(S),
                                             read_from_device=True, reload_inputs=True)).reshape(-1)  # fmt: skip
    for r, v in live.items():
        seq[r].append(int(o[v]))


PROMPTS = {"r": prompt_ids(0, 70), "a": prompt_ids(1, 41), "b": prompt_ids(2, 57)}
STEPS = 24  # decode steps per request in total
CUT = 9  # R is preempted after this many decode steps: n = 70 + 1 + 9 - 1 = 79 computed tokens (off the 64 grid)


def _run(hf_config, monkeypatch, *, preempt: bool, flag: bool):
    monkeypatch.setattr(G, "RESUME_AS_DECODE", flag)
    model = _adapter(hf_config, max_batch=4)
    pages = Pages(256, 64)
    seq = {k: list(v) for k, v in PROMPTS.items()}
    live = {"r": 0, "a": 1, "b": 2}
    for r, v in live.items():
        _prefill(model, pages, r, seq, v)
    for _ in range(CUT):
        _decode(model, pages, live, seq)
    if preempt:
        n = len(seq["r"]) - 1  # computed tokens: the last sampled token was never an input
        blob = model.export_kda_state(model.kv_cache, live["r"])  # the exact checkpoint at n (Stage 12's save)
        assert blob["next_pos"] == n and n % 64 != 0, (blob["next_pos"], n)
        model.release_request(live.pop("r"))  # preempted (no token ids); R's KV pages stay where they are
        for _ in range(3):  # the others run on without R
            _decode(model, pages, live, seq)
        mark = len(model.generator.backend.events)
        _prefill(model, pages, "r", seq, 3, start=n, blob=blob)  # the resume: ONE token at position n
        model.resume_events = model.generator.backend.events[mark:]
        live["r"] = 3
    while min(len(seq[r]) - len(PROMPTS[r]) for r in live) < STEPS:
        _decode(model, pages, live, seq)
    return {r: seq[r][len(PROMPTS[r]):][:STEPS] for r in PROMPTS}, model


def test_a_preempted_request_resumes_with_one_decode_step_and_the_same_stream(hf_config, monkeypatch):
    ref, _ = _run(hf_config, monkeypatch, preempt=False, flag=True)
    got, model = _run(hf_config, monkeypatch, preempt=True, flag=True)
    assert got == ref, (got, ref)
    gen = model.generator
    for r, p in PROMPTS.items():  # and every stream is the oracle's greedy continuation
        assert got[r] == gen.backend.reference_greedy(p, STEPS), r
    assert gen.counters.get("resume_decodes", 0) == 1
    # the resume call ran no prefill chunk, and exactly one decode step with one active row
    kinds = [e.get("event") for e in model.resume_events]
    steps = [e for e in model.resume_events if e.get("event") == "decode_step"]
    assert "prefill_chunk" not in kinds and len(steps) == 1 and steps[0]["active"] == 1, model.resume_events


def test_without_the_flag_an_off_grid_resume_is_refused(hf_config, monkeypatch):
    """today's contract: the live export and the continuation both need the 64-token page grid"""
    monkeypatch.setattr(G, "RESUME_AS_DECODE", False)
    model = _adapter(hf_config, max_batch=4)
    pages = Pages(256, 64)
    seq = {"r": list(PROMPTS["r"])}
    _prefill(model, pages, "r", seq, 0)
    for _ in range(CUT):
        _decode(model, pages, {"r": 0}, seq)
    with pytest.raises(ValueError, match="page boundary"):
        model.export_kda_state(model.kv_cache, 0)


def test_a_multi_token_row_at_an_off_grid_start_is_still_refused(hf_config, monkeypatch):
    """the flag admits ONE token at an exact checkpoint, nothing else: a 2-token continuation off the grid is refused"""
    monkeypatch.setattr(G, "RESUME_AS_DECODE", True)
    model = _adapter(hf_config, max_batch=4)
    pages = Pages(256, 64)
    seq = {"r": list(PROMPTS["r"])}
    _prefill(model, pages, "r", seq, 0)
    for _ in range(CUT):
        _decode(model, pages, {"r": 0}, seq)
    n = len(seq["r"]) - 1
    blob = model.export_kda_state(model.kv_cache, 0)
    model.release_request(0)
    seq["r"].append(5)  # two uncomputed tokens
    model.import_kda_state(model.kv_cache, 1, blob)
    pages.ensure("r", len(seq["r"]) + 1)
    with pytest.raises(ValueError, match="not a multiple"):
        model.prefill_forward(tokens=torch.tensor([seq["r"]], dtype=torch.int32), page_table=pages.table(["r"]),
                              kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([n + 2]),
                              start_pos=torch.tensor([n]), empty_slots=[1], sampling_params=_sampling(1))  # fmt: skip
