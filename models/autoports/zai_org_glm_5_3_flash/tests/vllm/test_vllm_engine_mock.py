# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end host-only serving tests: the real vLLM 0.26 V1 engine (in-process), the TT plugin's scheduler,
worker, model runner and async-decode controller, the GLM-5.3-Flash adapter, 06a's Glm53Generator and its CPU
MockModel backend. Only the mesh open/close are faked (engine_harness.fake_tt_mesh); nothing opens a TT device.

Every greedy output is compared token for token with 06a's unpaged oracle (MockModel.reference_greedy), so stale
tokens / positions, page-table or state-slot mistakes anywhere in the vLLM -> plugin -> adapter -> generator path
change the result. Trace bookkeeping: the generator's host-work counters (captures, replays, refreshes, readbacks)
and the mock device (any compile while a trace is live raises; the compile log must not grow after warm-up).
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    DSA_MODE_ENV,
    GREEDY_ONLY_ENV,
    PER_WIDTH_ENV,
    PF_BATCH_ENV,
    PF_PACKED_CAPS_ENV,
    PF_PACKED_ENV,
    prompt_ids,
    reference_greedy,
)

H.engine_env()

from vllm import SamplingParams  # noqa: E402  (after engine_env: in-process engine core)
from vllm.inputs import TokensPrompt  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import KV_POOL_TOKENS_ENV  # noqa: E402

GREEDY = dict(temperature=0.0, ignore_eos=True)


def _prompt(seed: int, n: int) -> list[int]:
    return prompt_ids(seed, n)


class Engine:
    def __init__(self, **kwargs):
        self._mesh = H.fake_tt_mesh()
        self.meshes = self._mesh.__enter__()
        t0 = time.time()
        self.llm = H.make_llm(**kwargs)
        self.init_s = time.time() - t0
        self.gen = H.live_generator()
        self.backend = self.gen.backend
        self.baseline = self.trace_report()

    def trace_report(self):
        """Generator counters + mock-device program log: what the mixed-shape check needs."""
        dev = self.backend.device
        return {
            "counters": dict(self.gen.counters),
            "programs_compiled": len(dev.compile_log),
            "live_traces": len(dev.live_traces),
            "copy_slot_calls": sum(1 for e in self.backend.events if e["event"] == "copy_slot"),
        }

    @property
    def events(self):
        return self.backend.events

    @property
    def runner(self):
        core = self.llm.llm_engine.engine_core.engine_core
        return core.model_executor.driver_worker.worker.model_runner

    def close(self):
        del self.llm
        gc.collect()
        self._mesh.__exit__(None, None, None)

    def run(self, schedule, max_steps=20000):
        """schedule: [(arrival_step, request_id, prompt_ids, SamplingParams)] -> per-request record."""
        eng = self.llm.llm_engine
        pending = sorted(schedule, key=lambda x: x[0])
        rec = {}
        step = 0
        while pending or eng.has_unfinished_requests():
            while pending and pending[0][0] <= step:
                _, rid, prompt, params = pending.pop(0)
                eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), params)
                rec[rid] = {
                    "prompt_len": len(prompt),
                    "arrival_step": step,
                    "t_arrival": time.time(),
                    "temperature": params.temperature,
                    "prompt": prompt,
                }
            for out in eng.step():
                r = rec[out.request_id]
                toks = list(out.outputs[0].token_ids)
                if toks and "first_token_step" not in r:
                    r["first_token_step"] = step
                    r["ttft_s"] = time.time() - r["t_arrival"]
                if out.finished:
                    r.update(finish_step=step, tokens=toks, finish_reason=out.outputs[0].finish_reason)
            step += 1
            if step > max_steps:
                raise RuntimeError(f"workload did not finish in {max_steps} steps")
        for r in rec.values():
            r.pop("t_arrival", None)
        return rec

    def check_greedy(self, rec):
        bad = {}
        for rid, r in rec.items():
            if r["temperature"] == 0.0:
                ref = reference_greedy(self.backend, r["prompt"], len(r["tokens"]))
                if r["tokens"] != ref:
                    first = next(i for i, (a, b) in enumerate(zip(r["tokens"], ref)) if a != b)
                    bad[rid] = {"prompt_len": r["prompt_len"], "first_mismatch": first}
        return bad


def _device_sampled_steps(counters) -> int:
    """Decode steps the device sampler served: sampler-trace replays, or decode replays of a backend whose decode
    trace samples (samples_in_decode_trace)."""
    return counters["sampling_trace_replays"] + counters.get("fused_sampling_steps", 0)


def _assert_clean_trace(e, baseline=None):
    """One decode trace per bucket width + two sampler traces (none when the decode trace samples:
    samples_in_decode_trace), captured in warm-up and never again; no program compiled after warm-up (the mock device
    raises on a compile while a trace is live); slot state is copied only by the adapter's compaction moves (slot_remap
    itself is an indirection)."""
    baseline = baseline or e.baseline
    now = e.trace_report()
    c = now["counters"]
    widths = len(e.gen.decode_buckets)
    samplers = 0 if e.gen.fused_sampling else 2
    assert c["decode_trace_captures"] == widths and c["sampling_trace_captures"] == samplers, c
    assert now["programs_compiled"] == baseline["programs_compiled"], "a program compiled after warm-up"
    assert now["live_traces"] == widths + samplers, now
    assert now["copy_slot_calls"] == e.runner.model.counters["compaction_moves"], now
    assert c["eager_decode_steps"] == 0, "decode left the traced path"
    return now


def _strip(rec):
    return {
        k: {f: v for f, v in r.items() if f not in ("prompt", "tokens")} | {"n_out": len(r.get("tokens", []))}
        for k, r in rec.items()
    }


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


# ------------------------------------------------------------------------------------------------ non-aligned
def test_engine_nonaligned_prompt_lengths(evidence):
    """Non-aligned logical prompt lengths (not multiples of the 32/64-token physical granules, the 128-token chunk
    grid or the 2048-token chunk) through vLLM's scheduler, the plugin runner's padding and the adapter."""
    e = Engine(max_num_seqs=8, max_num_batched_tokens=512)
    try:
        lengths = [1, 31, 33, 129, 130, 131, 999, 1951]
        sched = [(0, f"n{n}", _prompt(n, n), SamplingParams(max_tokens=16, **GREEDY)) for n in lengths]
        rec = e.run(sched)
        assert e.check_greedy(rec) == {}
        assert all(len(r["tokens"]) == 16 for r in rec.values())
        _assert_clean_trace(e)
        chunks = [(c["start"], c["n"], c["bucket"]) for c in e.events if c["event"] == "prefill_chunk"]
        assert sum(n for _, n, _ in chunks) == sum(lengths)
        assert all(b % 64 == 0 and b - 64 < n <= b for _, n, b in chunks)
        evidence(
            prompt_lengths=lengths,
            prefill_chunks=chunks,
            outputs_match_oracle=True,
            max_model_len=e.llm.llm_engine.model_config.max_model_len,
            engine_init_s=round(e.init_s, 1),
        )
    finally:
        e.close()


# -------------------------------------------------------------------------------------------- mixed-shape reuse
def test_engine_trace_reuse_mixed_shape(evidence):
    """Mixed-shape trace reuse check in mock form ($vllm-integration): after generator-wide preparation, unseen
    logical lengths within prepared buckets, chunked prefill with changing start positions and partial final
    chunks while other requests decode, repeats of known lengths in new batch combinations / orders, and sampling-mode
    switches (device greedy -> host -> device sampled -> device greedy) before returning to earlier trace keys. Must
    hold: the warm-up's decode + 2 sampler traces are the only captures, nothing compiles afterwards, every greedy
    output equals the oracle."""
    e = Engine(max_num_seqs=8, max_num_batched_tokens=512)
    try:
        before = e.trace_report()
        host = SamplingParams(max_tokens=24, repetition_penalty=1.1, temperature=0.0, ignore_eos=True)
        sampled = SamplingParams(max_tokens=24, temperature=0.8, top_k=20, seed=7, ignore_eos=True)
        sched = [
            # wave 1: decode-heavy requests
            *[(0, f"d{i}", _prompt(100 + i, 40 + 17 * i), SamplingParams(max_tokens=160, **GREEDY)) for i in range(4)],
            # wave 2 (while wave 1 decodes): long prompts, chunked at the 512-token budget -> continuation starts
            (6, "long1500", _prompt(201, 1500), SamplingParams(max_tokens=32, **GREEDY)),
            (7, "long1777", _prompt(202, 1777), SamplingParams(max_tokens=32, **GREEDY)),
            # wave 3: unseen logical lengths inside prepared buckets
            *[
                (20 + k, f"u{n}", _prompt(n, n), SamplingParams(max_tokens=20, **GREEDY))
                for k, n in enumerate((129, 130, 131))
            ],
            # wave 4: a penalty request is host-sampled (logits path), a top-k request is device-sampled ('sample')
            (40, "host_pen", _prompt(301, 77), host),
            (41, "sampled", _prompt(302, 64), sampled),
            # wave 5: repeats of known lengths in a new combination and order
            (70, "rep1777", _prompt(203, 1777), SamplingParams(max_tokens=24, **GREEDY)),
            (70, "rep40", _prompt(100, 40), SamplingParams(max_tokens=24, **GREEDY)),
            (71, "rep1500", _prompt(201, 1500), SamplingParams(max_tokens=24, **GREEDY)),
            (71, "rep130", _prompt(130, 130), SamplingParams(max_tokens=24, **GREEDY)),
        ]
        t0 = time.time()
        rec = e.run(sched)
        wall = time.time() - t0
        bad = e.check_greedy({k: v for k, v in rec.items() if k != "host_pen"})
        assert bad == {}, bad
        # host-sampled greedy with a repetition penalty is not the oracle's greedy; it must still complete
        assert len(rec["host_pen"]["tokens"]) == 24 and len(rec["sampled"]["tokens"]) == 24
        after = _assert_clean_trace(e, before)
        cb, ca = before["counters"], after["counters"]
        delta = {k: ca[k] - cb[k] for k in ca}
        assert delta["decode_trace_replays"] > 0 and _device_sampled_steps(delta) > 0
        assert delta["logits_readback"] > 0, "the host-sampled request must have used the logits path"
        events = e.events
        chunks = [ev for ev in events if ev["event"] == "prefill_chunk"]
        cont = [c for c in chunks if c["start"] > 0]
        assert len({c["start"] for c in cont}) >= 3, "chunked prefill must continue at several start positions"
        assert any(c["n"] < 512 and c["n"] % 64 for c in chunks), "a partial final chunk is expected"
        # decode work interleaved with a request's own prefill chunks (decode steps with active rows in between)
        idx = {id(ev): i for i, ev in enumerate(events)}
        interleaved = 0
        for slot in {c["slot"] for c in cont}:
            mine = [idx[id(c)] for c in chunks if c["slot"] == slot]
            between = [ev for ev in events[min(mine) : max(mine)] if ev["event"] == "decode_step" and ev["active"] > 0]
            interleaved += bool(between)
        assert interleaved >= 1, "no decode step ran between the chunks of a chunked prefill"
        ttft = [r["ttft_s"] for r in rec.values()]
        overlap = e.runner.async_decode._overlapped_submissions
        assert overlap > 0 and e.runner.async_decode._overlapped_unsafe_submissions == 0
        evidence(
            requests=_strip(rec),
            prefill_chunks=[(c["slot"], c["start"], c["n"], c["bucket"]) for c in chunks],
            continuation_starts=sorted({c["start"] for c in cont}),
            chunked_requests_with_decode_between_chunks=interleaved,
            captures={"decode": ca["decode_trace_captures"], "sampling": ca["sampling_trace_captures"]},
            programs_compiled_before_after=(before["programs_compiled"], after["programs_compiled"]),
            counter_delta=delta,
            decode_steps=sum(1 for ev in events if ev["event"] == "decode_step"),
            adapter_counters=dict(e.runner.model.counters),
            overlapped_submissions=overlap,
            mock_ttft_s={"p50": _pct(ttft, 50), "p95": _pct(ttft, 95), "p99": _pct(ttft, 99)},
            mock_wall_s=round(wall, 2),
            completed=sum(1 for r in rec.values() if r.get("finish_reason") == "length"),
            note="mock timings (CPU mock model) are workload bookkeeping, not performance",
        )
    finally:
        e.close()


# -------------------------------------------------------------------------------------------------- async decode
def test_engine_async_decode_matches_sync(evidence):
    """The async split (submit / deferred read / host processing, overlapped with the next schedule) produces the
    same tokens as synchronous scheduling, and actually overlapped."""
    sched = [
        (i // 2, f"r{i}", _prompt(400 + i, 30 + 61 * i), SamplingParams(max_tokens=40, **GREEDY)) for i in range(6)
    ]
    results = {}
    for mode in (True, False):
        e = Engine(max_num_seqs=8, max_num_batched_tokens=512, async_scheduling=mode)
        try:
            rec = e.run(sched)
            assert e.check_greedy(rec) == {}
            now = _assert_clean_trace(e)
            results[mode] = {
                "tokens": {k: r["tokens"] for k, r in rec.items()},
                "overlapped": e.runner.async_decode._overlapped_submissions,
                "unsafe": e.runner.async_decode._overlapped_unsafe_submissions,
                "token_refresh": now["counters"]["token_refresh"] - e.baseline["counters"]["token_refresh"],
                "decode_trace_replays": now["counters"]["decode_trace_replays"],
                "async_scheduling": e.llm.llm_engine.vllm_config.scheduler_config.async_scheduling,
            }
        finally:
            e.close()
    assert results[True]["tokens"] == results[False]["tokens"]
    assert results[True]["async_scheduling"] and not results[False]["async_scheduling"]
    assert results[True]["overlapped"] > 0 and results[True]["unsafe"] == 0
    assert results[False]["overlapped"] == 0
    evidence(**{("async" if k else "sync"): {f: v for f, v in r.items() if f != "tokens"} for k, r in results.items()})


# ------------------------------------------------------------------------------------- request context limit
def test_engine_request_context_limit_dsa_directive(evidence, expect_error):
    """max_model_len stays 1,048,576 (advertised); requests beyond the contract's current_supported_context (DSA
    dense-exact, 2051) are refused at admission and default max_tokens stop at the limit."""
    e = Engine(max_num_seqs=4, max_num_batched_tokens=512)
    try:
        assert e.llm.llm_engine.model_config.max_model_len == 1048576
        with expect_error(ValueError, "request context limit of 2051"):
            e.llm.generate([TokensPrompt(prompt_token_ids=_prompt(1, 2100))], SamplingParams(max_tokens=4))
        with expect_error(ValueError, "2000 prompt \\+ 100 output tokens"):
            e.llm.generate([TokensPrompt(prompt_token_ids=_prompt(2, 2000))], SamplingParams(max_tokens=100))
        out = e.llm.generate(
            [TokensPrompt(prompt_token_ids=_prompt(3, 2000))],
            SamplingParams(max_tokens=None, temperature=0.0, ignore_eos=True),
        )
        toks = list(out[0].outputs[0].token_ids)
        assert len(toks) == 51 and out[0].outputs[0].finish_reason == "length"
        assert toks == reference_greedy(e.backend, _prompt(3, 2000), 51)
        evidence(max_model_len=1048576, request_context_limit=2051, default_max_tokens_for_2000=len(toks))
    finally:
        e.close()


# --------------------------------------------------------------------------------------------------- logprobs
def test_engine_logprob_requests_take_the_host_sampler(evidence):
    """On the 8-device mesh the plugin sends logprob requests to the device sampler unless the model declares
    supports_device_logprobs=False; the generator's sampler returns tokens only (06a contract section 7), so these
    requests must be host-sampled from read-back logits -- with logprobs, and the greedy tokens unchanged."""
    e = Engine(max_num_seqs=4, max_num_batched_tokens=512)
    try:
        assert e.runner.num_devices == 8
        lps = {}
        for lp in (0, 1, 5):
            out = e.llm.generate(
                [TokensPrompt(prompt_token_ids=_prompt(40 + lp, 50))],
                SamplingParams(max_tokens=6, temperature=0.0, logprobs=lp, ignore_eos=True),
            )
            o = out[0].outputs[0]
            assert list(o.token_ids) == reference_greedy(e.backend, _prompt(40 + lp, 50), 6)
            assert o.logprobs is not None and len(o.logprobs) == 6
            assert all(len(step) >= max(lp, 1) for step in o.logprobs)
            lps[lp] = [round(next(iter(step.values())).logprob, 4) for step in o.logprobs][:3]
        _assert_clean_trace(e)
        assert e.gen.counters["logits_readback"] > e.baseline["counters"]["logits_readback"]
        evidence(num_devices=8, first_logprobs=lps)
    finally:
        e.close()


# ------------------------------------------------------------------------------------------------ batch coverage
def test_engine_32_concurrent_sequences(evidence):
    e = Engine(max_num_seqs=32, max_num_batched_tokens=2048)
    try:
        lengths = [1 + 23 * i + (i % 5) for i in range(32)]
        rec = e.run(
            [(0, f"b{i}", _prompt(500 + i, n), SamplingParams(max_tokens=24, **GREEDY)) for i, n in enumerate(lengths)]
        )
        assert e.check_greedy(rec) == {}
        _assert_clean_trace(e)
        widest = max(ev["active"] for ev in e.events if ev["event"] == "decode_step")
        assert widest == 32
        evidence(max_num_seqs=32, widest_decode_rows=widest, prompt_lengths=lengths)
    finally:
        e.close()


_WAVE_B = (101, 142, 183, 224, 265, 306, 347, 388)  # 1,956 tokens: one prefill step of the 2,048-token budget
_WAVE_C = (300, 350, 400, 450, 1200)  # 2,700 tokens: the last one's first chunk is intermediate and within the bucket


def _batched_admission_schedule():
    """Wave A (step 0, idle engine): 6 greedy prompts -> one group of 4 + 2 per-user (rows == slots 0..5). Wave B
    (step 4, while wave A decodes): 8 prompts in one prefill step, greedy and seeded top-k sampled alternating -> two
    groups of 4; its rows are 0..7 of its prefill call while its slots are 6..13 (rows != slots). Wave C (step 8):
    5 prompts past the step's token budget: 4 fresh prompts within the bucket + one whose first chunk is intermediate
    and also within the bucket, so the step would form a group of 4 if it were device-sampled. The plugin samples a
    step holding an intermediate chunk on the host (vllm_tt_plugin model_runner, the intermediate-prefill rule: device
    sampling would advance an intermediate row's RNG) and the adapter passes no sampling params: no batch, every row
    per user through the logits path, the continuation per user."""
    sampled = dict(temperature=0.8, top_k=20, ignore_eos=True)
    wave_a = [(0, f"a{i}", _prompt(900 + i, 17 + 41 * i), SamplingParams(max_tokens=40, **GREEDY)) for i in range(6)]
    wave_b = [
        (4, f"b{i}", _prompt(950 + i, n), SamplingParams(max_tokens=16, **(GREEDY if i % 2 == 0 else sampled),
                                                         **({} if i % 2 == 0 else {"seed": 100 + i})))
        for i, n in enumerate(_WAVE_B)
    ]  # fmt: skip
    wave_c = [
        (8, f"c{i}", _prompt(970 + i, n), SamplingParams(max_tokens=16, **(sampled if i % 2 else GREEDY),
                                                         **({"seed": 200 + i} if i % 2 else {})))
        for i, n in enumerate(_WAVE_C)
    ]  # fmt: skip
    return wave_a + wave_b + wave_c


def test_engine_batched_prefill_admission(evidence, monkeypatch):
    """Stage 16's batched admission (Glm53Generator._admit_batch -> StackBackend.prefill_batch_tp) through the adapter
    and the real engine: with the backend reporting a captured batched-prefill trace (the mock's GLM53_MOCK_PF_BATCH:
    bucket 512, nb 4) a device-sampled prefill step's fresh requests (start 0, chunk <= the bucket) are admitted in
    groups of exactly 4, one prefill_batch_tp call each; the rest (a group's remainder) take the per-user path. A
    second wave is admitted while the first decodes (rows != slots) with seeded sampled requests inside its groups; a
    third wave's step holds an intermediate chunk beside 4 fresh prompts within the bucket and is not batched: the
    plugin samples it on the host (_batched_admission_schedule). Checks: every greedy output equals the reference
    oracle, and every output (greedy and seeded sampled) equals the same workload with batching off; the exact
    groups, per-user prompts (and which were device-sampled) and the continuation."""
    sched = _batched_admission_schedule()
    runs = {}
    for batched in (True, False):
        if batched:
            monkeypatch.setenv(PF_BATCH_ENV, "512,4")
        else:
            monkeypatch.delenv(PF_BATCH_ENV, raising=False)
        e = Engine(max_num_seqs=32, max_num_batched_tokens=2048)
        try:
            rec = e.run(sched)
            assert e.check_greedy(rec) == {}
            _assert_clean_trace(e)
            runs[batched] = {"rec": rec, "events": list(e.events), "batches": e.gen.counters.get("prefill_batches", 0)}
        finally:
            e.close()
    on, off = runs[True], runs[False]
    tokens = {k: r["tokens"] for k, r in on["rec"].items()}
    assert tokens == {k: r["tokens"] for k, r in off["rec"].items()}, "batched admission changed an output"
    assert all(len(t) == 16 or k.startswith("a") for k, t in tokens.items())
    assert off["batches"] == 0 and not [ev for ev in off["events"] if ev["event"] == "prefill_batch"]
    evs = on["events"]
    batches = [ev for ev in evs if ev["event"] == "prefill_batch"]
    fresh = [ev for ev in evs if ev["event"] == "prefill_chunk" and ev["start"] == 0]
    cont = [ev for ev in evs if ev["event"] == "prefill_chunk" and ev["start"] > 0]
    # wave A: one group of 4 + 2 per-user; wave B: two groups of 4; wave C (4 fresh prompts + the long one's
    # intermediate first chunk, all within the bucket) depends on the plugin's rule for a step holding an
    # intermediate chunk: the whole step on the host (5 per-user rows, no group), or -- a plugin with partial-prefill
    # sampling and this model's supports_partial_prefill_sampling -- device sampling with the intermediate row
    # unsampled (one group of the 4 complete prompts + the intermediate chunk per user without a head); then the
    # continuation per user
    import inspect

    from vllm_tt_plugin import model_runner as _runner

    partial = "supports_partial_prefill_sampling" in inspect.getsource(_runner)
    assert on["batches"] == len(batches) == 3 + partial and all(len(ev["slots"]) == 4 for ev in batches), batches
    long_first = [c for c in fresh if c["n"] not in (181, 222) + _WAVE_C]
    per_user = [181, 222] if partial else [181, 222, 300, 350, 400, 450]
    assert sorted(c["n"] for c in fresh if c is not long_first[0]) == per_user, fresh
    assert len(long_first) == 1 and long_first[0]["n"] <= 512 and len(cont) == 1, (fresh, cont)
    assert batches[0]["slots"] == [0, 1, 2, 3] and batches[0]["n"] == [17, 58, 99, 140]
    # wave B's groups: admitted while wave A's rows decode, in slots above its prefill call's rows 0..7
    idx = {id(ev): i for i, ev in enumerate(evs)}
    first_decode = min(i for i, ev in enumerate(evs) if ev["event"] == "decode_step" and ev["active"] > 0)
    late = batches[1:3]
    assert all(idx[id(ev)] > first_decode for ev in late)
    assert sorted(n for ev in late for n in ev["n"]) == list(_WAVE_B)
    assert min(s for ev in late for s in ev["slots"]) >= 6, late
    # seeded sampled requests inside wave B's groups (per-request sampling params reach prefill_batch_tp)
    assert sorted(t > 0 for ev in late for t in ev["temperature"]) == [False] * 4 + [True] * 4
    assert sorted(s for ev in late for s in ev["seed"] if s is not None) == [101, 103, 105, 107]
    # wave C's step holds the intermediate chunk
    if partial:  # device sampling kept: the 4 complete prompts in one group, the intermediate chunk without a head
        assert batches[3]["n"] == [300, 350, 400, 450] and idx[id(batches[3])] > idx[id(late[-1])], batches
        assert not long_first[0]["device_sample"] and long_first[0]["rows"] == "none", long_first
        assert all(c["device_sample"] for c in fresh if c is not long_first[0]), fresh
    else:  # sampled on the host (no device sampling for any of its rows); wave A's two per-user rows were on device
        assert all(c["device_sample"] == (c["n"] in (181, 222)) for c in fresh), fresh
    # the continuation runs per user in the same slot
    assert cont[0]["slot"] == long_first[0]["slot"] and cont[0]["start"] == long_first[0]["n"]
    assert cont[0]["start"] + cont[0]["n"] == 1200
    evidence(
        batched_groups=[{"slots": ev["slots"], "n": ev["n"], "temperature": ev["temperature"], "seed": ev["seed"]}
                        for ev in batches],
        per_user_fresh=[(c["slot"], c["n"], c["device_sample"]) for c in fresh],
        continuation_chunks=[(c["slot"], c["start"], c["n"], c["device_sample"]) for c in cont],
        outputs_equal_to_unbatched=True,
        requests=len(tokens),
    )  # fmt: skip


_WAVE_M1 = (16, 100, 300, 700, 1200)  # 2,316 tokens: the 1,200 prompt's first chunk (932) is intermediate
_WAVE_M2 = (40, 77, 129, 161, 250, 333, 512, 65)  # mixed lengths admitted while wave 1 decodes


def _packed_admission_schedule():
    """Mixed ISL (program 2026-10-22 03:15:30Z / 06:10:41Z): wave 1 = ONE vLLM step of 4 complete prompts of
    different lengths + a 1,200-token prompt whose first chunk is intermediate; wave 2 = 8 mixed lengths at step 6
    while wave 1 decodes (greedy and seeded sampled alternating)."""
    sampled = dict(temperature=0.8, top_k=20, ignore_eos=True)
    w1 = [(0, f"m{i}", _prompt(1300 + i, n), SamplingParams(max_tokens=12, **GREEDY)) for i, n in enumerate(_WAVE_M1)]
    w2 = [
        (6, f"n{i}", _prompt(1350 + i, n), SamplingParams(max_tokens=12, **(GREEDY if i % 2 == 0 else sampled),
                                                          **({} if i % 2 == 0 else {"seed": 300 + i})))
        for i, n in enumerate(_WAVE_M2)
    ]  # fmt: skip
    return w1 + w2


def test_engine_packed_prefill_admission(evidence, monkeypatch):
    """Stage 16's PACKED admission (Glm53Generator._admit_packed -> StackBackend.prefill_packed_tp: fresh requests of
    DIFFERENT lengths back to back in one (T, K) replay) through the adapter and the real engine, the mock reporting
    packed traces T = 2,048, KMAX = 4 (GLM53_MOCK_PF_PACKED). vLLM's step packs several requests of different lengths
    (and a long prompt's intermediate chunk) into ONE prefill_forward call. With a plugin keeping device sampling for
    such a step (supports_partial_prefill_sampling) the step's complete prompts are admitted packed (groups of 2..4
    whose 32-row-rounded lengths fit T) and the intermediate chunk runs per user without a head; with the base rule
    the whole step is host-sampled and nothing is packed. Checks: every greedy output equals the reference oracle,
    every output equals the same workload with packing off, the packed groups, the continuation."""
    import inspect

    from vllm_tt_plugin import model_runner as _runner


    partial = "supports_partial_prefill_sampling" in inspect.getsource(_runner)
    sched = _packed_admission_schedule()
    runs = {}
    for packed in (True, False):
        monkeypatch.delenv(PF_BATCH_ENV, raising=False)
        if packed:
            monkeypatch.setenv(PF_PACKED_ENV, "2048,4")
        else:
            monkeypatch.delenv(PF_PACKED_ENV, raising=False)
        e = Engine(max_num_seqs=32, max_num_batched_tokens=2048)
        try:
            rec = e.run(sched)
            assert e.check_greedy(rec) == {}
            _assert_clean_trace(e)
            runs[packed] = {"rec": rec, "events": list(e.events), "packed": e.gen.counters.get("prefill_packed", 0)}
        finally:
            e.close()
    on, off = runs[True], runs[False]
    tokens = {k: r["tokens"] for k, r in on["rec"].items()}
    assert tokens == {k: r["tokens"] for k, r in off["rec"].items()}, "packed admission changed an output"
    assert all(len(t) == 12 for t in tokens.values()), {k: len(t) for k, t in tokens.items()}
    assert off["packed"] == 0 and not [ev for ev in off["events"] if ev["event"] == "prefill_packed"]
    evs = on["events"]
    groups = [ev for ev in evs if ev["event"] == "prefill_packed"]
    fresh = [ev for ev in evs if ev["event"] == "prefill_chunk" and ev["start"] == 0]
    cont = [ev for ev in evs if ev["event"] == "prefill_chunk" and ev["start"] > 0]
    assert on["packed"] == len(groups)
    for g in groups:  # every group fits its trace: 2..4 requests, 32-row-rounded lengths <= T
        assert 2 <= len(g["n"]) <= 4 and sum(-(-n // 32) * 32 for n in g["n"]) <= g["t"] == 2048, g
    packed_n = sorted(n for g in groups for n in g["n"])
    long_first = [c for c in fresh if c["n"] not in _WAVE_M1[:4] + _WAVE_M2]
    assert len(long_first) == 1 and len(cont) >= 1 and cont[0]["slot"] == long_first[0]["slot"], (fresh, cont)
    assert cont[0]["start"] == long_first[0]["n"] and sum(c["n"] for c in cont) + long_first[0]["n"] == 1200
    if partial:  # wave 1's 4 complete prompts in one group; the intermediate chunk without a head
        assert groups and groups[0]["n"] == list(_WAVE_M1[:4]), groups
        assert not long_first[0]["device_sample"] and long_first[0]["rows"] == "none", long_first
    else:  # wave 1 host-sampled as a whole: its complete prompts per user
        assert not set(_WAVE_M1[:4]) & set(packed_n), groups
    # every fresh complete prompt is either packed or per user, exactly once
    per_user = sorted(c["n"] for c in fresh if c is not long_first[0])
    assert sorted(packed_n + per_user) == sorted(_WAVE_M1[:4] + _WAVE_M2), (packed_n, per_user)
    evidence(
        plugin_partial_sampling=partial,
        packed_groups=[{"slots": g["slots"], "n": g["n"], "temperature": g["temperature"]} for g in groups],
        per_user_fresh=[(c["slot"], c["n"], c["device_sample"]) for c in fresh],
        continuation_chunks=[(c["slot"], c["start"], c["n"], c["device_sample"]) for c in cont],
        outputs_equal_to_unpacked=True,
        requests=len(tokens),
    )  # fmt: skip


_WAVE_W1 = (100, 300, 700, 2000)  # step 0: three complete prompts + the 2,000 prompt's intermediate first chunk
_WAVE_W2 = (40, 150, 260)  # admitted while the 2,000 prompt continues: its continuation + fresh prompts, one step


def _whole_step_schedule():
    sampled = dict(temperature=0.8, top_k=20, ignore_eos=True)
    w1 = [(0, f"w{i}", _prompt(1400 + i, n), SamplingParams(max_tokens=10, **GREEDY)) for i, n in enumerate(_WAVE_W1)]
    w2 = [
        (1, f"x{i}", _prompt(1450 + i, n), SamplingParams(max_tokens=10, **(GREEDY if i % 2 == 0 else sampled),
                                                          **({} if i % 2 == 0 else {"seed": 400 + i})))
        for i, n in enumerate(_WAVE_W2)
    ]  # fmt: skip
    return w1 + w2


def test_engine_packed_whole_step(evidence, monkeypatch):
    """The operator's packed step (2026-10-22 07:32:59Z): with a backend whose packed call also takes continuation
    segments and head-less intermediate chunks (prefill_packed_caps; the mock: GLM53_MOCK_PF_PACKED_CAPS) EVERY row of
    a vLLM step goes into ONE packed replay when the 32-row-rounded segments fit T (budget 1,920 of T 2,048: room for
    the rounding of K <= 4 segments). Wave 1's step = 3 complete prompts + the 2,000 prompt's intermediate first chunk
    (head-less); wave 2's step = that prompt's continuation + fresh prompts. Checks: outputs equal to the same workload
    unpacked and to the greedy oracle; a packed replay holds a head-less segment and one holds a continuation."""
    import inspect

    from vllm_tt_plugin import model_runner as _runner

    if "supports_partial_prefill_sampling" not in inspect.getsource(_runner):
        pytest.skip("needs the per-row prefill sampling plugin (a step holding an intermediate chunk is host-sampled)")
    sched = _whole_step_schedule()
    runs = {}
    for mode in ("whole", "off"):
        monkeypatch.delenv(PF_BATCH_ENV, raising=False)
        if mode == "whole":
            monkeypatch.setenv(PF_PACKED_ENV, "2048,4")
            monkeypatch.setenv(PF_PACKED_CAPS_ENV, "continuation,no_head")
        else:
            monkeypatch.delenv(PF_PACKED_ENV, raising=False)
            monkeypatch.delenv(PF_PACKED_CAPS_ENV, raising=False)
        e = Engine(max_num_seqs=32, max_num_batched_tokens=1920)
        try:
            rec = e.run(sched)
            assert e.check_greedy(rec) == {}
            _assert_clean_trace(e)
            runs[mode] = {"rec": rec, "events": list(e.events), "steps": e.gen.counters.get("prefill_packed_steps", 0)}
        finally:
            e.close()
    on, off = runs["whole"], runs["off"]
    tokens = {k: r["tokens"] for k, r in on["rec"].items()}
    assert tokens == {k: r["tokens"] for k, r in off["rec"].items()}, "the packed step changed an output"
    assert all(len(t) == 10 for t in tokens.values()), {k: len(t) for k, t in tokens.items()}
    groups = [ev for ev in on["events"] if ev["event"] == "prefill_packed"]
    assert on["steps"] == len(groups) >= 2, groups
    assert any(not h for g in groups for h in g["heads"]), groups  # a long prompt's intermediate chunk, packed
    assert any(st > 0 for g in groups for st in g["starts"]), groups  # a continuation, packed
    for g in groups:
        assert sum(-(-n // 32) * 32 for n in g["n"]) <= 2048 and 2 <= len(g["n"]) <= 4, g
    per_user = [ev for ev in on["events"] if ev["event"] == "prefill_chunk"]
    evidence(
        packed_steps=[{"slots": g["slots"], "n": g["n"], "starts": g["starts"], "heads": g["heads"]} for g in groups],
        per_user_chunks=[(c["slot"], c["start"], c["n"]) for c in per_user],
        outputs_equal_to_unpacked=True,
        requests=len(tokens),
    )  # fmt: skip


# ------------------------------------------------------------------------- KDA state slots across chunked prefill
def _slot_reuse_schedule():
    # a finishes while c is between prompt chunks; c's continuation is then re-added at a row whose slot is free
    return [
        (0, "a", _prompt(11, 50), SamplingParams(max_tokens=2, **GREEDY)),
        (0, "b", _prompt(12, 60), SamplingParams(max_tokens=60, **GREEDY)),
        (0, "c", _prompt(13, 1500), SamplingParams(max_tokens=8, **GREEDY)),
    ]


def test_engine_chunked_continuation_keeps_kda_slot(evidence):
    e = Engine(max_num_seqs=4, max_num_batched_tokens=256)
    try:
        rec = e.run(_slot_reuse_schedule())
        assert e.check_greedy(rec) == {}
        _assert_clean_trace(e)
        chunks = [(ev["slot"], ev["start"], ev["n"]) for ev in e.events if ev["event"] == "prefill_chunk"]
        # every continuation chunk of c must run in the generator slot that holds c's KDA state: the slot of its first
        # chunk, followed through any compaction move (copy_slot) the adapter made between chunks
        evs = [ev for ev in e.events if ev["event"] in ("prefill_chunk", "copy_slot")]
        cont = [i for i, ev in enumerate(evs) if ev["event"] == "prefill_chunk" and ev["start"] > 0]
        first = evs[cont[0]]
        head = max(
            i
            for i in range(cont[0])
            if evs[i]["event"] == "prefill_chunk" and evs[i]["start"] == 0 and evs[i]["n"] == first["start"]
        )
        slot, followed = evs[head]["slot"], 0
        for ev in evs[head + 1 :]:
            if ev["event"] == "copy_slot" and ev["src"] == slot:
                slot, followed = ev["dst"], followed + 1
            elif ev["event"] == "prefill_chunk" and ev["start"] > 0:
                assert (
                    ev["slot"] == slot
                ), f"c's continuation ran in slot {ev['slot']}, its state is in {slot}: {chunks}"
        evidence(
            prefill_chunks_slot_start_n=chunks,
            state_moves_followed=followed,
            finish_steps={k: r["finish_step"] for k, r in rec.items()},
        )
    finally:
        e.close()


def test_engine_chunked_continuation_negative_control_on_base_plugin(evidence):
    """The same workload on the unmodified pinned plugin (a723030, worktree work-09a/plugin-base): its prefill slot
    allocation moves the continuation to a fresh slot and the adapter's slot check stops the engine. Proves the
    positive test above exercises the bug."""
    base = os.environ.get("GLM53_09A_PLUGIN_BASE_SRC", "/opt/glm53/work-09a/plugin-base/src")
    if not os.path.isdir(base):
        pytest.fail(f"base plugin checkout missing: {base}")
    code = textwrap.dedent(
        """
        import vllm_tt_plugin, sys
        assert vllm_tt_plugin.__file__.startswith(sys.argv[1]), vllm_tt_plugin.__file__
        from models.autoports.zai_org_glm_5_3_flash.tests.vllm import test_vllm_engine_mock as t
        e = t.Engine(max_num_seqs=4, max_num_batched_tokens=256)
        try:
            e.run(t._slot_reuse_schedule())
        except Exception as exc:
            print("CONTROL_FAILED_AS_EXPECTED", type(exc).__name__, str(exc)[:400])
        else:
            print("CONTROL_DID_NOT_FAIL")
        """
    )
    # the base plugin has no GLM registration: use its per-launch override. The bare architecture name registers
    # both forms; a TT-prefixed key registers only the TT alias, which the nested glm5_next config cannot resolve.
    env = dict(
        os.environ,
        PYTHONPATH=f"{base}:{os.environ.get('PYTHONPATH', '')}",
        TT_MODEL_CLASS_OVERRIDES="Glm5NextForConditionalGeneration="
        "models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm:Glm5NextForCausalLM",
    )
    proc = subprocess.run([sys.executable, "-c", code, base], env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert "CONTROL_FAILED_AS_EXPECTED" in out, out[-3000:]
    line = next(l for l in out.splitlines() if "CONTROL_FAILED_AS_EXPECTED" in l)
    assert "KDA state" in line, line
    evidence(base_plugin=base, control_error=line[:400])


# ------------------------------------------------------------------------- long context, request limit lifted
def test_engine_long_prompt_kda_semantics_when_limit_lifted(evidence, monkeypatch, lifted_contract):
    """After the sparse DSA path lands the contract lifts the request limit (lifted_contract) and the generator
    computes every position (the mock's unbounded DSA); the serving path then takes prompts past the 2048-token
    physical chunk with 128-aligned vLLM chunks (here 5003 tokens at a 2048 budget and 3001 at a 512-token sub-budget
    share) and the generator's 2048-token sub-chunks."""
    monkeypatch.setenv(DSA_MODE_ENV, "unbounded")
    e = Engine(max_num_seqs=4, max_num_batched_tokens=2048)
    try:
        rec = e.run(
            [
                (0, "L5003", _prompt(901, 5003), SamplingParams(max_tokens=6, **GREEDY)),
                (0, "L3001", _prompt(902, 3001), SamplingParams(max_tokens=6, **GREEDY)),
                (2, "s70", _prompt(903, 70), SamplingParams(max_tokens=40, **GREEDY)),
            ]
        )
        assert e.check_greedy(rec) == {}
        _assert_clean_trace(e)
        chunks = [(c["slot"], c["start"], c["n"]) for c in e.events if c["event"] == "prefill_chunk"]
        assert max(s + n for _, s, n in chunks) == 5003
        evidence(prefill_chunks=chunks)
    finally:
        e.close()


# --------------------------------------------------------------------------- context ladder (rung contracts)
@pytest.fixture
def rung_contract(monkeypatch, tmp_path):
    """A context-ladder rung's contract (program 2026-10-21 02:57:57Z; a launch sets GLM53_CONTEXT_CONTRACT, read at
    import): the program contract with current_supported_context = the rung; here the loader and the class capability
    are patched in this process (the engine runs in-process)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv

    def make(supported: int) -> Path:
        data = json.loads(Path(gv.CONTEXT_CONTRACT_PATH).read_text())  # the program contract (conftest pins it)
        data["current_supported_context"] = supported
        path = tmp_path / f"context_contract_{supported}.json"
        path.write_text(json.dumps(data))
        load = gv.load_context_contract
        monkeypatch.setattr(gv, "load_context_contract", lambda p=path: load(p))
        limit = gv.load_context_contract().request_context_limit
        monkeypatch.setitem(gv.Glm5NextForCausalLM.model_capabilities, "tt_request_context_limit", limit)
        return path

    return make


def test_engine_ladder_rung_32k_long_prompt_chunked_prefill(evidence, expect_error, monkeypatch, rung_contract):
    """Context-ladder rung 32K (ladder_contract.md: 32K x 32 users; here 4 seqs): the rung contract (32,768),
    --max-model-len 32768, a KV pool of 4 x 32,768 tokens. A 32,000-token prompt prefills through vLLM's chunked
    prefill (2,048-token budget) next to short requests that keep decoding; greedy tokens equal the full-sequence
    reference; a 32,769-token prompt is refused at admission; the published per-user context table is the rung's."""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv

    monkeypatch.setenv(DSA_MODE_ENV, "unbounded")  # the lifted stack (the rung's sparse DSA path), not 2,051
    monkeypatch.setenv(KV_POOL_TOKENS_ENV, str(4 * 32768))
    rung_contract(32768)
    e = Engine(max_num_seqs=4, max_num_batched_tokens=2048, max_model_len=32768)
    try:
        vcfg = e.llm.llm_engine.vllm_config
        assert vcfg.model_config.max_model_len == 32768 and e.gen.supported_context == 32768
        assert vcfg.additional_config.get("_tt_request_context_limit") == 0  # the rung limit == max_model_len
        with expect_error(ValueError, "maximum context length is 32768"):
            e.llm.generate([TokensPrompt(prompt_token_ids=_prompt(5, 32769))], SamplingParams(max_tokens=1))
        rec = e.run(
            [
                (0, "s300", _prompt(951, 300), SamplingParams(max_tokens=60, **GREEDY)),
                (1, "L32000", _prompt(950, 32000), SamplingParams(max_tokens=8, **GREEDY)),
                (2, "s70", _prompt(952, 70), SamplingParams(max_tokens=40, **GREEDY)),
            ]
        )
        assert e.check_greedy(rec) == {}
        _assert_clean_trace(e)
        chunks = [(c["slot"], c["start"], c["n"]) for c in e.events if c["event"] == "prefill_chunk"]
        long_slot = next(s for s, st, n in chunks if st + n == 32000)
        long_chunks = sorted((st, n) for s, st, n in chunks if s == long_slot)
        assert long_chunks[0][0] == 0 and all(a + n == b for (a, n), (b, _) in zip(long_chunks, long_chunks[1:]))
        table = gv.context_by_users(e.runner.model.cache_spec.num_blocks, 32768, 0, 4)
        assert table[1] == 32768 and table[4] == 32768, table
        evidence(long_prompt=32000, long_chunks=len(long_chunks), kv_num_blocks=e.runner.model.cache_spec.num_blocks,
                 context_by_users=table, requests=len(rec))  # fmt: skip
    finally:
        e.close()


# ------------------------------------------------------------------------------------- first serving profile
def test_engine_first_serve_profile_max_model_len_2048(evidence, expect_error):
    """The first serving profile (DECISIONS 2026-10-17 ~20:45Z): --max-model-len 2048, 32 sequences, the generator in
    its dense-exact regime. 2048 < the contract's 2051 request limit, so the plugin resolves no extra limit and vLLM's
    own max_model_len bounds every request: a 2049-token prompt is refused, a default max_tokens stops at 2048."""
    e = Engine(max_num_seqs=32, max_num_batched_tokens=2048, max_model_len=2048)
    try:
        vcfg = e.llm.llm_engine.vllm_config
        assert vcfg.model_config.max_model_len == 2048 and e.gen.supported_context == 2048
        assert vcfg.additional_config.get("_tt_request_context_limit") == 0
        with expect_error(ValueError, "maximum context length is 2048"):
            e.llm.generate([TokensPrompt(prompt_token_ids=_prompt(1, 2049))], SamplingParams(max_tokens=1))
        lengths = [1 + 61 * i + (i % 7) for i in range(31)]
        sched = [
            (i // 4, f"f{i}", _prompt(600 + i, n), SamplingParams(max_tokens=8 + 3 * (i % 9), **GREEDY))
            for i, n in enumerate(lengths)
        ]
        sched.append((2, "edge2000", _prompt(640, 2000), SamplingParams(max_tokens=None, **GREEDY)))
        rec = e.run(sched)
        assert e.check_greedy(rec) == {}
        assert len(rec["edge2000"]["tokens"]) == 48 and rec["edge2000"]["finish_reason"] == "length"
        _assert_clean_trace(e)
        widest = max(ev["active"] for ev in e.events if ev["event"] == "decode_step")
        evidence(
            max_model_len=2048,
            max_num_seqs=32,
            resolved_request_limit=0,
            kv_num_blocks=e.runner.model.cache_spec.num_blocks,
            edge_prompt_2000_default_max_tokens=len(rec["edge2000"]["tokens"]),
            widest_decode_rows=widest,
        )
    finally:
        e.close()


def test_engine_serve_profile_max_model_len_2051(evidence, expect_error):
    """The Stage 6 / int3 serving profile: --max-model-len 2051 = the dense-exact program contract's
    current_supported_context (tests/vllm/contracts/context_contract_dense_exact_2051.json, pinned by conftest: the
    largest length the dense-exact DSA path computes faithfully), 32 sequences, async scheduling. vLLM's
    max_model_len equals the contract limit, so the plugin resolves no extra limit: a 2052-token prompt is refused,
    a default max_tokens stops at 2051 total, and requests ending exactly at 2051 complete although async scheduling
    looks one position ahead (that step consumes position 2050, the last the generator holds)."""
    e = Engine(max_num_seqs=32, max_num_batched_tokens=2048, max_model_len=2051)
    try:
        vcfg = e.llm.llm_engine.vllm_config
        assert vcfg.model_config.max_model_len == 2051 and e.gen.supported_context == 2051
        assert vcfg.scheduler_config.async_scheduling
        assert vcfg.additional_config.get("_tt_request_context_limit") == 0
        with expect_error(ValueError, "maximum context length is 2051"):
            e.llm.generate([TokensPrompt(prompt_token_ids=_prompt(2, 2052))], SamplingParams(max_tokens=1))
        lengths = [1 + 63 * i + (i % 5) for i in range(29)]
        sched = [
            (i // 4, f"f{i}", _prompt(700 + i, n), SamplingParams(max_tokens=6 + 3 * (i % 7), **GREEDY))
            for i, n in enumerate(lengths)
        ]
        sched += [
            (1, "edge2000", _prompt(740, 2000), SamplingParams(max_tokens=None, **GREEDY)),
            (2, "edge2050", _prompt(741, 2050), SamplingParams(max_tokens=1, **GREEDY)),
            (3, "edge1990", _prompt(742, 1990), SamplingParams(max_tokens=61, **GREEDY)),
        ]
        rec = e.run(sched)
        assert e.check_greedy(rec) == {}
        assert len(rec["edge2000"]["tokens"]) == 51 and rec["edge2000"]["finish_reason"] == "length"
        assert len(rec["edge2050"]["tokens"]) == 1 and len(rec["edge1990"]["tokens"]) == 61
        top = max(max(p for p in ev["positions"]) for ev in e.events if ev["event"] == "decode_step")
        assert top <= 2050, top
        _assert_clean_trace(e)
        evidence(
            max_model_len=2051,
            max_num_seqs=32,
            resolved_request_limit=0,
            kv_num_blocks=e.runner.model.cache_spec.num_blocks,
            edge_prompt_2000_default_max_tokens=len(rec["edge2000"]["tokens"]),
            highest_decode_position=top,
            requests=len(rec),
        )
    finally:
        e.close()


# --------------------------------------------------------------------------------------- default sampling route
def test_engine_greedy_only_backend_routes_sampled_requests_to_host(evidence, monkeypatch):
    """A backend whose device sampler is greedy only (max_device_top_k 1, a sampled row refused -- the Stage 14b stack)
    narrows the adapter instance's max_device_top_k, so the plugin routes the profile's default sampled requests
    (T 1.0, top_k 32) to vLLM's host sampler over the read-back logits, while greedy requests stay on the device
    sampler; the class capability is unchanged."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import Glm5NextForCausalLM

    monkeypatch.setenv(GREEDY_ONLY_ENV, "1")
    e = Engine(max_num_seqs=4, max_num_batched_tokens=512, override_generation_config={"top_k": 32})
    try:
        assert e.runner.model.model_capabilities["max_device_top_k"] == 1
        assert Glm5NextForCausalLM.model_capabilities["max_device_top_k"] == 32
        keys = ("logits_readback", "sampling_trace_replays", "fused_sampling_steps")
        rows = {}
        for name, params in (("sampled", e.llm.get_default_sampling_params()), ("greedy", SamplingParams(**GREEDY))):
            params = params.clone()
            params.max_tokens, params.ignore_eos = 10, True
            before = dict(e.gen.counters)
            out = e.llm.generate(
                [TokensPrompt(prompt_token_ids=_prompt(1100 + i, 40 + 7 * i)) for i in range(3)], params
            )
            assert all(len(o.outputs[0].token_ids) == 10 for o in out)
            rows[name] = {k: e.gen.counters.get(k, 0) - before.get(k, 0) for k in keys}
        assert rows["sampled"]["logits_readback"] > 0 and _device_sampled_steps(rows["sampled"]) == 0
        assert rows["greedy"]["logits_readback"] == 0 and _device_sampled_steps(rows["greedy"]) > 0
        evidence(**rows)
    finally:
        e.close()


def test_engine_default_top_k_override_keeps_default_requests_on_device(evidence):
    """The checkpoint's generation defaults (T 1.0, top_p 0.95, no top_k) send every default request -- and the whole
    decode batch with it -- to vLLM's host sampler (full-logits readback), because top_k unset exceeds
    max_device_top_k. The serving profile's `--override-generation-config '{"top_k": 32}'` keeps them on the device
    sampler (06a: the default nucleus never exceeds 11 tokens). Control: the same requests without the override."""
    results = {}
    for override in (True, False):
        kwargs = {"override_generation_config": {"top_k": 32}} if override else {}
        e = Engine(max_num_seqs=4, max_num_batched_tokens=512, **kwargs)
        try:
            defaults = e.llm.get_default_sampling_params()
            params = defaults.clone()
            params.max_tokens, params.ignore_eos = 12, True
            before = dict(e.gen.counters)
            out = e.llm.generate(
                [TokensPrompt(prompt_token_ids=_prompt(1000 + i, 60 + 9 * i)) for i in range(3)], params
            )
            assert all(len(o.outputs[0].token_ids) == 12 for o in out)
            keys = ("logits_readback", "sampling_trace_replays", "fused_sampling_steps")
            delta = {k: e.gen.counters.get(k, 0) - before.get(k, 0) for k in keys}
            results[override] = {
                "defaults": {"temperature": defaults.temperature, "top_p": defaults.top_p, "top_k": defaults.top_k},
                **delta,
            }
            _assert_clean_trace(e)
        finally:
            e.close()
    on, off = results[True], results[False]
    assert on["defaults"]["top_k"] == 32 and on["defaults"]["temperature"] == pytest.approx(1.0)
    assert on["defaults"]["top_p"] == pytest.approx(0.95)
    assert on["logits_readback"] == 0 and _device_sampled_steps(on) > 0
    assert off["logits_readback"] > 0, "control: without the override the default requests are host-sampled"
    evidence(with_override=on, without_override=off)


# ------------------------------------------------------------------------------------------------- preemption
def test_engine_preemption_resumes_by_recompute(evidence, monkeypatch):
    """A KV pool too small for the running set (GLM53_VLLM_KV_POOL_TOKENS) makes vLLM preempt: the plugin releases
    the request's slot (release_request -> Glm53Generator.release_slot), vLLM frees its blocks, and the request later
    resumes with a fresh prefill of prompt + generated tokens into whatever slot is free. Greedy outputs stay equal
    to the oracle."""
    monkeypatch.setenv(KV_POOL_TOKENS_ENV, "2048")
    e = Engine(max_num_seqs=4, max_num_batched_tokens=1024, max_model_len=2048)
    try:
        prompts = {f"p{i}": _prompt(700 + i, 380 + 9 * i) for i in range(4)}
        rec = e.run([(0, rid, p, SamplingParams(max_tokens=320, **GREEDY)) for rid, p in prompts.items()])
        assert e.check_greedy(rec) == {}
        _assert_clean_trace(e)
        fresh = [ev for ev in e.events if ev["event"] == "prefill_chunk" and ev["start"] == 0]
        resumed = len(fresh) - len(prompts)
        assert resumed > 0, "the pool was meant to force at least one preemption"
        counters = dict(e.runner.model.counters)
        assert counters["slot_releases"] >= resumed + len(prompts) - 1
        evidence(
            kv_pool_tokens=2048,
            kv_num_blocks=e.runner.model.cache_spec.num_blocks,
            admissions=len(fresh),
            resumed_after_preemption=resumed,
            adapter_counters=counters,
            outputs_match_oracle=True,
        )
    finally:
        e.close()


# -------------------------------------------------------------------------------------- seeded sampling state
def _spy_sampling_commands(model, forward_reset=False):
    """Count the reset_sampling_state commands the adapter receives; with `forward_reset` (the control) also pass the
    command on to Glm53Generator.decode_forward, which then re-seeds every decoding slot."""
    seen = {"resets": 0, "last": False}
    adapter_decode, gen_decode = model.decode_forward, model.generator.decode_forward

    def adapter(*a, **kw):
        seen["last"] = bool(kw.get("reset_sampling_state"))
        seen["resets"] += seen["last"]
        return adapter_decode(*a, **kw)

    def generator(*a, **kw):
        return gen_decode(*a, reset_sampling_state=seen["last"], **kw)

    model.decode_forward = adapter
    if forward_reset:
        model.generator.decode_forward = generator
    return seen


@pytest.mark.parametrize("widths", ["pow2", "per_width"])
def test_engine_seeded_sampling_state_follows_the_request(evidence, monkeypatch, widths):
    """reset_sampling_state ("rebuild RNG state for the current layout") is executed by the layout: each request's
    device RNG lives in its generator slot from admission (seeded there) and slot_remap is an indirection, so a seeded
    request samples the same tokens alone and amid churn (requests joining and leaving -> transitions with
    reset_sampling_state). A host-sampling interlude (a logprob request makes the batch host-sampled) does not
    re-seed it either. Control: forwarding the command to the generator re-seeds at every transition and the
    seeded stream changes. "per_width": the backend has widths 1 / 2 / 4 + 8 and its copy_slot re-seeds (like
    tt/per_width.py), so the adapter keeps seeded requests in place while it compacts the others."""
    if widths == "per_width":
        monkeypatch.setenv(PER_WIDTH_ENV, "1,2,4")
    e = Engine(max_num_seqs=8, max_num_batched_tokens=512)
    try:
        model = e.runner.model
        seeded = SamplingParams(max_tokens=40, temperature=1.0, top_k=32, seed=1234, ignore_eos=True)
        prompt = _prompt(800, 90)

        def churn(tag):
            return [
                (0, tag, prompt, seeded),
                (3, f"{tag}c1", _prompt(801, 70), SamplingParams(max_tokens=10, **GREEDY)),
                (5, f"{tag}c2", _prompt(802, 130), SamplingParams(max_tokens=25, **GREEDY)),
                (9, f"{tag}c3", _prompt(803, 33), SamplingParams(max_tokens=5, **GREEDY)),
                (14, f"{tag}c4", _prompt(804, 64), SamplingParams(max_tokens=12, temperature=0.9, top_k=8, seed=5)),
            ]

        def phase(sched, spy):
            seeds0, resets0 = e.gen.counters["seed_writes"], spy["resets"]
            rec = e.run(sched)
            return rec, e.gen.counters["seed_writes"] - seeds0, spy["resets"] - resets0

        spy = _spy_sampling_commands(model)
        alone, seeds_alone, _ = phase([(0, "A1", prompt, seeded)], spy)
        churned, seeds_churn, resets_churn = phase(churn("A2"), spy)
        assert churned["A2"]["tokens"] == alone["A1"]["tokens"]
        assert resets_churn >= 3 and seeds_churn == 5, (resets_churn, seeds_churn)  # one seed write per admission
        assert e.check_greedy({k: v for k, v in churned.items() if k in ("A2c1", "A2c2", "A2c3")}) == {}
        readback0 = e.gen.counters["logits_readback"]
        lp = SamplingParams(max_tokens=8, temperature=0.0, logprobs=1, ignore_eos=True)
        mixed, seeds_mixed, _ = phase([(0, "A3", prompt, seeded), (4, "lp", _prompt(805, 50), lp)], spy)
        assert e.gen.counters["logits_readback"] > readback0, "the logprob request must make the batch host-sampled"
        assert seeds_mixed == 2 and mixed["A3"]["tokens"][:3] == alone["A1"]["tokens"][:3]
        _assert_clean_trace(e)
        # control: execute reset_sampling_state as a re-seed of every decoding slot
        model.decode_forward = type(model).decode_forward.__get__(model)
        control_spy = _spy_sampling_commands(model, forward_reset=True)
        control, seeds_control, resets_control = phase(churn("A4"), control_spy)
        assert control["A4"]["tokens"] != alone["A1"]["tokens"], "the control must see the re-seeding"
        assert seeds_control > 5
        diverge = next(i for i, (a, b) in enumerate(zip(control["A4"]["tokens"], alone["A1"]["tokens"])) if a != b)
        evidence(
            seeded_tokens_alone_equal_churn=True,
            reset_commands_in_churn=resets_churn,
            seed_writes={"alone": seeds_alone, "churn": seeds_churn, "host_interlude": seeds_mixed},
            control_reseed={"seed_writes": seeds_control, "reset_commands": resets_control, "first_diff": diverge},
        )
    finally:
        e.close()


# ------------------------------------------------------------------------------------- pow2 decode buckets
def test_engine_pow2_bucket_slot_packing(evidence):
    """The generator decodes bucketed widths (`decode_buckets` (1, 2, 4, 8) at 8 slots), so the adapter keeps live
    requests in the lowest generator slots under vLLM's real scheduling: at every decode step the highest live slot
    is below the bucket covering the live count, and the generator replays the narrower traces. Survivors move with
    Glm53Generator.remap_slots (copy_slot); greedy outputs equal the oracle."""
    e = Engine(max_num_seqs=8, max_num_batched_tokens=512)
    try:
        model = e.runner.model
        assert model.pack_slots and e.gen.decode_buckets == (1, 2, 4, 8)
        widths = []
        decode = model.generator.decode_forward

        def spy(tokens, start_pos, **kw):
            live = sorted(model.slot_table.slots)
            widths.append((len(live), max(live) + 1 if live else 0))
            return decode(tokens, start_pos, **kw)

        model.generator.decode_forward = spy
        sched = [
            *[
                (0, f"w{i}", _prompt(900 + i, 30 + 41 * i), SamplingParams(max_tokens=6 + 9 * i, **GREEDY))
                for i in range(8)
            ],
            *[
                (40 + 6 * i, f"x{i}", _prompt(950 + i, 50 + 13 * i), SamplingParams(max_tokens=20, **GREEDY))
                for i in range(4)
            ],
        ]
        rec = e.run(sched)
        assert e.check_greedy(rec) == {}
        pow2 = lambda n: 1 << max(n - 1, 0).bit_length()
        assert all(top <= pow2(n) for n, top in widths), [w for w in widths if w[1] > pow2(w[0])]
        _assert_clean_trace(e)
        moves = [(ev["src"], ev["dst"]) for ev in e.events if ev["event"] == "copy_slot"]
        assert moves and model.counters["compaction_moves"] == len(moves)
        replays = dict(e.gen._binding(model.kv_cache).width_replays)
        assert all(replays[w] > 0 for w in (1, 2, 4, 8)), replays  # every width served steps of this workload
        evidence(
            compaction_moves=moves,
            width_replays=replays,
            slot_swaps=model.counters["slot_swaps"],
            decode_steps=len(widths),
            live_vs_highest_slot_sample=widths[:: max(1, len(widths) // 20)],
        )
    finally:
        e.close()
