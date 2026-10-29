# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving robustness (SRV lifecycle, 2026-10-22): the request lifecycle through the real vLLM V1 engine (in-process),
the TT plugin (scheduler, runner, async decode), the GLM-5.3-Flash adapter, Glm53Generator and the CPU MockModel --
no TT device (engine_harness.fake_tt_mesh). Every greedy output is checked token for token against the unpaged oracle
and, across runs, bitwise against the no-abort run; slot hygiene against the mock device's own SlotLedger.

1 / 8  abort / cancel mid-generation at random steps (mid packed prefill and mid decode) among 32 mixed-ISL users
2      per-request wall-clock timeouts and exact max_tokens at the scheduler step
3      prompts over max_model_len refused before any device call
4      n > 1 and stop strings / stop token ids: host-side, no extra device step
6      logprobs parity: host full-vocab logprobs vs the device sampling path and the oracle's top-K
7      a device exception latches the engine unhealthy; no retry, no device re-open
(5, the streamed first token, and the engine exit status are HTTP checks: test_srv_lifecycle_http.py)
"""

from __future__ import annotations

import random
import time

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    FAULT_DECODE_STEP_ENV,
    prompt_ids,
    reference_greedy,
    tokenizer,
)

H.engine_env()

from vllm import SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import test_vllm_engine_mock as T  # noqa: E402

GREEDY = dict(temperature=0.0, ignore_eos=True)


def _mixed_isl(seed: int, n: int = 32, lo: int = 5, hi: int = 1900) -> list[int]:
    """Distinct prompt lengths across the served ISL range (5 .. 1,900 of the 2,051 context), seeded."""
    rnd = random.Random(seed)
    lens = sorted(rnd.sample(range(lo, hi), n))
    return [lens[(i * 7) % n] for i in range(n)]  # shuffled so arrival order is not length order


def _core(e):
    return e.llm.llm_engine.engine_core.engine_core


def _ledger(e):
    return e.backend.ledger


def _decode_steps(e) -> int:
    return sum(1 for ev in e.events if ev["event"] == "decode_step")


def run_with_aborts(e, schedule, aborts: dict[str, int], max_steps=20000):
    """Like Engine.run, plus `aborts` {request_id: step}: the request is aborted (LLMEngine.abort_request = the API
    server's client-disconnect path) just before that step. Returns (records, abort_log)."""
    eng = e.llm.llm_engine
    pending = sorted(schedule, key=lambda x: x[0])
    rec, log = {}, {}
    step = 0
    while pending or eng.has_unfinished_requests():
        while pending and pending[0][0] <= step:
            _, rid, prompt, params = pending.pop(0)
            eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt), params)
            rec[rid] = {"prompt_len": len(prompt), "arrival_step": step, "prompt": prompt, "tokens": []}
        for rid, at in aborts.items():
            if at == step and rid in rec and "finish_step" not in rec[rid] and rid not in log:
                # the engine core keys requests by its internal id: find the live request by its (distinct) prompt
                by_prompt = {tuple(r.prompt_token_ids): r for r in _core(e).scheduler.requests.values()}
                req = by_prompt.get(tuple(rec[rid]["prompt"]))
                log[rid] = {
                    "step": step,
                    "status": str(req.status) if req else "gone",
                    "computed": int(req.num_computed_tokens) if req else None,
                    "prompt_len": rec[rid]["prompt_len"],
                    "streamed": len(rec[rid]["tokens"]),
                }
                eng.abort_request([rid])
        for out in eng.step():
            r = rec[out.request_id]
            r["tokens"] = list(out.outputs[0].token_ids)
            if r["tokens"] and "first_token_step" not in r:
                r["first_token_step"] = step
            if out.finished:
                r.update(finish_step=step, finish_reason=out.outputs[0].finish_reason)
        step += 1
        if step > max_steps:
            raise RuntimeError(f"workload did not finish in {max_steps} steps")
    return rec, log


def _check_oracle(e, rec, ids):
    bad = {}
    for rid in ids:
        r = rec[rid]
        ref = reference_greedy(e.backend, r["prompt"], len(r["tokens"]))
        if r["tokens"] != ref:
            bad[rid] = next((i for i, (a, b) in enumerate(zip(r["tokens"], ref)) if a != b), len(r["tokens"]))
    return bad


def _slots_clean(e):
    """Slot leaks, in every ledger: the plugin's ownership map, the adapter's mirror, the generator's slots and the
    mock device's. A request finished in the LAST step is released by the NEXT step's `_update_states` (vLLM hands
    `finished_req_ids` to the runner with the following SchedulerOutput; the engine idles when nothing is left), so
    the slots of exactly those requests (`scheduler.finished_req_ids`, still undrained) are legitimately held and
    are subtracted: anything else held is a leak. Returns the non-empty leak views (empty dict = clean)."""
    sched = _core(e).scheduler
    deferred_reqs = set(sched.finished_req_ids)
    plugin_map = dict(e.runner._req_state_slot)
    model = e.runner.model
    deferred_vslots = {plugin_map[r] for r in deferred_reqs if r in plugin_map}
    deferred_gslots = {model.gen_slot[v] for v in deferred_vslots}
    views = {
        "plugin_req_state_slot": sorted(set(plugin_map) - deferred_reqs),
        "adapter_slot_table": sorted(set(model.slot_table.slots) - deferred_gslots),
        "generator_active_slots": sorted(
            {i for i, st in enumerate(e.gen._binding(model.kv_cache).slots) if st.active} - deferred_gslots
        ),
        "mock_ledger_held": sorted(set(_ledger(e).held) - deferred_gslots),
        "mock_ledger_violations": list(_ledger(e).violations),
    }
    return {k: v for k, v in views.items() if v}


# =============================================================================== 1 + 8: aborts among 32 mixed-ISL users
def test_lifecycle_32_mixed_isl_users_with_8_random_aborts(evidence):
    """Run A: 32 users, mixed ISL, no aborts (the reference). Run B: the same 32 prompts with 8 aborts injected at
    seeded random steps -- during the multi-step packed prefill (co-packed requests keep going) and mid-decode. The
    24 survivors' tokens are bitwise those of run A (and the oracle's); the aborted requests never finish; every
    slot record (plugin map, adapter mirror, generator, mock ledger) is empty afterwards, with no violation (no
    leak, no double release, no fresh prefill into a held or un-zeroed slot). Run C: 32 fresh users reuse the slots
    and are oracle-exact (the next request on an aborted slot is clean)."""
    e = T.Engine(max_num_seqs=32, max_num_batched_tokens=2048)
    try:
        lens = _mixed_isl(1022)
        outs = [12 + (i * 5) % 21 for i in range(32)]  # 12 .. 32 output tokens
        prompts = {i: prompt_ids(2000 + i, n) for i, n in enumerate(lens)}

        def schedule(tag):
            return [(0, f"{tag}{i}", prompts[i], SamplingParams(max_tokens=outs[i], **GREEDY)) for i in range(32)]

        rec_a, _ = run_with_aborts(e, schedule("a"), {})
        assert _check_oracle(e, rec_a, rec_a) == {}
        assert all(len(rec_a[f"a{i}"]["tokens"]) == outs[i] for i in range(32))
        clean_a = _slots_clean(e)
        assert clean_a == {}, clean_a

        rnd = random.Random(8)
        victims = sorted(rnd.sample(range(32), 8))
        # the packed prefill of ~30k prompt tokens spans ~15 steps: half the aborts land in it, half in decode
        aborts = {f"b{i}": (rnd.randint(1, 8) if j % 2 == 0 else rnd.randint(14, 30)) for j, i in enumerate(victims)}
        rec_b, log = run_with_aborts(e, schedule("b"), aborts)
        survivors = [f"b{i}" for i in range(32) if i not in victims]
        mismatch = {s: rec_b[s]["tokens"] != rec_a["a" + s[1:]]["tokens"] for s in survivors}
        assert not any(mismatch.values()), {s for s, m in mismatch.items() if m}
        assert _check_oracle(e, rec_b, survivors) == {}
        for i in victims:
            assert "finish_step" not in rec_b[f"b{i}"], (i, rec_b[f"b{i}"])
        assert len(log) == 8, log
        mid_prefill = [r for r, l in log.items() if l["computed"] is not None and l["computed"] < l["prompt_len"]]
        mid_decode = [r for r, l in log.items() if l["streamed"] > 0]
        assert mid_prefill and mid_decode, log
        clean_b = _slots_clean(e)
        assert clean_b == {}, clean_b
        assert not e.llm.llm_engine.has_unfinished_requests()
        T._assert_clean_trace(e)

        rec_c, _ = run_with_aborts(e, schedule("c"), {})
        assert _check_oracle(e, rec_c, rec_c) == {}
        assert all(rec_c[f"c{i}"]["tokens"] == rec_a[f"a{i}"]["tokens"] for i in range(32))
        led = _ledger(e).summary()
        assert _slots_clean(e) == {} and led["violations"] == []
        # every admission was released (minus the last step's deferred releases); a request aborted while still
        # WAITING (never prefilled) took no slot, so run B admitted fewer than 32
        never_prefilled = [r for r, l in log.items() if l["computed"] == 0 and l["streamed"] == 0]
        assert led["occupies"] == 96 - len(never_prefilled), (led, never_prefilled)
        assert led["occupies"] - led["releases"] == len(_core(e).scheduler.finished_req_ids), led
        evidence(
            prompt_lens=lens, output_lens=outs, aborts=log, mid_prefill_aborts=sorted(mid_prefill),
            never_prefilled_aborts=sorted(never_prefilled),
            mid_decode_aborts=sorted(mid_decode), survivors_bitwise=len(survivors), ledger=led,
            adapter_counters=dict(e.runner.model.counters), generator_counters=dict(e.gen.counters),
        )  # fmt: skip
    finally:
        e.close()


# ====================================================================================== 2: timeouts and max_tokens
def test_lifecycle_request_timeout_and_exact_max_tokens(evidence):
    """tt.request_timeout_s = 200 with the scheduler's clock driven 10 s per engine step: requests that need more
    than ~20 steps finish with finish_reason 'abort' (stop_reason 'timeout'), their slots released, while the others
    finish 'length' with exactly their max_tokens (1, 5, 13 tokens) and oracle-exact tokens."""
    e = T.Engine(max_num_seqs=8, max_num_batched_tokens=512, tt_config={"request_timeout_s": 200})
    try:
        sched = _core(e).scheduler
        assert sched._request_timeout_s == 200.0
        clock = {"steps": 0}
        sched._tt_clock = lambda: time.time() + 10.0 * clock["steps"]
        eng = e.llm.llm_engine
        short = {f"s{k}": k for k in (1, 5, 13)}
        reqs = {**short, "long0": 400, "long1": 400}
        prompts = {rid: prompt_ids(300 + i, 40 + 17 * i) for i, rid in enumerate(reqs)}
        for rid, n in reqs.items():
            eng.add_request(rid, TokensPrompt(prompt_token_ids=prompts[rid]), SamplingParams(max_tokens=n, **GREEDY))
        rec = {}
        while eng.has_unfinished_requests():
            for out in eng.step():
                if out.finished:
                    rec[out.request_id] = (list(out.outputs[0].token_ids), out.outputs[0].finish_reason)
            clock["steps"] += 1
            assert clock["steps"] < 200
        for rid, n in short.items():
            toks, why = rec[rid]
            assert why == "length" and len(toks) == n, (rid, why, len(toks))
            assert toks == reference_greedy(e.backend, prompts[rid], n)
        for rid in ("long0", "long1"):
            toks, why = rec[rid]
            assert why == "abort" and 0 < len(toks) < 400, (rid, why, len(toks))
            assert toks == reference_greedy(e.backend, prompts[rid], len(toks))
        clean = _slots_clean(e)
        assert clean == {}, clean
        evidence(finish={r: (len(t), w) for r, (t, w) in rec.items()}, steps=clock["steps"],
                 ledger=_ledger(e).summary())  # fmt: skip
    finally:
        e.close()


# ============================================================================================ 3: input limits
def test_lifecycle_prompts_over_max_model_len_are_refused_before_any_device_call(evidence, expect_error):
    """The served profile (max_model_len 2051 = the DSA request limit): a 2,052-token prompt and a 2,051-token
    prompt (no room for one output token) raise ValueError at add_request -- the generator sees no prefill call --
    and a 2,050-token prompt with max_tokens 1 is served; aborting a request that is still WAITING touches no slot."""
    e = T.Engine(max_num_seqs=2, max_num_batched_tokens=2048, max_model_len=2051)
    try:
        eng = e.llm.llm_engine
        before = dict(e.gen.counters)
        errors = {}
        for n in (2052, 2051):
            with expect_error(ValueError, "2051") as exc:
                eng.add_request(f"over{n}", TokensPrompt(prompt_token_ids=prompt_ids(n, n)),
                                SamplingParams(max_tokens=4, **GREEDY))  # fmt: skip
            errors[n] = str(exc.value)[:160]
            assert "2051" in errors[n] and "maximum" in errors[n], errors[n]
        assert e.gen.counters["prefill_chunks"] == before["prefill_chunks"]
        assert not eng.has_unfinished_requests()
        # a request aborted while WAITING (never prefilled): no slot was ever taken or released
        releases = e.runner.model.counters["slot_releases"]
        eng.add_request("w", TokensPrompt(prompt_token_ids=prompt_ids(9, 9)), SamplingParams(max_tokens=4, **GREEDY))
        eng.abort_request(["w"])
        assert not eng.has_unfinished_requests() and e.runner.model.counters["slot_releases"] == releases
        rec = e.run([(0, "fit", prompt_ids(2050, 2050), SamplingParams(max_tokens=1, **GREEDY))])
        assert rec["fit"]["tokens"] == reference_greedy(e.backend, prompt_ids(2050, 2050), 1)
        assert rec["fit"]["finish_reason"] == "length"
        evidence(max_model_len=2051, refused=errors, served_2050_plus_1=rec["fit"]["tokens"])
    finally:
        e.close()


# =================================================================================== 4: n > 1, stop strings / ids
def test_lifecycle_n_sequences_and_stop_conditions_are_host_side(evidence):
    """stop_token_ids and stop strings end a request from the scheduler / detokenizer (finish 'stop'), the device
    only ever runs the steps the longest live request needs; n = 2 (seeded sampling) fans out on the host into two
    engine requests (two slots), deterministic across runs."""
    e = T.Engine(max_num_seqs=8, max_num_batched_tokens=512)
    try:
        tok = tokenizer()
        p = prompt_ids(77, 60)
        ref = reference_greedy(e.backend, p, 24)
        k = 7
        stop_text = tok.decode(ref[k : k + 1])
        assert stop_text.strip(), (k, ref[k], stop_text)
        d0 = _decode_steps(e)
        rec = e.run(
            [
                (0, "ids", p, SamplingParams(max_tokens=24, stop_token_ids=[ref[k]], temperature=0.0)),
                (0, "str", p, SamplingParams(max_tokens=24, stop=[stop_text], temperature=0.0)),
                (0, "len", p, SamplingParams(max_tokens=24, **GREEDY)),
            ]
        )
        assert rec["len"]["tokens"] == ref and rec["len"]["finish_reason"] == "length"
        assert rec["ids"]["finish_reason"] == "stop" and rec["ids"]["tokens"] == ref[: k + 1], rec["ids"]
        assert rec["str"]["finish_reason"] == "stop" and rec["str"]["tokens"][:k] == ref[:k], rec["str"]
        assert len(rec["str"]["tokens"]) <= k + 1
        steps = _decode_steps(e) - d0
        assert steps <= 24 + 1, steps  # the stopped requests cost no device step beyond the 24-token one
        clean = _slots_clean(e)
        assert clean == {}, clean
        pairs = []
        for _ in range(2):
            out = e.llm.generate(
                [TokensPrompt(prompt_token_ids=p)],
                SamplingParams(n=2, max_tokens=10, temperature=0.8, top_k=20, seed=1234, ignore_eos=True),
            )[0]
            pairs.append([list(c.token_ids) for c in out.outputs])
        assert len(pairs[0]) == 2 and all(len(t) == 10 for t in pairs[0]) and pairs[0] == pairs[1], pairs
        led = _ledger(e).summary()
        assert _slots_clean(e) == {} and led["violations"] == []
        evidence(stop_token=ref[k], stop_text=stop_text, finish={r: rec[r]["finish_reason"] for r in rec},
                 decode_steps=steps, n2_pairs=pairs[0], ledger=led)  # fmt: skip
    finally:
        e.close()


# ================================================================================================ 6: logprobs
def test_lifecycle_logprobs_parity_with_the_device_sampling_path(evidence):
    """The same greedy prompt through the device sampling path (no logprobs) and the host full-vocab path
    (logprobs=5) gives identical tokens; every served logprob step has the sampled token at rank 0 and its top-5
    ids equal the oracle's top-5 of the full-sequence logits (the candidate set a device top-K sampler sees); a
    logprob request batched with device-sampled users leaves all of them oracle-exact."""
    e = T.Engine(max_num_seqs=4, max_num_batched_tokens=512)
    try:
        p = prompt_ids(55, 90)
        n = 8
        dev = e.llm.generate([TokensPrompt(prompt_token_ids=p)], SamplingParams(max_tokens=n, **GREEDY))[0].outputs[0]
        host = e.llm.generate([TokensPrompt(prompt_token_ids=p)], SamplingParams(max_tokens=n, logprobs=5, **GREEDY))[
            0
        ].outputs[0]
        ref = reference_greedy(e.backend, p, n)
        assert list(dev.token_ids) == list(host.token_ids) == ref
        assert host.logprobs is not None and len(host.logprobs) == n
        import torch

        top_disagree = []
        for i, step in enumerate(host.logprobs):
            tok = ref[i]
            ranked = sorted(step.items(), key=lambda kv: kv[1].rank)
            assert ranked[0][0] == tok and step[tok].rank == 1, (i, ranked[:3])
            ref_logits = e.backend.reference_logits(p + ref[:i], last_only=True).reshape(-1)
            oracle_top = torch.topk(ref_logits.float(), 5).indices.tolist()
            host_top = [t for t, _ in ranked[:5]]
            if host_top != oracle_top:
                top_disagree.append((i, host_top, oracle_top))
        assert top_disagree == [], top_disagree
        rec = e.run(
            [(0, f"d{i}", prompt_ids(60 + i, 30 + 11 * i), SamplingParams(max_tokens=6, **GREEDY)) for i in range(3)]
            + [(0, "lp", prompt_ids(70, 44), SamplingParams(max_tokens=6, logprobs=3, **GREEDY))]
        )
        assert e.check_greedy(rec) == {}
        evidence(tokens=ref, first_logprobs=[round(s[ref[i]].logprob, 4) for i, s in enumerate(host.logprobs)],
                 mixed_batch_oracle_exact=True)  # fmt: skip
    finally:
        e.close()


# ========================================================================================== 7: device exception
def test_lifecycle_device_exception_latches_the_engine_unhealthy(evidence, monkeypatch, expect_error):
    """The mock raises at its 5th decode step. The engine step raises the fault; afterwards check_health raises
    TTDeviceFaultError, the next engine step is refused by the latch without any device call, the mesh was opened
    once and is closed once at shutdown -- nothing retries or re-opens."""
    from vllm_tt_plugin.model_runner import TTDeviceFaultError

    monkeypatch.setenv(FAULT_DECODE_STEP_ENV, "5")
    e = T.Engine(max_num_seqs=4, max_num_batched_tokens=512)
    try:
        assert e.backend.fault_decode_step == 5
        eng = e.llm.llm_engine
        for i in range(2):
            eng.add_request(f"f{i}", TokensPrompt(prompt_token_ids=prompt_ids(80 + i, 50)),
                            SamplingParams(max_tokens=20, **GREEDY))  # fmt: skip
        err = None
        for _ in range(40):
            try:
                eng.step()
            except Exception as exc:  # the device fault, through the engine core
                err = exc
                break
        assert err is not None and "injected device fault" in str(err), err
        assert e.runner.device_fault is not None
        with expect_error(TTDeviceFaultError, "no retry"):
            e.runner.check_health()
        with expect_error(TTDeviceFaultError, "unhealthy"):  # the worker's health probe (executor check_health)
            _core(e).model_executor.driver_worker.worker.check_health()
        steps = _decode_steps(e)
        faults = [ev for ev in e.events if ev["event"] == "device_fault"]
        with expect_error(TTDeviceFaultError, "previously failed"):
            eng.step()
        assert _decode_steps(e) == steps and len(faults) == 1, "a step reached the device after the fault"
        # the latch closes the mesh at the fault (plugin P2-5: the exit report reads its close marker), once
        assert len(e.meshes) == 1 and e.meshes[0].closed
        evidence(fault=str(err)[:200], decode_steps_before_fault=steps, meshes_opened=len(e.meshes))
        del err  # its traceback pins the engine
    finally:
        _core(e).shutdown()  # what EngineCoreProc's finally does after the fatal error: executor -> worker.shutdown
        assert e.meshes[0].closed, "shutdown must close the mesh once"
        e.close()
    assert len(e.meshes) == 1, "a second mesh was opened"
