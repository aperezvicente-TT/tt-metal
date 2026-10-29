# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 S4 host tests: the adapter's speculative-decoding path (tt/generator_vllm.py spec_plan / decode_forward
verify / propose_draft_tokens) and the generator's spec_enter / spec_step / spec_leave over 06a's CPU MockModel, driven
the way the plugin's runner drives a model-owned drafter (vllm_tt_plugin.spec_decode: VerifyOutput / DraftOutput, the
runner's own accept_greedy_drafts).

`MockSpec` stands in for tt/spec_mtp.SpecMTP's host API (K, set_page_table, enter, step, read, leave) with the
device step's semantics on the mock backend: the verify consumes [bonus, d1..dK] at positions c..c+K of slot 0 and
stops at the first rejection (a_{j-1} != d_j) — the committed state is exactly the accepted prefix, which is what the
device's candidate rows + selector deliver — the bonus a_n is left in token row 0, and the next drafts come from a
drafter (here: the oracle's true continuation with controlled corruption, so rejections at every depth happen). The
served token stream must equal 06a's independent oracle (MockModel.reference_greedy) token for token.
"""

from __future__ import annotations

import random

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import prompt_ids
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_generator_vllm import Pages, _adapter, _sampling
from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv

spec_decode = pytest.importorskip("vllm_tt_plugin.spec_decode")


class MockSpec:
    def __init__(self, backend, kv, K: int, truth: dict, corrupt: float, seed: int = 7):
        self.be, self.kv, self.K = backend, kv, K
        self.truth = truth  # absolute position -> the oracle's token there (prompt + greedy continuation)
        self.corrupt, self.rng = corrupt, random.Random(seed)
        self.rec = None
        self.sel, self.mesh = None, None
        self.calls = dict(enter=0, step=0, leave=0)

    def set_page_table(self, row):
        pass

    def _drafts(self, c_next: int) -> list:
        """drafts for positions c_next + 1 .. c_next + K (the bonus sits at c_next): the truth, a wrong token at a
        random depth with probability `corrupt` (and every later one is then wrong too: a chain diverges)"""
        out, bad = [], False
        for i in range(self.K):
            t = self.truth.get(c_next + 1 + i, 1)
            bad = bad or self.rng.random() < self.corrupt
            out.append((t + 1 + i) % 2000 if bad else t)
        return out

    def _io(self):
        return self.kv.extra["io"]

    def enter(self, page_row):
        io = self._io()
        c = int(io.cur_pos._t[0]) - 1
        b = int(io.tokens._t[0])
        d = self._drafts(c + 1)
        self.rec = dict(n=0, c=c, a=[b] * (self.K + 1), committed=[b], drafts=d, steps=0)
        io.tokens._t[1 : 1 + self.K] = torch.tensor(d, dtype=torch.int32)
        self.calls["enter"] += 1

    def step(self):
        io, rec, K = self._io(), self.rec, self.K
        base = rec["c"] + rec["n"] + 1
        ins = [rec["a"][rec["n"]]] + list(rec["drafts"])
        a = []
        for j in range(K + 1):
            io.tokens._t[0] = ins[j]
            io.cur_pos._t[:] = -1
            io.cur_pos._t[0] = base + j
            self.be.decode(io, self.kv, 1)  # slot 0's plain decode at base + j (positions advance on device)
            row = io.logits._t[0, : self.be.vocab_size].double()
            a.append(int(torch.nonzero(row == row.max())[0]))  # greedy, lowest id on ties
            if j == K or a[j] != ins[j + 1]:
                break
        n = len(a) - 1
        io.tokens._t[0] = a[n]  # the bonus = the next input (device token row 0)
        io.cur_pos._t[:] = -1
        io.cur_pos._t[0] = base + n + 1
        d = self._drafts(base + n + 1)
        self.rec = dict(n=n, c=base, a=a + [a[-1]] * (K + 1 - len(a)), committed=a[: n + 1], drafts=d,
                        steps=rec["steps"] + 1)  # fmt: skip
        self.calls["step"] += 1

    def read(self):
        return dict(self.rec)

    def leave(self):
        self.calls["leave"] += 1
        return dict(self.rec)


def _serve_spec(model, pages, req, prompt, gen, K, *, other=None, other_at=None, trim=None, ordinary_at=()):
    """the runner's loop for one greedy request (+ an optional second request admitted at step `other_at`, which turns
    drafting off): prefill, then propose -> verify (or an ordinary step) -> accept_greedy_drafts -> commit. trim
    {loop step: m}: the scheduler's budget leaves m of the offered drafts (the rest placeholders); ordinary_at: loop
    steps the runner demotes to an ordinary decode although drafts were offered (a logprobs row in the batch)"""
    S = model.max_batch_size
    sampling = _sampling(S)  # greedy rows
    out, pos, steps = [], len(prompt), dict(verify=0, ordinary=0, drafted=0)
    pages.ensure(req, len(prompt))
    first = model.prefill_forward(tokens=torch.tensor([prompt], dtype=torch.int32), page_table=pages.table([req]),
                                  kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([len(prompt)]),
                                  start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_sampling(1))  # fmt: skip
    last = int(torch.as_tensor(first).reshape(-1)[0])
    out.append(last)
    counts = torch.ones(S, dtype=torch.int32)
    committed = torch.full((S, K + 1), -1, dtype=torch.int32)
    committed[0, 0] = last
    t = -1
    while len(out) < gen:
        t += 1
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        positions[0] = pos + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        assert isinstance(dr, spec_decode.DraftOutput)
        nv = dr.num_valid.clone()
        if t in ordinary_at:
            nv[:] = 0
        elif trim and t in trim and int(nv[0]) > 0:
            nv[0] = min(int(nv[0]), int(trim[t]))
        pages.ensure(req, pos + K + 2)
        if int(nv.sum()) > 0:
            steps["drafted"] += 1
            tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
            start = torch.full((S, K + 1), -1, dtype=torch.int64)
            tokens[0] = torch.tensor([last] + dr.draft_token_ids[0].tolist())
            tokens[0, 1 + int(nv[0]):] = -1  # (the placeholders past the scheduled drafts)
            start[0] = pos + torch.arange(K + 1)
            vo = model.decode_forward(tokens, start, pages.table([req] + [None] * (S - 1)), model.kv_cache,
                                      sampling_params=sampling, num_valid_drafts=nv.to(torch.int32),
                                      accepted_counts=counts, spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
            assert isinstance(vo, spec_decode.VerifyOutput)
            com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, tokens[:, 1:].to(torch.int32), nv.to(torch.int32))
            n_ = int(cnt[0])
            got = [int(v) for v in com[0, :n_]]
            steps["verify"] += 1
        else:
            tokens = torch.full((S, 1), -1, dtype=torch.int64)
            tokens[0, 0] = last
            start = torch.full((S,), -1, dtype=torch.int64)
            start[0] = pos
            o = model.decode_forward(tokens, start, pages.table([req] + [None] * (S - 1)), model.kv_cache,
                                     sampling_params=sampling, read_from_device=True, reload_inputs=True)  # fmt: skip
            got = [int(torch.as_tensor(o).reshape(-1)[0])]
            n_ = 1
            steps["ordinary"] += 1
        out += got
        pos += n_
        last = got[-1]
        counts = torch.ones(S, dtype=torch.int32)
        counts[0] = n_
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        committed[0, :n_] = torch.tensor(got, dtype=torch.int32)
    return out[:gen], steps


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("K,corrupt", [(1, 0.3), (3, 0.25), (3, 0.0), (3, 1.0)])
def test_spec_greedy_identity_through_the_runner_contract(hf_config, monkeypatch, K, corrupt, async_):
    """greedy identity: the speculative path's served stream == the oracle's greedy continuation, for drafts that
    are always right, always wrong, or wrong at random depths (every rejection depth exercised); async_ =
    GLM53_SPEC_ASYNC (each step enqueued at the previous verify)"""
    monkeypatch.setattr(gv, "SPEC_K", K)
    monkeypatch.setattr(gv, "SPEC_ASYNC", async_)
    model = _adapter(hf_config, max_batch=4)
    gen = model.generator
    prompt = prompt_ids(0, 37)
    ref = gen.backend.reference_greedy(prompt, 40)
    truth = {len(prompt) + i: t for i, t in enumerate(ref)}
    sp = MockSpec(gen.backend, model.kv_cache, K, truth, corrupt)
    gen.backend.spec = sp
    pages = Pages(256, 64)
    got, steps = _serve_spec(model, pages, "r0", prompt, 40, K)
    assert got == ref[:40], (got, ref[:40], steps)
    # async: each verify enqueues the next step when slot 0's pages already hold its positions (else it is skipped and
    # the next verify submits its own); a step enqueued after the last verify runs unread when the request ends
    pre, skip = model.counters.get("spec_presubmits", 0), model.counters.get("spec_presubmit_skips", 0)
    assert steps["verify"] > 0 and sp.calls["step"] in (steps["verify"], steps["verify"] + 1), (sp.calls, steps)
    assert (pre + skip == steps["verify"] and pre > skip) if async_ else pre == skip == 0, (pre, skip, steps)
    if corrupt == 0.0:  # every draft accepted: K + 1 tokens per verify
        assert model.counters["spec_tokens"] == (K + 1) * model.counters["spec_steps"]
    # the per-verify host timeline: one pre / collect / post per lone step; a turn for each step that followed one
    c = model.counters
    assert c["spec_t_collect_n"] == c["spec_t_pre_n"] == c["spec_t_post_n"] == c["spec_steps"], c
    assert 0 < c.get("spec_t_turn_n", 0) < c["spec_steps"] and c["spec_t_collect_s_total"] >= 0.0, c


@pytest.mark.parametrize("K,trim,ordinary",[(3, {2: 1, 6: 0, 9: 2}, ()), (3, {}, (3, 7)), (7, {1: 2, 4: 5}, (8,)),
                                              (1, {3: 0}, (5, 6))])
def test_async_runner_deviations_are_answered_from_the_held_tokens(hf_config, monkeypatch, K, trim, ordinary):
    """GLM53_SPEC_ASYNC: the step after each verify is enqueued before the runner has seen the next block. When the
    runner then commits fewer tokens than that step (the scheduler's budget trims the drafts to m, or a logprobs row
    demotes the step to an ordinary decode), the adapter holds the tokens the device committed and answers slot 0's
    next inputs from them (the row sits out those replays) until the runner reaches the device's position; the stream
    == the oracle's greedy continuation, and the held path ran"""
    monkeypatch.setattr(gv, "SPEC_K", K)
    monkeypatch.setattr(gv, "SPEC_ASYNC", True)
    model = _adapter(hf_config, max_batch=8)
    gen = model.generator
    prompt = prompt_ids(0, 37)
    ref = gen.backend.reference_greedy(prompt, 60)
    truth = {len(prompt) + i: t for i, t in enumerate(ref)}
    sp = MockSpec(gen.backend, model.kv_cache, K, truth, 0.0)  # every draft right: the device accepts K each step
    gen.backend.spec = sp
    pages = Pages(256, 64)
    got, steps = _serve_spec(model, pages, "r0", prompt, 60, K, trim=trim, ordinary_at=ordinary)
    assert got == ref[:60], (got, ref[:60], steps)
    assert model.counters.get("spec_held", 0) >= 1 and model.counters.get("spec_held_answers", 0) >= 1, model.counters
    assert steps["verify"] > 0 and steps["ordinary"] > 0, steps


@pytest.mark.parametrize("K", [3, 7])
def test_drafter_cache_cap_turns_drafting_off_past_it(hf_config, monkeypatch, K):
    """GLM53_DF2_CACHE_CAP (SpecMTP.pos_cap, the 1M line's 131,072-token DFlash2 drafter cache): a request whose
    speculative steps would pass the cap leaves speculation before that step and continues with plain steps; the
    stream == the oracle's greedy continuation, and no spec step ever touches a position >= the cap"""
    monkeypatch.setattr(gv, "SPEC_K", K)
    model = _adapter(hf_config, max_batch=8)  # (K + 1 token rows)
    gen = model.generator
    prompt = prompt_ids(0, 37)
    ref = gen.backend.reference_greedy(prompt, 48)
    truth = {len(prompt) + i: t for i, t in enumerate(ref)}
    sp = MockSpec(gen.backend, model.kv_cache, K, truth, 0.25)
    sp.pos_cap = cap = len(prompt) + 24
    gen.backend.spec = sp
    top = []
    step0 = sp.step

    def step():
        base = sp.rec["c"] + sp.rec["n"] + 1
        top.append(base + 2 * K)  # the verify's last position (base + K) + the next drafts' block (up to base + 2K)
        step0()

    sp.step = step
    pages = Pages(256, 64)
    got, steps = _serve_spec(model, pages, "r0", prompt, 48, K)
    assert got == ref[:48], (got, ref[:48], steps)
    assert steps["verify"] > 0 and steps["ordinary"] > 0, steps  # it drafted, then decoded plainly past the cap
    assert max(top) < cap, (max(top), cap)
    assert sp.calls["leave"] >= 1


def test_spec_plan_declares_one_k_and_rejects_smaller(monkeypatch):
    monkeypatch.setattr(gv, "SPEC_K", 3)
    plan = gv.Glm5NextForCausalLM.spec_plan(None, 32, 3)
    assert isinstance(plan, spec_decode.SpecPlan) and plan.effective_k == 3 and "argmax_ids" in plan.accept_modes
    assert plan.drafter_state == "internal" and plan.supports_narrow_decode
    assert isinstance(gv.Glm5NextForCausalLM.spec_plan(None, 32, 7), spec_decode.SpecPlan)
    rej = gv.Glm5NextForCausalLM.spec_plan(None, 32, 1)
    assert isinstance(rej, spec_decode.SpecReject) and rej.supported_k == (3,)
    monkeypatch.setattr(gv, "SPEC_K", 0)
    assert isinstance(gv.Glm5NextForCausalLM.spec_plan(None, 32, 3), spec_decode.SpecReject)


def test_no_drafts_without_the_device_step(hf_config, monkeypatch):
    """a backend without the speculative step drafts nothing (num_valid 0): the plugin then sends ordinary steps"""
    monkeypatch.setattr(gv, "SPEC_K", 3)
    model = _adapter(hf_config, max_batch=4)
    model.generator.backend.spec = None
    S = model.max_batch_size
    dr = model.propose_draft_tokens(3, torch.full((S, 4), -1, dtype=torch.int32), torch.zeros((S, 4), dtype=torch.int64),
                                    torch.ones(S, dtype=torch.int32))  # fmt: skip
    assert int(dr.num_valid.sum()) == 0


@pytest.mark.parametrize("async_", [False, True])
def test_second_request_turns_drafting_off_and_back_on(hf_config, monkeypatch, async_):
    """request A speculates alone; B is admitted into slot 1 (a candidate row of A's step: the adapter leaves
    speculation before the prefill), both decode with ordinary steps (num_valid 0 for every row: the plain path is
    unchanged), B finishes, A speculates again. Both streams == the oracle. async_ (GLM53_SPEC_ASYNC): B's prefill
    meets A's pre-enqueued step -> its tokens are held and answer A's next ordinary steps while B decodes on device."""
    K = 3
    monkeypatch.setattr(gv, "SPEC_K", K)
    monkeypatch.setattr(gv, "SPEC_ASYNC", async_)
    model = _adapter(hf_config, max_batch=4)
    gen = model.generator
    S = model.max_batch_size
    pa, pb = prompt_ids(0, 29), prompt_ids(1, 23)
    ra, rb = gen.backend.reference_greedy(pa, 48), gen.backend.reference_greedy(pb, 12)
    truth = {len(pa) + i: t for i, t in enumerate(ra)}
    sp = MockSpec(gen.backend, model.kv_cache, K, truth, 0.3)
    gen.backend.spec = sp
    pages = Pages(256, 64)
    seq = {"a": list(pa), "b": list(pb)}
    out = {"a": [], "b": []}
    slot = {"a": 0}

    def prefill(req, s):
        pages.ensure(req, len(seq[req]))
        o = model.prefill_forward(tokens=torch.tensor([seq[req]], dtype=torch.int32), page_table=pages.table([req]),
                                  kv_cache=model.kv_cache, enable_trace=False,
                                  prompt_lens=torch.tensor([len(seq[req])]), start_pos=torch.tensor([0]),
                                  empty_slots=[s], sampling_params=_sampling(1))  # fmt: skip
        t = int(torch.as_tensor(o).reshape(-1)[0])
        out[req].append(t)
        seq[req].append(t)

    prefill("a", 0)
    counts = torch.ones(S, dtype=torch.int32)
    last_n = {"a": 1}
    step = 0
    while len(out["a"]) < 48:
        live = [r for r in ("a", "b") if r in slot]
        rows = [slot.get("a"), slot.get("b")]
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        for r in live:
            n_ = last_n.get(r, 1)
            committed[slot[r], :n_] = torch.tensor(seq[r][-n_:], dtype=torch.int32)
            positions[slot[r]] = len(seq[r]) - 1 + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        if live == ["a"] or (len(live) == 1):
            pass
        else:
            assert int(dr.num_valid.sum()) == 0, "drafting must be off with two live requests"
        tbl = pages.table([("a" if slot.get("a") == i else "b" if slot.get("b") == i else None) for i in range(S)])
        for r in live:
            pages.ensure(r, len(seq[r]) + K + 2)
        tbl = pages.table([("a" if slot.get("a") == i else "b" if slot.get("b") == i else None) for i in range(S)])
        if int(dr.num_valid.sum()) > 0:
            tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
            start = torch.full((S, K + 1), -1, dtype=torch.int64)
            tokens[0] = torch.tensor([seq["a"][-1]] + dr.draft_token_ids[0].tolist())
            start[0] = len(seq["a"]) - 1 + torch.arange(K + 1)
            vo = model.decode_forward(tokens, start, tbl, model.kv_cache, sampling_params=_sampling(S),
                                      num_valid_drafts=dr.num_valid.to(torch.int32), accepted_counts=counts,
                                      spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
            com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, dr.draft_token_ids, dr.num_valid.to(torch.int32))
            got = {"a": [int(v) for v in com[0, : int(cnt[0])]]}
        else:
            tokens = torch.full((S, 1), -1, dtype=torch.int64)
            start = torch.full((S,), -1, dtype=torch.int64)
            for r in live:
                tokens[slot[r], 0] = seq[r][-1]
                start[slot[r]] = len(seq[r]) - 1
            o = torch.as_tensor(model.decode_forward(tokens, start, tbl, model.kv_cache, sampling_params=_sampling(S),
                                                     read_from_device=True, reload_inputs=True)).reshape(-1)  # fmt: skip
            got = {r: [int(o[slot[r]])] for r in live}
        counts = torch.ones(S, dtype=torch.int32)
        for r, g in got.items():
            out[r] += g
            seq[r] += g
            last_n[r] = len(g)
            counts[slot[r]] = len(g)
        step += 1
        if step == 4:  # B arrives: prefilled into slot 1 (one of A's candidate rows)
            prefill("b", 1)
            slot["b"] = 1
            last_n["b"] = 1
        if "b" in slot and len(out["b"]) >= 12:  # B finishes
            model.release_request(slot.pop("b"))
            pages.release("b")
    assert out["a"][:48] == ra[:48]
    assert out["b"][:12] == rb[:12]
    assert model.counters.get("spec_leaves", 0) >= 1 and model.counters.get("spec_enters", 0) >= 2
    if async_:
        assert model.counters.get("spec_held", 0) >= 1 and model.counters.get("spec_held_answers", 0) >= 1, model.counters


@pytest.mark.parametrize("fixed", [True, False])
def test_no_spec_enter_while_a_plain_step_is_in_flight(hf_config, monkeypatch, fixed):
    """(async scheduling; the twinmtp row of 2026-10-27 09:29Z) the plugin submits slot 0's next plain step before it
    asks for drafts after the previous one (an overlapped step). Entering speculation then starts the state one step
    stale: the runner's next commit is not the device's bonus and propose's guard raises (the served engine died).
    The adapter drafts nothing while a plain step is in flight and enters once none is; the verify after that commits
    the oracle's tokens. fixed=False: the control -- the stale enter the check prevents, then the guard fires"""
    K = 3
    monkeypatch.setattr(gv, "SPEC_K", K)
    monkeypatch.setattr(gv, "SPEC_ASYNC", True)
    model = _adapter(hf_config, max_batch=4)
    gen = model.generator
    S = model.max_batch_size
    prompt = prompt_ids(0, 37)
    ref = gen.backend.reference_greedy(prompt, 24)
    sp = MockSpec(gen.backend, model.kv_cache, K, {len(prompt) + i: t for i, t in enumerate(ref)}, 0.0)
    gen.backend.spec = sp
    pages = Pages(256, 64)
    pages.ensure("r0", len(prompt) + 2 * K + 8)
    tbl = pages.table(["r0"] + [None] * (S - 1))
    P = len(prompt)
    first = model.prefill_forward(tokens=torch.tensor([prompt], dtype=torch.int32), page_table=pages.table(["r0"]),
                                  kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([P]),
                                  start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_sampling(1))  # fmt: skip
    got = [int(torch.as_tensor(first).reshape(-1)[0])]

    def propose(pos):
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        committed[0, 0] = got[-1]
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        positions[0] = pos + torch.arange(K + 1)
        return model.propose_draft_tokens(K, committed, positions, torch.ones(S, dtype=torch.int32), hidden=None)

    def plain(pos, read, reload=True):
        tokens = torch.full((S, 1), -1, dtype=torch.int64)
        tokens[0, 0] = got[-1]
        start = torch.full((S,), -1, dtype=torch.int64)
        start[0] = pos
        return model.decode_forward(tokens, start, tbl, model.kv_cache, sampling_params=_sampling(S),
                                    read_from_device=read, reload_inputs=reload)  # fmt: skip

    got.append(int(torch.as_tensor(plain(P, True)).reshape(-1)[0]))  # plain step A at P, read: its token is committed
    pend = plain(P + 1, False, reload=False)  # plain step B at P + 1 submitted before the runner asks for drafts
    if fixed:
        dr = propose(P + 1)
        assert int(dr.num_valid.sum()) == 0 and sp.calls["enter"] == 0, (dr.num_valid, sp.calls)
        assert model.counters.get("spec_declines_inflight") == 1, model.counters
    else:  # the control: A's token offered at B's position (what entering without the check did) -> a stale enter
        dr = propose(P + 2)
        assert int(dr.num_valid[0]) == K and sp.calls["enter"] == 1, (dr.num_valid, sp.calls)
    tok = model.process_decode_output_host(model.read_decode_output(pend), is_tokens=True)
    got.append(int(tok.reshape(-1)[0]))  # B's token, committed by the runner
    if not fixed:  # the runner's next commit is B's token, not the device's bonus: the guard fires (the served death)
        with pytest.raises(RuntimeError, match="propose: the runner committed"):
            propose(P + 2)
        return
    dr = propose(P + 2)  # nothing in flight: enter
    assert int(dr.num_valid[0]) == K and sp.calls["enter"] == 1, (dr.num_valid, sp.calls)
    tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
    start = torch.full((S, K + 1), -1, dtype=torch.int64)
    tokens[0] = torch.tensor([got[-1]] + dr.draft_token_ids[0].tolist())
    start[0] = P + 2 + torch.arange(K + 1)
    vo = model.decode_forward(tokens, start, tbl, model.kv_cache, sampling_params=_sampling(S),
                              num_valid_drafts=dr.num_valid.to(torch.int32), accepted_counts=torch.ones(S, dtype=torch.int32),
                              spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
    com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, dr.draft_token_ids, dr.num_valid.to(torch.int32))
    got += [int(v) for v in com[0, : int(cnt[0])]]
    assert got == ref[: len(got)] and len(got) == 3 + K + 1, (got, ref[: len(got)])


@pytest.mark.parametrize("swap", [False, True])
def test_stale_drafts_after_an_arrival_bind_per_row(hf_config, monkeypatch, swap):
    """the served runner (n29df2e, 2026-10-23 07:11Z) scheduled a verify block from drafts proposed BEFORE a second
    request arrived (the adapter had left speculation for B's prefill): the block must not fail -- each live row binds
    its own drafts (A the stale ones, B none), verified by ordinary decode sub-steps, and both streams stay == the
    oracle; A speculates again once alone. swap: the same block arrives with the rows swapped by a slot remap (B in
    row 0, A and its stale drafts in row 1, slot_remap [1, 0, 2, 3]) -- the served message exactly"""
    K = 3
    monkeypatch.setattr(gv, "SPEC_K", K)
    model = _adapter(hf_config, max_batch=4)
    gen = model.generator
    S = model.max_batch_size
    pa, pb = prompt_ids(0, 29), prompt_ids(1, 23)
    ra, rb = gen.backend.reference_greedy(pa, 40), gen.backend.reference_greedy(pb, 8)
    truth = {len(pa) + i: t for i, t in enumerate(ra)}
    gen.backend.spec = MockSpec(gen.backend, model.kv_cache, K, truth, 0.3)
    pages = Pages(256, 64)
    seq = {"a": list(pa), "b": list(pb)}
    out = {"a": [], "b": []}
    slot = {"a": 0}

    def prefill(req, s_):
        pages.ensure(req, len(seq[req]))
        o = model.prefill_forward(tokens=torch.tensor([seq[req]], dtype=torch.int32), page_table=pages.table([req]),
                                  kv_cache=model.kv_cache, enable_trace=False,
                                  prompt_lens=torch.tensor([len(seq[req])]), start_pos=torch.tensor([0]),
                                  empty_slots=[s_], sampling_params=_sampling(1))  # fmt: skip
        t = int(torch.as_tensor(o).reshape(-1)[0])
        out[req].append(t)
        seq[req].append(t)

    def table():
        return pages.table([("a" if slot.get("a") == i else "b" if slot.get("b") == i else None) for i in range(S)])

    def verify(dr, rows_live, remap=None, src=None):
        """A's row (slot["a"]) carries its drafts (row `src` of the proposal dr: default A's row), B's row column 0"""
        ra = slot["a"]
        src = ra if src is None else src
        tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
        start = torch.full((S, K + 1), -1, dtype=torch.int64)
        drafts = torch.full((S, K), -1, dtype=torch.int32)
        nv = torch.zeros(S, dtype=torch.int32)
        tokens[ra] = torch.tensor([seq["a"][-1]] + dr.draft_token_ids[src].tolist())
        start[ra] = len(seq["a"]) - 1 + torch.arange(K + 1)
        drafts[ra], nv[ra] = dr.draft_token_ids[src], dr.num_valid[src]
        if "b" in rows_live:  # B's row: no drafts, its last token / position in column 0
            tokens[slot["b"], 0] = seq["b"][-1]
            start[slot["b"], 0] = len(seq["b"]) - 1
        cn = torch.ones(S, dtype=torch.int32)
        cn[ra] = counts[0] if remap is not None else counts[ra]
        vo = model.decode_forward(tokens, start, table(), model.kv_cache, sampling_params=_sampling(S),
                                  num_valid_drafts=nv, accepted_counts=cn, spec_mode="argmax_ids",
                                  reload_inputs=True, slot_remap=remap)  # fmt: skip
        if remap is not None:  # the runner confirms the gather after the step
            model.note_state_slots_moved({j: i for i, j in enumerate(remap.tolist()) if i != j})
        vd = int(vo.argmax_ids.shape[1]) - 1  # (the runner trims the drafts to the width the verify answered at)
        com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, drafts[:, :vd], nv)
        return vo, com, cnt

    prefill("a", 0)
    counts = torch.ones(S, dtype=torch.int32)
    last_n = 1
    stale = None
    for step in range(3):  # A alone: speculative steps
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        committed[0, :last_n] = torch.tensor(seq["a"][-last_n:], dtype=torch.int32)
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        positions[0] = len(seq["a"]) - 1 + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        assert int(dr.num_valid[0]) == K
        for r in ("a",):
            pages.ensure(r, len(seq[r]) + K + 2)
        vo, com, cnt = verify(dr, ["a"])
        got = [int(v) for v in com[0, : int(cnt[0])]]
        out["a"] += got
        seq["a"] += got
        last_n = len(got)
        counts = torch.ones(S, dtype=torch.int32)
        counts[0] = last_n
    # the runner proposes once more for A (drafts for the next block) ...
    committed = torch.full((S, K + 1), -1, dtype=torch.int32)
    committed[0, :last_n] = torch.tensor(seq["a"][-last_n:], dtype=torch.int32)
    positions = torch.full((S, K + 1), -1, dtype=torch.int64)
    positions[0] = len(seq["a"]) - 1 + torch.arange(K + 1)
    stale = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
    assert int(stale.num_valid[0]) == K
    # ... then B arrives (prefilled into slot 1: the adapter leaves A's speculation first) and the block the runner
    # had scheduled from the earlier proposal still carries A's drafts, with B live in row 1
    prefill("b", 1)
    slot["b"] = 1
    pages.ensure("a", len(seq["a"]) + K + 2)
    pages.ensure("b", len(seq["b"]) + K + 2)
    plain0 = model.counters.get("spec_plain_verifies", 0)
    remap = None
    if swap:  # the served shape: a remap puts B in row 0 and A (with its stale drafts) in row 1
        slot = {"a": 1, "b": 0}
        remap = torch.tensor([1, 0, 2, 3], dtype=torch.int32)
    vo, com, cnt = verify(stale, ["a", "b"], remap, src=0)  # (raised 'verify block: rows [..] drafted' before the fix)
    assert model.counters.get("spec_plain_verifies", 0) == plain0 + 1, "the block ran as ordinary sub-steps"
    row_a, row_b = slot["a"], slot["b"]
    assert tuple(vo.argmax_ids.shape) == (S, K + 1) and int(cnt[row_b]) == 1, "B commits one token, A its drafts' prefix"
    got = {"a": [int(v) for v in com[row_a, : int(cnt[row_a])]], "b": [int(com[row_b, 0])]}
    for r_, g_ in got.items():
        out[r_] += g_
        seq[r_] += g_
    a_n = int(cnt[row_a])
    # both decode with ordinary steps until B finishes, then A speculates again
    counts = torch.ones(S, dtype=torch.int32)
    counts[slot["a"]] = a_n
    last = {"a": a_n, "b": 1}
    while len(out["a"]) < 40:
        live = [r for r in ("a", "b") if r in slot]
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        for r in live:
            committed[slot[r], : last[r]] = torch.tensor(seq[r][-last[r]:], dtype=torch.int32)
            positions[slot[r]] = len(seq[r]) - 1 + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        for r in live:
            pages.ensure(r, len(seq[r]) + K + 2)
        if int(dr.num_valid.sum()) > 0:
            assert live == ["a"]
            vo, com, cnt = verify(dr, ["a"])
            got = {"a": [int(v) for v in com[0, : int(cnt[0])]]}
        else:
            tokens = torch.full((S, 1), -1, dtype=torch.int64)
            start = torch.full((S,), -1, dtype=torch.int64)
            for r in live:
                tokens[slot[r], 0] = seq[r][-1]
                start[slot[r]] = len(seq[r]) - 1
            o = torch.as_tensor(model.decode_forward(tokens, start, table(), model.kv_cache, sampling_params=_sampling(S),
                                                     read_from_device=True, reload_inputs=True)).reshape(-1)  # fmt: skip
            got = {r: [int(o[slot[r]])] for r in live}
        counts = torch.ones(S, dtype=torch.int32)
        for r, g in got.items():
            out[r] += g
            seq[r] += g
            counts[slot[r]] = last[r] = len(g)
        if "b" in slot and len(out["b"]) >= 8:
            model.release_request(slot.pop("b"))
            pages.release("b")
            if slot["a"] != 0:  # the runner condenses: A moves back to row 0 by a remap on its next (plain) step
                remap = torch.tensor([slot["a"]] + [j for j in range(S) if j != slot["a"]], dtype=torch.int32)
                slot = {"a": 0}
                tokens = torch.full((S, 1), -1, dtype=torch.int64)
                start = torch.full((S,), -1, dtype=torch.int64)
                tokens[0, 0], start[0] = seq["a"][-1], len(seq["a"]) - 1
                o = torch.as_tensor(model.decode_forward(tokens, start, table(), model.kv_cache,
                                                         sampling_params=_sampling(S), read_from_device=True,
                                                         reload_inputs=True, slot_remap=remap)).reshape(-1)  # fmt: skip
                model.note_state_slots_moved({j: i for i, j in enumerate(remap.tolist()) if i != j})
                out["a"].append(int(o[0]))
                seq["a"].append(int(o[0]))
                counts = torch.ones(S, dtype=torch.int32)
                last["a"] = 1
    assert out["a"][:40] == ra[:40]
    assert out["b"][:8] == rb[:8]
    assert model.counters.get("spec_enters", 0) >= 2, "A speculates again once alone"


class ZeroSpec:
    """the image mocks' device step (GLM53_MOCK=1): no kernel runs, every record / token row reads back as zeros"""

    def __init__(self, K: int):
        self.K, self.sel, self.mesh = K, None, None
        self.calls = dict(enter=0, step=0, leave=0)

    def set_page_table(self, row):
        pass

    def _rec(self):
        return dict(n=0, c=-1, a=[0] * (self.K + 1), committed=[0], drafts=[0] * self.K, steps=0)

    def enter(self, page_row):
        self.calls["enter"] += 1

    def step(self):
        self.calls["step"] += 1

    def read(self):
        return self._rec()

    def leave(self):
        self.calls["leave"] += 1
        return self._rec()


def test_mock_device_step_rejects_its_zero_drafts(hf_config, monkeypatch):
    """GLM53_MOCK image mocks: the device step reads back zeros, so the generator's mock branch commits one token
    (n = 0). Its token must reject the block's first draft (also 0), else the runner's accept walk takes the K drafts
    the device never committed and the request runs K positions ahead of its slot (9's df2f image mock, 2026-10-23
    07:52Z: 'decode row 0 consumes position 17 ... slot 0 holds next_pos=10'). A speculates through a few steps, then a
    plain request B arrives: every decode's slot check passes and every walk commits one token per step."""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as G

    K = 3
    monkeypatch.setattr(gv, "SPEC_K", K)
    monkeypatch.setattr(G, "_MOCK_DEVICE", True)
    model = _adapter(hf_config, max_batch=4)
    gen = model.generator
    S = model.max_batch_size
    gen.backend.spec = ZeroSpec(K)
    pages = Pages(256, 64)
    pa, pb = prompt_ids(0, 29), prompt_ids(1, 23)
    seq = {"a": list(pa), "b": list(pb)}

    def prefill(req, s_):
        pages.ensure(req, len(seq[req]))
        o = model.prefill_forward(tokens=torch.tensor([seq[req]], dtype=torch.int32), page_table=pages.table([req]),
                                  kv_cache=model.kv_cache, enable_trace=False,
                                  prompt_lens=torch.tensor([len(seq[req])]), start_pos=torch.tensor([0]),
                                  empty_slots=[s_], sampling_params=_sampling(1))  # fmt: skip
        seq[req].append(int(torch.as_tensor(o).reshape(-1)[0]))

    prefill("a", 0)
    counts = torch.ones(S, dtype=torch.int32)
    for _ in range(4):  # A alone: mock speculative steps
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        committed[0, : int(counts[0])] = torch.tensor(seq["a"][-int(counts[0]):], dtype=torch.int32)
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        positions[0] = len(seq["a"]) - 1 + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        assert int(dr.num_valid[0]) == K
        pages.ensure("a", len(seq["a"]) + K + 2)
        tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
        start = torch.full((S, K + 1), -1, dtype=torch.int64)
        tokens[0] = torch.tensor([seq["a"][-1]] + dr.draft_token_ids[0].tolist())
        start[0] = len(seq["a"]) - 1 + torch.arange(K + 1)
        vo = model.decode_forward(tokens, start, pages.table(["a", None, None, None]), model.kv_cache,
                                  sampling_params=_sampling(S), num_valid_drafts=dr.num_valid.to(torch.int32),
                                  accepted_counts=counts, spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
        com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, dr.draft_token_ids, dr.num_valid.to(torch.int32))
        assert int(cnt[0]) == 1, f"the walk took {int(cnt[0]) - 1} drafts the mock device did not commit"
        seq["a"].append(int(com[0, 0]))
        counts = torch.ones(S, dtype=torch.int32)
    # a plain arrival: both rows decode; A's slot must sit at A's request position
    prefill("b", 1)
    for _ in range(2):
        tokens = torch.full((S, 1), -1, dtype=torch.int64)
        start = torch.full((S,), -1, dtype=torch.int64)
        for r_, s_ in (("a", 0), ("b", 1)):
            pages.ensure(r_, len(seq[r_]) + 1)
            tokens[s_, 0] = seq[r_][-1]
            start[s_] = len(seq[r_]) - 1
        o = torch.as_tensor(model.decode_forward(tokens, start, pages.table(["a", "b", None, None]), model.kv_cache,
                                                 sampling_params=_sampling(S), read_from_device=True,
                                                 reload_inputs=True)).reshape(-1)  # fmt: skip
        seq["a"].append(int(o[0]))
        seq["b"].append(int(o[1]))
    assert model.counters.get("spec_steps", 0) == 4


def test_spec_with_the_prefix_tier_snapshots_only_plain_decode_state(hf_config, monkeypatch):
    """Prefix tier x speculation (doc/ttnn/17-mtp-spec-decode/lmcache_spec_contract.md): the request decodes plainly
    across 128 (trailing snapshot), then speculates across 256 and finishes. While it speculates, a snapshot or export
    of slot 0 is refused (its committed state sits in candidate row n). Spec steps take no trailing snapshot, so the
    release commits the plain-decode snapshot as the G entry (128 tokens)."""
    from models.autoports.zai_org_glm_5_3_flash.tt import prefix_cache as PC

    K = 3
    monkeypatch.setattr(gv, "SPEC_K", K)
    model = _adapter(hf_config, max_batch=4)
    monkeypatch.setattr(type(model), "_vllm_prefix_caching", True, raising=False)
    model._prefix_init()
    assert model.prefix is not None and model.prefix.boundary == PC.BOUNDARY == 128
    gen = model.generator
    S = model.max_batch_size
    prompt = prompt_ids(0, 120)
    ref = gen.backend.reference_greedy(prompt, 160)
    truth = {len(prompt) + i: t for i, t in enumerate(ref)}
    sp = MockSpec(gen.backend, model.kv_cache, K, truth, 0.25)
    gen.backend.spec = sp
    snaps = []
    orig_snapshot = gen.snapshot_slot

    def snapshot_slot(kv, slot, entry, **kw):
        snaps.append((int(slot), model.slot_table.slots[int(slot)].next_pos))
        orig_snapshot(kv, slot, entry, **kw)

    monkeypatch.setattr(gen, "snapshot_slot", snapshot_slot)
    orig_tee = gen.set_decode_tee

    def set_decode_tee(kv, pairs):  # GLM53_DECODE_TEE=1: the G snapshot is the crossing step's in-trace tee
        snaps.extend((int(s_), int(t1)) for s_, _, _, t1 in pairs)
        orig_tee(kv, pairs)

    monkeypatch.setattr(gen, "set_decode_tee", set_decode_tee)
    pages = Pages(256, 64)
    pages.ensure("r0", len(prompt))
    first = model.prefill_forward(tokens=torch.tensor([prompt], dtype=torch.int32), page_table=pages.table(["r0"]),
                                  kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([len(prompt)]),
                                  start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_sampling(1))  # fmt: skip
    out = [int(torch.as_tensor(first).reshape(-1)[0])]
    pos = len(prompt)
    while pos < 130:  # plain decode across 128
        pages.ensure("r0", pos + 2)
        tokens = torch.full((S, 1), -1, dtype=torch.int64)
        tokens[0, 0] = out[-1]
        start = torch.full((S,), -1, dtype=torch.int64)
        start[0] = pos
        o = model.decode_forward(tokens, start, pages.table(["r0"] + [None] * (S - 1)), model.kv_cache,
                                 sampling_params=_sampling(S), read_from_device=True, reload_inputs=True)  # fmt: skip
        out.append(int(torch.as_tensor(o).reshape(-1)[0]))
        pos += 1
    assert snaps == [(0, 128)], snaps
    counts = torch.ones(S, dtype=torch.int32)
    committed = torch.full((S, K + 1), -1, dtype=torch.int32)
    committed[0, 0] = out[-1]
    while pos < 262:  # speculation across 256
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        positions[0] = pos + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        nv = dr.num_valid.to(torch.int32)
        assert int(nv[0]) == K
        pages.ensure("r0", pos + K + 2)
        tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
        start = torch.full((S, K + 1), -1, dtype=torch.int64)
        tokens[0] = torch.tensor([out[-1]] + dr.draft_token_ids[0].tolist())
        start[0] = pos + torch.arange(K + 1)
        vo = model.decode_forward(tokens, start, pages.table(["r0"] + [None] * (S - 1)), model.kv_cache,
                                  sampling_params=_sampling(S), num_valid_drafts=nv, accepted_counts=counts,
                                  spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
        com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, dr.draft_token_ids, nv)
        got = [int(v) for v in com[0, : int(cnt[0])]]
        out += got
        pos += len(got)
        counts = torch.ones(S, dtype=torch.int32)
        counts[0] = len(got)
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        committed[0, : len(got)] = torch.tensor(got, dtype=torch.int32)
    assert out == ref[: len(out)]
    assert model._spec is not None and model.counters["spec_steps"] > 0
    with pytest.raises(RuntimeError, match="speculates"):
        orig_snapshot(model.kv_cache, 0, 0)
    with pytest.raises(RuntimeError, match="speculates"):
        gen.export_slot_state(model.kv_cache, 0, n_tokens=256)
    if gen.decode_tee(model.kv_cache):  # nor may the decode tee copy slot 0 while it speculates
        with pytest.raises(RuntimeError, match="speculates"):
            orig_tee(model.kv_cache, [(0, 0, 0, 256)])
    assert snaps == [(0, 128)], snaps  # the spec steps crossed 256 without a snapshot
    assert model.prefix.trailing_position(0) == 128
    # spec_abort's selector clear is a device write (spec_mtp.write_rows): on the mock only the binding's flag drops
    monkeypatch.setattr(gen, "spec_abort", lambda kv: setattr(gen._binding(kv), "spec", None))
    model.release_request(0, token_ids=prompt + out)
    pages.release("r0")
    g = [e for e in model.prefix.entries.values() if e.kind == "G"]
    assert len(g) == 1 and g[0].n_tokens == 128, g
    assert model.counters.get("prefix_generation_entries", 0) == 1


class PoisonSpec(MockSpec):
    """MockSpec + the device step's footprint on the OTHER rows: the real step writes its candidate KDA rows / conv
    histories into rows 1..K of the shared 32-row state, so every enter / step here overwrites rows 1..K of every KDA
    layer with junk (a request whose live state sits in those rows is then corrupted, as on the device)"""

    def __init__(self, backend, kv, bkv, K, truth, corrupt):
        super().__init__(backend, kv, K, truth, corrupt)
        self.bkv, self.poisoned = bkv, 0

    def _poison(self):
        for layer, c in zip(self.be.layers, self.bkv.caches):
            if layer.is_kda:
                c.state[1 : 1 + self.K].fill_(7.0)
                c.conv[1 : 1 + self.K].fill_(-3.0)
        self.poisoned += 1

    def enter(self, page_row):
        super().enter(page_row)
        self._poison()

    def step(self):
        super().step()
        self._poison()


def _mid_prefill_run(hf_config, monkeypatch, *, spec: bool, sole: bool, blind: bool, asyn: bool = False):
    """Program 2026-10-26T23:57:28Z (2): a lone greedy decoder A (generator slot 0) and a 4-chunk prefill of B into
    vLLM slot 1, interleaved P1 / D3 (one chunk of B, then three steps of A); after B's last chunk both decode plainly,
    B is released and A runs alone again. spec: A's drafter is a PoisonSpec (else no speculative step at all, the
    reference). sole: GLM53_SPEC_SOLE_STATE. blind: the generator's spec_enter does not see the other binding slots
    (its own live-slot check bypassed), so only the adapter's guard stands. -> (A's tokens, B's tokens, B's KDA state
    after its last chunk, B's generator slot, the speculative steps taken while B was mid-prefill, the poison count)"""
    K = 3
    monkeypatch.setattr(gv, "SPEC_K", K)
    monkeypatch.setattr(gv, "SPEC_SOLE_STATE", sole)
    monkeypatch.setattr(gv, "SPEC_ASYNC", asyn)  # (GLM53_SPEC_ASYNC: a pre-enqueued step in flight)
    model = _adapter(hf_config, max_batch=8)
    gen = model.generator
    S = model.max_batch_size
    bkv = gen._binding(model.kv_cache).kv_cache
    pa, pb = prompt_ids(0, 29), prompt_ids(1, 250)  # B: chunks [0, 64) [64, 128) [128, 192) [192, 250)
    ra = gen.backend.reference_greedy(pa, 64)
    sp = None
    if spec:
        sp = PoisonSpec(gen.backend, model.kv_cache, bkv, K, {len(pa) + i: t for i, t in enumerate(ra)}, 0.25)
    gen.backend.spec = sp
    if blind:
        orig_enter = gen.spec_enter

        def spec_enter_blind(kv, slot=0):
            b = gen._binding(kv)
            saved = [(s, b.slots[s].active) for s in range(1, b.max_batch)]
            for s, _ in saved:
                b.slots[s].active = False
            try:
                return orig_enter(kv, slot)
            finally:
                for s, a in saved:
                    b.slots[s].active = a

        monkeypatch.setattr(gen, "spec_enter", spec_enter_blind)
    pages = Pages(256, 64)
    seq, out, slot, last_n = {"a": list(pa)}, {"a": [], "b": []}, {"a": 0}, {"a": 1}
    pages.ensure("a", len(pa))
    o = model.prefill_forward(tokens=torch.tensor([pa], dtype=torch.int32), page_table=pages.table(["a"]),
                              kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([len(pa)]),
                              start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_sampling(1))  # fmt: skip
    out["a"].append(int(torch.as_tensor(o).reshape(-1)[0]))
    seq["a"].append(out["a"][-1])
    counts = torch.ones(S, dtype=torch.int32)

    def table():
        return pages.table([next((r for r, s_ in slot.items() if s_ == i), None) for i in range(S)])

    def step():
        nonlocal counts
        live = sorted(slot, key=slot.get)
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        for r in live:
            committed[slot[r], : last_n[r]] = torch.tensor(seq[r][-last_n[r] :], dtype=torch.int32)
            positions[slot[r]] = len(seq[r]) - 1 + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        for r in live:
            pages.ensure(r, len(seq[r]) + K + 2)
        if int(dr.num_valid.sum()) > 0:
            tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
            start = torch.full((S, K + 1), -1, dtype=torch.int64)
            tokens[0] = torch.tensor([seq["a"][-1]] + dr.draft_token_ids[0].tolist())
            start[0] = len(seq["a"]) - 1 + torch.arange(K + 1)
            vo = model.decode_forward(tokens, start, table(), model.kv_cache, sampling_params=_sampling(S),
                                      num_valid_drafts=dr.num_valid.to(torch.int32), accepted_counts=counts,
                                      spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
            com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, dr.draft_token_ids, dr.num_valid.to(torch.int32))
            got = {"a": [int(v) for v in com[0, : int(cnt[0])]]}
        else:
            tokens = torch.full((S, 1), -1, dtype=torch.int64)
            start = torch.full((S,), -1, dtype=torch.int64)
            for r in live:
                tokens[slot[r], 0] = seq[r][-1]
                start[slot[r]] = len(seq[r]) - 1
            o_ = torch.as_tensor(model.decode_forward(tokens, start, table(), model.kv_cache,
                                                      sampling_params=_sampling(S), read_from_device=True,
                                                      reload_inputs=True)).reshape(-1)  # fmt: skip
            got = {r: [int(o_[slot[r]])] for r in live}
        counts = torch.ones(S, dtype=torch.int32)
        for r, g in got.items():
            out[r] += g
            seq[r] += g
            last_n[r] = len(g)
            counts[slot[r]] = len(g)

    for _ in range(3):  # A alone: it speculates
        step()
    spec_before = model.counters.get("spec_steps", 0)
    ends = [64, 128, 192, 250]
    b_state = None
    for c, end in enumerate(ends):  # P1 / D3: one chunk of B, then three steps of A
        start_c = 0 if c == 0 else ends[c - 1]
        pages.ensure("b", end)
        o = model.prefill_forward(tokens=torch.tensor([pb[:end]], dtype=torch.int32), page_table=pages.table(["b"]),
                                  kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([end]),
                                  start_pos=torch.tensor([start_c]), empty_slots=[1], sampling_params=_sampling(1),
                                  sample_rows=[c == len(ends) - 1])  # fmt: skip
        if c < len(ends) - 1:
            assert int(torch.as_tensor(o).reshape(-1)[0]) == -1  # an intermediate chunk: head-less
            for _ in range(3):
                step()
        else:
            gb = model.gen_slot[1]
            b_state = [(d["S"].clone(), d["C"].clone()) for d in gen.backend.export_kda_state(bkv, gb)]
            out["b"].append(int(torch.as_tensor(o).reshape(-1)[0]))
            seq["b"] = list(pb) + out["b"]
            slot["b"], last_n["b"] = 1, 1
    spec_mid = model.counters.get("spec_steps", 0) - spec_before
    while len(out["b"]) < 8:  # both decode (two live rows: no drafts)
        step()
    model.release_request(slot.pop("b"))
    pages.release("b")
    while len(out["a"]) < 48:  # A alone again
        step()
    return out["a"][:48], out["b"][:8], b_state, gb, spec_mid, (sp.poisoned if sp is not None else 0), model


@pytest.mark.parametrize("sole,blind,asyn", [(True, False, False), (True, True, False), (False, False, False),
                                             (True, False, True), (True, True, True)])
def test_spec_guard_mid_prefill_state_bitwise(hf_config, monkeypatch, sole, blind, asyn):
    """Program 2026-10-26T23:57:28Z (14's mechanism for the MTP twin's lossless FALSE at 8 / 16 users x 32K):
    the speculative step writes candidate KDA rows / conv histories into rows 1..K, which would clobber a request
    gathered into those rows between the chunks of its multi-chunk prefill. With the adapter's guard
    (GLM53_SPEC_SOLE_STATE=1, also with the generator's own live-slot check blinded) and with the generator's check
    alone, no speculative step runs while B is mid-prefill: B's KDA state after its last chunk and B's tokens are
    BITWISE the no-spec run's, A's and B's streams equal the oracle, and A speculates before and after B."""
    a_ref, b_ref, st_ref, gb_ref, _, _, _ = _mid_prefill_run(hf_config, monkeypatch, spec=False, sole=sole, blind=False)
    a, b, st, gb, spec_mid, poisoned, model = _mid_prefill_run(hf_config, monkeypatch, spec=True, sole=sole,
                                                               blind=blind, asyn=asyn)  # fmt: skip
    gen = model.generator
    assert 1 <= gb <= 3 and gb == gb_ref  # B's state sits in one of the K candidate rows
    assert spec_mid == 0, spec_mid
    assert len(st) == len(st_ref)
    assert all(torch.equal(s, s0) and torch.equal(c, c0) for (s, c), (s0, c0) in zip(st, st_ref))
    assert b == b_ref and a == a_ref
    assert a == gen.backend.reference_greedy(prompt_ids(0, 29), 48)
    assert b == gen.backend.reference_greedy(prompt_ids(1, 250), 8)
    assert poisoned > 0 and model.counters.get("spec_steps", 0) > 0 and model.counters.get("spec_enters", 0) >= 2
    if sole:
        assert model.counters.get("spec_declines_state", 0) > 0
    if asyn:  # the async arm ran with pre-enqueued steps (the next step submitted at each lone verify)
        assert model.counters.get("spec_presubmits", 0) > 0, model.counters


def test_spec_guard_negative_control_detects_the_clobber(hf_config, monkeypatch):
    """the unit's sensitivity: with the adapter's guard OFF (GLM53_SPEC_SOLE_STATE=0) and the generator's live-slot
    check blinded, A speculates while B is mid-prefill, the poisoned rows 1..K reach B's continuation chunks and B's
    state / tokens move off the no-spec run (the collision the guard closes)"""
    _, b_ref, st_ref, _, _, _, _ = _mid_prefill_run(hf_config, monkeypatch, spec=False, sole=False, blind=False)
    _, b, st, gb, spec_mid, poisoned, _ = _mid_prefill_run(hf_config, monkeypatch, spec=True, sole=False, blind=True)
    assert spec_mid > 0 and poisoned > 0 and 1 <= gb <= 3
    assert not all(torch.equal(s, s0) and torch.equal(c, c0) for (s, c), (s0, c0) in zip(st, st_ref))


@pytest.mark.parametrize("restore", [True, False])
def test_seeded_row_beside_a_sub_step_block_draws_like_a_plain_step(hf_config, monkeypatch, restore):
    """review round 5 P2-4: a verify block computed by ordinary sub-steps (_verify_plain) replays the decode trace 1 + n
    times, and every replay draws from every row's device RNG (samples_in_decode_trace). A seeded sampled request B
    beside A's accepted stale drafts sits sub-steps 1..K out: its stream must still advance ONE draw for the block,
    so its tokens == B served alone by a plain twin under the same seed. restore False (negative control): without
    the RNG restore B takes K extra draws and its stream differs."""
    K, SEED, N = 3, 4242, 12
    monkeypatch.setattr(gv, "SPEC_K", K)
    pa, pb = prompt_ids(0, 29), prompt_ids(1, 23)

    def b_row(S, row):
        temp = torch.zeros(S)
        seed = torch.full((S,), -1, dtype=torch.int64)
        temp[row], seed[row] = 1.0, SEED
        return _sampling(S, temperature=temp, seed=seed)

    def prefill(model, pages, req, prompt, slot, sp):
        pages.ensure(req, len(prompt))
        o = model.prefill_forward(tokens=torch.tensor([prompt], dtype=torch.int32), page_table=pages.table([req]),
                                  kv_cache=model.kv_cache, enable_trace=False, prompt_lens=torch.tensor([len(prompt)]),
                                  start_pos=torch.tensor([0]), empty_slots=[slot], sampling_params=sp)  # fmt: skip
        return int(torch.as_tensor(o).reshape(-1)[0])

    # the plain twin: B alone, seeded, N decode steps
    twin = _adapter(hf_config, max_batch=4, fused_sampling=True)
    tp = Pages(256, 64)
    want = [prefill(twin, tp, "b", pb, 0, b_row(1, 0))]
    for _ in range(N):
        S = twin.max_batch_size
        tokens = torch.full((S, 1), -1, dtype=torch.int64)
        start = torch.full((S,), -1, dtype=torch.int64)
        tokens[0, 0], start[0] = want[-1], len(pb) + len(want) - 1
        tp.ensure("b", len(pb) + len(want) + 1)
        o = twin.decode_forward(tokens, start, tp.table(["b"] + [None] * (S - 1)), twin.kv_cache,
                                sampling_params=b_row(S, 0), read_from_device=True, reload_inputs=True)  # fmt: skip
        want.append(int(torch.as_tensor(o).reshape(-1)[0]))

    model = _adapter(hf_config, max_batch=4, fused_sampling=True)
    gen = model.generator
    if not restore:
        monkeypatch.setattr(gen, "restore_sampler_rng", lambda rows: None)
    S = model.max_batch_size
    ra = gen.backend.reference_greedy(pa, 40)
    gen.backend.spec = MockSpec(gen.backend, model.kv_cache, K, {len(pa) + i: t for i, t in enumerate(ra)}, 0.0)
    pages = Pages(256, 64)
    seq_a = list(pa) + [prefill(model, pages, "a", pa, 0, _sampling(1))]
    counts, last_n = torch.ones(S, dtype=torch.int32), 1
    for _ in range(2):  # A alone: speculative steps (exact drafts: every one accepted)
        committed = torch.full((S, K + 1), -1, dtype=torch.int32)
        committed[0, :last_n] = torch.tensor(seq_a[-last_n:], dtype=torch.int32)
        positions = torch.full((S, K + 1), -1, dtype=torch.int64)
        positions[0] = len(seq_a) - 1 + torch.arange(K + 1)
        dr = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
        assert int(dr.num_valid[0]) == K
        pages.ensure("a", len(seq_a) + K + 2)
        tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
        start = torch.full((S, K + 1), -1, dtype=torch.int64)
        tokens[0] = torch.tensor([seq_a[-1]] + dr.draft_token_ids[0].tolist())
        start[0] = len(seq_a) - 1 + torch.arange(K + 1)
        vo = model.decode_forward(tokens, start, pages.table(["a", None, None, None]), model.kv_cache,
                                  sampling_params=_sampling(S), num_valid_drafts=dr.num_valid, accepted_counts=counts,
                                  spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
        com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, tokens[:, 1:].to(torch.int32), dr.num_valid)
        last_n = int(cnt[0])
        seq_a += [int(v) for v in com[0, :last_n]]
        counts = torch.ones(S, dtype=torch.int32)
        counts[0] = last_n
    committed = torch.full((S, K + 1), -1, dtype=torch.int32)
    committed[0, :last_n] = torch.tensor(seq_a[-last_n:], dtype=torch.int32)
    positions = torch.full((S, K + 1), -1, dtype=torch.int64)
    positions[0] = len(seq_a) - 1 + torch.arange(K + 1)
    stale = model.propose_draft_tokens(K, committed, positions, counts, hidden=None)
    assert int(stale.num_valid[0]) == K
    got = [prefill(model, pages, "b", pb, 1, b_row(1, 0))]  # B arrives (seeded, sampled) in slot 1
    pages.ensure("a", len(seq_a) + K + 2)
    pages.ensure("b", len(pb) + 2)
    tokens = torch.full((S, K + 1), -1, dtype=torch.int64)
    start = torch.full((S, K + 1), -1, dtype=torch.int64)
    tokens[0] = torch.tensor([seq_a[-1]] + stale.draft_token_ids[0].tolist())
    start[0] = len(seq_a) - 1 + torch.arange(K + 1)
    tokens[1, 0], start[1, 0] = got[-1], len(pb)
    nv = torch.zeros(S, dtype=torch.int32)
    nv[0] = K
    sub0 = model.counters.get("spec_plain_substeps", 0)
    vo = model.decode_forward(tokens, start, pages.table(["a", "b", None, None]), model.kv_cache,
                              sampling_params=b_row(S, 1), num_valid_drafts=nv, accepted_counts=counts,
                              spec_mode="argmax_ids", reload_inputs=True)  # fmt: skip
    assert model.counters.get("spec_plain_substeps", 0) - sub0 == K + 1, "A's exact drafts run every sub-step"
    com, cnt = spec_decode.accept_greedy_drafts(vo.argmax_ids, tokens[:, 1:].to(torch.int32), nv)
    assert int(cnt[0]) == K + 1 and int(cnt[1]) == 1
    seq_a += [int(v) for v in com[0, : K + 1]]
    got.append(int(com[1, 0]))
    while len(got) < N + 1:  # both decode with ordinary steps (two live: no drafts)
        pages.ensure("a", len(seq_a) + 1)
        pages.ensure("b", len(pb) + len(got) + 1)
        tokens = torch.full((S, 1), -1, dtype=torch.int64)
        start = torch.full((S,), -1, dtype=torch.int64)
        tokens[0, 0], start[0] = seq_a[-1], len(seq_a) - 1
        tokens[1, 0], start[1] = got[-1], len(pb) + len(got) - 1
        o = model.decode_forward(tokens, start, pages.table(["a", "b", None, None]), model.kv_cache,
                                 sampling_params=b_row(S, 1), read_from_device=True, reload_inputs=True)  # fmt: skip
        o = torch.as_tensor(o).reshape(S, -1)
        seq_a.append(int(o[0, 0]))
        got.append(int(o[1, 0]))
    assert seq_a[len(pa):] == ra[: len(seq_a) - len(pa)], "A (greedy) == the oracle"
    if restore:
        assert got == want, (got, want)
        assert gen.counters.get("sampler_rng_restores", 0) == 1, gen.counters
    else:
        assert got[:2] == want[:2] and got != want, "the negative control: K extra draws change B's later tokens"
