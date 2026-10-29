# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of `tt/generator.py` against CPU mocks of the device model (stage 06a; no device, no ttnn).

The MockModel stacks one MockDecoder per layer kind (Stage 1 contract) and has an independent unpaged, unchunked
full-sequence reference; every generator path is compared token for token (and logits within fp tolerance) against
it. Run under the no-tt-devices sandbox:

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_generator.py
"""

from __future__ import annotations

import ast
import inspect
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mock_model as M  # noqa: E402
from host_evidence import expect_error, record  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

KINDS = ("dense_kda", "sparse_dsa", "sparse_kda")


def prompt(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(1234 + 7 * n + seed)
    return torch.randint(0, M.TOKENIZER_VOCAB, (n,), generator=g).tolist()


def make(dsa_mode="dense_exact", max_batch=8, max_seq_len=None, **kw):
    m = M.MockModel(
        KINDS,
        max_batch=max_batch,
        dsa_mode=dsa_mode,
        **{k: v for k, v in kw.items() if k in ("logits_width", "bug", "seed")},
    )
    gk = {k: v for k, v in kw.items() if k in ("host_sampling", "stop_token_ids", "sampler_seed")}
    gk.setdefault("stop_token_ids", ())
    g = G.Glm53Generator(m, max_batch=max_batch, max_seq_len=max_seq_len or min(m.supported_context, 8192), **gk)
    return m, g


def steady_counters(g, before):
    return {k: g.counters[k] - before.get(k, 0) for k in g.counters}


def check_decode_log(m, seq_at, tol=1e-3):
    """Logits-level oracle: for every logged decode step and every active row s (cur_pos p >= 0) the token input is
    seq[p] of the request held by slot s and the logits equal the unpaged reference of seq[:p + 1] (max abs error
    < tol). A corrupted cache entry moves the logits by ~0.2 even when the greedy token survives.
    `seq_at(step, slot)` -> full token sequence of the request in `slot` at that step (None = skip)."""
    worst, n = 0.0, 0
    for step, (pos, tok, logits) in enumerate(m.decode_log or []):
        for s in range(pos.numel()):
            p = int(pos[s])
            seq = seq_at(step, s) if p >= 0 else None
            if seq is None:
                continue
            assert int(tok[s]) == seq[p], ("token feedback", step, s, p, int(tok[s]), seq[p])
            ref = m.reference_logits(seq[: p + 1], last_only=True)[-1]
            err = float((logits[s, : M.VOCAB].double() - ref.double()).abs().max())
            assert err < tol, ("decode logits", step, s, p, err)
            worst, n = max(worst, err), n + 1
    assert n > 0, "no decode rows checked"
    return {"checked_rows": n, "max_abs_err": worst}


# --------------------------------------------------------------------------------------------- contract constants
def test_stage1_constants_match():
    """Generator constants equal Stage 1's (parsed from tt/functional_decoder.py without importing ttnn)."""
    src = open(os.path.join(os.path.dirname(HERE), "tt", "functional_decoder.py")).read()
    consts = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert consts["PREFILL_CHUNK"] == G.PREFILL_CHUNK == 2048
    assert consts["PREFILL_ALIGN"] == G.PREFILL_ALIGN == {"kda": 32, "mla": 64}
    assert consts["BLOCK_SIZE"] == G.BLOCK_SIZE == 64
    for L in (1, 2051, 4096, 32768, 1048576):  # Stage 1 MLA: round_up(ceil((L + 63) / 64), 8)
        assert G.blocks_per_seq(L) == G.round_up(G.round_up(L + 63, 64) // 64, 8)
    record("generator_stage1_constants", prefill_chunk=2048, align=consts["PREFILL_ALIGN"], block=64)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 63, 64, 65, 2047, 2048, 2049, 2051, 2052, 4095, 4096, 4097, 32773])
def test_plan_prefill(n):
    plans = G.plan_prefill(n)
    assert sum(p.n_valid for p in plans) == n
    assert all(p.n_valid == 2048 for p in plans[:-1])
    assert all(p.physical % 64 == 0 and 0 <= p.physical - p.n_valid < 64 for p in plans)
    assert [p.start for p in plans] == [2048 * i for i in range(len(plans))]


# --------------------------------------------------------------------------------------------- non-aligned lengths
def _check_prefill_logits(m, g, ids):
    got = g.prefill_logits(ids)
    assert got.shape == (1, len(ids), M.VOCAB)
    ref = m.reference_logits(ids)
    err = (got[0] - ref).abs().max().item()
    agree = bool(torch.equal(got[0].argmax(-1), ref.argmax(-1)))
    return err, agree


def test_generator_nonaligned():
    """Public paths at non-aligned logical lengths: 1, 31, 33, 2051 on the Stage-1 contract (MLA/DSA dense-exact,
    supported context 2051) and 2052, 4097 on the lifted-bound contract (Stage 5): prefill_logits at every logical
    position and greedy generation equal the unpaged full-sequence reference; 2052 is rejected at the public API
    under the dense-exact contract."""
    rows = []
    m, g = make("dense_exact", max_batch=4)
    for n in (1, 31, 33, 63, 64, 65, 2047, 2048, 2049, 2051):
        ids = prompt(n)
        err, agree = _check_prefill_logits(m, g, ids)
        assert agree and err < 1e-3, (n, err)
        new = min(8, 2052 - n)
        out = g.generate(ids, new)
        assert out == m.reference_greedy(ids, new), n
        rows.append(
            {"len": n, "contract": "dense_exact_2051", "prefill_logits_max_abs_err": err, "generated": len(out)}
        )
    with expect_error(ValueError, "supported context 2051"):
        g.generate(prompt(2052), 1)
    with expect_error(ValueError, "supported context 2051"):
        g.generate(prompt(2000), 53)
    m2, g2 = make("unbounded", max_batch=4, max_seq_len=8192)
    for n in (2052, 4097):
        ids = prompt(n)
        err, agree = _check_prefill_logits(m2, g2, ids)
        assert agree and err < 1e-3, (n, err)
        out = g2.generate(ids, 4)
        assert out == m2.reference_greedy(ids, 4), n
        rows.append(
            {
                "len": n,
                "contract": "unbounded (Stage 5 DSA path)",
                "prefill_logits_max_abs_err": err,
                "generated": len(out),
            }
        )
    record(
        "generator_nonaligned",
        lengths=rows,
        rejected_beyond_context=[2052, "2000+53"],
        counters=g.counters,
        counters_unbounded=g2.counters,
    )


def test_mixed_prompt_lengths_one_batch():
    """Eight users with different lengths and max_new_tokens prefilled into shuffled slots, decoded together."""
    m, g = make("unbounded", max_batch=8, max_seq_len=8192)
    lens = [1, 31, 33, 64, 129, 700, 2052, 4097]
    news = [5, 9, 3, 7, 6, 4, 2, 3]
    slots = [5, 0, 7, 2, 1, 6, 3, 4]
    prompts = [prompt(n, seed=1) for n in lens]
    m.decode_log = []
    outs = g.generate_batch(prompts, news, slots=slots)
    for p, n, o in zip(prompts, news, outs):
        assert o == m.reference_greedy(p, n), len(p)
    seqs = {s: prompts[i] + outs[i] for i, s in enumerate(slots)}
    chk = check_decode_log(m, lambda step, s: seqs.get(s))
    record(
        "generator_mixed_lengths_batch",
        lengths=lens,
        max_new_tokens=news,
        slots=slots,
        logits_check=chk,
        counters=g.counters,
    )


def test_batch32_mixed_and_deterministic():
    """32 users (the decode-slot cap) with mixed lengths, budgets and early stops in one batch; two identical prompts in
    different slots / batch positions give identical tokens; a repeated run is identical."""
    m = M.MockModel(KINDS, max_batch=32)
    gen = torch.Generator().manual_seed(77)
    lens = torch.randint(1, 400, (32,), generator=gen).tolist()
    lens[0], lens[1], lens[31] = 1, 2051 - 8, 64
    prompts = [prompt(n, seed=40 + i) for i, n in enumerate(lens)]
    prompts[17] = list(prompts[3])  # identical prompt, different slot and batch position
    news = torch.randint(1, 9, (32,), generator=gen).tolist()
    news[3] = news[17] = 8
    refs = [m.reference_greedy(p, n) for p, n in zip(prompts, news)]
    stop = refs[5][1] if len(refs[5]) > 1 else refs[5][0]  # a real early stop for user 5 (and anyone emitting it)
    exp = [r[: r.index(stop) + 1] if stop in r else r for r in refs]
    early = [i for i in range(32) if len(exp[i]) < news[i]]
    assert early, "no early stop exercised"
    g2 = G.Glm53Generator(m, max_batch=32, max_seq_len=2051, stop_token_ids=[stop])
    slots = list(range(31, -1, -1))
    m.decode_log = []
    outs = g2.generate_batch(prompts, news, slots=slots)
    assert outs == exp and outs[3] == outs[17]
    seqs = {s: prompts[i] + outs[i] for i, s in enumerate(slots)}
    chk = check_decode_log(m, lambda step, s: seqs.get(s))
    m.decode_log = None
    assert g2.generate_batch(prompts, news, slots=slots) == outs
    record(
        "generator_batch32_mixed",
        users=32,
        lengths=lens,
        max_new_tokens=news,
        identical_prompt_slots=[28, 14],
        stop_token=stop,
        early_stopped_users=early,
        logits_check=chk,
        counters=g2.counters,
    )


def test_slot_reuse_no_leakage():
    m, g = make("dense_exact", max_batch=4)
    a, b = prompt(300, seed=2), prompt(77, seed=3)
    g.generate_batch([a], 6, slots=[3])
    m.decode_log = []
    out_b = g.generate_batch([b], 6, slots=[3])[0]
    assert out_b == m.reference_greedy(b, 6)
    chk = check_decode_log(m, lambda step, s: b + out_b if s == 3 else None)
    _, fresh = make("dense_exact", max_batch=4)
    assert fresh.generate_batch([b], 6, slots=[3])[0] == out_b
    kv = g._own.kv_cache
    st_reused = [c.state[3].clone() for c in kv.caches if c.state is not None]
    st_fresh = [c.state[3].clone() for c in fresh._own.kv_cache.caches if c.state is not None]
    assert all(torch.equal(x, y) for x, y in zip(st_reused, st_fresh))
    record(
        "generator_slot_reuse",
        first_len=300,
        second_len=77,
        slot=3,
        kda_state_bitwise_equal_to_fresh=True,
        logits_check=chk,
    )


def test_inactive_slots_untouched():
    """Low-level decode with slot 2 inactive (start_pos -1): its KDA state, conv rows and latent pages stay bitwise
    identical over several replays; the other rows still match the reference; slot 2 then resumes correctly."""
    m, g = make("dense_exact", max_batch=4)
    kv = g.allocate_kv_cache(4, 2051)
    g.warmup(kv)
    prompts = [prompt(n, seed=4) for n in (40, 65, 100, 17)]
    lens = [len(p) for p in prompts]
    P = max(lens)
    tokens = torch.zeros(4, P, dtype=torch.int64)
    for i, p in enumerate(prompts):
        tokens[i, : len(p)] = torch.tensor(p)
    bps = G.blocks_per_seq(2051)
    pt = torch.arange(4 * bps).reshape(4, bps)
    first = g.prefill_forward(
        tokens, page_table=pt, kv_cache=kv, prompt_lens=lens, sampling_params=[SamplingParams()] * 4
    )
    seqs = [p + [int(t)] for p, t in zip(prompts, first)]
    pos = torch.tensor(lens)
    snap = None
    m.decode_log = []
    mla = [c for c in kv.caches if c.latent is not None][0]
    kda = [c for c in kv.caches if c.state is not None]
    for step in range(6):
        sp = pos.clone()
        if step < 4:
            sp[2] = -1
        if step == 0:
            snap = ([c.state[2].clone() for c in kda], [c.conv[2].clone() for c in kda], mla.latent[pt[2]].clone())
        out = g.decode_forward(
            torch.tensor([s[-1] for s in seqs]), sp, page_table=pt, kv_cache=kv, sampling_params=[SamplingParams()] * 4
        )
        if step < 4:
            assert all(torch.equal(a, c.state[2]) for a, c in zip(snap[0], kda))
            assert all(torch.equal(a, c.conv[2]) for a, c in zip(snap[1], kda))
            nn = lambda t: torch.nan_to_num(t, nan=1234.5)  # unwritten pages hold NaN; compare bitwise incl. NaN slots
            assert torch.equal(nn(snap[2]), nn(mla.latent[pt[2]]))
        for u in range(4):
            if sp[u] >= 0:
                seqs[u].append(int(out[u]))
                pos[u] += 1
    for u in range(4):
        n_gen = len(seqs[u]) - lens[u]
        assert seqs[u][lens[u] :] == m.reference_greedy(prompts[u], n_gen), u
    chk = check_decode_log(m, lambda step, s: seqs[s])
    record(
        "generator_inactive_slots_untouched",
        inactive_slot=2,
        inactive_replays=4,
        resumed_steps=2,
        logits_check=chk,
        counters=g.counters,
    )


def test_page_table_rewrite_under_live_trace():
    """A new request with a shuffled block mapping arrives while the decode trace is live: the page table is
    rewritten in place (one refresh, no recapture) and the replay reads the new mapping. Negative control: rebinding
    a NEW device buffer instead of writing in place leaves the trace on the stale table and the outputs diverge."""
    m, g = make("dense_exact", max_batch=4)
    a = prompt(150, seed=5)
    g.generate_batch([a], 3, slots=[0])
    b_ = g._own
    assert b_.decode_traces
    bps = b_.blocks_per_seq
    pt = b_.pt_shadow.clone()
    pt[1] = pt[1].flip(0)  # slot 1's blocks in reverse order: a different physical layout
    before = dict(g.counters)
    p1 = prompt(130, seed=6)
    m.decode_log = []
    out = g.generate_batch([p1], 8, slots=[1], page_table=pt)[0]
    d = steady_counters(g, before)
    assert out == m.reference_greedy(p1, 8)
    chk = check_decode_log(m, lambda step, s: p1 + out if s == 1 else None)
    assert d["page_table_refresh"] == 1 and d["decode_trace_captures"] == 0 and d["sampling_trace_captures"] == 0
    dev = m.read(b_.io.page_table).long()
    want = pt[1] if b_.obm is None else b_.phys[1]  # GLM53_MOCK_DSA_SHARD: the device holds the physical row
    assert torch.equal(b_.pt_shadow[1], pt[1]) and torch.equal(dev[1], want)
    assert int(b_.coverage[1]) == bps  # the full supplied row is kept

    # negative control: a generator that rebinds the page table to a fresh buffer (stale under the live trace)
    m2, g2 = make("dense_exact", max_batch=4)
    g2.generate_batch([a], 3, slots=[0])
    orig = g2._write_page_rows

    def rebinding(b, rows, used):
        new = b.pt_shadow.clone()
        for s, r in rows.items():
            r = torch.as_tensor(r).long().reshape(-1)
            new[s] = r[: b.blocks_per_seq]
        b.io.page_table = M.DeviceTensor(new.to(torch.int32), "page_table")  # wrong: not the captured buffer
        m2.set_page_table_host(b.kv_cache, new.to(torch.int32))
        b.pt_shadow = new

    g2._write_page_rows = rebinding
    m2.decode_log = []
    try:
        bad = g2.generate_batch([p1], 8, slots=[1], page_table=pt)[0]
    except M.MockDeviceError as e:  # the stale table reads never-written (NaN) cache blocks: non-finite sampler input
        assert "not finite" in str(e), e
        detector = "sampler_nonfinite_active_row"
    except M.ContractViolation as e:  # GLM53_MOCK_DSA_SHARD: a logical table never reaches the sharded pool
        assert "owner chip" in str(e), e
        detector = "sharded_pool_owner_check"
    else:
        with expect_error(AssertionError):  # the logits oracle flags the stale table at the first decode step
            check_decode_log(m2, lambda step, s: p1 + bad if s == 1 else None)
        detector = "logits_oracle"
    finally:
        g2._write_page_rows = orig
    record(
        "generator_page_table_rewrite_live_trace",
        blocks_per_seq=bps,
        refreshes=d["page_table_refresh"],
        recaptures=d["decode_trace_captures"],
        logits_check=chk,
        negative_control_detected=True,
        negative_control_detector=detector,
    )


def test_stop_tokens_and_max_new_tokens():
    m, g = make("dense_exact", max_batch=4)
    p0, p1, p2 = prompt(60, seed=7), prompt(91, seed=8), prompt(33, seed=9)
    traj0 = m.reference_greedy(p0, 12)
    stop = traj0[4]
    first = traj0.index(stop)
    outs = g.generate_batch([p0, p1, p2], [12, 10, 1], stop_token_ids=[stop])
    assert outs[0] == traj0[: first + 1]
    traj1 = m.reference_greedy(p1, 10)
    exp1 = traj1[: traj1.index(stop) + 1] if stop in traj1 else traj1
    assert outs[1] == exp1 and len(outs[2]) == 1 and outs[2] == m.reference_greedy(p2, 1)
    assert g.generate(p0, 12, stop_token_ids=[stop], next_input=lambda s, t: traj0[s]) == traj0  # TF: no early stop
    for n in (1, 2, 17):
        assert len(g.generate(p1, n, stop_token_ids=[])) == n
    record("generator_stop_tokens_max_new_tokens", stop_token=stop, stopped_at=first, user1_len=len(outs[1]))


def test_default_stop_set_is_tokenizer_eos():
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    m = M.MockModel(KINDS, max_batch=2)
    g = G.build_generator(
        "models/autoports/zai_org_glm_5_3_flash",
        None,
        backend=m,
        tokenizer=load_tokenizer(),
        max_batch=2,
        max_seq_len=2051,
    )
    assert g.stop_token_ids == frozenset({154820, 154827, 154829})
    record("generator_default_stop_set", stop_token_ids=sorted(g.stop_token_ids))


def test_split_sampling_trace_counters_and_feedback():
    """Steady-state token-out loop = decode replay + sampler replay + one token readback; no token, position,
    page-table or parameter refresh; the token sampled at step N is the token input of step N+1 (tt_out_tok), and
    positions advance on device. Greedy -> sampled -> greedy reuses the captured traces."""
    m, g = make("dense_exact", max_batch=4)
    ids = prompt(80, seed=10)
    g.warmup()
    seen = []
    orig = m.decode

    def spy(io, kv, width):
        seen.append((io.tokens._t.clone(), io.cur_pos._t.clone(), io.sampled._t.clone()))
        orig(io, kv, width)

    m.decode = spy
    before = dict(g.counters)
    out = g.generate(ids, 10)
    d = steady_counters(g, before)
    m.decode = orig
    assert out == m.reference_greedy(ids, 10)
    assert d["decode_trace_replays"] == 9 and d["sampling_trace_replays"] == 9 and d["token_readback"] == 9
    assert d["token_refresh"] == 0 and d["page_table_refresh"] == 0 and d["logits_readback"] == 0
    assert d["pos_refresh"] == 1  # request admission only
    for step in range(1, len(seen)):
        assert int(seen[step][0][0]) == int(seen[step][2][0]) == out[step]  # step-N sample is step-N+1 input
        assert int(seen[step][1][0]) == int(seen[step - 1][1][0]) + 1  # device-side position advance
    caps = (g.counters["decode_trace_captures"], g.counters["sampling_trace_captures"])
    g.generate(ids, 6, sampling=SamplingParams(temperature=0.7, top_k=20, top_p=0.9, seed=3))
    g.generate(ids, 4)
    assert (
        (g.counters["decode_trace_captures"], g.counters["sampling_trace_captures"]) == caps == (3, 2)
    )  # widths 1,2,4
    assert g._own.width_replays[1] == 9 + 5 + 3  # single-user decode always replays the width-1 trace
    record(
        "generator_split_sampling_trace",
        steady_state_per_10_tokens=d,
        captures={"decode": 1, "sampling": 2},
        token_feedback_verified_steps=len(seen) - 1,
    )


def test_teacher_forcing_next_input():
    m, g = make("dense_exact", max_batch=2)
    ids = prompt(45, seed=11)
    forced = prompt(12, seed=12)
    calls = []

    def nxt(step, pred):
        calls.append((step, pred))
        return forced[step]

    seen = []
    orig = m.decode

    def spy(io, kv, width):  # device buffers at the start of each decode step
        seen.append((int(io.tokens._t[0]), int(io.sampled._t[0])))
        orig(io, kv, width)

    m.decode = spy
    assert "enable_trace" in inspect.signature(g.generate).parameters
    out = g.generate(ids, 12, next_input=nxt, enable_trace=True)
    m.decode = orig
    assert [c[0] for c in calls] == list(range(12)) and [c[1] for c in calls] == out
    exp = [int(torch.argmax(m.reference_logits(ids + forced[:i], last_only=True)[-1])) for i in range(12)]
    assert out == exp
    # host-written (forced) inputs: io.tokens holds the forced token while io.sampled still holds the prediction; the
    # generator only reads io.sampled, so this divergence is the documented pipeline semantics (contract 5)
    differ = sum(1 for t, smp in seen[1:] if t != smp)
    assert [t for t, _ in seen] == forced[:11] and differ >= 1
    record(
        "generator_teacher_forcing",
        steps=12,
        callbacks=len(calls),
        overrides=g.counters["teacher_forced_overrides"],
        steps_with_tokens_ne_sampled=differ,
    )


def test_host_sampling_compat_mode():
    m, g = make("dense_exact", max_batch=2, host_sampling=True)
    ids = prompt(70, seed=13)
    before = dict(g.counters)
    out = g.generate(ids, 8)
    d = steady_counters(g, before)
    assert out == m.reference_greedy(ids, 8)
    assert d["logits_readback"] == 7 and d["host_sampling_steps"] == 8 and d["sampling_trace_replays"] == 0
    sp = SamplingParams(temperature=0.9, top_k=50, top_p=0.95, seed=21)
    assert g.generate(ids, 6, sampling=sp) == g.generate(ids, 6, sampling=sp)
    record("generator_host_sampling_compat", explicit_flag="host_sampling=True", per_step=d)


def test_sampling_seeded_and_within_topk():
    m, g = make("dense_exact", max_batch=2)
    ids = prompt(50, seed=14)
    sp = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=7)
    a = g.generate(ids, 10, sampling=sp)
    b = g.generate(ids, 10, sampling=sp)
    c = g.generate(ids, 10, sampling=SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=8))
    assert a == b and a != c
    for i, t in enumerate(a):
        top = torch.topk(m.reference_logits(ids + a[:i], last_only=True)[-1], 20).indices.tolist()
        assert t in top
    record("generator_sampling_seeded", tokens=a, other_seed=c)


def test_two_phase_warmup_and_negative_control():
    """Prefill of every physical bucket while traces are live compiles nothing; skipping warmup phase 1 makes the
    first unprepared prefill compile under a live trace, which the mock device rejects."""
    m, g = make("dense_exact", max_batch=2)
    g.warmup()
    n_prog = len(m.device.program_cache)
    for n in list(range(1, 2048, 61)) + [2048, 2049, 2051]:
        g.prefill_logits(prompt(n, seed=15)) if n <= 300 else g.generate(prompt(n, seed=15), 1)
    assert len(m.device.program_cache) == n_prog
    m2, g2 = make("dense_exact", max_batch=2)
    m2.prepare_prefill = lambda kv: None
    g2.warmup()
    with expect_error(M.MockDeviceError, "compiled while"):
        g2.generate(prompt(77), 2)
    record("generator_two_phase_warmup", programs=n_prog, negative_control="compile under live trace raised")


def test_low_level_vllm_style():
    """Caller-owned cache, non-identity slots and blocks, on-device sampled first tokens, then decode_forward with
    tokens=-1 (device-owned tt_out_tok) and with explicit tokens; logits path without sampling params."""
    m, g = make("dense_exact", max_batch=6)
    kv = g.allocate_kv_cache(6, 1024)
    g.warmup(kv)
    bps = G.blocks_per_seq(1024)
    users = {5: prompt(90, seed=16), 2: prompt(200, seed=17)}
    free = list(range(6 * bps))[::-1]
    rows = {s: torch.tensor(free[i * bps : (i + 1) * bps]) for i, s in enumerate(users)}
    P = 256
    toks = torch.zeros(2, P, dtype=torch.int64)
    for i, (s, p) in enumerate(users.items()):
        toks[i, : len(p)] = torch.tensor(p)
    first = g.prefill_forward(
        toks,
        page_table=torch.stack(list(rows.values())),
        kv_cache=kv,
        prompt_lens=[len(p) for p in users.values()],
        empty_slots=list(users),
        sampling_params=[SamplingParams()] * 2,
    )
    seqs = {s: p + [int(t)] for (s, p), t in zip(users.items(), first)}
    before = dict(g.counters)
    m.decode_log = []
    for step in range(5):
        sl = list(users)
        tok = torch.full((2,), -1) if step % 2 else torch.tensor([seqs[s][-1] for s in sl])
        sp = torch.tensor([len(seqs[s]) - 1 for s in sl])
        out = g.decode_forward(
            tok,
            sp,
            page_table=torch.stack([rows[s] for s in sl]),
            kv_cache=kv,
            slots=sl,
            sampling_params=[SamplingParams()] * 2,
        )
        for s, t in zip(sl, out.tolist()):
            seqs[s].append(int(t))
    d = steady_counters(g, before)
    for s, p in users.items():
        assert seqs[s][len(p) :] == m.reference_greedy(p, 6)
    assert d["token_refresh"] == 0 and d["page_table_refresh"] == 0 and d["pos_refresh"] == 1
    lg = g.decode_forward(
        torch.tensor([seqs[5][-1]]), torch.tensor([len(seqs[5]) - 1]), page_table=None, kv_cache=kv, slots=[5]
    )
    assert lg.shape == (1, M.VOCAB)
    assert int(lg.argmax()) == m.reference_greedy(users[5], 7)[-1]
    chk = check_decode_log(m, lambda step, s: seqs.get(s))
    record("generator_low_level_vllm_style", slots=list(users), steady=d, logits_check=chk)


def test_async_decode_split_and_slot_remap():
    """vLLM async split: decode_forward(read_from_device=False) leaves the sampled tokens on device (already the next
    input), read_decode_output / process_decode_output_host read and format them; then a scheduler slot_remap moves
    two live requests (a chain 0 -> 3, 1 -> 0) with their KDA state and page rows, and decoding continues exactly."""
    m, g = make("dense_exact", max_batch=4)
    kv = g.allocate_kv_cache(4, 1024)
    g.warmup(kv)
    bps = G.blocks_per_seq(1024)
    users = [prompt(50, seed=20), prompt(77, seed=21)]
    toks = torch.zeros(2, 128, dtype=torch.int64)
    for i, p in enumerate(users):
        toks[i, : len(p)] = torch.tensor(p)
    pt = torch.arange(4 * bps).reshape(4, bps)
    first = g.prefill_forward(
        toks, page_table=pt[:2], kv_cache=kv, prompt_lens=[50, 77], sampling_params=[SamplingParams()] * 2
    )
    seqs = [p + [int(t)] for p, t in zip(users, first)]
    slots = [0, 1]
    before = dict(g.counters)
    m.decode_log = []
    for step in range(6):
        if step == 3:
            g.remap_slots(kv, {0: 3, 1: 0})
            slots = [3, 0]
        sp = torch.tensor([len(s) - 1 for s in seqs])
        h = g.decode_forward(
            torch.full((2,), -1),
            sp,
            page_table=None,
            kv_cache=kv,
            slots=slots,
            sampling_params=[SamplingParams()] * 2,
            read_from_device=False,
        )
        assert isinstance(h, G.DecodeOutput)
        out = g.process_decode_output_host(g.read_decode_output(h), is_tokens=True)
        assert out.shape == (2, 1)
        for s, t in zip(seqs, out.reshape(-1).tolist()):
            s.append(int(t))
    d = steady_counters(g, before)
    for u, p in enumerate(users):
        assert seqs[u][len(p) :] == m.reference_greedy(p, 7), u
    assert d["token_refresh"] <= 1 and d["page_table_refresh"] == 1
    owner = lambda step, s: ({0: seqs[0], 1: seqs[1]} if step < 3 else {3: seqs[0], 0: seqs[1]}).get(s)
    chk = check_decode_log(m, owner)
    record("generator_async_split_and_slot_remap", remap={0: 3, 1: 0}, steps=6, counters=d, logits_check=chk)


def test_serving_commands_09a_convention():
    """09a adapter calling convention: caller-sized block pool (num_blocks smaller than slots x blocks_per_seq),
    warmup_serving(prepare -> capture), prefill with `slots=`, decode with reload commands (reload_inputs=False reuses
    device-resident tokens / positions), list-form slot_remap, release_slot, trace_report."""
    m, g = make("dense_exact", max_batch=4)
    bps = G.blocks_per_seq(1024)
    kv = g.allocate_kv_cache(4, 1024, num_blocks=40)  # < 4 x 24 blocks: a shared pool
    g.warmup_serving(kv, phase="prepare")
    g.warmup_serving(kv, phase="capture")
    users = [prompt(64, seed=30), prompt(70, seed=31)]
    rows = torch.tensor([[3, 7, 9] + [9] * (bps - 3), [12, 5, 21] + [21] * (bps - 3)])
    toks = torch.zeros(2, 128, dtype=torch.int64)
    for i, p in enumerate(users):
        toks[i, : len(p)] = torch.tensor(p)
    first = g.prefill_forward(
        toks, page_table=rows, kv_cache=kv, prompt_lens=[64, 70], slots=[1, 2], sampling_params=[SamplingParams()] * 2
    )
    seqs = {1: users[0] + [int(first[0])], 2: users[1] + [int(first[1])]}
    sp = [SamplingParams()] * 2
    out = g.decode_forward(
        torch.tensor([seqs[1][-1], seqs[2][-1]]),
        torch.tensor([64, 70]),
        page_table=rows,
        kv_cache=kv,
        slots=[1, 2],
        sampling_params=sp,
        reload_inputs=True,
    )
    for s, t in zip((1, 2), out.tolist()):
        seqs[s].append(int(t))
    before = dict(g.counters)
    for _ in range(3):  # steady state: nothing reloaded
        out = g.decode_forward(None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False)
        for s, t in zip((1, 2), out.tolist()):
            seqs[s].append(int(t))
    d = steady_counters(g, before)
    assert d["token_refresh"] == d["pos_refresh"] == d["page_table_refresh"] == 0
    seqs[3] = seqs.pop(1)  # slot 1's request moves to slot 3 (rows of a reload_inputs=False step: live slots, sorted)
    for remap in ([-1, -1, -1, 1], None):
        out = g.decode_forward(
            None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False, slot_remap=remap
        )
        for s, t in zip((2, 3), out.tolist()):
            seqs[s].append(int(t))
    g.release_slot(kv, 2)
    rep = g.trace_report()
    assert rep["bindings"][0]["num_blocks"] == 40 and rep["counters"]["decode_trace_captures"] == 3
    assert seqs[3][64:] == m.reference_greedy(users[0], len(seqs[3]) - 64)
    assert seqs[2][70:] == m.reference_greedy(users[1], len(seqs[2]) - 70)
    record("generator_serving_commands_09a", num_blocks=40, steady=d, trace_report=rep)


def test_null_block_row_tail():
    """int24svc 2026-10-22: a caller row's tail past its coverage is the cache's null block (vLLM's block 0) -- the TP
    prefill writes a bucket's pad pages through the tail (tt/prefill_tp.py write_inputs), and a tail repeating the
    request's last page took the pad rows' latents. Without a null block the tail repeats the last entry; a row
    holding the null block is refused. The host shadow the prefill reads equals the device table."""
    bps = G.blocks_per_seq(1024)
    p = prompt(100, seed=7)
    gens = {}
    for null, tail in ((0, 0), (None, 9)):
        m, g = make("dense_exact", max_batch=4)
        kv = g.allocate_kv_cache(4, 1024, num_blocks=40, null_block=null)
        g.warmup_serving(kv, phase="prepare")
        g.warmup_serving(kv, phase="capture")
        row = torch.tensor([[3, 9] + [-1] * (bps - 2)])
        first = g.prefill_forward(_prefill_rows([p]), page_table=row, kv_cache=kv, prompt_lens=[100], slots=[1],
                                  sampling_params=[SamplingParams()])  # fmt: skip
        b = g._binding(kv)
        assert b.pt_shadow[1, :2].tolist() == [3, 9] and (b.pt_shadow[1, 2:] == tail).all() and int(b.coverage[1]) == 2
        host = next(c.page_table_host for l, c in zip(m.layers, kv.caches) if not l.is_kda)
        assert torch.equal(host[1].long(), b.pt_shadow[1] if b.obm is None else b.phys[1])  # (GLM53_MOCK_DSA_SHARD)
        out = g.decode_forward(torch.tensor([int(first[0])]), torch.tensor([100]), page_table=row, kv_cache=kv,
                               slots=[1], sampling_params=[SamplingParams()], reload_inputs=True)  # fmt: skip
        assert int(out[0]) == m.reference_greedy(p, 2)[1]
        gens[null] = (g, kv)
    g, kv = gens[0]
    with pytest.raises(ValueError, match="null block"):
        g.prefill_forward(_prefill_rows([p]), page_table=torch.tensor([[5, 0] + [-1] * (bps - 2)]), kv_cache=kv,
                          prompt_lens=[100], slots=[2], sampling_params=[SamplingParams()])  # fmt: skip
    m, g = make("dense_exact", max_batch=4)
    with pytest.raises(ValueError, match="null block"):
        g.allocate_kv_cache(4, 1024, num_blocks=40, null_block=40)
    record("generator_null_block_row_tail", bps=bps)


def _prefill_rows(prompts):
    P = max(len(p) for p in prompts)
    toks = torch.zeros(len(prompts), P, dtype=torch.int64)
    for i, p in enumerate(prompts):
        toks[i, : len(p)] = torch.tensor(p)
    return toks


def test_page_coverage_and_identity_restore():
    """Review P1 regression: (B) a full caller row is kept, so decoding past the prefill's blocks with
    page_table=None stays on the request's own blocks; (C) a slot whose row was replaced gets its own blocks back for a
    later request without a page table, and reset() restores the identity table; a row that covers too few blocks for
    a decode position raises instead of aliasing another block."""
    m, g = make("dense_exact", max_batch=4)
    kv = g.allocate_kv_cache(4, 1024)
    g.warmup(kv)
    bps = G.blocks_per_seq(1024)
    p = prompt(60, seed=50)
    row = torch.arange(bps, 2 * bps).flip(0)  # slot 1's own blocks, reversed: a full valid row
    first = g.prefill_forward(
        _prefill_rows([p]),
        page_table=row[None],
        kv_cache=kv,
        prompt_lens=[60],
        empty_slots=[1],
        sampling_params=[SamplingParams()],
    )
    seq = p + [int(first[0])]
    m.decode_log = []
    for _ in range(10):  # 60 -> 70 crosses the first 64-token block with page_table=None
        out = g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([len(seq) - 1]),
            page_table=None,
            kv_cache=kv,
            slots=[1],
            sampling_params=[SamplingParams()],
        )
        seq.append(int(out[0]))
    assert seq[60:] == m.reference_greedy(p, 11)
    chk_b = check_decode_log(m, lambda step, s: seq if s == 1 else None)
    # a short row: prefill fits one block, decoding into the second block must raise
    g.release_slot(kv, 1)
    g.prefill_forward(
        _prefill_rows([p]),
        page_table=torch.tensor([[bps + 3]]),
        kv_cache=kv,
        prompt_lens=[60],
        empty_slots=[1],
        sampling_params=[SamplingParams()],
    )
    for pos in range(60, 64):
        g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([pos]),
            page_table=None,
            kv_cache=kv,
            slots=[1],
            sampling_params=[SamplingParams()],
        )
    with expect_error(ValueError, "page-table coverage"):
        g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([64]),
            page_table=None,
            kv_cache=kv,
            slots=[1],
            sampling_params=[SamplingParams()],
        )
    # (C) generator-owned slot reuse after a caller page table, then reset()
    m2, g2 = make("dense_exact", max_batch=4)
    g2.generate_batch([prompt(40, seed=51)], 3, slots=[1])
    own = g2._own
    flipped = own.pt_shadow.clone()
    flipped[1] = flipped[1].flip(0)
    g2.generate_batch([prompt(40, seed=52)], 3, slots=[1], page_table=flipped)
    q = prompt(300, seed=53)
    m2.decode_log = []
    out_q = g2.generate_batch([q], 6, slots=[1])[0]
    assert out_q == m2.reference_greedy(q, 6) and torch.equal(own.pt_shadow[1], own.identity_row(1))
    chk_c = check_decode_log(m2, lambda step, s: q + out_q if s == 1 else None)
    g2.generate_batch([prompt(40, seed=54)], 2, slots=[2], page_table=flipped.flip(1))
    g2.reset()
    ident = torch.stack([own.identity_row(s) for s in range(4)])
    assert torch.equal(own.pt_shadow, ident)
    assert torch.equal(m2.read(own.io.page_table).long(), ident if own.obm is None else own.phys)
    record(
        "generator_page_coverage_identity_restore",
        full_row_decode=chk_b,
        slot_reuse_after_page_table=chk_c,
        short_row_rejected=True,
        reset_restores_identity=True,
    )


def test_release_slot_then_reuse_blocks():
    """Review P1 regression: release a slot, give its blocks to a new request in another slot, decode with
    reload_inputs=False: only live slots decode (the released slot gets cur_pos -1), the new owner's prompt pages stay
    bitwise intact, and all outputs match the reference."""
    m, g = make("dense_exact", max_batch=4)
    bps = G.blocks_per_seq(1024)
    kv = g.allocate_kv_cache(4, 1024, num_blocks=40)
    g.warmup_serving(kv, phase="prepare")
    g.warmup_serving(kv, phase="capture")
    u1, u2, u0 = prompt(64, seed=60), prompt(70, seed=61), prompt(100, seed=62)
    rows = torch.tensor([[3, 7, 9] + [-1] * (bps - 3), [12, 5, 21] + [-1] * (bps - 3)])
    first = g.prefill_forward(
        _prefill_rows([u1, u2]),
        page_table=rows,
        kv_cache=kv,
        prompt_lens=[64, 70],
        slots=[1, 2],
        sampling_params=[SamplingParams()] * 2,
    )
    seqs = {1: u1 + [int(first[0])], 2: u2 + [int(first[1])]}
    sp = [SamplingParams()] * 2
    out = g.decode_forward(
        torch.tensor([-1, -1]), torch.tensor([64, 70]), page_table=None, kv_cache=kv, slots=[1, 2], sampling_params=sp
    )
    for s, t in zip((1, 2), out.tolist()):
        seqs[s].append(int(t))
    g.release_slot(kv, 2)
    f0 = g.prefill_forward(
        _prefill_rows([u0]),
        page_table=torch.tensor([[12, 5]]),
        kv_cache=kv,
        prompt_lens=[100],
        slots=[0],
        sampling_params=[SamplingParams()],
    )
    seqs[0] = u0 + [int(f0[0])]
    mla = [c for c in kv.caches if c.latent is not None][0]
    snap = mla.latent[torch.tensor([12, 5])].clone()
    m.decode_log = []
    out = g.decode_forward(None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False)
    assert out.numel() == 2  # live rows: slots 0 and 1 (slot 2 released)
    for s, t in zip((0, 1), out.tolist()):
        seqs[s].append(int(t))
    pos, _, _ = m.decode_log[-1]
    assert int(pos[2]) == -1
    after = mla.latent[torch.tensor([12, 5])]
    flat_b, flat_a = snap.reshape(-1, snap.shape[-1]), after.reshape(-1, after.shape[-1])
    keep = torch.arange(flat_b.shape[0]) != 100  # position 100 = slot 0's own new decode token
    nn = lambda t: torch.nan_to_num(t, nan=1234.5)
    assert torch.equal(nn(flat_b[keep]), nn(flat_a[keep]))
    assert seqs[0][100:] == m.reference_greedy(u0, 2) and seqs[1][64:] == m.reference_greedy(u1, 3)
    chk = check_decode_log(m, lambda step, s: seqs.get(s) if s in (0, 1) else None)
    record("generator_release_slot_block_reuse", released=2, reused_blocks=[12, 5], logits_check=chk)


def test_unlisted_slot_device_token_preserved():
    """Review P2 regression: an async step leaves slot 3's next input only on device; a step that does not list slot 3
    would let the sampler overwrite that row - the generator reads it back first, so slot 3 later resumes with -1."""
    m, g = make("dense_exact", max_batch=4)
    kv = g.allocate_kv_cache(4, 1024)
    g.warmup(kv)
    bps = G.blocks_per_seq(1024)
    a, c = prompt(40, seed=70), prompt(55, seed=71)
    pt = torch.arange(4 * bps).reshape(4, bps)
    first = g.prefill_forward(
        _prefill_rows([a, c]),
        page_table=pt[[0, 3]],
        kv_cache=kv,
        prompt_lens=[40, 55],
        empty_slots=[0, 3],
        sampling_params=[SamplingParams()] * 2,
    )
    seqs = {0: a + [int(first[0])], 3: c + [int(first[1])]}
    sp = SamplingParams()
    h = g.decode_forward(
        torch.tensor([-1, -1]),
        torch.tensor([40, 55]),
        page_table=None,
        kv_cache=kv,
        slots=[0, 3],
        sampling_params=[sp] * 2,
        read_from_device=False,
    )
    tok = g.read_decode_output(h)  # caller reads only slot 0 later; slot 3's value stays device-owned for the generator
    seqs[0].append(int(tok[0]))
    seqs[3].append(int(tok[1]))
    g._binding(kv).tok_val[3] = -1  # forget the host copy: the device row is now the only copy
    before = g.counters["token_readback"]
    out = g.decode_forward(
        torch.tensor([-1]), torch.tensor([41]), page_table=None, kv_cache=kv, slots=[0], sampling_params=[sp]
    )
    seqs[0].append(int(out[0]))
    assert g.counters["token_readback"] - before >= 2  # preservation read + the step's own token read
    m.decode_log = []
    out = g.decode_forward(
        torch.tensor([-1]), torch.tensor([56]), page_table=None, kv_cache=kv, slots=[3], sampling_params=[sp]
    )
    seqs[3].append(int(out[0]))
    assert seqs[3][55:] == m.reference_greedy(c, 3) and seqs[0][40:] == m.reference_greedy(a, 3)
    chk = check_decode_log(m, lambda step, s: seqs[3] if s == 3 else None)
    record("generator_unlisted_slot_token_preserved", logits_check=chk)


def test_prefill_continuation_interleaved_and_all_logits():
    """Chunked prefill continuation (start_pos > 0, page-aligned) of slot 0 with decode steps of slot 1 between its
    chunks; prefill_forward(return_all_logits=True) vs the reference at every logical position."""
    m, g = make("dense_exact", max_batch=4)
    kv = g.allocate_kv_cache(4, 2051)
    g.warmup(kv)
    bps = G.blocks_per_seq(2051)
    pt = torch.arange(4 * bps).reshape(4, bps)
    a, b = prompt(700, seed=80), prompt(90, seed=81)
    f1 = g.prefill_forward(
        _prefill_rows([b]),
        page_table=pt[[1]],
        kv_cache=kv,
        prompt_lens=[90],
        empty_slots=[1],
        sampling_params=[SamplingParams()],
    )
    seq_b = b + [int(f1[0])]
    m.decode_log = []
    for start, end in ((0, 128), (128, 512), (512, 700)):
        r = g.prefill_forward(
            _prefill_rows([a[start:end]]),
            page_table=pt[[0]],
            kv_cache=kv,
            prompt_lens=[end - start],
            empty_slots=[0],
            start_pos=[start],
            sampling_params=[SamplingParams()] if end == 700 else None,
        )
        out = g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([len(seq_b) - 1]),
            page_table=None,
            kv_cache=kv,
            slots=[1],
            sampling_params=[SamplingParams()],
        )
        seq_b.append(int(out[0]))
    seq_a = a + [int(r[0])]
    for _ in range(3):
        out = g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([len(seq_a) - 1]),
            page_table=None,
            kv_cache=kv,
            slots=[0],
            sampling_params=[SamplingParams()],
        )
        seq_a.append(int(out[0]))
    assert seq_a[700:] == m.reference_greedy(a, 4) and seq_b[90:] == m.reference_greedy(b, 4)
    chk = check_decode_log(m, lambda step, s: {0: seq_a, 1: seq_b}.get(s))
    g.release_slot(kv, 0)
    g.release_slot(kv, 1)
    users = [prompt(31, seed=82), prompt(129, seed=83)]
    lg = g.prefill_forward(
        _prefill_rows(users),
        page_table=pt[[2, 3]],
        kv_cache=kv,
        prompt_lens=[31, 129],
        empty_slots=[2, 3],
        return_all_logits=True,
    )
    assert lg.shape == (2, 129, M.VOCAB)
    errs = []
    for i, u in enumerate(users):
        ref = m.reference_logits(u)
        errs.append(float((lg[i, : len(u)] - ref).abs().max()))
        assert errs[-1] < 1e-3 and float(lg[i, len(u) :].abs().max() if len(u) < 129 else 0.0) == 0.0
    record(
        "generator_prefill_continuation_and_all_logits",
        continuation_chunks=[[0, 128], [128, 512], [512, 700]],
        interleaved_decode_steps=3,
        logits_check=chk,
        all_logits_max_abs_err=errs,
    )


def test_rejections():
    """Invalid inputs raise ValueError / RuntimeError instead of corrupting state."""
    m, g = make("dense_exact", max_batch=4)
    kv = g.allocate_kv_cache(4, 1024)
    g.warmup(kv)
    bps = G.blocks_per_seq(1024)
    pt = torch.arange(4 * bps).reshape(4, bps)
    p = prompt(100, seed=90)
    t = _prefill_rows([p])
    cases = {}

    def expect(name, exc, fn, match=None):
        with expect_error(exc, match):
            fn()
        cases[name] = exc.__name__

    g.prefill_forward(
        t, page_table=pt[[0]], kv_cache=kv, prompt_lens=[100], empty_slots=[0], sampling_params=[SamplingParams()]
    )
    expect(
        "page_overlap",
        ValueError,
        lambda: g.prefill_forward(t, page_table=pt[[0]], kv_cache=kv, prompt_lens=[100], empty_slots=[1]),
        "mapped by live slots",
    )
    expect(
        "block_out_of_range",
        ValueError,
        lambda: g.prefill_forward(
            t, page_table=torch.full((1, bps), 4 * bps), kv_cache=kv, prompt_lens=[100], empty_slots=[1]
        ),
        "outside",
    )
    expect(
        "short_row",
        ValueError,
        lambda: g.prefill_forward(t, page_table=pt[[1], :1], kv_cache=kv, prompt_lens=[100], empty_slots=[1]),
        "covers",
    )
    expect(
        "continuation_not_page_aligned",
        ValueError,
        lambda: g.prefill_forward(
            t[:, :10], page_table=pt[[0]], kv_cache=kv, prompt_lens=[10], empty_slots=[0], start_pos=[100]
        ),
        "multiple of 64",
    )
    expect(
        "continuation_wrong_length",
        ValueError,
        lambda: g.prefill_forward(
            t[:, :10], page_table=pt[[0]], kv_cache=kv, prompt_lens=[10], empty_slots=[0], start_pos=[64]
        ),
        "cannot continue",
    )
    expect(
        "all_logits_with_sampling",
        ValueError,
        lambda: g.prefill_forward(
            t,
            page_table=pt[[2]],
            kv_cache=kv,
            prompt_lens=[100],
            empty_slots=[2],
            return_all_logits=True,
            sampling_params=[SamplingParams()],
        ),
        "exclusive",
    )
    g.decode_forward(
        torch.tensor([-1]),
        torch.tensor([100]),
        page_table=None,
        kv_cache=kv,
        slots=[0],
        sampling_params=[SamplingParams()],
    )
    expect(
        "decode_position_repeat",
        ValueError,
        lambda: g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([100]),
            page_table=None,
            kv_cache=kv,
            slots=[0],
            sampling_params=[SamplingParams()],
        ),
        "cache length",
    )
    h = g.decode_forward(
        torch.tensor([-1]),
        torch.tensor([101]),
        page_table=None,
        kv_cache=kv,
        slots=[0],
        sampling_params=[SamplingParams()],
        read_from_device=False,
    )
    g.decode_forward(
        torch.tensor([-1]),
        torch.tensor([102]),
        page_table=None,
        kv_cache=kv,
        slots=[0],
        sampling_params=[SamplingParams()],
    )
    expect("stale_decode_output", RuntimeError, lambda: g.read_decode_output(h), "stale")
    g.prefill_forward(
        t, page_table=pt[[1]], kv_cache=kv, prompt_lens=[100], empty_slots=[1]
    )  # logits path: token unknown
    expect(
        "unknown_next_token",
        ValueError,
        lambda: g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([100]),
            page_table=None,
            kv_cache=kv,
            slots=[1],
            sampling_params=[SamplingParams()],
        ),
        "unknown",
    )
    expect("remap_destination_live", ValueError, lambda: g.remap_slots(kv, {0: 1}), "live request")
    expect("remap_cycle", ValueError, lambda: g.remap_slots(kv, {0: 1, 1: 0}), "cyclic")
    expect("context", ValueError, lambda: g.generate(prompt(2050), 3), "supported context")
    expect(
        "implicit_huge_max_seq_len",
        ValueError,
        lambda: G.Glm53Generator(M.MockModel(KINDS, max_batch=2, dsa_mode="unbounded"), max_batch=2),
        "pass max_seq_len",
    )
    record("generator_rejections", cases=cases)


def test_warmup_idempotent_with_live_request():
    """Review P2 regression: a repeated warmup (both entry points) is a no-op for live request state; a warmup that
    still needs program preparation while a request is live raises instead of wiping it."""
    m, g = make("dense_exact", max_batch=2)
    kv = g.allocate_kv_cache(2, 1024)
    g.warmup_serving(kv, phase="prepare", prefill=True, decode=False)
    bps = G.blocks_per_seq(1024)
    p = prompt(80, seed=95)
    first = g.prefill_forward(
        _prefill_rows([p]),
        page_table=torch.arange(bps)[None],
        kv_cache=kv,
        prompt_lens=[80],
        sampling_params=[SamplingParams()],
    )
    with expect_error(RuntimeError, "before any request"):
        g.warmup_serving(kv, phase="capture")  # decode preparation would run with a live request
    g.release_slot(kv, 0)
    g.warmup_serving(kv, phase="capture")
    first = g.prefill_forward(
        _prefill_rows([p]),
        page_table=torch.arange(bps)[None],
        kv_cache=kv,
        prompt_lens=[80],
        sampling_params=[SamplingParams()],
    )
    seq = p + [int(first[0])]
    for i in range(4):
        if i == 2:
            g.warmup(kv)
            g.warmup_serving(kv, phase="capture")
        out = g.decode_forward(
            torch.tensor([-1]),
            torch.tensor([len(seq) - 1]),
            page_table=None,
            kv_cache=kv,
            slots=[0],
            sampling_params=[SamplingParams()],
        )
        seq.append(int(out[0]))
    assert seq[80:] == m.reference_greedy(p, 5)
    record(
        "generator_warmup_idempotent",
        repeated_warmups=2,
        live_request_continued=True,
        prepare_with_live_request="RuntimeError",
    )


def test_remap_moves_sampler_rng():
    """Seeded sampled requests produce the same tokens whether or not they are moved between slots mid-generation
    (KDA state, page row and per-slot sampler RNG move together)."""

    def run(remap):
        m, g = make("dense_exact", max_batch=4)
        kv = g.allocate_kv_cache(4, 1024)
        g.warmup(kv)
        bps = G.blocks_per_seq(1024)
        users = [prompt(45, seed=100), prompt(66, seed=101)]
        sp = [
            SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=7),
            SamplingParams(temperature=1.0, top_k=8, seed=9),
        ]
        pt = torch.arange(4 * bps).reshape(4, bps)
        first = g.prefill_forward(
            _prefill_rows(users), page_table=pt[:2], kv_cache=kv, prompt_lens=[45, 66], sampling_params=sp
        )
        seqs = [u + [int(t)] for u, t in zip(users, first)]
        slots = [0, 1]
        for step in range(6):
            mapping = {0: 2, 1: 0} if (remap and step == 3) else None
            if mapping:
                slots = [2, 0]
            out = g.decode_forward(
                torch.tensor([-1, -1]),
                torch.tensor([len(s) - 1 for s in seqs]),
                page_table=None,
                kv_cache=kv,
                slots=slots,
                sampling_params=sp,
                slot_remap=mapping,
            )
            for s, t in zip(seqs, out.tolist()):
                s.append(int(t))
        return seqs

    plain, moved = run(False), run(True)
    assert plain == moved
    record("generator_remap_moves_sampler_rng", tokens=[s[-7:] for s in plain])


def test_decode_buckets():
    """pow2 decode buckets (DECISIONS 2026-10-17 ~22:15Z): every width is compiled before the first capture and
    captured once; each step replays the smallest width covering its decoding slots; slots at or beyond the width
    (here a paused live request in slot 9) are untouched, and every row matches the reference at the logits level."""
    m, g = make("dense_exact", max_batch=16)
    kv = g.allocate_kv_cache(16, 1024)
    g.warmup(kv)
    b = g._binding(kv)
    assert g.decode_buckets == b.widths == (1, 2, 4, 8, 16) and sorted(b.decode_traces) == [1, 2, 4, 8, 16]
    n_prog = len(m.device.program_cache)
    bps = G.blocks_per_seq(1024)
    pt = torch.arange(16 * bps).reshape(16, bps)
    users = {s: prompt(30 + 7 * s, seed=110 + s) for s in (0, 1, 3, 9)}
    first = g.prefill_forward(
        _prefill_rows(list(users.values())),
        page_table=pt[list(users)],
        kv_cache=kv,
        prompt_lens=[len(p) for p in users.values()],
        empty_slots=list(users),
        sampling_params=[SamplingParams()] * 4,
    )
    seqs = {s: p + [int(t)] for (s, p), t in zip(users.items(), first)}
    kda = [c for c in kv.caches if c.state is not None]
    snap9 = [c.state[9].clone() for c in kda]
    m.decode_log = []
    plan = [([0], 1), ([0, 1], 2), ([0, 1, 3], 4), ([1, 3], 4), ([0], 1), ([0, 1, 3, 9], 16), ([9], 16)]
    widths = []
    unconsumed = 0
    for sl, w in plan:
        before = dict(b.width_replays)
        tv_before = b.tok_val.clone()
        out = g.decode_forward(
            torch.full((len(sl),), -1),
            torch.tensor([len(seqs[s]) - 1 for s in sl]),
            page_table=None,
            kv_cache=kv,
            slots=sl,
            sampling_params=[SamplingParams()] * len(sl),
        )
        used = [k for k in b.width_replays if b.width_replays[k] != before[k]]
        assert used == [w], (sl, used)
        for s in range(16):  # never consumed: the sampler's output for non-decoding rows is not recorded
            if s not in sl:
                assert int(b.tok_val[s]) == int(tv_before[s]) or int(tv_before[s]) == -1
                unconsumed += 1
        widths.append(used[0])
        for s, t in zip(sl, out.tolist()):
            seqs[s].append(int(t))
        if 9 not in sl and len(widths) < 6:
            assert all(torch.equal(a, c.state[9]) for a, c in zip(snap9, kda))  # slot 9 beyond the width: untouched
    assert len(m.device.program_cache) == n_prog  # nothing compiled after the captures
    masked_rows = 0  # inactive rows inside a replayed width: masked by the backend, finite sampler input
    for (pos, _tok, logits), w in zip(m.decode_log, widths):
        for s in range(w):
            if int(pos[s]) < 0:
                assert bool(torch.isfinite(logits[s]).all()), ("inactive row not masked", s, w)
                masked_rows += 1
    assert masked_rows > 0
    for s, p in users.items():
        assert seqs[s][len(p) :] == m.reference_greedy(p, len(seqs[s]) - len(p))
    chk = check_decode_log(m, lambda step, s: seqs.get(s))
    record(
        "generator_decode_buckets",
        widths=list(b.widths),
        steps=[[sl, w] for sl, w in plan],
        logits_check=chk,
        replays_by_width=dict(b.width_replays),
        decode_buckets_attr=list(g.decode_buckets),
        inactive_rows_in_width_finite=masked_rows,
        non_decoding_rows_unconsumed=unconsumed,
    )


def test_reset_sampling_state_reseeds():
    """reset_sampling_state=True re-seeds the decoding slots from their params, so a seeded sampled continuation does
    not depend on how much RNG the slot consumed before (here: an earlier sampled request in the same slot)."""

    def run(history: bool, reset: bool):
        m, g = make("dense_exact", max_batch=2)
        kv = g.allocate_kv_cache(2, 1024)
        g.warmup(kv)
        bps = G.blocks_per_seq(1024)
        row = torch.arange(bps)[None]
        if history:  # an earlier sampled request in slot 0 consumes device RNG
            q = prompt(20, seed=120)
            f = g.prefill_forward(
                _prefill_rows([q]),
                page_table=row,
                kv_cache=kv,
                prompt_lens=[20],
                sampling_params=[SamplingParams(temperature=1.0, top_k=32)],
            )
            seq = q + [int(f[0])]
            for _ in range(3):
                o = g.decode_forward(
                    torch.tensor([-1]),
                    torch.tensor([len(seq) - 1]),
                    page_table=None,
                    kv_cache=kv,
                    slots=[0],
                    sampling_params=[SamplingParams(temperature=1.0, top_k=32)],
                )
                seq.append(int(o[0]))
            g.release_slot(kv, 0)
        a = prompt(50, seed=121)
        f = g.prefill_forward(
            _prefill_rows([a]), page_table=row, kv_cache=kv, prompt_lens=[50], sampling_params=[SamplingParams()]
        )
        seq = a + [int(f[0])]
        sp = [SamplingParams(temperature=0.9, top_k=16, top_p=0.95, seed=11)]
        for i in range(5):
            o = g.decode_forward(
                torch.tensor([-1]),
                torch.tensor([len(seq) - 1]),
                page_table=None,
                kv_cache=kv,
                slots=[0],
                sampling_params=sp,
                reset_sampling_state=reset and i == 0,
            )
            seq.append(int(o[0]))
        return seq[50:]

    with_reset = run(True, True), run(False, True)
    without = run(True, False), run(False, False)
    assert with_reset[0] == with_reset[1]
    assert without[0] != without[1]  # control: without the re-seed the continuation depends on the slot's history
    record("generator_reset_sampling_state_reseeds", tokens=with_reset[0], control_differs=True)


def test_padded_vocab_masking():
    m, g = make("dense_exact", max_batch=2, logits_width=155648)
    ids = prompt(37, seed=18)
    assert g.generate(ids, 6) == m.reference_greedy(ids, 6)
    assert g.prefill_logits(ids).shape[-1] == M.VOCAB
    record("generator_padded_vocab_masking", logits_width=155648, vocab=M.VOCAB)


def test_pad_leak_negative_control():
    """A decoder that lets padded rows update KDA state is caught by the reference comparison."""
    m, g = make("dense_exact", max_batch=2, bug="pad_leak")
    ids = prompt(33, seed=19)  # 33 -> 64 physical rows: 31 padded rows in layer 0
    assert g.generate(ids, 6) != m.reference_greedy(ids, 6)
    record("generator_pad_leak_negative_control", detected=True)


def test_inactive_row_mask_negative_control():
    """Backend requirement (contract 5, Stage 2 review C-9): rows with cur_pos == -1 inside a replayed width reach the
    sampler masked. Positive: user 0 finishes, user 1 keeps decoding in the width-2 trace -> finite inactive row 0.
    Negative control: the same schedule without the backend's layer-boundary mask feeds the decoders' undefined (NaN)
    row to the sampler, which refuses it."""
    sched = ([prompt(40, seed=31), prompt(52, seed=32)], [2, 6])
    m, g = make("dense_exact", max_batch=2)
    m.decode_log = []
    outs = g.generate_batch(*sched)
    assert outs == [m.reference_greedy(p, n) for p, n in zip(*sched)]
    rows = [bool(torch.isfinite(lg[0]).all()) for pos, _t, lg in m.decode_log if int(pos[0]) < 0]
    assert rows and all(rows)
    m2, g2 = make("dense_exact", max_batch=2)
    m2.mask_inactive = False
    with expect_error(M.MockDeviceError, "inactive"):
        g2.generate_batch(*sched)
    record("generator_inactive_row_mask", inactive_steps_finite=len(rows), negative_control_detected=True)


def test_readiness_contract_surface():
    from readiness_check.contract import Generator as ReadinessGenerator

    m = M.MockModel(KINDS, max_batch=2)
    g = G.build_generator(
        "models/autoports/zai_org_glm_5_3_flash", None, backend=m, max_batch=2, max_seq_len=2051, stop_token_ids=()
    )
    assert isinstance(g, ReadinessGenerator)
    sig = inspect.signature(g.generate).parameters["enable_trace"]
    assert sig.kind == inspect.Parameter.KEYWORD_ONLY
    with expect_error(ValueError):
        G.build_generator("x", None)
    with expect_error(RuntimeError, "no decode trace"):
        g.warmup(capture=False)
        b = g._own
        g._decode_step(b, [], enable_trace=True, mode="greedy")
    record("generator_readiness_contract_surface", base=ReadinessGenerator.__module__)


def test_mock_decoder_stage1_contract():
    """The MockDecoder enforces Stage 1's physical-row contract, page-table permutation rule and DSA guard."""
    dev = M.MockDevice()
    kda = M.MockDecoder("sparse_kda", 4, dev)
    mla = M.MockDecoder("sparse_dsa", 3, dev)
    ck = kda.allocate_cache(max_batch=2, max_seq_len=4096)
    cm = mla.allocate_cache(max_batch=2, max_seq_len=2051)
    mla.make_page_table(cm)
    kda.begin_prefill_chunk(ck, user_slot=0, chunk_start=0, n_valid=33)
    with expect_error(M.ContractViolation):
        kda.prefill_chunk(M.DeviceTensor(torch.zeros(1, 1, 96, kda.width, dtype=torch.float64)), cache=ck, user_slot=0)
    with expect_error(NotImplementedError):
        mla.allocate_cache(max_batch=1, max_seq_len=2052)
    with expect_error(ValueError):
        mla.begin_prefill_chunk(cm, user_slot=0, chunk_start=10, n_valid=5)
    with expect_error(ValueError):
        mla.update_page_table(cm, M.DeviceTensor(torch.zeros(1)), torch.zeros(2 * cm.blocks_per_seq))
    x = torch.randn(1, 70, kda.width, dtype=torch.float64)
    full = kda.prefill_forward(x, cache=ck, user_slot=1)
    st = ck.state[1].clone()
    ck2 = kda.allocate_cache(max_batch=2, max_seq_len=4096)
    kda.prefill_forward(x[:, :40], cache=ck2, user_slot=1)
    part = kda.prefill_forward(x[:, 40:], cache=ck2, user_slot=1, start_pos=40)
    assert torch.allclose(full[:, 40:], part, atol=1e-12) and torch.allclose(st, ck2.state[1], atol=1e-12)
    record(
        "mock_decoder_stage1_contract",
        checks=["physical rows", "dsa guard", "mla page start", "permutation", "kda continuation"],
    )
