# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill chunk grid through the real vLLM 0.26 engine (Stage 9 gx2c5 regression, host-only).

2026-10-23 gx2c5 (32 live rows, LMCache on) died in the TT step with `ValueError: row 4: chunk start 649 is not a
multiple of the 64-token latent page`: a 668-token prompt with a 640-token LMCache hit was handed the step's 9-token
leftover budget, its chunk ran [640, 649) and its next chunk started off the latent page. vllm-tt-plugin's
`TTScheduler._align_prefill_chunk_end` returned the unaligned chunk of any request that did not start at 0;
e3e7072 (srv/s9gx2c6) makes a request whose aligned chunk would be empty wait for the next step.

Why the workload needs a page hit: in a TT prefill step the running partial prefill is scheduled first and a fresh
admission waits for the grid, so with fresh prompts only there is at most one partial prefill and no continuation is
ever offered a leftover (the same four prompts without a shared prefix keep every start on the grid with the pre-fix
plugin too). A request admitted at a page hit is: its start is `computed + local hit + external hit` (vLLM prefix
caching here; an LMCache / connector hit on device takes the same scheduler path, the connector itself is not in this
loop). The four prompts (301 / 457 / 668 / 1001 tokens) share a 256-token system prompt, so every request after the
first is admitted at 256 on whatever the step has left; 300 tokens per step leave 44-, 127- and 255-token leftovers.
A 1,664-token KV pool (30 blocks) admits the four prompts but not their decodes: vLLM preempts a request and recomputes
it (from its own cached pages).

The mock is the 06a CPU MockModel behind the real adapter and Glm53Generator (glm53_mock_stack.py); greedy outputs are
compared with its unpaged oracle (MockModel.reference_greedy), i.e. each prompt run alone.

Run (host-8chip, host only; the plugin's src first on PYTHONPATH; ~50 s on 32 cores with OMP_NUM_THREADS=8 -- unset,
torch oversubscribes a taskset slice and the first test takes ~160 s):
  T=models/autoports/zai_org_glm_5_3_flash
  no-tt-devices venv-lmcache/bin/python -m pytest $T/tests/vllm/test_chunk_grid_engine.py \\
      -q -p no:cacheprovider --confcutdir $T
"""

from __future__ import annotations

import inspect

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import DSA_MODE_ENV, prompt_ids
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_vllm_engine_mock import GREEDY, Engine, rung_contract  # noqa: F401

H.engine_env()

from vllm import SamplingParams  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import BLOCK_SIZE, KV_POOL_TOKENS_ENV  # noqa: E402

ALIGN = 128  # tt.prefill_chunk_alignment (the served value; also the model's prefix-cache boundary)


class OffGridChunkError(AssertionError):
    """The model was handed a prefill chunk starting off the grid (the TT step raises on the 64-token latent page)."""


def _external(rid: str) -> str:
    """vLLM's internal request id is '<caller's id>-<8 random chars>' (InputProcessor); the tests' ids have no '-'."""
    return rid.rsplit("-", 1)[0]


def _ints(x, rows):
    if x is None:
        return [0] * rows
    return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


class ChunkSpy:
    """What the model received and why, recorded without changing it:
    rows      -- every prefill row the adapter's prefill_forward got (before Glm53Generator checks it), as
                 (call, request, start, end); the request is the one the scheduler gave exactly that chunk;
    splits    -- every chunk-size decision of vLLM's block-aligned-split hook (TTScheduler's alignment): the request,
                 its start (computed + local hit + external hit), the tokens offered and the tokens granted;
    preempted -- vLLM's preemptions (Scheduler._preempt_request)."""

    def __init__(self, e):
        self.rows, self.splits, self.preempted = [], [], []
        self._chunks: dict[tuple[int, int], list[str]] = {}  # scheduled (start, end) -> request ids
        self._calls = 0
        model = e.runner.model
        sched = e.llm.llm_engine.engine_core.engine_core.scheduler
        schedule, preempt = sched.schedule, sched._preempt_request
        split, prefill = sched._mamba_block_aligned_split, model.prefill_forward
        sig = inspect.signature(prefill)

        def schedule_spy(*a, **kw):
            out = schedule(*a, **kw)
            for rid, n in out.num_scheduled_tokens.items():
                req = sched.requests.get(rid)
                if req is not None and n > 0:  # num_computed_tokens is already past this step's chunk
                    end = int(req.num_computed_tokens)
                    self._chunks.setdefault((end - int(n), end), []).append(_external(rid))
            return out

        def preempt_spy(request, *a, **kw):
            self.preempted.append(
                {"request": _external(request.request_id), "computed": int(request.num_computed_tokens),
                 "num_tokens": int(request.num_tokens), "prompt_len": int(request.num_prompt_tokens)}
            )  # fmt: skip
            return preempt(request, *a, **kw)

        def split_spy(request, num_new_tokens, *a, **kw):
            granted = split(request, num_new_tokens, *a, **kw)
            hits = sum(int(h) for h in list(a) + list(kw.values()))  # local + external computed tokens (waiting loop)
            self.splits.append(
                {"request": _external(request.request_id), "start": int(request.num_computed_tokens) + hits,
                 "offered": int(num_new_tokens), "granted": int(granted), "prompt_len": int(request.num_prompt_tokens),
                 "num_tokens": int(request.num_tokens)}
            )  # fmt: skip
            return granted

        def prefill_spy(*a, **kw):
            b = sig.bind(*a, **kw).arguments
            ends = _ints(b["prompt_lens"], 0)
            starts = _ints(b.get("start_pos"), len(ends))
            for s, end in zip(starts, ends):
                rids = self._chunks.get((s, end))
                self.rows.append((self._calls, rids.pop(0) if rids else "?", s, end))
            self._calls += 1
            return prefill(*a, **kw)

        sched.schedule, sched._preempt_request = schedule_spy, preempt_spy
        sched._mamba_block_aligned_split, model.prefill_forward = split_spy, prefill_spy

    def off_grid(self, grid):
        return [r for r in self.rows if r[2] % grid]

    def chunks_of(self, rid):
        return [(s, end) for _, r, s, end in self.rows if r == rid]


def _run(e, spy, schedule):
    """Engine.run; an engine that died on an off-grid chunk start fails with OffGridChunkError, naming the chunk."""
    try:
        return e.run(schedule)
    except Exception as exc:
        cause = exc
        while cause.__cause__ is not None:
            cause = cause.__cause__
        if not spy.off_grid(BLOCK_SIZE) and "latent page" not in str(cause):
            raise
        raise OffGridChunkError(
            f"engine died: {type(cause).__name__}: {cause}; off-grid chunk starts the model received "
            f"(call, request, start, end): {spy.off_grid(BLOCK_SIZE)}"
        ) from exc


def _check_completed(e, rec, n):
    assert e.runner.device_fault is None, f"the runner latched a device fault: {e.runner.device_fault!r}"
    bad = {rid: (r.get("finish_reason"), len(r.get("tokens", []))) for rid, r in rec.items()
           if r.get("finish_reason") != "length" or len(r.get("tokens", [])) != n}  # fmt: skip
    assert not bad, f"requests that did not complete: {bad}"


def _check_recomputed(spy, rec):
    """vLLM preempted at least one request; each preempted request was prefilled again (a chunk starting below an end
    the model had already seen for it) and completed."""
    assert spy.preempted, "the KV pool was meant to force a preemption"
    for p in spy.preempted:
        ch = spy.chunks_of(p["request"])
        assert any(s < max(end for _, end in ch[:i]) for i, (s, _) in enumerate(ch) if i), ("no recompute", p, ch)
        assert p["request"] in rec and rec[p["request"]].get("finish_reason") == "length", p


# --------------------------------------------------------------------------------------------- gx2c5 regression
SYSTEM = 256  # shared system prompt: the page hit of every request after the first
LENGTHS = (301, 457, 668, 1001)
BUDGET = 300  # max_num_batched_tokens
MAX_TOKENS = 48
POOL_TOKENS = 1664  # GLM53_VLLM_KV_POOL_TOKENS: 26 pages + 1 per user = 30 blocks
MAX_MODEL_LEN = 1088


def test_hit_remainders_and_preemption_keep_every_chunk_on_the_grid(monkeypatch, evidence):
    """4 concurrent odd prompts behind a shared system prompt, 300 tokens per step, alignment 128, a pool that forces a
    preemption: the workload offers page-hit remainders sub-grid leftovers (the gx2c5 shape) and they wait; every
    prefill chunk the model receives starts on the 128-token grid (so on the 64-token latent page); the engine survives
    and every request completes with its unpaged-oracle greedy tokens; vLLM preempted and recomputed a request."""
    monkeypatch.setenv(KV_POOL_TOKENS_ENV, str(POOL_TOKENS))
    e = Engine(max_num_seqs=4, max_num_batched_tokens=BUDGET, max_model_len=MAX_MODEL_LEN, enable_prefix_caching=True,
               tt_config={"prefill_chunk_alignment": ALIGN})  # fmt: skip
    try:
        spy = ChunkSpy(e)
        system = prompt_ids(4099, SYSTEM)
        prompts = {f"p{n}": system + prompt_ids(n, n - SYSTEM) for n in LENGTHS}
        rec = _run(e, spy, [(0, rid, p, SamplingParams(max_tokens=MAX_TOKENS, **GREEDY)) for rid, p in prompts.items()])
        _check_completed(e, rec, MAX_TOKENS)

        # the gx2c5 shape was offered: a request starting past 0 on a page hit, given a leftover that cannot reach the
        # next grid line before its prompt ends -- and it waited for a later step instead of taking the leftover
        subgrid = [s for s in spy.splits if 0 < s["start"] and s["start"] + s["offered"] < s["prompt_len"]
                   and (s["start"] + s["offered"]) // ALIGN * ALIGN <= s["start"]]  # fmt: skip
        assert subgrid, f"no page-hit remainder was offered a sub-grid leftover (scenario lost): {spy.splits}"
        assert all(s["granted"] == 0 for s in subgrid), f"a sub-grid leftover was taken: {subgrid}"
        assert [rid for rid in prompts if spy.chunks_of(rid)[0][0] == SYSTEM] == list(prompts)[1:], spy.rows

        # every chunk the model received starts on the grid; so does every chunk the backend ran (after the prefix plan)
        assert spy.rows and all(r != "?" for _, r, _, _ in spy.rows), f"rows the scheduler never gave out: {spy.rows}"
        assert not spy.off_grid(ALIGN), f"off-grid chunk starts (call, request, start, end): {spy.off_grid(ALIGN)}"
        backend = [ev["start"] for ev in e.events if ev["event"] == "prefill_chunk"]
        assert backend and all(s % ALIGN == 0 for s in backend), backend

        _check_recomputed(spy, rec)
        assert e.check_greedy(rec) == {}
        evidence(
            plugin=__import__("vllm_tt_plugin").__file__,
            prompt_lengths=list(LENGTHS), shared_prefix=SYSTEM, max_num_batched_tokens=BUDGET, alignment=ALIGN,
            kv_pool_tokens=POOL_TOKENS, kv_num_blocks=e.runner.model.cache_spec.num_blocks,
            prefill_rows=spy.rows, subgrid_leftovers_deferred=subgrid, preemptions=spy.preempted,
            outputs_match_oracle=True,
        )  # fmt: skip
    finally:
        e.close()


# ------------------------------------------------------------------------------ resumed request
def test_resumed_request_recompute_stays_on_the_grid(monkeypatch, evidence):
    """A preempted request re-prefills its prompt AND its generated tokens. Two odd prompts (129 / 131), 150 tokens per
    step (off the 64-token grid), alignment 128, a 320-token pool (7 blocks): the decodes outgrow the pool, vLLM
    preempts one request, and its recompute must still hand the model only on-grid chunk starts and reproduce the
    oracle's tokens. With e3e7072 the recompute runs [0, 150) (150 >= its 129-token prompt, < its 191 tokens) and the
    TT step raised on start 150; the plugin's glm53/glm53-s9-grid2 aligns against num_tokens."""
    monkeypatch.setenv(KV_POOL_TOKENS_ENV, "320")
    e = Engine(max_num_seqs=2, max_num_batched_tokens=150, max_model_len=256,
               tt_config={"prefill_chunk_alignment": ALIGN})  # fmt: skip
    try:
        spy = ChunkSpy(e)
        prompts = {"a129": prompt_ids(129, 129), "b131": prompt_ids(131, 131)}
        rec = _run(e, spy, [(0, rid, p, SamplingParams(max_tokens=80, **GREEDY)) for rid, p in prompts.items()])
        evidence(prefill_rows=spy.rows, preemptions=spy.preempted)
        _check_completed(e, rec, 80)
        if spy.off_grid(ALIGN):
            raise OffGridChunkError(f"off-grid chunk starts (call, request, start, end): {spy.off_grid(ALIGN)}")
        _check_recomputed(spy, rec)
        assert e.check_greedy(rec) == {}
    finally:
        e.close()


# ------------------------------------------------------------------------- the 64K tail chunk (n29h5's death, 03:28Z)
# n29h5 died on the 64K needle: TPPrefill.plan raised 'chunk [59392, 67584) past the cache (66048 tokens)'. The 8,192
# bucket of the prompt's last chunk (4,557 tokens left at 59,392) ran past the cache / the captured 65,536 window.
# 16's 710d565d6e3 clamps it in TPPrefill.slice_plan. Here the real engine schedules the served shape (2,048 tokens per
# step, alignment 128, the generator chunking at the largest bucket, 8,192), and every chunk the generator hands the
# backend goes through the TP prefill's own slice_plan / plan on the served geometry (TPPrefill stand-in: buckets
# 512 / 2,048 / 8,192, sparse window 65,536, max_seq = the served cache row). plan() raises exactly as on the device.
# The CPU mock computes each chunk in <= 2,048-row pieces (its own limit; the mock math is chunking-invariant). No
# oracle run: the unchunked 64K reference alone would need ~32 GB.
LONG = 64000
SERVED_BUCKETS = (512, 2048, 8192)
SERVED_WINDOW = 65536
SERVED_MML = 65536


def _served_planner(e, monkeypatch):
    """the TP prefill's bucket chooser (tt/prefill_tp.py) behind the mock backend, the way StackBackend exposes it;
    returns the list of (chunk_start, n_valid, (bucket, window)) the backend replayed"""
    from types import MethodType

    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.test_prefill_slice_boundary import tpp
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
    from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as S

    be = e.gen.backend
    pf = tpp(SERVED_BUCKETS, dsa=True, sparse_ts=(SERVED_WINDOW,),
             max_seq=G.blocks_per_seq(SERVED_MML) * G.BLOCK_SIZE)  # fmt: skip
    assert pf.max_seq == 66048  # the served cache row (n29h5's error message)
    be.tpp = pf
    monkeypatch.setattr(be, "prefill_slice_plan", MethodType(S.StackBackend.prefill_slice_plan, be), raising=False)
    monkeypatch.setattr(be, "prefill_slice_rows", MethodType(S.StackBackend.prefill_slice_rows, be), raising=False)
    monkeypatch.setattr(be, "prefill_chunk_tokens", max(SERVED_BUCKETS), raising=False)  # StackBackend's: 8,192
    replayed, orig = [], be.prefill_chunk
    mock_max = G.PREFILL_CHUNK

    def chunk(tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None, state_src=None,
              state_snap=None):  # fmt: skip
        replayed.append((int(chunk_start), int(n_valid), pf.plan(int(chunk_start), int(n_valid))))
        flat = tokens.reshape(-1)
        out, off = None, 0
        while off < n_valid:
            n = min(mock_max, n_valid - off)
            last = off + n == n_valid
            out = orig(flat[off: off + G.round_up(n, G.MODEL_PREFILL_ALIGN)], kv_cache=kv_cache, slot=slot,
                       chunk_start=chunk_start + off, n_valid=n, logits_rows=logits_rows if last else "none",
                       sample=sample if last else None, io=io, state_src=state_src if off == 0 else None,
                       state_snap=state_snap if last else None)  # fmt: skip
            off += n
        return out

    monkeypatch.setattr(be, "prefill_chunk", chunk, raising=False)
    return replayed, pf


def test_64k_prompt_and_tail_hit_fit_the_served_cache(monkeypatch, evidence, rung_contract):
    """the 64,000-token prompt (holmix's long row; the needle's 63,949 has the same tail) and a second request that
    shares its first 61,440 tokens (a page hit; its 2,560-token continuation is the 8,192-bucket tail case at 61,440)
    complete through the served chunk planner: every replayed (bucket, window) fits the cache row and the captured
    window, every chunk start is on the 128 grid, the clamp fired"""
    monkeypatch.setenv(DSA_MODE_ENV, "unbounded")  # the mock MLA layer serves the whole 64K context
    monkeypatch.setenv(KV_POOL_TOKENS_ENV, str(2 * SERVED_MML + 4096))
    rung_contract(SERVED_MML)  # the served contract: 65,536
    e = Engine(max_num_seqs=2, max_num_batched_tokens=2048, max_model_len=SERVED_MML, enable_prefix_caching=True,
               tt_config={"prefill_chunk_alignment": ALIGN, "prefill_chunk_run": 4})  # fmt: skip  (the served tt config)
    try:
        replayed, pf = _served_planner(e, monkeypatch)
        spy = ChunkSpy(e)
        a = prompt_ids(6400, LONG)
        b = a[:61440] + prompt_ids(6401, LONG - 61440)
        rec = _run(e, spy, [(0, "a64k", a, SamplingParams(max_tokens=4, **GREEDY)),
                            (40, "b64k", b, SamplingParams(max_tokens=4, **GREEDY))])  # fmt: skip
        _check_completed(e, rec, 4)
        assert replayed and all(s % ALIGN == 0 for s, _, _ in replayed), replayed
        for s, n, (c, t) in replayed:
            assert s + c <= pf.max_seq and (t == 0 or s + c <= t <= SERVED_WINDOW), (s, n, c, t)
        b_starts = [s for s, _ in spy.chunks_of("b64k")]
        assert b_starts and b_starts[0] >= 59392, ("b64k was meant to start at a deep page hit", b_starts)
        clamped = int(e.gen.counters.get("prefill_max_seq_clamped", 0))
        assert clamped > 0, ("the tail chunks fit without the clamp: scenario lost", replayed[-6:])
        evidence(replayed=replayed, b64k_first_start=b_starts[0], clamped=clamped, max_seq=pf.max_seq,
                 prefill_rows=spy.rows[-12:])  # fmt: skip
    finally:
        e.close()


def test_64k_tail_without_the_clamp_dies_like_n29h5(monkeypatch, rung_contract):
    """the control: the same engine and planner with TPPrefill.clamp_lengths reverted to the pre-710d565d6e3 behaviour
    (every chunk as one piece) dies in plan() on the 64K prompt's tail chunk with n29h5's error"""
    monkeypatch.setenv(DSA_MODE_ENV, "unbounded")
    monkeypatch.setenv(KV_POOL_TOKENS_ENV, str(SERVED_MML + 4096))
    rung_contract(SERVED_MML)
    e = Engine(max_num_seqs=2, max_num_batched_tokens=2048, max_model_len=SERVED_MML,
               tt_config={"prefill_chunk_alignment": ALIGN, "prefill_chunk_run": 4})  # fmt: skip
    try:
        _, pf = _served_planner(e, monkeypatch)
        monkeypatch.setattr(pf, "clamp_lengths", lambda s, n: [int(n)], raising=False)
        with pytest.raises(Exception) as err:
            e.run([(0, "a64k", prompt_ids(6400, LONG), SamplingParams(max_tokens=4, **GREEDY))])
        chain, exc = [], err.value
        while exc is not None:
            chain.append(str(exc))
            exc = exc.__cause__ or exc.__context__
        assert any("past the cache (66048 tokens)" in m for m in chain), chain
    finally:
        e.close()
