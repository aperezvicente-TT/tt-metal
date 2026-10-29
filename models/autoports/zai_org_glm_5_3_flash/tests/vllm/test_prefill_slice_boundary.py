# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16's slice-boundary mitigation (program 2026-10-22T23:13:59Z), host-only. In a sequence-parallel (SP) TP8
prefill bucket chip r holds rows [r R, (r + 1) R), R = bucket / 8; a chunk whose last valid row is the first row of a
chip's slice ((L - 1) mod R == 0) got a wrong first token (golden[:97] in the 256 bucket). The rule
(tt/generator.py slice_plan; TPPrefill.slice_choice / plan / slice_plan; Glm53Generator._slice_safe and the packed /
batched regrouping) keeps every chosen geometry off the boundary. Checks, over n = 1..2,048 and multi-chunk prompts up
to 8,192 with continuation starts: no chosen (bucket, chunk length L >= 2) has (L - 1) mod (bucket / 8) == 0 for an SP
bucket, the chunks cover the prompt exactly in order, continuation starts stay 128-aligned, the counters count every
firing per branch; the mock model's outputs equal the rule-off run (the split is only a different chunking); packed and
batched groups never hold a boundary segment. TPPrefill runs as a host stand-in (TPPrefill.__new__ + its geometry
attributes: the methods under test are plain Python)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    PF_BATCH_ENV,
    PF_PACKED_ENV,
    RecordingMockModel,
)
from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as P
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as S
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY

NDEV = 8
FRONT = 128
BRANCHES = ("larger", "split", "regrouped", "unavoidable")


def tpp(buckets, *, dsa=False, sparse_ts=(), max_seq=65536, rule=True, ndev=NDEV):
    """TPPrefill's bucket chooser without a mesh: the attributes its plan / slice methods read"""
    o = P.TPPrefill.__new__(P.TPPrefill)
    o.buckets, o.ndev, o.dsa = tuple(sorted(int(c) for c in buckets)), ndev, dsa
    o.sparse_ts, o.max_seq, o.slice_rule = tuple(sparse_ts), max_seq, rule
    o.layers = [SimpleNamespace(kind=P.SPARSE_DSA)]
    o.mla_form, o.fresh_min = "absorbed", 1024
    return o


def boundary(pf, start: int, n: int) -> bool:
    """the defect geometry of a chunk as the TP prefill would replay it (the bucket plan() picks)"""
    c, _ = pf.plan(start, n)
    r = pf.slice_rows(c)
    return n >= 2 and r > 0 and (n - 1) % r == 0


def natural_boundary(pf, start: int, n: int) -> bool:
    c = pf.bucket_for(n, sparse=pf._sparse(start, n))
    r = pf.slice_rows(c)
    return n >= 2 and r > 0 and (n - 1) % r == 0


class SliceModel(RecordingMockModel):
    """the mock stack's model + StackBackend's slice-boundary hooks over a TPPrefill stand-in with TP8 geometry; every
    backend chunk records (start, n_valid, the bucket TPPrefill.plan replays it in, head)"""

    prefill_slice_plan = S.StackBackend.prefill_slice_plan
    prefill_slice_rows = S.StackBackend.prefill_slice_rows

    def __init__(self, *a, pf=None, **k):
        super().__init__(*a, **k)
        self.tpp = pf
        self.calls: list[tuple] = []

    @property
    def prefill_chunk_tokens(self) -> int:  # StackBackend's: the largest bucket
        return max(self.tpp.buckets)

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None,
                      state_src=None, state_snap=None):  # fmt: skip
        c = self.tpp.plan(chunk_start, n_valid)[0]
        self.calls.append((int(chunk_start), int(n_valid), c, logits_rows != "none"))
        return super().prefill_chunk(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start, n_valid=n_valid,
                                     logits_rows=logits_rows, sample=sample, io=io, state_src=state_src,
                                     state_snap=state_snap)  # fmt: skip


def gen_of(pf, *, kinds=("dense_kda", "sparse_kda"), max_batch=8, max_seq_len=8192 + 4096, dsa_mode="dense_exact"):
    m = SliceModel(kinds, max_batch=max_batch, seed=1, dsa_mode=dsa_mode, pf=pf)
    return m, G.Glm53Generator(m, max_batch=max_batch, max_seq_len=max_seq_len, stop_token_ids=())


def check_pieces(pf, plans, n: int, start: int, align: int = FRONT):
    """the chunks cover [start, start + n) in order, start on aligned positions, never on the defect geometry"""
    pos, off = start, 0
    for pl in plans:
        assert pl.start == pos and pl.offset == off and 1 <= pl.n_valid <= max(pf.buckets), (n, start, plans)
        assert pl.start % align == 0 or pl.start == start, (n, start, plans)
        assert pl.physical == G.round_up(pl.n_valid, G.MODEL_PREFILL_ALIGN)
        assert not boundary(pf, pl.start, pl.n_valid), (n, start, pl, pf.plan(pl.start, pl.n_valid))
        pos, off = pos + pl.n_valid, off + pl.n_valid
    assert off == n


# --------------------------------------------------------------------------------------------- the bucket chooser
# the rule's branches over n = 1..2,048 (one chunk at position 0); 512,2048 = the program's rule verbatim
EXPECTED = {
    "512,2048": dict(larger=[65, 129, 193, 321, 385, 449], split=[257, 513, 769, 1025, 1281, 1537, 1793]),
    "256,512,2048": dict(larger=[33, 65, 97, 129, 161, 193, 225, 321, 385, 449],
                         split=[257, 513, 769, 1025, 1281, 1537, 1793]),  # fmt: skip
    "256,512,2048,8192": dict(larger=[33, 65, 97, 129, 161, 193, 225, 321, 385, 449],
                              split=[257, 513, 769, 1025, 1281, 1537, 1793]),  # fmt: skip
    # 128 is replicated at TP8: 193 = 128 + 65 both in the 128 bucket (256 rows) beats one 2,048 replay
    "128,512,2048": dict(larger=[129, 321, 385, 449], split=[193, 257, 513, 769, 1025, 1281, 1537, 1793]),
}


@pytest.mark.parametrize("buckets", sorted(EXPECTED))
def test_tpprefill_rule_over_1_to_2048(buckets, evidence):
    pf = tpp(buckets.split(","))
    off = tpp(buckets.split(","), rule=False)
    got = {}
    for n in range(1, 2049):
        lens, branch = pf.slice_plan(0, n, FRONT)
        got.setdefault(branch, []).append(n)
        assert sum(lens) == n and (branch == "split") == (len(lens) == 2), (n, lens, branch)
        pieces, s = [], 0
        for ln in lens:
            pieces.append(G.ChunkPlan(start=s, offset=s, n_valid=ln, physical=G.round_up(ln, 64)))
            s += ln
        check_pieces(pf, pieces, n, 0)
        assert (branch in ("larger", "split")) == natural_boundary(pf, 0, n), (n, branch)
        if branch == "larger":  # one chunk, moved from its natural bucket
            assert pf.plan(0, n)[0] > off.plan(0, n)[0], n
        if branch == "split":  # the rest runs in the 2,048 bucket (or a replicated one)
            rest = pf.plan(FRONT, lens[1])[0]
            assert lens[0] == FRONT and (rest == 2048 or pf.slice_rows(rest) == 0), (n, lens, rest)
        if not branch or branch == "row0":
            assert pf.plan(0, n) == off.plan(0, n), n  # the rule leaves every other chunk alone
        assert off.slice_plan(0, n, FRONT) == ([n], "")  # GLM53_PF_SLICE_RULE=0: the control
    assert got.get("larger", []) == EXPECTED[buckets]["larger"], got.get("larger")
    assert got.get("split", []) == EXPECTED[buckets]["split"], got.get("split")
    assert "unavoidable" not in got, got["unavoidable"]
    assert got.get("row0", []) == ([1] if pf.slice_rows(pf.bucket_for(1)) else [])
    evidence(buckets=buckets, larger=got.get("larger", []), split=got.get("split", []), row0=got.get("row0", []),
             untouched=len(got.get("", [])))  # fmt: skip


def test_tpprefill_sparse_regime_needs_a_captured_window():
    """past position 2,051 (DSA top-512) a larger bucket must be sequence-parallel with a captured key window that holds
    chunk_start + bucket; without one the chunk is split (or, with neither, runs unavoidably in its natural bucket)"""
    pf = tpp((256, 512, 2048), dsa=True, sparse_ts=(32768,), max_seq=33280)
    # [30720, +97): 256 -> 512 (window 32,768 >= 31,232)
    assert pf.plan(30720, 97) == (512, 32768) and pf.slice_plan(30720, 97, FRONT) == ([97], "larger")
    # [30720, +257): 512 / 2,048 boundaries -> 128 + 129, the rest's 2,048 bucket needs 30,848 + 2,048 > 32,768
    assert pf.slice_plan(30720, 257, FRONT) == ([257], "unavoidable")
    pf_wide = tpp((256, 512, 2048), dsa=True, sparse_ts=(32768, 65536), max_seq=65536)
    assert pf_wide.slice_plan(30720, 257, FRONT) == ([128, 129], "split")
    assert pf_wide.plan(30848, 129) == (2048, 65536)
    # a replicated bucket never runs a sparse chunk; the dense regime keeps it (128 at TP8 is replicated)
    pf_rep = tpp((128, 512, 2048), dsa=True, sparse_ts=(32768,))
    assert pf_rep.slice_rows(128) == 0 and pf_rep.plan(0, 97) == (128, 0)
    assert pf_rep.plan(4096, 97) == (512, 32768)  # sparse: the SP 512 bucket (96 mod 64 != 0)
    assert pf_rep.plan(4096, 65)[0] == 2048  # 512 is a boundary at 65 -> the next SP bucket


# ------------------------------------------------------------------------------------- the generator's chunk plans
@pytest.mark.parametrize("buckets", ["512,2048", "256,512,2048", "256,512,2048,8192"])
def test_generator_chunk_plans_avoid_the_boundary(buckets, evidence):
    """Glm53Generator._slice_safe over plan_prefill: n = 1..2,048 fresh, and multi-chunk prompts up to 8,192 tokens
    from continuation starts (128-aligned); the counters equal the firings"""
    pf = tpp(buckets.split(","))
    m, g = gen_of(pf)
    chunk, firings = m.prefill_chunk_tokens, dict.fromkeys(BRANCHES + ("row0",), 0)
    cases = [(0, n) for n in range(1, 2049)]
    starts = (0, 128, 2048, 4224)
    cases += [(s, n) for s in starts for n in range(2049, 8193 - s, 1 if s == 0 else 7)]
    cases += [(s, n) for s in starts[1:] for n in range(1, 2049, 3)]
    for start, n in cases:
        raw = G.plan_prefill(n, start, chunk, m.prefill_align)
        for pl in raw:
            _, br = pf.slice_plan(pl.start, pl.n_valid, FRONT)
            if br:
                firings[br] += 1
        plans = g._slice_safe(raw)
        check_pieces(pf, plans, n, start)
    for br in BRANCHES:
        assert g.counters[f"prefill_slice_boundary_{br}"] == firings[br], (br, g.counters, firings)
    assert g.counters["prefill_slice_row0_chunks"] == firings["row0"]
    assert g.counters["prefill_slice_boundary_avoided"] == firings["larger"] + firings["split"] > 0
    assert firings["unavoidable"] == 0
    evidence(buckets=buckets, cases=len(cases), firings=firings)


def test_split_front_follows_the_continuation_alignment():
    """the front is SLICE_SPLIT_FRONT rounded up to continuation_align (q chunk 256: a 256-token front)"""
    pf = tpp((256, 512, 2048, 8192))
    m, g = gen_of(pf)
    m.continuation_align = 256
    plans = g._slice_safe(G.plan_prefill(2049, 0, 8192, 64))  # 8,192 boundary (2,048 mod 1,024) -> 256 + 1,793
    assert [(p.start, p.n_valid) for p in plans] == [(0, 256), (256, 1793)], plans
    check_pieces(pf, plans, 2049, 0, align=256)


# ---------------------------------------------------------------------------- end to end on the mock model (per user)
def _serve(m, g, lens, *, start=0, prefix=None):
    """fresh device-sampled prefill of rows `lens` (one call), slot i = row i; returns the first tokens"""
    kv = g.allocate_kv_cache(m.max_batch, 8192 + 4096)
    g.warmup(kv)
    bps = G.blocks_per_seq(8192 + 4096)
    pt = torch.arange(m.max_batch * bps).reshape(m.max_batch, bps)
    width = max(lens)
    toks = torch.zeros(len(lens), width, dtype=torch.int64)
    gen = torch.Generator().manual_seed(7)
    for i, n in enumerate(lens):
        toks[i, :n] = torch.randint(0, 1000, (n,), generator=gen)
    out = g.prefill_forward(toks, page_table=pt[: len(lens)], kv_cache=kv, prompt_lens=list(lens),
                            sampling_params=[GREEDY] * len(lens))  # fmt: skip
    return [int(v) for v in out], kv


@pytest.mark.parametrize("n", [97, 98, 129, 193, 257, 769, 1793, 2049, 2048 + 257, 4096 + 1025])
def test_mock_outputs_equal_the_rule_off_run(n, evidence):
    """the rule only re-chunks / re-buckets: the mock (geometry-free) gives the rule-off first token and KDA state; the
    backend sees no boundary geometry and one head (the last chunk)"""
    on_m, on_g = gen_of(tpp((256, 512, 2048)), max_batch=2)
    off_m, off_g = gen_of(tpp((256, 512, 2048), rule=False), max_batch=2)
    first_on, kv_on = _serve(on_m, on_g, [n])
    first_off, kv_off = _serve(off_m, off_g, [n])
    assert first_on == first_off, (n, first_on, first_off)
    for a, b in zip(kv_on.caches, kv_off.caches):
        if a.state is not None:
            assert torch.allclose(a.state[0], b.state[0], atol=1e-5, rtol=1e-4), n
    plans = [G.ChunkPlan(start=s, offset=s, n_valid=v, physical=G.round_up(v, 64)) for s, v, _, _ in on_m.calls]
    check_pieces(on_m.tpp, plans, n, 0)
    assert [h for *_, h in on_m.calls] == [False] * (len(on_m.calls) - 1) + [True]
    fired = sum(on_g.counters[f"prefill_slice_boundary_{b}"] for b in ("larger", "split"))
    assert fired == sum(natural_boundary(on_m.tpp, p.start, p.n_valid) for p in G.plan_prefill(n, 0, 2048, 64))
    assert off_g.counters["prefill_slice_boundary_avoided"] == 0
    assert [c[1] for c in off_m.calls] == [p.n_valid for p in G.plan_prefill(n, 0, 2048, 64)]
    evidence(n=n, chunks=on_m.calls, rule_off_chunks=off_m.calls, first=first_on)


# ------------------------------------------------------------------------------------------ packed / batched groups
def test_packed_admission_never_packs_a_boundary_segment(monkeypatch, evidence):
    """_admit_packed (T 2,048 -> R 256): a 257-token prompt first in the group would end on chip 1's first row: it
    leaves the group (per-user path: 128 + 129); a segment pushed onto a boundary by its predecessors leaves too"""
    monkeypatch.setenv(PF_PACKED_ENV, "2048,4")
    monkeypatch.delenv(PF_BATCH_ENV, raising=False)
    m, g = gen_of(tpp((256, 512, 2048)))
    # rows: 257 @0 (last 256: boundary) | 100 @0 after the drop | 156 @128 (last 283) | 33 @288 (last 320)
    lens = [257, 100, 156, 33]
    first, _ = _serve(m, g, lens)
    packed = [ev for ev in m.events if ev["event"] == "prefill_packed"]
    assert len(packed) == 1 and packed[0]["n"] == [100, 156, 33], packed
    assert [(s, v) for s, v, *_ in m.calls] == [(0, 128), (128, 129)], m.calls  # the dropped prompt, split
    assert g.counters["prefill_slice_boundary_regrouped"] == 1 and g.counters["prefill_slice_boundary_split"] == 1
    assert g.counters["prefill_slice_boundary_avoided"] == 2
    # a segment that only becomes a boundary behind its predecessors: 224 @0 (rows 0..223) then 33 @224 -> last 256
    m2, g2 = gen_of(tpp((256, 512, 2048)))
    _serve(m2, g2, [224, 33, 40])
    packed2 = [ev for ev in m2.events if ev["event"] == "prefill_packed"]
    assert packed2[0]["n"] == [224, 40] and g2.counters["prefill_slice_boundary_regrouped"] == 1, packed2
    assert [(s, v) for s, v, *_ in m2.calls] == [(0, 33)]  # per user: 33 in the 512 bucket (32 mod 64 != 0)
    # every packed segment's last row is off the boundary of its T
    for ev in packed + packed2:
        o = 0
        for n in ev["n"]:
            assert not G.slice_boundary(o + n - 1, ev["t"] // NDEV), ev
            o += G.round_up(n, 32)
    # the same rows with the rule off: one group of all four (the control)
    m3, g3 = gen_of(tpp((256, 512, 2048), rule=False))
    first3, _ = _serve(m3, g3, lens)
    assert [ev["n"] for ev in m3.events if ev["event"] == "prefill_packed"] == [lens]
    assert first3 == first
    evidence(packed=[ev["n"] for ev in packed + packed2], per_user=m.calls + m2.calls, counters=dict(g.counters))


def test_batched_admission_never_batches_a_boundary_segment(monkeypatch, evidence):
    """_admit_batch (nb 4 x s 512 = 2,048 rows -> R 256): a 257-token prompt's last row j 512 + 256 is a slice start
    at any position j: it takes the per-user path, the next four prompts form the group"""
    monkeypatch.setenv(PF_BATCH_ENV, "512,4")
    monkeypatch.delenv(PF_PACKED_ENV, raising=False)
    m, g = gen_of(tpp((256, 512, 2048)))
    lens = [10, 257, 20, 30, 40]
    first, _ = _serve(m, g, lens)
    batches = [ev for ev in m.events if ev["event"] == "prefill_batch"]
    assert [ev["n"] for ev in batches] == [[10, 20, 30, 40]], batches
    assert [(s, v) for s, v, *_ in m.calls] == [(0, 128), (128, 129)], m.calls
    assert g.counters["prefill_slice_boundary_regrouped"] == 1 and g.counters["prefill_slice_boundary_split"] == 1
    m3, g3 = gen_of(tpp((256, 512, 2048), rule=False))
    assert _serve(m3, g3, lens)[0] == first
    evidence(batches=[ev["n"] for ev in batches], per_user=m.calls)


def test_stack_backend_hooks_without_the_tp_prefill():
    """StackBackend before prepare_prefill (no TP prefill) or with the rule off: the chunk unchanged, no geometry"""
    be = SimpleNamespace(tpp=None)
    assert S.StackBackend.prefill_slice_plan(be, 0, 257, FRONT) == ([257], "")
    assert S.StackBackend.prefill_slice_rows(be, 2048) == 0
    be.tpp = tpp((256, 512, 2048))
    assert S.StackBackend.prefill_slice_plan(be, 0, 257, FRONT) == ([128, 129], "split")
    assert S.StackBackend.prefill_slice_rows(be, 2048) == 256
    be.tpp = tpp((256, 512, 2048), rule=False)
    assert S.StackBackend.prefill_slice_rows(be, 2048) == 0


# ---- the max_seq clamp (program 2026-10-23 03:30:50Z: n29h5's 64K prompt -- plan() raised 'chunk [59392, 67584)
# past the cache (66048 tokens)': the 8,192-token chunk grid gave the prompt's last chunk the 8,192 bucket)
CLAMP_MAX_SEQ = 66048  # n29h5: max_model_len 65,536 + 512


@pytest.mark.parametrize("sparse_ts", [(65536,), ()])
@pytest.mark.parametrize("buckets", [(128, 512, 2048, 8192), (512, 2048, 8192), (256, 512, 2048, 8192)])
def test_max_seq_clamp_last_chunk(buckets, sparse_ts):
    """every chunk of the 8,192 grid that ends inside the window (start 59,392 + n, n = k x 128 and odd lengths, up to
    the 65,536 cap) is planned as pieces whose replays all fit: plan() never raises, the pieces cover the chunk in
    order, every next start stays 128-aligned, and the chunks that already fit are not touched"""
    pf = tpp(buckets, dsa=True, sparse_ts=sparse_ts, max_seq=CLAMP_MAX_SEQ)
    pf.start_align = 128
    start, cap = 59392, 65536
    ns = sorted({k * 128 for k in range(1, (cap - start) // 128 + 1)} | {k * 128 - 37 for k in range(1, (cap - start) // 128 + 1)})
    clamped = 0
    for n in ns:
        assert start + n <= cap
        lens, branch = pf.slice_plan(start, n, FRONT)
        assert sum(lens) == n and all(x >= 1 for x in lens), (n, lens)
        s = start
        for x in lens:
            assert s % 128 == 0, (n, lens)
            c, t = pf.plan(s, x)  # no ValueError: every piece has a bucket and a key window inside the cache
            assert s + c <= max(sparse_ts or (CLAMP_MAX_SEQ,)) or (not sparse_ts and -(-(s + c) // 2048) * 2048 <= CLAMP_MAX_SEQ)
            s += x
        nat = pf.bucket_for(n, sparse=True)
        if start + nat > 65536:
            assert branch.startswith("clamp") and len(lens) > 1, (n, lens, branch)
            clamped += 1
        else:
            assert not branch.startswith("clamp"), (n, lens, branch)
    assert clamped > 0
    # the reported case: 4,608 tokens left at 59,392 under the 65,536 window -> 2,048 + 2,048 + 512
    if sparse_ts and 512 in buckets:
        assert pf.slice_plan(59392, 4608, FRONT)[0] == [2048, 2048, 512]


def test_max_seq_clamp_leaves_fitting_chunks_alone():
    """chunks whose replay fits keep their plan (the clamp only fires where plan() would raise)"""
    pf = tpp((512, 2048, 8192), dsa=True, sparse_ts=(65536,), max_seq=CLAMP_MAX_SEQ)
    pf.start_align = 128
    for start in range(0, 57345, 8192):
        assert pf.clamp_lengths(start, 8192) == [8192]
    assert pf.clamp_lengths(0, 100) == [100]


def test_generator_counts_the_clamp():
    model = SliceModel.__new__(SliceModel)
    counters = {"prefill_slice_boundary_avoided": 0}
    gen = SimpleNamespace(counters=counters)
    G.Glm53Generator._count_slice(gen, "clamp+split")
    G.Glm53Generator._count_slice(gen, "clamp")
    assert counters["prefill_max_seq_clamped"] == 2 and counters["prefill_slice_boundary_split"] == 1
    assert counters["prefill_slice_boundary_avoided"] == 1
    del model
