# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""segtrace on the served bucket set (program ORDER 2026-10-27T15:14:02Z; 9's k4d3 re-mock at GLM53_PF_SEGMENTS=3 died
in warmup: 'segment 0 of (512, 0) ends on a non-persistent activation (cuts [0, 13, 29, 45])').

The served k4d construction plans PMK programs for the 2,048 bucket only; at 512 every PMK hook falls back to the op-graph
layer (a fresh tensor). A segment cut must lie only after a layer whose PMK program RUNS at the bucket, and buckets below
GLM53_PF_SEGMENT_MIN_BUCKET (default 2048 = the plugin's PREFILL_SEGMENT_MIN_TOKENS 513 rule) are not segmented."""

import types
from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT

N = 45  # the served stack: L0-L2 dense_kda, L % 4 == 3 sparse_dsa (11), the rest sparse_kda
SERVED_BUCKETS = (512, 2048)
SPARSE_T = (0, 16384, 32768, 65536, 131072, 262144, 524288, 1050624)


def _kind(i):
    if i % 4 == 3:
        return PT.SPARSE_DSA
    return PT.DENSE_KDA if i < 3 else PT.SPARSE_KDA


def _hook_stack(progs):
    """a PmkStack-like owner with the given planned (c, L) programs; its real _hook builds the served hooks"""
    return SimpleNamespace(progs=progs, counters={"hook_calls": 0, "copies": 0}, res={})


def _tpp(segments=3, min_bucket=2048, pmk_buckets=(2048,), hook_layers=None):
    layers = [SimpleNamespace(L=i, kind=_kind(i)) for i in range(N)]
    hook_layers = [i for i in range(N)] if hook_layers is None else hook_layers
    owner = _hook_stack({(c, L): object() for c in pmk_buckets for L in hook_layers})
    fake = SimpleNamespace(
        segments=segments,
        segment_min_bucket=min_bucket,
        layers=layers,
        pmk_layers={L: PS.PmkStack._hook(owner, L) for L in hook_layers},
        SEG_COST_MS=PT.TPPrefill.SEG_COST_MS,
        mode=lambda c: "sp",
    )
    fake._cut_ok = types.MethodType(PT.TPPrefill._cut_ok, fake)
    return fake


def _plan(tpp, c, t):
    return PT.TPPrefill.segment_plan(tpp, c, t)


def _replay_persistence(tpp, c, cuts):
    """forward_segment's boundary rule on a model of the outputs: a layer whose hook runs at c returns the persistent
    residual, any other layer a fresh tensor; every interior cut must end on the persistent one"""
    persistent = object()
    x = None
    for s in range(len(cuts) - 1):
        for li in range(cuts[s], cuts[s + 1]):
            hook = tpp.pmk_layers.get(tpp.layers[li].L)
            x = persistent if (hook is not None and hook.runs(c)) else object()
        if s < len(cuts) - 2:
            assert x is persistent, f"segment {s} of ({c}, ...) ends on a non-persistent activation (cuts {cuts})"


def test_hook_runs_reflects_the_planned_programs():
    owner = _hook_stack({(2048, 5): object()})
    run = PS.PmkStack._hook(owner, 5)
    assert run.runs(2048) and not run.runs(512) and not PS.PmkStack._hook(owner, 6).runs(2048)


@pytest.mark.parametrize("t", SPARSE_T)
@pytest.mark.parametrize("c", SERVED_BUCKETS)
def test_served_bucket_set_s3_cuts_are_persistent(c, t):
    tpp = _tpp()
    cuts = _plan(tpp, c, t)
    assert cuts[0] == 0 and cuts[-1] == N and cuts == sorted(cuts)
    if c < 2048:
        assert cuts == [0, N]  # the 512 bucket: one trace (the plugin never segments it either)
    else:
        assert len(cuts) == 4  # S = 3 on the PMK bucket
    _replay_persistence(tpp, c, cuts)


def test_the_k4d3_failure_case_is_closed():
    """the re-mock's plan: (512, 0) at S = 3 cut after layers 13 / 29 although no PMK program runs at 512"""
    tpp = _tpp()
    assert _plan(tpp, 512, 0) == [0, N]


def test_min_bucket_lowered_still_never_cuts_after_an_op_layer():
    """GLM53_PF_SEGMENT_MIN_BUCKET=512 alone does not segment the 512 bucket: no layer's program runs there"""
    tpp = _tpp(min_bucket=512)
    assert _plan(tpp, 512, 0) == [0, N]


def test_min_bucket_lowered_with_512_programs_segments_it():
    tpp = _tpp(min_bucket=512, pmk_buckets=(512, 2048))
    cuts = _plan(tpp, 512, 0)
    assert len(cuts) == 4
    _replay_persistence(tpp, 512, cuts)


def test_cuts_only_after_layers_whose_program_runs():
    """only the KDA layers carry programs at 2048 -> no cut after a sparse_dsa layer"""
    kda = [i for i in range(N) if _kind(i) != PT.SPARSE_DSA]
    tpp = _tpp(hook_layers=kda)
    for t in SPARSE_T:
        cuts = _plan(tpp, 2048, t)
        assert all(_kind(i - 1) != PT.SPARSE_DSA for i in cuts[1:-1]), cuts
        _replay_persistence(tpp, 2048, cuts)


def test_hook_without_runs_is_never_cut():
    tpp = _tpp()
    tpp.pmk_layers = {L: (lambda x, **kw: x) for L in range(N)}  # no .runs
    assert _plan(tpp, 2048, 0) == [0, N]


def test_s1_unchanged():
    tpp = _tpp(segments=1)
    assert _plan(tpp, 2048, 32768) == [0, N] and _plan(tpp, 512, 0) == [0, N]
