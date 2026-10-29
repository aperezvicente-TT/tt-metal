"""TPPrefill lazy capture (GLM53_PF_LAZY_KEYS, Stage 12): deferred keys are skipped by capture() and captured on first
use by replay / prefill_batch / prefill_packed; the head-less twin can be deferred alone; "" = everything at start."""

from types import MethodType, SimpleNamespace

import pytest

ttnn = pytest.importorskip("ttnn")
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: E402


class FakeMesh:
    def num_program_cache_entries(self):
        return 0


def _tp(lazy, nohead=True):
    tp = SimpleNamespace(mesh=FakeMesh(), traces={}, traces_nh={}, nohead=nohead, _warmed=True, buckets=(512, 2048, 8192),
                         batch_keys=[(512, 4)], packed_keys=[(2048, 3)], counters=dict(replays=0, captures=0, eager=0),
                         lazy=set(lazy), deferred=set(), cost=[], captured=[], nohead_keys=None, packed_cont=False,
                         _key_inputs=lambda key, full: None)  # fmt: skip  (e604417e4b7: capture() rebinds a key's inputs first)
    tp.fresh_keys = lambda buckets=None: []
    tp.sparse_keys = lambda buckets=None: [(2048, 4096)]
    for name in ("_lazy", "_lazy_capture", "_cost", "_cost_report", "capture", "replay", "nohead_ready", "_twin"):
        setattr(tp, name, MethodType(getattr(PT.TPPrefill, name), tp))

    def capture_key(key, head, leak_dbg=False, twin=True):
        tp.captured.append(("key", key, twin))
        tp.traces[key] = len(tp.captured)
        if twin and tp.nohead:
            tp.traces_nh[key] = -len(tp.captured)

    tp._capture_key = capture_key
    tp._capture_key_twin = lambda key: (tp.captured.append(("twin", key)), tp.traces_nh.__setitem__(key, 0))
    tp.capture_batch = lambda s, nb, head=True: (tp.captured.append(("batch", s, nb)), tp.traces.__setitem__(("batch", s, nb), 0))
    tp.capture_packed = lambda t, k, head=True: (tp.captured.append(("packed", t, k)), tp.traces.__setitem__(("packed", t, k), 0))
    tp.plan = lambda chunk_start, n_valid: (2048, 4096) if chunk_start else (2048, 0)
    return tp


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    monkeypatch.setattr(PT.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(PT.ttnn, "execute_trace", lambda *a, **k: None)
    monkeypatch.delenv("GLM53_MOCK", raising=False)
    monkeypatch.setenv("GLM53_PF_COST", "0")


def test_capture_defers_the_listed_classes_and_first_use_captures_once():
    tp = _tp({"8192", "sparse", "packed"})
    tp.capture()
    assert {c[1] for c in tp.captured if c[0] == "key"} == {(512, 0), (2048, 0)}
    assert ("batch", 512, 4) in tp.captured and ("packed", 2048, 3) not in tp.captured
    assert tp.deferred == {(8192, 0), (2048, 4096), ("packed", 2048, 3)}
    n = len(tp.captured)
    tp.replay(8192)  # first use: captured now, then replayed
    assert tp.captured[n] == ("key", (8192, 0), True) and (8192, 0) not in tp.deferred
    tp.replay(8192)  # second use: no capture
    assert len(tp.captured) == n + 1 and tp.counters["replays"] == 2 and tp.counters["lazy_captures"] == 1
    assert tp.nohead_ready(4096, 128)  # a deferred key counts as ready for the head-less route
    tp.replay(2048, t=4096, head=False)  # the head-less first use captures head + twin
    assert (2048, 4096) in tp.traces_nh and not tp.deferred - {("packed", 2048, 3)}


def test_nohead_alone_deferred_and_the_default_defers_nothing():
    tp = _tp({"nohead"})
    tp.capture()
    assert all(c[2] is False for c in tp.captured if c[0] == "key") and not tp.traces_nh
    assert ("nohead", (2048, 0)) in tp.deferred
    tp.replay(2048, head=False)
    assert ("twin", (2048, 0)) in tp.captured and (2048, 0) in tp.traces_nh
    tp0 = _tp(set())
    tp0.capture()
    assert not tp0.deferred and len([c for c in tp0.captured if c[0] == "key"]) == 4


def test_a_key_neither_captured_nor_deferred_still_raises():
    tp = _tp(set())
    with pytest.raises(RuntimeError, match="was not captured"):
        tp.replay(1024)
