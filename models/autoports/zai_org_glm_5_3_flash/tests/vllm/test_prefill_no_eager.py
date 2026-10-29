# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the TP prefill never runs an uncaptured key eager on device (operator 2026-10-22 09:18Z: eager execution
is prohibited). `replay`, `prefill_batch` and `prefill_packed` raise for a key with no captured trace; only the image
mock (GLM53_MOCK=1, no traces) runs the forward eager. TPPrefill runs as a host stand-in (TPPrefill.__new__ + the
attributes those methods read); a captured key goes to ttnn.execute_trace, stubbed here."""

from __future__ import annotations

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as P


def stand_in(monkeypatch, traces=(), traces_nh=(), batch=(), packed=()):
    o = P.TPPrefill.__new__(P.TPPrefill)
    o.mesh = object()
    o.traces = {k: f"tid{k}" for k in (*traces, *batch, *packed)}
    o.traces_nh = {k: f"tid_nh{k}" for k in traces_nh}
    o.counters = {"eager": 0, "replays": 0}
    o.deferred = set()  # GLM53_PF_LAZY_KEYS (Stage 12): no key deferred to its first use here
    o._warmed = False  # e604417e4b7: capture() rebinds each key's inputs (_key_inputs) before its forward / capture
    o._key_inputs = lambda key, full: None  # the device-side rebinding: nothing on the host stand-in
    o.ran = []
    o.forward = lambda c, t=0, head=True: o.ran.append(("forward", c, t, head))
    o.forward_batch = lambda s, nb, head=True: o.ran.append(("forward_batch", s, nb))
    o.forward_packed = lambda t, k, head=True, cont=False: o.ran.append(("forward_packed", t, k))
    o.write_batch = lambda s, reqs: None
    o.write_packed = lambda t, reqs, cont=False: [0] * len(reqs)
    # (tokens, slot, n_valid[, chunk_start[, head]]) -> 5-tuples, as TPPrefill.packed_reqs (the continuation twin's key)
    o.packed_reqs = lambda reqs: [tuple(r) + (0, True)[len(r) - 3:] for r in reqs]
    monkeypatch.setattr(P.ttnn, "execute_trace", lambda mesh, tid, cq_id=0, blocking=False: o.ran.append(("trace", tid)))
    return o


@pytest.fixture
def device(monkeypatch):
    monkeypatch.delenv("GLM53_MOCK", raising=False)


def test_replay_captured_key_replays(monkeypatch, device):
    o = stand_in(monkeypatch, traces=[(512, 0)], traces_nh=[(512, 0)])
    o.replay(512, t=0, head=True)
    o.replay(512, t=0, head=False)
    assert o.ran == [("trace", "tid(512, 0)"), ("trace", "tid_nh(512, 0)")]
    assert o.counters == {"eager": 0, "replays": 2}


@pytest.mark.parametrize("key, head", [((2048, 0), True), ((512, 0), False), ((512, 128), True)])
def test_replay_uncaptured_key_raises(monkeypatch, device, key, head):
    o = stand_in(monkeypatch, traces=[(512, 0)])
    with pytest.raises(RuntimeError, match="was not captured"):
        o.replay(key[0], t=key[1], head=head)
    assert o.ran == [] and o.counters == {"eager": 0, "replays": 0}


def test_batch_and_packed_uncaptured_keys_raise(monkeypatch, device):
    o = stand_in(monkeypatch, batch=[("batch", 512, 2)], packed=[("packed", 2048, 3)])
    reqs = lambda n: [([1, 2, 3], i, 3) for i in range(n)]  # noqa: E731
    o.prefill_batch(512, reqs(2))
    assert o.prefill_packed(2048, reqs(3)) == [0, 0, 0]
    with pytest.raises(RuntimeError, match="batch key"):
        o.prefill_batch(512, reqs(3))
    with pytest.raises(RuntimeError, match="packed key"):
        o.prefill_packed(2048, reqs(2))
    assert [r[0] for r in o.ran] == ["trace", "trace"] and o.counters["eager"] == 0


def test_image_mock_runs_eager(monkeypatch):
    monkeypatch.setenv("GLM53_MOCK", "1")
    o = stand_in(monkeypatch)
    o.replay(512, t=0)
    o.prefill_batch(512, [([1], 0, 1)])
    o.prefill_packed(2048, [([1], 0, 1)])
    assert [r[0] for r in o.ran] == ["forward", "forward_batch", "forward_packed"]
    assert o.counters["eager"] == 3 and o.counters["replays"] == 0


def _mock_flow(monkeypatch, o, lazy=()):
    monkeypatch.setenv("GLM53_MOCK", "1")
    monkeypatch.setenv("GLM53_MOCK_CAPTURE", "1")
    o.buckets, o.nohead, o.lazy = (512, 2048), False, set(lazy)
    o.fresh_keys = lambda b: []
    o.sparse_keys = lambda b: [(512, 65536)]
    o.batch_keys, o.packed_keys = [(512, 2)], []
    o.mock_captured = set()
    return o


def test_mock_capture_flow_replays_only_what_capture_recorded(monkeypatch):
    """GLM53_MOCK_CAPTURE=1: capture() records the keys the device would capture; a replay of one of them runs, a replay
    with NO capture before it raises as on the device (the n29h7s2 1542 start: capture() never ran at 16 seats)"""
    o = _mock_flow(monkeypatch, stand_in(monkeypatch))
    with pytest.raises(RuntimeError, match="was not captured \\(mock capture flow"):
        o.replay(512, t=0)  # nothing captured yet: the 1542 failure on the mock
    o.capture()
    assert o.mock_captured == {(512, 0), (2048, 0), (512, 65536), ("batch", 512, 2)}
    o.replay(512, t=0)
    o.replay(512, t=65536, head=False)
    o.prefill_batch(512, [([1], 0, 1), ([2], 1, 1)])
    with pytest.raises(RuntimeError, match="was not captured"):
        o.prefill_packed(2048, [([1], 0, 1)])  # no packed key in this construction
    with pytest.raises(RuntimeError, match="was not captured"):
        o.replay(512, t=131072)


def test_mock_capture_flow_lazy_keys_are_captured_at_first_use(monkeypatch):
    o = _mock_flow(monkeypatch, stand_in(monkeypatch), lazy=("sparse",))
    o.capture()
    assert (512, 65536) in o.deferred and (512, 65536) not in o.mock_captured
    o.replay(512, t=65536)  # the first use captures it, as _lazy_capture does on the device
    assert (512, 65536) in o.mock_captured and (512, 65536) not in o.deferred


def test_mock_without_the_capture_flow_stays_eager(monkeypatch):
    monkeypatch.setenv("GLM53_MOCK", "1")
    monkeypatch.delenv("GLM53_MOCK_CAPTURE", raising=False)
    o = stand_in(monkeypatch)
    o.replay(2048, t=0)
    assert o.ran == [("forward", 2048, 0, True)]
