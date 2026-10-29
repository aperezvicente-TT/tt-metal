# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: GLM53_PF_NOHEAD_KEYS (Stage 9 cf3g, program 2026-10-23T19:07:21Z (3)) limits the head-less twins to the
listed (bucket, window) keys: the mock capture flow runs the head-less forward only for them, nohead_ready() is False for
every other key (its non-final chunks keep the head trace), and unset keeps a twin per key. TPPrefill runs as a host
stand-in (TPPrefill.__new__ + the attributes those methods read)."""

from __future__ import annotations

from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as P


def stand_in(nohead=True, keys=None):
    o = P.TPPrefill.__new__(P.TPPrefill)
    o.nohead, o.nohead_keys = nohead, keys
    o.traces, o.traces_nh, o.deferred, o.mock_captured = {}, {}, set(), set()
    o.lazy, o.buckets, o.batch_keys, o.packed_keys = set(), (512, 2048), [], []
    o.ran = []
    o._warmed = False  # e604417e4b7: capture() rebinds each key's inputs (_key_inputs) before its forward / capture
    o._key_inputs = lambda key, full: None  # the device-side rebinding: nothing on the host stand-in
    o.forward = lambda c, t=0, head=True: o.ran.append((c, t, head))
    o.fresh_keys = lambda buckets=None: []
    o.sparse_keys = lambda buckets=None: [(512, 65536), (2048, 65536), (2048, 262144)]
    return o


def test_twin_filter():
    o = stand_in(keys={(2048, 65536)})
    assert o._twin((2048, 65536)) and o._twin([2048, 65536])
    assert not o._twin((2048, 262144)) and not o._twin((512, 65536)) and not o._twin((2048, 0))
    assert stand_in(keys=None)._twin((512, 0))
    assert not stand_in(nohead=False, keys={(2048, 65536)})._twin((2048, 65536))


def test_mock_capture_runs_twins_of_listed_keys_only(monkeypatch):
    monkeypatch.setenv("GLM53_MOCK", "1")
    monkeypatch.setenv("GLM53_MOCK_CAPTURE", "1")
    o = stand_in(keys={(2048, 65536)})
    o.capture()
    assert [r for r in o.ran if not r[2]] == [(2048, 65536, False)]
    assert sorted(o.mock_captured) == [(512, 0), (512, 65536), (2048, 0), (2048, 65536), (2048, 262144)]
    o = stand_in(keys=None)
    o.capture()
    assert len([r for r in o.ran if not r[2]]) == 5


def test_nohead_ready_follows_the_filter():
    o = stand_in(keys={(2048, 65536)})
    o.traces_nh[(2048, 65536)] = "tid"
    o.deferred.add((2048, 262144))  # a lazily captured head key: still no twin for it
    o.plan = lambda s, n: (2048, 65536) if s + 2048 <= 65536 else (2048, 262144)
    assert o.nohead_ready(2048, 2048)
    assert not o.nohead_ready(65536, 2048)
