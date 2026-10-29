# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the segtrace open-chunk guard (14's review 2026-10-27T09:16:21Z MUST + SHOULD). Between two
segments of a chunk its live state sits in shared buffers (every layer's KDA scratch, the bucket's inputs, the PMK
residual), so every other prefill entry must refuse while a segmented chunk is open; open_chunk() lets a scheduler ask
first."""
import types

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PF


def _tp(pending):
    ns = types.SimpleNamespace(_seg_pending=pending)
    ns.open_chunk = types.MethodType(PF.TPPrefill.open_chunk, ns)
    ns._refuse_if_open = types.MethodType(PF.TPPrefill._refuse_if_open, ns)

    def _never(*a, **k):
        raise AssertionError("_prefill reached while a segmented chunk is open")

    ns._prefill = _never
    return ns


def test_open_chunk_is_none_when_idle():
    assert _tp({}).open_chunk() is None


def test_open_chunk_names_the_slot():
    assert _tp({3: dict(chunk_start=0, n_valid=2048, next=1)}).open_chunk() == 3


def test_refusal_only_while_open():
    _tp({})._refuse_if_open("a whole-chunk prefill")  # idle: no refusal
    with pytest.raises(RuntimeError, match="open segmented chunk"):
        _tp({5: dict(chunk_start=2048, n_valid=2048, next=2)})._refuse_if_open("a batched prefill")


def test_whole_chunk_prefill_refuses_before_any_device_work():
    ns = _tp({1: dict(chunk_start=0, n_valid=2048, next=1)})
    with pytest.raises(RuntimeError, match="whole-chunk prefill of slot 2"):
        PF.TPPrefill.prefill(ns, torch.zeros(2048, dtype=torch.int64), slot=2, chunk_start=0, n_valid=2048)


def test_w1_group_refuses_while_open():
    from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM

    be = types.SimpleNamespace(tpp=_tp({0: dict(chunk_start=0, n_valid=2048, next=1)}))
    with pytest.raises(RuntimeError, match="W1 group of slot 4"):
        SM.StackBackend.prefill_group(be, torch.zeros(4096, dtype=torch.int64), kv_cache=None, slot=4, chunk_start=0,
                                      n_valid=[2048, 2048], logits_rows="none")


# ---- the state I/O control words across segments (14's review 2026-10-27T09:20:00Z MUST): StateIO.ctl is ONE table read
# in the trace by the last segment's STATE_OUT; a copy() between two segments re-points it, so the last segment re-issues
# the chunk's words before its replay, and copy() refuses an open chunk's slot.
def _sio_tp(monkeypatch, S=3):
    words = []  # every ctl upload, in order: [slot, fresh, src, snap]
    monkeypatch.setattr(PF.ttnn, "from_torch", lambda v, **k: v)
    monkeypatch.setattr(PF.ttnn, "copy_host_to_device_tensor", lambda v, ctl: words.append([int(x) for x in v[0, :4]]))
    tp = types.SimpleNamespace(_seg_pending={}, start_align=64, max_batch=32, slot_rows=40, counters={}, nohead=False,
                               traces_nh={}, repl=None, replays=[])
    sio = types.SimpleNamespace(tpp=tp, ctl=None, runs=[])
    sio.write = types.MethodType(PF.StateIO.write, sio)
    sio.copy = types.MethodType(PF.StateIO.copy, sio)
    sio._run = lambda mode: sio.runs.append((mode, list(words[-1])))
    tp.sio = sio
    for m in ("prefill", "_prefill", "_seg_rewrite_ctl", "_refuse_if_open", "open_chunk"):
        setattr(tp, m, types.MethodType(getattr(PF.TPPrefill, m), tp))
    tp.nsegments = lambda cs, nv: S
    tp.plan = lambda cs, nv: (2048, 0)
    tp.bucket_for = lambda n, sparse=False: 2048
    tp._sparse = lambda cs, nv: False
    tp._check_window = lambda *a: None
    tp.b, tp.kv = types.SimpleNamespace(_dsa_ops=lambda *a, **k: []), types.SimpleNamespace(extra={})  # no DSA pages

    def write_inputs(c, tokens, *, slot, chunk_start, n_valid, state_src=None, state_snap=None, snap_at=None):
        tp.sio.write(slot, fresh=chunk_start == 0 and state_src is None, src=state_src, snap=state_snap)  # = the real one's
        return False

    tp.write_inputs = write_inputs
    tp.replay = lambda c, t=0, head=True, segment=None: tp.replays.append((segment, list(words[-1])))
    return tp, sio, words


def _run_chunk(tp, sio, *, slot, between=None, snap=None, S=3):
    tok = torch.zeros(2048, dtype=torch.int64)
    for s in range(S):
        tp.prefill(tok, slot=slot, chunk_start=0, n_valid=2048, state_snap=snap, segment=s)
        if s == 0 and between is not None:
            between()


def test_last_segment_writes_the_open_slot_after_a_copy(monkeypatch):
    tp, sio, words = _sio_tp(monkeypatch)
    _run_chunk(tp, sio, slot=2, between=lambda: sio.copy(33, 7))  # a prefix-cache attach copy between two segments
    assert sio.runs == [("copy", [7, 0, 33, -1])]
    seg, w = tp.replays[-1]
    assert seg == 2 and w == [2, 1, -1, -1], tp.replays  # STATE_OUT of the last segment writes slot 2, not the copy's 7
    assert tp.open_chunk() is None


def test_last_segment_keeps_the_snapshot_entry(monkeypatch):
    tp, sio, words = _sio_tp(monkeypatch)
    _run_chunk(tp, sio, slot=4, snap=1, between=lambda: sio.copy(6, 34))
    assert tp.replays[0][1] == [4, 1, -1, 33] and tp.replays[-1][1] == [4, 1, -1, 33], tp.replays


def test_without_the_reissue_the_copy_destination_leaks(monkeypatch):
    # the control: what the MUST describes -- the last segment's STATE_OUT would target the copy's destination
    tp, sio, words = _sio_tp(monkeypatch)
    tp._seg_rewrite_ctl = lambda *a: None
    _run_chunk(tp, sio, slot=2, between=lambda: sio.copy(33, 7))
    assert tp.replays[-1][1][0] == 7


def test_copy_refuses_an_open_chunks_slot(monkeypatch):
    tp, sio, words = _sio_tp(monkeypatch)
    tp.prefill(torch.zeros(2048, dtype=torch.int64), slot=3, chunk_start=0, n_valid=2048, segment=0)
    for a, b in ((3, 9), (9, 3)):
        with pytest.raises(RuntimeError, match="open segmented chunk"):
            sio.copy(a, b)
    assert sio.runs == []
    sio.copy(9, 10)  # other slots: allowed
    assert sio.runs[-1][0] == "copy"


def test_segments_carry_segment_0s_prefix_args(monkeypatch):
    tp, sio, words = _sio_tp(monkeypatch)
    tok = torch.zeros(2048, dtype=torch.int64)
    tp.prefill(tok, slot=1, chunk_start=0, n_valid=2048, state_snap=2, segment=0)
    with pytest.raises(AssertionError, match="!= segment 0's"):
        tp.prefill(tok, slot=1, chunk_start=0, n_valid=2048, segment=1)  # the snapshot entry dropped
