# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The adapter's segtrace face, its failure paths (14's review 2026-10-27T13:07:08Z): a later segment that raises leaves no
open chunk behind (the model's, the record, the reserved snapshot entries), so the slot's next segment 0 is accepted;
open_chunk() names vLLM slots only; prefill_abort() tolerates an unknown slot. Host-only, on a stand-in adapter."""

from types import SimpleNamespace

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import Glm5NextForCausalLM


class _Gen:
    def __init__(self, fail_at=None, open_slot=None):
        self.fail_at, self.aborted, self.calls = fail_at, [], []
        self.backend = SimpleNamespace(prefill_segments=lambda s, n: 3,
                                       tpp=SimpleNamespace(open_chunk=lambda: open_slot))  # fmt: skip

    def prefill_forward(self, chunk, **kw):
        self.calls.append(kw.get("segment_rows"))
        if kw.get("segment_rows") == [self.fail_at]:
            raise RuntimeError("device step failed")
        return torch.tensor([-1])

    def prefill_abort(self, slot):
        self.aborted.append(slot)


def _adapter(gen):
    a = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    a.generator, a.gen_slot, a.counters, a.prefix = gen, [5, 6, 7, 8], {}, None
    a._seg_open = {1: {"raw": (0, 1024), "n_seg": 3, "chunk": torch.zeros(1, 1024), "page_rows": None,
                       "lengths": [1024], "gslots": [6], "starts": [0], "keys": [0], "tokens": None, "table": None,
                       "pc_attach": {}, "pc_blob": {}, "pc_snap": {}, "pc_key": {}, "extra": {}}}  # fmt: skip
    a._traces_captured, a.kv_cache, a._spec, a.max_seq_len = True, object(), None, 1 << 20
    a._in_call, a._engine_tid, a._last_exit_t, a._last_kind, a._events_path = False, 1, None, None, None
    a.slot_table = SimpleNamespace(slots={6: object()})
    a._prefix_states = {}
    return a


def _segment(a, s):
    return a.prefill_forward(torch.zeros(1, 1024, dtype=torch.int64), torch.zeros(1, 16, dtype=torch.int32),
                             a.kv_cache, [1024], start_pos=[0], empty_slots=[1], segment_rows=[s])  # fmt: skip


def test_a_failed_later_segment_drops_the_open_chunk():
    gen = _Gen(fail_at=1)
    a = _adapter(gen)
    with pytest.raises(RuntimeError, match="device step failed"):
        _segment(a, 1)
    assert gen.aborted == [6] and 1 not in a._seg_open, (gen.aborted, a._seg_open)
    assert a.counters["prefill_segment_aborts"] == 1


def test_a_later_segment_re_sends_segment_0s_record():
    gen = _Gen()
    a = _adapter(gen)
    _segment(a, 1)
    assert gen.calls == [[1]] and 1 in a._seg_open  # not the last segment: still open, nothing committed


def test_open_chunk_names_vllm_slots_only():
    assert _adapter(_Gen(open_slot=7)).open_chunk() == 2  # generator slot 7 = vLLM slot 2
    assert _adapter(_Gen(open_slot=42)).open_chunk() is None  # no vLLM slot maps to it: None, not the generator's id
    assert _adapter(_Gen(open_slot=None)).open_chunk() is None


def test_prefill_abort_tolerates_an_unknown_slot():
    gen = _Gen()
    a = _adapter(gen)
    a.prefill_abort(99)  # out of range: nothing to abort on the model, no IndexError
    assert gen.aborted == []
    a.prefill_abort(1)
    assert gen.aborted == [6] and not a._seg_open
