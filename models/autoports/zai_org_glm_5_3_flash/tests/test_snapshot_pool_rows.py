# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefix-cache snapshot pool honours its entry count (GLM53_PREFIX_DEVICE_ENTRIES not a multiple of the state
chunk: the pool's last chunk holds the remainder rows). Host-only: StackBackend's slot-op row math on fake state
tensors (shape / padded_shape / buffer_address), no device."""

from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM

CHUNK, HEADS = 32, 8  # B = 32 rows per state chunk; TP8: 8 KDA heads per chip -> 8 x 4 x 4 = 128 S tiles per user
N_S = HEADS * 4 * 4


class _T:
    def __init__(self, shape, addr, padded=None):
        self.shape, self.padded_shape, self._addr = tuple(shape), tuple(padded or shape), addr

    def buffer_address(self):
        return self._addr


def _state(rows, base):
    S = _T((rows, HEADS, 128, 128), base)
    C = _T((1, 3, rows, 3072), base + 1, padded=(1, 3, -(-rows // 32) * 32, 3072))  # user rows inside 32-row tiles
    return S, C


def _cache(pool_rows):
    """one live chunk of 32 slots + a pool chunk of `pool_rows` entries; layers: KDA, sparse_dsa, KDA"""
    states = [[_state(CHUNK, 1000), None, _state(CHUNK, 2000)],
              [_state(pool_rows, 3000), None, _state(pool_rows, 4000)]]  # fmt: skip
    kv = SimpleNamespace(states=states, extra={"snapshot_chunks": 1, "snapshot_entries": pool_rows}, max_batch=CHUNK)
    be = SimpleNamespace(chunk=CHUNK, seq=[dict(sparse=False), dict(sparse="dsa"), dict(sparse=False)])
    return be, kv


def test_slot_to_the_last_entry_of_a_16_row_pool():
    be, kv = _cache(16)
    entry = 15
    dst = SM.StackBackend._entry_slot(be, kv, entry)
    assert dst == CHUNK + entry
    ops = SM.StackBackend._state_ops(be, kv, dst, 3)
    s_ops = [op for op in ops if op[0] == 1]
    c_ops = [op for op in ops if op[0] == 2]
    assert len(s_ops) == 2 and len(c_ops) == 2  # the two KDA layers; sparse_dsa: no DSA store in this cache
    for op, (src_base, dst_base) in zip(s_ops, ((1000, 3000), (2000, 4000))):
        kind, sa, sp, da, dp, n, nbytes, _ = op
        assert (sa, da, n, nbytes) == (src_base, dst_base, N_S, 4096)
        assert sp == 3 * N_S and dp == entry * N_S  # row 15 of the 16-row tensor
        assert dp + n <= 16 * N_S  # stays inside the short tensor
    for op in c_ops:
        assert op[-1] == (3 << 16) | entry  # conv rows: src row 3 -> dst row 15


def test_restore_from_a_short_pool_into_a_live_slot():
    be, kv = _cache(16)
    ops = SM.StackBackend._state_ops(be, kv, 7, SM.StackBackend._entry_slot(be, kv, 4))
    s0 = [op for op in ops if op[0] == 1][0]
    assert s0[1] == 3000 and s0[2] == 4 * N_S and s0[3] == 1000 and s0[4] == 7 * N_S and s0[5] == N_S


def test_entry_bound_is_the_exact_count():
    be, kv = _cache(16)
    SM.StackBackend._entry_slot(be, kv, 15)
    with pytest.raises(ValueError, match="outside the pool of 16"):
        SM.StackBackend._entry_slot(be, kv, 16)


def test_full_chunk_pool_unchanged():
    be, kv = _cache(32)
    ops = SM.StackBackend._state_ops(be, kv, CHUNK + 31, 0)
    s0 = [op for op in ops if op[0] == 1][0]
    assert s0[4] == 31 * N_S and s0[5] == N_S


@pytest.mark.parametrize("entries,chunk,rows", [(16, 32, [16]), (40, 32, [32, 8]), (32, 32, [32]), (0, 32, []),
                                                (6, 4, [4, 2]), (3, 2, [2, 1])])  # fmt: skip
def test_pool_chunk_rows(entries, chunk, rows):
    assert SM.snapshot_chunk_rows(entries, chunk) == rows
    assert sum(rows) == entries


def _cache_rows(pool_rows):
    """one live chunk of 32 slots + one pool chunk per entry of pool_rows (e.g. [32, 8] = 40 entries)"""
    states = [[_state(CHUNK, 1000), None, _state(CHUNK, 2000)]]
    for i, r in enumerate(pool_rows):
        states.append([_state(r, 5000 + 100 * i), None, _state(r, 6000 + 100 * i)])
    n = sum(pool_rows)
    kv = SimpleNamespace(states=states, extra={"snapshot_chunks": len(pool_rows), "snapshot_entries": n},
                         max_batch=CHUNK)  # fmt: skip
    be = SimpleNamespace(chunk=CHUNK, seq=[dict(sparse=False), dict(sparse="dsa"), dict(sparse=False)])
    return be, kv


def test_two_pool_chunks_full_plus_tail():
    be, kv = _cache_rows(SM.snapshot_chunk_rows(40, CHUNK))  # [32, 8]
    for entry, (addr, row) in ((31, (5000, 31)), (32, (5100, 0)), (39, (5100, 7))):
        ops = SM.StackBackend._state_ops(be, kv, SM.StackBackend._entry_slot(be, kv, entry), 5)
        s0 = [op for op in ops if op[0] == 1][0]
        assert (s0[3], s0[4]) == (addr, row * N_S), (entry, s0)
    with pytest.raises(ValueError, match="outside the pool of 40"):
        SM.StackBackend._entry_slot(be, kv, 40)


def test_prefill_state_ops_reach_a_short_pool_row():
    from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT
    from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA

    be, kv = _cache_rows([16])
    scr = SimpleNamespace(S=_T((1, HEADS, 128, 128), 9000), C=_T((1, 3, 1, 3072), 9001, padded=(1, 3, 32, 3072)))
    layers = [SimpleNamespace(kind=DENSE_KDA, scratch=scr), SimpleNamespace(kind=SPARSE_DSA, scratch=None),
              SimpleNamespace(kind=DENSE_KDA, scratch=scr)]  # fmt: skip
    tpp = SimpleNamespace(b=be, kv=kv, layers=layers)
    out = PT.TPPrefill.state_ops(tpp, CHUNK + 15, into_scratch=False)  # scratch -> the pool's last row
    s_ops = [op for op in out if op[0] == 1]
    assert [(op[3], op[4], op[5]) for op in s_ops] == [(5000, 15 * N_S, N_S), (6000, 15 * N_S, N_S)]
