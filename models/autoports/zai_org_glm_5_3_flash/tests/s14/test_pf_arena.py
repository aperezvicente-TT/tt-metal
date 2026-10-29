# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TPPrefill.arena (GLM53_PF_ARENA): one tensor per (kind, key) shared by every SparseCtx of a TPPrefill; 0 = a tensor
per call (the per-key form); in_arena marks the shared ones so a key's release leaves them. Host only."""
import types

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PF


def _tpp():
    tpp = object.__new__(PF.TPPrefill)
    tpp._arena, tpp._arena_ids, tpp._spa_relay = {}, set(), None
    return tpp


def test_one_tensor_per_kind_and_key(monkeypatch):
    monkeypatch.delenv("GLM53_PF_ARENA", raising=False)
    tpp, made = _tpp(), []
    mk = lambda: made.append(object()) or made[-1]  # noqa: E731
    a = tpp.arena("pool_buf", 16384, mk)
    b = tpp.arena("pool_buf", 16384, mk)  # the other bucket at the same window
    c = tpp.arena("pool_buf", 32768, mk)  # a wider window: its own
    d = tpp.arena("ids", 256, mk)
    assert a is b and a is not c and len(made) == 3
    assert all(tpp.in_arena(t_) for t_ in (a, c, d)) and not tpp.in_arena(object())


def test_off_is_a_tensor_per_call(monkeypatch):
    monkeypatch.setenv("GLM53_PF_ARENA", "0")
    tpp = _tpp()
    a = tpp.arena("pool_buf", 16384, object)
    b = tpp.arena("pool_buf", 16384, object)
    assert a is not b and not tpp.in_arena(a) and not tpp.in_arena(b)


@pytest.mark.parametrize("arena", ["1", "0"])
def test_a_key_release_leaves_the_shared_tensors(monkeypatch, arena):
    monkeypatch.setenv("GLM53_PF_ARENA", arena)
    tpp, freed = _tpp(), []
    monkeypatch.setattr(PF.ttnn, "deallocate", lambda t_: freed.append(t_))
    shared = tpp.arena("pool_buf", 16384, object)
    own = object()
    ctx = object.__new__(PF.SparseCtx)
    ctx.tpp, ctx.c_add, ctx.nvis, ctx.spu = tpp, None, None, None
    for name in ("perm", "tau", "iota_c", "rows_local", "pool_iota", "lat_buf", "ids", "row_base", "pool_loc", "pool_g2l",
                 "lat_g2l", "shard_page", "counts", "ids_new"):
        setattr(ctx, name, None)
    ctx.pool_buf, ctx.ids = shared, own
    ctx.release()
    assert own in freed
    assert (shared in freed) == (arena == "0")
