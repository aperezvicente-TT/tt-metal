# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Glm53Generator on the sequence-sharded DSA pool's block contract (GLM53_DSA_SHARD; doc/ttnn/
14-production-tp8-megakernel/dsa/shard_tp8_design.md §6.1). The CPU MockModel with dsa_shard = C keeps its latent pool
as C x the per-chip physical blocks and refuses any page table that is not owner-aware; the generator maps vLLM's
logical blocks through tt/owner_blocks.py. Tokens and decode logits must equal the unpaged reference through
admission, release + reuse of the blocks at other columns, slot compaction and a full reset, with every physical block
back on its owner's free list at the end. Negative controls: a map that puts a block on the wrong owner is refused by
the mock; a map that aliases two live logical blocks onto one physical block changes the logits.

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_generator_dsa_shard.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mock_model as M  # noqa: E402
from test_generator import _prefill_rows, check_decode_log, make, prompt  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import owner_blocks as OB  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402


def _sharded(C=2, max_batch=4, num_blocks=40, null_block=0):
    m, g = make("dense_exact", max_batch=max_batch)
    m.dsa_shard = C
    kv = g.allocate_kv_cache(max_batch, 1024, num_blocks=num_blocks, null_block=null_block)
    g.warmup_serving(kv, phase="prepare")
    g.warmup_serving(kv, phase="capture")
    return m, g, kv


def _owner_ok(b):
    C, bpu = b.obm.C, b.obm.bpu
    col = (torch.arange(b.phys.shape[1]) // bpu) % C
    assert ((b.phys % C) == col).all()
    host = b.phys  # the backend got exactly the physical table
    return host


def _row(blocks, bps):
    return torch.tensor([list(blocks) + [-1] * (bps - len(blocks))])


@pytest.mark.parametrize("C", [2, 4])
def test_sharded_pool_admit_release_reuse_remap(C):
    m, g, kv = _sharded(C=C)
    b = g._binding(kv)
    assert b.obm is not None and b.obm.C == C
    full = b.obm.free_counts()
    bps = b.blocks_per_seq
    u1, u2, u0 = prompt(130, seed=60), prompt(200, seed=61), prompt(150, seed=62)
    rows = torch.cat([_row([3, 7, 9], bps), _row([12, 5, 21, 30], bps)])
    first = g.prefill_forward(_prefill_rows([u1, u2]), page_table=rows, kv_cache=kv, prompt_lens=[130, 200],
                              slots=[1, 2], sampling_params=[SamplingParams()] * 2)  # fmt: skip
    seqs = {1: u1 + [int(first[0])], 2: u2 + [int(first[1])]}
    _owner_ok(b)
    host = next(c.page_table_host for l, c in zip(m.layers, kv.caches) if not l.is_kda)
    assert torch.equal(host.long(), b.phys)  # the prefill's host table is the physical one
    assert b.pt_shadow[1, :3].tolist() == [3, 7, 9]  # the generator's own shadow stays logical
    m.decode_log = []
    sp = [SamplingParams()] * 2
    log_owner = []
    for step in range(3):
        out = g.decode_forward(None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False)
        for s, t in zip((1, 2), out.tolist()):
            seqs[s].append(int(t))
        log_owner.append({1: list(seqs[1]), 2: list(seqs[2])})
    # release slot 2; its logical blocks (cached, still mapped) go to a new request in slot 0 at OTHER columns: 30
    # moves from chip 1 (column 3) to chip 0 (column 0), 12 / 5 swap columns within chip 0's unit
    g.release_slot(kv, 2)
    f0 = g.prefill_forward(_prefill_rows([u0]), page_table=_row([30, 12, 5], bps), kv_cache=kv, prompt_lens=[150],
                           slots=[0], sampling_params=[SamplingParams()])  # fmt: skip
    seqs[0] = u0 + [int(f0[0])]
    del seqs[2]
    _owner_ok(b)
    for step in range(3):
        if step == 1:  # slot compaction: 1 -> 3, 0 -> 1 (a chain; the latents stay on their physical blocks)
            phys_before = {1: b.phys[1].clone(), 0: b.phys[0].clone()}
            g.remap_slots(kv, {1: 3, 0: 1})
            seqs = {3: seqs[1], 1: seqs[0]}
            assert torch.equal(b.phys[3], phys_before[1]) and torch.equal(b.phys[1], phys_before[0])
        out = g.decode_forward(None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False)
        live = sorted(seqs)
        for s, t in zip(live, out.tolist()):
            seqs[s].append(int(t))
        log_owner.append({s: list(v) for s, v in seqs.items()})
    assert seqs[3][130:] == m.reference_greedy(u1, len(seqs[3]) - 130)
    assert seqs[1][150:] == m.reference_greedy(u0, len(seqs[1]) - 150)
    chk = check_decode_log(m, lambda step, s: log_owner[step].get(s))
    assert chk["checked_rows"] >= 12
    for s in (1, 3):
        g.release_slot(kv, s)
    assert not b.obm.refs and [f + e for f, e in zip(b.obm.free_counts(), b.obm.evictable_counts())] == full


def test_sharded_pool_full_reset_frees_everything():
    m, g, kv = _sharded(C=2)
    b = g._binding(kv)
    full = b.obm.free_counts()
    bps = b.blocks_per_seq
    g.prefill_forward(_prefill_rows([prompt(100, seed=3)]), page_table=_row([4, 8], bps), kv_cache=kv,
                      prompt_lens=[100], slots=[0], sampling_params=[SamplingParams()])  # fmt: skip
    assert b.obm.free_counts() != full
    g.reset()  # a caller-owned cache: every slot loses its coverage (the blocks stay cached, evictable)
    assert not b.obm.refs and torch.equal(b.phys, b.null_row.expand_as(b.phys))
    assert [f + e for f, e in zip(b.obm.free_counts(), b.obm.evictable_counts())] == full
    _owner_ok(b)


def test_sharded_pool_refused_row_keeps_previous_rows():
    """an owner chip out of physical blocks: the refused row keeps its previous mapping (nothing leaked), the rows
    mapped before it in the same write are published"""
    m, g, kv = _sharded(C=2)
    b = g._binding(kv)
    bps = b.blocks_per_seq
    g._write_page_rows(b, {0: _row([1, 2, 3, 4], bps)[0]})
    before, cov0 = b.phys.clone(), b.coverage.clone()
    b.obm.free[0] = b.obm.free[0][-2:]  # chip 0 keeps two free blocks (nothing cached to evict)
    free_before = b.obm.free_counts()
    with pytest.raises(RuntimeError, match="no free physical block"):
        # slot 2 takes chip 0's last two blocks (columns 0, 1); slot 1's column 0 finds chip 0 dry
        g._write_page_rows(b, {2: _row([5, 6], bps)[0], 1: _row([7, 8, 9], bps)[0]})
    assert int(b.coverage[2]) == 2 and int(b.coverage[1]) == 0 and torch.equal(b.coverage[[0, 3]], cov0[[0, 3]])
    assert torch.equal(b.phys[[0, 1, 3]], before[[0, 1, 3]]) and not torch.equal(b.phys[2], before[2])
    assert b.obm.free_counts() == [0, free_before[1]] and b.obm.physical(7) is None  # nothing of slot 1 stayed
    host = next(c.page_table_host for l, c in zip(m.layers, kv.caches) if not l.is_kda)
    assert torch.equal(host.long(), b.phys)
    _owner_ok(b)


def test_negative_control_wrong_owner_is_refused(monkeypatch):
    m, g, kv = _sharded(C=2)
    b = g._binding(kv)
    real = OB.OwnerBlockMap.set_row

    def off_owner(self, slot, row, covered, keep=0):
        out = real(self, slot, row, covered, keep)
        return [out[0] + 1] + out[1:]  # column 0's block on chip 1

    monkeypatch.setattr(OB.OwnerBlockMap, "set_row", off_owner)
    with pytest.raises(M.ContractViolation, match="owner chip"):
        g.prefill_forward(_prefill_rows([prompt(100, seed=4)]), page_table=_row([4, 8], b.blocks_per_seq),
                          kv_cache=kv, prompt_lens=[100], slots=[0], sampling_params=[SamplingParams()])  # fmt: skip


def test_negative_control_aliased_blocks_change_the_logits(monkeypatch):
    """two live logical blocks on ONE physical block (owner-correct): the second request's prefill overwrites the
    first's latents -> the first request's decode logits leave the reference"""
    m, g, kv = _sharded(C=2)
    b = g._binding(kv)
    bps = b.blocks_per_seq
    real = OB.OwnerBlockMap.set_row
    first_phys = {}

    def alias(self, slot, row, covered, keep=0):
        out = real(self, slot, row, covered, keep)
        if slot == 2:
            out[0] = first_phys[0]  # column 0 = owner 0 on both rows: owner-correct, aliased
        elif slot == 1:
            first_phys[0] = out[0]
        return out

    monkeypatch.setattr(OB.OwnerBlockMap, "set_row", alias)
    u1, u2 = prompt(100, seed=70), prompt(100, seed=71)
    f = g.prefill_forward(_prefill_rows([u1]), page_table=_row([3, 7], bps), kv_cache=kv, prompt_lens=[100],
                          slots=[1], sampling_params=[SamplingParams()])  # fmt: skip
    g.prefill_forward(_prefill_rows([u2]), page_table=_row([12, 5], bps), kv_cache=kv, prompt_lens=[100], slots=[2],
                      sampling_params=[SamplingParams()])  # fmt: skip
    seq = u1 + [int(f[0])]
    m.decode_log = []
    g.decode_forward(torch.tensor([seq[-1]]), torch.tensor([100]), page_table=None, kv_cache=kv, slots=[1],
                     sampling_params=[SamplingParams()], reload_inputs=True)  # fmt: skip
    with pytest.raises(AssertionError):
        check_decode_log(m, lambda step, s: seq if s == 1 else None)


def test_unsharded_backend_has_no_map():
    m, g = make("dense_exact", max_batch=4)
    m.dsa_shard = 1  # (the suite may run with GLM53_MOCK_DSA_SHARD set)
    kv = g.allocate_kv_cache(4, 1024, num_blocks=40, null_block=0)
    assert g._binding(kv).obm is None


def test_sharded_pool_b32_c8_native_rows_and_null_blocks():
    """B = 32 native slot rows on the 8-chip pool (the served geometry): 32 users admitted on vLLM-style logical
    blocks, every physical row owner-aware, idle rows = the owners' null blocks (local 0 of every chip, never handed
    out), the per-chip LOCAL tables of the whole table (tt/dsa_shard.py page_tables_local) consistent with the
    physical rows; decode / release / re-admission keep the logits at the unpaged reference; nothing leaks."""
    from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS

    C, B = 8, 32
    m, g, kv = _sharded(C=C, max_batch=B, num_blocks=B * 16, null_block=0)
    b = g._binding(kv)
    assert b.obm is not None and b.obm.C == C and b.phys.shape == (B, b.blocks_per_seq)
    assert torch.equal(b.phys[0], b.null_row) and torch.equal(b.null_row, DS.ShardMap(C).null_row(b.blocks_per_seq))
    full = b.obm.free_counts()
    bps = b.blocks_per_seq
    lens = [70 + 9 * s for s in range(B)]  # 70 .. 349 tokens: 2 .. 6 blocks per user
    users = [prompt(n, seed=100 + s) for s, n in enumerate(lens)]
    nxt = 1  # logical ids as vLLM hands them out: unique, in admission order
    rows = []
    for n in lens:
        k = -(-(n + 1) // 64)
        rows.append(_row(list(range(nxt, nxt + k)), bps))
        nxt += k
    first = g.prefill_forward(_prefill_rows(users), page_table=torch.cat(rows), kv_cache=kv, prompt_lens=lens,
                              slots=list(range(B)), sampling_params=[SamplingParams()] * B)  # fmt: skip
    seqs = {s: users[s] + [int(first[s])] for s in range(B)}
    _owner_ok(b)
    # every chip's local 0 stays the null block: no live row maps to it; the per-chip local tables agree
    assert all((b.phys[s, : b.coverage[s]] // C > 0).all() for s in range(B))
    lt = DS.ShardMap(C).page_tables_local(b.phys, None)
    for s in range(B):
        for j in range(int(b.coverage[s])):
            r, gblk = (j // 2) % C, int(b.phys[s, j])
            assert gblk % C == r and int(lt[r, s, (j // 2 // C) * 2 + j % 2]) == gblk // C
    m.decode_log = []
    sp = [SamplingParams()] * B
    log = []
    for step in range(3):
        out = g.decode_forward(None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False)
        for s, t in enumerate(out.tolist()):
            seqs[s].append(int(t))
        log.append({s: list(v) for s, v in seqs.items()})
    chk = check_decode_log(m, lambda step, s: log[step].get(s))
    assert chk["checked_rows"] >= 3 * B
    for s in range(B):
        assert seqs[s][lens[s]:] == m.reference_greedy(users[s], len(seqs[s]) - lens[s])
    # release half the users; their cached blocks are reused by new users at other columns (moves within the map)
    for s in range(0, B, 2):
        g.release_slot(kv, s)
        assert torch.equal(b.phys[s], b.null_row)
    new = {s: prompt(90 + s, seed=300 + s) for s in range(0, B, 2)}
    reuse = {s: _row([int(v) for v in rows[(s + 2) % B][0, :2]], bps) for s in new}  # another released user's blocks
    f2 = g.prefill_forward(_prefill_rows([new[s] for s in sorted(new)]), page_table=torch.cat([reuse[s] for s in sorted(new)]),
                           kv_cache=kv, prompt_lens=[len(new[s]) for s in sorted(new)], slots=sorted(new),
                           sampling_params=[SamplingParams()] * len(new))  # fmt: skip
    for s, t in zip(sorted(new), f2.tolist()):
        seqs[s] = new[s] + [int(t)]
    _owner_ok(b)
    out = g.decode_forward(None, None, page_table=None, kv_cache=kv, sampling_params=sp, reload_inputs=False)
    for s, t in enumerate(out.tolist()):
        seqs[s].append(int(t))
    for s in range(B):
        base = users[s] if s % 2 else new[s]
        assert seqs[s][len(base):] == m.reference_greedy(base, len(seqs[s]) - len(base))
    for s in range(B):
        g.release_slot(kv, s)
    assert not b.obm.refs and [f + e for f, e in zip(b.obm.free_counts(), b.obm.evictable_counts())] == full


def test_connector_shard_map_exposes_the_block_map_and_local_pages():
    """Stage 12 x 14: the KV connector gets the owner-aware map (physical / admit) and the backend's local pools"""
    m, g, kv = _sharded(C=2)
    b = g._binding(kv)
    assert g.connector_shard_map(kv) == dict(C=2, nb_local=b.obm.nb_local, bpu=2, num_blocks=40, map=b.obm)
    m.latent_pages = lambda kv_: {3: "local-pool-L3"}  # the mock has no latent pages; the generator passes them through
    assert g.latent_pages(kv) == {3: "local-pool-L3"}
    # a hit admitted by the connector before its prefill row: the row's keep columns find the admitted blocks
    phys = b.obm.admit([(4, 0), (8, 1), (12, 2)])
    assert [p % 2 for p in phys] == [0, 0, 1]
    g.prefill_forward(_prefill_rows([prompt(200, seed=5)]), page_table=_row([4, 8, 12, 16], b.blocks_per_seq),
                      kv_cache=kv, prompt_lens=[200], slots=[0], sampling_params=[SamplingParams()])  # fmt: skip
    assert b.phys[0, :3].tolist() == phys and b.pt_shadow[0, :4].tolist() == [4, 8, 12, 16]
    _owner_ok(b)
    m2, g2 = make("dense_exact", max_batch=4)
    m2.dsa_shard = 1
    kv2 = g2.allocate_kv_cache(4, 1024, num_blocks=40, null_block=0)
    assert g2.connector_shard_map(kv2) is None  # the replicated pool: unchanged


def test_null_block_never_on_a_covered_column(monkeypatch):
    """lmshard N1: a non-owner chip's local block 0 takes KV-connector load rows (scratch); the generator refuses any
    table that would let a covered column read a null block"""
    m, g, kv = _sharded(C=2)
    b = g._binding(kv)
    real = OB.OwnerBlockMap.set_row

    def null_in_live(self, slot, row, covered, keep=0):
        out = real(self, slot, row, covered, keep)
        return [self.null(1) if j == 1 else v for j, v in enumerate(out)]  # column 1 (covered) on its owner's null block

    monkeypatch.setattr(OB.OwnerBlockMap, "set_row", null_in_live)
    with pytest.raises(RuntimeError, match="null-block rule"):
        g.prefill_forward(_prefill_rows([prompt(100, seed=9)]), page_table=_row([4, 8], b.blocks_per_seq), kv_cache=kv,
                          prompt_lens=[100], slots=[0], sampling_params=[SamplingParams()])  # fmt: skip


@pytest.mark.parametrize("C", [2, 4])
def test_unchanged_rows_skip_the_page_table_rewrite(C):
    """Stage 9 cf3g (n29h7p2 2026-10-23: ~35 ms of host per reloaded decode step at 1M context in _write_page_rows'
    Python passes): a reloaded decode step whose rows equal the published ones does no page-table work (the
    page_rows_unchanged counter, no refresh, the physical map untouched) and decodes the reference tokens; a row that
    grows by a block takes the full path (one refresh, owner-aware); a position past the coverage still raises."""
    m, g, kv = _sharded(C=C)
    b = g._binding(kv)
    bps = b.blocks_per_seq
    u1, u2 = prompt(120, seed=70), prompt(60, seed=71)
    rows = {1: _row([3, 7], bps)[0], 2: _row([12], bps)[0]}
    first = g.prefill_forward(_prefill_rows([u1, u2]), page_table=torch.stack([rows[1], rows[2]]), kv_cache=kv,
                              prompt_lens=[120, 60], slots=[1, 2], sampling_params=[SamplingParams()] * 2)  # fmt: skip
    seqs = {1: u1 + [int(first[0])], 2: u2 + [int(first[1])]}
    sp = [SamplingParams()] * 2
    phys0, before = b.phys.clone(), dict(g.counters)
    for _ in range(3):  # positions 120..122 / 60..62: inside the 2 / 1 published blocks
        sl = [1, 2]
        out = g.decode_forward(torch.tensor([seqs[s][-1] for s in sl]), torch.tensor([len(seqs[s]) - 1 for s in sl]),
                               page_table=torch.stack([rows[s] for s in sl]), kv_cache=kv, slots=sl, sampling_params=sp)  # fmt: skip
        for s, t in zip(sl, out.tolist()):
            seqs[s].append(int(t))
    assert g.counters["page_rows_unchanged"] - before["page_rows_unchanged"] == 3
    assert g.counters["page_table_refresh"] == before["page_table_refresh"] and torch.equal(b.phys, phys0)
    # slot 2 crosses its first block (position 64 needs a second block): vLLM appends one -> the full path
    while len(seqs[2]) - 1 < 64:
        out = g.decode_forward(torch.tensor([seqs[2][-1]]), torch.tensor([len(seqs[2]) - 1]), page_table=rows[2][None],
                               kv_cache=kv, slots=[2], sampling_params=sp[:1])  # fmt: skip
        seqs[2].append(int(out[0]))
    with pytest.raises(ValueError):  # position 64 outside the published coverage (1 block): refused as before
        g.decode_forward(torch.tensor([seqs[2][-1]]), torch.tensor([len(seqs[2]) - 1]), page_table=rows[2][None],
                         kv_cache=kv, slots=[2], sampling_params=sp[:1])  # fmt: skip
    rows[2] = _row([12, 9], bps)[0]
    r0 = g.counters["page_table_refresh"]
    out = g.decode_forward(torch.tensor([seqs[2][-1]]), torch.tensor([len(seqs[2]) - 1]), page_table=rows[2][None],
                           kv_cache=kv, slots=[2], sampling_params=sp[:1])  # fmt: skip
    seqs[2].append(int(out[0]))
    assert g.counters["page_table_refresh"] == r0 + 1 and b.pt_shadow[2, :2].tolist() == [12, 9]
    _owner_ok(b)
    assert seqs[1][120:] == m.reference_greedy(u1, len(seqs[1]) - 120)
    assert seqs[2][60:] == m.reference_greedy(u2, len(seqs[2]) - 60)
