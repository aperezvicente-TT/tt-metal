# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host model of GLM53_PF_BQ_HEAD_BLOCK (MlaPrefill._bq in head blocks, 2026-10-23 08:02Z): the B-q split-K
attention's head bookkeeping, mirrored in torch over n simulated chips. Full form: every chip gathers all H = n h_r
heads (rank order: chip r's heads at [h_r r, h_r (r + 1))), computes its owner partial (o_c, m_c, l_c) per head, the
owner merge (tt/dsa_shard.py bq_merge math) and a head-dim reduce-scatter (chip r keeps block r). Block form: block j
gathers local heads [j p, (j + 1) p) of every chip (p = hb / n; chip r's at [p r, p (r + 1))), merges and
reduce-scatters them (chip r keeps its own p heads), and the blocks concatenate in local head order. Checks: every chip
ends with exactly its own heads in local order, and the result is BITWISE the full form (per-head math, the same owner
sum order), for TP2 / TP8 head counts and every legal block size. The per-head math is computed head by head here
(the device's sparse_sdpa_stats is per head; the device bitwise A/B of the knob is the pair unit's job)."""
from __future__ import annotations

import pytest
import torch


def owner_partials(q_heads: torch.Tensor, chip: int, n: int, keys: torch.Tensor, scale: float):
    """chip `chip`'s partial for every head of q_heads [Hb, D] over the keys it owns (keys [K, D] split by chip):
    (o_c [Hb, D] normalised, m_c [Hb], l_c [Hb])"""
    k = keys[chip::n]
    outs = []
    for h in range(q_heads.shape[0]):  # per head (the device kernel's per-head math does not depend on the head count;
        s = (q_heads[h : h + 1] @ k.T) * scale  # a batched torch matmul's CPU blocking would, which is not modelled)
        m = s.max(-1).values
        e = torch.exp(s - m[:, None])
        lsum = e.sum(-1)
        outs.append(((e @ k) / lsum[:, None], m / scale, lsum))
    return tuple(torch.cat(x) for x in zip(*outs))


def merge(parts, scale):
    """the owner merge, head by head (torch's vectorised exp / column sums change bits with the column count; the
    device's per-element ops do not)"""
    hb = parts[0][0].shape[0]
    out = [torch.empty_like(o) for o, _, _ in parts]
    for h in range(hb):
        ms = torch.stack([m[h : h + 1] for _, m, _ in parts])  # [n, 1]
        ls = torch.stack([lv[h : h + 1] for _, _, lv in parts])
        big = ms.max(0).values
        w = torch.exp((ms - big) * scale) * ls
        w = w / w.sum(0)
        for c, (o, _, _) in enumerate(parts):
            out[c][h : h + 1] = o[h : h + 1] * w[c][:, None]
    return out  # per owner, weighted


def reduce_scatter_heads(weighted_per_owner, n):
    """the head-dim reduce-scatter: sum over owners in owner order, chip r keeps head block r"""
    tot = weighted_per_owner[0].clone()
    for t in weighted_per_owner[1:]:
        tot = tot + t
    hb = tot.shape[0] // n
    return [tot[r * hb : (r + 1) * hb] for r in range(n)]


def b_q(local_q, keys, n, scale, hb=0):
    """local_q: per chip [h_r, D] -> per chip [h_r, D] (its own heads, merged over the owners)"""
    h_r = local_q[0].shape[0]
    if not hb or hb >= h_r * n:
        blocks = [(0, h_r)]
    else:
        p = hb // n
        assert p * n == hb and h_r % p == 0
        blocks = [(j, j + p) for j in range(0, h_r, p)]
    per_chip = [[] for _ in range(n)]
    for a, b in blocks:
        q_all = torch.cat([q[a:b] for q in local_q])  # gather: rank order
        parts = [owner_partials(q_all, c, n, keys, scale) for c in range(n)]
        mine = reduce_scatter_heads(merge(parts, scale), n)
        for r in range(n):
            per_chip[r].append(mine[r])
    return [torch.cat(x) for x in per_chip]


@pytest.mark.parametrize("n,heads", [(2, 64), (8, 64)])
def test_head_blocks_bitwise_and_owned(n, heads):
    torch.manual_seed(0)
    h_r, d, scale = heads // n, 64, 0.125
    local_q = [torch.randn(h_r, d) for _ in range(n)]
    keys = torch.randn(48 * n, d)
    full = b_q(local_q, keys, n, scale)
    # every chip holds exactly its own heads (vs a direct per-head reference)
    for r in range(n):
        ref = []
        for h in range(h_r):
            q = local_q[r][h : h + 1]
            parts = [owner_partials(q, c, n, keys, scale) for c in range(n)]
            ref.append(sum(merge(parts, scale)[1:], merge(parts, scale)[0]))
        assert torch.allclose(full[r], torch.cat(ref), atol=1e-5)
    # legal on device: a multiple of the chips dividing the heads AND of 32 (sparse_sdpa's heads are tile rows); the
    # host model also checks the smaller blocks (bookkeeping only)
    legal = [hb for hb in range(n, heads + 1, n) if h_r % (hb // n) == 0 and hb < heads]
    assert 32 in legal and 16 in legal
    for hb in legal:
        blk = b_q(local_q, keys, n, scale, hb=hb)
        assert all(torch.equal(a, b) for a, b in zip(blk, full)), hb


def test_illegal_block_sizes_rejected():
    local_q = [torch.randn(32, 8) for _ in range(2)]
    with pytest.raises(AssertionError):
        b_q(local_q, torch.randn(16, 8), 2, 0.1, hb=3)  # not a multiple of the chips
    with pytest.raises(AssertionError):
        b_q(local_q, torch.randn(16, 8), 2, 0.1, hb=2 * 3)  # 3 local heads per block does not divide 32
