# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-replay store A (GLM53_PF_KDA_SNAPSHOT / the plugin's GLM53_KDA_CKPT_IN_REPLAY): a prefill row carrying
snapshot_at (a storing request's checkpoint cut inside its chunk) must take the per-user path, whose chunk writes the
KDA state at the cut. The packed step (_packed_step) and the batched admission (_admit_batch) have no per-segment
snapshot: before this fix a storing row packed or batched with other rows ran there, the replay kept no snapshot, and
the plugin's export found it not ready (no checkpoint for that prompt). Host-only, the CPU mock stack."""

from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    PF_BATCH_ENV,
    PF_PACKED_CAPS_ENV,
    PF_PACKED_ENV,
    RecordingMockModel,
)
from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY

MAXSEQ = 4096


def _serve(lens, snapshot_at):
    """fresh device-sampled prefill of rows `lens` in ONE call (slot i = row i) with `snapshot_at` {row: cut}"""
    m = RecordingMockModel(("dense_kda", "sparse_kda"), max_batch=8, seed=1, dsa_mode="dense_exact")
    g = G.Glm53Generator(m, max_batch=8, max_seq_len=MAXSEQ, stop_token_ids=())
    kv = g.allocate_kv_cache(m.max_batch, MAXSEQ)
    g.warmup(kv)
    bps = G.blocks_per_seq(MAXSEQ)
    pt = torch.arange(m.max_batch * bps).reshape(m.max_batch, bps)
    toks = torch.zeros(len(lens), max(lens), dtype=torch.int64)
    gen = torch.Generator().manual_seed(7)
    for i, n in enumerate(lens):
        toks[i, :n] = torch.randint(0, 1000, (n,), generator=gen)
    out = g.prefill_forward(toks, page_table=pt[: len(lens)], kv_cache=kv, prompt_lens=list(lens),
                            sampling_params=[GREEDY] * len(lens), snapshot_at=snapshot_at)  # fmt: skip
    return m, g, kv, [int(v) for v in out]


@pytest.mark.parametrize("path", ["packed", "batched"])
def test_snapshot_row_leaves_the_packed_and_batched_groups(monkeypatch, path, evidence):
    """three fresh rows, the middle one storing (cut 128 inside its 200 tokens): it runs per user with snap_at and its
    snapshot is held for export; the other two still form the packed / batched group"""
    monkeypatch.setenv("GLM53_PF_KDA_SNAPSHOT", "1")
    if path == "packed":
        monkeypatch.setenv(PF_PACKED_ENV, "2048,4")
        monkeypatch.setenv(PF_PACKED_CAPS_ENV, "continuation,no_head")
        monkeypatch.delenv(PF_BATCH_ENV, raising=False)
        lens = [100, 200, 60]
    else:
        monkeypatch.setenv(PF_BATCH_ENV, "512,2")
        monkeypatch.delenv(PF_PACKED_ENV, raising=False)
        monkeypatch.delenv(PF_PACKED_CAPS_ENV, raising=False)
        lens = [100, 200, 60]
    m, g, kv, first = _serve(lens, {1: 128})
    groups = [ev for ev in m.events if ev["event"] in ("prefill_packed", "prefill_batch")]
    per_user = [ev for ev in m.events if ev["event"] == "prefill_chunk" and ev["slot"] == 1]
    assert [ev["slots"] for ev in groups] == [[0, 2]], groups  # the group holds the two plain rows only
    assert [ev["snap_at"] for ev in per_user] == [128], per_user  # the storing row's chunk carries the cut
    assert m.kda_snapshot_holds(kv, 1, 128)
    assert g.counters.get("kda_snapshot_chunks", 0) == 1
    # the snapshot is the state after the first 128 tokens: a separate 128-token prefill's live state
    toks = torch.zeros(1, 200, dtype=torch.int64)
    gen = torch.Generator().manual_seed(7)
    for n in lens[:2]:  # replay the generator's draws up to row 1
        toks[0, :n] = torch.randint(0, 1000, (n,), generator=gen)
    ref = _ref_state(toks[0, :128])
    snap = m.export_kda_state(kv, 1, n_tokens=128, source="snapshot")
    for a, b in zip(snap, ref):
        assert torch.equal(a["S"], b["S"]) and torch.equal(a["C"], b["C"]), a["L"]
    evidence(path=path, groups=[ev["slots"] for ev in groups], per_user=[(ev["start"], ev["n"]) for ev in per_user],
             first=first)  # fmt: skip


def _ref_state(ids):
    """the live KDA state of a slot that prefilled exactly `ids` (one fresh request, per-user path)"""
    m = RecordingMockModel(("dense_kda", "sparse_kda"), max_batch=8, seed=1, dsa_mode="dense_exact")
    g = G.Glm53Generator(m, max_batch=8, max_seq_len=MAXSEQ, stop_token_ids=())
    kv = g.allocate_kv_cache(m.max_batch, MAXSEQ)
    g.warmup(kv)
    bps = G.blocks_per_seq(MAXSEQ)
    pt = torch.arange(m.max_batch * bps).reshape(m.max_batch, bps)
    g.prefill_forward(ids.reshape(1, -1), page_table=pt[1:2], kv_cache=kv, prompt_lens=[int(ids.numel())],
                      empty_slots=[1], sampling_params=[GREEDY])  # fmt: skip
    return m.export_kda_state(kv, 1)


def test_no_snapshot_rows_keep_the_groups(monkeypatch):
    """control: without snapshot_at the same three rows form one packed group"""
    monkeypatch.setenv("GLM53_PF_KDA_SNAPSHOT", "1")
    monkeypatch.setenv(PF_PACKED_ENV, "2048,4")
    monkeypatch.setenv(PF_PACKED_CAPS_ENV, "continuation,no_head")
    monkeypatch.delenv(PF_BATCH_ENV, raising=False)
    m, g, _kv, _ = _serve([100, 200, 60], None)
    assert [ev["slots"] for ev in m.events if ev["event"] == "prefill_packed"] == [[0, 1, 2]]
    assert g.counters.get("kda_snapshot_chunks", 0) == 0
