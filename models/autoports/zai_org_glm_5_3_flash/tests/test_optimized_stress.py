# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stress / repeated-run coverage of the OPTIMIZED decoder: Stage 2's `test_fused_stress.test_stress` protocol (64-step
traced B=4 decode vs the goldens with an inactive slot whose carried state must stay bitwise unchanged, the whole
sequence repeated from a pre-capture snapshot -> active rows bitwise, 50 identical replays -> active rows bitwise,
3 identical 2-chunk non-aligned prefills -> bitwise) run unmodified against `OptimizedDecoder` (loader swapped by an
autouse fixture; rows tagged `impl: optimized`), and `test_stress_batch`: the same decode protocol at B = 8 / 32."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_fused_stress as s2
from models.autoports.zai_org_glm_5_3_flash.tests.common import PCC_BAR, check, pcc, record, streams4
from models.autoports.zai_org_glm_5_3_flash.tests.test_optimized_decoder import _load_optimized_checked


@pytest.fixture(autouse=True)
def _optimized_path(monkeypatch):
    monkeypatch.setattr(s2, "load_impl", lambda cfg, layer, mesh: _load_optimized_checked(cfg, layer, mesh))
    monkeypatch.setattr(s2, "IMPL", "optimized")


mesh = s2.mesh
test_stress = s2.test_stress


@pytest.mark.parametrize("layer,batch", [(0, 8), (4, 8), (3, 8), (0, 32), (4, 32), (3, 32)])
def test_stress_batch(mesh, layer, batch):
    """Stage 2's stress protocol at B = 8 / 32 on the optimized decoder (Stage 3 self-review R9): B users with mixed
    golden prompts / start positions in all slots, slot 2 inactive (cur_pos -1) for the first half, STEPS consecutive
    replays of one captured trace vs the fp32 goldens (per step: batch PCC >= min(0.995, prefill path of the same rows -
    0.002), Stage 2's accepted batch criterion; the plain value is recorded), slot 2's KDA state bitwise unchanged while
    inactive, the whole sequence repeated from the pre-capture snapshot (active rows bitwise), REPLAYS replays of one
    step from the same state (active rows bitwise)."""
    import random

    cfg = s2.hf_config()
    dec = _load_optimized_checked(cfg, layer, mesh)
    kind = dec.spec.kind
    steps, replays = s2.STEPS, s2.REPLAYS
    max_seq = 4096 + 64 if dec.is_kda else s2.DSA_DENSE_EXACT_MAX_KV
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=max_seq)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    dec.prepare_prefill(cache)
    rng = random.Random(4242 + layer + batch)
    prompts = [512, 2048, 128, 2048] + ([4096] if dec.is_kda else [])
    users = []  # slot -> (golden prompt, first decode position)
    for slot in range(batch):
        T = prompts[slot % len(prompts)]
        hi = (T if dec.is_kda else min(T, s2.DSA_DENSE_EXACT_MAX_KV)) - steps
        users.append((T, rng.randint(1, hi)))
    g = {T: s2.golden(layer, T) for T in {T for T, _ in users}}
    # prefill-path rows of every user (single-user cache, same prompt through position n + steps - 1)
    ref_cache = dec.allocate_cache(max_batch=1, max_seq_len=max_seq)
    if not dec.is_kda:
        dec.make_page_table(ref_cache)
    pre_rows = []
    for T, n in users:
        y = streams4(dec.prefill_forward(g[T]["streams_in"][:, : n + steps], cache=ref_cache, user_slot=0))
        pre_rows.append(y[:, n : n + steps])
    del ref_cache
    for slot, (T, n) in enumerate(users):
        dec.prefill_forward(g[T]["streams_in"][:, :n], cache=cache, user_slot=slot)
    inactive_until = steps // 2

    def step_inputs(step):
        rows, pos, off = [], [], []
        for slot, (T, n) in enumerate(users):
            if slot == 2 and step < inactive_until:
                rows.append(torch.zeros(1, 1, 4, 4096))
                pos.append(-1)
                off.append(None)
                continue
            o = step - inactive_until if slot == 2 else step
            rows.append(g[T]["streams_in"][:, n + o : n + o + 1])
            pos.append(n + o)
            off.append(o)
        return torch.cat(rows, dim=1), pos, off

    td = s2.TracedDecode(dec, cache, pt, batch=batch)
    rows, pos, _ = step_inputs(0)
    td.set_inputs(rows, pos)
    td.capture()
    snap = td.snap
    slot2_before = ttnn.to_torch(cache.kda.recurrent[2]) if dec.is_kda else None

    def run_sequence():
        outs, steps_rows, slot2_mid = [], [], None
        for step in range(steps):
            rows, pos, off = step_inputs(step)
            td.set_inputs(rows, pos)
            out = td.step()
            act = [i for i, p in enumerate(pos) if p >= 0]
            outs.append((out, act))
            ref = torch.cat([g[users[i][0]]["streams_out"][:, pos[i] : pos[i] + 1] for i in act], dim=1)
            pre = torch.cat([pre_rows[i][:, off[i] : off[i] + 1] for i in act], dim=1)
            steps_rows.append((pcc(out[:, act], ref), pcc(pre, ref)))
            if dec.is_kda and step == inactive_until - 1:
                slot2_mid = ttnn.to_torch(cache.kda.recurrent[2])
        return outs, steps_rows, slot2_mid

    outs1, rows1, slot2_mid = run_sequence()
    dec.restore_state(cache, snap)
    outs2, _, _ = run_sequence()
    seq_bitwise = all(torch.equal(a[0][:, a[1]], b[0][:, b[1]]) for a, b in zip(outs1, outs2))
    slot2_unchanged = bool(torch.equal(slot2_mid, slot2_before)) if dec.is_kda else None
    dec.restore_state(cache, snap)
    rows, pos, _ = step_inputs(0)
    td.set_inputs(rows, pos)
    act0 = [i for i, p in enumerate(pos) if p >= 0]
    first, replay_bitwise = None, True
    for _ in range(replays):
        dec.restore_state(cache, snap)
        out = td.step()
        first = out if first is None else first
        replay_bitwise &= bool(torch.equal(out[:, act0], first[:, act0]))
    td.release()
    fails = [i for i, (p, q) in enumerate(rows1) if p < min(PCC_BAR, q - 0.002)]
    row = dict(
        impl="optimized",
        test="stress_batch",
        layer_kind=kind,
        layer=layer,
        batch=batch,
        steps=steps,
        users=users,
        min_step_pcc=min(p for p, _ in rows1),
        mean_step_pcc=sum(p for p, _ in rows1) / len(rows1),
        min_step_prefill_path_pcc=min(q for _, q in rows1),
        batch_criterion="pcc >= min(0.995, prefill_path_pcc - 0.002)",
        failing_steps=fails,
        sequence_repeat_bitwise_equal=seq_bitwise,
        inactive_slot_state_unchanged=slot2_unchanged,
        replays=replays,
        replays_bitwise_equal=replay_bitwise,
    )
    record(row, name="stress.jsonl")
    bad = fails or not (seq_bitwise and replay_bitwise) or slot2_unchanged is False
    check([row] if bad else [], "stress / repeated runs at B = 8 / 32")
