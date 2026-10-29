# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stress / repeated-run coverage of the fused decoder (real weights, one layer per kind).

* long traced decode: B = 4 users (different prompts and positions, one slot inactive for the first half), STEPS
  consecutive replays of ONE captured trace with refreshed streams/positions; every step's active rows vs the fp32
  goldens (batch PCC >= the bar), the inactive slot's carried state bitwise unchanged while inactive;
* repeat: the whole STEPS-step sequence re-run from a device snapshot taken before it -> every step's active rows
  bitwise equal;
* REPLAYS replays of the same step from the same restored state -> active rows bitwise equal.
  (An inactive slot's output row is don't-care by the decode contract: MLA skips its attention, so the row holds
  whatever the trace buffer held; whole-output equality is recorded for information.)
* REPEAT_PREFILL identical prefills of a 2,100 / 2,000-token request (2 chunks, non-aligned tail) -> bitwise equal.
Rows -> $GLM53_STAGE_EVIDENCE/stress.jsonl. `GLM53_DECODER_IMPL=functional` runs the same protocol on the Stage 1
decoder (baseline for the per-step PCC).
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import PCC_BAR, check, golden, hf_config, pcc, record
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import IMPL, load_impl, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tests.test_functional_decoder import TracedDecode
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV

STEPS = int(os.environ.get("GLM53_STRESS_STEPS", "64"))
REPLAYS = int(os.environ.get("GLM53_STRESS_REPLAYS", "50"))
REPEAT_PREFILL = int(os.environ.get("GLM53_STRESS_PREFILLS", "3"))


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh(trace_region_size=300_000_000)
    yield m
    ttnn.close_mesh_device(m)


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_stress(mesh, layer):
    cfg = hf_config()
    dec = load_impl(cfg, layer, mesh)
    kind = dec.spec.kind
    batch = 4
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=4096 + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    dec.prepare_prefill(cache)
    users = [(512, 200), (2048, 1500), (128, 40), (2048, 900)]  # slot -> (golden prompt, first decode position)
    for slot, (T, n) in enumerate(users):
        dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=slot)
    inactive_until = STEPS // 2  # slot 2 inactive (cur_pos -1) for the first half: its state must not move
    g = {T: golden(layer, T) for T in {T for T, _ in users}}

    def step_inputs(step):
        rows, pos = [], []
        for slot, (T, n) in enumerate(users):
            if slot == 2 and step < inactive_until:
                rows.append(torch.zeros(1, 1, 4, 4096))
                pos.append(-1)
                continue
            p = n + (step - inactive_until if slot == 2 else step)
            rows.append(g[T]["streams_in"][:, p : p + 1])
            pos.append(p)
        return torch.cat(rows, dim=1), pos

    td = TracedDecode(dec, cache, pt, batch=batch)
    rows, pos = step_inputs(0)
    td.set_inputs(rows, pos)
    td.capture()
    snap = td.snap  # taken before the capture: no device buffer may be allocated (and kept) while the trace is live
    slot2_before = ttnn.to_torch(cache.kda.recurrent[2]) if dec.is_kda else None

    def run_sequence():
        outs, pccs = [], []
        for step in range(STEPS):
            rows, pos = step_inputs(step)
            td.set_inputs(rows, pos)
            out = td.step()
            act = [i for i, p in enumerate(pos) if p >= 0]
            outs.append((out, act))
            ref = torch.cat([g[users[i][0]]["streams_out"][:, pos[i] : pos[i] + 1] for i in act], dim=1)
            pccs.append(pcc(out[:, act], ref))
            if dec.is_kda and step == inactive_until - 1:
                outs.append(("slot2_state", ttnn.to_torch(cache.kda.recurrent[2])))
        return outs, pccs

    outs1, pccs1 = run_sequence()
    dec.restore_state(cache, snap)  # MLA: the latent writes of the sequence are rewritten with the same values
    outs2, pccs2 = run_sequence()
    seq_bitwise = all(
        torch.equal(a[1], b[1]) if isinstance(a[1], torch.Tensor) else torch.equal(a[0][:, a[1]], b[0][:, b[1]])
        for a, b in zip(outs1, outs2)
    )
    seq_full_bitwise = all(torch.equal(a[0], b[0]) for a, b in zip(outs1, outs2) if not isinstance(a[1], torch.Tensor))
    slot2_unchanged = None  # MLA keeps no per-slot recurrent state (its inactive slot skips the cache write)
    if dec.is_kda:
        s2 = [o[1] for o in outs1 if isinstance(o[1], torch.Tensor)][0]
        slot2_unchanged = bool(torch.equal(s2, slot2_before))

    # REPLAYS replays of step 0 from the same restored state
    dec.restore_state(cache, snap)
    rows, pos = step_inputs(0)
    td.set_inputs(rows, pos)
    act0 = [i for i, p in enumerate(pos) if p >= 0]
    first = None
    replay_bitwise = replay_full_bitwise = True
    for _ in range(REPLAYS):
        dec.restore_state(cache, snap)
        out = td.step()
        first = out if first is None else first
        replay_bitwise &= bool(torch.equal(out[:, act0], first[:, act0]))
        replay_full_bitwise &= bool(torch.equal(out, first))
    td.release()

    # repeated identical prefills of a 2-chunk request with a non-aligned tail (slot 0, fresh request each time)
    n_req = 2100 if dec.is_kda else 2000
    req = golden(layer, 4096 if dec.is_kda else 2048)["streams_in"][:, :n_req]
    ys = [dec.prefill_forward(req, cache=cache, user_slot=0) for _ in range(REPEAT_PREFILL)]
    prefill_bitwise = all(torch.equal(y, ys[0]) for y in ys[1:])

    row = dict(
        impl=IMPL,
        test="stress",
        layer_kind=kind,
        layer=layer,
        batch=batch,
        steps=STEPS,
        min_step_pcc=min(pccs1),
        mean_step_pcc=sum(pccs1) / len(pccs1),
        sequence_repeat_bitwise_equal=seq_bitwise,
        sequence_repeat_full_output_bitwise_equal=seq_full_bitwise,
        inactive_slot_state_unchanged=slot2_unchanged,
        replays=REPLAYS,
        replays_bitwise_equal=replay_bitwise,
        replays_full_output_bitwise_equal=replay_full_bitwise,
        repeated_prefills=REPEAT_PREFILL,
        prefill_tokens=n_req,
        prefills_bitwise_equal=prefill_bitwise,
    )
    record(row, name="stress.jsonl")
    bad = row["min_step_pcc"] < PCC_BAR or not (seq_bitwise and replay_bitwise and prefill_bitwise)
    bad = bad or slot2_unchanged is False
    check([row] if bad else [], "stress / repeated runs")
