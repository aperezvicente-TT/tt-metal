# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight tests of the GLM-5.3-Flash FUSED decoder (`tt/fused_decoder.py`).

Stage 1's gates are reused verbatim: the test functions of `test_functional_decoder.py` (golden prompts of every
captured layer, non-aligned prefill lengths / continuation / traced follow-on decode, batched paged decode with a
permuted page table, prefill under a live decode trace, decode between the chunks of an interleaved prefill with
inactive slots, NaN / Inf / huge values in an inactive slot's input row) are imported unmodified and run against
`FusedDecoder` (the module-level loader of the Stage 1 module is swapped by an autouse fixture, so every test exercises
the fused path; `_load_fused_checked` asserts it). Two adaptations:
the router recorder keeps the fused router's (top-8 ids, unbiased scores) and rebuilds the dense routing rows on the
host for the expert-agreement metric, and every evidence row is tagged `impl: fused`.

Fused-only tests: `test_batched_paged_decode_rows` (Stage 1's batched paged decode gate at the row counts where the
fused decode switches programs: 2, 3, 5, 16, 17), `test_prefill_slot_isolation_nonfinite` (a prefill reads / writes its
slot of the batched KDA state independently of the values held by the other slots), and `test_fused_vs_functional`
(the unfused (Stage 1) and fused layer on identical inputs: prompt-128 prefill, traced decode0, B=4 traced decode with
distinct users; PCC fused vs functional and both vs the fp32 goldens: graph-fusing "verify every rewrite" on device,
whole layer).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_functional_decoder as s1
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    PCC_BAR,
    check,
    golden,
    int_tensor,
    load_decoder,
    pcc,
    record,
    streams4,
    to_dev,
    to_host,
)
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import plan_prefill_chunks
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder

mesh = s1.mesh  # module-scoped device fixture (same as Stage 1)
cfg = s1.cfg


class FusedRecordingRouter:
    """Wraps FusedRouter: records (ids [1,1,P,8] uint16, unbiased scores [1,1,P,288] bf16) device tensors of the calls
    made while `recording` (prefill); no device reads, so it is trace-safe."""

    def __init__(self, router):
        self.router = router
        self.num_experts = router.num_experts
        self.recording = False
        self.log = []

    def __call__(self, x):
        out = self.router(x)
        if self.recording:
            self.log.append(out)
        return out


def fused_routing_rows(rec, seq_len: int, n_experts: int, align: int) -> torch.Tensor:
    """Dense [T, E] host routing rows (the selected experts' unbiased scores, 0 elsewhere) from the recorded calls:
    `.topk(8)` of a row returns exactly the selected expert set."""
    rows = []
    for (off, n, phys), (idx, scores) in zip(plan_prefill_chunks(seq_len, align), rec.log):
        ids = ttnn.to_torch(idx).reshape(phys, -1)[:n].long()
        sc = ttnn.to_torch(scores).float().reshape(phys, n_experts)[:n]
        dense = torch.zeros(n, n_experts)
        dense.scatter_(1, ids, sc.gather(1, ids).clamp_min(1e-12))
        rows.append(dense)
    return torch.cat(rows)


def _record_fused(row: dict, name: str = "pcc_real_weights.jsonl"):
    return record({"impl": "fused", **row}, name=name)


def _load_fused_checked(cfg_, layer, mesh_):
    dec = load_fused_decoder(cfg_, layer, mesh_)
    assert isinstance(dec, FusedDecoder), type(dec)
    return dec


@pytest.fixture(autouse=True)
def _fused_path(monkeypatch):
    monkeypatch.setattr(s1, "load_decoder", _load_fused_checked)
    monkeypatch.setattr(s1, "RecordingRouter", FusedRecordingRouter)
    monkeypatch.setattr(s1, "_routing_rows", fused_routing_rows)
    monkeypatch.setattr(s1, "record", _record_fused)


# Stage 1 gates, unmodified, on the fused decoder
test_golden_prompts = s1.test_golden_prompts
test_prefill_lengths = s1.test_prefill_lengths
test_batched_paged_decode = s1.test_batched_paged_decode
test_prefill_under_live_trace = s1.test_prefill_under_live_trace
test_decode_between_prefill_chunks = s1.test_decode_between_prefill_chunks
test_inactive_slot_nonfinite_input = s1.test_inactive_slot_nonfinite_input


# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer", [0, 4, 3])
def test_fused_vs_functional(mesh, cfg, layer):
    """Unfused (Stage 1) vs fused layer on identical inputs: prompt-128 prefill, traced decode0 (B=1) and a B=4 traced
    decode of 4 users at different positions. Gate: fused vs golden >= the bar and fused vs functional >= 0.999."""
    kind = s1.KIND_OF[layer]
    g = golden(layer, 128)
    outs = {}
    for impl, loader in (("functional", load_decoder), ("fused", load_fused_decoder)):
        dec = loader(cfg, layer, mesh)
        cache = dec.allocate_cache(max_batch=1, max_seq_len=128 + 64)
        pt = None if dec.is_kda else dec.make_page_table(cache)
        y = dec.prefill_forward(g["streams_in"], cache=cache, user_slot=0).reshape(1, -1, 4, 4096)
        td = s1.TracedDecode(dec, cache, pt, batch=1)
        td.set_inputs(g["decode0_streams_in"], [128])
        td.capture()
        d0 = td.step()
        td.release()
        # B=4: users = prefixes of the 2048 golden at different lengths
        g2 = golden(layer, 2048)
        c4 = dec.allocate_cache(max_batch=4, max_seq_len=2048 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
        pt4 = None if dec.is_kda else dec.make_page_table(c4)
        lens = [100, 700, 1500, 333]
        for slot, n in enumerate(lens):
            dec.prefill_forward(g2["streams_in"][:, :n], cache=c4, user_slot=slot)
        td4 = s1.TracedDecode(dec, c4, pt4, batch=4)
        td4.set_inputs(torch.cat([g2["streams_in"][:, n : n + 1] for n in lens], dim=1), lens)
        td4.capture()
        d4 = td4.step()
        td4.release()
        outs[impl] = (y, d0, d4)
        del dec, cache, c4
    ref4 = torch.cat([golden(layer, 2048)["streams_out"][:, n : n + 1] for n in [100, 700, 1500, 333]], dim=1)
    f, u = outs["fused"], outs["functional"]
    row = dict(
        impl="fused",
        test="fused_vs_functional",
        layer_kind=kind,
        layer=layer,
        prefill128_pcc_fused_vs_functional=pcc(f[0], u[0]),
        prefill128_pcc_fused=pcc(f[0], g["streams_out"]),
        prefill128_pcc_functional=pcc(u[0], g["streams_out"]),
        decode0_pcc_fused_vs_functional=pcc(f[1], u[1]),
        decode0_pcc_fused=pcc(f[1], g["decode0_streams_out"]),
        decode0_pcc_functional=pcc(u[1], g["decode0_streams_out"]),
        decode_b4_pcc_fused_vs_functional=pcc(f[2], u[2]),
        decode_b4_pcc_fused=pcc(f[2], ref4),
        decode_b4_pcc_functional=pcc(u[2], ref4),
    )
    record(row, name="fused_vs_functional.jsonl")
    bad = [k for k, v in row.items() if k.endswith("_fused") and isinstance(v, float) and v < PCC_BAR]
    bad += [k for k, v in row.items() if k.endswith("fused_vs_functional") and isinstance(v, float) and v < 0.999]
    check(bad, "fused path below the bar or off the functional path")


# ---------------------------------------------------------------------------------------------------------------------
# Row-count dispatch coverage (fused only): the fused decode switches programs by decode rows (stacked mHC projection
# up to 2 rows, fused P2a mHC up to 4 / stream-major from 5, sigmoid_gated_rms_norm from 2 with users padded to 32, one
# 64-head flash-MLA call from 16, fused_experts bound 8 x rows). Stage 1's batched paged decode gate (same body, same
# per-user criterion) at the boundary row counts the Stage 1 parametrization does not reach.
@pytest.mark.parametrize("layer,batch", [(4, 2), (4, 3), (4, 5), (4, 17), (0, 2), (0, 5), (3, 2), (3, 5), (3, 16)])
def test_batched_paged_decode_rows(mesh, cfg, layer, batch):
    """Stage 1's `test_batched_paged_decode` body (same users / seeds, shuffled slots, permuted page table, 4 traced
    steps, same per-user criterion: decode >= min(0.995, prefill path of the row - 0.002)). The per-step batch PCC is
    gated the same way, against the prefill path of the same rows: with 2-5 users a single token at the bfloat4_b
    weights floor sets the batch PCC (L4 prompt 128 position 101: the bfp4 control with the golden routing is 0.9887,
    Stage 1 F7), which no implementation with the mandated expert dtype lifts to 0.995; a slot / page-table / position /
    trace bug would still put the batch below both."""
    kind = s1.KIND_OF[layer]
    dec = _load_fused_checked(cfg, layer, mesh)
    max_seq = DSA_DENSE_EXACT_MAX_KV if not dec.is_kda else 4096 + 64
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=max_seq)
    pt = None
    if not dec.is_kda:
        perm = torch.randperm(batch * cache.blocks_per_seq, generator=torch.Generator().manual_seed(7))
        pt = dec.make_page_table(cache, permutation=perm)
    steps = 4
    users = s1._batched_users(dec, batch, steps, 1234 + layer + batch)
    ref_cache = dec.allocate_cache(max_batch=1, max_seq_len=max_seq)
    if not dec.is_kda:
        dec.make_page_table(ref_cache)
    prefill_rows = {}
    for slot, T, n in users:
        y = streams4(dec.prefill_forward(golden(layer, T)["streams_in"][:, : n + steps], cache=ref_cache, user_slot=0))
        prefill_rows[slot] = y[:, n : n + steps]
    del ref_cache
    for slot, T, n in users:
        dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=slot, start_pos=0)
    by_slot = sorted(users)
    td = s1.TracedDecode(dec, cache, pt, batch=batch)

    def inputs(step):
        rows = torch.cat([golden(layer, T)["streams_in"][:, n + step : n + step + 1] for _, T, n in by_slot], dim=1)
        return rows, [n + step for _, _, n in by_slot]

    rows, pos = inputs(0)
    td.set_inputs(rows, pos)
    td.capture()
    failures = []
    for step in range(steps):
        rows, pos = inputs(step)
        td.set_inputs(rows, pos)
        out = td.step()
        ref = torch.cat([golden(layer, T)["streams_out"][:, n + step : n + step + 1] for _, T, n in by_slot], dim=1)
        pre = torch.cat([prefill_rows[slot][:, step : step + 1] for slot, _, _ in by_slot], dim=1)
        users_rows = []
        for i, (slot, T, n) in enumerate(by_slot):
            p_dec = pcc(out[:, i : i + 1], ref[:, i : i + 1])
            p_pre = pcc(pre[:, i : i + 1], ref[:, i : i + 1])
            p_cons = pcc(out[:, i : i + 1], pre[:, i : i + 1])
            users_rows.append(
                dict(
                    slot=slot,
                    prompt=T,
                    position=n + step,
                    decode_pcc=p_dec,
                    prefill_path_pcc=p_pre,
                    decode_vs_prefill_path_pcc=p_cons,
                )
            )
            if p_dec < min(PCC_BAR, p_pre - 0.002):
                failures.append(dict(layer=layer, step=step, **users_rows[-1]))
        row = dict(
            layer_kind=kind,
            mode="decode",
            layer=layer,
            batch=batch,
            step=step,
            pcc=pcc(out, ref),
            prefill_path_pcc=pcc(pre, ref),
            batch_criterion="pcc >= min(0.995, prefill_path_pcc - 0.002)",
            min_user_pcc=min(u["decode_pcc"] for u in users_rows),
            min_decode_vs_prefill_path_pcc=min(u["decode_vs_prefill_path_pcc"] for u in users_rows),
            seq_len=max(pos) + 1,
            positions=pos,
            weights="real",
            trace=True,
            test="batched_paged_decode",
        )
        _record_fused(row)
        _record_fused(
            dict(layer_kind=kind, layer=layer, batch=batch, step=step, users=users_rows),
            name="batched_decode_users.jsonl",
        )
        if row["pcc"] < min(PCC_BAR, row["prefill_path_pcc"] - 0.002):
            failures.append(row)
    td.release()
    check(failures, "rows below the acceptance bar")


@pytest.mark.parametrize("layer", [0, 4])
def test_prefill_slot_isolation_nonfinite(mesh, cfg, layer):
    """The batched KDA state holds every user's conv history / recurrent state in one tensor: a prefill of slot 2 must
    not depend on what the other slots hold. Inf / NaN / -3e38 planted in the other slots' conv rows and NaN / Inf /
    1e30 in their recurrent states (host writes into the persistent tensors) -> slot 2's prefill output (non-aligned
    200 tokens), recurrent state and conv rows bitwise equal to the clean-cache run."""
    dec = _load_fused_checked(cfg, layer, mesh)
    t = 200
    x = golden(layer, 512)["streams_in"][:, :t]

    def run(poison: bool):
        cache = dec.allocate_cache(max_batch=4, max_seq_len=t + 64)
        if poison:
            c = ttnn.to_torch(cache.kda.C)  # [1, 3, slots, 3HK] bf16
            c[0, :, 0, :], c[0, :, 1, :], c[0, :, 3, :] = float("inf"), float("nan"), -3.0e38
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), cache.kda.C
            )
            st = ttnn.to_torch(cache.kda.S)  # [slots, H, K, V] fp32
            st[0], st[1], st[3] = float("nan"), float("inf"), 1.0e30
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(st, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), cache.kda.S
            )
        y = dec.prefill_forward(x, cache=cache, user_slot=2)
        return y, ttnn.to_torch(cache.kda.recurrent[2]), ttnn.to_torch(cache.kda.conv[2])

    clean, poisoned = run(False), run(True)
    same = [bool(torch.equal(a, b)) for a, b in zip(clean, poisoned)]
    row = dict(
        impl="fused",
        test="prefill_slot_isolation_nonfinite",
        layer_kind=s1.KIND_OF[layer],
        layer=layer,
        tokens=t,
        output_bitwise_equal=same[0],
        recurrent_bitwise_equal=same[1],
        conv_bitwise_equal=same[2],
        output_finite=bool(torch.isfinite(poisoned[0]).all()),
    )
    record(row, name="slot_isolation.jsonl")
    check([] if all(same) and row["output_finite"] else [row], "prefill of one slot depends on another slot's state")
