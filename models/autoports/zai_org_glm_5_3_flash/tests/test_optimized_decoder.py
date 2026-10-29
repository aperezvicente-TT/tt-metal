# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight tests of the GLM-5.3-Flash OPTIMIZED decoder (`tt/optimized_decoder.py`).

Stage 1's gates are reused verbatim (golden prompts of every captured layer, non-aligned prefill lengths /
continuation / traced follow-on decode, batched paged decode with a permuted page table, prefill under a live decode
trace, decode between the chunks of an interleaved prefill with inactive slots, NaN / Inf / huge values in an inactive
slot's input row), plus Stage 2's row-count dispatch coverage and slot-isolation test, all run against
`OptimizedDecoder`: the module-level loaders are swapped by an autouse fixture and `_load_optimized_checked` asserts the
optimized class AND that every decode projection is a DRAM-sharded `DSLinear` (no functional fallback). Rows are tagged
`impl: optimized`.

Optimized-only tests: `test_optimized_vs_fused` (Stage 2 and Stage 3 layers on identical inputs: prompt-128 prefill,
traced decode0, B=4 / B=8 traced decode of distinct users; both vs the fp32 goldens), `test_prepare_prefill_once_per_kind`
(Stage 1 C-4: the second layer of a kind compiles nothing new in any prefill bucket after its short warm-up),
`test_ds_weights_shared_by_prefill` (one weight copy: the prefill matmuls read the DRAM-sharded tensors).
"""

from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests import test_functional_decoder as s1
from models.autoports.zai_org_glm_5_3_flash.tests import test_fused_decoder as s2
from models.autoports.zai_org_glm_5_3_flash.tests.common import PCC_BAR, check, golden, pcc, record
from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import DSLinear, OptimizedDecoder

mesh = s1.mesh  # module-scoped device fixture (same as Stage 1)
cfg = s1.cfg


class OptimizedRecordingRouter(s2.FusedRecordingRouter):
    """FusedRecordingRouter for the optimized router (which takes the decode flag)."""

    def __call__(self, x, decode: bool = False):
        out = self.router(x, decode=decode)
        if self.recording:
            self.log.append(out)
        return out


def _record_optimized(row: dict, name: str = "pcc_real_weights.jsonl"):
    return record({"impl": "optimized", **row}, name=name)


def assert_optimized_path(dec):
    """The decoder is the optimized class and every decode projection is a DRAM-sharded DSLinear."""
    assert isinstance(dec, OptimizedDecoder), type(dec)
    ds = dec.layer.ds_linears()
    roles = {d.role for d in ds}
    need = {"kda_qkv", "kda_o"} if dec.is_kda else {"mla_qa_kva", "mla_qb", "mla_o"}
    need |= {"mlp_gate_up", "mlp_down"} if dec.spec.mlp == "dense" else {"shared_gate_up", "shared_down", "router"}
    assert need <= roles, (sorted(roles), sorted(need))
    assert all(isinstance(d, DSLinear) and d.weight.memory_config().is_sharded() for d in ds)


def _load_optimized_checked(cfg_, layer, mesh_):
    dec = load_optimized_decoder(cfg_, layer, mesh_)
    assert_optimized_path(dec)
    return dec


@pytest.fixture(autouse=True)
def _optimized_path(monkeypatch):
    monkeypatch.setattr(s1, "load_decoder", _load_optimized_checked)
    monkeypatch.setattr(s1, "RecordingRouter", OptimizedRecordingRouter)
    monkeypatch.setattr(s1, "_routing_rows", s2.fused_routing_rows)
    monkeypatch.setattr(s1, "record", _record_optimized)
    monkeypatch.setattr(s2, "_load_fused_checked", _load_optimized_checked)
    monkeypatch.setattr(s2, "record", _record_optimized)
    # Stage 2's row-count test writes through _record_fused ({"impl": "fused", **row}), whose key would override the
    # optimized label (self-review R6)
    monkeypatch.setattr(s2, "_record_fused", _record_optimized)


# Stage 1 gates, unmodified, on the optimized decoder
test_golden_prompts = s1.test_golden_prompts
test_prefill_lengths = s1.test_prefill_lengths
test_batched_paged_decode = s1.test_batched_paged_decode
test_prefill_under_live_trace = s1.test_prefill_under_live_trace
test_decode_between_prefill_chunks = s1.test_decode_between_prefill_chunks
test_inactive_slot_nonfinite_input = s1.test_inactive_slot_nonfinite_input
# Stage 2's row-count dispatch coverage and slot isolation, on the optimized decoder
test_batched_paged_decode_rows = s2.test_batched_paged_decode_rows
test_prefill_slot_isolation_nonfinite = s2.test_prefill_slot_isolation_nonfinite


# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer", [0, 4, 3])
def test_optimized_vs_fused(mesh, cfg, layer):
    """Stage 2 (fused) vs Stage 3 (optimized) layer on identical inputs: prompt-128 prefill, traced decode0 (B=1),
    B=4 and B=8 traced decodes of users at different positions. Gate: optimized vs golden >= the bar; the deltas vs the
    fused layer are recorded (the routed experts run at LoFi and the decode projections with a different reduction
    order, so the two are not bitwise equal)."""
    kind = s1.KIND_OF[layer]
    g = golden(layer, 128)
    g2 = golden(layer, 2048)
    lens = {4: [100, 700, 1500, 333], 8: [100, 700, 1500, 333, 64, 1999, 1024, 47]}
    outs = {}
    for impl, loader in (("fused", load_fused_decoder), ("optimized", _load_optimized_checked)):
        dec = loader(cfg, layer, mesh)
        cache = dec.allocate_cache(max_batch=1, max_seq_len=128 + 64)
        pt = None if dec.is_kda else dec.make_page_table(cache)
        y = dec.prefill_forward(g["streams_in"], cache=cache, user_slot=0).reshape(1, -1, 4, 4096)
        td = s1.TracedDecode(dec, cache, pt, batch=1)
        td.set_inputs(g["decode0_streams_in"], [128])
        td.capture()
        res = [y, td.step()]
        td.release()
        for b, ls in lens.items():
            cb = dec.allocate_cache(max_batch=b, max_seq_len=2048 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
            ptb = None if dec.is_kda else dec.make_page_table(cb)
            for slot, n in enumerate(ls):
                dec.prefill_forward(g2["streams_in"][:, :n], cache=cb, user_slot=slot)
            tdb = s1.TracedDecode(dec, cb, ptb, batch=b)
            tdb.set_inputs(torch.cat([g2["streams_in"][:, n : n + 1] for n in ls], dim=1), ls)
            tdb.capture()
            res.append(tdb.step())
            tdb.release()
            del cb
        outs[impl] = res
        del dec, cache
    refs = [g["streams_out"], g["decode0_streams_out"]] + [
        torch.cat([g2["streams_out"][:, n : n + 1] for n in ls], dim=1) for ls in lens.values()
    ]
    names = ["prefill128", "decode0", "decode_b4", "decode_b8"]
    o, f = outs["optimized"], outs["fused"]
    row = dict(impl="optimized", test="optimized_vs_fused", layer_kind=kind, layer=layer)
    for i, nm in enumerate(names):
        row[f"{nm}_pcc_optimized"] = pcc(o[i], refs[i])
        row[f"{nm}_pcc_fused"] = pcc(f[i], refs[i])
        row[f"{nm}_pcc_optimized_vs_fused"] = pcc(o[i], f[i])
    record(row, name="optimized_vs_fused.jsonl")
    bad = [k for k, v in row.items() if k.endswith("_pcc_optimized") and v < PCC_BAR]
    check(bad, "optimized path below the bar")


@pytest.mark.parametrize("layers", [(0, 1), (4, 5), (3, 7)])
def test_prepare_prefill_once_per_kind(mesh, cfg, layers):
    """Stage 1 C-4: the first layer of a kind compiles every prefill bucket; the second layer of the same kind only runs
    its smallest bucket (its own fused_experts program), and afterwards a prefill of any bucket adds no program-cache
    entry. Records the warm-up wall times."""
    import time

    kind = s1.KIND_OF[layers[0]]
    times, decs = [], []
    for layer in layers:
        dec = _load_optimized_checked(cfg, layer, mesh)
        cache = dec.allocate_cache(max_batch=2, max_seq_len=4096 + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
        if not dec.is_kda:
            dec.make_page_table(cache)
        t0 = time.perf_counter()
        dec.prepare_prefill(cache)
        times.append(time.perf_counter() - t0)
        decs.append((dec, cache))
    dec, cache = decs[1]
    before = mesh.num_program_cache_entries()
    g = golden(layers[0], 2048)  # any real stream rows: the check is on compiled programs (goldens exist for L0/L4/L3)
    for n in (17, 333, 2048, 1000):  # buckets 32 / 64 .. 2048, incl. the full chunk and non-aligned tails
        n = min(n, cache.max_seq_len)
        dec.prefill_forward(g["streams_in"][:, :n], cache=cache, user_slot=1)
    after = mesh.num_program_cache_entries()
    row = dict(
        impl="optimized",
        test="prepare_prefill_once_per_kind",
        layer_kind=kind,
        layers=list(layers),
        warm_s_first=times[0],
        warm_s_second=times[1],
        program_cache_entries_before=before,
        program_cache_entries_after=after,
    )
    record(row, name="prepare_prefill.jsonl")
    check([] if after == before else [row], "a prefill of the second layer compiled a new program")


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_ds_weights_shared_by_prefill(mesh, cfg, layer):
    """One copy of each DRAM-sharded projection: the prefill linears hold the DSLinear's DRAM-sharded tensor."""
    dec = _load_optimized_checked(cfg, layer, mesh)
    lay = dec.layer
    pairs = []
    if dec.is_kda:
        a = lay.attn
        pairs += [
            (a.w_qkv, a.ds_qkv),
            (a.w_lowrank, a.ds_lowrank),
            (a.w_fb, a.ds_fb),
            (a.w_gb, a.ds_gb),
            (a.w_o, a.ds_o),
        ]
    else:
        a = lay.attn
        pairs += [(a.w_qa_kva, a.ds_qa_kva), (a.w_qb, a.ds_qb), (a.w_o, a.ds_o)]
    ffn = lay.mlp if dec.spec.mlp == "dense" else lay.mlp.shared
    pairs += [(ffn.gate_up, ffn.ds_gate_up), (ffn.down, ffn.ds_down)]
    pairs = [(w, ds) for w, ds in pairs if ds is not None]  # roles in ds_skip_roles keep their interleaved linear
    same = [w is ds.weight for w, ds in pairs]
    record(
        dict(impl="optimized", test="ds_weights_shared", layer=layer, roles=[ds.role for _, ds in pairs], shared=same),
        name="ds_weights.jsonl",
    )
    assert all(same), same
