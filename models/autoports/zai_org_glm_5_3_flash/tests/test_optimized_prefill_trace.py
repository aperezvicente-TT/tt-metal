# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced prefill of the OPTIMIZED decoder (DECISIONS 2026-10-17 ~20:20Z): one trace per bucket of the bounded set
(`OptimizedDecoder.trace_buckets`: KDA 32..2048, MLA 64..2048, powers of two), persistent inputs / outputs, prompts split
by `plan_traced_chunks`.

  * test_prefill_trace          — per layer kind: setup in the generator order (prepare_prefill_traces -> decode trace
                                  -> capture_prefill_traces); a 1999-token prompt (6 bucket chunks) in slot 1 and a
                                  1024-token prompt + 476-token continuation in slot 0, interleaved with replays of the
                                  live decode trace (the other slot inactive); prefill / decode PCC vs the fp32 goldens;
                                  replay determinism (bitwise); no program compiled after the captures; the persistent
                                  inputs survive; with TT_METAL_TRACE_ALLOC_TRACKING=1 every replay is checked for unsafe
                                  allocations (execute_trace raises); traced vs eager prompt PCC recorded.
  * test_prefill_trace_full_bucket — the 2048-token bucket: KDA a 4095-token traced prompt (every bucket, two
                                  rounds of the planner) + decode vs the 4096-token golden; MLA a 2048-token traced
                                  prompt vs the golden + decode at position 2048 vs the eager slot's.
  * test_prefill_trace_single_slot — max_batch = 1 (the KDA slot slice is a view of the whole state): traced prompt +
                                  continuation + traced decode vs the goldens, persistent buffers and state alive.
  * test_prefill_trace_perf     — warmed wall of every bucket untraced (begin_prefill_chunk + prefill_chunk, host
                                  dispatch included) vs traced (run_prefill_trace), device-synchronised, and end-to-end
                                  2048 / 1999 / 128-token prompts (prefill_forward vs prefill_forward_traced, host
                                  transfers included). Rows -> prefill_trace_perf.jsonl.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_functional_decoder as s1
from models.autoports.zai_org_glm_5_3_flash.tests.common import PCC_BAR, check, golden, pcc, record, streams4
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoder, plan_traced_chunks

TRACE_REGION = int(os.environ.get("GLM53_PREFILL_TRACE_REGION", str(1 << 30)))
cfg = s1.cfg


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh(trace_region_size=TRACE_REGION)
    yield m
    ttnn.close_mesh_device(m)


def _trace_bytes(mesh) -> int | None:
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)
    except Exception:  # memory view not available for this device type: not recorded
        return None


def _setup(dec: OptimizedDecoder, mesh, batch: int = 2, kda_max_seq: int = 2048 + 64, before=None):
    """Generator setup order; `before(cache)` runs eager work (e.g. an eager prompt in a slot other than 0, which the
    setup overwrites) after the cache allocation and before anything is captured."""
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=kda_max_seq if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    if before is not None:
        before(cache)
    t0 = time.perf_counter()
    traces = dec.prepare_prefill_traces(cache)
    t_prep = time.perf_counter() - t0
    td = s1.TracedDecode(dec, cache, pt, batch=batch)
    td.set_inputs(torch.zeros(1, batch, 4, 4096), [-1] * batch)
    td.capture()
    before = _trace_bytes(mesh)
    t0 = time.perf_counter()
    dec.capture_prefill_traces(cache, traces)
    t_cap = time.perf_counter() - t0
    after = _trace_bytes(mesh)
    setup = dict(
        buckets=sorted(traces),
        prepare_s=t_prep,
        capture_s=t_cap,
        prefill_trace_bytes=None if before is None or after is None else after - before,
    )
    return cache, pt, traces, td, setup


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_prefill_trace(mesh, cfg, layer):
    kind = s1.KIND_OF[layer]
    dec = load_optimized_decoder(cfg, layer, mesh)
    g = golden(layer, 2048)
    xin, ref = g["streams_in"], g["streams_out"]  # [1, 2048, 4, 4096]
    cache, pt, traces, td, setup = _setup(dec, mesh)
    entries = mesh.num_program_cache_entries()
    row = dict(impl="optimized", test="prefill_trace", layer_kind=kind, layer=layer, **setup)
    failures = []

    def gate(mode, out, want, **extra):
        r = dict(layer_kind=kind, mode=mode, layer=layer, pcc=pcc(out, want), weights="real", trace=True, **extra)
        record({"impl": "optimized", **r})
        if r["pcc"] < PCC_BAR:
            failures.append(r)
        return r["pcc"]

    # slot 1: 1999 tokens (1024 + 512 + 256 + 128 + 64 + a padded tail) while slot 0 is idle
    plan1 = plan_traced_chunks(1999, dec.prefill_align, tuple(sorted(traces)))
    y1 = streams4(dec.prefill_forward_traced(xin[:, :1999], cache=cache, traces=traces, user_slot=1))
    gate("prefill_traced", y1, ref[:, :1999], seq_len=1999, chunks=[c[2] for c in plan1])
    td.set_inputs(torch.cat([xin[:, :1], xin[:, 1999:2000]], dim=1), [-1, 1999])
    d1 = td.step()[:, 1:2]
    gate("decode_after_traced_prefill", d1, ref[:, 1999:2000], position=1999, batch=2)
    # slot 0: 1024 tokens, then a 476-token continuation, between replays of the live decode trace
    y0a = streams4(dec.prefill_forward_traced(xin[:, :1024], cache=cache, traces=traces, user_slot=0))
    td.set_inputs(torch.cat([xin[:, :1], xin[:, :1]], dim=1), [-1, -1])
    td.step()  # both slots inactive: state / pages untouched
    y0b = streams4(
        dec.prefill_forward_traced(xin[:, 1024:1500], cache=cache, traces=traces, user_slot=0, start_pos=1024)
    )
    gate("prefill_traced_continuation", torch.cat([y0a, y0b], dim=1), ref[:, :1500], seq_len=1500, start=1024)
    td.set_inputs(torch.cat([xin[:, 1500:1501], xin[:, :1]], dim=1), [1500, -1])
    d0 = td.step()[:, :1]
    gate("decode_after_traced_prefill", d0, ref[:, 1500:1501], position=1500, batch=2)
    # determinism: the same prompt again in slot 1 (reset at chunk start 0) is bitwise equal
    y1b = streams4(dec.prefill_forward_traced(xin[:, :1999], cache=cache, traces=traces, user_slot=1))
    row["replay_bitwise_equal"] = bool(torch.equal(y1, y1b))
    row["program_cache_entries_after_capture"] = entries
    row["program_cache_entries_end"] = mesh.num_program_cache_entries()
    row["inputs_alive"] = all(t.x.is_allocated() and t.y.is_allocated() for t in traces.values())
    td.release()
    dec.release_prefill_traces(traces)
    # eager prompt (one padded chunk) on the same layer: the traced split is not bitwise equal to it (other chunking)
    ye = streams4(dec.prefill_forward(xin[:, :1999], cache=cache, user_slot=1))
    row["pcc_traced_vs_eager_1999"] = pcc(y1, ye)
    row["pcc_eager_1999"] = pcc(ye, ref[:, :1999])
    record(row, name="prefill_trace.jsonl")
    if not row["replay_bitwise_equal"]:
        failures.append("replay not bitwise equal")
    if row["program_cache_entries_end"] != entries:
        failures.append("a traced prefill compiled a program")
    if not row["inputs_alive"]:
        failures.append("a persistent trace buffer was freed")
    check(failures, "traced prefill")


@pytest.mark.parametrize("layer", [0, 3])
def test_prefill_trace_single_slot(mesh, cfg, layer):
    """max_batch = 1 (single-user serving): the KDA slot state is the whole state tensor (its dim-0 slice is a view),
    traced 333-token prompt + 64-token continuation + traced decode vs the goldens, state bitwise equal to the eager path.
    """
    kind = s1.KIND_OF[layer]
    dec = load_optimized_decoder(cfg, layer, mesh)
    g = golden(layer, 2048)
    xin, ref = g["streams_in"], g["streams_out"]
    cache, pt, traces, td, setup = _setup(dec, mesh, batch=1)
    failures = []
    y = streams4(dec.prefill_forward_traced(xin[:, :320], cache=cache, traces=traces, user_slot=0))
    y2 = streams4(dec.prefill_forward_traced(xin[:, 320:397], cache=cache, traces=traces, user_slot=0, start_pos=320))
    td.set_inputs(xin[:, 397:398], [397])
    d = td.step()
    rows = [
        dict(
            layer_kind=kind,
            mode="prefill_traced",
            layer=layer,
            seq_len=397,
            batch=1,
            pcc=pcc(torch.cat([y, y2], dim=1), ref[:, :397]),
        ),
        dict(
            layer_kind=kind,
            mode="decode_after_traced_prefill",
            layer=layer,
            position=397,
            batch=1,
            pcc=pcc(d, ref[:, 397:398]),
        ),
    ]
    for r in rows:
        record({"impl": "optimized", "test": "prefill_trace_single_slot", "weights": "real", "trace": True, **r})
        if r["pcc"] < PCC_BAR:
            failures.append(r)
    alive = all(t.x.is_allocated() and t.y.is_allocated() for t in traces.values())
    if dec.is_kda:
        alive = alive and cache.kda.S.is_allocated() and cache.kda.prefill_S.is_allocated()
    record(
        dict(
            impl="optimized",
            test="prefill_trace_single_slot",
            layer_kind=kind,
            layer=layer,
            buffers_alive=alive,
            **setup,
        ),
        name="prefill_trace.jsonl",
    )
    td.release()
    dec.release_prefill_traces(traces)
    check(failures + ([] if alive else ["a persistent buffer was freed"]), "single-slot traced prefill")


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_prefill_trace_full_bucket(mesh, cfg, layer):
    """The 2048-token bucket (Stage 3 self-review R1) and, for KDA, a traced prompt longer than one chunk. KDA: a
    4095-token prompt traced in slot 0 (2048 + 1024 + 512 + 256 + 128 + 64 + 32 + a padded 31: every bucket) and eager
    (prefill_forward) in slot 1, then a B=2 traced decode at position 4095 of both, all vs the fp32 4096-token golden.
    MLA (DSA dense-exact up to 2051 keys; goldens of 2048 tokens): a 2048-token prompt traced in slot 0 (one 2048-bucket
    chunk) vs the golden and eager in slot 1, then a B=2 traced decode at position 2048 of both from the same input row
    (no golden row 2048: the traced slot's decode vs the eager slot's checks the pages the 2048-bucket trace wrote).
    Plus: traced prompt replay bitwise, no program compiled after the captures, persistent buffers alive; with
    TT_METAL_TRACE_ALLOC_TRACKING=1 every replay is checked for unsafe allocations."""
    kind = s1.KIND_OF[layer]
    dec = load_optimized_decoder(cfg, layer, mesh)
    n = 4095 if dec.is_kda else 2048
    g = golden(layer, 4096 if dec.is_kda else 2048)
    xin, ref = g["streams_in"], g["streams_out"]
    eager = {}

    def before(cache):  # eager prompt in slot 1 before any capture (it compiles the eager programs)
        eager["y"] = streams4(dec.prefill_forward(xin[:, :n], cache=cache, user_slot=1))

    cache, pt, traces, td, setup = _setup(dec, mesh, kda_max_seq=4096 + 64, before=before)
    entries = mesh.num_program_cache_entries()
    failures = []

    def gate(mode, out, want, bar=PCC_BAR, **extra):
        r = dict(layer_kind=kind, mode=mode, layer=layer, pcc=pcc(out, want), weights="real", trace=True, **extra)
        record({"impl": "optimized", "test": "prefill_trace_full_bucket", **r})
        if r["pcc"] < bar:
            failures.append(r)
        return r["pcc"]

    plan = plan_traced_chunks(n, dec.prefill_align, tuple(sorted(traces)))
    y = streams4(dec.prefill_forward_traced(xin[:, :n], cache=cache, traces=traces, user_slot=0))
    gate("prefill_traced", y, ref[:, :n], seq_len=n, chunks=[c[2] for c in plan])
    gate("prefill", eager["y"], ref[:, :n], seq_len=n, path="eager (slot 1)")
    row_in = xin[:, n : n + 1] if dec.is_kda else xin[:, n - 1 : n]  # MLA: no input row 2048 in the golden
    td.set_inputs(torch.cat([row_in, row_in], dim=1), [n, n])
    d = td.step()
    if dec.is_kda:
        gate("decode_after_traced_prefill", d[:, :1], ref[:, n : n + 1], position=n, batch=2, slot="traced")
        gate("decode", d[:, 1:2], ref[:, n : n + 1], position=n, batch=2, slot="eager")
    p_te = gate("decode_traced_vs_eager_slot", d[:, :1], d[:, 1:2], bar=0.999, position=n, batch=2)
    # determinism: the same prompt again (reset at chunk start 0) is bitwise equal
    y2 = streams4(dec.prefill_forward_traced(xin[:, :n], cache=cache, traces=traces, user_slot=0))
    row = dict(
        impl="optimized",
        test="prefill_trace_full_bucket",
        layer_kind=kind,
        layer=layer,
        seq_len=n,
        chunks=[c[2] for c in plan],
        decode_traced_vs_eager_slot_pcc=p_te,
        decode_traced_vs_eager_slot_bitwise=bool(torch.equal(d[:, :1], d[:, 1:2])),
        pcc_traced_vs_eager_prompt=pcc(y, eager["y"]),
        replay_bitwise_equal=bool(torch.equal(y, y2)),
        program_cache_entries_after_capture=entries,
        program_cache_entries_end=mesh.num_program_cache_entries(),
        inputs_alive=all(t.x.is_allocated() and t.y.is_allocated() for t in traces.values()),
        **setup,
    )
    td.release()
    dec.release_prefill_traces(traces)
    record(row, name="prefill_trace.jsonl")
    if not row["replay_bitwise_equal"]:
        failures.append("replay not bitwise equal")
    if row["program_cache_entries_end"] != entries:
        failures.append("a traced prefill compiled a program")
    if not row["inputs_alive"]:
        failures.append("a persistent trace buffer was freed")
    check(failures, "traced prefill: 2048 bucket / > 2048 tokens")


def _median_wall(mesh, fn, reps: int = 3) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_prefill_trace_perf(mesh, cfg, layer):
    """Untraced runs first (they compile what they use while no trace exists: a program compiled after a capture could
    get its binary buffer where a trace's intermediates live), then the traced setup and the traced runs."""
    kind = s1.KIND_OF[layer]
    dec = load_optimized_decoder(cfg, layer, mesh)
    g = golden(layer, 2048)
    xin = g["streams_in"]
    cache = dec.allocate_cache(max_batch=2, max_seq_len=2048 + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    buckets = dec.trace_buckets(cache)
    untraced, prompt_untraced = {}, {}
    for p in buckets:
        x = ttnn.from_torch(
            xin[:, :p].reshape(1, 1, p, -1).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh
        )

        def eager():
            dec.begin_prefill_chunk(cache, user_slot=1, chunk_start=0, n_valid=p)
            ttnn.deallocate(dec.prefill_chunk(x, cache=cache, user_slot=1))

        untraced[p] = _median_wall(mesh, eager)
        ttnn.deallocate(x)
    for n in (2048, 1999, 128):
        prompt_untraced[n] = _median_wall(
            mesh, lambda: dec.prefill_forward(xin[:, :n], cache=cache, user_slot=1), reps=2
        )
    t0 = time.perf_counter()
    traces = dec.prepare_prefill_traces(cache)
    t_prep = time.perf_counter() - t0
    before = _trace_bytes(mesh)
    t0 = time.perf_counter()
    dec.capture_prefill_traces(cache, traces)
    t_cap = time.perf_counter() - t0
    after = _trace_bytes(mesh)
    entries = mesh.num_program_cache_entries()
    base = dict(
        impl="optimized",
        test="prefill_trace_perf",
        layer_kind=kind,
        layer=layer,
        buckets=sorted(traces),
        prepare_s=t_prep,
        capture_s=t_cap,
        prefill_trace_bytes=None if before is None or after is None else after - before,
    )
    for p, t in sorted(traces.items()):
        ttnn.copy_host_to_device_tensor(dec.prefill_host_chunk(xin[0, :p], t), t.x)
        tr = _median_wall(mesh, lambda: dec.run_prefill_trace(t, cache, user_slot=1, chunk_start=0, n_valid=p))
        e = untraced[p]
        record(dict(base, bucket=p, untraced_ms=e, traced_ms=tr, speedup=e / tr), name="prefill_trace_perf.jsonl")
    for n in (2048, 1999, 128):
        tr = _median_wall(
            mesh, lambda: dec.prefill_forward_traced(xin[:, :n], cache=cache, traces=traces, user_slot=1), reps=2
        )
        chunks = [c[2] for c in plan_traced_chunks(n, dec.prefill_align, tuple(sorted(traces)))]
        record(
            dict(base, prompt=n, traced_chunks=chunks, prompt_traced_ms=tr, prompt_untraced_ms=prompt_untraced[n]),
            name="prefill_trace_perf.jsonl",
        )
    assert mesh.num_program_cache_entries() == entries, "a traced run compiled a program"
    dec.release_prefill_traces(traces)
