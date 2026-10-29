# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced warmed decode of one decoder layer per kind, delimited by Tracy signposts; the SAME harness
measures the optimized decoder (default) and the Stage 2 fused / Stage 1 functional baselines
(GLM53_DECODER_IMPL=optimized|fused|functional), so before/after rows are like-for-like.

    python -m tracy -r -p -v -m pytest models/autoports/zai_org_glm_5_3_flash/tests/test_optimized_perf.py
Signposts (one Tracy window each, all warmed: the programs are compiled by an identical call first):
    PERF_PREFILL .. _END           one 2048-token prefill_chunk pass (eager ops, the dominant prefill bucket)
    PERF_PREFILL_SHORT .. _END     one 128-token prefill_chunk pass (short prompt / tail bucket)
    PERF_PREFILL_TRACED[_SHORT]    the same chunks replayed from a captured prefill trace (optimized decoder only:
                                   prepare_prefill_traces / capture_prefill_traces / run_prefill_trace)
    PERF_DECODE[_B<n>] .. _END     DECODE_REPLAYS replays of the captured decode trace, n users at distinct golden
                                   positions (B=1 untagged)
    PERF_ATTN_B<n> / PERF_FFN_B<n> the attention block (KDA / MLA decode) and the FFN block (dense MLP / MoE: router,
                                   routed experts, shared expert, add) alone, each captured in its own trace on the
                                   step's normalised input (n in GLM53_PERF_BLOCK_BATCHES, default 8: the plan gates
                                   "KDA layer <= 0.5 ms" and "MoE layer <= 1.5 ms" at B=8)
`test_prefill_wall` (not profiled) times whole prompts through `prefill_forward` (host chunk loop incl. transfers) at a
short, a non-aligned and a long length per kind. Wall-clock rows go to perf_wall.jsonl tagged with the implementation
and the config overrides.
"""

from __future__ import annotations

import os
import time

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, hf_config, int_tensor, record, to_dev
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import IMPL, load_impl, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


DECODE_REPLAYS = int(os.environ.get("GLM53_PERF_DECODE_REPLAYS", "4"))
WALL_REPLAYS = int(os.environ.get("GLM53_PERF_WALL_REPLAYS", "20"))
PROFILING = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
PREFILL_LEN = int(os.environ.get("GLM53_PERF_PREFILL_LEN", "2048"))
SHORT_LEN = int(os.environ.get("GLM53_PERF_SHORT_LEN", "128"))
KINDS = [("dense_kda", 0), ("sparse_kda", 4), ("sparse_dsa", 3)]
ONLY = os.environ.get("GLM53_PERF_KINDS")  # tracy re-runs pytest through /bin/sh: select by env
BATCHES = [int(b) for b in os.environ.get("GLM53_PERF_BATCHES", "1,8,32").split(",") if b]
BLOCK_BATCHES = [int(b) for b in os.environ.get("GLM53_PERF_BLOCK_BATCHES", "8").split(",") if b]
DO_PREFILL = os.environ.get("GLM53_PERF_PREFILL", "1") == "1"
OVERRIDES = os.environ.get("GLM53_OCFG", "") if IMPL == "optimized" else os.environ.get("GLM53_FCFG", "")


# GLM53_PERF_KIND_SIGNPOSTS=1: windows are named PERF_<KIND>_<window> (one Tracy process for every kind; the device is
# opened once per process, DECISIONS 2026-10-17 ~21:50Z); else PERF_<window> (one kind per process)
KIND_SIGNPOSTS = os.environ.get("GLM53_PERF_KIND_SIGNPOSTS") == "1"
_SP = {"prefix": "PERF"}


def _sp(name: str) -> str:
    return f"{_SP['prefix']}_{name}"


@pytest.fixture(scope="module")
def perf_mesh():
    """One device open for every kind of the module (trace region for the decode and prefill traces)."""
    m = open_mesh(trace_region_size=int(os.environ.get("GLM53_TRACE_REGION", "1000000000")))
    yield m
    ttnn.close_mesh_device(m)


def _flush(mesh):
    if PROFILING:
        ttnn.ReadDeviceProfiler(mesh)


def _kinds():
    return [k for k in KINDS if not ONLY or k[0] in ONLY.split(",")]


def _timed_prefill(dec, cache, mesh, streams, n: int, tag: str) -> float:
    x = to_dev(streams[:, :n], mesh)
    dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=n)
    ttnn.deallocate(dec.prefill_chunk(x, cache=cache, user_slot=0))  # warm-up (compile)
    dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=n)
    ttnn.synchronize_device(mesh)
    _flush(mesh)
    signpost(_sp(f"PREFILL{tag}"))
    t0 = time.perf_counter()
    y = dec.prefill_chunk(x, cache=cache, user_slot=0)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3
    signpost(_sp(f"PREFILL{tag}_END"))
    ttnn.deallocate(y)
    ttnn.deallocate(x)
    _flush(mesh)
    return ms


def _timed_traced_prefill(dec, cache, mesh, streams, n: int, tag: str) -> float:
    traces = dec.capture_prefill_traces(cache, dec.prepare_prefill_traces(cache, buckets=(n,)))
    t = traces[n]
    ttnn.copy_host_to_device_tensor(dec.prefill_host_chunk(streams[0, :n], t), t.x)
    run = lambda: dec.run_prefill_trace(t, cache, user_slot=0, chunk_start=0, n_valid=n)
    run()  # warm replay
    ttnn.synchronize_device(mesh)
    _flush(mesh)
    signpost(_sp(f"PREFILL_TRACED{tag}"))
    t0 = time.perf_counter()
    run()
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3
    signpost(_sp(f"PREFILL_TRACED{tag}_END"))
    _flush(mesh)
    dec.release_prefill_traces(traces)
    return ms


def _block_windows(dec, cache, pt, mesh, streams, cur_pos, batch: int) -> dict:
    """Attention block and FFN block alone (each in its own trace) on the step's stream-0 rows normalised by the layer's
    own norms: signposted windows PERF_ATTN_B<n> / PERF_FFN_B<n>; returns wall ms per replay."""
    lay = dec.layer
    h = ttnn.typecast(ttnn.slice(streams, (0, 0, 0, 0), (1, 1, batch, dec.cfg.hidden_size)), ttnn.float32)
    xa, xf = lay._norm(h, lay.input_norm), lay._norm(h, lay.post_attn_norm)
    if dec.is_kda:
        active = dec._active_masks(cur_pos)
        attn = lambda: lay.attn.decode(xa, cache.kda, active)
    else:
        attn = lambda: lay.attn.decode(xa, cache.latent, pt, cur_pos)
    takes_flag = lay.spec.mlp == "moe" or IMPL == "optimized"
    ffn = (lambda: lay.mlp(xf, decode=True)) if takes_flag else (lambda: lay.mlp(xf))
    out = {}
    snap = dec.snapshot_state(cache)
    for name, fn in (("ATTN", attn), ("FFN", ffn)):
        warm = fn()
        ttnn.synchronize_device(mesh)
        del warm
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            res = fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)  # never leave the shared device in capture mode
        for _ in range(2):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        _flush(mesh)
        reps = DECODE_REPLAYS if PROFILING else WALL_REPLAYS
        signpost(_sp(f"{name}_B{batch}"))
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            if PROFILING:
                ttnn.synchronize_device(mesh)
                _flush(mesh)
        ttnn.synchronize_device(mesh)
        out[name] = (time.perf_counter() - t0) * 1e3 / reps
        _flush(mesh)
        signpost(_sp(f"{name}_B{batch}_END"))
        ttnn.release_trace(mesh, tid)
        del res
    dec.restore_state(cache, snap)
    return out


@pytest.mark.parametrize("kind,layer", _kinds())
def test_perf(perf_mesh, kind, layer):
    cfg = hf_config()
    mesh = perf_mesh
    _SP["prefix"] = f"PERF_{kind.upper()}" if KIND_SIGNPOSTS else "PERF"
    try:
        dec = load_impl(cfg, layer, mesh)
        g = golden(layer, 2048)
        for batch in BATCHES:
            tag = "" if batch == 1 else f"_B{batch}"
            cache = dec.allocate_cache(
                max_batch=batch, max_seq_len=PREFILL_LEN + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV
            )
            pt = None if dec.is_kda else dec.make_page_table(cache)
            prefill_ms = short_ms = traced_ms = traced_short_ms = None
            if DO_PREFILL and batch == BATCHES[0]:
                short_ms = _timed_prefill(dec, cache, mesh, g["streams_in"], SHORT_LEN, "_SHORT")
                prefill_ms = _timed_prefill(dec, cache, mesh, g["streams_in"], PREFILL_LEN, "")
                if hasattr(dec, "prepare_prefill_traces"):
                    traced_short_ms = _timed_traced_prefill(dec, cache, mesh, g["streams_in"], SHORT_LEN, "_SHORT")
                    traced_ms = _timed_traced_prefill(dec, cache, mesh, g["streams_in"], PREFILL_LEN, "")

            last = PREFILL_LEN - 1 if dec.is_kda else min(PREFILL_LEN, 2050) - 1
            positions = list(range(last - batch + 1, last + 1))
            streams = to_dev(g["streams_in"][:, positions[0] : last + 1], mesh)
            cur_pos = int_tensor(positions, mesh)
            snap = dec.snapshot_state(cache)
            ttnn.deallocate(dec.decode_forward(streams, cache=cache, page_table=pt, cur_pos=cur_pos))
            dec.restore_state(cache, snap)
            ttnn.synchronize_device(mesh)
            _flush(mesh)
            trace = dec.capture_decode_trace(streams, cache=cache, page_table=pt, cur_pos=cur_pos)
            for _ in range(2):
                dec.restore_state(cache, snap)
                trace.replay()
            ttnn.synchronize_device(mesh)
            _flush(mesh)
            reps = DECODE_REPLAYS if PROFILING else WALL_REPLAYS
            signpost(_sp(f"DECODE{tag}"))
            t0 = time.perf_counter()
            for _ in range(reps):
                trace.replay(blocking=False)
                if PROFILING:
                    ttnn.synchronize_device(mesh)
                    _flush(mesh)
            ttnn.synchronize_device(mesh)
            decode_ms = (time.perf_counter() - t0) * 1e3 / reps
            _flush(mesh)
            signpost(_sp(f"DECODE{tag}_END"))
            trace.release()
            blocks = _block_windows(dec, cache, pt, mesh, streams, cur_pos, batch) if batch in BLOCK_BATCHES else {}
            record(
                dict(
                    impl=IMPL,
                    overrides=OVERRIDES,
                    layer_kind=kind,
                    layer=layer,
                    batch=batch,
                    prefill_tokens=PREFILL_LEN if prefill_ms is not None else None,
                    prefill_wall_ms=prefill_ms,
                    short_prefill_tokens=SHORT_LEN if short_ms is not None else None,
                    short_prefill_wall_ms=short_ms,
                    traced_prefill_wall_ms=traced_ms,
                    traced_short_prefill_wall_ms=traced_short_ms,
                    decode_wall_ms_per_step=decode_ms,
                    attn_block_wall_ms=blocks.get("ATTN"),
                    ffn_block_wall_ms=blocks.get("FFN"),
                    decode_replays=reps,
                    profiler=PROFILING,
                ),
                name="perf_wall.jsonl",
            )
            print(
                f"PERF {IMPL} {kind} L{layer} B={batch}: prefill {prefill_ms} ms (traced {traced_ms}), short prefill "
                f"{short_ms} ms (traced {traced_short_ms}), "
                f"traced decode {decode_ms:.3f} ms/step, attn block {blocks.get('ATTN')} ms, ffn block {blocks.get('FFN')} ms"
            )
            del cache
    finally:
        ttnn.synchronize_device(mesh)


PROMPT_LENS = {"kda": (128, 333, 2048, 4096 + 333), "mla": (128, 333, 2048)}


@pytest.mark.parametrize("kind,layer", _kinds())
def test_prefill_wall(perf_mesh, kind, layer):
    """Whole-prompt prefill through `prefill_forward` (host chunk loop: pad, transfer, begin_prefill_chunk,
    prefill_chunk, read back), warmed: every bucket used is compiled by a first identical prompt."""
    cfg = hf_config()
    mesh = perf_mesh
    try:
        dec = load_impl(cfg, layer, mesh)
        lens = PROMPT_LENS["kda" if dec.is_kda else "mla"]
        g = golden(layer, 4096) if dec.is_kda else golden(layer, 2048)
        cache = dec.allocate_cache(max_batch=1, max_seq_len=max(lens) + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
        if not dec.is_kda:
            dec.make_page_table(cache)
        for n in lens:
            src = g["streams_in"][:, :n]
            dec.prefill_forward(src, cache=cache, user_slot=0)  # warm (compile the buckets of this length)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            dec.prefill_forward(src, cache=cache, user_slot=0)
            ms = (time.perf_counter() - t0) * 1e3
            record(
                dict(
                    impl=IMPL, overrides=OVERRIDES, layer_kind=kind, layer=layer, prompt_tokens=n, prefill_forward_ms=ms
                ),
                name="perf_prefill_wall.jsonl",
            )
            print(f"PREFILL_WALL {IMPL} {kind} L{layer} n={n}: {ms:.1f} ms")
    finally:
        ttnn.synchronize_device(mesh)
