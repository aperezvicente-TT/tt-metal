# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunk pipelining host mock (doc/ttnn/09-vllm/chunk_pipelining.md): the vLLM step sequence of a long prompt driven
through `Glm53Generator.prefill_forward` on the CPU MockModel (no device, no ttnn), three ways:

  served   -- one 2,048-chunk per step, the head + sample on EVERY step (what int27c does: the plugin host-samples
              intermediate steps, the generator sees each step's chunk as its "last");
  headless -- one chunk per step, `head_rows=[False]` on intermediate steps (supports_partial_prefill_sampling);
  run K    -- the plugin's tt.prefill_chunk_run: continuation steps carry K chunks in ONE prefill call.

Asserted: the backend sees the SAME chunk replays in the SAME order (one bucket replay per chunk, nothing eager),
the KV / KDA state after the prompt is bitwise equal, the first token is equal, and the host waits (chunks that run
the head and read) drop from one per chunk to one per prompt. Run under the no-tt-devices sandbox:

    GLM53_MOCK=1 no-tt-devices python -m pytest --noconftest -q models/.../tests/s16/test_chunk_run.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mock_model as M  # noqa: E402
from host_evidence import expect_error  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

CHUNK = 2048


class RecordingModel(M.MockModel):
    """the MockModel with the backend's prefill_chunk calls recorded: (chunk_start, n_valid, head, sampled)"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls: list[tuple] = []

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None):
        self.calls.append((int(chunk_start), int(n_valid), logits_rows != "none", sample is not None))
        return super().prefill_chunk(
            tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start, n_valid=n_valid,
            logits_rows=logits_rows, sample=sample, io=io,
        )  # fmt: skip


def prompt(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(1234 + 7 * n + seed)
    return torch.randint(0, M.TOKENIZER_VOCAB, (n,), generator=g).tolist()


def steps_of(n: int, run: int) -> list[tuple[int, int]]:
    """vLLM's chunk ends for a lone request: the first step one bucket (TTFT chunk of a fresh request), then
    `run` buckets per continuation step (TTScheduler._prefill_chunk_run_budget), the last step the remainder"""
    out, s = [], 0
    while s < n:
        e = min(n, s + (CHUNK if s == 0 else run * CHUNK))
        out.append((s, e))
        s = e
    return out


def drive(kinds, n: int, run: int, mode: str, dsa_mode: str = "dense_exact"):
    """returns (first token, state snapshot, backend calls, host waits, chunks per call max, wall s)"""
    m = RecordingModel(kinds, max_batch=2, dsa_mode=dsa_mode, seed=1)
    g = G.Glm53Generator(m, max_batch=2, max_seq_len=n + CHUNK, stop_token_ids=())
    kv = g.allocate_kv_cache(2, n + CHUNK)
    g.warmup(kv)
    ids = torch.tensor(prompt(n))
    bps = G.blocks_per_seq(n + CHUNK)
    pt = torch.arange(2 * bps).reshape(2, bps)
    m.calls.clear()
    waits = 0
    first = None
    t0 = time.perf_counter()
    for s, e in steps_of(n, run):
        final = e == n
        head = None if mode == "served" else [final]
        out = g.prefill_forward(
            ids[s:e][None], page_table=pt[:1], kv_cache=kv, prompt_lens=[e - s], start_pos=[s],
            sampling_params=[SamplingParams()], head_rows=head,
        )  # fmt: skip
        if head is None or final:
            waits += 1  # the call read the sampled token: a host wait on the device
        if final:
            first = int(out[0])
    wall = time.perf_counter() - t0
    snap = []
    for c in kv.caches:
        if c.state is not None:
            snap.append(("state", c.state[0].clone()))
            snap.append(("conv", c.conv[0].clone()))
        else:
            snap.append(("latent", c.latent[pt[0]].clone()))
    per_call = max((e - s + CHUNK - 1) // CHUNK for s, e in steps_of(n, run))
    return first, snap, list(m.calls), waits, per_call, wall


def _same_state(a, b) -> bool:
    """bitwise equality; the mock fills unwritten latent pages with NaN, so NaN == NaN on both sides"""

    def eq(x, y):
        return x.shape == y.shape and bool(torch.all((x == y) | (torch.isnan(x) & torch.isnan(y))))

    return len(a) == len(b) and all(x[0] == y[0] and eq(x[1], y[1]) for x, y in zip(a, b))


def _record(row: dict):
    row = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    print("CHUNK_RUN_MOCK " + json.dumps(row), flush=True)
    path = os.environ.get("GLM53_CHUNKRUN_EVIDENCE")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    "n, run, kinds, dsa_mode",
    [
        (32000, 4, ("dense_kda", "sparse_dsa", "sparse_kda"), "unbounded"),  # the served ladder's 32K request
        (131072, 8, ("dense_kda", "sparse_kda"), "dense_exact"),  # 128K: the KDA kinds (the mock's MLA is O(T^2))
    ],
)
def test_chunk_run_same_replays_bitwise_state_one_wait(n, run, kinds, dsa_mode):
    chunks = -(-n // CHUNK)
    served = drive(kinds, n, 1, "served", dsa_mode)
    headless = drive(kinds, n, 1, "headless", dsa_mode)
    pipelined = drive(kinds, n, run, "headless", dsa_mode)

    # the same bucket replays in the same order: (chunk_start, n_valid) per backend call, one call per chunk
    key = lambda calls: [(s, v) for s, v, _, _ in calls]  # noqa: E731
    assert key(served[2]) == key(headless[2]) == key(pipelined[2])
    assert len(served[2]) == chunks
    assert [s for s, _, _, _ in served[2]] == [i * CHUNK for i in range(chunks)]
    # served: the head + sample on every chunk; headless / run: only on the prompt's last chunk
    assert all(h and smp for _, _, h, smp in served[2])
    assert [h for _, _, h, _ in headless[2]] == [False] * (chunks - 1) + [True]
    assert [h for _, _, h, _ in pipelined[2]] == [False] * (chunks - 1) + [True]
    # host waits (a call that reads the device): one per chunk -> one per prompt
    assert served[3] == chunks and headless[3] == 1 and pipelined[3] == 1
    # bitwise: the first token and every slot-0 state / latent page after the prompt
    assert served[0] == headless[0] == pipelined[0]
    assert _same_state(served[1], headless[1]) and _same_state(served[1], pipelined[1])
    steps = steps_of(n, run)
    _record(
        {
            "test": "chunk_run_mock",
            "n": n,
            "chunks": chunks,
            "run": run,
            "steps_served": chunks,
            "steps_run": len(steps),
            "chunks_per_step_run": [(e - s + CHUNK - 1) // CHUNK for s, e in steps],
            "bucket_replays_per_chunk": 1,
            "eager": 0,
            "host_waits": {"served": served[3], "headless": headless[3], "run": pipelined[3]},
            "headless_chunks": {"served": 0, "headless": chunks - 1, "run": chunks - 1},
            "first_token": served[0],
            "state_bitwise": True,
            "mock_wall_s": {
                "served": round(served[5], 2),
                "headless": round(headless[5], 2),
                "run": round(pipelined[5], 2),
            },
        }
    )


def test_head_rows_contract():
    m = RecordingModel(("dense_kda",), max_batch=2, seed=1)
    g = G.Glm53Generator(m, max_batch=2, max_seq_len=4096, stop_token_ids=())
    kv = g.allocate_kv_cache(2, 4096)
    g.warmup(kv)
    bps = G.blocks_per_seq(4096)
    pt = torch.arange(2 * bps).reshape(2, bps)
    ids = torch.tensor(prompt(3000))
    # an intermediate step without sampling params returns a zero logits row and reads nothing
    out = g.prefill_forward(ids[:2048][None], page_table=pt[:1], kv_cache=kv, prompt_lens=[2048], head_rows=[False])
    assert out.shape == (1, 1, m.vocab_size) and not out.any()
    assert m.calls[-1][2] is False and g.counters["prefill_chunks_headless"] == 1
    # the final chunk samples; the slot continues from the head-less chunk's state
    tok = g.prefill_forward(
        ids[2048:][None], page_table=pt[:1], kv_cache=kv, prompt_lens=[952], start_pos=[2048],
        sampling_params=[SamplingParams()], head_rows=[True],
    )  # fmt: skip
    assert int(tok[0]) >= 0 and m.calls[-1][2] is True
    # a run of two chunks in one call: counters
    g2 = G.Glm53Generator(
        RecordingModel(("dense_kda",), max_batch=2, seed=1), max_batch=2, max_seq_len=4096, stop_token_ids=()
    )
    kv2 = g2.allocate_kv_cache(2, 4096)
    g2.warmup(kv2)
    g2.prefill_forward(
        ids[None], page_table=pt[:1], kv_cache=kv2, prompt_lens=[3000], sampling_params=[SamplingParams()]
    )
    assert g2.counters["prefill_runs"] == 1 and g2.counters["prefill_run_chunks_max"] == 2
    assert g2.counters["prefill_chunks_headless"] == 1  # the first chunk of the run: head-less, no wait
    with expect_error(ValueError, "one head_rows entry per row"):
        g2.prefill_forward(ids[None], page_table=pt[:1], kv_cache=kv2, prompt_lens=[3000], head_rows=[False, True])
