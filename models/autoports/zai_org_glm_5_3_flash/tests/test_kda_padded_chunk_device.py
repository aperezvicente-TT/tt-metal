# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-chip device validation of the KDA dead-chunk fix (`tt/kda.py` `keep_if_dead`): a prompt prefilled as ONE padded
chunk whose tail holds fully padded 32-row recurrence chunks must leave the same KDA state as the pure power-of-two
plan (Stage 4 v2 finding: PCC ~0 without the fix; doc/ttnn/04-multichip-decoder §3c), independently of what the op's
output buffers held before.

Per KDA layer kind (L0 dense_kda, L4 sparse_kda; MLA has no recurrence) on ONE chip, real weights, golden activations:
  slot 0: n tokens as one padded chunk of the smallest bucket >= n (Stage 3 run_prefill_trace, n_valid = n)
  slot 1: the same n tokens through the pure plan (plan_traced_chunks: pow2 chunks + tail, Stage 1 contract)
  slot 2: an unrelated 2048-token prompt first (rewrites every bucket buffer of the op), then n tokens padded again
  check: S / C of slot 0 vs slot 1 PCC >= 0.9999 (different chunking: not bitwise) and slot 0 == slot 2 bitwise
n in 65 (128-row chunk, 63 padded), 130 (256, 126), 1999 (2048, 49) and the control 100 (128, 28).
Expected: FAIL at glm53/glm53-flash-s4 (no gate), PASS at glm53/glm53-flash-s4-kdafix.

run (maintenance window, one chip):  tt-lock run chip4 -- python -m pytest -q --timeout 0 -p no:cacheprovider \
    models/autoports/zai_org_glm_5_3_flash/tests/test_kda_padded_chunk_device.py
GLM53_KDA_FIX_EAGER=1 runs the same per-chunk bodies eagerly (no capture: mock-cluster code-path check).
Rows -> $GLM53_OUT_DIR (default $GLM53_WS/doc/ttnn/04-multichip-decoder/kda_fix)/kda_padded_chunk.jsonl.
"""
import json
import os
import time

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests import test_functional_decoder as s1
from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import plan_traced_chunks

EAGER = os.environ.get("GLM53_KDA_FIX_EAGER") == "1"
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
OUT = os.environ.get("GLM53_OUT_DIR", os.path.join(WS, "doc/ttnn/04-multichip-decoder/kda_fix"))
CASES = [(65, True), (130, True), (1999, True), (100, False)]  # (tokens, holds a fully padded 32-row chunk)
cfg = s1.cfg  # pytest fixture (the HF config), as in test_optimized_prefill_trace.py


def record(row):
    os.makedirs(OUT, exist_ok=True)
    row = dict(
        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        eager=EAGER,
        code=os.environ.get("GLM53_CODE_SHA"),
        visible=os.environ.get("TT_VISIBLE_DEVICES"),
        **row,
    )
    with open(os.path.join(OUT, "kda_padded_chunk.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row)[:300], flush=True)


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh(trace_region_size=1 << 30)
    yield m
    ttnn.close_mesh_device(m)


def run_chunk(dec, cache, t, rows, *, slot, start, n):
    """One chunk of `slot` (Stage 3 run_prefill_trace; eager: the same steps around the uncaptured body)."""
    host = dec.prefill_host_chunk(rows, t)
    if not EAGER:
        dec.run_prefill_trace(t, cache, user_slot=slot, chunk_start=start, n_valid=n, chunk=host)
        return
    ttnn.copy_host_to_device_tensor(host, t.x)
    dec.begin_prefill_chunk(cache, user_slot=slot, chunk_start=start, n_valid=n)
    dec._scratch_state_in(cache, slot)
    dec._traced_body(t, cache)
    ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, slot)


def prompt(dec, cache, traces, rows, *, slot, padded):
    n = rows.shape[0]
    if padded:
        p = min(b for b in traces if b >= n)
        run_chunk(dec, cache, traces[p], rows, slot=slot, start=0, n=n)
        return [p]
    plan = plan_traced_chunks(n, dec.prefill_align, tuple(sorted(traces)))
    for off, cn, p in plan:
        run_chunk(dec, cache, traces[p], rows[off : off + cn], slot=slot, start=off, n=cn)
    return [p for _, _, p in plan]


def state(cache, slot):
    S = ttnn.to_torch(cache.kda.S)[slot].float()
    C = ttnn.to_torch(cache.kda.C)[:, :, slot].float()
    return S, C


@pytest.mark.parametrize("layer", [0, 4])
def test_padded_chunk_kda_state(mesh, cfg, layer):
    dec = load_optimized_decoder(cfg, layer, mesh)
    assert dec.is_kda
    xin = golden(layer, 2048)["streams_in"][0]  # [2048, 4, 4096]
    cache = dec.allocate_cache(max_batch=3, max_seq_len=2048 + 64)
    traces = dec.prepare_prefill_traces(cache)
    if not EAGER:
        dec.capture_prefill_traces(cache, traces)
    failures = []
    for n, dead in CASES:
        rows = xin[:n]
        c0 = prompt(dec, cache, traces, rows, slot=0, padded=True)
        c1 = prompt(dec, cache, traces, rows, slot=1, padded=False)
        prompt(dec, cache, traces, xin[:2048], slot=2, padded=False)  # rewrites every bucket's buffers
        prompt(dec, cache, traces, rows, slot=2, padded=True)
        ttnn.synchronize_device(mesh)
        (s0, k0), (s1_, k1), (s2, k2) = state(cache, 0), state(cache, 1), state(cache, 2)
        r = dict(
            test="kda_padded_chunk_state",
            layer=layer,
            layer_kind=s1.KIND_OF[layer],
            tokens=n,
            dead_chunk=dead,
            padded_chunks=c0,
            plan_chunks=c1,
            pcc_S_padded_vs_plan=round(pcc(s0, s1_), 8),
            pcc_C_padded_vs_plan=round(pcc(k0, k1), 8),
            S_padded_repeat_bitexact=bool(torch.equal(s0, s2)),
            C_padded_repeat_bitexact=bool(torch.equal(k0, k2)),
            S_finite=bool(torch.isfinite(s0).all()),
        )
        ok = (
            r["pcc_S_padded_vs_plan"] >= 0.9999
            and r["pcc_C_padded_vs_plan"] >= 0.9999
            and r["S_padded_repeat_bitexact"]
            and r["C_padded_repeat_bitexact"]
            and r["S_finite"]
        )
        record(dict(**r, status="pass" if ok else "fail"))
        if not ok:
            failures.append(n)
    if not EAGER:
        dec.release_prefill_traces(traces)
    assert not failures, f"layer {layer}: padded-chunk KDA state wrong for prompts {failures}"
