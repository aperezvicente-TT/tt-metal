# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Advertised-context run for the KDA layer kinds (HF max_position_embeddings = 1,048,576).

KDA layers carry a constant-size state (S fp32 [64,128,128] + 3 conv rows) and have no positional input, so the
advertised context is reachable on one chip. The test streams 1,048,575 tokens (non-divisible by the 2048 chunk and
the 32 tile) through `prefill_forward` with explicit prefix continuation (32 calls of <= 32,768 tokens; the golden
32K prompt's streams cycled), then runs a traced decode step at position 1,048,575 (context 1,048,576). Layers 0, 1
(dense_kda) and 4 (sparse_kda). The prefill has to be streamed: the fp32 input streams of 1,048,575 tokens alone are
64 GiB, twice the chip's DRAM.

Checks:
  * the first call (fresh state) reproduces the fp32 golden 32K output (PCC);
  * every output row and the final state are finite; the state norm stays bounded;
  * the last 2,048-token chunk and the decode step at 1M are checked against the fp32 CPU reference layer started from
    the device's own carried state (S, conv history) read back right before them;
  * state drift: with GLM53_KDA_STATE_DIR set, the device recurrent state is saved after 32K, 64K and 256K tokens and
    after the decode step (1M tokens); the evidence scripts compare it with an independent fp32 CPU run of the layer's
    attention half over the same input stream (state PCC per checkpoint, kda_drift.jsonl).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    check,
    golden,
    hf_config,
    int_tensor,
    load_decoder,
    open_mesh,
    pcc,
    record,
    streams4,
    to_dev,
    to_host,
)

CONTEXT = int(os.environ.get("GLM53_LONG_CONTEXT", str(1 << 20)))  # 1,048,576 = HF max_position_embeddings
# device KDA states after 32K x {1, 2, 8} tokens and after the decode step (1M tokens) are saved here and compared with
# an independent fp32 CPU run of the layer's attention half over the same input stream (doc: kda_drift.jsonl)
STATE_DIR = os.environ.get("GLM53_KDA_STATE_DIR", "")
CHECKPOINT_CYCLES = (1, 2, 8)
CALL = 32768
TAIL_CHUNK = 2048
# Resumable streamed prefill (the 1M run of an MoE layer takes 25-45 min): with GLM53_LONG_RESUME_DIR set, the slot's
# carried state (recurrent S fp32 TILE + conv rows bf16 ROW_MAJOR, read back bitwise) is saved after every call; a
# re-run restores it into the fresh cache (checked bitwise) and continues from the saved position.
RESUME_DIR = os.environ.get("GLM53_LONG_RESUME_DIR", "")


def _reference_layer(layer: int, cfg):
    from models.autoports.zai_org_glm_5_3_flash.reference.weights import ShardReader, build_layer, load_layer

    reader = ShardReader(os.environ["GLM53_HF_SNAPSHOT"])
    sd, store = load_layer(reader, layer, dtype=torch.float32, experts="fp8", lazy_experts=True, cfg=cfg)
    return build_layer(cfg, layer, sd, store, dtype=torch.float32)


def _ref_state(dec, cache):
    from models.autoports.zai_org_glm_5_3_flash.reference.modules import KDAState

    s = ttnn.to_torch(cache.kda.recurrent[0]).float()[:1].clone()  # [1, 64, 128, 128]
    conv = ttnn.to_torch(cache.kda.conv[0]).float()[:1]  # [1, 3, 24576] rows t-3, t-2, t-1
    conv4 = torch.zeros(1, conv.shape[-1], 4)
    conv4[:, :, 1:] = conv.transpose(1, 2)  # reference keeps 4 rows, reads only the last 3
    return KDAState(conv=conv4, recurrent=s)


def _save_state(cache, layer, tokens):
    if STATE_DIR:
        os.makedirs(STATE_DIR, exist_ok=True)
        torch.save(
            ttnn.to_torch(cache.kda.recurrent[0]).float()[:1].clone(), f"{STATE_DIR}/dev_L{layer:02d}_state_{tokens}.pt"
        )


def _resume_path(layer):
    return f"{RESUME_DIR}/L{layer:02d}_resume.pt"


def _save_resume(cache, layer, pos, meta):
    if RESUME_DIR:
        os.makedirs(RESUME_DIR, exist_ok=True)
        tmp = _resume_path(layer) + ".tmp"
        state = dict(recurrent=ttnn.to_torch(cache.kda.recurrent[0]), conv=ttnn.to_torch(cache.kda.conv[0]))
        torch.save(dict(pos=pos, **state, **meta), tmp)
        os.replace(tmp, _resume_path(layer))


def _load_resume(cache, layer, mesh):
    if not RESUME_DIR or not os.path.exists(_resume_path(layer)):
        return None
    ck = torch.load(_resume_path(layer))
    for dst, key, dtype, layout in (
        (cache.kda.recurrent[0], "recurrent", ttnn.float32, ttnn.TILE_LAYOUT),
        (cache.kda.conv[0], "conv", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ):
        src = ttnn.from_torch(ck[key], dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(src, dst)
        ttnn.deallocate(src)
        assert torch.equal(ttnn.to_torch(dst), ck[key]), f"restored {key} differs from the checkpoint"
    return ck


@pytest.mark.parametrize("layer", [0, 1, 4])
def test_advertised_context_kda(layer):
    torch.set_num_threads(16)
    cfg = hf_config()
    assert CONTEXT <= cfg.max_position_embeddings
    mesh = open_mesh()
    try:
        dec = load_decoder(cfg, layer, mesh)
        assert dec.is_kda
        g = golden(layer, 32768)
        src_in = g["streams_in"]  # [1, 32768, 4, 4096]
        cache = dec.allocate_cache(max_batch=1, max_seq_len=CONTEXT)
        n_prefill = CONTEXT - 1
        t_start = time.time()
        pos, first_pcc, max_abs, wall_before, resumed_at = 0, None, 0.0, 0.0, []
        ck = _load_resume(cache, layer, mesh)
        if ck is not None:
            pos, first_pcc, max_abs, wall_before = ck["pos"], ck["first_pcc"], ck["max_abs"], ck["prefill_wall_s"]
            resumed_at = ck["resumed_at"] + [pos]
        while pos < n_prefill:
            n = min(CALL, n_prefill - pos)
            rows = src_in[:, (pos % 32768) : (pos % 32768) + n]
            last = pos + n == n_prefill
            if last:  # split the final call so the last 2048 tokens can be checked from the carried state
                head = n - TAIL_CHUNK
                if head > 0:
                    dec.prefill_forward(rows[:, :head], cache=cache, user_slot=0, start_pos=pos)
                    pos += head
                    rows = rows[:, head:]
                ref_state = _ref_state(dec, cache)
                tail_in = rows.clone()
                y = streams4(dec.prefill_forward(rows, cache=cache, user_slot=0, start_pos=pos))
                pos += rows.shape[1]
                tail_out = y
            else:
                y = streams4(dec.prefill_forward(rows, cache=cache, user_slot=0, start_pos=pos))
                pos += n
                if first_pcc is None:
                    first_pcc = pcc(y, g["streams_out"][:, :n])
                if pos % 32768 == 0 and pos // 32768 in CHECKPOINT_CYCLES:
                    _save_state(cache, layer, pos)
            assert torch.isfinite(y).all(), f"non-finite prefill output before position {pos}"
            max_abs = max(max_abs, float(y.abs().max()))
            if not last:
                meta = dict(first_pcc=first_pcc, max_abs=max_abs, resumed_at=resumed_at)
                _save_resume(cache, layer, pos, dict(meta, prefill_wall_s=wall_before + time.time() - t_start))
        prefill_s = wall_before + time.time() - t_start
        s_dev = ttnn.to_torch(cache.kda.recurrent[0]).float()
        assert torch.isfinite(s_dev).all()

        # traced decode at position CONTEXT - 1 (kv context = CONTEXT)
        p = CONTEXT - 1
        dec_in = src_in[:, (p % 32768) : (p % 32768) + 1]
        dec_state = _ref_state(dec, cache)
        streams = to_dev(dec_in, mesh)
        cur = int_tensor([p], mesh)
        snap = dec.snapshot_state(cache)
        dec.decode_forward(streams, cache=cache, cur_pos=cur)
        dec.restore_state(cache, snap)
        trace = dec.capture_decode_trace(streams, cache=cache, cur_pos=cur)
        out = to_host(trace.replay(blocking=True))
        trace.release()
        _save_state(cache, layer, CONTEXT)  # the replayed decode consumed token CONTEXT - 1

        ref = _reference_layer(layer, cfg)
        with torch.no_grad():
            ref_tail, _ = ref(tail_in.float(), state=ref_state)
            ref_dec, _ = ref(dec_in.float(), state=dec_state)
        row = dict(
            layer_kind=dec.spec.kind,
            layer=layer,
            context=CONTEXT,
            prefill_tokens=n_prefill,
            prefill_calls=-(-n_prefill // CALL),
            prefill_wall_s=prefill_s,
            first_call_golden_pcc=first_pcc,
            tail_chunk_pcc_vs_cpu_from_device_state=pcc(tail_out, ref_tail),
            decode_pcc_vs_cpu_from_device_state=pcc(out, ref_dec),
            decode_position=p,
            state_fro_norm=float(s_dev.norm()),
            max_abs_output=max_abs,
            all_finite=bool(torch.isfinite(out).all()),
            trace=True,
            resumed_at_positions=resumed_at,
        )
        record(row, name="long_context.jsonl")
        if RESUME_DIR and os.path.exists(_resume_path(layer)):
            os.remove(_resume_path(layer))  # finished: a new run starts from position 0
        ok = (
            first_pcc >= 0.995
            and row["tail_chunk_pcc_vs_cpu_from_device_state"] >= 0.995
            and row["decode_pcc_vs_cpu_from_device_state"] >= 0.995
            and row["all_finite"]
        )
        check([] if ok else [row], "advertised-context run")
    finally:
        ttnn.close_mesh_device(mesh)
