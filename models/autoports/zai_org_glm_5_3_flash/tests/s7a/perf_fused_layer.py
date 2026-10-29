# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a single-layer decode A/B of the fused streaming programs, one process, one layer, one chip:

    python -m tracy -r -p -v -o <dir> models/autoports/zai_org_glm_5_3_flash/tests/s7a/perf_fused_layer.py \
        --layer 4 --batches 1,8,32 [--variants "BASE:,MHC:mhc_pre,ALL:mhc_pre+kda_pre+router_tail+moe_post"]

The layer is loaded ONCE with every selected fused program built; each variant toggles the program bindings on the same
layer object (BASE = today's op chain, FUSED = the selected programs), captures the decode trace of the layer at each
batch (distinct users at golden positions, as tests/test_optimized_perf.py) and replays it inside signposted windows
PERF_<VARIANT>_DECODE_B<n> .. _END (DECODE_REPLAYS replays; under the device profiler every replay is synchronised and
read out). The FUSED output of each replay is compared with BASE's (same inputs, same state restored): PCC rows.
Without the profiler the same script gives the warmed traced wall clock (WALL_REPLAYS replays). Rows ->
$GLM53_STAGE_EVIDENCE/perf.jsonl and pcc.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, hf_config, int_tensor, pcc, to_dev
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder, open_mesh
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


PROFILING = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
DECODE_REPLAYS = int(os.environ.get("GLM53_PERF_DECODE_REPLAYS", "4"))
WALL_REPLAYS = int(os.environ.get("GLM53_PERF_WALL_REPLAYS", "20"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", "/tmp")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("perf_%Y%m%dT%H%M%SZ", time.gmtime()))
PREFILL_LEN = 2048


def _flush(mesh):
    if PROFILING:
        ttnn.ReadDeviceProfiler(mesh)


def set_variant(dec, programs: set):
    """Bind / unbind the fused programs on the loaded layer (BASE = empty set). The programs were all built at load;
    unbinding restores today's op chain on the same layer object (same weights, same state)."""
    lay = dec.layer
    on = "mhc_pre" in programs
    lay.ocfg.fused_mhc_pre = on
    if hasattr(lay, "fused_mhc"):
        attn, ffn = lay.fused_mhc
        lay.attn_hc.fused = lay.attn_hc_sm.fused = attn if on else None
        lay.ffn_hc.fused = lay.ffn_hc_sm.fused = ffn if on else None
    lay._mhc_pending = None
    if dec.is_kda:
        if not hasattr(lay.attn, "_kda_pre_built"):
            lay.attn._kda_pre_built = lay.attn.kda_pre
        lay.attn.kda_pre = lay.attn._kda_pre_built if "kda_pre" in programs else None
        if not hasattr(lay.attn, "_kda_rec_built"):
            lay.attn._kda_rec_built = getattr(lay.attn, "kda_rec", None)
        lay.attn.kda_rec = lay.attn._kda_rec_built if "kda_rec" in programs else None
    if lay.spec.mlp == "moe":
        router = lay.mlp.router
        if not hasattr(router, "_tail_built"):
            router._tail_built = router.tail
        router.tail = router._tail_built if "router_tail" in programs else None
    if not hasattr(lay, "_mhc_post_built"):
        lay._mhc_post_built = getattr(lay, "mhc_post", None)
    lay.mhc_post = lay._mhc_post_built if "moe_post" in programs else None  # needs mhc_pre (its coefficient tile)


def parse_variants(spec: str) -> list:
    """ "BASE:,MHC:mhc_pre,ALL:mhc_pre+kda_pre" -> [(name, {programs})]."""
    out = []
    for item in filter(None, spec.split(",")):
        name, _, progs = item.partition(":")
        out.append((name, set(filter(None, progs.split("+")))))
    return out


def write(name, row):
    with open(os.path.join(OUT, name), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--variants", default="BASE:,MHC:mhc_pre,ALL:mhc_pre+kda_pre+router_tail+moe_post")
    a = ap.parse_args()
    variants = parse_variants(a.variants)
    programs = set().union(*(p for _, p in variants))
    batches = [int(b) for b in a.batches.split(",")]
    cfg = hf_config()
    mesh = open_mesh(trace_region_size=int(os.environ.get("GLM53_TRACE_REGION", "1000000000")))
    try:
        dec = load_optimized_decoder(
            cfg,
            a.layer,
            mesh,
            config=OptimizedDecoderConfig(
                fused_mhc_pre="mhc_pre" in programs,
                fused_kda_pre="kda_pre" in programs,
                fused_router_tail="router_tail" in programs,
                fused_moe_post="moe_post" in programs,
                fused_kda_rec="kda_rec" in programs,
            ),
        )
        kind = ("dense" if dec.spec.mlp == "dense" else "sparse") + ("_kda" if dec.is_kda else "_dsa")
        g = golden(a.layer, PREFILL_LEN)
        outs = {}
        for variant, progs in variants:
            set_variant(dec, progs)
            for batch in batches:
                cache = dec.allocate_cache(
                    max_batch=batch, max_seq_len=PREFILL_LEN + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV
                )
                pt = None if dec.is_kda else dec.make_page_table(cache)
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
                    y = trace.replay()
                ttnn.synchronize_device(mesh)
                outs[(variant, batch)] = ttnn.to_torch(y).float()
                _flush(mesh)
                reps = DECODE_REPLAYS if PROFILING else WALL_REPLAYS
                tag = f"PERF_{variant}_DECODE_B{batch}"
                signpost(tag)
                t0 = time.perf_counter()
                for _ in range(reps):
                    trace.replay(blocking=False)
                    if PROFILING:
                        ttnn.synchronize_device(mesh)
                        _flush(mesh)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) * 1e3 / reps
                _flush(mesh)
                signpost(f"{tag}_END")
                trace.release()
                write("perf.jsonl", dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN,
                                         layer=a.layer, layer_kind=kind, batch=batch, variant=variant,
                                         programs=sorted(progs),
                                         decode_wall_ms_per_step=ms, replays=reps, profiler=PROFILING,
                                         window=tag))  # fmt: skip
                del cache
        for batch in batches:
            for variant, progs in variants:
                if variant == "BASE" or ("BASE", batch) not in outs or (variant, batch) not in outs:
                    continue
                b, f = outs[("BASE", batch)], outs[(variant, batch)]
                write("pcc.jsonl", dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN,
                                        test="layer_fused_vs_base", layer=a.layer, layer_kind=kind, batch=batch,
                                        variant=variant, programs=sorted(progs), pcc=pcc(f, b),
                                        max_abs=float((f - b).abs().max()), status="pass" if pcc(f, b) >= 0.9999 else "fail"))  # fmt: skip
    finally:
        ttnn.synchronize_device(mesh)
        ttnn.close_mesh_device(mesh)
    print("PERF_FUSED_LAYER_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
