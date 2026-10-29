# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 04a: per-chip speed of the same decoder work (the partition assumes identical chips; the 2-chip pipeline test
showed the same layers ~9 % slower on chip 5 than on chip 4 at B=32). One chip per process (1x1 descriptor, tt-lock
chipN): load one sparse_kda layer (L4) and one sparse_dsa layer (L3) with the Stage 2 fused decoder, traced decode at
B=1 and B=32 (Stage 2 perf protocol: warm, capture, 3 x 20 non-blocking replays, median). Rows ->
$GLM53_E4A/chip_speed.jsonl.
"""
import json
import os
import statistics
import time

import torch

os.environ.setdefault(
    "TT_MESH_GRAPH_DESC_PATH",
    "/opt/glm53/tt-metal-s4a/models/autoports/zai_org_glm_5_3_flash/mesh/glm_1x1_mesh_graph_descriptor.textproto",
)
import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder  # noqa: E402

E = os.environ.get(
    "GLM53_E4A", "/opt/glm53/kit/doc/ttnn/04a-multichip-handoff-prework"
)
REPS = 20


def record(row):
    row = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chip": os.environ.get("TT_VISIBLE_DEVICES"),
        **row,
    }
    with open(os.path.join(E, "chip_speed.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row), flush=True)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=300_000_000)
    try:
        cfg = hf_config()
        for layer, kind in ((4, "sparse_kda"), (3, "sparse_dsa")):
            dec = FusedDecoder.from_weight_cache(
                hf_config=cfg, layer_idx=layer, mesh_device=mesh, cache_dir=CACHE_DIR, manifest_path=MANIFEST
            )
            g = golden(layer, 2048)["streams_in"].reshape(-1, 16384).float()
            for batch in (1, 32):
                cache = dec.allocate_cache(max_batch=batch, max_seq_len=2112 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
                pt = None if dec.is_kda else dec.make_page_table(cache)
                pos = list(range(batch))
                x = ttnn.from_torch(
                    g[:batch].reshape(1, 1, batch, 16384).contiguous(),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                cp = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, device=mesh)
                snap = dec.snapshot_state(cache)
                ttnn.deallocate(dec.decode_forward(x, cache=cache, page_table=pt, cur_pos=cp))
                dec.restore_state(cache, snap)
                ttnn.synchronize_device(mesh)
                tr = dec.capture_decode_trace(x, cache=cache, page_table=pt, cur_pos=cp)
                tr.replay(blocking=True)
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        tr.replay(blocking=False)
                    ttnn.synchronize_device(mesh)
                    ts.append((time.perf_counter() - t0) / REPS * 1e3)
                tr.release()
                record(
                    dict(
                        test="chip_speed/decode",
                        layer=layer,
                        layer_kind=kind,
                        batch=batch,
                        decode_ms=round(statistics.median(ts), 4),
                        samples_ms=[round(t, 4) for t in ts],
                    )
                )
                del cache, snap, x, cp
            del dec
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
