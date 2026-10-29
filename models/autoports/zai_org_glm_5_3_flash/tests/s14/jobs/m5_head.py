# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): M5 piece — the TP vocab-sharded lm_head (tt/tp8_head.py) as a DS reader-phase program on every chip.
h = random [1,1,B,4096] bf16 rows (replicated); compile launch, traced replay -> every chip's logits shard, gathered on
host in vocab order; vs the CPU oracle (h @ lm_head, fp32 over the exact bfp8 weights): PCC, max-abs, top-1 / top-5
agreement; traced single-launch median and steady state of `chain` back-to-back launches.
ARGS: b (1), reps (20), chain (8), record (False), tag, kb (8), splits (None = 8 / tp)."""
import gc
import importlib
import os
import sys

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.tp8_head as TPH  # noqa: E402

importlib.reload(TPH)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, REPS, CHAIN = int(args.get("b", 1)), int(args.get("reps", 20)), int(args.get("chain", 8))
key = ("head", int(args.get("kb", 8)), args.get("splits"))
if key not in ctx or type(ctx[key]) is not TPH.TPHead:
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "head"]:
        ctx.pop(k)
    gc.collect()
    ctx[key] = TPH.TPHead(mesh, FdTileCacheSource(CACHE_DIR, MANIFEST), OptimizedDecoderConfig(), kb=int(args.get("kb", 8)),
                          splits=args.get("splits"))  # fmt: skip
head = ctx[key]
g = torch.Generator().manual_seed(1)
h_host = (torch.randn(1, 1, B, 4096, generator=g) * 0.5).to(torch.bfloat16)
h = ttnn.from_torch(h_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                    memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
tr = C.Traced(ttnn, mesh, lambda: head(h))
tr.replay()
got = head.gather(tr.out)
walls = tr.replays(REPS)
tr.release()
st = C.stats(walls)
chain = None
if CHAIN > 1:
    trc = C.Traced(ttnn, mesh, lambda: [head(h) for _ in range(CHAIN)], warm=False)
    wc = C.stats(trc.replays(REPS))
    trc.release()
    chain = dict(n=CHAIN, median_us=wc["median_us"], steady_per_launch_us=round((wc["median_us"] - st["median_us"]) / (CHAIN - 1), 2))
want = h_host.float().reshape(B, 4096) @ head.host_w.float()
p = C.pcc(got, want)
mae = float((got - want).abs().max())
top1 = float((got.argmax(-1) == want.argmax(-1)).float().mean())
t5 = got.topk(5, dim=-1).indices
w5 = want.topk(5, dim=-1).indices
top5 = float(sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(t5, w5)) / (5 * B))
row = dict(test=f"tp lm_head B={B} tp{mesh.get_num_devices()} {CTX['mgd']}", status="pass" if p >= 0.999 else "fail",  # noqa: F821
           milestone="M5-piece", B=B, tp=mesh.get_num_devices(), pcc_vs_cpu=p, max_abs_vs_cpu=mae, top1_agree=top1,
           top5_overlap=top5, traced=True, us=st["median_us"], replay=st, chain=chain, splits=head.splits,
           cols_per_chip=head.w, tag=args.get("tag", "m5-head"))  # fmt: skip
if args.get("record"):
    C.record("perf", row)
print("M5HEAD", row, flush=True)
RESULT = row
