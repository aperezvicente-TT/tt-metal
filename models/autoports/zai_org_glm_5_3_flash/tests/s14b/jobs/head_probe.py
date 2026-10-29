# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd 2x4 | pair; outside the tree): Stage 14b head PROBE (GLM53_HEAD_PROBE=1): per-role phase marks of the head program, FLAT residual for T > 4 — the model head on the TP mesh (tt/mk_head.py MkHead: the final
norm replicated on every chip, chip r's lm_head vocab shard [W r, W (r + 1)) as DS reader phases, device top-K of its
shard) on the replicated stacked residual of the reference model's last layer (golden L44 streams_out at T positions):
traced; the CPU oracle once (logits of the whole vocab) -> PCC, top-1 / top-5; every chip's top-K exact vs its own logits
shard; the chips' candidates merged on the host == the top-K of the gathered logits; traced replay timing.
ARGS: t (1), reps (30), topk (5), tag, record."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402

for mod in (M5, MKR, MKH):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
for k_ in ("GLM53_MK_RING_KB", "GLM53_MK_READERS"):
    os.environ.pop(k_, None)
T = int(args.get("t", 1))
REPS = int(args.get("reps", 30))
TOPK = int(args.get("topk", 5))
tag = args.get("tag", "m5b-tp-head")
TPN = int(mesh.get_num_devices())
cfg = hf_config()


def log(*a):
    print(f"M5B-HEAD {C.now()}", *a, flush=True)


os.environ["GLM53_HEAD_PROBE"] = "1"
key = ("m5b_tp_head_probe", T, TOPK)
M5.prune_ctx(ctx, [key])
hk = ("m5_head_host",)
if hk not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    ctx[hk] = dict(lm=MKH.lm_head_host(source), norm=MKH.norm_host(source))
if key not in ctx:
    t0 = time.time()
    ctx[key] = MKH.MkHead(mesh, ctx[hk]["lm"], ctx[hk]["norm"], t=T, eps=cfg.rms_norm_eps, topk=TOPK)
    log(f"TP head built in {time.time() - t0:.1f}s: tp {TPN}, {ctx[key].splits} split(s) of {ctx[key].cw} columns")
head = ctx[key]
lm_w, norm_w = ctx[hk]["lm"], ctx[hk]["norm"]
g = golden(44, 2048)["streams_out"].float()
HID = 4096
rows = g.reshape(g.shape[0], g.shape[1], -1)[0, 500 : 500 + 41 * T : 41].reshape(T, 4, HID)
# T > 4: the flat residual [1, 1, T, 4 x HID] in DRAM (row u = its 4 streams back to back: the native-t layout the
# serving head reads), else the stacked [T, 1, 4, HID] in L1
flat_in = T > 4
res = ttnn.from_torch(rows.reshape(1, 1, T, 4 * HID) if flat_in else rows.reshape(T, 1, 4, HID), dtype=ttnn.float32,
                      layout=ttnn.TILE_LAYOUT, device=mesh,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG if flat_in else ttnn.L1_MEMORY_CONFIG,
                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None)  # fmt: skip
tr = C.Traced(ttnn, mesh, lambda: head(res))
for _ in range(5):
    tr.replay()
st = C.stats(tr.replays(int(args.get("reps", 200))))
tr.release()
os.environ.pop("GLM53_HEAD_PROBE", None)
nread = len(head.readers.assign)
summ = []
for chip, rows in enumerate(head.probe_rows()):
    sel = [int(v) for v in rows[2 * nread].tolist()]
    sel = [v for v in sel if v]
    d = lambda a, b: (b - a) & 0xFFFFFFFF  # noqa: E731
    sel_d = [d(sel[i], sel[i + 1]) for i in range(len(sel) - 1)]
    movers = []
    for i in range(nread):
        m = [int(v) for v in rows[2 * i + 1].tolist() if v]
        movers.append([d(m[j], m[j + 1]) for j in range(len(m) - 1)])
    streams = []
    for i in range(nread):
        w = [int(v) for v in rows[2 * i].tolist() if v]
        streams.append(d(w[0], w[-1]) if len(w) > 1 else 0)
    mv = torch.tensor([x for x in movers if x], dtype=torch.float64)
    summ.append(dict(chip=chip, select_marks=len(sel), select_spans=sel_d, select_total=d(sel[0], sel[-1]) if sel else 0,
                     mover_spans_median=[float(v) for v in mv.median(0).values] if len(mv) else [],
                     mover_spans_max=[float(v) for v in mv.max(0).values] if len(mv) else [],
                     stream_total_median=float(torch.tensor(streams, dtype=torch.float64).median()) if streams else 0))
res = dict(T=T, topk=TOPK, tp=TPN, nread=nread, replay_median_us=st["median_us"], probe=summ[:2])
log("PROBE", res)
RESULT = dict(ok=True, replay_median_us=st["median_us"], probe=summ[:1])
