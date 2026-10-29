# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd 2x4 | pair): Stage 14b M5b piece — the model head on the TP mesh (tt/mk_head.py MkHead: the final
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


key = ("m5b_tp_head", T, TOPK)
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
res = ttnn.from_torch(rows.reshape(T, 1, 4, HID), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                      memory_config=ttnn.L1_MEMORY_CONFIG,
                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None)  # fmt: skip
tr = C.Traced(ttnn, mesh, lambda: head(res))
tr.replay()
got = head.gather(tr.out)[:, :154880]
cand = head.topk_host()  # [tp, T, K]
hs = [ttnn.to_torch(d).float().reshape(-1, HID)[:T] for d in ttnn.get_device_tensors(head.h)]
tr.release()
trt = C.Traced(ttnn, mesh, lambda: head(res), warm=False)
st = C.stats(trt.replays(REPS))
bm = trt.batch(REPS)
trt.release()
m = rows.double().mean(dim=1)
h_ref = (m * torch.rsqrt(m.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w.double()).float()
h_ref = h_ref.to(torch.bfloat16).float()
want = (h_ref.double() @ lm_w.double()).float()
h_same = all(torch.equal(h, h_ref) for h in hs)
p = C.pcc(got, want)
top1 = [int(got[r].argmax()) == int(want[r].argmax()) for r in range(T)]
top5 = [len(set(got[r].topk(5).indices.tolist()) & set(want[r].topk(5).indices.tolist())) for r in range(T)]
W_ = head.w
shard_exact, merged_exact = True, True
for r in range(TPN):  # chip r's candidates vs its own shard of the device logits (sorted: value desc, id asc)
    lo = W_ * r
    sh = got[:, lo : min(lo + W_, 154880)]
    _, idx = torch.sort(sh, dim=-1, descending=True, stable=True)
    shard_exact &= bool(torch.equal(cand[r], idx[:, :TOPK] + lo))
_, gidx = torch.sort(got, dim=-1, descending=True, stable=True)
for t_ in range(T):  # host merge of the chips' candidates == the global top-K
    pool = sorted({int(i) for i in cand[:, t_].reshape(-1)}, key=lambda i: (-float(got[t_, i]), i))[:TOPK]
    merged_exact &= pool == gidx[t_, :TOPK].tolist()
ok = p >= 0.999 and h_same and all(top1) and shard_exact and merged_exact
r = dict(test=f"m5b tp head: final norm + lm_head vocab shard + device top-{TOPK} per chip, tp{TPN} T={T}",
         status="pass" if ok else "fail", milestone="M5b-piece", T=T, tp=TPN, mgd=CTX["mgd"], pcc_logits_vs_cpu=p,  # noqa: F821
         h_bitwise_vs_cpu_all_chips=h_same, top1_agree=top1, top5_overlap=top5, per_chip_topk_exact=shard_exact,
         host_merged_topk_exact=merged_exact, vocab_cols_per_chip=W_, tag=tag)  # fmt: skip
prow = dict(test=f"m5b tp head tp{TPN} T={T} step", status="info", milestone="M5b-piece", traced=True, T=T, tp=TPN,
            us=st["median_us"], replay=st, batch_mean_us=round(bm, 1) if bm else None, topk=TOPK, tag=tag,
            note="ONE program per chip: final norm + this chip's lm_head vocab shard + device top-K of the shard "
                 "(the cross-chip merge is not in-program yet)")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, us=st["median_us"], pcc=p)
