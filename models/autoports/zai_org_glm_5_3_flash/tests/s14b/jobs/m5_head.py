# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b — the model head as ONE program (tt/mk_head.py MkHead: final norm on
the owners + the lm_head vocab splits on the DS readers) on the stacked residual of the reference model's LAST layer
(golden streams_out of layer 44 at T positions): traced; the CPU oracle once (h = RMSNorm(mean of the 4 streams) * w in
fp32, rounded to bf16, @ the cache's bfloat8_b lm_head values) -> PCC of the logits, top-1 / top-5 agreement; traced
replay timing. ARGS: t (1), reps (30), tag, record, splits (default 8 / tp), kb (8)."""
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

for mod in (MKR, MKH):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
for k_ in ("GLM53_MK_RING_KB", "GLM53_MK_READERS"):
    os.environ.pop(k_, None)
T = int(args.get("t", 1))
REPS = int(args.get("reps", 30))
tag = args.get("tag", "m5-head")
cfg = hf_config()


def log(*a):
    print(f"M5-HEAD {C.now()}", *a, flush=True)


hk = ("m5_head_host",)
if hk not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    t0 = time.time()
    ctx[hk] = dict(lm=MKH.lm_head_host(source), norm=MKH.norm_host(source))
    log(f"lm_head {tuple(ctx[hk]['lm'].shape)} + norm loaded in {time.time() - t0:.1f}s")
lm_w, norm_w = ctx[hk]["lm"], ctx[hk]["norm"]
TOPK = int(args.get("topk", 5))
key = ("m5_head", T, args.get("splits"), int(args.get("kb", 8)), TOPK)
M5.prune_ctx(ctx, [key])
gc.collect()
if key not in ctx:
    t0 = time.time()
    ctx[key] = MKH.MkHead(mesh, lm_w, norm_w, t=T, eps=cfg.rms_norm_eps, splits=args.get("splits"),
                          kb=int(args.get("kb", 8)), topk=TOPK)  # fmt: skip
    log(f"MkHead built in {time.time() - t0:.1f}s: {ctx[key].splits} splits of {ctx[key].cw} columns")
head = ctx[key]

g = golden(44, 2048)["streams_out"].float()  # [1, P, 4, 4096]? -> rows of 16384
HID = 4096
rows = g.reshape(g.shape[0], g.shape[1], -1)[0, 500 : 500 + 41 * T : 41].reshape(T, 4, HID)
res = ttnn.from_torch(rows.reshape(T, 1, 4, HID), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                      memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
tr = C.Traced(ttnn, mesh, lambda: head(res))
tr.replay()
got = head.gather(tr.out)  # [T, V]
h_dev = ttnn.to_torch(head.h).float().reshape(-1, HID)[:T]
dev_topk = ttnn.to_torch(head.topk_out).to(torch.int64).reshape(32, 64)[:T, :TOPK] if TOPK else None
tr.release()
trt = C.Traced(ttnn, mesh, lambda: head(res), warm=False)
st = C.stats(trt.replays(REPS))
bm = trt.batch(REPS)
trt.release()
# CPU oracle (once): mean over the streams -> RMSNorm (fp32) * w -> bf16 -> @ lm_head (the bfloat8_b values)
m = rows.double().mean(dim=1)
hh = (m * torch.rsqrt(m.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * norm_w.double()).float()
h_ref = hh.to(torch.bfloat16).float()
want = (h_ref.double() @ lm_w.double()).float()[:, : got.shape[-1]]
p_h = C.pcc(h_dev, h_ref)
h_maxabs = float((h_dev - h_ref).abs().max())
p = C.pcc(got, want)
top1 = [int(got[r].argmax()) == int(want[r].argmax()) for r in range(T)]
top5 = [len(set(got[r].topk(5).indices.tolist()) & set(want[r].topk(5).indices.tolist())) for r in range(T)]
topk_exact = None
if TOPK:  # device top-K vs the same device logits, host-sorted by (value desc, vocab id asc): exact
    want_ids = []
    for r_ in range(T):
        v = got[r_]
        vals, idx = torch.sort(v, descending=True, stable=True)  # stable: equal values keep the lower id first
        want_ids.append(idx[:TOPK])
    want_ids = torch.stack(want_ids)
    topk_exact = bool(torch.equal(dev_topk, want_ids))
    log("TOPK device", dev_topk.tolist(), "host", want_ids.tolist())
ok = p >= 0.999 and p_h >= 0.9999 and all(top1) and (topk_exact is not False)
r = dict(test=f"m5 head: final norm + lm_head ({head.splits} vocab splits) in ONE program T={T}", status="pass" if ok
         else "fail", milestone="M5b-piece", T=T, pcc_logits_vs_cpu=p, pcc_h_vs_cpu=p_h, h_max_abs=h_maxabs,
         top1_agree=top1, top5_overlap=top5, splits=head.splits, vocab_cols=head.cw * head.splits, topk=TOPK,
         device_topk_exact_vs_device_logits=topk_exact, tag=tag)  # fmt: skip
prow = dict(test=f"m5 head T={T} step", status="info", milestone="M5b-piece", traced=True, T=T, us=st["median_us"],
            replay=st, batch_mean_us=round(bm, 1) if bm else None, splits=head.splits, tag=tag,
            note="ONE program: final norm (64 owners) + lm_head 4096 x 154,880 bf8 (DS readers), logits to DRAM; "
                 "single chip (all vocab)")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, us=st["median_us"], pcc=p, top1=top1)
