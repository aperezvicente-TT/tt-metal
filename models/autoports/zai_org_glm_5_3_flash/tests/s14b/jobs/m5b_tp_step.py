# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd 2x4 | pair): Stage 14b M5b — a whole TP decode step, token in -> token out, as ONE trace on the
mesh with no host work inside: MkEmbed (replicated tokens -> replicated stacked residual) -> the dense_kda layers as ONE
TP stack program per chip (tt/tp_stack.py: layer loop + 2 in-program fabric all-reduces per layer) -> MkHead (final
norm, this chip's lm_head vocab shard, device top-K of the shard, the chips' candidates merged IN the program with the
same fabric phase -> every chip writes the same token rows = the next step's MkEmbed input).

Checks (device vs device): every step of an S-step replay run: the token rows are identical on every chip and equal
the top-K of the gathered logits (sorted on the host: value desc, id asc); the embedding residual is bitwise the host
embedding on every chip. Timing: one step (blocking median), S steps back to back. A model prefix (the dense layers
only), so the tokens are not the model's tokens. ARGS: layers ([0, 1, 2]), repeat (1), t (1), steps (8), reps (20),
topk (5), tag, record, rev."""
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

import models.autoports.zai_org_glm_5_3_flash.tt.fabric_ar as FAR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_pre as KPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_dense as MKD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8 as TP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for mod in (M5, KREC, KPRE, MKR, MKD, MKL, FAR, TP, TPL, MKS, TPS, MKH):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
M5.set_env({"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3"})
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2])]
REP = int(args.get("repeat", 1))
T = int(args.get("t", 1))
STEPS = int(args.get("steps", 8))
REPS = int(args.get("reps", 20))
TOPK = int(args.get("topk", 5))
tag = args.get("tag", "m5b-tp-step")
TPN = int(mesh.get_num_devices())
cfg = hf_config()


def log(*a):
    print(f"M5B-STEP {C.now()}", *a, flush=True)


key = ("m5b_tp_step", T, tuple(LAYERS), REP, TOPK)
M5.prune_ctx(ctx, [key])
M5.keep_decoders(ctx, [])
hk = ("m5_head_host",)
if hk not in ctx:
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    ctx[hk] = dict(lm=MKH.lm_head_host(source), norm=MKH.norm_host(source))
if "emb" not in ctx[hk]:
    ctx[hk]["emb"] = MKH.embed_host(FdTileCacheSource(CACHE_DIR, MANIFEST))
if key not in ctx:
    t0 = time.time()
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    tps = {L: TPL.TPDenseKDALayer(mesh, cfg, L, source, OptimizedDecoderConfig(), b=T, program="layer")
           for L in LAYERS}  # fmt: skip
    fl = tps[LAYERS[0]]
    head = MKH.MkHead(mesh, ctx[hk]["lm"], ctx[hk]["norm"], t=T, eps=cfg.rms_norm_eps, topk=TOPK,
                      fabric=fl.fabric, n_prod=fl.n_readers)  # fmt: skip
    emb = MKH.MkEmbed(mesh, ctx[hk]["emb"], t=T, place=head.place)
    ctx[key] = dict(tps=tps, fl=fl, head=head, emb=emb)
    log(f"TP layers {LAYERS}, head (fabric gather), embed ready in {time.time() - t0:.1f}s")
st_ = ctx[key]
tps, fl, head, emb = st_["tps"], st_["fl"], st_["head"], st_["emb"]
sh = fl.shards
table = ctx[hk]["emb"]
seq = [tps[L].mk for L in LAYERS] * REP
n = len(seq)
gen = torch.Generator().manual_seed(7)
S0 = torch.randn(T, 64, 128, 128, generator=gen) * 0.1
C0 = (torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16)
sts = [sh.shard_state(S0, C0) for _ in range(n)]
act = fl.masks(torch.ones(T, dtype=torch.bool))
res = fl.replicated(torch.zeros(T, 1, 4, 4096), memory_config=ttnn.L1_MEMORY_CONFIG)
stk = TPS.TPStack(mesh, seq, fl, t=T, rev=args.get("rev"), res=res)
tokens = head.topk_out
first = torch.randint(0, 154880, (T,), generator=gen)
tok_host = torch.zeros(32, 64, dtype=torch.int32)
tok_host[:T, 0] = first.to(torch.int32)


def write_tokens():
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(tok_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)), tokens)  # fmt: skip


def step():
    emb(tokens, res)
    stk([Cm for _, Cm in sts], act, [Sm for Sm, _ in sts])
    return head(res)


write_tokens()
emb(tokens, res)
ttnn.synchronize_device(mesh)
got_x = [ttnn.to_torch(d).float().reshape(T, 4, 4096) for d in ttnn.get_device_tensors(res)]
want_x = table[first].float().reshape(T, 1, 4096).expand(T, 4, 4096)
embed_bitwise = all(torch.equal(g_, want_x) for g_ in got_x)
write_tokens()
trw = C.Traced(ttnn, mesh, step)
write_tokens()
fed, ok_feed, replicas = [], True, True
for s_ in range(STEPS):
    trw.replay()
    logits = head.gather(trw.out)[:, :154880]
    rows = [ttnn.to_torch(d).to(torch.int64).reshape(32, 64)[:T, :TOPK] for d in ttnn.get_device_tensors(tokens)]
    replicas &= all(torch.equal(rows[0], r_) for r_ in rows[1:])
    _, idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    exact = bool(torch.equal(rows[0], idx[:, :TOPK]))
    ok_feed &= exact
    fed.append(dict(step=s_, tok=rows[0][:, 0].tolist(), topk_exact=exact))
log("FEED", fed)
walls = C.stats(trw.replays(REPS))
bm = trw.batch(REPS)
trw.release()
ok = embed_bitwise and ok_feed and replicas
r = dict(test=f"m5b tp step: embed + {n} dense_kda layers (ONE TP stack program) + head with in-program top-{TOPK} "
              f"gather, one trace, token fed back on device, tp{TPN} T={T}", status="pass" if ok else "fail",
         milestone="M5b-piece", T=T, tp=TPN, mgd=CTX["mgd"], n_layers=n, embed_bitwise_all_chips=embed_bitwise,  # noqa: F821
         token_rows_identical_on_chips=replicas, topk_exact_vs_gathered_logits=ok_feed, steps=STEPS, fed=fed[:4],
         tag=tag)  # fmt: skip
prow = dict(test=f"m5b tp step {n} dense_kda layers tp{TPN} T={T}", status="info", milestone="M5b-piece", traced=True,
            T=T, tp=TPN, n_layers=n, us=walls["median_us"], replay=walls, batch_mean_us=round(bm, 1) if bm else None,
            launches_per_step=3, tag=tag,
            note="one trace per step: MkEmbed + ONE TP stack program (2 in-program all-reduces per layer) + MkHead (vocab "
                 "shard + in-program candidate gather); no host work between steps")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, us=walls["median_us"], feed=ok_feed, replicas=replicas)
