# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b — a whole single-chip decode step, token in -> token out, as ONE
trace with no host work inside: MkEmbed (the device-resident tokens -> the stacked residual) -> one stack program per
layer-kind run (tt/mk_stack.py; real dense_kda L0-L2 + sparse_kda layers, each its own weights / state) -> MkHead
(final norm + lm_head + device top-K -> the SAME token tensor the next step's MkEmbed reads: token feedback on device).

Checks (device vs device): (1) MkEmbed's residual == the host embedding of the tokens in 4 streams (bitwise); (2) the
step's layers == the same layers as single-layer launches on that residual (bitwise residual + every state, stack vs
chain); (3) every step of an S-step replay run: the fed-back token == the argmax of that step's own logits (exact,
lower id on ties); timing: one step replay (blocking median) and S steps back to back. The sequence is a prefix of the
model (no sparse_dsa layer yet: 7b), so the tokens are not the model's tokens.
ARGS: dense ([0, 1, 2]), sparse ([4, 5, 6]), t (1), steps (8), reps (20), topk (1), tag, record, rev, soak_s (0)."""
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

import models.autoports.zai_org_glm_5_3_flash.tt.kda_pre as KPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_dense as MKD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_experts as MKX  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse as MKSP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402

for mod in (M5, KREC, KPRE, MKR, MKD, MKX, MKL, MKSP, MKS, MKH):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [4, 5, 6])]
T = int(args.get("t", 1))
STEPS = int(args.get("steps", 8))
REPS = int(args.get("reps", 20))
TOPK = int(args.get("topk", 1))
tag = args.get("tag", "m5-step")
ENV_D = {**M5.DENSE_KNOBS, **{k: str(v) for k, v in (args.get("env") or {}).items()}}
ENV_S = {**M5.SPARSE_KNOBS, **{k: str(v) for k, v in (args.get("sparse_env") or {}).items()}}
cfg = hf_config()


def log(*a):
    print(f"M5-STEP {C.now()}", *a, flush=True)


key = ("m5_step", T, tuple(DENSE), tuple(SPARSE), tuple(sorted(ENV_D.items())), tuple(sorted(ENV_S.items())), TOPK)
M5.prune_ctx(ctx, [key])
gc.collect()
log("dropped cached decoders", M5.keep_decoders(ctx, DENSE + SPARSE))
if key not in ctx:
    t0 = time.time()
    seq = M5.build_layers(mesh, cfg, ctx, DENSE, SPARSE, ENV_D, ENV_S)
    hk = ("m5_head_host",)
    if hk not in ctx:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        ctx[hk] = dict(lm=MKH.lm_head_host(source), norm=MKH.norm_host(source))
    if "emb" not in ctx[hk]:
        ctx[hk]["emb"] = MKH.embed_host(FdTileCacheSource(CACHE_DIR, MANIFEST))
    head = MKH.MkHead(mesh, ctx[hk]["lm"], ctx[hk]["norm"], t=T, eps=cfg.rms_norm_eps, topk=TOPK)
    emb = MKH.MkEmbed(mesh, ctx[hk]["emb"], t=T, place=head.place)
    ctx[key] = dict(seq=seq, head=head, emb=emb)
    log(f"layers {DENSE} + {SPARSE}, head, embed ready in {time.time() - t0:.1f}s; L1", M5.l1_state(mesh))
seq, head, emb = ctx[key]["seq"], ctx[key]["head"], ctx[key]["emb"]
table = ctx[("m5_head_host",)]["emb"]
kda0 = seq[0]["kda"]
act = masks(mesh, torch.tensor([(i % 3) != 2 for i in range(T)]) if T > 1 else torch.tensor([True]), kda0.lower_bound)
sts = [kda0.allocate_state(T) for _ in seq]
gen = torch.Generator().manual_seed(99)
first_tokens = torch.randint(0, 154880, (T,), generator=gen)
tok_host = torch.zeros(32, 64, dtype=torch.int32)
tok_host[:T, 0] = first_tokens.to(torch.int32)
tokens = head.topk_out  # the step's input AND output (MkHead writes row t word 0 = the greedy token)
res = ttnn.empty((T, 1, 4, 4096), ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG)
skey = ("m5_step_stacks", key, args.get("rev"))
if skey not in ctx:
    stacks, pool = [], None
    for r in M5.kind_runs(seq):
        M5.set_env(ENV_S if seq[r[0]]["sparse"] else ENV_D)
        stk = MKS.MkStack(mesh, [seq[i]["mk"] for i in r], t=T, rev=args.get("rev"), res=res, pool=pool)
        pool = stk.pool
        stacks.append((r, stk))
    ctx[skey] = stacks
stacks = ctx[skey]


def write_tokens():
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(tok_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), tokens)


def layers(restore_states=None):
    for r, stk in stacks:
        M5.set_env(ENV_S if seq[r[0]]["sparse"] else ENV_D)
        stk([sts[i].C for i in r], act, [sts[i].S for i in r])


def step():
    emb(tokens, res)
    layers()
    return head(res)


# (1) embed vs host, bitwise
write_tokens()
emb(tokens, res)  # compile launch
ttnn.synchronize_device(mesh)
write_tokens()
tr = C.Traced(ttnn, mesh, lambda: emb(tokens, res), warm=False)
tr.replay()
got_x = ttnn.to_torch(res).float().reshape(T, 4, 4096)
want_x = table[first_tokens].float().reshape(T, 1, 4096).expand(T, 4, 4096)
embed_bitwise = bool(torch.equal(got_x, want_x))
tr.release()
# (2) the layers: stacks vs single-layer launches on the same residual and states (restored)
x_flat = ttnn.from_torch(want_x.reshape(1, 1, T, 16384).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                         device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
x_stk = ttnn.from_torch(want_x.reshape(T, 1, 4, 4096).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
s0 = ttnn.from_torch(torch.randn(T, 64, 128, 128, generator=gen) * 0.1, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                     device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
h0 = ttnn.from_torch((torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def restore():
    for st in sts:
        ttnn.copy(h0, st.C)
        ttnn.copy(s0, st.S)


def chain():
    restore()
    y = x_flat
    for e, st in zip(seq, sts):
        M5.set_env(ENV_S if e["sparse"] else ENV_D)
        y = e["mk"](y, st.C, act, t=T, state=st.S)["out"]
    return y


trc = C.Traced(ttnn, mesh, chain)
trc.replay()
want = ttnn.to_torch(trc.out).float().reshape(-1, 16384)[:T]
want_s = [ttnn.to_torch(st.S).float() for st in sts]
trc.release()
trc = None
gc.collect()


def stacked_layers():
    restore()
    ttnn.copy(x_stk, res)
    layers()
    return res


trs = C.Traced(ttnn, mesh, stacked_layers)
trs.replay()
got = ttnn.to_torch(res).float().reshape(T, 16384)
got_s = [ttnn.to_torch(st.S).float() for st in sts]
trs.release()
layers_bitwise = bool(torch.equal(got, want)) and all(torch.equal(a, b) for a, b in zip(got_s, want_s))
# (3) the whole step, token in -> token out, S steps of one trace (token fed back on device)
write_tokens()
trw = C.Traced(ttnn, mesh, step)  # compile launch (the head's first call), then the capture
write_tokens()
fed, ok_feed = [], True
for s_ in range(STEPS):
    tok_in = ttnn.to_torch(tokens).to(torch.int64).reshape(32, 64)[:T, 0].clone()
    trw.replay()
    logits = head.gather(trw.out)
    tok_out = ttnn.to_torch(tokens).to(torch.int64).reshape(32, 64)[:T, 0]
    vals, idx = torch.sort(logits, dim=-1, descending=True, stable=True)
    exact = bool(torch.equal(tok_out, idx[:, 0]))
    ok_feed &= exact
    fed.append(dict(step=s_, tok_in=tok_in.tolist(), tok_out=tok_out.tolist(), argmax_exact=exact))
log("FEED", fed)
walls = C.stats(trw.replays(REPS))
bm = trw.batch(REPS)
soak = None
if float(args.get("soak_s", 0)) > 0:
    n_rep, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < float(args["soak_s"]):
        for _ in range(20):
            ttnn.execute_trace(mesh, trw.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        n_rep += 20
    soak = dict(seconds=round(time.perf_counter() - t0, 1), steps=n_rep,
                us_per_step=round((time.perf_counter() - t0) * 1e6 / n_rep, 1))  # fmt: skip
trw.release()
n = len(seq)
names = [f"L{e['L']}{'s' if e['sparse'] else 'd'}" for e in seq]
ok = embed_bitwise and layers_bitwise and ok_feed
r = dict(test=f"m5 step: embed + {names} as {len(stacks)} stack programs + head (top-{TOPK}) in ONE trace, token fed "
              f"back on device T={T}", status="pass" if ok else "fail", milestone="M5b-piece", T=T, sequence=names,
         embed_bitwise=embed_bitwise, layers_bitwise_vs_single_launches=layers_bitwise, steps=STEPS,
         token_feedback_argmax_exact=ok_feed, fed=fed[:4], tag=tag)  # fmt: skip
prow = dict(test=f"m5 step {names} T={T}", status="info", milestone="M5b-piece", traced=True, T=T, n_layers=n,
            us=walls["median_us"], replay=walls, batch_mean_us=round(bm, 1) if bm else None, soak=soak,
            launches_per_step=2 + len(stacks), tag=tag,
            note="one trace per step: MkEmbed + one stack program per kind run + MkHead (final norm, lm_head 154,880 "
                 "columns on one chip, device top-K); no host work between steps (token fed back on device)")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, us=walls["median_us"], embed=embed_bitwise, layers=layers_bitwise, feed=ok_feed)
