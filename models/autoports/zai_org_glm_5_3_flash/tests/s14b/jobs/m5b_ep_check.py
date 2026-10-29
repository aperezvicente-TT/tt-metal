# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd 2x4): Stage 14b — EP-SHARDED routed-expert weights (tt/ep_experts.py: chip r holds only its 36
experts, slot i = one mesh tensor at one address) vs the decoder's REPLICATED experts (all 288 per chip), one sparse_kda
layer as Stage 14's TP / EP single-layer program (TPSparseKDALayer.run_layer): the layer output on every chip and the
state / conv history, bitwise, from the same input and state; the DRAM the layer's experts take per chip both ways.
ARGS: layer (4), t (1), tag, record, rev."""
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

import models.autoports.zai_org_glm_5_3_flash.tt.ep_experts as EP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse as TPSP  # noqa: E402

for mod in (M5, EP, TPSP):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
T = int(args.get("t", 1))
REV = args.get("rev")
tag = args.get("tag", "m5b-ep-check")
TPN = mesh.get_num_devices()
cfg = hf_config()
M5.set_env({"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"})


def log(*a):
    print(f"M5B-EP {C.now()}", *a, flush=True)


def dram_per_bank():
    return int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)


M5.prune_ctx(ctx, [])
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("tp", "moe", "m5b_tp", "m5b_moe")]:
    ctx.pop(k)
gc.collect()
source = FdTileCacheSource(CACHE_DIR, MANIFEST)
ocfg = OptimizedDecoderConfig()
d0 = dram_per_bank()
t0 = time.time()
moe_r = TPSP.build_moe(mesh, cfg, source, L, ocfg)
d1 = dram_per_bank()
moe_e = EP.build_moe_ep(mesh, cfg, source, L, ocfg)
d2 = dram_per_bank()
log(f"MoE L{L}: replicated {(d1 - d0) / 2**20:.0f} MiB / bank, EP {(d2 - d1) / 2**20:.0f} MiB / bank "
    f"({time.time() - t0:.1f}s)")  # fmt: skip
lay_r = TPSP.TPSparseKDALayer(mesh, cfg, L, source, ocfg, b=T, moe=moe_r, rev=REV)
lay_e = TPSP.TPSparseKDALayer(mesh, cfg, L, source, ocfg, b=T, moe=moe_e, rev=REV)
sh = lay_r.shards
g = golden(L, 2048)["streams_in"].float()
x = lay_r.replicated(g[0, 300 : 300 + 37 * T : 37].reshape(1, 1, T, 16384))
act = lay_r.masks(torch.ones(T, dtype=torch.bool))
gen = torch.Generator().manual_seed(77)
S0 = torch.randn(T, 64, 128, 128, generator=gen) * 0.1
C0 = (torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16)
res = {}
for name, lay in (("replicated", lay_r), ("ep", lay_e)):
    Sm, Cm = sh.shard_state(S0, C0)
    tr = C.Traced(ttnn, mesh, lambda: lay.run_layer(x, Cm, act, Sm, t=T)["out"])
    tr.replay()
    outs = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(tr.out)]
    st = sh.gather_state(Sm, Cm)
    us = C.stats(tr.replays(int(args.get("reps", 10))))
    tr.release()
    res[name] = dict(outs=outs, st=st, us=us)
o_r, o_e = res["replicated"]["outs"], res["ep"]["outs"]
out_bit = all(torch.equal(a, b) for a, b in zip(o_r, o_e))
st_bit = torch.equal(res["replicated"]["st"][0], res["ep"]["st"][0]) and torch.equal(res["replicated"]["st"][1],
                                                                                      res["ep"]["st"][1])  # fmt: skip
ok = out_bit and st_bit
r = dict(test=f"m5b EP-sharded routed experts (36 / chip) vs replicated (288 / chip): sparse_kda L{L} TP / EP layer, "
              f"tp{TPN} T={T}", status="pass" if ok else "fail", milestone="M5b", layer=L, T=T, tp=TPN,
         out_bitwise_all_chips=out_bit, state_bitwise=st_bit,
         expert_dram_mib_per_bank=dict(replicated=round((d1 - d0) / 2**20, 1), ep=round((d2 - d1) / 2**20, 1)),
         us=dict(replicated=res["replicated"]["us"].get("median_us"), ep=res["ep"]["us"].get("median_us")),
         tag=tag)  # fmt: skip
log("ROW", r)
if args.get("record"):
    C.record("pcc", r)
lay_r = lay_e = moe_r = moe_e = None
gc.collect()
RESULT = dict(ok=ok)
