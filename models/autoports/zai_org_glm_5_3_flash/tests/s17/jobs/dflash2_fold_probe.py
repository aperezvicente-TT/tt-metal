# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK or a device): the rung-2b stock-op folds of the drafter's attention chain, probed one by one on
this mesh (the mock validates shapes / memory configs / program factories, a device the numerics):
  hnorm_sharded   ttnn.rms_norm on the 32-core height-sharded head tensor [1, 32, 32, hd] with a sharded program
                  config (block_h 1, block_w hd / 32): the per-head norm WITHOUT the two reshards (3 -> 1 launch;
                  16 head norms per step: -32); the weight as [1, 1, 1, hd] tile and as the row-major [1, 1, hd/32, 32]
  sdpa_sharded    the paged SDPA decode writing its output height-sharded on the batch cores (the concat_heads
                  reshard in: -8)
  ds_sharded_in   a DS reader-phase matmul (tt/dflash2_mm.py DsMM) reading nlp_concat_heads_decode's sharded output
                  in place through the TensorAccessor (the concat_heads reshard out: -8)
Every probe records ok / the exception text; on a device also the PCC vs the stock chain. ARGS: tag."""
import os
import sys
import traceback

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_mm as DM  # noqa: E402

DK, DM = importlib.reload(DK), importlib.reload(DM)
TILE = 32


def log(*a):
    print(f"S17-DF2FOLD {C.now()}", *a, flush=True)


ops = DK.TtnnOps(mesh, log=log)
tp = ops.tp
nh_l, nkv_l, hd, H = 32 // tp, 8 // tp, 128, 4096
res = {}


def probe(name, fn):
    try:
        out = fn()
        res[name] = dict(ok=True, **(out or {}))
    except Exception as e:  # noqa: BLE001
        res[name] = dict(ok=False, error=str(e).splitlines()[0][:300], trace=traceback.format_exc()[-1500:])
    log(name, res[name].get("ok"), res[name].get("error", ""))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


# the chain's tensors: q heads [1, 32, nh_l (32 padded), hd] height-sharded on the 32 batch cores (create_heads' layout)
torch.manual_seed(0)
qkv_h = torch.randn(1, 1, TILE, (nh_l + 2 * nkv_l) * hd) * 0.5
qkv = ops.rep(qkv_h)
q, k, v = ops.create_heads(qkv, nh_l, nkv_l, hd)
w_h = (1.0 + 0.1 * torch.randn(hd)).to(torch.bfloat16)
w_tile = ops.rep(w_h.reshape(1, 1, 1, hd))
w_rm = ops.rep(w_h.reshape(1, 1, hd // TILE, TILE), torch.bfloat16, "rm")
eps = 1e-5


def hnorm_sharded(w):
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(8, 4), subblock_w=hd // TILE,
                                                     block_h=1, block_w=hd // TILE, inplace=False)  # fmt: skip
    y = ttnn.rms_norm(q, epsilon=eps, weight=w, program_config=pc, memory_config=q.memory_config(),
                      compute_kernel_config=ops.ckc)  # fmt: skip
    ref = ops.head_norm(q, w_tile, eps)
    out = dict(shape=[int(s) for s in y.shape], sharded=bool(y.is_sharded()))
    if not C.MOCK:
        out["pcc"] = pcc(ops.host(y).float(), ops.host(ref).float())
    return out


probe("hnorm_sharded_tilew", lambda: hnorm_sharded(w_tile))
probe("hnorm_sharded_rmw", lambda: hnorm_sharded(w_rm))

# sdpa with a sharded output
nblk = 8
Kc = dict(
    t=ops.zeros((nblk, nkv_l, TILE, hd)),
    pt=ops.shards([torch.arange(nblk, dtype=torch.int32).reshape(1, -1).repeat(TILE, 1)] * tp, torch.int32, "rm"),
)
Vc = dict(t=ops.zeros((nblk, nkv_l, TILE, hd)), pt=Kc["pt"])
cur = ops.shards([torch.full((TILE,), 40, dtype=torch.int32)] * tp, torch.int32, "rm")


def sdpa_sharded():
    a = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q, Kc["t"], Vc["t"], page_table_tensor=Kc["pt"], cur_pos_tensor=cur, scale=1.0 / hd**0.5,
        compute_kernel_config=ops.hifi2, memory_config=ops._hs(hd))  # fmt: skip
    c = ttnn.experimental.nlp_concat_heads_decode(a, num_heads=nh_l)
    out = dict(shape=[int(s) for s in a.shape], sharded=bool(a.is_sharded()), concat_sharded=bool(c.is_sharded()),
               concat_mem=str(c.memory_config())[:200])  # fmt: skip
    if not C.MOCK:
        ref = ops.sdpa(q, Kc, Vc, cur, 1.0 / hd**0.5)
        out["pcc"] = pcc(ops.host(a).float(), ops.host(ref).float())
    return out


probe("sdpa_sharded", sdpa_sharded)


# a DS matmul reading the concat_heads output in place (sharded input through the TensorAccessor)
def ds_sharded_in():
    a = ops.sdpa(q, Kc, Vc, cur, 1.0 / hd**0.5)
    s = ttnn.to_memory_config(a, ops._hs(hd))
    c = ttnn.experimental.nlp_concat_heads_decode(s, num_heads=nh_l)  # [1, 1, 32, nh_l hd] sharded
    w = torch.randn(nh_l * hd, H) * 0.02
    u = DM.DsMM(ops, [w] * tp, out=torch.float32, name="probe_o")
    y = u(c)
    out = dict(in_mem=str(c.memory_config())[:200], shape=[int(v) for v in y.shape])
    if not C.MOCK:
        ref = u(ttnn.to_memory_config(c, ops.DRAM))
        out["pcc"] = pcc(ops.host(y).float(), ops.host(ref).float())
    return out


probe("ds_sharded_in", ds_sharded_in)
ttnn.synchronize_device(mesh)
row = dict(test="dflash2-fold-probe", milestone="DF2", status="pass" if all(r["ok"] for r in res.values()) else "info",
           tp=tp, mock=C.MOCK, probes=res, tag=args.get("tag"))  # fmt: skip
S.record("tests", row)
RESULT = dict(probes={k: v.get("ok") for k, v in res.items()})
