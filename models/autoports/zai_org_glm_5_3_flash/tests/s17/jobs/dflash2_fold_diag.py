# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair or its mock): is the multi-core drafter fold ('spec64') a rounding difference or a defect?
The fold unit (2026-10-27 02:05Z, d0:card2) found 'spec' (one core per batch row) BITWISE equal to today and 'spec64'
(27 cores per kv head) off by max |diff| 0.044 at 4K / 0.19 at 32K. Here, on the same standalone kind (random weights,
the bfp8 drafter cache filled at every position), layer 0's attention (its input is the same in every form) is computed
by today's call ('off') and by the fold at every (cores per kv head, candidates per call) of ARGS, and each is compared
with an fp32 HOST reference built from the device's own q (the probe) and the device's own cache (read back after the
block wrote its rows): max |err|, mean |err| and the PCC of every form vs the reference, per context.
ARGS: ctxs ("4096,32768"), cores ("1,2,4,8,16,27"), tgs ("4,2"), kchunk (32), cap (33,280), tag."""
import importlib
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
K = 7
T = K + 1
CAP = int(args.get("cap", 33280))
CTXS = [int(v) for v in str(args.get("ctxs", "4096,32768")).split(",")]
CORES = [int(v) for v in str(args.get("cores", "1,2,4,8,16,27")).split(",")]
TGS = [int(v) for v in str(args.get("tgs", "4,2")).split(",")]
KCH = int(args.get("kchunk", 32))
os.environ["GLM53_DF2_WEIGHTS"] = "random"
os.environ["GLM53_DF2_LAYER"] = "ops"
os.environ["GLM53_DF2_KV_DTYPE"] = "bfp8"
os.environ["GLM53_DF2_OPS_FOLD"] = "spec64"
os.environ["GLM53_DF2_FOLD_TG"] = str(TGS[0])
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402

DK = importlib.reload(DK)


def log(*a):
    print(f"S17-DF2FOLD-D {C.now()}", *a, flush=True)


S.M5.prune_ctx(ctx, [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "dflash2_kind"])
ops0 = DK.TtnnOps(mesh, log=log)
tin = ops0.zeros((32, 64), torch.int64, "rm")
rec = ops0.zeros((1, 64), torch.int64, "rm")
kind = DK.build_dflash2_kind(mesh, K=K, cap=CAP, tin=tin, rec=rec, log=log)
ops = kind.ops
nh = kind.S.nh_l
L0 = kind.L[0]
g = torch.Generator().manual_seed(1234)
rep = ttnn.ReplicateTensorToMesh(mesh)
for Ly in kind.L:
    for c_ in (Ly["Kc"], Ly["Vc"]):
        shp = tuple(int(v) for v in c_["t"].shape)
        src = ttnn.from_torch((torch.randn(*shp, generator=g) * 0.5).to(torch.bfloat16), dtype=ttnn.bfloat8_b,
                              layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        ttnn.copy(src, c_["t"])
        ttnn.deallocate(src)
ttnn.synchronize_device(mesh)
# the fold forms of layer 0: every (Tg, cores); each Tg its own views (mask), each cores its own program config
forms = {}
for tg in TGS:
    os.environ["GLM53_DF2_FOLD_TG"] = str(tg)
    base = ops.fold_views(L0["Kc"], L0["Vc"], nh, T)
    for c in CORES:
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ops.grid, q_chunk_size=32, k_chunk_size=KCH,
                                     max_cores_per_head_batch=c)  # fmt: skip
        forms[(tg, c)] = dict(base, cfg=cfg, cores=c, kchunk=KCH)


def chips(t):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


def stats(x, ref):
    d = (x - ref).abs()
    xf, rf = x.flatten().double(), ref.flatten().double()
    pcc = float(torch.corrcoef(torch.stack([xf, rf]))[0, 1])
    return dict(max_err=float(d.max()), mean_err=float(d.mean()), pcc=round(pcc, 7), ref_absmax=float(ref.abs().max()))


rows = []
for Lc in CTXS:
    rh = torch.zeros(1, 64, dtype=torch.int64)
    rh[0, 1] = Lc - 16
    ops.write(rec, rh)
    th = torch.zeros(32, 64, dtype=torch.int64)
    th[0, 0] = 1234
    ops.write(tin, th)
    kind.prep_program()
    kind.ctx_program()
    L0["fold"] = None
    kind.probe = {}
    kind.block_program()
    ttnn.synchronize_device(mesh)
    q = chips(kind.probe["L0.q"])
    att_off = chips(kind.probe["L0.att"])
    kind.probe = None
    bound = Lc - 16 + 1 + K  # c' + K (n = 0)
    Kc, Vc = chips(L0["Kc"]["t"]), chips(L0["Vc"]["t"])  # the device cache after the block's rows were written
    refs = []
    for qi, Ki, Vi in zip(q, Kc, Vc):  # per chip: [1, 32, 32 (nh valid), hd]; cache [nblk, nkv, 32, hd]
        nkv, hd = Ki.shape[1], Ki.shape[3]
        G = nh // nkv
        Kf = Ki.float().permute(1, 0, 2, 3).reshape(nkv, -1, hd)[:, : bound + 1]
        Vf = Vi.float().permute(1, 0, 2, 3).reshape(nkv, -1, hd)[:, : bound + 1]
        out = torch.zeros(T, nh, hd)
        for j in range(T):
            qj = qi[0, j, :nh].float().reshape(nkv, G, hd)
            s = torch.einsum("kgd,ksd->kgs", qj, Kf) * kind.scale
            out[j] = torch.einsum("kgs,ksd->kgd", torch.softmax(s, -1), Vf).reshape(nh, hd)
        refs.append(out)
    ref = torch.stack(refs)
    off = torch.stack([a[0, :T, :nh].float() for a in att_off])
    r = dict(ctx=Lc, form="off", **stats(off, ref))
    rows.append(r)
    log(f"ctx {Lc} off vs fp32 ref: {r}")
    for (tg, c), f in forms.items():
        L0["fold"] = f
        kind.probe = {}
        kind.block_program()
        ttnn.synchronize_device(mesh)
        att = chips(kind.probe["L0.att"])
        kind.probe = None
        x = torch.stack([a[0, :T, :nh].float() for a in att])
        r = dict(ctx=Lc, form=f"fold Tg{tg} cores{c}", bitwise_vs_off=bool(torch.equal(x, off)), **stats(x, ref))
        rows.append(r)
        log(f"ctx {Lc} fold Tg {tg} cores {c} vs fp32 ref: {r}")
    L0["fold"] = None
S.record("tests", dict(test="dflash2-fold-diag" + ("-mock" if C.MOCK else ""), milestone="DF2-CTX", rows=rows,
                       status="pass" if C.MOCK else "info", tag=args.get("tag")))  # fmt: skip
RESULT = dict(ok=True, rows=len(rows))
