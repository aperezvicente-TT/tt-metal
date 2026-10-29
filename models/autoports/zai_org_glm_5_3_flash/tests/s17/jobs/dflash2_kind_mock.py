# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK or a pair): the in-trace DFlash2 kind STANDALONE (tt/dflash2_kind.py on TtnnOps, the real
config with random weights by default: GLM53_DF2_WEIGHTS) -- the op chain builds on this mesh, the launch census per
step part (PREP / ctx / block), the program-cache keys (a second call adds none: the capture-key gate), the DRAM bytes
of the resident weights, the per-step weight bytes streamed; MOCK: no numerics. ARGS: k (7), cap (4096), weights
("random" | "safetensors"), tap_mode ("fold"), conv ("fused" | "ops": GLM53_DF2_CONV, the rung-2 conv program / the
stock chain), mm ("stream" | "ds": GLM53_DF2_MM, rung 3 M1), mm_cores / conv_cores (GLM53_DF2_MM_CORES /
GLM53_DF2_CONV_CORES: the core caps), tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
K = int(args.get("k", 7))
os.environ["GLM53_DF2_WEIGHTS"] = str(args.get("weights", "random"))
os.environ["GLM53_DF2_TAP_MODE"] = str(args.get("tap_mode", "fold"))
os.environ["GLM53_DF2_CONV"] = str(args.get("conv", "fused"))
os.environ["GLM53_DF2_MM"] = str(args.get("mm", "stream"))
for k_, e_ in (("mm_cores", "GLM53_DF2_MM_CORES"), ("conv_cores", "GLM53_DF2_CONV_CORES")):
    if args.get(k_) is not None:
        os.environ[e_] = str(args[k_])
    else:
        os.environ.pop(e_, None)
os.environ["GLM53_DF2_CONV_RES"] = str(args.get("conv_res", "0"))
os.environ["GLM53_DF2_HEADS_SHARDED"] = str(args.get("heads_sharded", "0"))
os.environ["GLM53_DF2_LAYER"] = str(args.get("layer", "ops"))
os.environ["GLM53_DF2_TAPS"] = str(args.get("taps", "0"))
os.environ["GLM53_DF2_TOPK"] = str(args.get("topk", "ttnn"))
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402

DK = importlib.reload(DK)


def log(*a):
    print(f"S17-DF2K {C.now()}", *a, flush=True)


S.M5.prune_ctx(ctx, [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "dflash2_kind"])
t0 = time.time()
ops = DK.TtnnOps(mesh, log=log)
tin = ops.zeros((32, 64), torch.int64, "rm")
rec = ops.zeros((1, 64), torch.int64, "rm")
mv0 = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
kind = DK.build_dflash2_kind(mesh, K=K, cap=int(args.get("cap", 4096)), tin=tin, rec=rec, log=log)
mv1 = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
ops = kind.ops  # (the kind's own TtnnOps: its launch counter)
dram_mb = (mv1.total_bytes_allocated_per_bank - mv0.total_bytes_allocated_per_bank) * 8 / 2**20
log(f"kind built in {time.time() - t0:.1f}s: DRAM +{dram_mb:.1f} MiB / chip; host layout {kind.S.bytes_per_chip()}")
# the pages a step reads: BOOT state (n = 0, c = 40, bonus 1234)
rh = torch.zeros(1, 64, dtype=torch.int64)
rh[0, 1] = 40
ops.write(rec, rh)
th = torch.zeros(32, 64, dtype=torch.int64)
th[0, 0] = 1234
ops.write(tin, th)
census = {}
for name, fn in (("prep", kind.prep_program), ("ctx", kind.ctx_program), ("block", kind.block_program)):
    ops.launches = 0
    ops.census.clear()
    t1 = time.time()
    fn()
    ttnn.synchronize_device(mesh)
    census[name] = dict(launches=ops.launches, s=round(time.time() - t1, 1), by_op=dict(ops.census))
    log(name, census[name])
n_keys = mesh.num_program_cache_entries()
ops.launches = 0
kind.step()
ttnn.synchronize_device(mesh)
new_keys = mesh.num_program_cache_entries() - n_keys
log("capture keys: warm", n_keys, "new on a further step", new_keys, "step launches", ops.launches)
row = dict(test="dflash2-kind-mock" if C.MOCK else "dflash2-kind-standalone", milestone="DF2I", status="pass" if new_keys == 0 else "fail",
           K=K, tp=int(mesh.get_num_devices()), weights=os.environ["GLM53_DF2_WEIGHTS"], tap_mode=os.environ["GLM53_DF2_TAP_MODE"],
           conv=os.environ["GLM53_DF2_CONV"], mm=os.environ["GLM53_DF2_MM"], mm_cores=args.get("mm_cores"),
           conv_cores=args.get("conv_cores"), conv_res=os.environ["GLM53_DF2_CONV_RES"], heads_sharded=os.environ["GLM53_DF2_HEADS_SHARDED"], mm_plans=ops.mm_plans,
           layer=os.environ["GLM53_DF2_LAYER"], mk_l1=(kind.mk.l1 if kind.mk is not None else None),
           mk_place=(dict(readers=len(kind.mk.assign), owners=len(kind.mk.owners), lanes=kind.mk.lanes) if kind.mk is not None else None),
           census=census, step_launches=ops.launches, program_cache_keys=n_keys, new_keys=new_keys,
           dram_mib_per_chip=round(dram_mb, 1), resident_bytes=kind.S.bytes_per_chip(), read_bytes_per_step=kind.S.read_bytes_per_step(),
           tag=args.get("tag"))  # fmt: skip
S.record("tests", row)
RESULT = dict(ok=new_keys == 0, census=census, dram_mib=dram_mb, step_launches=ops.launches)
