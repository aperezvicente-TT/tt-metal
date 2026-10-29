# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair): does the SHIPPED fused7a mHC pre (tt/mhc_pre_pf.MhcPrePf, PF_FPU = lever A, in the served
env as GLM53_PF_MHC_PRE=fused7a + GLM53_PF_MHC_FPU=1) run its 20 Sinkhorn iterations on silicon? (PMK mhc unit 11:03Z:
the PMK phase's comb planes equal the host 20-iteration Sinkhorn to 1.9e-7 while its mismatch vs fused7a, 0.7373, equals
the 20-vs-1-iteration delta -> the reference looks like ONE iteration.)
ONE traced replay of MhcPrePf(attn sublayer of layer L) on the golden residual rows (chip c: rows [SP c, SP c + SP)),
comb / post read back, compared with tt/mhc_pre.reference_step on the same rows and weights at iters = cfg's (20) and
iters = 1. Criterion: maxabs(comb_dev - comb_host_20) <= 1e-3 -> pass (the shipped kernel iterates); a match to the
1-iteration host instead is the defect. ARGS: layer (4), sp (256), tag, record (True)."""
import os
import sys

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_mhc_replica as R  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import reference_step  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf import MhcPrePf  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

C = S.C
mesh, args = MESH, ARGS  # noqa: F821
L = int(args.get("layer", 4))
SP = int(args.get("sp", 256))
tag = args.get("tag", f"fused7a-sinkhorn-L{L}-sp{SP}")
NDEV = int(mesh.get_num_devices())
cfg = hf_config()
source = FdTileCacheSource(CACHE_DIR, MANIFEST)
rows = golden(L, 2048)["streams_in"].float().reshape(1, 1, 2048, 16384)[:, :, : SP * NDEV]
mapper = ttnn.ShardTensorToMesh(mesh, dim=2) if NDEV > 1 else None
x = ttnn.from_torch(rows, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)  # fmt: skip
pf = MhcPrePf.from_source(mesh, source, cfg, L, "attn", "input_layernorm.weight", fpu=True)


def fn():
    return pf(x, hn_dtype=ttnn.bfloat16)  # -> (hn, post, comb)


out = fn()  # the compile launch (the one eager launch; its output is not compared)
ttnn.synchronize_device(mesh)
for t_ in out:
    ttnn.deallocate(t_)
tid = None
if not C.MOCK:
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        out = fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
per_chip = lambda t_: [ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731
row = dict(test=f"shipped fused7a mHC pre (MhcPrePf PF_FPU) Sinkhorn iterations on silicon: comb vs host at "
                f"{cfg.hc_sinkhorn_iters} and at 1 iteration, layer {L} attn, SP {SP}, traced", milestone="PMK",
           layer=L, sp=SP, tp=NDEV, iters_cfg=int(cfg.hc_sinkhorn_iters), traced=not C.MOCK, mock=C.MOCK, tag=tag)  # fmt: skip
if not C.MOCK:
    fn_t, scale, bias, w = R.weights("attn", False, source, cfg)
    d20, d1, dpost = [], [], []
    for c, (comb_c, post_c) in enumerate(zip(per_chip(out[2]), per_chip(out[1]))):
        xs = rows[0, 0, c * SP : (c + 1) * SP]
        kw = dict(hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps)
        r20 = reference_step(xs, fn_t, scale, bias, w, iters=int(cfg.hc_sinkhorn_iters), **kw)
        r1 = reference_step(xs, fn_t, scale, bias, w, iters=1, **kw)
        cd = comb_c.reshape(-1, comb_c.shape[-1])[:SP, :16].double()
        pd = post_c.reshape(-1, post_c.shape[-1])[:SP, :4].double()
        # the device's 16-wide row may be C[i][j] row-major or its transpose: take the closer layout per reference
        lay = lambda r_: min(float((cd - r_["comb"].reshape(SP, 16)).abs().max()),  # noqa: E731
                             float((cd - r_["comb"].transpose(-1, -2).reshape(SP, 16)).abs().max()))
        d20.append(lay(r20))
        d1.append(lay(r1))
        dpost.append(float((pd - r20["post"]).abs().max()))
        row.setdefault("comb_rows_sum_dev", []).append(float((cd.reshape(SP, 4, 4).sum(-1) - 1).abs().max()))
    row.update(maxabs_comb_vs_host20=d20, maxabs_comb_vs_host1=d1, maxabs_post_vs_host=dpost,
               host20_vs_host1=float((r20["comb"] - r1["comb"]).abs().max()))  # fmt: skip
    row["verdict"] = ("iterates (20)" if max(d20) <= 1e-3 else
                      "ONE ITERATION" if max(d1) <= 1e-3 else "neither (other defect)")  # fmt: skip
    row["status"] = "pass" if max(d20) <= 1e-3 else "fail"
    ttnn.release_trace(mesh, tid)
else:
    row["status"] = "mock"
S.log(row)
if bool(args.get("record", True)) and not C.MOCK:
    C.record("pcc", row)
RESULT = row
