# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (mock cluster, --mgd 2x4; also a real mesh): the PMK mHC phases (tt/pmk_mhc.py) as ONE standalone
program per geometry -- pre(attn) -> post(attn) -> pre(ffn) -> post(ffn) on random weights and random rows -- built
and launched through ttnn.generic_op. Under GLM53_MOCK=1 the kernels JIT-compile and the CBs / L1 / semaphores are
allocated and validated (enqueues are no-ops): that is the evidence this job records. On a real mesh it also compares
the outputs with the oracle (PCC). ARGS: sps ([256, 1024]), tag."""

from __future__ import annotations

import json
import os
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as M

mesh = MESH  # noqa: F821
A = ARGS or {}  # noqa: F821
MOCK = os.environ.get("GLM53_MOCK") == "1"
SPS = [int(s) for s in A.get("sps", [256, 1024])]
TAG = A.get("tag", "pmk-mhc-mock")
rows = []


def row(**kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tag=TAG, mock=MOCK, **kw)
    rows.append(r)
    print("ROW", json.dumps(r, default=str), flush=True)
    return r


def put(t, dt):
    n = int(mesh.get_num_devices())
    mapper = ttnn.ReplicateTensorToMesh(mesh) if n > 1 else None
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=mapper)  # fmt: skip


for sp in SPS:
    try:
        plan = M.Plan(sp_rows=sp, y_lane=False)  # standalone: y from DRAM
        subs = [M.SubWeights.random(mesh, seed=s) for s in range(plan.nsub)]
        prog = M.PmkMhc(mesh, subs, plan, hc_eps=1e-6, x_eps=1e-5, h_eps=1e-5, iters=20)
        g = torch.Generator().manual_seed(7)
        x_host = torch.randn(1, 1, sp, M.N_STREAMS * M.HIDDEN, generator=g)
        ys_host = [torch.randn(1, 1, sp, M.HIDDEN, generator=g) for _ in range(plan.nsub)]
        x = put(x_host, ttnn.float32)
        ys = [put(y, ttnn.float32) for y in ys_host]
        t0 = time.perf_counter()
        hns, posts, combs = prog(x, ys)
        ttnn.synchronize_device(mesh)
        build_s = time.perf_counter() - t0
        r = dict(test=f"pmk_mhc standalone program, SP {sp} ({'resident' if plan.resident else 'streamed'}, nrb {plan.nrb})",
                 status="pass" if MOCK else "info", sp=sp, nrb=plan.nrb, resident=plan.resident, build_s=round(build_s, 2),
                 l1=plan.l1_bytes(), estimate_us=plan.estimate_us()["per_sublayer"], grid=plan.grid)  # fmt: skip
        if not MOCK:  # a real mesh: the oracle (chip 0's copy)
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import reference_step as post_ref
            from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import reference_step as pre_ref

            first = lambda t: (ttnn.get_device_tensors(t)[0] if mesh.get_num_devices() > 1 else t)  # noqa: E731
            host = lambda t: ttnn.to_torch(first(t)).float()  # noqa: E731
            xr = x_host[0, 0].double()
            pccs = []
            for s in range(plan.nsub):
                h = subs[s].host
                ref = pre_ref(xr.float(), h["fn_t"], h["scale"], h["bias"], h["w"], hc_eps=1e-6, x_eps=1e-5, h_eps=1e-5,
                              iters=20)  # fmt: skip
                hn = host(hns[s])[0, 0].double()
                pccs.append(float(torch.corrcoef(torch.stack([hn.reshape(-1), ref["hn"].reshape(-1)]))[0, 1]))
                xr = post_ref(ys_host[s][0, 0], xr.reshape(sp, 4, M.HIDDEN).float(), ref["post"], ref["comb"]).reshape(
                    sp, -1
                )
            xo = host(x)[0, 0].double()
            pccs.append(float(torch.corrcoef(torch.stack([xo.reshape(-1), xr.reshape(-1)]))[0, 1]))
            r.update(pcc_hn=pccs[:-1], pcc_residual=pccs[-1], status="pass" if min(pccs) >= 0.9999 else "fail")
        row(**r)
        if MOCK:  # the per-part bodies (P1 / P10 / P16) and the probe variant: JIT-compile + allocate (no execution)
            for part in M.PARTS:
                prog(x, ys, part=part)
            pplan = M.Plan(sp_rows=sp, y_lane=False, probe=True)
            pprog = M.PmkMhc(mesh, subs, pplan, hc_eps=1e-6, x_eps=1e-5, h_eps=1e-5, iters=20)
            outs = pprog(x, ys)
            ttnn.synchronize_device(mesh)
            row(test=f"pmk_mhc parts P1 / P10 / P16 + probe variant compile, SP {sp}", status="pass", sp=sp,
                taps_shape=list(outs[-1].shape))  # fmt: skip
        for t_ in [x] + ys + hns + posts + combs:
            ttnn.deallocate(t_)
    except Exception:
        traceback.print_exc()
        row(test=f"pmk_mhc standalone program, SP {sp}", status="fail", error=traceback.format_exc()[-1200:])

fails = [r for r in rows if r.get("status") == "fail"]
RESULT = dict(status="pass" if not fails else "fail", rows=rows)
print("S16", json.dumps(RESULT, default=str)[:2000], flush=True)
