# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 MOCK or any mesh): the PMK MoE phase set (tt/pmk_moe.py) built and launched ONCE on synthetic
XSL expert slices of a chosen width (TP8: 256, TP2: 1,024) with random routing -- in the mock the kernels JIT-build
(the phase composition compiles), the CBs allocate (cb_guard + the L1 ledger) and the program validates; on a device
the launch is the compile launch and the output is compared with EpMoe (xffn forced) on the same inputs (PCC).
ARGS: experts (288), width (256), c (2048), k (8), seed (0), blocks ("" = the budget's choice, else "mg,kc,nc"),
fid ("" = HiFi2), tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.pmk_moe as PM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import _nd_dram, gate_up_tile_perm  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
E, W, CR, K = (
    int(args.get("experts", 288)),
    int(args.get("width", 256)),
    int(args.get("c", 2048)),
    int(args.get("k", 8)),
)
H, LIMIT = 4096, 10.0
tag = args.get("tag", "pmk-moe-mock")
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
g = torch.Generator().manual_seed(int(args.get("seed", 0)))
os.environ["GLM53_PF_MOE_FID"] = str(args.get("fid", ""))
os.environ["GLM53_PF_MOE_FFN"] = "xffn"
os.environ["GLM53_PF_MOE_PLAN"] = "mc2"

perm = gate_up_tile_perm(2 * W // 32)
gus, dns = [], []
t0 = time.time()
for e in range(E):
    gate = torch.randn(H, W, generator=g) * H**-0.5
    up = torch.randn(H, W, generator=g) * H**-0.5
    gu = torch.cat([gate, up], dim=1).reshape(H, 2 * W // 32, 32)[:, perm, :].reshape(H, 2 * W)
    dn = torch.randn(W, H, generator=g) * W**-0.5
    gus.append(
        ttnn.from_torch(
            gu,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=_nd_dram(H, 64),
            mesh_mapper=repl,
        )
    )
    dns.append(
        ttnn.from_torch(
            dn,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=_nd_dram(W, 64),
            mesh_mapper=repl,
        )
    )
S.log(f"weights on device in {time.time() - t0:.1f} s")

x_h = torch.randn(1, 1, CR, H, generator=g) * 0.5
idx_h = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(CR)]).reshape(1, 1, CR, K)
sel_h = torch.rand(1, 1, CR, K, generator=g) + 0.05
dev = lambda t_, dt: ttnn.from_torch(
    t_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=PF.DRAM, mesh_mapper=repl
)  # noqa: E731
x, idx, sel = dev(x_h, ttnn.bfloat16), dev(idx_h.to(torch.int32), ttnn.uint16), dev(sel_h, ttnn.float32)

blocks = tuple(int(v) for v in str(args.get("blocks", "")).split(",")) if args.get("blocks") else None
pm = PM.PmkMoe(mesh, num_experts=E, top_k=K, hidden=H, limit=LIMIT, gate_up=gus, down=dns, blocks=blocks,
               fid=str(args.get("fid", "")))  # fmt: skip
geo = pm.geometry(CR)
mg, kc, nc = pm.blocks(CR)
plan = PM.cb_plan(geo, mg, kc, nc)
model = PM.streaming_model(geo, mg, pm.fid)
S.log(
    f"grid P={pm.p} W_T={pm.w_t} blocks={(mg, kc, nc)} CB total={plan['total']} B groups="
    f"{[(m, t) for m, t, _ in plan['groups']]}"
)
S.log(f"streaming model: { {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in model.items()} }")

x_ext, ids_rm, wt = PM.inputs_from_op_path(x, idx, sel, 1.0)
t0 = time.time()
y = pm(x_ext, ids_rm, wt, CR)  # the compile launch (mock: JIT build + validation only)
ttnn.synchronize_device(mesh)
S.log(f"pmk_moe program launched in {time.time() - t0:.1f} s (calls {pm.calls})")
row = dict(test=f"pmk_moe mock build W={W} C={CR} E={E} tp{TPN} P={pm.p}", milestone="PMK", status="pass" if C.MOCK else "info",
           blocks=[mg, kc, nc], cb_total=plan["total"], fid=pm.fid, mock=C.MOCK, tag=tag,
           model={k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in model.items()})  # fmt: skip
if not C.MOCK:  # a device: the op path (EpMoe, xffn forced) on the same inputs
    import types

    class _Tpp:
        def __init__(self):
            self.mesh, self.ndev, self.repl = mesh, TPN, repl

    moe = types.SimpleNamespace(num_experts=E, top_k=K, hidden=H, intermediate=W, limit=LIMIT, scaling=1.0,
                                gate_up=[None] * E, down=[None] * E)  # fmt: skip
    ep = PF.EpMoe(_Tpp(), moe, xsl=(TPN, gus, dns))
    y_ref = ep(x, idx, sel)
    ttnn.synchronize_device(mesh)
    a = ttnn.to_torch(ttnn.get_device_tensors(y)[0] if TPN > 1 else y).float()
    b = ttnn.to_torch(ttnn.get_device_tensors(y_ref)[0] if TPN > 1 else y_ref).float()
    row.update(pcc=C.pcc(a, b), bitwise=bool(torch.equal(a, b)), maxabs=float((a - b).abs().max()))
    row["status"] = "pass" if row["bitwise"] or row["pcc"] >= 0.9999 else "fail"
S.log(row)
if args.get("record", True):
    C.record("tests", row)
RESULT = row
