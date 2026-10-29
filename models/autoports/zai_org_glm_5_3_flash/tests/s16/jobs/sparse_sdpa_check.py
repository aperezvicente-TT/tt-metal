# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): Stage 16 P3 diagnosis -- ttnn.transformer.sparse_sdpa alone, rows with the DSA prefill's
valid-key counts (expand_rows: 4 x 512 selected-pool tokens + the open pool's (p + 1) % 4 tail = 2,048..2,051 valid ids,
then 0xFFFFFFFF sentinels to SPARSE_W), vs a host fp32 reference (absorbed MLA: K = V = the latent rows).

The 32K L3 check (06:24Z) gave ~1e36 rows for 2,048 / 2,049 / 2,050 valid ids and correct rows for 2,051, deterministic,
traced and eager alike, where the same code passed at 05:04Z (right after a DRAM-wiping reset): an uninitialized read?
Each pass runs the op twice: "clean" (fresh process state) and "dirty" (after ops that leave 1e30 values in L1 CBs and in
freed DRAM), so a dependence on leftover memory shows as a clean / dirty difference.

ARGS: t (32768), rows (256), heads (32), nvs ([2048, 2049, 2050, 2051]: cycled over the rows), k_chunk (256),
check_rows (32), seed (0), tag, record."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
T = int(args.get("t", 32768))
R = int(args.get("rows", 256))
H = int(args.get("heads", 32))
NVS = [int(v) for v in args.get("nvs", [2048, 2049, 2050, 2051])]
KC = int(args.get("k_chunk", PF.SPARSE_KCHUNK))
NCHK = int(args.get("check_rows", 32))
W = PF.SPARSE_W
SCALE = 0.0625
tag = args.get("tag", "p3-sparse-sdpa-check")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
DRAM = ttnn.DRAM_MEMORY_CONFIG
g = torch.Generator().manual_seed(int(args.get("seed", 0)))


def dev(t_, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t_, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=repl)


def dev0(t_):
    return ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()


kv_h = (torch.randn(1, 1, T, 512, generator=g) * 0.3).to(torch.bfloat16)
q_h = (torch.randn(1, H, R, 512, generator=g) * 0.3).to(torch.bfloat16)
ids_h = torch.full((R, W), 0xFFFFFFFF, dtype=torch.int64)
nv_of = []
for r in range(R):
    nv = NVS[r % len(NVS)]
    ids_h[r, :nv] = torch.randperm(T, generator=g)[:nv]
    nv_of.append(nv)
kv = dev(kv_h, ttnn.bfloat16)
q = dev(q_h, ttnn.bfloat16)
ids = dev(ids_h.reshape(1, 1, R, W).to(torch.int32), ttnn.uint32)
ckc = PF._ckc_fp32(mesh)

# host reference on check rows (every nv represented)
rows = sorted({r for r in range(R) if r < NCHK} | {r for r in range(R - 4, R)})
ref = {}
for r in rows:
    nv = nv_of[r]
    k = kv_h[0, 0, ids_h[r, :nv]].float()  # [nv, 512]
    s = (q_h[0, :, r].float() @ k.T) * SCALE  # [H, nv]
    ref[r] = torch.softmax(s, dim=-1) @ k  # [H, 512]


def run_once():
    o = ttnn.transformer.sparse_sdpa(q, kv, ids, 512, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=SCALE,
                                     k_chunk_size=KC, compute_kernel_config=ckc)  # fmt: skip
    ttnn.synchronize_device(mesh)
    y = dev0(o).reshape(1, -1, R, 512)[0, :H]  # [H, R, 512] (rows past H heads: the op's tile padding)
    ttnn.deallocate(o)
    out = {}
    for nv in NVS:
        rs = [r for r in rows if nv_of[r] == nv]
        got = torch.stack([y[:, r] for r in rs])
        want = torch.stack([ref[r] for r in rs])
        out[str(nv)] = dict(pcc=round(C.pcc(got, want), 6), max_abs=float("%.3g" % float(got.abs().max())),
                            nonfinite=int((~torch.isfinite(got)).sum()), rows=len(rs))  # fmt: skip
    return out


def dirty():
    """leave 1e30 values in the L1 CBs of every core (a full-grid eltwise) and in freed DRAM"""
    big = ttnn.from_torch(torch.full((1, 1, 4096, 4096), 1e30), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                          device=mesh, memory_config=DRAM, mesh_mapper=repl)  # fmt: skip
    for _ in range(2):
        y_ = ttnn.multiply(big, 1.5)
        z_ = ttnn.to_layout(y_, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y_)
        ttnn.deallocate(z_)
    ttnn.deallocate(big)
    ttnn.synchronize_device(mesh)


res = {"clean": run_once()}
dirty()
res["dirty"] = run_once()
res["clean_again"] = run_once()
ok = all(v["pcc"] >= 0.99 and v["nonfinite"] == 0 for p in res.values() for v in p.values())
row = dict(test=f"p3 sparse_sdpa alone: {H} heads x {R} rows, T {T}, valid ids {NVS} cycled, k_chunk {KC}, vs host fp32 "
                f"(clean / after 1e30 L1+DRAM residue / clean again) tp{TPN}", milestone="P3", heads=H, rows=R, t=T,
           k_chunk=KC, by_nv=res, status="pass" if ok else "fail", tag=tag)  # fmt: skip
S.log(row)
if rec:
    C.record("pcc", row)
for t_ in (kv, q, ids):
    ttnn.deallocate(t_)
RESULT = [row]
