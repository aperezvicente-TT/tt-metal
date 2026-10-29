# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK or a device): the DFlash2 drafter's 1M form STANDALONE (tt/dflash2_kind.py + tt/dflash2_1m.py;
dflash2_intrace.md §10): the kind at a long capacity (131K / 1M-synthetic) with the cache dtype and the sequence-shard
knobs -- the op chain builds, the kernels JIT (df2_cache / df2_mk_dm1 bfp8 path, df2_ctl DF2_SHARD), the launch
census per part, the DRAM bytes per chip (the caches are allocated on the device: no host staging), the program-cache
keys (0 new on a further run), the bfp8 host mirror vs ttnn's packer, the LMCache unit export / import API path (the
page view's read / write on the mesh). MOCK: no numerics. ARGS: cap (131072), kv_dtype ("bf16" | "bfp8"), shard
(0 | 1), layer ("ops" | "mk"), parts ("prep,ctx,block"; the shard's block attention is not built on device: the
job records the refusal), k (7), weights ("random"), units (2: the export / import round trip), reps (device: the
traced boot replays of the timing row; 0 on the mock), tag."""
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
K = int(args.get("k", 7))
cap = int(args.get("cap", 131072))
os.environ["GLM53_DF2_WEIGHTS"] = str(args.get("weights", "random"))
os.environ["GLM53_DF2_TAP_MODE"] = str(args.get("tap_mode", "mean"))
os.environ["GLM53_DF2_CONV"] = str(args.get("conv", "fused"))
os.environ["GLM53_DF2_LAYER"] = str(args.get("layer", "ops"))
os.environ["GLM53_DF2_MM"] = str(args.get("mm", "stream"))
os.environ["GLM53_DF2_KV_DTYPE"] = str(args.get("kv_dtype", "bf16"))
os.environ["GLM53_DF2_TOPK"] = str(args.get("topk", "ttnn"))  # rung 3: "kernel" = the two-launch exact top-16 of rows 1..K
os.environ["GLM53_DF2_FOLD"] = str(args.get("fold", "spec"))  # rung 3: the SDPA row fold ("off" = one user per q head)
os.environ["GLM53_DFLASH2_SHARD"] = str(args.get("shard", 0))
parts = str(args.get("parts", "prep,ctx,block")).split(",")
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_1m as M1  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_mk as DMK  # noqa: E402

M1, DMK, DK = importlib.reload(M1), importlib.reload(DMK), importlib.reload(DK)


def log(*a):
    print(f"S17-DF2-1M {C.now()}", *a, flush=True)


S.M5.prune_ctx(ctx, [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "dflash2_kind"])
row = dict(test="dflash2-1m-mock" if C.MOCK else "dflash2-1m-standalone", milestone="DF2-1M", K=K, cap=cap,
           topk=os.environ["GLM53_DF2_TOPK"], fold=os.environ["GLM53_DF2_FOLD"],
           tp=int(mesh.get_num_devices()), kv_dtype=os.environ["GLM53_DF2_KV_DTYPE"], shard=os.environ["GLM53_DFLASH2_SHARD"],
           layer=os.environ["GLM53_DF2_LAYER"], parts=parts, tag=args.get("tag"), status="fail")  # fmt: skip
# the bfp8 host mirror vs ttnn's packer (host tensors; no device data)
torch.manual_seed(0)
x = (torch.randn(64, 128) * 3).to(torch.bfloat16)
x[3, :16] = 0
y = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
row["bfp8_mirror_bitwise"] = bool(torch.equal(y, M1.bfp8_like(x)))
t0 = time.time()
tin = ttnn.from_torch(torch.zeros(32, 64, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
rec = ttnn.from_torch(torch.zeros(1, 64, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
mv0 = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
kind = DK.build_dflash2_kind(mesh, K=K, cap=cap, tin=tin, rec=rec, log=log)
mv1 = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
ops = kind.ops
dram_mb = (mv1.total_bytes_allocated_per_bank - mv0.total_bytes_allocated_per_bank) * 8 / 2**20
cfg = kind.S.d.__dict__
sz = M1.size_drafter(dict(hidden_size=kind.d.H, intermediate_size=kind.d.I, num_hidden_layers=kind.d.L,
                          num_attention_heads=kind.d.nh, num_key_value_heads=kind.d.nkv, head_dim=kind.d.hd,
                          rms_norm_eps=kind.d.eps, rope_parameters=dict(rope_theta=kind.d.theta), vocab_size=kind.d.V,
                          target_layer_ids=list(range(kind.d.n_taps)), mask_token_id=kind.d.mask_id,
                          dflash_config=dict(block_size=kind.d.K + 1, mask_token_id=kind.d.mask_id, conv_kernel_size=2,
                                             conv_group_size=kind.d.group, selector_rank=kind.d.rank, selector_top_k=kind.d.top_k)),
                     tp=ops.tp, ctx=cap, kv_dtype=os.environ["GLM53_DF2_KV_DTYPE"])  # fmt: skip
row.update(build_s=round(time.time() - t0, 1), dram_mib_per_chip=round(dram_mb, 1), nblk=kind.nblk, cap_cache=kind.cap_cache,
           resident_weights=kind.S.bytes_per_chip(), expected_cache_mib_per_chip=round(sz["forms"]["head"] / 2**20, 1),
           expected_rope_mib=round(sz["rope_tables_bytes"] / 2**20, 1), expected_pt_mib=round(sz["page_table_bytes"] / 2**20, 2),
           expected_attention_ms=[round(v, 2) for v in sz["attention_ms_per_step"]])  # fmt: skip
log(f"kind built in {row['build_s']}s: DRAM +{dram_mb:.1f} MiB / chip (expected cache {row['expected_cache_mib_per_chip']} + rope "
    f"{row['expected_rope_mib']} + weights {kind.S.bytes_per_chip()['total'] / 2**20:.0f} MiB); nblk {kind.nblk}")  # fmt: skip
# BOOT state far into the context (n = 0, c = cap - 64, bonus 1234): the positions / pages at the long end
rh = torch.zeros(1, 64, dtype=torch.int32)
rh[0, 1] = cap - 64
ops.write(rec, rh)
th = torch.zeros(32, 64, dtype=torch.int32)
th[0, 0] = 1234
ops.write(tin, th)
census, refused = {}, None
fns = dict(prep=kind.prep_program, ctx=kind.ctx_program, block=kind.block_program)
for name in parts:
    ops.launches = 0
    ops.census.clear()
    t1 = time.time()
    try:
        fns[name]()
        ttnn.synchronize_device(mesh)
    except NotImplementedError as e:
        refused = f"{name}: {e}"
        log("refused", refused)
        break
    census[name] = dict(launches=ops.launches, s=round(time.time() - t1, 1), by_op=dict(ops.census))
    log(name, census[name])
n_keys = mesh.num_program_cache_entries()
ops.launches = 0
for name in census:
    fns[name]()
ttnn.synchronize_device(mesh)
new_keys = mesh.num_program_cache_entries() - n_keys
log("capture keys: warm", n_keys, "new on a further run", new_keys, "launches", ops.launches)
# the LMCache unit export / import API on the mesh (the mock holds no data: the path, the shapes, the fill op)
n_units = int(args.get("units", 2))
t2 = time.time()
blob = M1.export_units(kind, 0, n_units)
M1.import_units(kind, blob, n_units)  # units 0..n-1 -> units n..2n-1
views = M1.connector_views(kind)
pg = views["drafter.layers.0.k"].read_pages([0, 1])
views["drafter.layers.0.v"].write_pages([2, 3], pg)
ttnn.synchronize_device(mesh)
if not C.MOCK:  # a device holds data: the imported units read back equal to the blob (bfp8: the encoder's values)
    back = M1.export_units(kind, n_units, n_units)
    row["units_round_trip_equal"] = bool(torch.equal(back, blob))
    pg2 = views["drafter.layers.0.v"].read_pages([2, 3])
    row["pages_round_trip_equal"] = bool(torch.equal(pg2, pg))
# the device timing row: the boot (PREP + block: the attention over the whole capacity, c = cap - 64) captured ONCE
# and replayed `reps` times (no eager row); the mock has no timing
reps = int(args.get("reps", 0))
if not C.MOCK and reps > 0 and refused is None and "block" in census:
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    kind.boot()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(10):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t3 = time.time()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    row["boot_us_per_replay"] = round((time.time() - t3) / reps * 1e6, 1)
    row["reps"] = reps
    ttnn.release_trace(mesh, tid)
    log("traced boot (PREP + block over the full capacity):", row["boot_us_per_replay"], "us per replay")
row.update(units_blob_shape=list(blob.shape), unit_bytes=M1.UnitFormat.__new__(M1.UnitFormat).__dict__ and int(blob[0].numel() * 2),
           units_api_s=round(time.time() - t2, 1), views=len(views))  # fmt: skip
row.update(census=census, step_launches=ops.launches, program_cache_keys=n_keys, new_keys=new_keys, refused=refused,
           status="pass" if (new_keys == 0 and row["bfp8_mirror_bitwise"] and (refused is None or os.environ["GLM53_DFLASH2_SHARD"] == "1")) else "fail")  # fmt: skip
S.record("tests", row)
RESULT = dict(ok=row["status"] == "pass", census=census, dram_mib=dram_mb, refused=refused)
