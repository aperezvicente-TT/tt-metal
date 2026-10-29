# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair / mesh): GLM53_DSA_PAGED_POOLS op unit (Stage 12's paged pooled keys rev 3, the padded layout:
page lb = latent block lb, its 16 pooled rows in rows 0-15). On device, TRACED (one eager compile launch, then the
capture, then replays): (1) the prefill write -- random pooled keys [1, 1, C / 4, 128] bf16 -> tt/prefill_tp.py
paged_pool_src (one padded [32, 128] tile row per 64-token block) -> tiles_to into a [NB, 1, 32, 128] store by a
block-indexed idx (block b -> page perm[b], tiles 4 perm[b] .. + 3); (2) the prefill read -- page_gather of the blocks'
pages in block order + the compaction of SparseCtx.paged_pools (rows 0-15 of each block) back to [1, 1, C / 4, 128].
Checks, bitwise vs the host: every written page's rows 0-15 == its block's pooled rows, untouched pages stay zero, and
the read-back == the original pooled keys. ARGS: c (2048), pages (2 x the blocks), replays (20), seed, tag, record."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as P  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
CH = int(args.get("c", 2048))
NBLK = CH // P.BLOCK
NB = int(args.get("pages", 2 * NBLK))
REPLAYS = int(args.get("replays", 20))
tag = args.get("tag", "p3-paged-pools-unit")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None

g = torch.Generator().manual_seed(int(args.get("seed", 7)))
pooled_h = torch.randn(1, 1, CH // 4, 128, generator=g).bfloat16()
perm = torch.randperm(NB, generator=g)[:NBLK]  # block b -> page perm[b] (distinct)
idx_h = torch.stack([4 * perm + j for j in range(4)], dim=1).reshape(1, -1).to(torch.int32)  # [1, NBLK x 4]
table_h = perm.to(torch.int32).reshape(1, -1)  # the read: page_gather's page table (block order)

pooled = ttnn.from_torch(pooled_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=P.DRAM,
                         mesh_mapper=repl)  # fmt: skip
idx = ttnn.from_torch(idx_h, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=P.DRAM,
                      mesh_mapper=repl)  # fmt: skip
table = ttnn.from_torch(table_h, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=P.DRAM,
                        mesh_mapper=repl)  # fmt: skip
store = ttnn.from_torch(torch.zeros(NB, 1, 32, 128).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh, memory_config=P.DRAM, mesh_mapper=repl)  # fmt: skip
pad = ttnn.from_torch(torch.zeros(1, 1, NBLK * 32, 128).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                      device=mesh, memory_config=P.DRAM, mesh_mapper=repl)  # fmt: skip
back = ttnn.from_torch(torch.zeros(1, 1, CH // 4, 128).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=mesh, memory_config=P.DRAM, mesh_mapper=repl)  # fmt: skip


def step():
    src = P.paged_pool_src(pooled, CH)  # the prefill write (DsaIndexer.keys under the shard + paged)
    P.tiles_to(src, idx, store)
    ttnn.deallocate(src)
    P.page_gather(store, table, pad, NBLK, 128 // P.TILE)  # the prefill read (SparseCtx.paged_pools' compaction)
    rm_ = ttnn.to_layout(pad, ttnn.ROW_MAJOR_LAYOUT, memory_config=P.DRAM)
    top = ttnn.slice(ttnn.reshape(rm_, (1, NBLK, P.TILE, 128)), (0, 0, 0, 0), (1, NBLK, P.POOL_BLOCK_ROWS, 128))
    ttnn.deallocate(rm_)
    flat = ttnn.to_layout(ttnn.reshape(top, (1, 1, NBLK * P.POOL_BLOCK_ROWS, 128)), ttnn.TILE_LAYOUT,
                          memory_config=P.DRAM)  # fmt: skip
    ttnn.deallocate(top)
    ttnn.copy(flat, back)
    ttnn.deallocate(flat)


host = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_)  # noqa: E731
step()  # the compile launch (eager, before the capture)
ttnn.synchronize_device(mesh)
traced = not C.MOCK
ts = []
if traced:
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    step()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.zeros(NB, 1, 32, 128).bfloat16(), dtype=ttnn.bfloat16,
                                                    layout=ttnn.TILE_LAYOUT, mesh_mapper=repl), store)  # fmt: skip
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
if C.MOCK:
    S.log(f"{tag}: mock -- built and launched once")
    RESULT = []
else:
    st = host(store).float()
    p0 = pooled_h.float().reshape(NBLK, P.POOL_BLOCK_ROWS, 128)
    write_ok = all(torch.equal(st[int(perm[b]), 0, : P.POOL_BLOCK_ROWS], p0[b]) for b in range(NBLK))
    untouched = sorted(set(range(NB)) - set(int(v) for v in perm))
    zero_ok = all(bool((st[u] == 0).all()) for u in untouched)
    rb = host(back).float()
    read_ok = torch.equal(rb, pooled_h.float())
    row = dict(test=f"paged pooled keys op unit (GLM53_DSA_PAGED_POOLS, padded layout): paged_pool_src + tiles_to into "
                    f"{NB} pages by a block-indexed idx, then the block-page gather + rows 0-15 compaction, C {CH}, traced, "
                    f"tp{TPN}", milestone="P3", write_bitwise=write_ok, untouched_pages_zero=zero_ok,
               readback_bitwise=read_ok, blocks=NBLK, pages=NB, traced=traced, replay_us=C.stats(ts).get("median_us"),
               replays=REPLAYS, status="pass" if (write_ok and zero_ok and read_ok and traced) else "fail", tp=TPN, tag=tag)  # fmt: skip
    S.log(row)
    if rec:
        C.record("tests", row)
    RESULT = [row]
