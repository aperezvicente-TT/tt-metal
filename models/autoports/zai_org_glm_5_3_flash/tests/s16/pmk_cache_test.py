# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python layout tests of tt/pmk_cache.py (no device): the SEGMENT TABLE round trip and its kernel-side constants
(kernels/pmk/cache_common.hpp), and the write plans against the op path's index math in tt/prefill_tp.py --
chunk_pages (v1: one chunk at a chunk start), write_packed's lat_idx / pool_idx / ring_rows loops (v2: packed
segments; the loops are repeated here verbatim from write_packed). usage: python pmk_cache_test.py"""

import os
import re
import sys

import torch

sys.path.insert(0, os.environ.get("PMK_TREE", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))))  # fmt: skip
import models.autoports.zai_org_glm_5_3_flash.tt.pmk_cache as PC  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import NULL_BLOCK, chunk_pages  # noqa: E402

BLOCK, TILE = PC.BLOCK, PC.TILE
Seg = PC.Segment
n_checks = 0


def check(cond, msg):
    global n_checks
    n_checks += 1
    assert cond, msg


# ---------------------------------------------------------------- 1. the table: encode / decode, kernel constants
hpp = open(os.path.join(PC.KDIR, "cache_common.hpp")).read()
consts = {m.group(1): int(m.group(2), 0) for m in re.finditer(r"\b([A-Z_0-9]+) = (0x[0-9a-fA-F]+|\d+)u?[,;]", hpp)}
for name in ("SEG_MAGIC", "SEG_NONE", "NULL_BLOCK", "HDR_WORDS", "SEG_WORDS", "S_SLOT", "S_ROW0", "S_NVALID",
             "S_POS0", "S_PTROW", "S_FLAGS", "S_NROWS", "F_SAMPLE", "F_FRESH", "F_CONT", "H_NSEG", "H_TROWS"):  # fmt: skip
    check(name in consts, f"{name} not in cache_common.hpp")
check(consts["SEG_MAGIC"] == PC.SEG_MAGIC and consts["SEG_NONE"] == PC.SEG_NONE, "magic / none")
check(consts["NULL_BLOCK"] == PC.NULL_BLOCK == NULL_BLOCK, "null block")
check((consts["HDR_WORDS"], consts["SEG_WORDS"]) == (PC.HDR_WORDS, PC.SEG_WORDS), "words")
check([consts[k] for k in ("S_SLOT", "S_ROW0", "S_NVALID", "S_POS0", "S_PTROW", "S_FLAGS", "S_NROWS")] == list(range(7)),
      "segment field order")  # fmt: skip
check((consts["F_SAMPLE"], consts["F_FRESH"], consts["F_CONT"]) == (PC.F_SAMPLE, PC.F_FRESH, PC.F_CONT), "flags")
tab = PC.SegTable(2048, [Seg(slot=3, n_valid=700), Seg(slot=9, n_valid=129, pos0=256, cont=True, sample=False),
                         Seg(slot=1, n_valid=1)])  # fmt: skip
w = tab.words()
check(w.numel() * 4 == PC.table_bytes(), "table bytes")
check([s.row0 for s, _ in tab.segments] == [0, 768, 1024], [s.row0 for s, _ in tab.segments])
check([n for _, n in tab.segments] == [768, 256, 128], "padded spans")
check(int(w[PC.HDR_WORDS + PC.SEG_WORDS * 3]) == PC.SEG_NONE, "empty entry")
t2 = PC.SegTable.decode(w)
check(t2.words().tolist() == w.tolist(), "round trip")
check(t2.segments[1][0].flags == PC.F_CONT and t2.segments[0][0].flags == (PC.F_SAMPLE | PC.F_FRESH), "flags encoded")
check(tab.seg_of_row(767) == 0 and tab.seg_of_row(1023) == 1 and tab.seg_of_row(1152) is None, "seg_of_row")
check(tab.valid(0, 699) and not tab.valid(0, 700) and tab.pos(1, 800) == 288, "valid / pos")
for bad in ([Seg(slot=3, n_valid=2049)], [Seg(slot=3, n_valid=10), Seg(slot=3, n_valid=10)],
            [Seg(slot=3, n_valid=10, pos0=100)]):  # fmt: skip
    try:
        PC.SegTable(2048, bad)
        check(False, f"accepted {bad}")
    except AssertionError as e:
        if "accepted" in str(e):
            raise

# ---------------------------------------------------------------- 2. v1 vs chunk_pages (one chunk of a slot's row)
BPS = 64  # max_seq 4096 / 64
g = torch.Generator().manual_seed(7)
for c, pos0, nv in [(2048, 0, 2048), (2048, 0, 1000), (2048, 0, 1), (2048, 2048, 2048), (2048, 2048, 300),
                    (1024, 0, 5), (1024, 1024, 1024), (512, 512, 1)]:  # fmt: skip
    # the generator's row: the slot's blocks up to its coverage, padded with the last covered page (chunk_pages' doc)
    cov = min(BPS, -(-(pos0 + nv) // BLOCK) + int(torch.randint(0, 3, (1,), generator=g)))
    blocks = torch.randperm(200, generator=g)[:cov] + 1  # never the null block
    row = torch.cat([blocks, blocks[-1:].repeat(BPS - cov)])
    tab = PC.SegTable(c, [Seg(slot=0, n_valid=nv, pos0=pos0)])
    pt = row.reshape(1, -1)
    plan = dict(PC.plan_latent(tab, pt))
    ref = chunk_pages(row, pos0, nv, c)
    end_p = -(-nv // BLOCK)  # chunk pages with valid rows
    for i in range(c // BLOCK):
        if i < end_p:
            check(plan.get(i) == int(ref[i]), f"c={c} pos0={pos0} nv={nv}: page {i} {plan.get(i)} vs {int(ref[i])}")
        else:
            check(i not in plan, f"pad page {i} written")
    plan_n = dict(PC.plan_latent(tab, pt, write_null=True))
    for i in range(end_p, c // BLOCK):
        check(plan_n[i] == NULL_BLOCK, "WRITE_NULL -> the null block")
        # the op path's spare page is the null block or a covered page with no valid row: never a page with valid rows
        check(int(ref[i]) == NULL_BLOCK or int(ref[i]) not in {int(ref[j]) for j in range(end_p)}, "spare aliases")

# ---------------------------------------------------------------- 3. v2 vs write_packed's loops (packed segments)
t, pps = 2048, 4096 // PC.POOL_ROWS
reqs = [(5, 700, 0), (9, 129, 256), (1, 1, 0), (2, 516, 0)]  # (slot, nv, st): rows 0, 768, 896, 1024 .. 1536
tab = PC.SegTable(t, [Seg(slot=s, n_valid=nv, pos0=st, cont=st > 0) for s, nv, st in reqs])
offs = [s.row0 for s, _ in tab.segments]
rows = {s: (torch.randperm(500, generator=g)[:BPS] + 1) for s, _, _ in reqs}
pt = torch.stack([rows[s] if s in rows else torch.full((BPS,), -1) for s in range(10)])
# write_packed's lat_idx (verbatim)
wt = 512 // TILE
lat = torch.full((t // TILE * wt,), 0xFFFFFFFF, dtype=torch.int64)
pidx = torch.full((t // PC.POOL_ROWS * (PC.INDEX_DIM // TILE),), 0xFFFFFFFF, dtype=torch.int64)
rr_all = {}
for (slot, nv, st), o in zip(reqs, offs):
    row = rows[slot]
    for p in range(-(-nv // TILE)):
        pos = st + p * TILE
        page = int(row[pos // BLOCK])
        dst = page * (BLOCK // TILE) + (pos % BLOCK) // TILE
        src = o // TILE + p
        lat[src * wt : (src + 1) * wt] = torch.arange(dst * wt, (dst + 1) * wt)
    page_ = PC.POOL_ROWS
    for p in range(-(-nv // page_)):
        src, dst = o // page_ + p, int(slot) * pps + st // page_ + p
        pidx[src * PC.POOL_TILES : (src + 1) * PC.POOL_TILES] = torch.arange(
            dst * PC.POOL_TILES, (dst + 1) * PC.POOL_TILES
        )
    rr = torch.full((TILE,), t, dtype=torch.int64)
    end = st + nv
    base = PC.KPOOL * (end // PC.KPOOL)
    for j in range(PC.KPOOL):
        pos = base + j if base + j < end else base - PC.KPOOL + j
        if 0 <= pos - st < nv:
            rr[j] = o + pos - st
    rr_all[slot] = rr
# latent: the op path writes 32-row tile rows up to ceil(nv / 32); the PMK writes whole 64-row pages (the extra tile
# row of a page is the segment's own pad rows): op tiles <= PMK tiles <= op tiles + pad tile rows of the same pages
plan = PC.plan_latent(tab, pt)
pmk_tiles = {}
for i, p in plan:
    for k in range(PC.TPB):
        pmk_tiles[i * PC.TPB + k] = p * PC.TPB + k
op_tiles = {i: int(v) for i, v in enumerate(lat.tolist()) if v != 0xFFFFFFFF}
for s_, d_ in op_tiles.items():
    check(pmk_tiles.get(s_) == d_, f"latent tile {s_}: pmk {pmk_tiles.get(s_)} vs op {d_}")
for s_, d_ in pmk_tiles.items():
    if s_ not in op_tiles:
        r_ = (s_ // wt) * TILE
        k = tab.seg_of_row(r_)
        check(k is not None and (r_ - tab.segments[k][0].row0) // BLOCK * BLOCK < tab.segments[k][0].n_valid,
              f"PMK tile {s_} outside the op path's tiles is not a pad tile row of a valid page")  # fmt: skip
        check(d_ // PC.TPB == pmk_tiles[(s_ // wt - 1) * wt] // PC.TPB, "same page as its valid tile row")
# pools: equal sets
pool_plan = PC.plan_pools(tab, pps)
pmk_pool = {}
for j, p in pool_plan:
    for c_ in range(PC.POOL_TILES):
        pmk_pool[j * PC.POOL_TILES + c_] = p * PC.POOL_TILES + c_
op_pool = {i: int(v) for i, v in enumerate(pidx.tolist()) if v != 0xFFFFFFFF}
check(pmk_pool == op_pool, f"pool tiles differ: {len(pmk_pool)} vs {len(op_pool)}")
# rings: equal rows
for slot, rws in PC.plan_ring(tab):
    exp = [None if int(v) == t else int(v) for v in rr_all[slot][: PC.KPOOL]]
    check(rws == exp, f"ring rows of slot {slot}: {rws} vs {exp}")
check(
    [r for _, r in PC.plan_ring(tab)][2] == [1024, None, None, None], "a 1-token segment at position 0: ring row 0 only"
)
# local-rows source: every chip's plan is a disjoint part of the full plan
full = set(PC.plan_latent(tab, pt))
parts = [set((i + r * 4, p) for i, p in PC.plan_latent(tab, pt, src_row0=r * 256, src_rows=256)) for r in range(8)]
check(set.union(*parts) == full and sum(len(p) for p in parts) == len(full), "sequence-parallel source split")
# state
check(PC.plan_state(tab, 32) == [(0, 0, 5), (1, 0, 9), (2, 0, 1), (3, 0, 2)], PC.plan_state(tab, 32))
check(PC.plan_state(PC.SegTable(128, [Seg(slot=40, n_valid=3)]), 32) == [(0, 1, 8)], "chunk 1 row 8")

# ---------------------------------------------------------------- 4. apply_* on host tensors (the unit's expectation)
src = torch.randn(t, 512).bfloat16()
cache = torch.randn(600, 1, BLOCK, 512).bfloat16()
out = PC.apply_latent(plan, src, cache)
for (slot, nv, st), o in zip(reqs, offs):
    for q in range(nv):
        p = int(rows[slot][(st + q) // BLOCK])
        check(torch.equal(out[p, 0, (st + q) % BLOCK], src[o + q]), f"row {q} of slot {slot}")
changed = {i for i in range(600) if not torch.equal(out[i], cache[i])}
check(changed <= {p for _, p in plan}, "only planned pages changed")

print(f"PMK_CACHE_TEST PASS checks={n_checks}")
for kind, tp in (("sparse_dsa", 8), ("sparse_kda", 8), ("sparse_dsa", 2), ("sparse_kda", 2)):
    print(f"estimate {kind} tp{tp} T=2048:", PC.estimate(2048, tp, kind))
