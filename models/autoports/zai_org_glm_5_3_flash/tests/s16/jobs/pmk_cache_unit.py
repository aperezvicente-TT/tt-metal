# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP2 pair -- or any mesh under the mock): the PMK cache-write phase (tt/pmk_cache.py, kernels/pmk/)
vs the op path on the SAME random inputs, read back and compared bitwise, per chip:
  L3 (sparse_dsa): latent pages (paged_fill_cache / tiles_to), pooled-key store pages (paged_fill_cache / tiles_to),
                   ring rows (DsaIndexer._ring_src / _ring_to);
  L4 (sparse_kda): the final S / conv carry into the slot's state store (kernels/prefill/state_io.cpp STATE_OUT).
Two segment tables on the same T = 1,024 rows: v1 (one fresh segment, 1,000 valid rows: a partial last page) and v2
(a fresh 500-row segment + a 300-row continuation at position 256 of another slot in another state chunk + 128 pad
rows). The PMK program (3 parts on 3 disjoint writer-core sets, one generic_op) is captured in a trace and replayed:
the write span per layer kind comes from the replay medians. Under the mock the kernels build and the plans run; the
compares are skipped (garbage readback).
ARGS: t (1024), writers (8), reps (5), max_seq (2048), tag, record. Criterion: every compare bitwise (status pass)."""
import os
import sys
import time
import types

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.pmk_cache as PC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402

C = S.C
mesh, args = MESH, ARGS  # noqa: F821
T = int(args.get("t", 1024))
NW = int(args.get("writers", 8))
REPS = int(args.get("reps", 5))
MAXSEQ = int(args.get("max_seq", 2048))
tag = args.get("tag", "pmk-cache-unit")
TPN = int(mesh.get_num_devices())
H_R, W_R = 64 // TPN, 24576 // TPN
CHUNK, B = 32, 64  # two state chunks
BPS, PPS = MAXSEQ // PC.BLOCK, MAXSEQ // PC.POOL_ROWS
NBLK = 2 * BPS + 1
BF16, FP32, DRAM = ttnn.bfloat16, ttnn.float32, ttnn.DRAM_MEMORY_CONFIG
repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
g = torch.Generator().manual_seed(16)
dev = lambda x, dt, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(x, dtype=dt, layout=layout, device=mesh,  # noqa: E731
                                                             memory_config=DRAM, mesh_mapper=repl)  # fmt: skip
host = lambda t_, i=0: ttnn.to_torch(ttnn.get_device_tensors(t_)[i] if TPN > 1 else t_)  # noqa: E731
S.log(f"pmk_cache_unit tp{TPN} T={T} writers={NW} H_r={H_R} W_r={W_R} mock={C.MOCK}")

# ---------------------------------------------------------------- inputs (host) and the two caches (op / pmk)
lat_h = torch.randn(T, 512, generator=g).bfloat16()
kg_h = torch.randn(T, PC.KG_W, generator=g).bfloat16()
pooled_h = torch.randn(T // PC.KPOOL, PC.INDEX_DIM, generator=g).bfloat16()
cache0 = torch.randn(NBLK, 1, PC.BLOCK, 512, generator=g).bfloat16()
store0 = torch.randn(B * PPS, 1, 32, PC.INDEX_DIM, generator=g).bfloat16()
ring0 = torch.randn(B, 1, 32, PC.KG_W, generator=g).bfloat16()
stackS0 = [torch.randn(CHUNK, H_R, 128, 128, generator=g) for _ in range(2)]
stackC0 = [torch.randn(1, 3, CHUNK, W_R, generator=g).bfloat16() for _ in range(2)]
scrS_h = [torch.randn(1, H_R, 128, 128, generator=g) for _ in range(2)]
scrC_h = [torch.randn(1, 3, 1, W_R, generator=g).bfloat16() for _ in range(2)]
pt_h = torch.full((B, BPS), -1, dtype=torch.int32)
pt_h[5] = torch.arange(1, BPS + 1)  # slot 5: blocks 1..32; slot 40: 33..64 (chunk 1, row 8); block 0 = null
pt_h[40] = torch.arange(BPS + 1, 2 * BPS + 1)

lat, kg, pooled = dev(lat_h.reshape(1, 1, T, 512), BF16), dev(kg_h.reshape(1, 1, T, -1), BF16), dev(pooled_h.reshape(1, 1, -1, PC.INDEX_DIM), BF16)  # fmt: skip
pt = dev(pt_h, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
scrS, scrC = [dev(x, FP32) for x in scrS_h], [dev(x, BF16) for x in scrC_h]
two = lambda x, dt: (dev(x, dt), dev(x, dt))  # noqa: E731  # (op, pmk) copies of the same initial bytes
cache_op, cache_pk = two(cache0, BF16)
store_op, store_pk = two(store0, BF16)
ring_op, ring_pk = two(ring0, BF16)
stS_op, stS_pk = zip(*[two(x, FP32) for x in stackS0])
stC_op, stC_pk = zip(*[two(x, BF16) for x in stackC0])
scenarios = {
    "v1": PC.SegTable(T, [PC.Segment(slot=5, n_valid=1000)]),
    "v2": PC.SegTable(T, [PC.Segment(slot=5, n_valid=500), PC.Segment(slot=40, n_valid=300, pos0=256, cont=True)]),
}
segtab = scenarios["v1"].tensor(mesh, repl)

# ---------------------------------------------------------------- the op path (prefill_tp's pieces, per segment)
ctl = dev(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
kg_rm = dev(kg_h.float().reshape(1, 1, T, -1), FP32, ttnn.ROW_MAJOR_LAYOUT)  # _ring_src typecasts to bf16: exact
n_s, n_c = H_R * 16, 3 * (W_R // 32)
grid = mesh.compute_with_storage_grid_size()
sio_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def state_out_op(k):
    """kernels/prefill/state_io.cpp STATE_OUT, as StateIO._run, on segment k's scratch (ctl = its slot)"""
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [ctl.buffer_address(), scrS[k].buffer_address(), scrC[k].buffer_address()] + sum(
        [[stS_op[c].buffer_address(), stC_op[c].buffer_address()] for c in range(2)], [])  # fmt: skip
    kd = ttnn.KernelDescriptor(kernel_source=os.path.join(PF.KDIR, "prefill", "state_io.cpp"),
                               source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=sio_core,
                               compile_time_args=[n_s, n_c, CHUNK, 2], runtime_args=rt,
                               defines=[("SIO_CB", "0"), ("STATE_OUT", "1")],
                               config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                        noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
    cb = ttnn.CBDescriptor(total_size=2 * 4096 + 2 * 2048, core_ranges=sio_core, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=2 * 4096 + 2 * 2048)])  # fmt: skip
    ttnn.generic_op([ctl, ctl], ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb]))


def op_path(tab: PC.SegTable):
    put = lambda v, dt, dst: ttnn.copy_host_to_device_tensor(ttnn.from_torch(v, dtype=dt,  # noqa: E731
                                                                              layout=ttnn.ROW_MAJOR_LAYOUT,
                                                                              mesh_mapper=repl), dst)  # fmt: skip
    segs = [s for s, _ in tab.segments]
    if len(segs) == 1:  # v1: one chunk at the segment's start (TPPrefill chunk_inputs / write_inputs)
        s = segs[0]
        ct = PF.chunk_pages(pt_h[s.slot].to(torch.int64), s.pos0, s.n_valid, T).to(torch.int32).reshape(1, -1)
        ttnn.experimental.paged_fill_cache(cache_op, lat, dev(ct, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), batch_idx=0)
        pool_t = (
            (s.slot * PPS + s.pos0 // PC.POOL_ROWS + torch.arange(T // PC.POOL_ROWS)).to(torch.int32).reshape(1, -1)
        )
        ttnn.experimental.paged_fill_cache(
            store_op, pooled, dev(pool_t, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), batch_idx=0
        )
    else:  # v2: write_packed's lat_idx / pool_idx (tiles_to)
        wt, pg = 512 // 32, PC.POOL_ROWS
        li = torch.full((T // 32 * wt,), 0xFFFFFFFF, dtype=torch.int64)
        pi = torch.full((T // pg * PC.POOL_TILES,), 0xFFFFFFFF, dtype=torch.int64)
        for s in segs:
            row = pt_h[s.slot].to(torch.int64)
            for p in range(-(-s.n_valid // 32)):
                pos = s.pos0 + p * 32
                d = int(row[pos // PC.BLOCK]) * (PC.BLOCK // 32) + (pos % PC.BLOCK) // 32
                sr = s.row0 // 32 + p
                li[sr * wt : (sr + 1) * wt] = torch.arange(d * wt, (d + 1) * wt)
            for p in range(-(-s.n_valid // pg)):
                sr, d = s.row0 // pg + p, s.slot * PPS + s.pos0 // pg + p
                pi[sr * PC.POOL_TILES : (sr + 1) * PC.POOL_TILES] = torch.arange(
                    d * PC.POOL_TILES, (d + 1) * PC.POOL_TILES
                )
        PF.tiles_to(lat, dev(li.reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), cache_op)
        PF.tiles_to(pooled, dev(pi.reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), store_op)
    ext = PF.DsaIndexer._ring_src(None, kg_rm, T)
    fake = types.SimpleNamespace(ring=ring_op)
    for k, s in enumerate(segs):
        rr = torch.full((1, 32), T, dtype=torch.int32)
        end, base = s.pos0 + s.n_valid, PC.KPOOL * ((s.pos0 + s.n_valid) // PC.KPOOL)
        for j in range(PC.KPOOL):
            pos = base + j if base + j < end else base - PC.KPOOL + j
            if 0 <= pos - s.pos0 < s.n_valid:
                rr[0, j] = s.row0 + pos - s.pos0
        PF.DsaIndexer._ring_to(fake, ext, dev(rr, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                               dev((torch.arange(PC.RING_TILES) + s.slot * PC.RING_TILES).to(torch.int32).reshape(1, -1),
                                   ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))  # fmt: skip
        v = torch.zeros(1, 32, dtype=torch.int32)
        v[0, 0] = s.slot
        put(v, ttnn.uint32, ctl)
        state_out_op(k)
    ttnn.deallocate(ext)


# ---------------------------------------------------------------- the PMK program (one generic_op, traced)
cores = PC.free_cores(mesh, 3 * NW, taken={(0, 0)})
c_lat, c_dsa, c_st = cores[:NW], cores[NW : 2 * NW], cores[2 * NW :]
parts = [
    PC.latent_parts(lat, cache_pk, pt, segtab, c_lat, t_rows=T),
    PC.dsa_parts(pooled, store_pk, kg, ring_pk, segtab, c_dsa, t_rows=T, pages_per_slot=PPS),
    PC.state_parts(list(zip(scrS, scrC)), list(zip(stS_pk, stC_pk)), segtab, c_st, chunk=CHUNK),
]
io = [lat, cache_pk, pt, segtab, pooled, store_pk, kg, ring_pk] + scrS + scrC + list(stS_pk) + list(stC_pk)
lat_only = [PC.latent_parts(lat, cache_pk, pt, segtab, c_lat, t_rows=T)]
dsa_only = [PC.dsa_parts(pooled, store_pk, kg, ring_pk, segtab, c_dsa, t_rows=T, pages_per_slot=PPS)]
st_only = [PC.state_parts(list(zip(scrS, scrC)), list(zip(stS_pk, stC_pk)), segtab, c_st, chunk=CHUNK)]
t0 = time.time()
# compile EVERY program (both scenarios' op paths + the four PMK variants) before the FIRST capture: a program first
# compiled after a capture keeps a live buffer ('device buffers still alive before trace replay', d0 10:10Z)
for tab_ in scenarios.values():
    tab_.write(segtab, repl)
    op_path(tab_)
for p_ in (parts, lat_only, dsa_only, st_only):
    PC.run(p_, io)
ttnn.synchronize_device(mesh)
tr_all = C.Traced(ttnn, mesh, lambda: PC.run(parts, io))
S.log(f"PMK program compiled + captured in {time.time() - t0:.1f} s (3 parts on {3 * NW} cores)")
timed = {}
for name, p in (("lat", lat_only), ("dsa", dsa_only), ("state", st_only)):
    tr = C.Traced(ttnn, mesh, lambda p=p: PC.run(p, io))
    timed[name] = C.stats(tr.replays(REPS)).get("median_us")
    tr.release()
S.log(f"replay medians us: all={C.stats(tr_all.replays(REPS)).get('median_us')} parts={timed}")


def compare(name: str, tab: PC.SegTable) -> dict:
    """op path then PMK on the same initial bytes: per chip, every written region bitwise + untouched regions intact"""
    segtab_w = tab
    segtab_w.write(segtab, repl)
    for t_, h0 in ((cache_op, cache0), (cache_pk, cache0), (store_op, store0), (store_pk, store0), (ring_op, ring0),
                   (ring_pk, ring0)):  # fmt: skip
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(h0, dtype=t_.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=repl), t_
        )
    for c in range(2):
        for t_, h0 in (
            (stS_op[c], stackS0[c]),
            (stS_pk[c], stackS0[c]),
            (stC_op[c], stackC0[c]),
            (stC_pk[c], stackC0[c]),
        ):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(h0, dtype=t_.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=repl), t_
            )
    op_path(tab)
    tr_all.replay()
    ttnn.synchronize_device(mesh)
    row = dict(scenario=name, segments=[dataclasses_dict(s) for s, _ in tab.segments])
    if C.MOCK:
        row["status"] = "mock"
        return row
    plan_l, plan_p = PC.plan_latent(tab, pt_h), PC.plan_pools(tab, PPS)
    exp_cache = PC.apply_latent(plan_l, lat_h, cache0)
    exp_store = PC.apply_pools(plan_p, pooled_h, store0)
    exp_ring = PC.apply_ring(PC.plan_ring(tab), kg_h, ring0)
    exp_state = PC.apply_state(PC.plan_state(tab, CHUNK), list(zip(scrS_h, scrC_h)), list(zip(stackS0, stackC0)))
    bad = []
    for chip in range(TPN):
        c_op, c_pk = host(cache_op, chip), host(cache_pk, chip)
        s_op, s_pk = host(store_op, chip), host(store_pk, chip)
        r_op, r_pk = host(ring_op, chip), host(ring_pk, chip)
        ne = lambda a, b: int((a != b).sum())  # noqa: E731
        for s, _ in tab.segments:  # the tokens' rows: pmk == op == expected
            for q0 in range(0, s.n_valid, PC.BLOCK):
                p = int(pt_h[s.slot][(s.pos0 + q0) // PC.BLOCK])
                r0, r1 = (s.pos0 + q0) % PC.BLOCK, (s.pos0 + q0) % PC.BLOCK + min(PC.BLOCK, s.n_valid - q0)
                for nm, a in (("op", c_op), ("exp", exp_cache)):
                    d = ne(c_pk[p, 0, r0:r1], a[p, 0, r0:r1])
                    if d:
                        bad.append(f"chip{chip} slot{s.slot} latent page {p} rows {r0}:{r1} pmk vs {nm}: {d} elements")
        touched = {p for _, p in plan_l}
        for p in range(NBLK):
            if p not in touched and ne(c_pk[p], cache0[p]):
                bad.append(f"chip{chip} latent page {p} untouched by the plan but changed (pmk)")
        if ne(s_pk, s_op) or ne(s_pk, exp_store):
            bad.append(f"chip{chip} pooled-key store: pmk vs op {ne(s_pk, s_op)} / vs exp {ne(s_pk, exp_store)}")
        for s, _ in tab.segments:
            if ne(r_pk[s.slot], r_op[s.slot]) or ne(r_pk[s.slot], exp_ring[s.slot]):
                bad.append(f"chip{chip} ring slot {s.slot}: pmk vs op {ne(r_pk[s.slot], r_op[s.slot])} / vs exp "
                           f"{ne(r_pk[s.slot], exp_ring[s.slot])}")  # fmt: skip
        if ne(r_pk, r_op):
            bad.append(f"chip{chip} ring other slots changed: {ne(r_pk, r_op)}")
        for c in range(2):
            S_op, S_pk, C_op, C_pk = (
                host(stS_op[c], chip),
                host(stS_pk[c], chip),
                host(stC_op[c], chip),
                host(stC_pk[c], chip),
            )
            # the op path's scratches are the SAME tensors (per segment): expected = apply_state on the host copies
            for nm, a, b in (("S op", S_pk, S_op), ("S exp", S_pk, exp_state[c][0]), ("C op", C_pk, C_op),
                             ("C exp", C_pk, exp_state[c][1])):  # fmt: skip
                if ne(a, b):
                    bad.append(f"chip{chip} state chunk {c} {nm}: {ne(a, b)} elements differ")
    row["mismatches"] = bad[:20]
    row["status"] = "pass" if not bad else "fail"
    return row


def dataclasses_dict(s):
    return dict(slot=s.slot, row0=s.row0, n_valid=s.n_valid, pos0=s.pos0, cont=s.cont)


rows = []
for name, tab in scenarios.items():
    r = compare(name, tab)
    S.log(r)
    rows.append(r)
tr_all.release()
est = {k: PC.estimate(T, TPN, k, writers=NW) for k in ("sparse_dsa", "sparse_kda")}
result = dict(test=f"PMK cache-write phase vs the op path, bitwise (T={T}, tp{TPN}, {NW} writers per part)",
              milestone="P2", status="mock" if C.MOCK else ("pass" if all(r["status"] == "pass" for r in rows) else "fail"),
              traced=not C.MOCK, t=T, scenarios=rows, replay_us=timed, estimate=est, tag=tag)  # fmt: skip
S.log(result)
if args.get("record", True) and not C.MOCK:
    C.record("pcc", result)
RESULT = result
