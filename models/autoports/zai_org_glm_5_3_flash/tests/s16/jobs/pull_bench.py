# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): Stage 14's pull-form go / no-go microbenchmark (w8/progb_2ms_design.md step 4; program
2026-10-25 22:26:06Z). Can compute cores PULL weight chunks out of bank-adjacent stagers' L1 rings at ~470-490 GB/s per
chip? One interleaved DRAM tensor of PB-byte pages (page p in bank p mod 8), traced, three forms:
  ADJ   the 24 bank-adjacent cores (ttnn MESH_ROW_PLUS_X, 3 per bank) read their own bank's pages straight into L1
        (7a's 493 GB/s pattern: the reference ceiling);
  ALL   every core of the grid reads its share of the pages straight from DRAM (the all-core rotating pattern ~304 GB/s,
        = today's self-stream B readers);
  ALLW  as ALL, but each core's page count proportional to its measured ALL rate (stride scheduling over the page
        sequence, so every core's pages stay spread over the 8 banks): the slow far cores get fewer pages;
  PULL  the 24 stagers stream their bank's pages into an S-slot L1 ring; the other cores pull chunk (stager j, n) with an
        L1 -> L1 NoC read once j's PRODUCED >= n + 1 and +1 j's CONSUMED[n mod S] after it (chunks dealt to the pullers
        by global index n x 24 + j: every puller draws from all 8 banks, as a MoE unit does).
Per form: the compile launch (outside any trace), one trace capture, `replays` replays; the per-core probe rows of the
last replay give the aggregate GB/s = bytes moved / (last end - first start) at 1,350 MHz, per chip. ARGS: pb_kb (64),
mb (256), slots (4), replays (5), forms (["ADJ", "ALL", "PULL"]), tag, record."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

C = S.C
mesh, args = MESH, ARGS  # noqa: F821
TPN = int(mesh.get_num_devices())
PB = int(args.get("pb_kb", 64)) * 1024
NP = int(args.get("mb", 256)) * 1024 * 1024 // PB
SL = int(args.get("slots", 4))
DEPTH = int(args.get("all_depth", 1))  # ALL: reads in flight per core
REPLAYS = int(args.get("replays", 5))
FORMS = list(args.get("forms", ["ADJ", "ALL", "PULL"]))
tag = args.get("tag", "pull-bench")
rec = bool(args.get("record", True))
KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "bench")
MHZ = 1350.0
row = dict(test=f"Stage 14 pull-form microbenchmark: {NP} x {PB // 1024}-KB pages, slots {SL}, all-depth {DEPTH}, tp{TPN}", milestone="W8", all_depth=DEPTH,
           tag=tag, pb_kb=PB // 1024, pages=NP, slots=SL, forms={})  # fmt: skip

repl = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
words = PB // 4
grid = mesh.compute_with_storage_grid_size()
GX, GY = int(grid.x), int(grid.y)
adj = ttnn._ttnn.operations.matmul.dram_bank_reader_assignments(
    mesh, num_workers_per_dram_bank=3, dram_reader_placement=ttnn._ttnn.operations.matmul.DramReaderPlacement.MESH_ROW_PLUS_X)
stagers = [((int(c.x), int(c.y)), int(b), int(i)) for c, b, i in adj]
assert len(stagers) == 24 and len({s[0] for s in stagers}) == 24, stagers
st_xy = [s[0] for s in stagers]
others = [(x, y) for y in range(GY) for x in range(GX) if (x, y) not in set(st_xy)]
allc = [(x, y) for y in range(GY) for x in range(GX)]
noc = lambda xy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))  # noqa: E731
crs_of = lambda cores: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])  # noqa: E731
row["stagers"] = [[list(xy), b, i] for xy, b, i in stagers]


def dram(shape, dtype=ttnn.uint32):
    return ttnn.from_torch(torch.zeros(*shape, dtype=torch.int32), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=repl)  # fmt: skip


try:
    w = ttnn.from_torch(torch.arange(NP * words, dtype=torch.int32).reshape(NP, words), dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=repl)  # fmt: skip
    probe = dram((len(allc), 8))
    acc = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    # the stagers' page lists: bank b, index i -> pages p = b + 8 (i + 3 k)
    st_pages = []
    for _, b, i in stagers:
        first, stride = b + 8 * i, 24
        count = len(range(first, NP, stride))
        st_pages.append((first, stride, count))

    def cbs(cores):
        crs = crs_of(cores)
        mk = lambda idx, total: ttnn.CBDescriptor(total_size=total, core_ranges=crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.uint32, page_size=total)])  # fmt: skip
        return [mk(0, SL * PB), mk(1, 16 * 17 + 64), mk(2, max(2, DEPTH) * PB)]

    def kd(src, cores, ct, rt, defines, proc, noc_=None):
        return ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                     core_ranges=crs_of(cores), compile_time_args=ct, runtime_args=rt, defines=defines,
                                     config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc_ or ttnn.NOC.RISCV_0_default))  # fmt: skip

    B0 = ttnn.DataMovementProcessor.RISCV_0

    core_rate = {}

    def deal_weighted(weights):
        """pages -> cores in proportion to weights (stride scheduling: the next page goes to the core with the smallest
        (assigned + 1) / weight), so each core's pages interleave over the banks"""
        import heapq
        h = [((1.0) / max(1e-6, weights[xy]), k, xy) for k, xy in enumerate(allc)]
        heapq.heapify(h)
        out = {xy: [] for xy in allc}
        cap = 150  # runtime args: 5 + 2 x pages per core
        for pg in range(NP):
            v, k, xy = heapq.heappop(h)
            out[xy].append(pg)
            if len(out[xy]) < cap:
                heapq.heappush(h, ((len(out[xy]) + 1) / max(1e-6, weights[xy]), k, xy))
        return out

    def program(form):
        pa = probe.buffer_address()
        prow = {xy: k for k, xy in enumerate(allc)}
        kernels, cores = [], []
        if form in ("ADJ", "PULL"):
            rt = ttnn.RuntimeArgs()
            for (xy, _, _), (first, stride, count) in zip(stagers, st_pages):
                rt[xy[0]][xy[1]] = [w.buffer_address(), first, stride, count, pa, prow[xy]]
            kernels.append(kd("pull_stager.cpp", st_xy, [PB, SL] + acc, rt, [("PULL_DIRECT", "1")] if form == "ADJ" else [], B0))
            cores += st_xy
        if form == "PULL":
            lists = {xy: [] for xy in others}
            for n in range(max(c for _, _, c in st_pages)):
                for j, (_, _, count) in enumerate(st_pages):
                    if n < count:
                        g = n * 24 + j
                        lists[others[g % len(others)]].append((j, n))
            xys = [v for xy in st_xy for v in (int(noc(xy).x), int(noc(xy).y))]
            rt = ttnn.RuntimeArgs()
            for xy in others:
                pr = [v for jn in lists[xy] for v in jn]
                rt[xy[0]][xy[1]] = [w.buffer_address(), pa, prow[xy], 24] + xys + [len(lists[xy])] + pr
            kernels.append(kd("pull_consumer.cpp", others, [PB, SL] + acc, rt, [], B0))
            cores += others
        if form in ("ALL", "ALLW", "ALL1"):  # ALL1 (Stage 14 B2.3, the combine's NoC-path question): ALL on the NCRISC / NoC1
            rt = ttnn.RuntimeArgs()
            dealt = deal_weighted(core_rate) if form == "ALLW" else None
            for k, xy in enumerate(allc):
                pages = dealt[xy] if dealt is not None else list(range(k, NP, len(allc)))
                rt[xy[0]][xy[1]] = [w.buffer_address(), pa, prow[xy], 0, len(pages)] + [v for p in pages for v in (p, 0)]
            # PULL_FORM_<form>: a distinct program per form -- generic_op's cache keys ignore runtime args, so ALLW (the same
            # kernels / CBs / CT args as ALL, other per-core arg counts) hit ALL's cached program and aborted on the arg
            # sizes (d0 card2 22:42:16Z)
            kernels.append(kd("pull_consumer.cpp", allc, [PB, SL] + acc, rt, [("PULL_DIRECT_ALL", "1"), ("PULL_ALL_DEPTH", str(DEPTH)),
                                                                            ("PULL_FORM_" + form, "1")],
                              ttnn.DataMovementProcessor.RISCV_1 if form == "ALL1" else B0,
                              ttnn.NOC.RISCV_1_default if form == "ALL1" else None))  # fmt: skip
            cores = allc
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs(allc)), cores

    def rate(cores, byte_cores=None):
        t = ttnn.to_torch(ttnn.get_device_tensors(probe)[0] if TPN > 1 else probe).to(torch.int64) & 0xFFFFFFFF
        idx = [k for k, xy in enumerate(allc) if xy in set(cores)]
        r = t[idx]
        ok = bool((r[:, 7] == 0x9017B001).all())
        start, end, kb = r[:, 0], r[:, 1], r[:, 2]
        kb_moved = kb  # PULL: the bytes DELIVERED (the pullers'), not the stagers' copy of the same data again
        if byte_cores is not None:
            kb_moved = t[[k for k, xy in enumerate(allc) if xy in set(byte_cores)]][:, 2]
        span = float((end - start.min()).max()) / MHZ
        mb = float(kb_moved.sum()) / 1024
        per = ((kb * 1024) / ((end - start).clamp(min=1).double() / MHZ) / 1e3).tolist()  # GB/s per core, own span
        if form_now[0] == "ALL":
            for kk, xy in enumerate([xy for xy in allc if xy in set(cores)]):
                core_rate[xy] = per[kk]
        return dict(magic_ok=ok, mb=round(mb, 1), span_us=round(span, 1), gbps=round(float(kb_moved.sum()) * 1024 / span / 1e3, 1),
                    core_gbps_min=round(min(per), 2), core_gbps_max=round(max(per), 2),
                    wait_frac_mean=round(float((r[:, 3] * 16).double().mean() / max(1.0, float((end - start).double().mean()))), 3),
                    polls_mean=round(float(r[:, 4].double().mean()), 1))  # fmt: skip

    form_now = [None]
    for form in FORMS:
        form_now[0] = form
        if form == "ALLW" and not core_rate:
            row["forms"][form] = "needs ALL first"
            continue
        prog, cores = program(form)
        io = [w, probe]
        ttnn.generic_op(io, prog)  # the compile launch (outside any trace)
        ttnn.synchronize_device(mesh)
        if C.MOCK:
            row["forms"][form] = "mock"
            continue
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            ttnn.generic_op(io, prog)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        row["forms"][form] = rate(cores, others if form == "PULL" else None)
        ttnn.release_trace(mesh, tid)
        S.log(f"{tag}: {form} {row['forms'][form]}")
    row["status"] = "mock" if C.MOCK else ("pass" if all(v.get("magic_ok") for v in row["forms"].values()) else "fail")
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:2000])
S.log(row)
if rec and not C.MOCK:
    C.record("perf", row)
RESULT = row
