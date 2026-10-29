# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS-O (GLM53_PMK_WS_O; program [session 10] 2026-10-28 05:24:26Z GO; doc/ttnn/16-prefill-production/plan_1m/
l5_ws_oproj_design.md): program A's P7 o_proj WEIGHT-STATIONARY under the KDA scan. Stage 1 = the kernel set + this host
op + the one-chip unit (tests/s16/jobs/pmk_wso_unit.py); the program-A wiring (the scan leader's SEM_ON on the lanes,
the RS8 at FS_RDY_MULT 64 / QSLICE 1, the composite) is stage 2.

y[M, N] = on[M, K] @ W_o[K, N] (TP8 per chip: M 2,048, K 1,024, N 4,096, W bf8):
- the 64 members = ONE exact 8 x 8 multicast rectangle are the receivers; member m holds N columns [64 m, 64 m + 64) =
  2 column tiles x 32 K tiles = 64 bf8 tiles (69,632 B) RESIDENT, read once in place from the image's o_proj DSLinear
  weight (DRAM WIDTH_SHARDED [K, n_pad], shard b = bank b = [K, SW tiles] row-major: the PMK_A_INPLACE layout P7's
  PfStream streams today) -- no new DRAM tensor;
- TWO lanes OUTSIDE the box, lane L on NoC L -- exactly one multicaster per NoC into the box (the stage-1c hang class:
  never a sender inside the box, never two on one NoC / VC 4) -- take the even / odd chunks c (= row tiles of `on`, as
  the scan finishes them: SEM_ON >= c + 1), read each from DRAM and multicast it (64 KB) into the members' S-slot ring
  with the PfStream credit protocol (EPOCH per-launch credits) as single packets with writes_flushed (mc 1, from the
  start; mc 2: a barrier each);
- per chunk each member accumulates its 2 column tiles over K ascending in 2 fp32 DST tiles (the PfStream member's
  per-tile MAC sequence: bitwise), packs once, stores them at today's y positions and raises block c's H5 word (64
  raises per block per launch: the RS8 readers' FS_RDY_MULT 64, QSLICE 1).
Protocol + word maps: kernels/pfwo/pfwo_common.hpp."""

from __future__ import annotations

import os
import re

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "pfwo")
TILE = 32
BANKS = 8
CB_XR, CB_WS, CB_OUT, CB_STG = 0, 2, 3, 5
SEM_V0, SEM_ON = 0, 12  # VALID of lane L = SEM_V0 + L; SEM_ON = the scan's chunk count (kda SEM_ON's role)
TB = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
H5_RAISES_MAX = 1 << 30  # the RS readiness words: 32-bit, waited with a wrap-safe signed difference (fs_common.hpp wait_ge)


def _header_consts() -> dict:
    out = {}
    for line in open(os.path.join(KDIR, "pfwo_common.hpp")):
        if line.lstrip().startswith("constexpr uint32_t"):
            out.update({m.group(1): int(m.group(2)) for m in re.finditer(r"(\w+) = (\d+)", line)})
    return out


_W = _header_consts()
SYNC_WORDS, WORD = _W["SYNC_WORDS"], _W["WORD"]


def kernel_text(name: str) -> str:
    src = open(os.path.join(KDIR, name)).read()
    return src.replace('#include "pfwo_common.hpp"', open(os.path.join(KDIR, "pfwo_common.hpp")).read())


def _crs(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


def ws_o_on() -> bool:
    """GLM53_PMK_WS_O=1 (default off: today's PfStream P7)"""
    return os.environ.get("GLM53_PMK_WS_O", "0") == "1"


def h5_check(raises_per_block: int, launches_between_resets: int | None = None) -> dict:
    """the program's 05:24:26Z condition (2): the H5 counter must cover the raises of one block per launch. The RS
    readiness words (FabricStreamRS LREADY / OWNRDY) are 32-bit and waited as wait_ge(word, g x FS_RDY_MULT) with a
    WRAP-SAFE signed difference (kernels/fabric_stream/fs_common.hpp wait_ge): correct while one launch's raises (and the
    gap between a waiter and its target) stay < 2^31, whatever the launch count g. Asserted here; the RS readers take
    FS_RDY_MULT = raises (rs_read.cpp / red_read.cpp, static_assert'd to QSLICE 1)."""
    r = int(raises_per_block)
    assert 0 < r <= H5_RAISES_MAX, ("H5 raises per block per launch past the wrap-safe range", r)
    if launches_between_resets is not None:  # an EPOCH-reset form: the word's absolute peak must stay below 2^32
        assert r * int(launches_between_resets) < (1 << 32), ("H5 word overflow before its reset", r, launches_between_resets)
    return dict(raises_per_block=r, counter_bits=32, compare="wrap-safe int32 difference", fs_rdy_mult=r, qslice=1)


def resident_weight(mesh, w: torch.Tensor, dtype) -> ttnn.Tensor:
    """the DSLinear resident layout (tt/optimized_decoder.DSLinear): [K, n_pad] DRAM WIDTH_SHARDED over the 8 banks,
    shard [K, n_pad / 8] row-major tiles -- replicated over the mesh (the unit's stand-in for the image's ds_o)"""
    k, n = int(w.shape[-2]), int(w.shape[-1])
    assert (n // TILE) % BANKS == 0 and k % TILE == 0, (k, n)
    shard = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(BANKS - 1, 0))}),
                           (k, n // BANKS), ttnn.ShardOrientation.ROW_MAJOR)  # fmt: skip
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard)
    return ttnn.from_torch(w.reshape(1, 1, k, n).float(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip


class PfStreamWO:
    """weight: the resident o_proj [K, n_pad] (resident_weight / the image's ds_o.weight). members: the 64 receivers,
    ONE exact multicast rectangle (members[m] holds columns [cols m, cols (m + 1)) in tiles). lanes: (lane 0 on NoC 0,
    lane 1 on NoC 1), both OUTSIDE the rectangle. slots: the ring depth. mc: 1 (default) / 2. In the PMK composite
    (pmk_assembly, GLM53_PMK_WS_O): y = today's o_proj output (the RS8's source), cb = the engine's CB indices, sem_v0 =
    PfStream's VALID ids (10 / 11: the launch-global block numbering continues: qbase), sem_on = kda SEM_ON (7), sync =
    the reused layer's lane sync shard (one L1 allocation for every layer)."""

    def __init__(self, mesh, weight, *, members, lanes, m_rows: int = 2048, slots: int = 2, mc: int = 1,
                 fidelity=ttnn.MathFidelity.HiFi3, y=None, cb: dict | None = None, sem_v0: int = SEM_V0,
                 sem_on: int = SEM_ON, sync=None, qbase: int = 0):  # fmt: skip
        self.mesh = mesh
        self.weight = weight
        self.members = [tuple(c) for c in members]
        self.lanes = [tuple(c) for c in lanes]
        assert len(self.lanes) == 2, ("exactly one lane per NoC: 2 lanes", self.lanes)
        assert not set(self.lanes) & set(self.members), ("the lanes stay OUTSIDE the members' box", self.lanes)
        self.mt = int(m_rows) // TILE
        self.k, self.n_pad = int(weight.shape[-2]), int(weight.shape[-1])
        self.kt, self.nt = self.k // TILE, self.n_pad // TILE
        self.cols = self.nt // len(self.members)
        assert self.cols * len(self.members) == self.nt and self.cols <= 4, (self.nt, len(self.members))
        self.sw = self.nt // BANKS
        assert self.sw % self.cols == 0, ("a member's columns inside one bank's shard", self.sw, self.cols)
        self.tbw = TB[weight.dtype]
        self.s, self.mc, self.fidelity = int(slots), int(mc), fidelity
        assert 1 <= self.s <= 8 and self.mc in (1, 2), (self.s, self.mc)
        self.cb = {**dict(xr=CB_XR, ws=CB_WS, out=CB_OUT, stg=CB_STG), **{k: int(v) for k, v in (cb or {}).items()}}
        self.sem_v0, self.sem_on, self.qbase = int(sem_v0), int(sem_on), int(qbase)
        self._noc = lambda xy: (lambda c: (int(c.x), int(c.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))
        self.box = self._rectangle(self.members)
        lx, ly, hx, hy = self.box
        for c in self.lanes:  # the lanes outside the box in NoC coordinates too
            x, y_ = self._noc(c)
            assert not (lx <= x <= hx and ly <= y_ <= hy), ("a lane inside the members' box", c, self.box)
        self.h5 = h5_check(len(self.members))
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.own_y = y is None
        self.y = y if y is not None else ttnn.from_torch(
            torch.zeros(1, 1, self.mt * TILE, self.n_pad, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        assert tuple(self.y.shape)[-2:] == (self.mt * TILE, self.n_pad), (self.y.shape, self.mt, self.n_pad)
        words = SYNC_WORDS * WORD // 4
        self.own_sync = sync is None
        if sync is None:
            smc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                    ttnn.ShardSpec(_crs(self.lanes), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
            sync = ttnn.from_torch(torch.zeros(2, words, dtype=torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=smc, mesh_mapper=rep)  # fmt: skip
        self.sync = sync
        self.srcs = {f: kernel_text(f) for f in ("pfwo_lane.cpp", "pfwo_rd.cpp", "pfwo_c.cpp", "pfwo_wr.cpp")}

    @staticmethod
    def pack_h5(h5: list) -> list:
        """[(noc x, noc y, L1 address)] per block -> [base] + [x | y << 8 | (address - base) << 16] (RT_H5)"""
        base = min(int(a) for _, _, a in h5)
        out = [base]
        for x, y, a in h5:
            off = int(a) - base
            assert 0 <= int(x) < 256 and 0 <= int(y) < 256 and 0 <= off < (1 << 16) and off % 4 == 0, (x, y, a, base)
            out.append(int(x) | (int(y) << 8) | (off << 16))
        return out

    def _rectangle(self, cores):
        """an exact multicast rectangle in NoC coordinates (pf_stream.PfStreamMM._rectangle's rule)"""
        v = [self._noc(c) for c in cores]
        lx, ly = min(x for x, _ in v), min(y for _, y in v)
        hx, hy = max(x for x, _ in v), max(y for _, y in v)
        g = self.mesh.compute_with_storage_grid_size()
        inside = {(x, y) for y in range(int(g.y)) for x in range(int(g.x))
                  if (lambda c: lx <= c[0] <= hx and ly <= c[1] <= hy)(self._noc((x, y)))}  # fmt: skip
        assert inside == set(cores), f"not an exact multicast rectangle: {sorted(inside ^ set(cores))}"
        return lx, ly, hx, hy

    def footprint(self) -> dict:
        """static CB bytes per role (the unit's census rows)"""
        return dict(member=self.s * self.kt * 2048 + self.kt * self.cols * self.tbw + 2 * self.cols * 2048,
                    lane=2 * self.kt * 2048, ring=self.s * self.kt * 2048, w_slice=self.kt * self.cols * self.tbw)  # fmt: skip

    def ct(self, x) -> list:
        """the ONE compile-time list (every kernel of the phase): x = the `on` rows (DRAM [M, K] bf16 tiles)"""
        c = self.cb
        ct = [self.kt, self.mt, self.s, self.cols, self.tbw, self.sw, 2048, len(self.members), self.mc, c["ws"], c["xr"],
              c["out"], c["stg"], self.sem_v0, self.sem_on, self.mt // 2, self.qbase]  # fmt: skip
        assert len(ct) == _W["CT_ACC"] and self.qbase % 2 == 0, (len(ct), self.qbase)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        return ct + acc(self.y) + acc(x)

    def rt_members(self, h5: list, incs: list | None = None) -> dict:
        """{member: the ONE runtime list} (the reader reads its first RT_H5 words; the writer the H5 table, then its raise
        value: incs[i] for member i, 1 by default = one +1 per block, QSLICE 1)"""
        assert len(h5) == self.mt, (len(h5), self.mt)
        incs = [1] * len(self.members) if incs is None else [int(v) for v in incs]
        assert len(incs) == len(self.members), (len(incs), len(self.members))
        sync = self.sync.buffer_address()
        lv = [self._noc(c) for c in self.lanes]
        h5p = self.pack_h5(h5)
        out = {}
        for i, c in enumerate(self.members):
            col0 = i * self.cols
            bank, j0 = divmod(col0, self.sw)
            out[c] = [sync, int(i == 0), self.weight.buffer_address(), bank, j0, self.y.buffer_address(), self.nt, col0,
                      lv[0][0], lv[0][1], lv[1][0], lv[1][1]] + h5p + [incs[i]]  # fmt: skip
            assert len(out[c]) == _W["RT_H5"] + 2 + self.mt
        return out

    def slice_incs(self, q: int, nt: int) -> tuple:
        """PMK_RS8_QSLICE Q > 1: (incs per member, holders per slice M). Member i holds output columns [i COLS, (i + 1)
        COLS) of every block, inside column slice s = i COLS // (NT / Q); each slice has M holders; a holder of slice s
        raises its block's readiness word by (M + 1)^s (the RS readers' FS_SLICE_DIGITS = M digit wait)"""
        q, nt = int(q), int(nt)
        assert q > 1 and nt % q == 0, (q, nt)
        w = nt // q
        sl = []
        for i in range(len(self.members)):
            a, b = i * self.cols, (i + 1) * self.cols - 1
            assert a // w == b // w, ("WS-O QSLICE: a member's columns straddle two slices", i, self.cols, w)
            sl.append(a // w)
        m = [sl.count(s_) for s_ in range(q)]
        assert len(set(m)) == 1 and m[0] > 0, ("WS-O QSLICE: equal holders per slice", m)
        base = m[0] + 1
        assert base ** q < 2 ** 32, (base, q)
        return [base ** s_ for s_ in sl], m[0]

    def rt_lanes(self, x) -> dict:
        lx, ly, hx, hy = self.box
        out = {c: [self.sync.buffer_address(), L, lx, ly, hx, hy, len(self.members), x.buffer_address()]
               for L, c in enumerate(self.lanes)}  # fmt: skip
        assert all(len(v) == _W["RT_XADDR"] + 1 for v in out.values())
        return out

    def footprint(self) -> dict:
        """static CB bytes per role (the unit's census rows)"""
        return dict(member=self.s * self.kt * 2048 + self.kt * self.cols * self.tbw + 2 * self.cols * 2048,
                    lane=2 * self.kt * 2048, ring=self.s * self.kt * 2048, w_slice=self.kt * self.cols * self.tbw)  # fmt: skip

    def parts(self, x, h5: list) -> tuple:
        """the STANDALONE program's (kernels, cbs, {semaphore id: cores}) (the unit): x = the `on` rows; h5 = per block
        c the (noc x, noc y, L1 address) of its H5 word"""
        rts = dict(self.rt_members(h5))
        rts.update(self.rt_lanes(x))

        def rt_of(cores):
            r = ttnn.RuntimeArgs()
            for c in cores:
                r[c[0]][c[1]] = rts[c]
            return r

        ct = self.ct(x)
        kd = lambda src, cores, cfg: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=_crs(cores),
            compile_time_args=ct, runtime_args=rt_of(cores), config=cfg, defines=[])  # fmt: skip
        dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True, dst_full_sync_en=False,
                                          math_approx_mode=False)  # fmt: skip
        kernels = [kd(self.srcs["pfwo_lane.cpp"], self.lanes, dyn),
                   kd(self.srcs["pfwo_rd.cpp"], self.members, ttnn.ReaderConfigDescriptor()),
                   kd(self.srcs["pfwo_c.cpp"], self.members, cc),
                   kd(self.srcs["pfwo_wr.cpp"], self.members, ttnn.WriterConfigDescriptor())]  # fmt: skip
        c = self.cb
        cb = lambda idx, dt, page, n, cores: ttnn.CBDescriptor(total_size=page * n, core_ranges=_crs(cores), format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)])  # noqa: E731  # fmt: skip
        cbs = [cb(c["xr"], ttnn.bfloat16, 2048, self.s * self.kt, self.members),
               cb(c["ws"], self.weight.dtype, self.tbw, self.kt * self.cols, self.members),
               cb(c["out"], ttnn.bfloat16, 2048, 2 * self.cols, self.members),
               cb(c["stg"], ttnn.bfloat16, 2048, 2 * self.kt, self.lanes)]  # fmt: skip
        both = self.members + self.lanes
        sems = {self.sem_v0: both, self.sem_v0 + 1: both, self.sem_on: both}
        return kernels, cbs, sems

    def lane_words(self, cycles_per_us: float = 1350.0) -> list:
        """per chip, per lane: GEN, the deadlock record and the last launch's waits (us)"""
        out = []
        for t in ttnn.get_device_tensors(self.sync) if int(self.mesh.get_num_devices()) > 1 else [self.sync]:
            h = ttnn.to_torch(t).reshape(2, SYNC_WORDS, WORD // 4)[:, :, 0].to(torch.int64) & 0xFFFFFFFF
            us = lambda L, w: round(float(h[L, _W[w]]) / cycles_per_us, 1)  # noqa: E731
            out.append([dict(gen=int(h[L, _W["GEN"]]), dl_what=hex(int(h[L, _W["DL_WHAT"]])), dl_seen=int(h[L, _W["DL_SEEN"]]),
                             span_us=us(L, "P_SPAN"), ready_us=us(L, "P_READY"), on_us=us(L, "P_ON"),
                             read_us=us(L, "P_READ"), mc_us=us(L, "P_MC")) for L in range(2)])  # fmt: skip
        return out

    def release(self):
        if self.own_y:
            ttnn.deallocate(self.y)
        if self.own_sync:
            ttnn.deallocate(self.sync)
