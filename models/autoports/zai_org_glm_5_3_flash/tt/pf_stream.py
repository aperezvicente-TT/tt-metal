# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a PfStream: the PMK PfStreamPhase BODY (prefill_megakernel.md §4c; program 10:33Z): a projection
y[M, N] = x[M, K] @ W[K, N] at prefill width, activation-stationary and weights-streamed-once.

- Members (64, an exact rectangle of the grid: ONE multicast target): member G holds row tile G of x (32 x K bf16, the
  PMK XS 35 / IN0 40 block; the unit fills it from DRAM) and computes its 32 x N outputs N-block by N-block: the NB fp32
  outputs of an N-block stay in DST across all K-blocks, one bf16 pack per N-block (OUT 42). NB = 4 in a half-sync DST
  is the PMK member composite's mode (one compute config for every phase of the members' TRISC); NB = 8 needs a
  full-sync DST (the standalone unit's v2c / v3 rows).
- Streams (NST = N / (32 NW)): stream st = the K x NW tiles of its columns, K-row-major, contiguous in DRAM bank st mod 8
  (tt/mm_stream.py "mk-stream-v1", cores = NST), NH = NW / NB N-blocks. TWO streamers per stream (BRISC, dynamic NoC),
  lane L = 0 / 1: lane L sends the K-blocks j = L, L + 2, .. of every N-block of the stream (KB contiguous NB-tile
  reads per block into a 3-slot ring) on NoC L, each ONCE into the members' PARK 29 slot Q mod P (Q = the
  launch-global block: the phase's QBASE + its order), followed on that NoC by a multicast SET of the members' lane-L
  VALID (sem 10 / 11) to Q / 2 + 1. Two DRAM readers and two NoC injectors per N-block.
- Credits: members credit PARK slots AHEAD of the data (up to P = PARK past the next landing) on the block's streamer,
  in one of two READY words by the block's parity in that streamer's list (one word mixed blocks: d0 10:57Z watcher
  stop); each lane multicasts in stream order (TOKEN chain per lane, after a write barrier). The protocol is simulated
  host-side for the geometry before any program is built (tt/pfs_credit_model.py).
- pmk/proj lever 1 (rects / rt / n_split): the members form R x n_split exact multicast RECTANGLES; rectangle
  (h, t) holds row group h (its members hold RT consecutive row tiles each) and computes column group t (N / n_split
  columns = the streams t nst / n_split ..). Each rectangle has its own TWO streamer cores (lanes), each serving the
  group's SPC = nst / n_split streams in order (no TOKEN chain: every rectangle's lanes run concurrently). The 96-member
  layout (6 x 16, rt 2, n_split 3) and the model behind it: tt/pfs_member_model.py, pmk/proj/README.md.
- PMK form: parts() = the phase's ONE compile-time list (every kernel reads its own CT_x of kernels/pfs/pfs_common.hpp)
  and ONE runtime list per core (RT_x), the role sources, CB uses and semaphores; QBASE / SLOT let several stream phases
  share the rings, PARK and VALID in one launch. The unit's program is built from the same parts().
"""

from __future__ import annotations

import os
import re
import struct

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import pfs_credit_model
from models.autoports.zai_org_glm_5_3_flash.tt.mm_stream import TILE_BYTES, stream_layout, stream_tiles

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "pfs")
TILE = 32
CB_PARK, CB_X, CB_OUT, CB_RING = 29, 35, 42, 43  # §4b ids (PARK 29, XS 35, OUT 42; rings 43..)
SEM_VALID0, SEM_VALID1 = (
    10,
    11,
)  # §4b-ids: the weight multicast's VALID (one per NoC lane; READY lives on the streamers)
RING = 3
BANKS = 8  # DRAM banks = the resident DSLinear's shards = the in-place streams


def _header_consts() -> dict:
    out = {}
    for line in open(os.path.join(KDIR, "pfs_common.hpp")):
        if line.lstrip().startswith("constexpr uint32_t"):
            out.update({m.group(1): int(m.group(2)) for m in re.finditer(r"(\w+) = (\d+)", line)})
    return out


_W = _header_consts()
SYNC_WORDS, WORD = _W["SYNC_WORDS"], _W["WORD"]
_SYNC = ("READY", "READY1", "TOKEN", "GEN", "PARKADDR", "VALIDADDR0", "VALIDADDR1", "VAL0", "VAL1", "P_TOKEN", "P_READ",
         "P_READY", "P_FLUSH", "P_SPAN")  # fmt: skip
assert max(_W[k] for k in _SYNC) < SYNC_WORDS, _W
RT_LEN_STREAMER = _W["RT_SB"] + 2  # 7a's one stream per core; a core serving SPC streams: RT_SB + 2 SPC


def kernel_text(name: str) -> str:
    """kernel source with the header inlined (SOURCE_CODE kernels: no include path for the pfs dir)"""
    src = open(os.path.join(KDIR, name)).read()
    return src.replace('#include "pfs_common.hpp"', open(os.path.join(KDIR, "pfs_common.hpp")).read())


def _noc_xy(mesh, c) -> tuple:
    w = mesh.worker_core_from_logical_core(ttnn.CoreCoord(int(c[0]), int(c[1])))
    return int(w.x), int(w.y)


def _crs(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


SYNC_WORDS_PFS = 16  # kernels/pfs/pfs_common.hpp SYNC_WORDS


def pfs_mc_pkt_defines() -> list:
    """GLM53_PMK_PFS_MC_PKT (class A, the P7 streamer multicast stall: doc/ttnn/16-prefill-production/pmk/w7/hang_mcast/
    README.md; program [session 10] 2026-10-28 03:53:29Z): each PfStream block multicast as single packets (<=
    NOC_MAX_BURST_SIZE) -- 2 = a write barrier after each packet (the proof arm), 1 = writes_flushed (the serving form if
    2 holds); 3 = the BOUNDED multi-burst (<= 2 packets per multicast, writes_flushed between: 05:04:55Z's serving
    candidate); "" / 0 = today (one multi-burst multicast). Process-wide, like PFS_EPOCH."""
    v = os.environ.get("GLM53_PMK_PFS_MC_PKT", "0").strip() or "0"
    assert v in ("0", "1", "2", "3", "4"), f"GLM53_PMK_PFS_MC_PKT={v!r}: 0 / 1 / 2 / 3 / 4 (4 = the tt-metal #15018 rule: a write barrier before every multicast)"
    return [] if v == "0" else [("PFS_MC_PKT", v)]


def pfs_unicast_on() -> bool:
    """GLM53_PMK_PFS_UNICAST=1 (class A arm c, operator 2026-10-28T06:30:24Z; the MECHANISM PROOF only: ~x4.7 per layer,
    the members' 4-block credit window keeps ~one sender per lane, so 64 copies of each block leave one NIU): no PfStream
    multicast at all -- per block one unicast write per member, a write barrier, a 4-B VALID SET per member
    (kernels/pfs/pfs_stream.cpp PFS_UNICAST). Process-wide; excludes GLM53_PMK_PFS_MC_PKT."""
    on = os.environ.get("GLM53_PMK_PFS_UNICAST", "0") == "1"
    assert not (on and pfs_mc_pkt_defines()), "GLM53_PMK_PFS_UNICAST excludes GLM53_PMK_PFS_MC_PKT"
    return on


def pfs_epoch_defines() -> list:
    """GLM53_PMK_PFS_EPOCH=1 (program 2026-10-26 19:41:35Z order 2b, the k2 hang's PfStream credit / generation drift):
    per-launch credit space -- the streamer's READY / READY1 / TOKEN targets count THIS launch only and the words are zeroed
    at the end of every launch (kernels/pfs/pfs_stream.cpp PFS_EPOCH), + a DEADLOCK record (DL_WHAT / DL_SEEN) instead of a
    hang past PFS_WAIT_BUDGET cycles. Process-wide: every program sharing a sync slot must run the same mode."""
    if os.environ.get("GLM53_PMK_PFS_EPOCH", "0") != "1":
        return []
    out = [("PFS_EPOCH", "1")]
    # the deadlock unit's hooks (never served): GLM53_PMK_PFS_BUDGET = the wait budget in cycles; GLM53_PMK_PFS_TEST_DROP = b:
    # member 0 never credits block b (a forced lost credit -> the DEADLOCK record + a completed launch, not a hang)
    if os.environ.get("GLM53_PMK_PFS_BUDGET", ""):
        out.append(("PFS_WAIT_BUDGET", str(int(os.environ["GLM53_PMK_PFS_BUDGET"])) + "ull"))
    if os.environ.get("GLM53_PMK_PFS_TEST_DROP", ""):
        out.append(("PFS_TEST_DROP", str(int(os.environ["GLM53_PMK_PFS_TEST_DROP"]))))
    return out


def deadlock_records(mesh, sync, slots: int = 8) -> list:
    """the PFS_EPOCH deadlock records of every streamer core's slots: [(chip, shard, slot, what, seen)] for DL_WHAT != 0
    (what = 0xDEAD0000 | (word << 12) | block; word 0 = READY, 1 = READY1, 0xFFF = TOKEN)"""
    import ttnn  # noqa: PLC0415

    out = []
    words = SYNC_WORDS_PFS
    for d, t in enumerate(ttnn.get_device_tensors(sync) if int(mesh.get_num_devices()) > 1 else [sync]):
        h = ttnn.to_torch(t).reshape(-1, slots, words, 4)[:, :, :, 0].to(torch.int64) & 0xFFFFFFFF
        for shard in range(int(h.shape[0])):
            for sl in range(slots):
                what = int(h[shard, sl, 14])
                if what:
                    out.append((d, shard, sl, hex(what), int(h[shard, sl, 15])))
    return out


# GLM53_PMK_PFS_ROUTE (class A, operator 2026-10-28T06:30:24Z arm (b), program [session 10] 06:39:40Z): the BH NoC
# torus the route model walks (translated coordinates; doc/ttnn/16-prefill-production/pmk/w7/hang_mcast/census/route_class.py)
NOC_GRID = (17, 12)


def pfs_route_on() -> bool:
    """GLM53_PMK_PFS_ROUTE=1: every PfStream phase in the multi form with ONE sender per lane (SPC = the phase's streams,
    no TOKEN chain), lane 0 on a core whose NoC 0 approach to the members' low corner and lane 1 on a core whose NoC 1
    approach to the high corner neither wrap the torus, nor cross a non-worker row (0 / 1), nor touch a box tile
    (route_senders). Bitwise by construction: the same blocks, members, PARK slots and K order; only the sender moves."""
    return os.environ.get("GLM53_PMK_PFS_ROUTE", "0") == "1"


def noc_approach(s: tuple, t: tuple, noc: int) -> list:
    """the dimension-order path from s to t (exclusive of s): NoC 0 X+ then Y+, NoC 1 Y- then X- (wrap-around)"""
    nx, ny = NOC_GRID
    (x, y), path = s, []
    if noc == 0:
        while x != t[0]:
            x = (x + 1) % nx
            path.append((x, y))
        while y != t[1]:
            y = (y + 1) % ny
            path.append((x, y))
    else:
        while y != t[1]:
            y = (y - 1) % ny
            path.append((x, y))
        while x != t[0]:
            x = (x - 1) % nx
            path.append((x, y))
    return path


def route_senders(noc_of, members, candidates) -> list:
    """[lane 0 sender, lane 1 sender] (logical cores from `candidates`, in order of preference): the fewest-hop core whose
    multicast approach on NoC L to the members' box start corner (NoC 0: low, NoC 1: high) has no torus wrap, no row 0
    / 1 router and no box tile before the corner. Raises if a lane has none."""
    mv = [noc_of(c) for c in members]
    lx, ly, hx, hy = min(x for x, _ in mv), min(y for _, y in mv), max(x for x, _ in mv), max(y for _, y in mv)
    inbox = lambda q: lx <= q[0] <= hx and ly <= q[1] <= hy  # noqa: E731
    mset = {tuple(c) for c in members}
    out = []
    for noc, corner in ((0, (lx, ly)), (1, (hx, hy))):
        best = None
        for c in candidates:
            c = tuple(c)
            if c in mset or c in out:
                continue
            s = noc_of(c)
            if inbox(s):
                continue
            path = noc_approach(s, corner, noc)
            hops = [s] + path
            wrap = any(abs(b[0] - a[0]) > 1 or abs(b[1] - a[1]) > 1 for a, b in zip(hops, hops[1:]))
            if wrap or any(q[1] < 2 for q in path) or any(inbox(q) for q in path[:-1]):
                continue
            if best is None or len(path) < best[0]:
                best = (len(path), c)
        assert best is not None, ("GLM53_PMK_PFS_ROUTE: no wrap-free NoC %d sender" % noc, (lx, ly, hx, hy),
                                  [(tuple(c), noc_of(c)) for c in candidates])  # fmt: skip
        out.append(best[1])
    return out


class PfStreamMM:
    """y = x @ W on `members` (row tile G of x on members[G]) with W [K, N] streamed by the first 2 N / (32 nw) of
    `streamers`. One call = one launch (traced by the caller). slots = the sync slots on the streamers (one per stream
    phase sharing them in a launch); pass `sync` to share one sync tensor between phases."""

    def __init__(self, mesh, w: torch.Tensor, *, members, streamers, nb: int = 8, nw: int = 32, kb: int = 8,
                 park: int = 4, w_dtype=ttnn.bfloat8_b, fidelity=ttnn.MathFidelity.HiFi2, probe: bool = False,
                 sync=None, slots: int = 1, m_rows: int | None = None, epi: int = 0, limit: float = 10.0,
                 xsrc: bool = False, cb_x: int | None = None, cb_out: int | None = None, cb_park: int | None = None,
                 rt: int = 1, n_split: int = 1, rects: list | None = None, lane_noc=(0, 1),
                 out_dtype=None, resident=None, weight_of=None, joint=None, zskip=None):  # fmt: skip
        """epi: 0 bf16 tiles stored by the writer; 1 SwiGLU into the next phase's in0 (cb_out = that CB, the weight's
        columns = (gate, up) tile pairs); 2 ROW_MAJOR rows (y is a ROW_MAJOR tensor). xsrc: in0 pushed by the previous
        phase's compute (cb_x = the act CB). cb_park: this phase's own PARK CB (block sizes differ from the other phases).
        rt: row tiles per member (RT x NB fp32 tiles in DST). rects: the member rectangles (lists of cores, each an
        exact multicast rectangle; `members` = their concatenation); len(rects) = R x n_split, rectangle g = (h = g //
        n_split, t = g % n_split); streamers = 2 per rectangle (lane 0, 1), each serving nst / n_split streams."""
        self.mesh = mesh
        # zskip (GLM53_PMK_A_ZSKIP, cf3h-k4d8a L3): dict(k_lo, k_hi, n_from) in TILES -- the weight's nonzero K rows are K
        # tiles [k_lo, k_hi) and its columns from N tile n_from on are zero: the member compute skips those matmuls (exact:
        # zero products; the caller asserts the zeros). None = today's kernel (no define)
        self.zskip = dict(zskip) if zskip else None
        # resident (Stage 12's image line, PMK_A_INPLACE): the DSLinear's DRAM WIDTH_SHARDED weight [K, n_pad] read IN
        # PLACE -- shard b = bank b = the K x (n_pad / 8) column slab as row-major tiles [K, SHARD_W] (mk_readers.py:98),
        # which IS one mk-stream-v1 stream of NW = SHARD_W columns at in-bank start m0 = 0: 8 streams = 8 banks, no
        # upload, no bank padding, the tensor's own dtype. The caller passes nw = SHARD_W and w = None.
        self.resident = resident
        # joint (2026-10-25, the f_b / g_b DRAM reclaim): dict(weight, lay, base) -- this phase's NST streams are streams
        # base .. base + NST - 1 of a JOINT mk-stream-v1 copy shared with other phases (pmk_assembly lays f_b | g_b as one
        # [K, 2N] copy of 2 NST streams: every bank holds whole streams, no bank padding -- one 4-stream f_b copy alone
        # padded every bank to the fullest, 2x its bytes at TP8; one stream, 8x). Bank / in-bank start come from the
        # joint layout; the kernels are unchanged (the streamer's (bank, m0) args).
        if joint is None and weight_of is not None and getattr(weight_of, "joint", None) is not None:
            joint, weight_of = weight_of.joint, None  # a shared copy of a joint phase is that joint slice (same base)
        self.joint = joint
        self.stream_base = int(joint["base"]) if joint is not None else 0
        if resident is not None:
            assert w is None, "resident: no host weight"
            w0 = resident
        else:
            w0 = w[0] if isinstance(w, (list, tuple)) else w
        k, n = int(w0.shape[-2]), int(w0.shape[-1])
        if resident is not None:
            assert (n // TILE) % BANKS == 0 and nw == (n // TILE) // BANKS, (n, nw, "nw = the shard width in tiles")
            w_dtype = resident.dtype
        self.kt, self.nt = k // TILE, n // TILE
        assert self.nt % nw == 0 and nw % nb == 0 and self.kt % kb == 0 and rt * nb <= 8, (
            self.kt,
            self.nt,
            kb,
            nb,
            nw,
            rt,
        )
        self.nst = self.nt // nw
        self.rt, self.n_split = int(rt), int(n_split)
        self.lane_noc = tuple(int(v) for v in lane_noc)  # the NoC of lane 0 / 1 (u2: wrap routes from far streamers)
        assert len(self.lane_noc) == 2 and set(self.lane_noc) <= {0, 1}
        self.members = [tuple(c) for c in members]
        # GLM53_PMK_PFS_ROUTE: the multi form over ONE rectangle (the members), two senders chosen by route_senders below
        self.route = pfs_route_on()
        if self.route:
            # several row rectangles of ONE column group (PMK_P12_RECTS: 8 rows x 2 senders = 8 concurrent multicasters per
            # NoC) collapse into the one box: the same member order, so the same row tiles (row0 = the member's index)
            assert n_split == 1 and tuple(self.lane_noc) == (0, 1), ("GLM53_PMK_PFS_ROUTE: n_split 1, lanes on NoC 0 / 1",
                                                                     n_split, self.lane_noc)  # fmt: skip
            assert rects is None or [tuple(c) for r in rects for c in r] == self.members, "rects = the members, in order"
            rects = [list(members)]
        self.rects = [[tuple(c) for c in r] for r in (rects if rects is not None else [members])]
        assert [c for r in self.rects for c in r] == self.members, "members = the rectangles' concatenation"
        assert len(self.rects) % self.n_split == 0 and self.nst % self.n_split == 0, (
            len(self.rects),
            self.nst,
            n_split,
        )
        self.n_rows_groups = len(self.rects) // self.n_split
        self.nst_g = self.nst // self.n_split  # streams per rectangle
        self.multi = rects is not None
        self.spc = self.nst_g if self.multi else 1  # streams per streamer core
        self.ns = 2 * len(self.rects) if self.multi else 2 * self.nst  # streamers: 2 per rectangle (multi) / per stream
        self.ndest = len(self.rects[0])
        assert all(len(r) == self.ndest for r in self.rects), "equal rectangles"
        self.all_streamers = [tuple(c) for c in streamers]
        if self.route:  # the two wrap-free senders among the sync cores (pmk_assembly puts them in its streamer list)
            self.streamers = route_senders(lambda c: _noc_xy(mesh, c), self.rects[0], self.all_streamers)
        else:
            self.streamers = self.all_streamers[: self.ns]
        assert (
            len(self.streamers) == self.ns
        ), f"{len(self.rects)} rectangles x 2 lanes / {self.nst} streams need {self.ns} streamers"
        assert not set(self.all_streamers) & set(self.members)
        self.k, self.n, self.nb, self.nw, self.kb, self.park, self.probe = k, n, nb, nw, kb, park, probe
        self.mt = self.ndest * self.rt * self.n_rows_groups  # row tiles of the phase
        self.epi, self.limit, self.xsrc = epi, float(limit), bool(xsrc)
        self.cb_x = CB_X if cb_x is None else cb_x
        self.cb_out = CB_OUT if cb_out is None else cb_out
        self.cb_park = CB_PARK if cb_park is None else cb_park
        self.out_dtype = ttnn.bfloat16 if out_dtype is None else out_dtype  # assembly: fp32 stores (kda f / gate_out)
        self.out_tb = TILE_BYTES[self.out_dtype]
        assert epi != 1 or nb % 2 == 0
        self.nbk = self.kt // kb
        self.nblk = self.nt // self.n_split // nb  # N-blocks per member (its rectangle's column group)
        assert self.nbk % 2 == 0 and (self.nw // self.nb * self.nbk // 2) % 2 == 0, "even blocks per lane"
        assert 2 <= park <= 4, "READY parity words hold credits ahead of at most 4 PARK slots (pfs_common.hpp)"
        # the credit protocol, simulated before any device launch (program 10:59Z): random asynchronous schedules of
        # this geometry; raises on a mixed parity word, a busy slot, unpublished addresses or a deadlock. A streamer
        # core serving SPC streams = one list of SPC x NH N-blocks (the lane's blocks in order): check it as such.
        pfs_credit_model.check(1 if self.multi else self.nst, self.spc * (nw // nb), self.nbk, park, seeds=3, members=4,
                               launches=2)  # fmt: skip
        self.w_dtype, self.fidelity = w_dtype, fidelity
        self.tb = TILE_BYTES[w_dtype]
        self.bt = kb * nb
        self._noc = lambda xy: (lambda c: (int(c.x), int(c.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))
        self.rect_boxes = [self._rectangle(r) for r in self.rects]
        self.rect = self.rect_boxes[0]
        rep = ttnn.ReplicateTensorToMesh(mesh)
        # weight_of: another PfStreamMM of the SAME weight (the GroupStack's chunk instances of one layer): its DRAM stream
        # copy is shared, not uploaded again (45 layers x 4 instances ran the 2x4 mock out of DRAM, 2026-10-24 03:44Z)
        self.weight_of = weight_of
        self.own_weight = resident is None and weight_of is None and joint is None
        if resident is not None:
            assert self.nst == BANKS, self.nst
            self.lay = dict(version="resident-ds-v1", k=k, n=n, kt=self.kt, nt=nw, cores=BANKS, m0=[0] * BANKS,
                            pages_per_bank=self.kt * nw)  # fmt: skip
            self.weight = resident
            self.weight_bytes = 0  # no copy: the image's own tensor
        elif joint is not None:
            self.lay = joint["lay"]
            assert self.lay["k"] == k and self.lay["nt"] == nw and self.stream_base + self.nst <= self.lay["cores"], (
                self.lay, k, nw, self.stream_base, self.nst)  # fmt: skip
            self.weight_bytes = 0  # the joint copy's bytes: its owner (pmk_assembly) reports them
        else:
            self.lay = stream_layout(k, n, self.nst)
            self.weight_bytes = (
                self.lay["pages_per_bank"] * BANKS * self.tb
            )  # the padded copy (every bank = the fullest)
        assert self.lay["nt"] == nw
        if resident is not None:
            pass
        elif joint is not None:
            self.weight = joint["weight"]
        elif weight_of is not None:
            assert weight_of.lay == self.lay and weight_of.w_dtype == w_dtype, (
                "weight_of: another geometry",
                weight_of.lay,
                self.lay,
            )
            self.weight, self.weight_bytes = weight_of.weight, 0
        elif isinstance(w, (list, tuple)):  # assembly: PER-CHIP weights (TP shards, mesh order) sharded over the mesh
            n_dev = int(mesh.get_num_devices())
            assert len(w) == n_dev, (len(w), n_dev)
            st = torch.stack([stream_tiles(wi.float(), self.lay).reshape(1, -1, TILE) for wi in w], dim=0)
            if w_dtype == ttnn.bfloat16 and torch.equal(st.to(torch.bfloat16).float(), st):
                st = st.to(torch.bfloat16)  # exact: a bf16 host tensor (from_torch staged the fp32 one on device at 2x)
            self.weight = ttnn.from_torch(st, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))  # fmt: skip
        else:
            self.weight = ttnn.from_torch(stream_tiles(w.float(), self.lay).reshape(1, 1, -1, TILE), dtype=w_dtype,
                                          layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                          mesh_mapper=rep)  # fmt: skip
        self.own_sync = sync is None
        self.sync = sync if sync is not None else self.make_sync(mesh, self.all_streamers, slots)
        self.mprobe = None
        if probe:
            pw = _W["M_WORDS"] * WORD // 4
            mcm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                    ttnn.ShardSpec(_crs(self.members), [1, pw], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
            self.mprobe = ttnn.from_torch(torch.zeros(len(self.members), pw, dtype=torch.int32), dtype=ttnn.uint32,
                                          layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mcm, mesh_mapper=rep)  # fmt: skip
        m = m_rows or self.mt * TILE
        assert m == self.mt * TILE, (m, self.mt)
        # epi 1 (SwiGLU into the next phase's in0 CB) stores NOTHING: its y only carries the writer's compile-time accessor
        # args (a tile page) -> one tile, not [m, n] (cf3h-k B2: P13 gate_up's 2,048 x 2 i bf16 per W7 / G layer, never written)
        if epi == 1:
            m, n = TILE, TILE
        self.y = ttnn.from_torch(torch.zeros(1, 1, m, n, dtype=torch.bfloat16), dtype=self.out_dtype,
                                 layout=ttnn.ROW_MAJOR_LAYOUT if epi == 2 else ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)  # fmt: skip
        self.srcs = {f: kernel_text(f) for f in ("pfs_stream.cpp", "pfs_member_rd.cpp", "pfs_member_c.cpp",
                                                 "pfs_member_wr.cpp")}  # fmt: skip

    @staticmethod
    def make_sync(mesh, streamers, slots: int = 1):
        """the streamers' persistent sync words: `slots` x SYNC_WORDS per streamer (one slot per stream phase)"""
        words = slots * SYNC_WORDS * WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(streamers), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        return ttnn.from_torch(torch.zeros(len(streamers), words, dtype=torch.int32), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip

    def _rectangle(self, members=None):
        """the members' virtual (NoC) bounding box must hold exactly the members: no other worker core inside (columns
        that are not Tensix, e.g. virtual x 8 / 9 on the p300a, do not count: ttnn's 2D mcast matmul spans them)"""
        members = self.members if members is None else members
        v = [self._noc(c) for c in members]
        lx, ly = min(x for x, _ in v), min(y for _, y in v)
        hx, hy = max(x for x, _ in v), max(y for _, y in v)
        g = self.mesh.compute_with_storage_grid_size()
        inside = {(x, y) for y in range(int(g.y)) for x in range(int(g.x))
                  if (lambda c: lx <= c[0] <= hx and ly <= c[1] <= hy)(self._noc((x, y)))}  # fmt: skip
        assert inside == set(members), f"members are not an exact multicast rectangle: {sorted(inside ^ set(members))}"
        return lx, ly, hx, hy

    def _check_publishers(self, publishers: dict, credit_sources: dict):
        """host-mock assertion (program 18:26Z): every streamer that reads the PARK / VALID addresses from its
        sync words gets them from a member that is one of ITS credit sources (so the addresses land before the
        streamer's first READY wait can pass: no VALID multicast to address 0), and exactly one such publisher"""
        for si in range(self.ns):
            pubs = [m for m, pair in publishers.items() if si in pair]
            assert len(pubs) == 1, f"streamer {si}: {len(pubs)} address publishers"
            assert pubs[0] in credit_sources[si], f"streamer {si}: publisher {pubs[0]} is not one of its credit sources"

    @property
    def blocks(self) -> int:
        """this phase's blocks (the next phase's QBASE adds them)"""
        return self.nblk * self.nbk

    def footprint(self) -> dict:
        """L1 bytes per role (the PMK census numbers for this phase)"""
        member = self.rt * self.kt * 2048 + self.park * self.bt * self.tb + 2 * self.rt * self.nb * 2048
        return dict(member=member, streamer=RING * self.bt * self.tb + SYNC_WORDS * WORD,
                    park_slot=self.bt * self.tb, blocks=self.blocks, rt=self.rt, rects=len(self.rects),
                    members_per_rect=self.ndest, streams_per_core=self.spc)  # fmt: skip

    def zskip_defines(self) -> list:
        """GLM53_PMK_A_ZSKIP: the member compute's zero-tile skip defines (none when off: today's source)"""
        if not self.zskip:
            return []
        lo, hi, nf = int(self.zskip.get("k_lo", 0)), int(self.zskip.get("k_hi", self.kt)), int(self.zskip.get("n_from", self.nt))
        assert 0 <= lo < hi <= self.kt and 0 < nf <= self.nt, ("zskip", lo, hi, nf, self.kt, self.nt)
        return [("PFS_ZSKIP", "1"), ("PFS_ZK_LO", str(lo)), ("PFS_ZK_HI", str(hi)), ("PFS_ZN_FROM", str(nf))]

    def parts(self, out, *, x=None, qbase: int = 0, slot: int = 0, cb_x: int | None = None, cb_out: int | None = None,
              signals: dict | None = None, wait: tuple = (0, 0), x_fill: bool = True,
              wait_rt: dict | None = None, sig_slices: int = 1) -> dict:  # fmt: skip
        """the phase for a program: ct (ONE list, every kernel of the phase), rt {core: ONE list} (equal lengths),
        sources per role / RISC, CB uses, semaphores, defines. x = the unit's DRAM input (PFS_UNIT); None in the layer
        (the stationary block is already in cb_x). out = the output store (DRAM [rows, N] tile-interleaved).
        signals = {member: (noc_x, noc_y, l1_addr)}: +1 there after the member's whole output row is stored (H5).
        wait = (semaphore id, n): H2 outside a PMK composite, the reader waits it before handing in0 to the compute.
        x_fill=False (with x given): this phase reuses the stationary block an earlier phase of the launch filled.
        wait_rt = {member: n} (PFS_WAIT_RT, with x given): each member waits semaphore wait[0] >= ITS n before reading x
        (o_proj row groups under the scan: n = the chunks its rectangle's rows need); one more RT word per core.
        sig_slices = Q > 1 (PMK_RS8_QSLICE, the TP8 RS8's producer): the H5 raise once per column slice of the row
        (PFS_SIG_SLICES on the member WRITER only: returned as "wr_defines"); one full-width row tile per member."""
        assert qbase % 2 == 0
        sig_slices = int(sig_slices or 1)
        if sig_slices > 1:  # a slice = NBLK / Q whole N-blocks = columns [s N / Q, (s + 1) N / Q) of the member's row
            assert signals and (not self.multi or (len(self.rects) == 1 and self.n_split == 1)) and self.rt == 1 \
                and int(self.epi) == 0, \
                ("PFS_SIG_SLICES: signals, one full-width row tile per member, bf16 tiles", bool(signals), self.multi,
                 self.rt, self.epi)  # fmt: skip
            assert self.nblk % sig_slices == 0 and self.nblk * self.nb == self.nt, (self.nblk, self.nb, self.nt, sig_slices)
        cb_x = self.cb_x if cb_x is None else cb_x
        cb_out = self.cb_out if cb_out is None else cb_out
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ct = [self.nbk, self.kb, self.nb, self.nw, self.tb, self.park, self.ndest, CB_RING, self.kt, cb_x,
              self.cb_park, SEM_VALID0, SEM_VALID1, self.nblk, cb_out, qbase, slot, int(wait[0]), int(wait[1]),
              self.epi, struct.unpack("<I", struct.pack("<f", self.limit))[0], int(self.xsrc), self.spc, self.rt,
              self.nt]  # fmt: skip
        assert len(ct) == _W["CT_ACC"] and ct[_W["CT_QBASE"]] == qbase and ct[_W["CT_SLOT"]] == slot
        ct += acc(out) + (acc(x) if x is not None else [])
        sync = self.sync.buffer_address()
        svirt = [self._noc(c) for c in self.streamers]
        rt_len_streamer = _W["RT_SB"] + 2 * self.spc
        uni = pfs_unicast_on()
        if uni:  # PFS_UNICAST: the member columns / rows of each rectangle's box as bit masks after the (bank, m0) pairs
            rt_len_streamer += 2
            masks = []
            for g_, box_ in enumerate(self.rect_boxes):
                lx_, ly_, hx_, hy_ = box_
                mv_ = {self._noc(c) for c in self.rects[g_]}
                cm_ = sum(1 << (x - lx_) for x in {x for x, _ in mv_})
                rm_ = sum(1 << (y - ly_) for y in {y for _, y in mv_})
                assert hx_ - lx_ < 32 and hy_ - ly_ < 32 and len(mv_) == bin(cm_).count("1") * bin(rm_).count("1"), (
                    "PFS_UNICAST: the members = the box's member columns x rows", box_, len(mv_))  # fmt: skip
                masks.append((cm_, rm_))
        n_rt = max(rt_len_streamer, _W["RT_SXY"] + 2 * self.ns + (1 if wait_rt else 0))
        rt = {c: [sync, 0] for c in self.all_streamers}  # RT_ACTIVE 0: no stream of this phase on the core
        w_addr = self.weight.buffer_address()
        for si, c in enumerate(self.streamers):
            if self.multi:  # streamer si = (rectangle si // 2, lane si % 2): its column group's SPC streams, no chain
                g, lane = divmod(si, 2)
                t = g % self.n_split
                lx, ly, hx, hy = self.rect_boxes[g]
                sts = [t * self.nst_g + j for j in range(self.spc)]
                st, nst, nx, ny = 0, 1, 0, 0
            else:  # 7a: streamer si = (stream si // 2, lane si % 2), the lane's TOKEN chain across streams
                st, lane = divmod(si, 2)
                lx, ly, hx, hy = self.rect
                sts, nst = [st], self.nst
                nx, ny = svirt[si + 2] if st + 1 < self.nst else (0, 0)  # (st + 1, lane)
            sb_ = self.stream_base  # (joint: this phase's streams inside the shared layout)
            pairs = [v for s_ in sts for v in ((sb_ + s_) % 8, self.lay["m0"][sb_ + s_])]
            rt[c] = [sync, 1, st, lane, nst, w_addr, pairs[0], pairs[1], si & 3, lx, ly, hx, hy, nx, ny,
                     self.lane_noc[lane]] + pairs  # fmt: skip
            if uni:
                rt[c] += list(masks[g if self.multi else 0])
            assert len(rt[c]) == rt_len_streamer
        pbase = self.mprobe.buffer_address() if self.mprobe is not None else 0
        rows_per_rect = self.ndest * self.rt
        # the GROUP form (W7-A(k), program 2026-10-26 20:33:37Z / H3): row_off shifts this phase's OUTPUT rows into the
        # n c-row group buffer (a runtime arg only), in ROW TILES like row0 (14's B1, 21:00:25Z: k c / 32, never k c); never
        # with a DRAM-filled x, whose rows row0 also addresses
        row_off = int(getattr(self, "row_off", 0) or 0)
        assert not (row_off and x is not None and x_fill), ("row_off with a DRAM-filled x", row_off)
        self.ct_slot = int(slot)  # the sync slot this phase's cumulative words live in (the H5 / S1 geometry key)
        row_lo, row_hi = None, 0
        # The address publisher (RT_FIRST: PARK base + the two VALID semaphore addresses -> the streamers' sync words)
        # must be a CREDIT SOURCE of every streamer it publishes to: a streamer passes its READY wait only when all
        # its NDEST members credited, and the publisher writes the addresses before its first credit (credit model
        # I4). With several rectangles each one's first member publishes to ITS pair (RT_NS = 2); one publisher for
        # all pairs let the other rectangles' streamers pass READY before the addresses landed (d0 card2 18:26Z
        # watcher stop: a VALID set multicast to L1 0x0). `_check_publishers` asserts the coverage on the host.
        publishers, credit_sources = {}, {}
        for g, rect in enumerate(self.rects):
            h, t = divmod(g, self.n_split)
            own = [svirt[2 * g], svirt[2 * g + 1]] if self.multi else []
            others = [xy for i, xy in enumerate(svirt) if not (self.multi and i // 2 == g)]
            flat = [v for xy in own + others for v in xy]  # the rectangle's own pair first (credit targets 0 / 1)
            pair = [2 * g, 2 * g + 1] if self.multi else list(range(self.ns))
            for i, c in enumerate(rect):
                sig = (signals or {}).get(c, (0, 0, 0))
                row0 = h * rows_per_rect + i * self.rt + row_off
                row_lo, row_hi = row0 if row_lo is None else min(row_lo, row0), max(row_hi, row0 + self.rt)
                first = int(i == 0) if self.multi else int(g == 0 and i == 0)
                rt[c] = [sync, first, 2 if self.multi else self.ns, pbase, out.buffer_address(), row0,
                         x.buffer_address() if x is not None and x_fill else 0] + list(sig) \
                        + [t * (self.nt // self.n_split)] + flat  # fmt: skip
                assert len(rt[c]) == _W["RT_SXY"] + 2 * self.ns
                if wait_rt:
                    rt[c] = rt[c] + [int(wait_rt.get(c, 0))]
                if first:
                    publishers[c] = pair
                for si in pair:
                    credit_sources.setdefault(si, set()).add(c)
        # 14's B1 bounds assert: every member's output tile rows inside `out` (32 rows per tile row, any layout) -- for the
        # epilogues that STORE rows to `out` (0: bf16 tiles, 2: ROW_MAJOR rows); epi 1 packs into the next phase's CB and
        # passes a placeholder `out`
        if int(self.epi) != 1:
            out_rows = int(out.shape[-2]) if len(out.shape) >= 2 else int(out.shape[0])
            assert row_hi * 32 <= out_rows, (
                "PfStream output rows past the tensor",
                row_lo,
                row_hi,
                out_rows,
                row_off,
                self.epi,
            )
        self.out_tile_rows = (row_lo, row_hi)
        self._check_publishers(publishers, credit_sources)
        for c in rt:
            rt[c] = rt[c] + [0] * (n_rt - len(rt[c]))
        defs = ([("PFS_UNIT", "1")] if x is not None else []) + ([("PFS_PROBE", "1")] if self.probe else []) \
            + pfs_epoch_defines() + pfs_mc_pkt_defines() + ([("PFS_UNICAST", "1")] if uni else []) \
            + ([("PFS_DST_FULL", "1")] if self.rt * self.nb > 4 else []) + ([("PFS_WAIT_RT", "1")] if wait_rt else []) \
            + self.zskip_defines()  # fmt: skip
        assert not wait_rt or (x is not None and int(wait[0]) > 0), ("PFS_WAIT_RT: the x read + the semaphore id", wait)
        return dict(ct=ct, rt=rt, defines=defs,
                    wr_defines=[("PFS_SIG_SLICES", str(sig_slices))] if sig_slices > 1 else [],
                    sources=dict(streamer=dict(dm0=self.srcs["pfs_stream.cpp"]),
                                 member=dict(dm1=self.srcs["pfs_member_rd.cpp"], c=self.srcs["pfs_member_c.cpp"],
                                             dm0=self.srcs["pfs_member_wr.cpp"])),
                    cbs=dict(streamer=[(CB_RING, self.w_dtype, self.tb, RING * self.bt)],
                             member=[(cb_x, ttnn.bfloat16, 2048, self.rt * self.kt), (self.cb_park, self.w_dtype, self.tb, self.park * self.bt),
                                     (cb_out, self.out_dtype, self.out_tb, 2 * self.rt * self.nb)]),
                    sems=dict(member=[SEM_VALID0, SEM_VALID1]),
                    compute=dict(math_fidelity=self.fidelity, fp32_dest_acc_en=True, dst_full_sync_en=self.rt * self.nb > 4),
                    streamer_noc_mode="DM_DYNAMIC_NOC")  # fmt: skip

    def program(self, x) -> ttnn.ProgramDescriptor:
        """the unit's standalone program, built from parts() exactly as the PMK composite would: the phase's ONE CT
        list and each core's ONE RT list go to every kernel of the core"""
        pt = self.parts(self.y, x=x)
        ct, defs = pt["ct"], pt["defines"]

        def rt_of(cores):
            r = ttnn.RuntimeArgs()
            for c in cores:
                r[c[0]][c[1]] = pt["rt"][c]
            return r

        dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
        cc = ttnn.ComputeConfigDescriptor(**pt["compute"], math_approx_mode=False)
        kd = lambda src, cores, cfg: ttnn.KernelDescriptor(
            kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=_crs(cores),
            compile_time_args=ct, runtime_args=rt_of(cores), config=cfg, defines=list(defs))  # fmt: skip
        s, m = pt["sources"]["streamer"], pt["sources"]["member"]
        kernels = [kd(s["dm0"], self.all_streamers, dyn), kd(m["dm1"], self.members, ttnn.ReaderConfigDescriptor()),
                   kd(m["c"], self.members, cc), kd(m["dm0"], self.members, ttnn.WriterConfigDescriptor())]  # fmt: skip
        cbs = []
        for role, cores in (("streamer", self.all_streamers), ("member", self.members)):
            for idx, dt, page, pages in pt["cbs"][role]:
                cbs.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=_crs(cores), format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]))  # fmt: skip
        sems = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=_crs(self.members), initial_value=0)
            for i in pt["sems"]["member"]
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def __call__(self, x):
        assert tuple(x.shape)[-2:] == (self.mt * TILE, self.k) and x.dtype == ttnn.bfloat16, (x.shape, x.dtype)
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(int(self.mesh.shape[0])):
            for c in range(int(self.mesh.shape[1])):
                mc = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = self.program(x)
        ttnn.generic_op([x, self.weight, self.y, self.sync] + ([self.mprobe] if self.probe else []), mpd)
        return self.y

    def probes(self, cycles_per_us: float = 1350.0, slot: int = 0) -> list:
        """PFS_PROBE readback per chip (the last launch): streamer waits (TOKEN / DRAM read / READY / NoC flush) and
        span, member landing-to-landing time / free-slot waits and span, in us (mean, max) over the phase's cores"""
        if not self.probe:
            return []
        out = []
        o = slot * SYNC_WORDS
        rows = [self.all_streamers.index(c) for c in self.streamers]  # (ROUTE: the senders are not the first ns)
        st = [ttnn.to_torch(t).reshape(len(self.all_streamers), -1)[rows, ::WORD // 4].double() / cycles_per_us
              for t in ttnn.get_device_tensors(self.sync)]  # fmt: skip
        mb = [ttnn.to_torch(t).reshape(len(self.members), -1)[:, ::WORD // 4].double() / cycles_per_us
              for t in ttnn.get_device_tensors(self.mprobe)]  # fmt: skip
        f = lambda v: [round(float(v.mean()), 1), round(float(v.max()), 1)]  # noqa: E731
        for s_, m_ in zip(st, mb):
            out.append(dict(s_token=f(s_[:, o + _W["P_TOKEN"]]), s_read=f(s_[:, o + _W["P_READ"]]),
                            s_ready=f(s_[:, o + _W["P_READY"]]), s_flush=f(s_[:, o + _W["P_FLUSH"]]),
                            s_span=f(s_[:, o + _W["P_SPAN"]]), m_reserve=f(m_[:, _W["M_RESERVE"]]),
                            m_land=f(m_[:, _W["M_VALID"]]), m_span=f(m_[:, _W["M_SPAN"]]),
                            m_xfill=f(m_[:, _W["M_XFILL"]]), m_wstall=f(m_[:, _W["M_WSTALL"]]),
                            m_cspan=f(m_[:, _W["M_CSPAN"]]),
                            m_throttle_blocks=f(m_[:, _W["M_THROTTLE"]] * cycles_per_us)))  # fmt: skip
        return out

    def release(self):
        own = ((self.weight,) if self.own_weight else ()) + (self.y,)
        for t in own + ((self.sync,) if self.own_sync else ()) + ((self.mprobe,) if self.probe else ()):
            ttnn.deallocate(t)


_HDR = {
    "dm": ['#include "api/dataflow/dataflow_api.h"'],
    "c": ['#include "api/compute/common.h"', '#include "api/compute/compute_kernel_api.h"',
          '#include "api/compute/compute_kernel_hw_startup.h"', '#include "api/compute/matmul.h"',
          '#include "api/compute/pack.h"', '#include "api/compute/reconfig_data_format.h"',
          '#include "api/compute/tile_move_copy.h"'],
}  # fmt: skip


def composite_source(srcs: list, ct_bases: list, rt_bases: list, risc: str, startup: str = "") -> str:
    """the PMK composite pattern (tt/pmk_layer.composite_source): every phase body in its own namespace with its own
    PH_CT0 / ph_rt0, `kernel_main` renamed to `phase_main`, then one main calling the phases in order"""
    incs = ["#include <cstdint>"] + _HDR["c" if risc == "c" else "dm"]
    bodies = []
    for src in srcs:  # every #include is hoisted to global scope (a header first included inside phase i's namespace
        body = []  # would be invisible to phase i + 1: include guards)
        for line in src.splitlines():
            t = line.strip()
            if t.startswith("#include"):
                if t not in incs:
                    incs.append(t)
            elif t != "#pragma once":
                body.append(line)
        bodies.append("\n".join(body))
    out = ["// generated by tt/pf_stream.composite_source: phases in one kernel", "#define MK_PHASED 1"] + incs
    for i, (src, cb, rb) in enumerate(zip(bodies, ct_bases, rt_bases)):
        out += [f"namespace ph{i} {{", f"constexpr uint32_t PH_CT0 = {cb};", f"uint32_t ph_rt0 = {rb};",
                "#define kernel_main phase_main", src, "#undef kernel_main", f"}}  // namespace ph{i}"]  # fmt: skip
    out += ["void kernel_main() {", startup] + [f"    ph{i}::phase_main();" for i in range(len(srcs))] + ["}"]
    return "\n".join(out) + "\n"


def fused_program(mms: list, xs: list) -> ttnn.ProgramDescriptor:
    kernels, cbs, sems = fused_parts(mms, [dict(x=x) for x in xs])
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)


def fused_parts(mms: list, kws: list):
    """ONE program running PfStream phases mms[0], mms[1], .. in order (one launch; the PMK layer's P3 -> P7 -> ..
    pattern): the phases share the streamers (one sync tensor, slot i), the members' PARK 29 / OUT 42 / VALID 10 / 11
    (QBASE i = the blocks of the phases before), and the stationary in0 CB (each phase pushes and pops it). The members
    and streamers must be the same objects' lists; xs[i] = phase i's DRAM input (PFS_UNIT), outputs = mms[i].y."""
    m0 = mms[0]
    assert all(m.members == m0.members and m.all_streamers == m0.all_streamers and m.sync is m0.sync for m in mms)
    for m in mms:  # phases sharing a PARK CB share its block geometry; a phase with its own PARK starts at its slot 0
        same = [o for o in mms if o.cb_park == m.cb_park]
        assert all(
            o.bt == m.bt and o.tb == m.tb and o.park == m.park for o in same
        ), "a shared PARK needs one block size"
    assert all(m.nb == m0.nb and m.rt == m0.rt for m in mms), "shared OUT pages"
    qbase, parts, seen = 0, [], set()
    for i, (m, kw) in enumerate(zip(mms, kws)):  # kw: parts() kwargs (x, signals, wait, cb_x, cb_out)
        if m.cb_park not in seen:
            assert qbase % m.park == 0, "a fresh PARK CB starts at slot 0: QBASE must be a multiple of its slots"
            seen.add(m.cb_park)
        parts.append(m.parts(m.y, qbase=qbase, slot=i, **kw))
        qbase += m.blocks
    assert not any(p.get("wr_defines") for p in parts), "PFS_SIG_SLICES: the PMK composites only (pmk_assembly._pfs_parts)"
    ct_bases = [sum(len(p["ct"]) for p in parts[:i]) for i in range(len(parts))]
    ct = [v for p in parts for v in p["ct"]]
    cores = m0.all_streamers + m0.members
    rt_len = [len(next(iter(p["rt"].values()))) for p in parts]
    rt_bases = [sum(rt_len[:i]) for i in range(len(parts))]
    defs = sorted({d for p in parts for d in p["defines"]})
    startup = "    compute_kernel_hw_startup<SrcOrder::Reverse>(%d, %d, %d);" % (CB_X, CB_PARK, CB_OUT)

    def rt_of(cs):
        r = ttnn.RuntimeArgs()
        for c in cs:
            r[c[0]][c[1]] = [v for p in parts for v in p["rt"][c]]
        return r

    dyn = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                            noc=ttnn.NOC.RISCV_0_default, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC)  # fmt: skip
    cc = ttnn.ComputeConfigDescriptor(**parts[0]["compute"], math_approx_mode=False)
    kd = lambda role, risc, cs, cfg: ttnn.KernelDescriptor(
        kernel_source=composite_source([p["sources"][role][risc] for p in parts], ct_bases, rt_bases, risc,
                                       startup if risc == "c" else ""),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=_crs(cs), compile_time_args=ct,
        runtime_args=rt_of(cs), config=cfg, defines=defs)  # fmt: skip
    kernels = [kd("streamer", "dm0", m0.all_streamers, dyn), kd("member", "dm1", m0.members, ttnn.ReaderConfigDescriptor()),
               kd("member", "c", m0.members, cc), kd("member", "dm0", m0.members, ttnn.WriterConfigDescriptor())]  # fmt: skip
    cbs = []
    for role, cs in (("streamer", m0.all_streamers), ("member", m0.members)):
        seen = {}
        for p in parts:
            for idx, dt, page, pages in p["cbs"][role]:
                prev = seen.get(idx)
                assert prev is None or (prev[0], prev[1]) == (dt, page), f"CB {idx} differs between phases"
                seen[idx] = (dt, page, max(pages, prev[2] if prev else 0))
        for idx, (dt, page, pages) in seen.items():
            cbs.append(ttnn.CBDescriptor(total_size=page * pages, core_ranges=_crs(cs), format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)]))  # fmt: skip
    sems = [
        ttnn.SemaphoreDescriptor(id=i, core_ranges=_crs(m0.members), initial_value=0)
        for i in parts[0]["sems"]["member"]
    ]
    del cores
    return kernels, cbs, sems


def fused_call(mms: list, xs: list):
    mesh = mms[0].mesh
    mpd = ttnn.MeshProgramDescriptor()
    for r in range(int(mesh.shape[0])):
        for c in range(int(mesh.shape[1])):
            mc = ttnn.MeshCoordinate(r, c)
            mpd[ttnn.MeshCoordinateRange(mc, mc)] = fused_program(mms, xs)
    io = list(xs) + [t for m in mms for t in (m.weight, m.y)] + [mms[0].sync] + [m.mprobe for m in mms if m.probe]
    ttnn.generic_op(io, mpd)
    return [m.y for m in mms]
