# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 SP-row sparse attention v2: the device-side UNION + request exchange + gather in ONE program per chip
(kernels/spa/spa_um.cpp members, spa_ur.cpp / spa_us.cpp lane). Design check: doc pmk/spaphase/spa_union_model.py (the
bitmap + prefix-popcount construction reproduces spa_gather_unit's host lists exactly).

Per chip (its R SP rows, the selected token ids [1, 1, R, W] uint32 ROW_MAJOR with a SKIP tail):
  members (M cores, rows split evenly): a window bitmap of selected POOLS per member -> an OR tree -> the chip's union
          bitmap (member 0 -> the lane's slot of this chip); after the lane's prefix table: ids_new = each id's compact
          row 4 rank(pool) + t % 4 (rank = the pool's index in the union, pool order; the SKIP tail kept);
  lane    BRISC: the bitmap to every peer (one slot per chip position, same L1 address on every chip) -- also the
          handshake that the peer left its previous launch; then v1's owner push (spa_send.cpp) until the end marker.
          NCRISC: the prefix table (-> the members), then the walk of every destination's bitmap over the units this chip
          owns (bitmap word w = 128-token unit w, owner = w % N: the served GLM53_DSA_SHARD layout) -> staged pools.
The compact buffer holds `cap` pools (4 cap rows); a larger union is flagged in the sync word STATUS (status()).
Lines: one line of chips (the pair / 1 x n: rank = position, owner = w % n) or the 2x4 (FABRIC_1D lines per axis, rank
k = r 4 + c, owner = w % 8) in TWO stages: the row lane exchanges the 4 row bitmaps and runs stage B (this chip's own
pools AND the relayed ones -> the row-mates' compact buffers); a COLUMN lane (spa_ca.cpp + spa_us.cpp) sends the row's
OR to the column partner and runs stage A (the pools the partner's row needs from this chip -> the partner's relay
rows, `relay`: rcap pools). Every pool is pushed exactly once (by its owner when it is in the destination's row, else by
the owner's column partner in that row); the bitmap / OR exchanges are the cross-launch handshakes."""

from __future__ import annotations

import os

import torch

import ttnn

from models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream import _crs, fabric_connect

KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "spa")
CB_BM, CB_SCR, CB_ROWS, CB_PFX, CB_STAGE, CB_META = 0, 1, 2, 3, 4, 5
S_CHILD, S_PFX, S_LOCAL, S_ORMINE = 0, 1, 2, 3
DROP = 31  # the lane's sync word counting the members' out-of-window ids (spa_um.cpp; status())
CB_ROWOR, CB_RP, CB_SLOTS, CB_ORS = 6, 7, 9, 10
# A0 parallel walkers (GLM53_SPA_WALKERS=1): CB_MW (lane + members: the header / bitmap slice), CB_MST (members' staging),
# CB_MSLOT (the row lane: 2 slots of 4 KiB + 16 B meta per member); semaphores S_WALK / S_FREE0 / S_FREE1 (members),
# S_DONE (the lane); sync word W_READY (the slots' valid words cleared for this launch)
CB_MW, CB_MST, CB_MSLOT = 11, 12, 13
S_WALK, S_FREE0, S_DONE, W_READY = 4, 5, 7, 30  # (W_READY: past RDY + n - 1 <= 29 at n <= 9, below DROP)
S_RELAY = 8  # A0r (GLM53_SPA_WALKERS=2, the 2x4): the members' go for pass 1 (the partner's stage A landed)
UNIT = 128  # tokens per shard unit = one bitmap word (32 pools of 4 tokens)
# GLM53_SPA_WALK_BOUND=<cycles> (16, program [session 10] 2026-10-28 04:03:56Z; doc pmk/merge/walkbound): the walker
# path's waits bounded (kernels/spa/spa_bound.hpp); an expiry's record = sync words 16..23 of the row lane (the
# GLM53_SPA_STAMPS words: exclusive), the expiry count = word 29. GLM53_SPA_WALK_BOUND_MODE: report (default, the k4d8s-line
# served form: record + count, the wait resumes) | bail (the unit form)
BOUND_REC, BOUND_CNT, BOUND_MAGIC = 16, 29, 0x5BB00000
BOUND_WAITS = {1: "mw_walk", 2: "mw_free", 3: "mw_relay", 4: "ur_local", 5: "ur_peer_bm", 6: "ur_arr_a", 7: "ur_arr_a_pass1",
               8: "ur_a2b_sweep", 9: "us_sem", 10: "us_rdy", 11: "us_wready", 12: "us_sweep", 13: "us_arr_data",
               14: "us_arr2"}  # fmt: skip


def walk_bound() -> int:
    """GLM53_SPA_WALK_BOUND in cycles (0 / unset: off = no define, today's binaries)"""
    v = os.environ.get("GLM53_SPA_WALK_BOUND", "0").strip() or "0"
    assert v.isdigit(), f"GLM53_SPA_WALK_BOUND={v!r}: cycles"
    return int(v)


def walk_bound_mode() -> str:
    """GLM53_SPA_WALK_BOUND_MODE: "report" (default; the k4d8s-line served form, program [session 10] 2026-10-28
    04:28:04Z): an expiry records + counts and the wait RESUMES (results unchanged, a real hang still hangs with the record);
    "bail" (the unit form): WAYPOINT + ASSERT + record, then the core bails (the program completes, the SpaUnion state is
    undefined afterwards: units only)"""
    m = os.environ.get("GLM53_SPA_WALK_BOUND_MODE", "report").strip() or "report"
    assert m in ("report", "bail"), f"GLM53_SPA_WALK_BOUND_MODE={m!r}: report | bail"
    return m


def walk_bound_defaulted() -> bool:
    """the bound came from K4_DEFAULTS (not the environment): it then switches itself off where it cannot apply (walkers
    off, GLM53_SPA_STAMPS on, a row past the record words) instead of refusing the build"""
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.k4_defaults import K4_FROM_DEFAULT  # noqa: PLC0415
    except ImportError:
        return False
    return "GLM53_SPA_WALK_BOUND" in K4_FROM_DEFAULT


class SpaUnion:
    def __init__(self, mesh, compact, rows: int, window: int, width: int, *, cap: int | None = None, rcap: int | None = None,
                 lane=(0, 0), col_lane=(0, 1), members=None, link: int = 0, depth: int = 16, syncs: tuple | None = None,
                 relay=None):  # fmt: skip
        self.mesh, self.compact = mesh, compact
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        self.two = self.R > 1 and self.C > 1
        if self.two:
            assert self.R == 2, "the 2x4: a column of 2 (one relay partner)"
        self.axis = 1 if self.C > 1 else 0
        self.n = self.C if self.axis == 1 else self.R  # the (row) line
        self.nall = self.R * self.C
        assert window % UNIT == 0, (window, UNIT)
        self.rows, self.words, self.width = int(rows), window // UNIT, int(width)
        self.cap = int(cap) if cap is not None else int(compact.shape[-2]) // 4
        assert 4 * self.cap <= int(compact.shape[-2]), (self.cap, tuple(compact.shape))
        self.lane, self.col_lane, self.link, self.depth = tuple(lane), tuple(col_lane), int(link), int(depth)
        if members is None:  # an 8 x 8 rectangle right of the lanes
            members = [ttnn.CoreCoord(1 + x, y) for y in range(8) for x in range(8)]
        self.members = list(members)
        lanes = {self.lane} | ({self.col_lane} if self.two else set())
        assert all((int(c.x), int(c.y)) not in lanes for c in self.members)
        self.m = len(self.members)
        self.nr = -(-self.rows // self.m)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        # the ONLY persistent L1: 32 sync words per lane (GEN, ARR_BM[n], ARR_DATA[n], STATUS, ARR_DATA_A, RDY[n], ...,
        # DROP = word 31); the bitmap / OR slots are CBs of the program (spa_us.cpp: the RDY handshake); `syncs` = (row
        # lane, column lane) sync tensors shared by every key's SpaUnion (one TPPrefill: the launches run in the same order
        # on every chip, and every counter is generation-relative)
        assert 3 + 3 * self.n <= DROP
        self.own_syncs = syncs is None
        if syncs is None:
            syncs = SpaUnion.make_syncs(mesh, self.two, lane, col_lane)
        self.sync, self.sync_c = syncs
        self.relay = None
        self.rcap = 0
        self.own_relay = relay is None
        if self.two:
            # the relay rows: the pools this chip's row needs from the partner = the row OR in the partner's words.
            # `relay` (TPPrefill.spa_relay, GLM53_PF_ARENA): one store shared by every key, sized at the largest key's rcap
            # (the kernels bound it by this key's rcap; written before it is read in every launch)
            self.rcap = int(rcap) if rcap else min(self.words * 32 // self.nall, self.n * self.rows * 513)
            if relay is not None:
                assert int(relay.shape[-2]) >= 4 * self.rcap, ("the shared relay store", tuple(relay.shape), self.rcap)
                self.relay = relay
            else:
                self.relay = ttnn.from_torch(torch.zeros(1, 1, 4 * self.rcap, 512, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                             mesh_mapper=rep)  # fmt: skip
        self.calls = 0

    @staticmethod
    def make_syncs(mesh, two: bool, lane=(0, 0), col_lane=(0, 1)) -> tuple:
        rep_ = ttnn.ReplicateTensorToMesh(mesh)

        def l1(core):  # 32 words on a lane core (same address on every chip)
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                   ttnn.ShardSpec(_crs([tuple(core)]), [1, 32], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
            return ttnn.from_torch(torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=mesh, memory_config=mc, mesh_mapper=rep_)  # fmt: skip

        return (l1(lane), l1(col_lane) if two else None)

    def _noc(self, c):
        p = self.mesh.worker_core_from_logical_core(c)
        return int(p.x), int(p.y)

    def program(self, lat, ids, ids_new, r: int, c: int, row_off: int = 0):
        p = c if self.axis == 1 else r
        k = r * self.C + c
        kp = ((1 - r) * self.C + c) if self.two else k
        up, dn = self.n - 1 - p, p
        lane = ttnn.CoreCoord(*self.lane)
        lane_crs = _crs([self.lane])
        mem_crs = ttnn.CoreRangeSet([ttnn.CoreRange(m_, m_) for m_ in self.members])
        both = ttnn.CoreRangeSet([ttnn.CoreRange(m_, m_) for m_ in self.members] + [ttnn.CoreRange(lane, lane)])
        lanes_crs = _crs([self.lane, self.col_lane]) if self.two else lane_crs
        u32, bf = ttnn.uint32, ttnn.bfloat16
        cb = lambda idx, fmt, page, pages, ranges: ttnn.CBDescriptor(total_size=page * pages, core_ranges=ranges, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])  # fmt: skip
        wb = self.words * 4
        wv = os.environ.get("GLM53_SPA_WALKERS", "0")
        assert wv in ("0", "1", "2"), f"GLM53_SPA_WALKERS={wv!r}"
        walk = wv in ("1", "2")  # A0: pass 0 on the members (spa_mw.cpp)
        wrel = wv == "2" and self.two  # A0r: the relay pass 1 on the members too (the row lane streams nothing)
        # GLM53_SPA_WALK_HDRB=1 (spaa0b): the lane NCRISC writes the members' headers from per-member scratch (after the
        # member slots in CB_MSLOT), one barrier, then the S_WALK increments; GLM53_SPA_A2B=1 (spaa0b, A2b): the lane NCRISC
        # (idle once the members walk: the pair, or the 2x4 under A0r) is a second sender on fabric link 1 for the odd
        # members' slots in the directions it connects (<= 3 connections per core); default off = no define / CB / arg
        hdrb = walk and os.environ.get("GLM53_SPA_WALK_HDRB", "0") == "1"
        a2b = walk and os.environ.get("GLM53_SPA_A2B", "0") == "1"
        bound = walk_bound()
        if bound and walk_bound_defaulted() and (not walk or os.environ.get("GLM53_SPA_STAMPS", "0") == "1"
                                                  or 3 + 3 * self.n > BOUND_REC):  # fmt: skip
            bound = 0  # the K4 default where it cannot apply: off (an explicit GLM53_SPA_WALK_BOUND keeps the asserts below)
        bmode = walk_bound_mode() if bound else ""
        if bound:
            assert walk, "GLM53_SPA_WALK_BOUND bounds the walker path: GLM53_SPA_WALKERS=1 / 2"
            assert os.environ.get("GLM53_SPA_STAMPS", "0") != "1", "GLM53_SPA_WALK_BOUND and GLM53_SPA_STAMPS share sync words 16..23"
            assert 3 + 3 * self.n <= BOUND_REC, ("GLM53_SPA_WALK_BOUND: the record words 16..23 above RDY", self.n)
        if a2b:
            assert not self.two or wrel, "GLM53_SPA_A2B: the lane NCRISC must be free (the pair / a line, or A0r on the 2x4)"
            assert self.n <= 4, ("GLM53_SPA_A2B: the link-1 arrival words 25 + q (q < n) below W_READY 30", self.n)
            assert self.link in (0, 1), self.link
        # the lane NCRISC's staged stream: today (both lanes); A0 + the 2x4: the relay pass only (both lanes); A0 without
        # the 2x4: none -- the row lane's CB_STAGE / CB_META are not allocated (14b's 1M L1 budget, 21:40:45Z)
        stage_crs = lanes_crs if (not walk or self.two) else None
        if wrel:  # only the column lane's stage-A stream keeps CB_STAGE / CB_META
            stage_crs = _crs([self.col_lane])
        cbs = [cb(CB_BM, u32, wb, 1, mem_crs), cb(CB_SCR, u32, wb, 1, mem_crs), cb(CB_ROWS, u32, self.nr * self.width * 4, 1, mem_crs),
               cb(CB_PFX, u32, 2 * wb, 1, both)]
        if stage_crs is not None:
            cbs += [cb(CB_STAGE, bf, 4096, self.depth, stage_crs), cb(CB_META, u32, 16, self.depth, stage_crs)]
        cbs += [
               cb(CB_SLOTS, u32, self.n * wb, 1, ttnn.CoreRangeSet([ttnn.CoreRange(self.members[0], self.members[0]),
                                                                  ttnn.CoreRange(lane, lane)]))]  # fmt: skip  # member 0 + the row lane
        if self.two:
            cbs += [cb(CB_ROWOR, u32, wb, 1, lane_crs), cb(CB_RP, u32, wb, 1, lane_crs), cb(CB_ORS, u32, 2 * wb, 1, lanes_crs)]
        rmax = -(-self.words // self.m)  # A0: a member's longest word slice
        if walk:
            assert self.n <= 8 and 2 * self.m <= 128, (self.n, self.m)
            rp = (rmax + 6 + 3) // 4 * 4  # spa_mw.cpp RP: a slice region with its 16-B alignment lead / tail words
            cbs += [cb(CB_MW, u32, (12 + (10 if wrel else 8) * rp) * 4, 1, both),
                    cb(CB_MST, bf, 3 * 4096 + 32 + (64 if bound else 0), 1, mem_crs),  # (bound: the member's record stash)
                    cb(CB_MSLOT, bf, 2 * self.m * (4096 + 16) + (48 * self.m if hdrb else 0) + (128 if bound else 0), 1,
                       lane_crs)]  # fmt: skip  # (bound: the lane BRISC's + NCRISC's record stashes)
        sems = [ttnn.SemaphoreDescriptor(id=S_CHILD, core_ranges=mem_crs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=S_PFX, core_ranges=mem_crs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=S_LOCAL, core_ranges=lane_crs, initial_value=0)]  # fmt: skip
        if self.two:
            sems.append(ttnn.SemaphoreDescriptor(id=S_ORMINE, core_ranges=_crs([self.col_lane]), initial_value=0))
        if walk:
            sems += [ttnn.SemaphoreDescriptor(id=S_WALK, core_ranges=mem_crs, initial_value=0),
                     ttnn.SemaphoreDescriptor(id=S_FREE0, core_ranges=mem_crs, initial_value=1),
                     ttnn.SemaphoreDescriptor(id=S_FREE0 + 1, core_ranges=mem_crs, initial_value=1),
                     ttnn.SemaphoreDescriptor(id=S_DONE, core_ranges=lane_crs, initial_value=0)]  # fmt: skip
            if wrel:
                sems.append(ttnn.SemaphoreDescriptor(id=S_RELAY, core_ranges=mem_crs, initial_value=0))
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        lx, ly = self._noc(lane)
        cx, cy = self._noc(ttnn.CoreCoord(*self.col_lane)) if self.two else (0, 0)
        mxy = [v for m_ in self.members for v in self._noc(m_)]
        sync = int(self.sync.buffer_address())
        rel = self.relay if self.two else lat
        rt_m, rt_r, rt_s = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, m_ in enumerate(self.members):
            r0 = min(i * self.nr, self.rows)
            nrw = max(0, min(self.nr, self.rows - r0))
            rt_m[int(m_.x)][int(m_.y)] = [int(ids.buffer_address()), int(ids_new.buffer_address()), row_off + r0, nrw, i, self.m, lx, ly,
                                          p, self.cap, sync + 4 * DROP] + mxy  # fmt: skip
        rt_r[lane.x][lane.y] = [sync, 0, int(lat.buffer_address()), p, self.n, self.cap, self.m, self.nall, k, kp,
                                int(rel.buffer_address()), cx, cy, 0, self.rcap] + mxy  # fmt: skip
        rt_s[lane.x][lane.y] = [sync, p, p, S_LOCAL, p, self.n, up, dn, int(self.compact.buffer_address()), 0, 0, 0] + (
            [self.m] + mxy if walk else [])  # (A0: the members, before the fabric connections)
        if a2b:  # the BRISC: which slots the NCRISC takes (its link-1 directions) + the peers whose NCRISC reaches this chip
            ncu, ncd = self._a2b_dirs(p)
            mask = sum(1 << q for q in range(self.n)
                       if q != p and ((p > q and self._a2b_dirs(q)[0]) or (p < q and self._a2b_dirs(q)[1])))  # fmt: skip
            rt_s[lane.x][lane.y].extend([ncu, ncd, mask])
            rt_r[lane.x][lane.y].extend([int(self.compact.buffer_address()), ncu, ncd])  # (after the members' NoC xy)
        # GLM53_SPA_INTERLEAVE=1 (16's A2'(a), 2026-10-27): the walker interleaves the destinations word by word (both
        # direction channels send concurrently); every pool's (slot, destination) is unchanged: the compact store is
        # bitwise the default's. Off ("0") = no define: the default walk order (the loop body shared as emit())
        il = [("SPA_INTERLEAVE", "1")] if os.environ.get("GLM53_SPA_INTERLEAVE", "0") == "1" else []
        stp = [("SPA_STAMPS", "1")] if os.environ.get("GLM53_SPA_STAMPS", "0") == "1" else []  # the slope unit's stamps
        # A0's defines: the ROW lane's two kernels + the member walker (the column lane's spa_us.cpp keeps none)
        wd = [("SPA_CB_MW", str(CB_MW)), ("SPA_CB_MST", str(CB_MST)), ("SPA_CB_MSLOT", str(CB_MSLOT)), ("SPA_S_WALK", str(S_WALK)),
              ("SPA_S_FREE0", str(S_FREE0)), ("SPA_S_DONE", str(S_DONE)), ("SPA_W_READY", str(W_READY))] if walk else []
        if wrel:
            wd = wd + [("SPA_WALK_RELAY", "1"), ("SPA_S_RELAY", str(S_RELAY))]
        if bound:  # the member walker + the row lane's two kernels (wrow takes wd)
            wd = wd + [("SPA_WALK_BOUND", f"{bound}ull")] + ([("SPA_WALK_BOUND_REPORT", "1")] if bmode == "report" else [])
        wrow = ([("SPA_WALKERS", "1")] + wd + ([("SPA_WALK_STAGE", "1")] if self.two and not wrel else [])) if walk else []
        wrow += ([("SPA_WALK_HDRB", "1")] if hdrb else []) + ([("SPA_A2B", "1")] if a2b else [])  # (the row lane's two kernels)
        kd = lambda src, ct, rt, crs, cfg, extra=(): ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src),  # noqa: E731
                                                                           source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                                                           core_ranges=crs, compile_time_args=[int(v) for v in ct],
                                                                           runtime_args=rt, config=cfg,
                                                                           defines=(list(il) if src == "spa_ur.cpp" else [])
                                                                           + (list(stp) if src in ("spa_ur.cpp", "spa_us.cpp") else [])
                                                                           + list(extra))  # fmt: skip
        kernels = [
            kd("spa_us.cpp", [CB_STAGE, CB_META, self.words, CB_SLOTS] + acc(self.compact), rt_s, lane_crs, ttnn.WriterConfigDescriptor(),
               wrow),
            kd("spa_ur.cpp", [CB_STAGE, CB_META, CB_PFX, self.words, int(self.two), CB_ROWOR, CB_RP, CB_SLOTS, CB_ORS] + acc(lat)
               + acc(rel) + (acc(self.compact) if a2b else []), rt_r, lane_crs, ttnn.ReaderConfigDescriptor(), wrow),
            kd("spa_um.cpp", [CB_BM, CB_SCR, CB_PFX, CB_ROWS, self.words, self.width, CB_SLOTS] + acc(ids) + acc(ids_new), rt_m,
               mem_crs, ttnn.ReaderConfigDescriptor()),
        ]  # fmt: skip
        col = ttnn.CoreCoord(*self.col_lane)
        if self.two:  # the column lane: stage A (its OR to the partner, then the partner row's pools into its relay rows)
            rt_cs, rt_ca = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            sync_c = int(self.sync_c.buffer_address())
            rt_cs[col.x][col.y] = [sync_c, 0, 1, S_ORMINE, r, 2, 1 - r, r, int(self.relay.buffer_address()), lx, ly,
                                   sync + (2 + 2 * self.n) * 4]  # fmt: skip
            rt_ca[col.x][col.y] = [sync_c, 0, int(lat.buffer_address()), r, self.nall, k, self.rcap]
            col_crs = _crs([self.col_lane])
            kernels += [kd("spa_us.cpp", [CB_STAGE, CB_META, self.words, CB_ORS] + acc(self.relay), rt_cs, col_crs, ttnn.WriterConfigDescriptor()),
                        kd("spa_ca.cpp", [CB_STAGE, CB_META, self.words, CB_ORS] + acc(lat), rt_ca, col_crs, ttnn.ReaderConfigDescriptor())]  # fmt: skip
        if walk:  # A0: the member walker on the members' BRISC (appended last: kernels[0] / [3] keep their fabric args)
            rt_w = ttnn.RuntimeArgs()
            for i, m_ in enumerate(self.members):
                rt_w[int(m_.x)][int(m_.y)] = [int(lat.buffer_address()), int(self.compact.buffer_address()), lx, ly, i, self.m,
                                              self.n, p, self.nall, k, self.cap, int(rel.buffer_address()), kp, self.rcap] + (
                                                  [sync] if bound else [])  # fmt: skip  # (bound: the record's sync words)
            kernels.append(kd("spa_mw.cpp", [self.words, rmax] + acc(lat) + acc(self.compact) + acc(rel), rt_w, mem_crs,
                              ttnn.WriterConfigDescriptor(), wd))  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        me = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        step = (lambda d: (r, c + d)) if self.axis == 1 else (lambda d: (r + d, c))  # noqa: E731
        for d, rng in ((1, up), (-1, dn)):  # up first, then down (spa_us.cpp's order)
            if rng:
                peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                prog.kernels[0].runtime_args[lane.x][lane.y].extend(fabric_connect(me, peer, self.link, prog, lane))
        if a2b:  # the lane NCRISC's link-1 connections (up first, then down: spa_ur.cpp's A2b order)
            ncu, ncd = self._a2b_dirs(p)
            for d, rng, on in ((1, up, ncu), (-1, dn, ncd)):
                if rng and on:
                    peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                    prog.kernels[1].runtime_args[lane.x][lane.y].extend(fabric_connect(me, peer, 1 - self.link, prog, lane))
        if self.two:  # the column connection (up = to row 1, down = to row 0)
            peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(1 - r, c))
            prog.kernels[3].runtime_args[col.x][col.y].extend(fabric_connect(me, peer, self.link, prog, col))
        return prog

    def _a2b_dirs(self, q: int) -> tuple:
        """A2b: (up, down) = the link-1 directions the lane NCRISC of line position q connects. A core holds <= 3 fabric
        connections per program (fabric_stream.FAB_SEM_BASE): with both link-0 directions on the BRISC (a middle chip) it
        takes ONE, the one with more destinations (ties: up); else the one direction q has"""
        nu, nd = self.n - 1 - q, q
        if nu and nd:
            return (1, 0) if nu >= nd else (0, 1)
        return (int(nu > 0), int(nd > 0))

    def tensors(self):
        return [t_ for t_ in (self.sync, self.sync_c, self.relay) if t_ is not None]

    def owned(self):
        """the tensors this instance allocated (the shared syncs are the owner's)"""
        return [t_ for t_ in ((self.sync, self.sync_c) if self.own_syncs else ()) + ((self.relay,) if self.own_relay else ())
                if t_ is not None]

    def __call__(self, lat, ids, ids_new, row_off: int = 0):
        """lat: the local row-major latent store [1, 1, T_local, 512] bf16 (sharded); ids: this chip's SP rows' selected
        token ids [1, 1, R', W] uint32 ROW_MAJOR (sharded, SKIP tail) -> the compact buffer + ids_new (same shape as ids:
        the compact rows). row_off: this launch covers rows [row_off, row_off + rows) of ids / ids_new (a row block of
        GLM53_PMK_SPA's blocked form: the compact buffer then holds that block's union only)"""
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self.program(lat, ids, ids_new, r, c, row_off)
        ttnn.generic_op([lat, ids, ids_new, self.compact] + self.tensors(), mpd)
        self.calls += 1
        return self.compact

    def stamps(self) -> list:
        """GLM53_SPA_STAMPS=1: per chip the row lane's phase times (us, the LAST launch): BRISC bitmap send, pool sends,
        arrival wait; NCRISC wait for the own bitmap, prefix + peers' bitmaps, the walk (AICLK at 1,350 MHz)"""
        out = []
        for t_ in ttnn.get_device_tensors(self.sync):
            w = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(t_).reshape(-1)[:32].tolist()]
            d = lambda a, b: round(((w[b] - w[a]) & 0xFFFFFFFF) / 1350.0, 1)  # noqa: E731
            out.append(dict(us_bitmaps=d(16, 17), us_pools=d(17, 18), us_arrivals=d(18, 19), us_total=d(16, 19),
                            ur_own_bitmap=d(20, 21), ur_prefix_peers=d(21, 22), ur_walk=d(22, 23)))  # fmt: skip
        return out

    def bound_records(self) -> list:
        """GLM53_SPA_WALK_BOUND: per chip the row lane's expiry record (sync words 16..23, the FIRST expiry since the sync
        words were made: with shared syncs, over every key) and the expiry count (word 29); host read, outside traces.
        -> [dict(count, fired, wait, risc, noc_xy, seen, want, ctx, gen, cycles, checksum_ok)]"""
        out = []
        for t_ in ttnn.get_device_tensors(self.sync):
            w = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(t_).reshape(-1)[:32].tolist()]
            r = w[BOUND_REC:BOUND_REC + 8]
            fired = (r[0] & 0xFFFF0000) == BOUND_MAGIC
            x = 0
            for v in r[:7]:
                x ^= v
            out.append(dict(count=w[BOUND_CNT], fired=fired)
                       if not fired else
                       dict(count=w[BOUND_CNT], fired=True, wait=BOUND_WAITS.get(r[0] & 0xFF, r[0] & 0xFF), fail=bool(r[0] & 0x8000),
                            risc="ncrisc" if (r[1] >> 16) & 1 else "brisc", noc_xy=[(r[1] >> 8) & 0xFF, r[1] & 0xFF],
                            seen=r[2], want=r[3], ctx=r[4], gen=r[5], cycles=r[6], checksum_ok=x == r[7]))  # fmt: skip
        return out

    def status(self) -> list:
        """per chip: (compact overflow: 0 or the union size, relay overflow: 0 or the relay set size, dropped: the
        out-of-window ids the members dropped since the sync words were made: with shared syncs (TPPrefill.spa_syncs) the
        last launch's overflow words and the DROP count over EVERY key -- read per-forward deltas); host read, outside traces"""
        n = self.n
        words = [ttnn.to_torch(t_).reshape(-1) for t_ in ttnn.get_device_tensors(self.sync)]
        rows = [int(w[1 + 2 * n]) for w in words]
        rel = [int(ttnn.to_torch(t_).reshape(-1)[5]) for t_ in ttnn.get_device_tensors(self.sync_c)] if self.two else [0] * len(rows)
        return list(zip(rows, rel, [int(w[DROP]) for w in words]))
