# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The sequence-sharded DSA + split-K attention half on the TP pair (1x2 / 2x1) and the TP8 2x4 mesh (tt/dsa_shard.py,
kernels/mla/dsa_shard.hpp; doc/ttnn/14-production-tp8-megakernel/dsa/shard_tp8_design.md): TPMla with
MkMla(dsa_shard=True) + in-program fabric collectives per launch (tt/fabric_stream.py, Stage 7a's kernels).
The pair (TP2, one round each):
  AG  all-gather of x_ag [2 T chunks: per token this chip's q_lat tile row | its candidate records] -> y_ag on every
      chip (the peer's q for pass 0; the program's global merge); the lane waits for T (chunk cores + the
      program) + 1 (the latent core's q_lat copy) producer increments, its waiter signals the index core (SEM_AGC);
  RS  the pair's reduce-scatter exchange of x_rs [2 blocks x T x 3 chunks: the partial (O, stats) for the peer's heads]
      -> y_rs (the peer's partial of this chip's heads); the lane waits for T root increments, its waiter signals the
      owners (SEM_RSC).
TP8 on the 2x4 mesh (rank = 4 row + col; 4 stream instances, the design's link table):
  AG1 (axis 0, link 0) x_ag [T chunks: the compact q piece + records] -> y_ag block ag_blk = 2 col + row, local copy on;
      its waiter signals AG2's lane;
  AG2 (axis 1, link 0) the column's 2-block bundle of y_ag multicast along the row (x = y = y_ag); its waiter signals
      the index core (SEM_AGC): y_ag holds all 8 blocks;
  RS1 (axis 0, link 1) the pass-0 partial of the other row's tile -> the column peer (the pair exchange per column);
      its waiter signals the owners (SEM_RSC, 1st);
  RS2 (axis 1, link 1) FabricStreamScatter: x_rs2 block q (the 8-row slice of row peer q of the column-combined
      partial) -> peer q's y_rs2 block p; its waiter signals the owners (SEM_RSC, 2nd).
  The o_proj AR lanes run FAR_LATE_OPEN: lane 0 releases AG1 + AG2 (link 0), lane 1 RS1 + RS2 (link 1) at launch start.
Token lanes (rev 60, W: t > 4 = native B=32 and widths 8 / 16; shard_tp8_design.md section 7): MkMla's att_lanes(t) = L
lanes, one index core per lane (reserved here AFTER the stream lanes: the t <= 4 placement is unchanged): the AG waits
for t (C / L + 1) + 1 producer increments and signals every lane index core (count-prefixed), the RS signals every lane
root; streams per t of stream_ts (ts + stream_ts + the stack's native widths GLM53_STACK_NATIVE_WIDTHS when 32 is in ts:
StackBackend's width calls t = 8 / 16 run on the 32-row FabricPhase, tp_mla.fabric, but on their own streams).
One MeshProgramDescriptor per call: the captured MkMla program + Stage 14's o_proj all-reduce lanes (TPMla) + the stream
lanes, with the chip's rank / pair-peer position patched into the shard runtime args per mesh coordinate; the link /
target / packet-header audits (tt/shard_plan.py) run on every build and raise. The sharded state (per-chip latent
pool, local page table, local pooled-key store) is the caller's (tests/s14/jobs/dsa_shard.py: tt/dsa_shard.py
ShardMap). GLM53_DSA_SHARD=1 selects this class in a stack (tt/stack_model.py: pending)."""

from __future__ import annotations

import os

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
from models.autoports.zai_org_glm_5_3_flash.tt import phase_merge as PM
from models.autoports.zai_org_glm_5_3_flash.tt.mk_dense import _cores_of
from models.autoports.zai_org_glm_5_3_flash.tt import shard_plan as SP
from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM
from models.autoports.zai_org_glm_5_3_flash.tt.shard_plan import audit_links  # noqa: F401  (re-export)
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import _Capture, _rt_dict, _src_is
from models.autoports.zai_org_glm_5_3_flash.tt.tp_mla import _with_rt

DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = DS.KV_LORA  # the pair's AG tensors / q_asm width (512 bf16 = 16 pages per 32-row chunk)
W_RS = 32 * MK.SHARD_RS_PPC  # rev 59: x_rs / y_rs width (384 bf16 = 12 pages per chunk: 3 chunks = the 36-page message)
W8 = 32 * MK.SHARD_AG8_PPC  # TP8: the compact AG / RS2 tensors' width (320 bf16 = 10 pages per chunk)
assert MK.SHARD_RS2_PPC == MK.SHARD_AG8_PPC


LINK_CB_STAGE0 = 20  # GLM53_DSA_SHARD_LINK_LANES: the streams' stage CBs from here up (one index per stage size)
# lanecore v3 (program ruling 2026-10-27T15:43:40Z (1)): stage batches per stage CB on the link lane cores. 1 =
# single-buffered staging: (8,2) RS1 24,576 B -> CB end 1,234,048, (8,0) AG1 / AG2 / RS2 20,480 B -> 1,229,952, under the
# served final floor 1,241,344 (L1_FLOOR_SERVED) at every t; v2's 2 slots ended 1,258,624 / 1,250,432, over it
def link_stage_slots(fab_rows: int = 32) -> int:  # (read at each build: a queued job's env applies after this module's import)
    """stage batches per stage CB on the link lane cores: GLM53_DSA_SHARD_STAGE_SLOTS if set, else 2 under a packed
    (b <= 16) all-reduce phase (its AR lane buffers are 64 KiB or less, so double-buffered staging fits) and 1 under the
    32-row phase (lanecore v3: 7 x 128 KiB of AR buffers leave room for one slot below the floor)"""
    v = os.environ.get("GLM53_DSA_SHARD_STAGE_SLOTS", "")
    return int(v) if v else (2 if int(fab_rows) <= 16 else 1)
L1_SIZE = 1_572_864  # Blackhole worker L1 (1.5 MiB): the CB region starts at L1_SIZE - the worker L1 bank size
CB_BASE_SERVED = 291_968  # the served worker_l1 1,280,896 -> CB base (program 2026-10-27T13:50:59Z)
# The L1 floor is NOT a constant: L1 buffers grow down from L1_ALLOC_TOP and every sparse_dsa decode shard program adds
# ~4,224 B of persistent L1 per layer, and every enqueue re-validates a program's static CBs against the L1 buffers live
# then (program 2026-10-27T15:39:32Z (3); 9's k4d4 d0 mock died at (8, 2): CB end 1,258,624 vs a buffer at 1,256,576)
L1_ALLOC_TOP = 1_570_048  # the L1 allocator's top: that clash's buffer 1,256,576 + its alloc_per_bank 313,472
L1_FLOOR_SERVED = 1_241_344  # the served build's FINAL floor = L1_ALLOC_TOP - its max alloc_per_bank 328,704 (the k4d3
# d0 mock's MK_CB_REPORT headers, scripts/cb_report_ends.py); was 1,270,912, the PMK floor, which no decode program meets
_CB_ENDS: dict = {}  # id(mesh) -> {(id(TPMlaShard), t): lane-core CB ends}: re-checked at every later build (_cb_model_check)


def live_l1_floor(mesh) -> int | None:
    """L1_ALLOC_TOP - the allocator's L1 bytes allocated per bank now (lock-step: chip 0 = every chip); None if unreadable"""
    try:
        return L1_ALLOC_TOP - int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank)
    except Exception:  # noqa: BLE001
        return None


def _merge_stage_cbs(scbs: list) -> list:
    """GLM53_DSA_SHARD_LINK_LANES: the streams' stage CB descriptors with ONE descriptor per (index, size) over the union of
    their cores (the phases of a lane core run one after another and share one stage CB: same batch size, capacity a
    multiple of it)"""
    stage, order = {}, []
    for cb in scbs:
        f0 = cb.format_descriptors[0]
        k_ = (int(f0.buffer_index), int(cb.total_size))
        if k_ not in stage:
            stage[k_] = set()
            order.append(k_)
        stage[k_] |= {(int(x), int(y)) for x, y in _cores_of(cb.core_ranges)}
    assert len({k[0] for k in order}) == len(order), ("one stage CB index with two sizes", order)
    return [ttnn.CBDescriptor(total_size=sz, core_ranges=_crs_xy(stage[(bix, sz)]), format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=bix, data_format=ttnn.bfloat16, page_size=FS.TB)])
        for bix, sz in order]  # fmt: skip  (every stage CB: bf16 pages of FS.TB, fabric_stream.py parts())


def _crs_xy(cores) -> "ttnn.CoreRangeSet":
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])


def _flag(cb, name: str) -> bool:
    v = getattr(cb, name, False)
    return bool(v() if callable(v) else v)


def cb_region_bytes(prog, cores) -> dict:
    """{core: the bytes of the CB REGION on it}: every CB descriptor covering the core, except the globally allocated ones
    (has_buffer / has_global_circular_buffer: they live in an L1 tensor, not in the CB region)"""
    per = {tuple(c): 0 for c in cores}
    for cb in prog.cbs:
        if _flag(cb, "has_buffer") or _flag(cb, "has_global_circular_buffer"):
            continue
        for x, y in _cores_of(cb.core_ranges):
            if (int(x), int(y)) in per:
                per[(int(x), int(y))] += int(cb.total_size)
    return per


class TPMlaShard(TPM.TPMla):
    """TPMla on the pair / the 2x4 mesh with the split-K DSA phase: same call / outputs (o_lat: this chip's heads,
    merged over every chip's selected rows; s1 identical on every chip after the o_proj all-reduce)."""

    def __init__(self, mesh, mhc_args: dict, attn, *, ts=(1, 2, 3, 4), lanes_extra=None, stream_ts=None,
                 stream_share=None, **mk_kwargs):  # fmt: skip
        """stream_ts (rev 60): extra row counts with their own streams (no FabricPhase of their own: tp_mla.fabric
        serves t > 4 from the 32-row phase); the stack's native widths are added when 32 is in ts"""
        assert mk_kwargs.get("dsa") is not None, "TPMlaShard: dsa=DsaIndexer"
        super().__init__(mesh, mhc_args, attn, ts=ts, dsa_shard=True, **mk_kwargs)
        self.geo = SP.Geometry(self.R, self.C)  # raises unless 1x2 / 2x1 / 2x4
        self.n, self.L = self.geo.n, self.geo.L
        wts = set(ts) | set(int(v) for v in (stream_ts or ()))
        if 32 in ts:  # StackBackend's native widths (t = w calls on this object, the 32-row FabricPhase)
            nspec = os.environ.get("GLM53_STACK_NATIVE_WIDTHS", "none")
            wts |= {int(v) for v in nspec.split(",") if v.strip() and 4 < int(v) < 32} if nspec not in ("", "none") else set()
        # GLM53_DSA_SHARD_STREAM_TS ("1,8,16,32"; default all): build the streams of these row counts only. Each t holds
        # 4 x 1.5 KB of L1 sync words per sparse_dsa layer = 67.6 KB per bank over the 11 layers; the served native-width
        # stack calls t in {1 (narrow), 8, 16 (native), 32}, never t = 2..4 (the 4-row chunk path), and the n29h7 env's
        # prefill MoE program needs that L1 (2026-10-23 07:29Z: its static CBs end 114.5 KB above the lowest L1 buffer;
        # dsa/shard/README.md section 6r). A call at a t without streams raises (__call__: 'no streams for t').
        only = os.environ.get("GLM53_DSA_SHARD_STREAM_TS", "")
        if only.strip():
            keep = {int(v) for v in only.split(",") if v.strip()}
            wts = {t for t in wts if t in keep}
            if not wts:
                raise ValueError(f"GLM53_DSA_SHARD_STREAM_TS={only!r} leaves no stream row count of {sorted(set(ts))}")
        self.stream_ts = tuple(sorted(wts))
        assert max(self.stream_ts) <= MK.SHARD_T_MAX, self.stream_ts
        assert self.mk.groups == 1, ("one head group per chip", self.mk.groups)
        self.sm = DS.ShardMap(self.n)
        grid = mesh.compute_with_storage_grid_size()
        used = set(self.mk.used) | set(tuple(v) for v in self.lanes)
        free = [(x, y) for y in range(int(grid.y)) for x in range(int(grid.x)) if (x, y) not in used]
        self.names = ["ag1", "rs1"] + (["ag2", "rs2"] if self.L > 1 else [])
        # GLM53_DSA_SHARD_LINK_LANES=1 (Stage 14, 2026-10-27; dsa/shard_w16/levers_design.md section 6): the PER-LINK LANE
        # CORES -- every stream runs on the AR lane core of its link (AG1 / AG2 on lane 0 = link 0, RS1 / RS2 on lane 1
        # = link 1) as phases of one composite per RISC (tt/phase_merge.py: BRISC fs_send x n -> lane_send, NCRISC
        # fs_read x n -> lane_recv), so the link's holders run in sequence on one worker and the 2 (pair) / 4 (2x4)
        # stream cores are freed for the token lanes (reserve_lane_cores below: 8 lanes on the 11 x 10 grid)
        self.link_lanes = os.environ.get("GLM53_DSA_SHARD_LINK_LANES", "1") == "1"  # default ON (k4d4, program 2026-10-27T14:28:47Z: lanecore v2 on the served pin, Galaxy dsaglx7 t16 1,763.1 -> 1,233.9 us = -529 per DSA layer, A == B bitwise; DSA_SHARD_CB_MODEL lane-core ends 1,250,432 / 1,258,624 < 1,270,912); =0 restores the 4-lane placement
        if self.link_lanes:
            assert not lanes_extra, "GLM53_DSA_SHARD_LINK_LANES: the streams take the AR lane cores (no lanes_extra)"
            # lanecore v2 (program ruling 2026-10-27T13:56:44Z): RS2 joins link 0 (after AG1 / AG2: link 0 is idle in
            # the RS rounds), so every phase of a lane core has ONE stage batch size (link 0: W8 streams, NB 10; link 1:
            # RS1 alone, NB 12) and one 2-slot stage CB per core fits the served L1 floor (the CB model below)
            link_of = {"ag1": 0, "ag2": 0, "rs1": 1, "rs2": 0}
            ex = []
            self.lane_core = {n: tuple(self.lanes[link_of[n]]) for n in self.names}
        else:
            ex = list(lanes_extra or free[: len(self.names)])
            assert len(ex) == len(self.names), f"{len(self.names)} more free cores for the stream lanes ({ex})"
            self.lane_core = dict(zip(self.names, ex))
        self.geo.link_lanes = self.link_lanes  # (shard_plan: RS2 on link 0 in the link-lane form)
        self.ag_lane, self.rs_lane = self.lane_core["ag1"], self.lane_core["rs1"]
        # rev 60: the token lanes' index cores (t > 4), after the stream lanes (t <= 4 builds: no lane cores, the same
        # placement as rev 59); every lane holds the 2 LINE merge helpers
        self.ix_lane_cores = []
        if max(self.stream_ts) > MK.DSA_T_MAX:
            rest = [v for v in free if tuple(v) not in {tuple(e) for e in ex}]
            self.ix_lane_cores = self.mk.reserve_lane_cores(rest, 2 * self.L)
        for t in self.stream_ts:
            self.mk.shard_lanes_check(t, self.L)
        self.link_audit, self.target_audit, self.header_audit, self.census, self.censuses = {}, {}, {}, {}, {}
        # GLM53_DSA_SHARD_SHARE_SYNC=1 (stream_share = the first sparse_dsa layer's TPMlaShard): every layer's stream of a
        # row count t uses that layer's sync tensor -- the counters are monotonic across launches (fs_common.hpp: launch g
        # waits >= g x per-launch count), every layer's stream at t has the same lane cores and per-launch counts, and the
        # layers launch one after another in the same order on every chip: one stream launched 11 x per step. Frees
        # 10 x 4 x 1.5 KB of L1 per t (the n29h7 1M env: 246 KB per bank at t = 1 / 8 / 16 / 32).
        self._sync_from = None
        if stream_share is not None and os.environ.get("GLM53_DSA_SHARD_SHARE_SYNC", "0") == "1":
            assert stream_share.lane_core == self.lane_core, (stream_share.lane_core, self.lane_core)
            self._sync_from = stream_share
        self.streams = {t: self._new_streams(t) for t in self.stream_ts}

    def _pos(self, r: int, c: int) -> int:
        """this chip's rank (the head shard / owner index: row-major mesh order)"""
        return self.geo.rank(r, c)

    @property
    def valid_noc(self) -> set:
        """the NoC (x, y) of every worker core of the grid (a semaphore / atomic target must be one of them)"""
        grid = self.mesh.compute_with_storage_grid_size()
        return {
            (int(w.x), int(w.y))
            for w in (
                self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                for y in range(int(grid.y))
                for x in range(int(grid.x))
            )
        }

    def _noc(self, core) -> tuple:
        w = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        return (int(w.x), int(w.y))

    def _new_streams(self, t: int) -> dict:
        mesh, geo = self.mesh, self.geo
        z = lambda rows, w: ttnn.zeros(
            (1, 1, rows, w), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )  # noqa: E731  # fmt: skip  (the same shape / address on every chip)
        nl = self.mk.att_lanes(t)  # rev 60: token lanes (1 at t <= 4)
        cl = self.mk.chunks // nl  # chunk cores per lane
        ix = [tuple(v) for v in self.mk._ix_nocs(t)]  # the index core / every lane's index core (the AG consumers)
        assert len(ix) == nl, (t, nl, ix)
        rs_rows = t * MK.SHARD_RS_CHUNKS * 32
        # t x (the token's lane chunk cores + its lane program) + the latent core's q_lat copy
        ag_wait = (MK.SEM_FSW, t * (cl + 1) + 1)
        assert ag_wait[1] == SP.ag_producers(t, self.mk.chunks, nl), (t, nl, ag_wait)  # (the plan's per-lane count)
        assert not SP.lanes_reasons(t, nl, self.mk.chunks, self.L, self.mk.groups), SP.lanes_reasons(t, nl)
        rs_wait = (MK.SEM_FSW, t * self.mk.groups)  # one root increment per token (its lane's root)
        lane = lambda name: [self.lane_core[name]]  # noqa: E731
        share = self._sync_from.streams.get(t, {}).get("objs", {}) if self._sync_from is not None else {}
        sy = lambda name: dict(sync=share[name].sync) if name in share else {}  # noqa: E731
        # rev 59 (review 7): the RS consumers are the roots only (chunk core 0 of each head group; the other owners
        # never wait on SEM_RSC); rev 60: chunk core 0 of every (group, lane)
        roots = [tuple(self.mk.owner_noc[g * self.mk.chunks + l_ * cl]) for g in range(self.mk.groups)
                 for l_ in range(nl)]  # fmt: skip
        sl = dict(slots=link_stage_slots(self.fabric(t).b)) if self.link_lanes else {}  # slots per the program's AR phase (the floor)
        if self.L == 1:  # the pair: the validated one-round form (link 0: AG, link 1: RS; both gated by the AR lanes)
            x_ag = z(2 * t * 32, HIDDEN)
            x_rs = z(2 * rs_rows, W_RS)
            ag = FS.FabricStreamAG(mesh, 2 * t * 32, HIDDEN, lanes=lane("ag1"), **sy("ag1"), local_copy=True, wait=ag_wait,
                                   cons=(MK.SEM_AGC, ix), link_base=0, after=MK.SEM_FSA, **sl)  # fmt: skip
            rs = FS.FabricStreamRS2(mesh, rs_rows, W_RS, lanes=lane("rs1"), **sy("rs1"), wait=rs_wait, cons=(MK.SEM_RSC, roots),
                                    link_base=1, after=MK.SEM_FSA, **sl)  # fmt: skip
            return dict(C=self.n, line=1, x_ag=x_ag, y_ag=ag.y, x_rs=x_rs, y_rs=rs.y, ag=ag, rs=rs,
                        ag_lane=ag.lane_noc()[0], rs_lane=rs.lane_noc()[0], t=t, lanes=nl, ag_wait=ag_wait[1],
                        objs=dict(ag1=ag, rs1=rs), x=dict(ag1=x_ag, rs1=x_rs))  # fmt: skip
        # TP8: AG1 (axis 0) -> AG2 (axis 1, the bundles) -> merge; RS1 (axis 0) -> combine -> RS2 (axis 1 scatter)
        x_ag = z(t * 32, W8)
        x_rs = z(2 * rs_rows, W_RS)
        x_rs2 = z(self.L * t * 32, W8)
        q_asm = z(2 * t * 32, HIDDEN)  # rev 59: the two head tiles per token, assembled by the index core (ix_dm1.cpp)
        ag2_noc = self._noc(self.lane_core["ag2"])
        ag1 = FS.FabricStreamAG(mesh, t * 32, W8, axis=geo.pair_axis, lanes=lane("ag1"), **sy("ag1"), local_copy=True, wait=ag_wait,
                                cons=(MK.SEM_FSW, [ag2_noc]), link_base=0, after=MK.SEM_FSA, **sl)  # fmt: skip
        # GLM53_DSA_SHARD_LINK_LANES: AG2 shares AG1's core and its SEM_FSW -> AG2 waits for AG1's producers + AG1's
        # waiter signal (cumulative); RS2 likewise for both rounds' root increments (RS2's follow RS1's landing)
        ag2_wait = (MK.SEM_FSW, ag_wait[1] + 1) if self.link_lanes else (MK.SEM_FSW, 1)
        # (v2: RS2 on the link-0 core after AG1's producers + AG1's waiter signal; RS1 alone on link 1)
        rs2_wait = (MK.SEM_FSW, ag_wait[1] + 1 + rs_wait[1]) if self.link_lanes else rs_wait
        ag2 = FS.FabricStreamAG(mesh, 2 * t * 32, W8, axis=geo.line_axis, lanes=lane("ag2"), **sy("ag2"), bundle=True, y=ag1.y,
                                wait=ag2_wait, cons=(MK.SEM_AGC, ix), link_base=0, after=MK.SEM_FSA, **sl)  # fmt: skip
        rs1 = FS.FabricStreamRS2(mesh, rs_rows, W_RS, axis=geo.pair_axis, lanes=lane("rs1"), **sy("rs1"), wait=rs_wait,
                                 cons=(MK.SEM_RSC, roots), link_base=1, after=MK.SEM_FSA, **sl)  # fmt: skip
        rs2 = FS.FabricStreamScatter(mesh, t * 32, W8, axis=geo.line_axis, lanes=lane("rs2"), **sy("rs2"), wait=rs2_wait,
                                     cons=(MK.SEM_RSC, roots), link_base=0 if self.link_lanes else 1, after=MK.SEM_FSA,
                                     **sl)  # fmt: skip
        assert int(ag1.y.shape[-2]) == geo.n * t * 32 and int(rs2.y.shape[-2]) == self.L * t * 32
        return dict(C=self.n, line=self.L, x_ag=x_ag, y_ag=ag1.y, x_rs=x_rs, y_rs=rs1.y, x_rs2=x_rs2, y_rs2=rs2.y,
                    q_asm=q_asm,
                    ag=ag1, rs=rs1, ag2=ag2, rs2=rs2, ag_lane=ag1.lane_noc()[0], rs_lane=rs1.lane_noc()[0],
                    ag2_lane=ag2_noc, rs2_lane=rs2.lane_noc()[0], t=t, lanes=nl, ag_wait=ag_wait[1],
                    objs=dict(ag1=ag1, ag2=ag2, rs1=rs1, rs2=rs2),
                    x=dict(ag1=x_ag, ag2=ag1.y, rs1=x_rs, rs2=x_rs2))  # fmt: skip

    def _audits(self, r: int, c: int, st: dict, fab, entries: list, late: dict) -> None:
        """the link-holder, NoC-target / rank-word and packet-header audits of chip (r, c)'s program: raise on any
        violation (every build, the mock included); the results go to self.*_audit[(r, c)] (recorded by the unit)"""
        geo, rank = self.geo, self.geo.rank(r, c)
        objs = st["objs"]
        # link holders from the objects themselves, then cross-checked against the plan (tt/shard_plan.py Geometry)
        holders = []
        for name, o in objs.items():
            assert o.after is not None, (name, "every stream is gated (FS_LATE_OPEN)")
            for h in range(len(o.lanes)):
                for d in o.directions(r, c):
                    holders.append((name, o.link_base + h, d, "gated", f"ar_lane{o.link_base + h}"))
        ar_dirs = geo.ar_directions(r, c)
        for h in (0, 1):
            for d in ar_dirs:
                holders.append((f"ar_lane{h}", h, d, "late", None))
        # rev 59 (review 5): every stream's block / destination offsets = the plan's (shard_plan.Geometry.block)
        for name, o in objs.items():
            p_, line = o._pos(r, c)
            src, dst, local, up, dn = o._block(p_, line)
            want = geo.block(name, st["t"], r, c)
            got = dict(p=p_, src=src if not o.scatter else None, dst=dst, local=bool(local), up=up, dn=dn,
                       dsts=list(o._dsts(p_)))  # fmt: skip
            if got != {k: want[k] for k in got}:
                raise ValueError(f"chip {(r, c)} stream {name}: blocks {got} differ from the plan {want}")
        plan = geo.holders(r, c, st["t"])
        if sorted(holders) != sorted(plan):
            raise ValueError(f"chip {(r, c)}: link holders differ from the plan: {sorted(holders)} vs {sorted(plan)}")
        if self.link_lanes:  # the per-link lane cores: every stream runs on its link's AR lane core, before that lane
            for name, o in objs.items():
                if [tuple(v) for v in o.lanes] != [tuple(self.lanes[o.link_base])] or late[o.link_base][1]:
                    raise ValueError(f"chip {(r, c)}: {name} on link {o.link_base} is not a phase of ar_lane{o.link_base}'s "
                                     f"core {tuple(self.lanes[o.link_base])} (lanes {o.lanes}, release list {late})")
        else:
            rel = {h: sorted(self._noc_names(late[h][1], st)) for h in late}
            for name, o in objs.items():  # every gated stream is released by the AR lane of its link
                if name not in rel[o.link_base]:
                    raise ValueError(f"chip {(r, c)}: {name} on link {o.link_base} is not released by ar_lane{o.link_base}")
        self.link_audit[f"{r},{c}"] = SP.audit_links(holders, sequential=self.link_lanes)
        # NoC targets: the patched shard blocks + every stream's consumer / lane list + the AR release lists
        extra = [(f"{n}.cons", tuple(xy)) for n, o in objs.items() for xy in (o.cons[1] if o.cons else [])]
        extra += [(f"{n}.lane", tuple(xy)) for n, o in objs.items() for xy in o.lane_noc()]
        extra += [(f"ar_lane{h}.release", tuple(xy)) for h in late for xy in late[h][1]]
        self.target_audit[f"{r},{c}"] = SP.audit_targets(entries, rank, 1 - geo.pair_pos(rank), self.valid_noc, extra)
        # fabric packet headers per (core, RISC): fs_send NHDR per direction, lane_send 2 + 1 flag per direction
        budget = {(n, "brisc"): o.headers(r, c) for n, o in objs.items()}
        budget.update({(f"ar_lane{h}", "brisc"): SP.FAR_NHDR * len(ar_dirs) for h in (0, 1)})
        if budget != geo.header_budget(r, c, st["t"]):
            raise ValueError(f"chip {(r, c)}: header budget differs from the plan: {budget}")
        self.header_audit[f"{r},{c}"] = SP.audit_headers(budget)

    def _cb_census(self, prog, st: dict) -> dict:
        """static CB bytes per core of chip (0, 0)'s program (every chip has the same CB layout) by role: the owners'
        maximum (the L1 floor binder), the index / latent cores, the AR / stream lane cores, and the overall maximum"""
        per = {}
        for cb in prog.cbs:
            for rg in cb.core_ranges.ranges():
                for x in range(rg.start.x, rg.end.x + 1):
                    for y in range(rg.start.y, rg.end.y + 1):
                        per[(x, y)] = per.get((x, y), 0) + int(cb.total_size)
        xy = lambda c: (int(c.x), int(c.y))  # noqa: E731
        owners = [xy(c) for c in self.mk.place["k"]]
        top = max(per.items(), key=lambda kv: kv[1])
        return dict(t=st["t"], lanes=st.get("lanes", 1), max_cb_bytes=top[1], max_cb_core=list(top[0]),
                    owner_max=max(per.get(o, 0) for o in owners),
                    ix_core=per.get(xy(self.mk.ix_core), 0), lat_core=per.get(xy(self.mk.lat_core), 0),
                    ix_lane_cores={f"{c.x},{c.y}": per.get(xy(c), 0) for c in self.mk._ix_cores(st["t"])}
                    if st.get("lanes", 1) > 1 else {},
                    ar_lanes=[per.get(tuple(v), 0) for v in self.lanes],
                    stream_lanes={n: per.get(tuple(self.lane_core[n]), 0) for n in st["objs"]})  # fmt: skip

    def _cb_report(self, prog, st: dict, top: int = 12) -> None:
        """GLM53_MK_CB_REPORT: chip (0, 0)'s static CB bytes per buffer index on the `top` cores with the most CB bytes
        (+ their role) and the L1 allocator's view, printed (MK_CB_REPORT lines: diff two builds' reports)"""
        per = {}
        for cb in prog.cbs:
            idx = tuple(int(f.buffer_index) for f in cb.format_descriptors)
            for rg in cb.core_ranges.ranges():
                for x in range(rg.start.x, rg.end.x + 1):
                    for y in range(rg.start.y, rg.end.y + 1):
                        d = per.setdefault((x, y), {})
                        d[idx] = d.get(idx, 0) + int(cb.total_size)
        xy = lambda c: (int(c.x), int(c.y))  # noqa: E731
        role = {xy(c): "owner" for c in self.mk.place["k"]}
        role[xy(self.mk.place["root"])] = "root"
        role[xy(self.mk.lat_core)] = "latent"
        role.update({xy(c): "qnorm" for c in self.mk.qn_cores})
        role[xy(self.mk.ix_core)] = "index"
        role.update({xy(c): "ix_lane" for c in self.mk.co_cores})
        role.update({tuple(v): "ar_lane" for v in self.lanes})
        role.update({tuple(v): f"stream_{n}" for n, v in self.lane_core.items()})
        for c_, _, _ in self.mk.last_readers.assign:
            role.setdefault(xy(c_), "reader")
        try:
            mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
            l1 = dict(alloc_per_bank=int(mv.total_bytes_allocated_per_bank),
                      largest_free_per_bank=int(mv.largest_contiguous_bytes_free_per_bank))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            l1 = dict(error=str(e)[:120])
        print(f"MK_CB_REPORT shard t={st['t']} lanes={st.get('lanes', 1)} l1={l1}", flush=True)
        if os.environ.get("GLM53_MK_CB_REPORT") == "2":  # + every live L1 buffer (largest per bank first), with its tensor
            self._l1_census()
        for core, d in sorted(per.items(), key=lambda kv: -sum(kv[1].values()))[:top]:
            print(f"MK_CB_REPORT shard t={st['t']} core={core} role={role.get(core, '?')} total={sum(d.values())} "
                  + " ".join(f"{'/'.join(map(str, k))}:{v}" for k, v in sorted(d.items())), flush=True)  # fmt: skip

    def _l1_census(self, top: int = 60) -> None:
        """GLM53_MK_CB_REPORT=2: the live L1 buffers of the mesh (ttnn reports.get_buffers: address, bytes per bank,
        layout) joined with the live ttnn tensors that own them (gc: shape, dtype, memory layout), largest first"""
        import gc

        try:
            bufs = ttnn._ttnn.reports.get_buffers(self.mesh)
        except Exception as e:  # noqa: BLE001
            print(f"MK_L1_CENSUS error {str(e)[:160]}", flush=True)
            return
        own = {}

        def where(o):  # the attribute / container names that hold tensor o (one level up)
            out = []
            for r in gc.get_referrers(o):
                if isinstance(r, dict):
                    keys = [str(k) for k, v in r.items() if v is o][:2]
                    holders = [type(h).__name__ for h in gc.get_referrers(r)
                               if getattr(h, "__dict__", None) is r][:1]  # fmt: skip
                    out.append(f"{holders[0] if holders else 'dict'}.{'/'.join(keys)}")
                elif isinstance(r, (list, tuple)):
                    out.append(type(r).__name__ + f"[{len(r)}]")
            return ",".join(out[:3])

        for o in gc.get_objects():
            if type(o).__name__ != "Tensor":
                continue
            try:
                a = int(o.buffer_address())
                mc = o.memory_config()
                if "L1" not in str(mc.buffer_type):
                    continue
                own.setdefault(a, f"{list(o.shape)} {o.dtype} {mc.memory_layout} held_by={where(o)}")
            except Exception:  # noqa: BLE001
                continue
        rows = {}
        for b_ in bufs:
            if "L1" not in str(b_.buffer_type) or int(b_.device_id) != 0:
                continue
            rows[int(b_.address)] = (int(b_.max_size_per_bank), str(b_.buffer_layout))
        tot = sum(v[0] for v in rows.values())
        print(f"MK_L1_CENSUS device 0: {len(rows)} L1 buffers, {tot} B per bank (sum of max sizes)", flush=True)
        for addr, (sz, lay) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"MK_L1_CENSUS addr={addr} per_bank={sz} layout={lay} tensor={own.get(addr, '?')}", flush=True)

    def _noc_names(self, nocs: list, st: dict) -> list:
        by = {tuple(o.lane_noc()[0]): n for n, o in st["objs"].items()}
        return [by.get(tuple(xy), str(xy)) for xy in nocs]

    def _hooked_shard(self, io, desc, t: int, st: dict):
        """TPMla._hooked (the o_proj all-reduce lanes) + the stream lanes + the per-chip rank / peer-position patch."""
        fab = self.fabric(t)
        o_bit = self.mk.phase_order.index("o")
        n_readers = len(self.mk.last_readers.assign)
        slots = dict(self.mk.shard_slots)
        assert {"owner_dm0.cpp", "owner_dm1.cpp", "ix_dm0.cpp", "ix_dm1.cpp", "lat_dm1.cpp"} <= set(slots), slots
        objs = st["objs"]
        order = list(objs)  # kernel order of the stream lanes after the AR lanes
        pa = self.mk.probe.buffer_address() if self.mk.probe is not None else None  # rev 59: the lanes' probe rows
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                p = self._pos(r, c)
                peer = 1 - self.geo.pair_pos(p)  # the pair peer's position: x_rs block of pass 0 / its y_ag block (TP2)
                kernels, seen, audit = [], {"ra": 0, "own": 0}, []
                n_cores, patched = {}, {}  # per slot source: the kernels' cores / the cores patched (review 4)
                for kd in desc.kernels:  # copies: patched per chip
                    if _src_is(kd, "mla/ra.cpp"):
                        rt = max(len(v) for v in _rt_dict(kd).values())
                        defs = list(kd.defines) + [("MK_FAB_MASK", str(1 << o_bit)), ("MK_FAB_BROWS", str(fab.brows)),
                                                   ("MK_FAB_RT", str(rt))]  # fmt: skip
                        kd = _with_rt(kd, fab.producer_args(), defs)[0]
                        seen["ra"] += 1
                    elif _src_is(kd, "mla/owner_dm0.cpp"):
                        rt = max(len(v) for v in _rt_dict(kd).values())
                        defs = [d for d in kd.defines if d[0] != "MK_POST_WAIT_COUNT"]
                        defs += [("MK_POST_WAIT_COUNT", "2"), ("MK_POST_Y_FABRIC", "1"), ("MK_POST_FAB_RT", str(rt))]
                        kd = _with_rt(kd, fab.consumer_args(), defs)[0]
                        seen["own"] += 1
                    name = os.path.basename(str(kd.kernel_source).replace("\\", "/"))
                    if name in slots:  # this chip's rank (and the peer's position) into the shard arg slots
                        n_cores[name] = n_cores.get(name, 0) + len(_rt_dict(kd))
                        for x, y in _rt_dict(kd):
                            for ri, pi, cores, nocs in slots[name]:  # the entry of this core's kernel group
                                if (x, y) not in cores:
                                    continue  # (lat_dm1.cpp also runs on the q-norm cores, without the shard block)
                                kd.runtime_args[x][y][ri] = p
                                if pi is not None:
                                    kd.runtime_args[x][y][pi] = peer
                                patched.setdefault(name, set()).add((x, y))
                                rt = list(kd.runtime_args[x][y])
                                audit.append(
                                    (
                                        name,
                                        (x, y),
                                        rt[ri],
                                        None if pi is None else rt[pi],
                                        [(rt[ri + dx], rt[ri + dy]) for dx, dy in nocs],
                                    )
                                )
                    kernels.append(kd)
                assert seen["ra"] >= 1 and seen["own"] >= 1, seen
                # rev 59 (review 4): every core of a slot source is patched (lat_dm1.cpp: exactly its slot cores = the
                # latent core; its q-norm cores run the same source without the shard block) -- a silently skipped core
                # was the pair's second hang (a zeroed NoC target)
                for name in slots:
                    want = (set().union(*(e[2] for e in slots[name])) if name == "lat_dm1.cpp" else None)
                    got = patched.get(name, set())
                    if (want is not None and got != want) or (want is None and len(got) != n_cores.get(name, -1)):
                        raise ValueError(f"chip {(r, c)}: {name} patched {len(got)} cores of "
                                         f"{n_cores.get(name)} (expected {len(want) if want is not None else 'all'})")
                # the AR lanes hold their link only around their own round (FAR_LATE_OPEN): at launch start lane h
                # closes and signals every stream lane borrowing link h (pair: AG / RS; TP8: AG1 + AG2 / RS1 + RS2)
                late = {h: (MK.SEM_FSA, [] if self.link_lanes else
                            [tuple(o.lane_noc()[0]) for o in objs.values() if o.link_base == h])
                        for h in (0, 1)}  # fmt: skip  (link lanes: the streams run before the AR phase, in-kernel)
                assert all(len(v[1]) == len(late[0][1]) for v in late.values()), late  # (same rt widths per lane)
                lk, lcbs = fab.lane_parts(r, c, 1, n_readers, self.mk.owner_noc, cons_sem=MK.SEM_A, late_open=late)
                sk, scbs, sems = [], [], list(desc.semaphores)
                idx = {}
                on_core = {}  # link lanes: the streams per lane core, in order; their stage CB = ONE index per stage
                # size from 20 up (the AR lane roles take 0..19, fabric_ar.py CB_L1 .. CB_OWN): a core's phases share it
                stage_ix = {sz: LINK_CB_STAGE0 + k for k, sz in enumerate(
                    sorted({int(objs[n].slots * objs[n].nb * FS.TB) for n in order}))}  # fmt: skip
                for nm in order:
                    probe = (pa, SP.PROBE_LANE_ROW[nm]) if pa is not None else None
                    assert probe is None or SP.PROBE_LANE_ROW[nm] < MK.PROBE_ROWS, (nm, MK.PROBE_ROWS)
                    kw = {}
                    if self.link_lanes:  # (no lane probe in a composite phase: tt/phase_merge.py)
                        core = tuple(self.lane_core[nm])
                        on_core.setdefault(core, []).append(nm)
                        probe = None
                        kw = dict(cb_stage=stage_ix[int(objs[nm].slots * objs[nm].nb * FS.TB)])
                    k_, cb_, _, _ = objs[nm].parts(st["x"][nm], r, c, probe=probe, **kw)
                    idx[nm] = len(kernels) + len(lk) + len(sk)
                    sk += k_
                    scbs += cb_
                    sems += objs[nm].semaphores()
                if self.link_lanes:
                    scbs = _merge_stage_cbs(scbs)
                prog = ttnn.ProgramDescriptor(kernels=kernels + lk + sk, semaphores=sems,
                                              cbs=list(desc.cbs) + lcbs + scbs)  # fmt: skip
                FS.CONNECT_MEMO = {} if self.link_lanes else None  # link lanes: the streams reuse the AR's connections
                try:
                    fab.add_connections(prog, r, c, send_kernel_index=len(kernels))
                    for nm in order:
                        objs[nm].connect(prog, r, c, idx[nm])
                finally:
                    FS.CONNECT_MEMO = None
                if self.link_lanes:
                    prog = self._link_composites(prog, len(kernels), idx, on_core)
                self._cb_model_check(prog, r, c, t)  # the lane cores' CB END ADDRESSES vs the floor (raises: host-side)
                self._audits(r, c, st, fab, audit, late)
                if (r, c) == (0, 0):
                    self.census = self._cb_census(prog, st)
                    self.censuses[t] = self.census
                    if os.environ.get("GLM53_MK_CB_REPORT"):  # static CB bytes per CB index on the top cores
                        self._cb_report(prog, st)
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = prog
        extra = [objs[nm].sync for nm in order]
        return list(io) + fab.io() + extra, mpd

    def _cb_model_check(self, prog, r: int, c: int, t: int) -> None:
        """The HOST CB MODEL (program ruling 2026-10-27T13:56:44Z): every AR / stream lane core's CB region END ADDRESS
        = CB base + the CB bytes on it (globally allocated CBs excluded) must stay <= GLM53_L1_FLOOR. The bar is the SERVED
        construction (t = 32): base GLM53_CB_BASE (default 291,968 = L1_SIZE - the served worker L1 1,280,896) and floor
        1,270,912; the CB bytes are the same program in a harness (whose own base / floor differ: e.g. worker_l1
        1,264,512 -> base 308,352 with a much higher floor), so the served bar is checked everywhere. A build over it
        raises here, never in a capture."""
        if getattr(self, "_cb_base", None) is None:
            self._cb_base = int(os.environ.get("GLM53_CB_BASE", "") or CB_BASE_SERVED)
        floor = int(os.environ.get("GLM53_L1_FLOOR", "") or L1_FLOOR_SERVED)
        live = live_l1_floor(self.mesh)  # the floor THIS build meets now; the constant covers the layers still to come
        if live is not None:
            floor = min(floor, live)
        cores = {tuple(v) for v in self.lanes} | {tuple(v) for v in self.lane_core.values()}
        per = cb_region_bytes(prog, cores)
        ends = {f"{x},{y}": self._cb_base + b for (x, y), b in sorted(per.items())}
        self.cb_model = getattr(self, "cb_model", {})
        self.cb_model[t] = dict(base=self._cb_base, floor=floor, ends=ends, bytes={f"{x},{y}": b for (x, y), b in per.items()})
        if (r, c) == (0, 0) and not getattr(self, "_cb_model_printed", {}).get(t):
            self.__dict__.setdefault("_cb_model_printed", {})[t] = True
            print("DSA_SHARD_CB_MODEL " + __import__("json").dumps(dict(t=t, link_lanes=self.link_lanes,
                                                                          **self.cb_model[t])), flush=True)  # fmt: skip
        over = {k: v for k, v in ends.items() if v > floor}
        if over:
            raise ValueError(f"TPMlaShard CB model (t = {t}, chip {(r, c)}): lane-core CB region ends {over} exceed the L1 "
                             f"floor {floor} (base {self._cb_base}; GLM53_L1_FLOOR / live {live})")
        # every program built earlier is re-validated at its next enqueue against the floor as it is NOW (it only moves
        # down as layers / widths are built): its registered lane-core ends must still clear it
        reg = _CB_ENDS.setdefault(id(self.mesh), {})
        stale = {key: {k: v for k, v in e.items() if v > floor} for key, e in reg.items()}
        stale = {key: o for key, o in stale.items() if o}
        if stale:
            raise ValueError(f"TPMlaShard CB model (t = {t}): the L1 floor moved to {floor} (live {live}) under an earlier "
                             f"shard program's lane-core CB ends {stale}")
        reg[(id(self), t)] = ends

    def _link_composites(self, prog, i_lane: int, idx: dict, on_core: dict):
        """GLM53_DSA_SHARD_LINK_LANES: chip program `prog` with, on each AR lane core h, ONE BRISC composite (the link's
        streams' fs_send in order, then lane_send) and ONE NCRISC composite (their fs_read, then lane_recv) in place of
        the separate kernels (tt/phase_merge.py); lane_reduce (TRISC) stays. i_lane = the index of lane_send in
        prog.kernels (lane_recv follows it, lane_reduce after); idx[name] = a stream's fs_send index (fs_read follows).
        The streams' FS_AFTER_SEM (the AR lane's release of the link) is dropped: in the composite they run first."""
        ks = list(prog.kernels)
        drop = {i_lane, i_lane + 1}
        comps = []
        for h, lane in enumerate(self.lanes):
            core = tuple(lane)
            names = on_core.get(core, [])
            assert names and all(self.lane_core[n] == core for n in names), (core, names)
            send = [(ks[idx[n]], {"FS_AFTER_SEM"}) for n in names] + [(ks[i_lane], set())]
            read = [(ks[idx[n] + 1], set()) for n in names] + [(ks[i_lane + 1], set())]
            comps += [PM.merge(send, core, name=f"link{h} BRISC: {' -> '.join(names)} -> lane_send"),
                      PM.merge(read, core, name=f"link{h} NCRISC: {' -> '.join(names)} -> lane_recv")]  # fmt: skip
            drop |= {idx[n] for n in names} | {idx[n] + 1 for n in names}
        lane_cores = {tuple(v) for v in self.lanes}
        rest = []
        for i, kd in enumerate(ks):
            if i in (i_lane, i_lane + 1):  # the AR lane send / recv on cores that got no composite (none today)
                kd = PM.restrict(kd, lane_cores)
                if kd is not None:
                    rest.append(kd)
            elif i not in drop:
                rest.append(kd)
        seen, sems = set(), []
        for sd in prog.semaphores:  # the streams of one core each declared its wait / release semaphore: once per core
            key = (int(sd.id), tuple(sorted(PM._cores_of(sd.core_ranges))), int(sd.initial_value))
            if key not in seen:
                seen.add(key)
                sems.append(sd)
        return ttnn.ProgramDescriptor(kernels=rest + comps, semaphores=sems, cbs=list(prog.cbs))

    def __call__(self, x, cache, page_table, cur_pos, *, t: int, dsa_state: dict | None = None, keep=None,
                 stack: bool = False):  # fmt: skip
        """MkMla's call on every chip with the split-K phase: cache / page_table / dsa_state are the per-chip SHARDED
        tensors (tt/dsa_shard.py ShardMap: local blocks, local page table, local pooled keys; the ring replicated)."""
        self.calls += 1
        st = self.streams.get(t)
        if st is None:
            raise ValueError(f"TPMlaShard: no streams for t = {t} (built for {sorted(self.streams)}: ts / stream_ts)")
        with _Capture() as cap:
            out = self.mk(x, cache, page_table, cur_pos, t=t, dsa_state=dsa_state, stack=stack, shard=st)
        ((io, desc),) = cap.calls
        io2, mpd = self._hooked_shard(io, desc, t, st)
        TPM.first_launch_sync(self.mesh, self.fabric(t))
        ttnn.generic_op(io2, mpd)
        if keep is not None:
            for k in [k for k in out if k not in keep]:
                v = out.pop(k)
                if v is not None:
                    ttnn.deallocate(v)
        return out

    def release_streams(self):
        for st in self.streams.values():
            for o in st["objs"].values():
                o.release()  # (AG2 lands in AG1's y: only AG1 frees it)
            for k in ("x_ag", "x_rs", "x_rs2", "q_asm"):
                if st.get(k) is not None:
                    ttnn.deallocate(st[k])
        self.streams = {}

    def shard_probe(self, mhz: float = 1350.0) -> dict:
        """the last replay's shard phase marks per chip (MkMla(probe=True)): {"r,c": tt/shard_plan.py decode_probe}"""
        import torch

        if self.mk.probe is None:
            return {}
        ar = self.fabric(max(self.streams)).lane_marks() if self.fabrics else {}
        out = {}
        for d, ten in enumerate(ttnn.get_device_tensors(self.mk.probe)):
            p = ttnn.to_torch(ten).to(torch.int64).reshape(-1, 16) & 0xFFFFFFFF
            r, c = self.geo.coord(d)
            out[f"{r},{c}"] = SP.decode_probe(p, line=self.L, mhz=mhz, ar=ar.get(d))
        return out
