# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""THE ROW FENCE (program 2026-10-26 20:06:53Z, 14's pre-review 20:09:45Z; design doc/ttnn/16-prefill-production/
pmk/w7/hop_audit.md): one tiny program at the transitions the per-hop drain does not cover -- before each prefill
chunk's first PMK program (the decode / head fabric AR is multi-hop and drains hop 1 only) and after the head.

Why a separate program works: it starts on a chip only after that chip's previous program finished, i.e. after every
arrival that program waited for landed there. What can remain is RELAYED traffic: another chip's multi-hop packets still
queued in this chip's onward router (FABRIC_1D, first-level ack off: drain_close certifies hop 1 only).

mode "mcast" (the fix): per link h of the line axis, every chip sends one LINE-MULTICAST token per direction to the line
end (+1 TOK on lane h of every chip it passes). It rides the source's worker channel and every relay's forwarding
channel, FIFO behind every earlier packet of that source on link h. Lane h waits TOK >= g x (nc - 1).
mode "hop1": 1-hop BAR_L / BAR_R sweeps, the 10-23 row barrier (the reproducer's A/B arm: a relay's token can
overtake relayed packets). mode "none": no tokens (the reproducer's control).

FenceBurst (the class reproducer, kernels/fabric_stream/fence_burst.cpp): position 0 of each line multicasts npkt 4-KiB
packets over the whole line, then +1s LAND on the fence's lane 0 of every other chip; nothing waits for it. The fence's
check (check=True) then counts LEAK when LAND < g after the fence passed.

GLM53_PF_ROW_FENCE = "" (off, default) | "mcast" | "hop1": the served placement (tt/prefill_tp.py). The sync tensor is
allocated at construction -- before any capture -- as a persistent L1 shard at the top of L1 (above every program's
CB_END, never in the kernel-config ring)."""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS

FENCE_WORDS = 16  # row_fence.cpp: 16 words x 16 B per lane core
F = dict(GEN=0, TOK=1, BAR_L=2, BAR_R=3, LAND=4, LEAK=5, SEEN=6, CHECKS=7, T0=8, T1=9, T2=10)
MODES = {"hop1": 0, "mcast": 1, "none": 2}
DEFAULT_LANES = ((0, 0), (1, 0))  # one lane core per fabric link (h = 0, 1); any free cores: the program runs alone


def fence_mode() -> str:
    """GLM53_PF_ROW_FENCE: "" (off) | "mcast" | "hop1" """
    v = os.environ.get("GLM53_PF_ROW_FENCE", "") or ""
    if v not in ("", "mcast", "hop1"):
        raise ValueError(f"GLM53_PF_ROW_FENCE={v!r}: '' | mcast | hop1")
    return v


def _line_axis(mesh, axis):
    r, c = int(mesh.shape[0]), int(mesh.shape[1])
    if axis is None:
        axis = 1 if c >= r else 0
    return axis, (c if axis == 1 else r), (r if axis == 1 else c)


def _pos(axis: int, line: int, p: int):
    """mesh coordinate of position p on line `line` of the line axis"""
    return ttnn.MeshCoordinate(line, p) if axis == 1 else ttnn.MeshCoordinate(p, line)


class RowFence:
    def __init__(self, mesh, *, axis=None, lanes=DEFAULT_LANES, mode: str = "mcast", check: bool = False):
        assert mode in MODES, mode
        self.mesh, self.mode, self.check = mesh, mode, check
        self.axis, self.nc, self.nlines = _line_axis(mesh, axis)
        self.lanes = [tuple(int(v) for v in c) for c in lanes]
        words = FENCE_WORDS * FS.WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(FS._crs(self.lanes), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        self.sync = ttnn.from_torch(torch.zeros(len(self.lanes), words, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self._src = FS.kernel_text("row_fence.cpp")
        self._noc = lambda xy: (lambda w: (int(w.x), int(w.y)))(mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy)))
        # generic_op needs >= 2 io tensors (generic_op_device_operation.cpp:136): a tiny persistent DRAM input, allocated
        # with the sync tensor (before any capture)
        self._io = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self.calls = 0

    def _prog(self, line: int, p: int):
        sync = self.sync.buffer_address()
        rt = ttnn.RuntimeArgs()
        for h, (lx, ly) in enumerate(self.lanes):
            nl = self._noc((lx, ly))
            args = [sync, nl[0], nl[1], p, self.nc, int(self.check and h == 0 and p != 0)]
            rt[lx][ly] = args + [len(args) + 1]  # fab_at: connect() appends the fabric args after this slot
        k = ttnn.KernelDescriptor(kernel_source=self._src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=FS._crs(self.lanes), compile_time_args=[MODES[self.mode]], runtime_args=rt,
                                  config=ttnn.WriterConfigDescriptor())  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
        if self.mode != "none":
            me_id = self.mesh.get_fabric_node_id(_pos(self.axis, line, p))
            for h, lane in enumerate(self.lanes):
                for d, rng in ((-1, p), (1, self.nc - 1 - p)):  # dir 0 = toward lower positions, dir 1 = higher
                    if rng:
                        peer = self.mesh.get_fabric_node_id(_pos(self.axis, line, p + d))
                        fab = FS.fabric_connect(me_id, peer, h, prog, ttnn.CoreCoord(*lane))
                        prog.kernels[0].runtime_args[lane[0]][lane[1]].extend(fab)
        return prog

    def __call__(self):
        mpd = ttnn.MeshProgramDescriptor()
        for line in range(self.nlines):
            for p in range(self.nc):
                m = _pos(self.axis, line, p)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self._prog(line, p)
        ttnn.generic_op([self._io, self.sync], mpd)
        self.calls += 1
        return self.sync

    def words(self) -> dict:
        """host read of every chip's lane words: {(line, position): [[word per F] per lane]}"""
        out = {}
        shards = ttnn.get_device_tensors(self.sync)  # row-major mesh order
        ncol = int(self.mesh.shape[1])
        for line in range(self.nlines):
            for p in range(self.nc):
                row, col = (line, p) if self.axis == 1 else (p, line)
                t = ttnn.to_torch(shards[row * ncol + col]).reshape(len(self.lanes), FENCE_WORDS, FS.WORD // 4)[:, :, 0]
                out[(line, p)] = t.to(torch.int64).tolist()
        return out

    def release(self):
        ttnn.deallocate(self.sync)
        ttnn.deallocate(self._io)


class FenceBurst:
    """position 0 of every line: npkt 4-KiB line-multicast packets into a DRAM landing buffer on every other chip, then
    +1 LAND on fence.lanes[0] of every other chip (link 0); nothing waits for it (kernels/fabric_stream/fence_burst.cpp)
    """

    CB_SRC = 0

    def __init__(self, mesh, fence: RowFence, *, npkt: int = 256, npages: int = 64):
        self.mesh, self.fence, self.npkt, self.npages = mesh, fence, npkt, npages
        self.dst = ttnn.from_torch(torch.zeros(1, 1, npages * 32, 32, dtype=torch.float32), dtype=ttnn.float32,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self._src = FS.kernel_text("fence_burst.cpp")

    def _prog(self, line: int, p: int):
        f = self.fence
        lane = f.lanes[0]
        nl = f._noc(lane)
        rng = f.nc - 1 if p == 0 else 0
        args = [f.sync.buffer_address(), nl[0], nl[1], rng, self.dst.buffer_address(), self.npkt, self.npages]
        rt = ttnn.RuntimeArgs()
        rt[lane[0]][lane[1]] = args + [len(args) + 1]
        ct = [self.CB_SRC] + list(ttnn.TensorAccessorArgs(self.dst).get_compile_time_args())
        k = ttnn.KernelDescriptor(kernel_source=self._src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=FS._crs([lane]), compile_time_args=ct, runtime_args=rt,
                                  config=ttnn.WriterConfigDescriptor())  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=4096, core_ranges=FS._crs([lane]), format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=self.CB_SRC, data_format=ttnn.float32, page_size=4096)])  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
        if rng:
            me_id = self.mesh.get_fabric_node_id(_pos(f.axis, line, p))
            peer = self.mesh.get_fabric_node_id(_pos(f.axis, line, p + 1))
            fab = FS.fabric_connect(me_id, peer, 0, prog, ttnn.CoreCoord(*lane))
            prog.kernels[0].runtime_args[lane[0]][lane[1]].extend(fab)
        return prog

    def __call__(self):
        mpd = ttnn.MeshProgramDescriptor()
        f = self.fence
        for line in range(f.nlines):
            for p in range(f.nc):
                m = _pos(f.axis, line, p)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self._prog(line, p)
        ttnn.generic_op([self.dst, f.sync], mpd)
        return self.dst

    def release(self):
        ttnn.deallocate(self.dst)
