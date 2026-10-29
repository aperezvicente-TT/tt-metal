# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 04a hand-off micro-benchmark: device -> device transfer of the pipeline boundary tensor of the GLM-5.3-Flash
decoder (Stage 2 contract: the 4 mHC residual streams, fp32 [1, 1, rows, 16384] DRAM interleaved; decode rows = B users
x 1 position, prefill chunk = 2048 positions of one user; no other per-layer carry crosses a layer boundary).

Run on ONE card (two Ethernet-adjacent chips), one mode per process (the parent mesh and its submeshes have separate
allocators for the same chips, so parent-mesh ops and submesh ops are never mixed in one process):

  sockets  FABRIC_1D 1x2 mesh, two 1x1 submeshes (one per chip = one pipeline stage), ttnn.MeshSocket pair per
           direction; ttnn.experimental.send_async/recv_async (FIFO) and send_direct_async/recv_direct_async (sender
           writes straight into the receiver's tensor). Every transfer is captured in per-chip traces.
  parent   FABRIC_1D 1x2 mesh used as one mesh: ttnn.point_to_point (mesh-sharded tensor movement, coord (0,0) ->
           (0,1)) and ttnn.all_gather (FABRIC_1D collective used as transport); captured in parent-mesh traces.
  host     no fabric; two 1x1 submeshes with 2 command queues: device -> host -> device through the pinned hugepage
           sysmem (ttnn.copy_device_to_host_tensor / copy_host_to_device_tensor), blocking and overlapped (D2H of
           slice i+1 on chip A's CQ1 while slice i is written to chip B on its CQ1; cross-chip ordering through host
           event_synchronize, as events do not cross meshes).

Latency = one-way hop time = round-trip / 2 of an A -> B -> A ping-pong captured K times in a trace (device-side
critical path: the receiver's op waits on device for the sender's data, no host in the loop); stream time = K
back-to-back A -> B transfers / K (occupancy, i.e. throughput). Rows -> $GLM53_E4A/handoff_bench.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

import ttnn

HC = 16384
E = os.environ.get(
    "GLM53_E4A", "/opt/glm53/kit/doc/ttnn/04a-multichip-handoff-prework"
)
TRACKING = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
PROFILING = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
DRAM = ttnn.DRAM_MEMORY_CONFIG


def quiesce(mesh):
    """Parent mesh with submeshes: tt-metal refuses to close a parent whose CQ is "in use" by a child (TT_THROW "cq ID 0
    is in use by child submesh") -- quiesce_devices() quiesces the submeshes and clears the flags (needed for the
    device profiler's final dump, 2026-10-17 22:29Z)."""
    try:
        mesh.quiesce_devices()
    except Exception as e:  # noqa: BLE001
        print(f"quiesce_devices failed: {e!r}", flush=True)


try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_a, **_k):
        pass


def flush(*meshes):
    """Device-profiler runs: drain each device's profiler buffer between cases (no-op otherwise)."""
    if PROFILING:
        for m in meshes:
            ttnn.ReadDeviceProfiler(m)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


PAYLOAD = {"v": 0}


def record(row, name="handoff_bench.jsonl"):
    row = {
        "time_utc": now(),
        **row,
        "fabric_payload_bytes": PAYLOAD["v"] or 4352,
        "trace_alloc_tracking": TRACKING,
        "profiler": PROFILING,
        "visible": os.environ.get("TT_VISIBLE_DEVICES"),
        "lock": os.environ.get("TT_LOCK_GROUP"),
    }
    path = os.environ.get("GLM53_HANDOFF_OUT", os.path.join(E, name))
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row), flush=True)


def cases(sel):
    """(mode, batch, layout, rows) -- decode TILE pads the B rows to 32 (2 MiB on the wire for every B <= 32);
    decode ROW_MAJOR moves exactly B x 64 KiB (untilize on the sender / tilize on the receiver counted in)."""
    out = []
    for b in (1, 8, 32):
        out.append(("decode", b, "tile", b))
    for b in (1, 8):
        out.append(("decode", b, "rm", b))
    out.append(("prefill", 1, "tile", 2048))
    return [c for c in out if not sel or f"{c[0]}:{c[1]}:{c[2]}" in sel]


def payload(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((1, 1, rows, HC), generator=g, dtype=torch.float32)


def bytes_of(rows, layout):
    logical = rows * HC * 4
    moved = (((rows + 31) // 32) * 32 * HC * 4) if layout == "tile" else logical
    return logical, moved


def dev_tensor(x, mesh, layout="tile"):
    return ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
    )


def to_host(t):
    return ttnn.to_torch(t).float()


def timed_replays(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


# ------------------------------------------------------------------------------------------------------- sockets
def set_fabric(a):
    """FABRIC_1D before the mesh opens; optional fabric packet payload override (default 4352 B on Blackhole, max
    14 x 1088 = 15232 B)."""
    if a.payload:
        rc = ttnn.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = a.payload
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, router_config=rc)
    else:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)


def run_sockets(a):
    set_fabric(a)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=a.trace_region)
    A = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    B = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
    c0 = ttnn.MeshCoordinate(0, 0)
    try:
        for transfer in a.transfers.split(","):
            for nconn in (int(x) for x in a.connections.split(",")):
                for fifo in [128] if transfer == "direct" else [int(x) for x in a.fifos.split(",")]:
                    socket_case(a, mesh, A, B, c0, transfer, nconn, fifo)
    finally:
        for s in (A, B):
            ttnn.synchronize_device(s)
        quiesce(mesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def make_pair(src, dst, c0, nconn, fifo, row):
    # sender cores on core row `row`, receiver cores on row `row + 1`; distinct rows per socket pair
    conns = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(i, row)), ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(i, row + 1))
        )
        for i in range(nconn)
    ]
    cfg = ttnn.SocketConfig(conns, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, fifo))
    return ttnn.create_socket_pair(src, dst, cfg)


def socket_case(a, mesh, A, B, c0, transfer, nconn, fifo):
    send = ttnn.experimental.send_direct_async if transfer == "direct" else ttnn.experimental.send_async
    recv = ttnn.experimental.recv_direct_async if transfer == "direct" else ttnn.experimental.recv_async
    for mode, batch, layout, rows in cases(a.cases):
        logical, moved = bytes_of(rows, layout)
        page = 4096 if layout == "tile" else HC * 4
        if transfer != "direct" and fifo < page:
            continue
        tag = f"sock_{transfer} conn={nconn} fifo={fifo} {mode} B={batch} {layout}"
        try:
            s_ab, r_ab = make_pair(A, B, c0, nconn, fifo, 0)
            s_ba, r_ba = make_pair(B, A, c0, nconn, fifo, 2)
            x = payload(rows, seed=batch)
            xa = dev_tensor(x, A, "tile")
            spec_rm = None
            if layout == "rm":
                xa_rm = ttnn.to_layout(xa, ttnn.ROW_MAJOR_LAYOUT)
                spec = xa_rm.spec
            else:
                spec = xa.spec
            yb = ttnn.allocate_tensor_on_device(spec, B)  # receive buffer on chip B (persistent)
            ya = ttnn.allocate_tensor_on_device(spec, A)  # return buffer on chip A

            def a_send():
                src = ttnn.to_layout(xa, ttnn.ROW_MAJOR_LAYOUT) if layout == "rm" else xa
                send(src, s_ab)
                return src

            def b_echo():
                recv(yb, r_ab)
                if layout == "rm":  # consumer view: tilize, then untilize again for the return hop
                    t = ttnn.to_layout(yb, ttnn.TILE_LAYOUT)
                    back = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                    send(back, s_ba)
                    return [t, back]
                send(yb, s_ba)
                return []

            def a_recv():
                recv(ya, r_ba)
                return ttnn.to_layout(ya, ttnn.TILE_LAYOUT) if layout == "rm" else ya

            # warm (compiles every program) + correctness: A -> B -> A must be bit-exact
            src = a_send()
            tmp = b_echo()
            back = a_recv()
            ttnn.synchronize_device(A)
            ttnn.synchronize_device(B)
            got_b = to_host(ttnn.to_layout(yb, ttnn.TILE_LAYOUT) if layout == "rm" else yb)
            got_a = to_host(back)
            exact = bool(torch.equal(got_b[..., :rows, :], x) and torch.equal(got_a[..., :rows, :], x))
            # ping-pong trace: K round trips
            K = a.k_prefill if mode == "prefill" else a.k
            ta = ttnn.begin_trace_capture(A, cq_id=0)
            tb = ttnn.begin_trace_capture(B, cq_id=0)
            keep = []
            for _ in range(K):
                keep.append(a_send())
                keep += b_echo()
                keep.append(a_recv())
            ttnn.end_trace_capture(A, ta, cq_id=0)
            ttnn.end_trace_capture(B, tb, cq_id=0)

            def pp():
                ttnn.execute_trace(A, ta, cq_id=0, blocking=False)
                ttnn.execute_trace(B, tb, cq_id=0, blocking=False)
                ttnn.synchronize_device(A)
                ttnn.synchronize_device(B)

            pp()
            flush(A, B)
            signpost(f"HANDOFF sock_{transfer} c{nconn} f{fifo} {mode} B{batch} {layout} pingpong K{K}")
            med, ts = timed_replays(pp, a.reps)
            signpost("HANDOFF_END")
            flush(A, B)
            rtt = med / K
            ttnn.release_trace(A, ta)
            ttnn.release_trace(B, tb)
            # stream trace: K one-way transfers back to back
            ta = ttnn.begin_trace_capture(A, cq_id=0)
            tb = ttnn.begin_trace_capture(B, cq_id=0)
            for _ in range(K):
                keep.append(a_send())
                recv(yb, r_ab)
            ttnn.end_trace_capture(A, ta, cq_id=0)
            ttnn.end_trace_capture(B, tb, cq_id=0)

            def st():
                ttnn.execute_trace(A, ta, cq_id=0, blocking=False)
                ttnn.execute_trace(B, tb, cq_id=0, blocking=False)
                ttnn.synchronize_device(A)
                ttnn.synchronize_device(B)

            st()
            smed, _ = timed_replays(st, a.reps)
            ttnn.release_trace(A, ta)
            ttnn.release_trace(B, tb)
            # post-replay correctness of the persistent receive buffer
            got_b2 = to_host(ttnn.to_layout(yb, ttnn.TILE_LAYOUT) if layout == "rm" else yb)
            exact2 = bool(torch.equal(got_b2[..., :rows, :], x))
            one_way = rtt / 2
            record(
                dict(
                    path="eth",
                    mechanism=f"socket_{transfer}",
                    mode=mode,
                    batch=batch,
                    layout=layout,
                    rows=rows,
                    logical_bytes=logical,
                    moved_bytes=moved,
                    connections=nconn,
                    fifo_bytes=fifo,
                    latency_us=round(one_way * 1e6, 2),
                    rtt_us=round(rtt * 1e6, 2),
                    stream_us_per_transfer=round(smed / K * 1e6, 2),
                    bw_logical_GBps=round(logical / one_way / 1e9, 2),
                    bw_moved_stream_GBps=round(moved / (smed / K) / 1e9, 2),
                    bitexact=exact and exact2,
                    in_trace=True,
                    k=K,
                    reps=a.reps,
                    replay_s=[round(t, 6) for t in ts],
                    chips="4-5 (card2, on-board)"
                    if os.environ.get("TT_VISIBLE_DEVICES") == "4,5"
                    else os.environ.get("TT_VISIBLE_DEVICES"),
                    method="ping-pong trace (RTT/2) + stream trace, wall clock of replay",
                )
            )
            for t in keep + tmp:
                if isinstance(t, ttnn.Tensor) and t is not xa and t is not yb and t is not ya:
                    try:
                        ttnn.deallocate(t)
                    except Exception:  # noqa: BLE001
                        pass
            for t in (xa, yb, ya):
                ttnn.deallocate(t)
        except Exception as e:  # noqa: BLE001
            record(
                dict(
                    path="eth",
                    mechanism=f"socket_{transfer}",
                    mode=mode,
                    batch=batch,
                    layout=layout,
                    connections=nconn,
                    fifo_bytes=fifo,
                    status="error",
                    error=repr(e)[:400],
                )
            )
            raise


# -------------------------------------------------------------------------------------------------------- parent
def run_parent(a):
    set_fabric(a)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=a.trace_region)
    c0, c1 = ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)
    try:
        for mech in a.mechanisms.split(","):
            for mode, batch, layout, rows in cases(a.cases):
                parent_case(a, mesh, c0, c1, mech, mode, batch, layout, rows)
    finally:
        ttnn.synchronize_device(mesh)
        quiesce(mesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def p2p_spec_fn():
    for path in ("ttnn._ttnn.operations.point_to_point", "ttnn._ttnn.operations.ccl", "ttnn"):
        mod = eval(path)
        if hasattr(mod, "p2p_compute_intermediate_tensor_spec"):
            return mod.p2p_compute_intermediate_tensor_spec
    raise AttributeError("p2p_compute_intermediate_tensor_spec not exposed")


def parent_case(a, mesh, c0, c1, mech, mode, batch, layout, rows):
    logical, moved = bytes_of(rows, layout)
    if mech == "all_gather" and layout == "rm":
        # all_gather's multicast kernels keep chunk sizes in uint16 (multicast_common.hpp:48): a 64 KiB ROW_MAJOR page
        # (16384 fp32) does not compile (-Werror=overflow), logs/bench/parent.log 2026-10-17T21:40:26Z
        record(
            dict(
                path="eth",
                mechanism=mech,
                mode=mode,
                batch=batch,
                layout=layout,
                status="unsupported",
                error="all_gather kernels: 64 KiB page overflows uint16 chunk size (multicast_common.hpp:48)",
            )
        )
        return
    x = payload(rows, seed=batch)
    lay = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    # mesh-sharded tensor: device (0,0) holds the payload, device (0,1) zeros
    full = torch.cat([x, torch.zeros_like(x)], dim=0)
    xm = ttnn.from_torch(
        full,
        dtype=ttnn.float32,
        layout=lay,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    try:
        if mech == "p2p":
            ym = ttnn.allocate_tensor_on_device(xm.spec, mesh)
            zm = ttnn.allocate_tensor_on_device(xm.spec, mesh)
            ib = ttnn.allocate_tensor_on_device(p2p_spec_fn()(xm, c0, c1, ttnn.Topology.Linear), mesh)

            def fwd():  # (input, sender_coord, receiver_coord): chip A -> chip B
                ttnn.point_to_point(xm, c0, c1, output_tensor=ym, intermediate_tensor=ib, topology=ttnn.Topology.Linear)

            def back():  # chip B -> chip A
                ttnn.point_to_point(ym, c1, c0, output_tensor=zm, intermediate_tensor=ib, topology=ttnn.Topology.Linear)

            fwd()
            back()
            ttnn.synchronize_device(mesh)
            yt = ttnn.to_torch(ym, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            zt = ttnn.to_torch(zm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            exact = bool(torch.equal(yt[1:2, :, :rows], x) and torch.equal(zt[0:1, :, :rows], x))
        elif mech == "all_gather":

            def fwd():
                return ttnn.all_gather(xm, dim=0, memory_config=DRAM, topology=ttnn.Topology.Linear)

            back = None
            out = fwd()
            ttnn.synchronize_device(mesh)
            ot = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            exact = bool(torch.equal(ot[2 + 0 : 2 + 1, :, :rows], x))  # device (0,1) received device (0,0)'s shard
            ttnn.deallocate(out)
        else:
            raise ValueError(mech)
        K = a.k_prefill if mode == "prefill" else a.k
        keep = []
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(K):
            r = fwd()
            if r is not None:
                keep.append(r)
            if back is not None:
                back()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)

        def run():
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)

        run()
        flush(mesh)
        signpost(f"HANDOFF {mech} {mode} B{batch} {layout} K{K}")
        med, ts = timed_replays(run, a.reps)
        signpost("HANDOFF_END")
        flush(mesh)
        ttnn.release_trace(mesh, tid)
        per = med / K
        one_way = per / 2 if mech == "p2p" else per  # p2p: ping-pong (2 hops per K); all_gather: one exchange per K
        record(
            dict(
                path="eth",
                mechanism=mech,
                mode=mode,
                batch=batch,
                layout=layout,
                rows=rows,
                logical_bytes=logical,
                moved_bytes=moved,
                latency_us=round(one_way * 1e6, 2),
                bw_logical_GBps=round(logical / one_way / 1e9, 2),
                bitexact=exact,
                in_trace=True,
                k=K,
                reps=a.reps,
                replay_s=[round(t, 6) for t in ts],
                chips=os.environ.get("TT_VISIBLE_DEVICES"),
                method=(
                    "ping-pong point_to_point in a parent-mesh trace (per K: 2 hops), RTT/2"
                    if mech == "p2p"
                    else "all_gather dim 0 over the 1x2 axis in a parent-mesh trace: both chips send their "
                    "payload (exchange), time per op"
                ),
                note="parent-mesh trace: cannot connect two per-chip (submesh) traces",
            )
        )
        for t in keep:
            ttnn.deallocate(t)
        if mech == "p2p":
            for t in (ym, zm, ib):
                ttnn.deallocate(t)
    except Exception as e:  # noqa: BLE001
        record(
            dict(path="eth", mechanism=mech, mode=mode, batch=batch, layout=layout, status="error", error=repr(e)[:400])
        )
        raise
    finally:
        ttnn.deallocate(xm)


# ---------------------------------------------------------------------------------------------------------- host
def run_host(a):
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=a.trace_region, num_command_queues=2)
    A = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    B = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
    d2h = h2d = None
    try:
        mechs = a.host_mechs.split(",")
        if "hd" in mechs:  # sockets allocate L1 on their cores: create them before any trace exists
            c0 = ttnn.MeshCoordinate(0, 0)
            d2h = ttnn.D2HSocket(A, ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(0, 3)), HD_D2H_FIFO)
            h2d = ttnn.H2DSocket(
                B, ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(0, 2)), ttnn.BufferType.L1, a.h2d_fifo, ttnn.H2DMode.HOST_PUSH
            )
            d2h.set_page_size(HD_PAGE)
            h2d.set_page_size(HD_PAGE)
            for mode, batch, layout, rows in cases(a.cases):
                if (mode == "prefill" and not a.hd_prefill) or layout == "rm":  # rm == tile here (logical rows)
                    continue
                hd_case(a, A, B, d2h, h2d, mode, batch, layout, rows)
        if "cq" in mechs:
            for mode, batch, layout, rows in cases(a.cases):
                host_case(a, A, B, mode, batch, layout, rows)
    finally:
        for s in (A, B):
            ttnn.synchronize_device(s)
        # the PCIe sockets must die before their mesh: ~D2HSocket after close_mesh_device segfaulted (2026-10-17 21:42Z)
        del d2h, h2d
        import gc

        gc.collect()
        quiesce(mesh)
        ttnn.close_mesh_device(mesh)


# D2H socket FIFO lives in pinned host memory; without an IOMMU (this host: identity groups) the KMD pins one 4 KiB
# page, so the FIFO is one page minus the bytes counter (models/experimental/deepseek_v4_flash/tt/model.py:160-176 of
# the dsv4 branch). Pages of 2 KiB = 512 fp32 (PCIe-aligned, divides every boundary tensor).
HD_D2H_FIFO = 4032
HD_PAGE = 2048


def hd_case(a, A, B, d2h, h2d, mode, batch, layout, rows):
    """chip A trace: [untilize ->] reshape to 2 KiB rows -> send_async_d2h; chip B trace: recv_async_h2d -> reshape
    [-> tilize]; host: D2HSocket.read_tensor then H2DSocket.write_tensor (pages relayed through host memory)."""
    logical, moved = bytes_of(rows, layout)
    x = payload(rows, seed=batch)
    xa = dev_tensor(x, A, "tile")
    n_el = rows * HC  # untilize keeps the logical rows: the PCIe sockets move B x 64 KiB for either layout
    prow = HD_PAGE // 4
    shape_pages = (1, 1, n_el // prow, prow)
    yb_rm = ttnn.from_torch(
        torch.zeros(shape_pages), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=B, memory_config=DRAM
    )
    host = torch.empty(shape_pages[2], prow, dtype=torch.float32)
    keep = []

    def a_send():
        rm = ttnn.to_layout(xa, ttnn.ROW_MAJOR_LAYOUT)
        pg = ttnn.reshape(rm, shape_pages)
        ttnn.experimental.send_async_d2h(pg, d2h)
        return [rm, pg]

    def b_recv():
        ttnn.experimental.recv_async_h2d(yb_rm, h2d)
        t = ttnn.reshape(yb_rm, (1, 1, n_el // HC, HC))
        return [t, ttnn.to_layout(t, ttnn.TILE_LAYOUT)]

    def relay():
        d2h.read_tensor(host)
        h2d.write_tensor(host)

    try:
        keep += a_send()
        outs = b_recv()
        relay()
        ttnn.synchronize_device(A)
        ttnn.synchronize_device(B)
        exact = bool(torch.equal(to_host(outs[-1])[..., :rows, :], x))
        ta = ttnn.begin_trace_capture(A, cq_id=0)
        keep += a_send()
        ttnn.end_trace_capture(A, ta, cq_id=0)
        tb = ttnn.begin_trace_capture(B, cq_id=0)
        outs_t = b_recv()
        ttnn.end_trace_capture(B, tb, cq_id=0)

        def hop():
            ttnn.execute_trace(A, ta, cq_id=0, blocking=False)
            ttnn.execute_trace(B, tb, cq_id=0, blocking=False)
            relay()
            ttnn.synchronize_device(B)

        hop()
        ex2 = bool(torch.equal(to_host(outs_t[-1])[..., :rows, :], x))
        reps = a.reps if mode == "decode" else 2
        med, ts = timed_replays(hop, reps)
        ttnn.synchronize_device(A)
        ttnn.release_trace(A, ta)
        ttnn.release_trace(B, tb)
        record(
            dict(
                path="host",
                mechanism="pcie_socket_d2h_h2d",
                mode=mode,
                batch=batch,
                layout=layout,
                rows=rows,
                logical_bytes=logical,
                moved_bytes=n_el * 4,
                latency_us=round(med * 1e6, 2),
                bw_logical_GBps=round(logical / med / 1e9, 3),
                bitexact=exact and ex2,
                in_trace=True,
                reps=reps,
                replay_s=[round(t, 6) for t in ts],
                d2h_fifo_bytes=HD_D2H_FIFO,
                h2d_fifo_bytes=a.h2d_fifo,
                page_bytes=HD_PAGE,
                chips=os.environ.get("TT_VISIBLE_DEVICES"),
                method="send_async_d2h in chip A's trace, recv_async_h2d (HOST_PUSH) in chip B's trace, host "
                "read_tensor + write_tensor; incl. untilize/tilize; wall clock per hop",
            )
        )
    except Exception as e:  # noqa: BLE001
        record(
            dict(
                path="host",
                mechanism="pcie_socket_d2h_h2d",
                mode=mode,
                batch=batch,
                layout=layout,
                status="error",
                error=repr(e)[:400],
            )
        )
        raise
    finally:
        for t in keep:
            try:
                ttnn.deallocate(t)
            except Exception:  # noqa: BLE001
                pass
        ttnn.deallocate(xa)


def host_case(a, A, B, mode, batch, layout, rows):
    logical, moved = bytes_of(rows, layout)
    x = payload(rows, seed=batch)
    xa = dev_tensor(x, A, layout)  # producer output on chip A (persistent)
    yb = ttnn.allocate_tensor_on_device(xa.spec, B)  # consumer input on chip B (persistent trace input)
    hb = ttnn.allocate_tensor_on_host(
        xa.spec, A
    )  # host staging buffer (reused => pinned-cache hit where pinning exists)
    try:
        # blocking hop: D2H (CQ0 of A) then H2D (CQ0 of B), host waits for completion of both
        def hop_blocking():
            ttnn.copy_device_to_host_tensor(xa, hb, blocking=True)
            ttnn.copy_host_to_device_tensor(hb, yb)
            ttnn.synchronize_device(B)

        hop_blocking()
        exact = bool(torch.equal(to_host(yb)[..., :rows, :], x))
        med, ts = timed_replays(hop_blocking, a.reps)
        record(
            dict(
                path="host",
                mechanism="cq_d2h_h2d_blocking",
                mode=mode,
                batch=batch,
                layout=layout,
                rows=rows,
                logical_bytes=logical,
                moved_bytes=moved,
                latency_us=round(med * 1e6, 2),
                bw_logical_GBps=round(logical / med / 1e9, 2),
                bitexact=exact,
                in_trace=False,
                trace_compatible=True,
                reps=a.reps,
                replay_s=[round(t, 6) for t in ts],
                chips=os.environ.get("TT_VISIBLE_DEVICES"),
                method="copy_device_to_host_tensor(blocking) + copy_host_to_device_tensor + synchronize(B)",
            )
        )
        # split timing of the two halves
        d2h, h2d = [], []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            ttnn.copy_device_to_host_tensor(xa, hb, blocking=True)
            t1 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(hb, yb)
            ttnn.synchronize_device(B)
            t2 = time.perf_counter()
            d2h.append(t1 - t0)
            h2d.append(t2 - t1)
        record(
            dict(
                path="host",
                mechanism="cq_d2h_h2d_split",
                mode=mode,
                batch=batch,
                layout=layout,
                rows=rows,
                logical_bytes=logical,
                moved_bytes=moved,
                latency_us=round((statistics.median(d2h) + statistics.median(h2d)) * 1e6, 2),
                d2h_us=round(statistics.median(d2h) * 1e6, 2),
                h2d_us=round(statistics.median(h2d) * 1e6, 2),
                d2h_GBps=round(moved / statistics.median(d2h) / 1e9, 2),
                h2d_GBps=round(moved / statistics.median(h2d) / 1e9, 2),
                in_trace=False,
                trace_compatible=True,
                reps=a.reps,
                chips=os.environ.get("TT_VISIBLE_DEVICES"),
                method="per-direction medians",
            )
        )
        # overlapped: S row-slices, D2H of slice i+1 (chip A, CQ1) overlaps H2D of slice i (chip B, CQ1)
        S = a.slices if mode == "prefill" else 1
        if S > 1:
            per = rows // S
            xs = [dev_tensor(x[..., i * per : (i + 1) * per, :], A, layout) for i in range(S)]
            ys = [ttnn.allocate_tensor_on_device(xs[0].spec, B) for _ in range(S)]
            hs = [ttnn.allocate_tensor_on_host(xs[0].spec, A) for _ in range(S)]

            def hop_overlap():
                evs = []
                for i in range(S):
                    ttnn.copy_device_to_host_tensor(xs[i], hs[i], blocking=False, cq_id=1)
                    evs.append(ttnn.record_event(A, cq_id=1))
                for i in range(S):
                    ttnn.event_synchronize(evs[i])  # slice i is on the host
                    ttnn.copy_host_to_device_tensor(hs[i], ys[i], cq_id=1)
                ev_b = ttnn.record_event(B, cq_id=1)
                ttnn.wait_for_event(0, ev_b)  # B's compute queue (trace replay) ordered after the writes
                ttnn.synchronize_device(B)

            hop_overlap()
            got = torch.cat([to_host(t) for t in ys], dim=-2)
            ex2 = bool(torch.equal(got[..., :rows, :], x))
            med2, ts2 = timed_replays(hop_overlap, a.reps)
            record(
                dict(
                    path="host",
                    mechanism=f"cq_d2h_h2d_overlap_{S}slices",
                    mode=mode,
                    batch=batch,
                    layout=layout,
                    rows=rows,
                    logical_bytes=logical,
                    moved_bytes=moved,
                    latency_us=round(med2 * 1e6, 2),
                    bw_logical_GBps=round(logical / med2 / 1e9, 2),
                    bitexact=ex2,
                    in_trace=False,
                    trace_compatible=True,
                    reps=a.reps,
                    replay_s=[round(t, 6) for t in ts2],
                    chips=os.environ.get("TT_VISIBLE_DEVICES"),
                    method="CQ1 D2H non-blocking per slice + events; host event_synchronize; CQ1 H2D; CQ0 wait_for_event",
                )
            )
            for t in xs + ys:
                ttnn.deallocate(t)
    except Exception as e:  # noqa: BLE001
        record(
            dict(
                path="host",
                mechanism="cq_d2h_h2d",
                mode=mode,
                batch=batch,
                layout=layout,
                status="error",
                error=repr(e)[:400],
            )
        )
        raise
    finally:
        ttnn.deallocate(xa)
        ttnn.deallocate(yb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sockets", "parent", "host"])
    ap.add_argument("--cases", default="", help="comma list of mode:batch:layout, e.g. decode:1:tile,prefill:1:tile")
    ap.add_argument("--transfers", default="direct,fifo")
    ap.add_argument("--connections", default="1,2")
    ap.add_argument("--fifos", default="65536,262144")
    ap.add_argument("--mechanisms", default="p2p,all_gather")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--k-prefill", type=int, default=2)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--slices", type=int, default=8)
    ap.add_argument("--payload", type=int, default=0, help="fabric max packet payload bytes (0 = default 4352)")
    ap.add_argument("--host-mechs", default="cq,hd")
    ap.add_argument("--h2d-fifo", type=int, default=65536)
    ap.add_argument(
        "--hd-prefill", action="store_true", help="also stream the 128 MiB prefill chunk over the PCIe sockets"
    )
    ap.add_argument("--trace-region", type=int, default=128 * 1024 * 1024)
    a = ap.parse_args()
    a.cases = set(filter(None, a.cases.split(",")))
    PAYLOAD["v"] = a.payload
    {"sockets": run_sockets, "parent": run_parent, "host": run_host}[a.mode](a)


if __name__ == "__main__":
    main()
