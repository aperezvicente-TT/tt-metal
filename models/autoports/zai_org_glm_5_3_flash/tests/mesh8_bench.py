# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 04a: 8-chip mesh checks on host-8chip's 2x4 Ethernet mesh (doc/hardware.md §3, 2026-10-17). One mode per
process; needs every chip it opens (run inside the all-chip window, `tt-lock run all`).

  pair  --chips a,b      FABRIC_1D 1x2 mesh over one Ethernet edge (pair descriptor): open, socket ping-pong of the
                         decode hand-off tensor (fp32 [1,1,32,16384] TILE = 2 MiB), bit-exact check, close.
  coll  --mgd 2x4|ring8  open the 8-chip mesh (FABRIC_1D for 2x4, FABRIC_1D_RING for the 1x8 ring), record the
                         coordinate -> chip map, time all_gather / reduce_scatter / all_reduce per cluster axis in traces
                         (K ops captured, replayed), check against torch.
  hops                   2x4 mesh + FABRIC_1D, 8 1x1 submeshes; socket (direct) hops along the pipeline ring
                         0-2-5-6-7-4-3-1-0: per-edge ping-pong latency for the decode hand-off (TILE 2 MiB and
                         ROW_MAJOR B=1 64 KiB) and one full ring pass (8 chained hops, one trace per chip).
Rows -> $GLM53_E4A/mesh8_bench.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

import ttnn

E = os.environ.get(
    "GLM53_E4A", "/opt/glm53/kit/doc/ttnn/04a-multichip-handoff-prework"
)
MESH_DIR = os.environ.get("GLM53_MESH_DIR", "/opt/glm53/tt-metal-s4a/models/autoports/zai_org_glm_5_3_flash/mesh")
RING = [0, 2, 5, 6, 7, 4, 3, 1]
HC = 16384
DRAM = ttnn.DRAM_MEMORY_CONFIG


def quiesce(mesh):
    """Parent mesh with submeshes: tt-metal refuses to close a parent whose CQ is "in use" by a child (TT_THROW "cq ID 0
    is in use by child submesh") -- quiesce_devices() quiesces the submeshes and clears the flags (needed for the
    device profiler's final dump, 2026-10-17 22:29Z)."""
    try:
        mesh.quiesce_devices()
    except Exception as e:  # noqa: BLE001
        print(f"quiesce_devices failed: {e!r}", flush=True)


def record(row):
    row = {
        "fabric_payload_bytes": KNOBS["payload"] or 4352,
        "socket_connections": KNOBS["connections"],
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **row,
        "visible": os.environ.get("TT_VISIBLE_DEVICES"),
        "lock": os.environ.get("TT_LOCK_GROUP"),
        "profiler": bool(os.environ.get("TT_METAL_DEVICE_PROFILER")),
    }
    with open(os.path.join(E, "mesh8_bench.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row), flush=True)


def med(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def chip_map(mesh):
    """mesh coordinate -> process device id (= chip index when TT_VISIBLE_DEVICES=0..7)."""
    ids = list(mesh.get_device_ids())
    r, c = mesh.shape[0], mesh.shape[1]
    vis = [int(x) for x in os.environ.get("TT_VISIBLE_DEVICES", ",".join(map(str, range(r * c)))).split(",")]
    return {(i // c, i % c): (vis[ids[i]] if ids[i] < len(vis) else ids[i]) for i in range(len(ids))}


KNOBS = {"connections": 1, "payload": 0}


def set_fab(cfg_name):
    """Fabric config (by FabricConfig name) before the mesh opens, with the optional packet payload override."""
    cfg = getattr(ttnn.FabricConfig, cfg_name)
    if KNOBS["payload"]:
        rc = ttnn.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = KNOBS["payload"]
        ttnn.set_fabric_config(cfg, router_config=rc)
    else:
        ttnn.set_fabric_config(cfg)


def pair_link(src, dst, row, connections=None):
    connections = connections or KNOBS["connections"]
    c0 = ttnn.MeshCoordinate(0, 0)
    conns = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(i, row)), ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(i, row + 1))
        )
        for i in range(connections)
    ]
    return ttnn.create_socket_pair(src, dst, ttnn.SocketConfig(conns, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 128)))


def pingpong(A, B, x, layout, k, reps, row):
    """A -> B -> A socket (direct) ping-pong inside per-chip traces; returns (one-way us, bitexact)."""
    s_ab, r_ab = pair_link(A, B, row)
    s_ba, r_ba = pair_link(B, A, row + 2)
    lay = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
    xa = ttnn.from_torch(x, dtype=ttnn.float32, layout=lay, device=A, memory_config=DRAM)
    yb = ttnn.allocate_tensor_on_device(xa.spec, B)
    ya = ttnn.allocate_tensor_on_device(xa.spec, A)

    def once():
        ttnn.experimental.send_direct_async(xa, s_ab)
        ttnn.experimental.recv_direct_async(yb, r_ab)
        ttnn.experimental.send_direct_async(yb, s_ba)
        ttnn.experimental.recv_direct_async(ya, r_ba)

    once()
    ttnn.synchronize_device(A)
    ttnn.synchronize_device(B)
    exact = bool(torch.equal(ttnn.to_torch(ya).float()[..., : x.shape[-2], :], x))
    ta = ttnn.begin_trace_capture(A, cq_id=0)
    tb = ttnn.begin_trace_capture(B, cq_id=0)
    for _ in range(k):
        once()
    ttnn.end_trace_capture(A, ta, cq_id=0)
    ttnn.end_trace_capture(B, tb, cq_id=0)

    def run():
        ttnn.execute_trace(A, ta, cq_id=0, blocking=False)
        ttnn.execute_trace(B, tb, cq_id=0, blocking=False)
        ttnn.synchronize_device(A)
        ttnn.synchronize_device(B)

    run()
    t = med(run, reps) / k / 2
    ttnn.release_trace(A, ta)
    ttnn.release_trace(B, tb)
    for tt in (xa, yb, ya):
        ttnn.deallocate(tt)
    return t * 1e6, exact


def run_pair(a):
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(MESH_DIR, "glm_pair_1x2_mesh_graph_descriptor.textproto")
    set_fab("FABRIC_1D")
    t0 = time.perf_counter()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=64 << 20)
    open_s = time.perf_counter() - t0
    try:
        A = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        B = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
        x = torch.randn(1, 1, 32, HC)
        us, exact = pingpong(A, B, x, "tile", a.k, a.reps, 0)
        record(
            dict(
                mode="pair",
                chips=os.environ.get("TT_VISIBLE_DEVICES"),
                status="pass" if exact else "fail",
                open_s=round(open_s, 2),
                coord_map={str(k): v for k, v in chip_map(mesh).items()},
                handoff="socket_direct",
                payload="fp32 [1,1,32,16384] TILE 2 MiB",
                latency_us=round(us, 2),
                bitexact=exact,
            )
        )
        for s in (A, B):
            ttnn.synchronize_device(s)
    finally:
        quiesce(mesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def run_coll(a):
    ring = a.mgd == "ring8"
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(
        MESH_DIR,
        "glm8_1x8_ring_mesh_graph_descriptor.textproto" if ring else "glm8_2x4_mesh_graph_descriptor.textproto",
    )
    set_fab("FABRIC_1D_RING" if ring else "FABRIC_1D")
    shape = (1, 8) if ring else (2, 4)
    t0 = time.perf_counter()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=256 << 20)
    open_s = time.perf_counter() - t0
    try:
        cmap = chip_map(mesh)
        record(
            dict(
                mode="coll_open",
                mgd=a.mgd,
                shape=list(shape),
                status="pass",
                open_s=round(open_s, 2),
                coord_map={str(k): v for k, v in cmap.items()},
            )
        )
        topo = ttnn.Topology.Ring if ring else ttnn.Topology.Linear
        axes = [1] if ring else [0, 1]
        payloads = [
            ("bf16_32x4096_256KiB", ttnn.bfloat16, 4096),
            ("bf16_32x16384_1MiB", ttnn.bfloat16, 16384),
            ("fp32_32x16384_2MiB", ttnn.float32, 16384),
        ]
        n = shape[0] * shape[1]
        for axis in axes:
            for name, dt, w in payloads:
                xt = torch.randn(n, 1, 32, w).to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32)
                x = ttnn.from_torch(
                    xt,
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                )
                for op in ("all_gather", "reduce_scatter", "all_reduce"):

                    def fn():
                        if op == "all_gather":
                            return ttnn.all_gather(x, dim=3, cluster_axis=axis, memory_config=DRAM, topology=topo)
                        if op == "reduce_scatter":
                            return ttnn.reduce_scatter(x, dim=3, cluster_axis=axis, memory_config=DRAM, topology=topo)
                        return ttnn.all_reduce(x, cluster_axis=axis, memory_config=DRAM, topology=topo)

                    try:
                        o = fn()
                        ttnn.synchronize_device(mesh)
                        ok, err, scale = check_coll(op, o, xt, mesh, shape, axis)
                        ttnn.deallocate(o)
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        outs = [fn() for _ in range(a.k)]
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)

                        def run():
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)

                        run()
                        t = med(run, a.reps) / a.k
                        ttnn.release_trace(mesh, tid)
                        for oo in outs:
                            ttnn.deallocate(oo)
                        record(
                            dict(
                                mode="coll",
                                mgd=a.mgd,
                                axis=axis,
                                group_size=shape[axis],
                                op=op,
                                payload=name,
                                topology=str(topo),
                                latency_us=round(t * 1e6, 2),
                                check=ok,
                                max_abs_err=err,
                                max_abs_ref=scale,
                                status="pass" if ok else "fail",
                                k=a.k,
                                reps=a.reps,
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        record(
                            dict(
                                mode="coll",
                                mgd=a.mgd,
                                axis=axis,
                                op=op,
                                payload=name,
                                status="error",
                                error=repr(e)[:300],
                            )
                        )
                ttnn.deallocate(x)
    finally:
        ttnn.synchronize_device(mesh)
        quiesce(mesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def check_coll(op, o, xt, mesh, shape, axis):
    """Compare against torch per device (groups along `axis`, row-major device order). Returns (ok, max |err|,
    max |ref|): all_gather must be bit-exact; reductions within 0.05 (bf16) / 1e-3 (fp32) of max(1, max |ref|)."""
    got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    R, C = shape
    x = xt.float()
    groups = (
        [[r * C + c for c in range(C)] for r in range(R)]
        if axis == 1
        else [[r * C + c for r in range(R)] for c in range(C)]
    )
    tol = 0.05 if xt.dtype == torch.bfloat16 else 1e-3
    ok, err, scale = True, 0.0, 0.0
    for g in groups:
        if op == "all_gather":
            ref = torch.cat([x[d] for d in g], dim=-1)
        else:
            ref = sum(x[d] for d in g)
        for k, d in enumerate(g):
            r = ref if op != "reduce_scatter" else torch.chunk(ref, len(g), dim=-1)[k]
            gd = got[d * (got.shape[0] // (R * C)) : (d + 1) * (got.shape[0] // (R * C))].reshape(r.shape)
            e = float((gd - r).abs().max())
            err, scale = max(err, e), max(scale, float(r.abs().max()))
            if op == "all_gather":
                ok &= bool(torch.equal(gd, r))
            elif e > tol * max(1.0, float(r.abs().max())):
                ok = False
    return ok, err, scale


MGDS = {
    "2x4": ("glm8_2x4_mesh_graph_descriptor.textproto", (2, 4), "FABRIC_1D"),
    "ring8": ("glm8_1x8_ring_mesh_graph_descriptor.textproto", (1, 8), "FABRIC_1D_RING"),
    "line8": ("glm8_1x8_line_mesh_graph_descriptor.textproto", (1, 8), "FABRIC_1D"),
}


def run_hops(a):
    mgd, shape, fab = MGDS[a.mgd]
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(MESH_DIR, mgd)
    set_fab(fab)
    t0 = time.perf_counter()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=128 << 20)
    open_s = time.perf_counter() - t0
    try:
        cmap = chip_map(mesh)
        coord_of = {chip: coord for coord, chip in cmap.items()}
        subs = {
            chip: mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(*coord))
            for chip, coord in coord_of.items()
        }
        record(
            dict(
                mode="hops_open",
                mgd=a.mgd,
                fabric=fab,
                status="pass",
                open_s=round(open_s, 2),
                coord_map={str(k): v for k, v in cmap.items()},
            )
        )
        for layout, rows in (("tile", 32), ("rm", 1)):
            x = torch.randn(1, 1, rows, HC)
            if a.mgd == "2x4":  # every hop of the ring 0-2-5-6-7-4-3-1-(0) is a row or column neighbour here
                edges = [(RING[i], RING[(i + 1) % len(RING)]) for i in range(len(RING))]
            else:  # 1x8 line / ring: consecutive coordinates of the embedding the runtime picked (no wrap edge:
                # sockets find no forwarding direction across the FABRIC_1D_RING wrap, 2026-10-17 22:16:56Z)
                order = [cmap[(0, j)] for j in range(8)]
                edges = list(zip(order[:-1], order[1:]))
            for i, (s, d) in enumerate(edges):
                try:
                    us, exact = pingpong(subs[s], subs[d], x, layout, a.k, a.reps, (2 * i) % 6)
                    record(
                        dict(
                            mode="hop",
                            mgd=a.mgd,
                            edge=f"{s}-{d}",
                            coords=[list(coord_of[s]), list(coord_of[d])],
                            layout=layout,
                            rows=rows,
                            bytes=(32 if layout == "tile" else rows) * HC * 4,
                            latency_us=round(us, 2),
                            bitexact=exact,
                            status="pass" if exact else "fail",
                        )
                    )
                except (
                    Exception
                ) as e:  # noqa: BLE001  record, then stop: a failed socket op can leave the fabric wedged
                    record(
                        dict(mode="hop", mgd=a.mgd, edge=f"{s}-{d}", layout=layout, status="error", error=repr(e)[:300])
                    )
                    raise
            if a.mgd != "2x4":
                continue
            # full ring pass: chip RING[0] -> ... -> RING[7] -> RING[0], one trace per chip, K passes
            links = [pair_link(subs[RING[i]], subs[RING[(i + 1) % 8]], 6) for i in range(8)]
            lay = ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT
            bufs = {
                c: ttnn.from_torch(
                    x if c == RING[0] else torch.zeros_like(x),
                    dtype=ttnn.float32,
                    layout=lay,
                    device=subs[c],
                    memory_config=DRAM,
                )
                for c in RING
            }
            back = ttnn.allocate_tensor_on_device(bufs[RING[0]].spec, subs[RING[0]])

            def ring_pass():
                for i in range(8):
                    s, d = RING[i], RING[(i + 1) % 8]
                    ttnn.experimental.send_direct_async(bufs[s], links[i][0])
                    ttnn.experimental.recv_direct_async(back if d == RING[0] else bufs[d], links[i][1])

            ring_pass()
            for c in RING:
                ttnn.synchronize_device(subs[c])
            exact = bool(torch.equal(ttnn.to_torch(back).float()[..., :rows, :], x))
            tids = {}
            for c in RING:
                tids[c] = ttnn.begin_trace_capture(subs[c], cq_id=0)
            for _ in range(a.k):
                ring_pass()
            for c in RING:
                ttnn.end_trace_capture(subs[c], tids[c], cq_id=0)

            def run():
                for c in RING:
                    ttnn.execute_trace(subs[c], tids[c], cq_id=0, blocking=False)
                for c in RING:
                    ttnn.synchronize_device(subs[c])

            run()
            t = med(run, a.reps) / a.k
            for c in RING:
                ttnn.release_trace(subs[c], tids[c])
            record(
                dict(
                    mode="ring_pass",
                    mgd=a.mgd,
                    layout=layout,
                    rows=rows,
                    hops=8,
                    ring=RING,
                    latency_us=round(t * 1e6, 2),
                    per_hop_us=round(t * 1e6 / 8, 2),
                    bitexact=exact,
                    status="pass" if exact else "fail",
                )
            )
        for s in subs.values():
            ttnn.synchronize_device(s)
    finally:
        quiesce(mesh)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pair", "coll", "hops"])
    ap.add_argument("--mgd", default="2x4", choices=["2x4", "ring8", "line8"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--connections", type=int, default=1)
    ap.add_argument("--payload", type=int, default=0, help="fabric max packet payload bytes (0 = default 4352)")
    a = ap.parse_args()
    KNOBS["connections"], KNOBS["payload"] = a.connections, a.payload
    {"pair": run_pair, "coll": run_coll, "hops": run_hops}[a.mode](a)


if __name__ == "__main__":
    main()
