# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c (copy of Stage 14 s14common.py) shared helpers: evidence rows (perf.jsonl / pcc.jsonl / tests.jsonl), mesh open with fabric, traced timing
(warm = the program's compile launch, then capture + replays; TT_METAL_TRACE_ALLOC_TRACKING=1 from env.sh), PCC."""

from __future__ import annotations

import json
import os
import statistics
import time

E = os.environ.get("GLM53_E14", "/opt/glm53/kit/doc/ttnn/14c-batch-scaling")
OUT = os.environ.get("GLM53_OUT_DIR", E)
MOCK = os.environ.get("GLM53_MOCK") == "1"
MESH_DIR = os.environ.get(
    "GLM53_MESH_DIR", "/opt/glm53/tt-metal-s14c/models/autoports/zai_org_glm_5_3_flash/mesh"
)
MGDS = {  # name -> (descriptor file, mesh shape); "single" = one chip (TT_VISIBLE_DEVICES), no fabric
    "single": ("glm_1x1_mesh_graph_descriptor.textproto", (1, 1)),  # opened WITHOUT fabric
    "pair": ("glm_pair_1x2_mesh_graph_descriptor.textproto", (1, 2)),
    "2x4": ("glm8_2x4_mesh_graph_descriptor.textproto", (2, 4)),
    "line8": ("glm8_1x8_line_mesh_graph_descriptor.textproto", (1, 8)),
    "3x2": ("glm6_3x2_mesh_graph_descriptor.textproto", (3, 2)),  # host-p300 all three p300a (rows = cards)
}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ctx():
    return dict(
        visible=os.environ.get("TT_VISIBLE_DEVICES"), lock=os.environ.get("TT_LOCK_GROUP"),
        watcher=bool(os.environ.get("TT_METAL_WATCHER")), trace_alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
        mock=MOCK, code=os.environ.get("GLM53_CODE_SHA"),
    )  # fmt: skip


def record(kind, row):
    """kind: perf | pcc | tests. Every perf / pcc row also lands in tests.jsonl with its status."""
    row = {"time_utc": now(), **row, **_ctx()}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{kind}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    if kind != "tests":
        t = {k: row[k] for k in ("time_utc", "test", "status", "milestone", "us", "pcc") if k in row}
        with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
            f.write(json.dumps({**t, "kind": kind, "mock": MOCK}) + "\n")
    print("ROW", kind, json.dumps(row), flush=True)
    return row


def open_mesh(ttnn, mgd, *, trace_region=64 << 20, worker_l1_size=0, payload=0, fabric="FABRIC_1D"):
    desc, shape = MGDS[mgd]
    extra = dict(worker_l1_size=worker_l1_size) if worker_l1_size else {}
    if mgd == "single":  # one chip (TT_VISIBLE_DEVICES), no fabric; the CUSTOM cluster still needs a descriptor
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(MESH_DIR, desc)
        t0 = time.perf_counter()
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=trace_region, **extra)
        return mesh, time.perf_counter() - t0
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = os.path.join(MESH_DIR, desc)
    cfg = getattr(ttnn.FabricConfig, fabric)
    if payload:
        rc = ttnn.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = payload
        ttnn.set_fabric_config(cfg, router_config=rc)
    else:
        ttnn.set_fabric_config(cfg)
    t0 = time.perf_counter()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=trace_region, **extra)
    return mesh, time.perf_counter() - t0


def close_mesh(ttnn, mesh, fabric=True):
    try:
        ttnn.synchronize_device(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def chip_map(mesh):
    """mesh coordinate (r, c) -> physical chip index (TT_VISIBLE_DEVICES position -> chip)."""
    ids = list(mesh.get_device_ids())
    r, c = mesh.shape[0], mesh.shape[1]
    vis = [int(x) for x in os.environ.get("TT_VISIBLE_DEVICES", ",".join(map(str, range(r * c)))).split(",")]
    return {(i // c, i % c): (vis[ids[i]] if ids[i] < len(vis) else ids[i]) for i in range(len(ids))}


class Traced:
    """fn() captured ONCE in a trace (after its compile launch, which is the only eager launch the dev-loop rule
    allows) and replayed: `replays(n)` -> per-replay host wall us (blocking replay = launch + device time + sync)."""

    def __init__(self, ttnn, mesh, fn, warm=True):
        self.ttnn, self.mesh, self.fn = ttnn, mesh, fn
        if warm:
            fn()  # compile / warm-up launch (never timed)
            ttnn.synchronize_device(mesh)
        self.tid = None
        if MOCK:
            return
        self.tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            self.out = fn()
        finally:
            ttnn.end_trace_capture(mesh, self.tid, cq_id=0)

    def replay(self):
        if self.tid is None:
            self.out = self.fn()
            self.ttnn.synchronize_device(self.mesh)
            return
        self.ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=True)

    def replays(self, n):
        d = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.replay()
            d.append((time.perf_counter() - t0) * 1e6)
        return d

    def batch(self, n):
        """n replays back to back (non-blocking), one sync: mean us per replay (launch overlap amortised)."""
        if self.tid is None:
            return None
        t0 = time.perf_counter()
        for _ in range(n):
            self.ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
        self.ttnn.synchronize_device(self.mesh)
        return (time.perf_counter() - t0) * 1e6 / n

    def release(self):
        if self.tid is not None:
            self.ttnn.release_trace(self.mesh, self.tid)
            self.tid = None


def stats(us):
    s = sorted(us)
    n = len(s)
    if not n:
        return {}
    return dict(n=n, median_us=round(statistics.median(s), 2), min_us=round(s[0], 2), max_us=round(s[-1], 2),
                p90_us=round(s[min(n - 1, int(0.9 * n))], 2), mean_us=round(sum(s) / n, 2))  # fmt: skip


def pcc(a, b):
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float((a @ b).item() / d) if d else float("nan")
