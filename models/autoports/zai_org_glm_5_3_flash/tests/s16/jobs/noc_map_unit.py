# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 mock or a device): the logical -> NoC 0 map of every tensix column 0..12 x row 0..9 on every
chip (HANG static check, program 2026-10-23 21:02:25Z: the dispatch cores sit at logical (11, 0) / (11, 1); a NoC
multicast rectangle spans every physical core between its corners, so a rectangle over the 11 compute columns covers a
dispatch core whenever that core's NoC x lies between the workers' NoC x). No program runs."""
import ttnn

mesh, args = MESH, ARGS  # noqa: F821
g = mesh.compute_with_storage_grid_size()
gx, gy = int(g.x), int(g.y)
row = dict(compute_grid=[gx, gy], chips={})
devs = [mesh]  # the mesh's translated (virtual) worker coordinates: the same on every chip
for i, d in enumerate(devs):
    m = {}
    for x in range(gx + 2):
        for y in range(gy):
            try:
                w = d.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                m[f"{x},{y}"] = [int(w.x), int(w.y)]
            except Exception as e:  # noqa: BLE001
                m[f"{x},{y}"] = str(type(e).__name__)
    wx = sorted({v[0] for k, v in m.items() if isinstance(v, list) and int(k.split(",")[0]) < gx})
    wy = sorted({v[1] for k, v in m.items() if isinstance(v, list) and int(k.split(",")[0]) < gx})
    disp = {k: v for k, v in m.items() if int(k.split(",")[0]) >= gx}
    inside = {k: v for k, v in disp.items() if isinstance(v, list) and wx[0] <= v[0] <= wx[-1] and wy[0] <= v[1] <= wy[-1]}
    row["chips"]["mesh"] = dict(worker_noc_x=wx, worker_noc_y=wy, beyond_grid=disp, inside_worker_bbox=inside)
    print(f"NOCMAP mesh: worker NoC x {wx} y {wy}; logical x >= {gx}: {disp}; INSIDE the worker bbox: {inside}",
          flush=True)  # fmt: skip
RESULT = row
