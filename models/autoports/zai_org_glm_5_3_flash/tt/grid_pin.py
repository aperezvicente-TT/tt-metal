# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM53_GRID_X / GLM53_GRID_Y: pin the compute grid the construction sees (program 2026-10-27 02:12:31Z / 02:12:56Z).

The Galaxy's Blackhole chips report compute_with_storage_grid_size [12, 10]; host-8chip's p300 chips report 11 x 10. The
construction reads the grid in ~65 places (the served PMK placement pmk_stack.py / pmk_layer.py
`Placement(gx=grid.x, gy=grid.y)`, the op path's matmul / shard program configs). A 12-column grid gives a different
PMK placement (the W4 run died on "the scan body runs on every MM-only core") and a different image map. Pinning the
grid to host-8chip's makes the image map, the traces and every bitwise unit carry across hosts.

install() wraps compute_with_storage_grid_size on the ttnn device classes so that every Python caller sees
min(device grid, (GLM53_GRID_X, GLM53_GRID_Y)). It is idempotent, and a no-op when neither variable is set (the served
path is unchanged on host-8chip / host-p300). Ops that size their program configs INSIDE C++ from the device's own grid
are not affected. Those programs do not enter the DRAM image map. The map sha after the pin is the check.
"""

import os

_PINNED_ATTR = "_glm53_grid_pin"


def pin() -> tuple:
    """(x, y) caps from the environment; None for an unset axis"""
    gx = os.environ.get("GLM53_GRID_X", "").strip()
    gy = os.environ.get("GLM53_GRID_Y", "").strip()
    return (int(gx) if gx else None, int(gy) if gy else None)


def _classes(ttnn):
    out = []
    for name in ("MeshDevice", "Device", "IDevice"):
        c = getattr(ttnn, name, None)
        if c is not None and c not in out and hasattr(c, "compute_with_storage_grid_size"):
            out.append(c)
    return out


def install() -> bool:
    """wrap the grid getter once per class when a pin is set; returns True when a pin is active"""
    cx, cy = pin()
    if cx is None and cy is None:
        return False
    import ttnn  # noqa: PLC0415

    for cls in _classes(ttnn):
        orig = cls.compute_with_storage_grid_size
        if getattr(orig, _PINNED_ATTR, False):
            continue

        def pinned(self, *a, _orig=orig, **k):
            g = _orig(self, *a, **k)
            x = min(int(g.x), cx) if cx is not None else int(g.x)
            y = min(int(g.y), cy) if cy is not None else int(g.y)
            return g if (x, y) == (int(g.x), int(g.y)) else ttnn.CoreCoord(x, y)

        setattr(pinned, _PINNED_ATTR, True)
        cls.compute_with_storage_grid_size = pinned
    return True


ACTIVE = install()
