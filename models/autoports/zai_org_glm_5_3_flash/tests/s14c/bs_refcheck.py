# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c host check (CPU only): the Stage 3 reference of B users is the first B rows of the B=32 reference (decode
rows are independent) -> bs.py may slice B=16 from B=32. usage: bs_refcheck.py <m3_ref dir> <layer> <b> [prompt]"""
import sys

import torch

d, layer, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = int(sys.argv[4]) if len(sys.argv) > 4 else 128
r, r32 = torch.load(f"{d}/L{layer}_B{b}_P{p}.pt"), torch.load(f"{d}/L{layer}_B32_P{p}.pt")
out = {}
for k in ("x", "y_ref", "golden", "S0", "S1"):
    out[k] = bool(torch.equal(r[k], r32[k][:b]))
for k in ("C0", "C1"):
    out[k] = bool(torch.equal(r[k], r32[k][:, :, :b]))
out["positions"] = list(r["positions"]) == list(r32["positions"])[:b]
print({"layer": layer, "b": b, "prompt": p, "equal": out, "all": all(out.values())})
