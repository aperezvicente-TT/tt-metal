# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): kda_rec value-column split (REC_SPREAD) device vs device — the standalone KdaRec program on H heads
with spread 1 vs spread s (H * s <= 64 cores) on the SAME inputs (random q / k / v / g / beta / gate_out rows, a random
state snapshot): the gated-normed output and the updated state must be bitwise equal (the split changes no arithmetic:
column-independent recurrence; the sum of squares of a head's V columns is reduced in a fixed tree, whose bits can
differ from the single-core row sum -> also PCC / max-abs). Compile launch, then traced replays (state restored).
ARGS: heads (16), spreads ([2, 4]), b (4), seed (0)."""
import gc
import importlib
import os
import sys
from types import SimpleNamespace

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402

importlib.reload(KREC)
mesh, args = MESH, ARGS  # noqa: F821
H, B = int(args.get("heads", 16)), int(args.get("b", 4))
g = torch.Generator().manual_seed(int(args.get("seed", 0)))
rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
DK = 128
rep = dict(device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), layout=ttnn.TILE_LAYOUT)
dev = lambda t, dt=ttnn.float32, mem=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t, dtype=dt, memory_config=mem, **rep)  # noqa: E731
o_norm = dev((1.0 + 0.1 * rnd(1, 1, 1, DK)), ttnn.bfloat16)
host = dict(q=rnd(1, 1, B, H * DK), k=rnd(1, 1, B, H * DK), v=rnd(1, 1, B, H * DK),
            g=-torch.rand(1, 1, B, H * DK, generator=g) * 0.2, beta=torch.rand(1, 1, B, H, generator=g),
            go=rnd(1, 1, B, H * DK), S=rnd(B, H, DK, DK) * 0.05)  # fmt: skip
ins = {k: dev(v) for k, v in host.items() if k != "S"}
S0 = dev(host["S"])


def run(spread):
    kda = SimpleNamespace(device=mesh, h=H, d=DK, eps=1e-6, o_norm=o_norm, kda_rec_spread=spread)
    rec = KREC.KdaRec(kda)
    S = dev(host["S"])

    def step():
        ttnn.copy(S0, S)
        return rec(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], ins["go"], S, rows=B)

    tr = C.Traced(ttnn, mesh, step)
    tr.replay()
    on = ttnn.to_torch(ttnn.get_device_tensors(tr.out)[0]).float()[..., :B, :]
    st = ttnn.to_torch(ttnn.get_device_tensors(S)[0]).float()
    walls = tr.replays(10)
    tr.release()
    return on, st, C.stats(walls)["median_us"]


base_on, base_st, base_us = run(1)
res = dict(heads=H, b=B, spread1_us=base_us)
for s in args.get("spreads", [2, 4]):
    on, st, us = run(int(s))
    res[f"s{s}"] = dict(on_bitwise=bool(torch.equal(on, base_on)), state_bitwise=bool(torch.equal(st, base_st)),
                        on_pcc=C.pcc(on, base_on) if hasattr(C, "pcc") else None,
                        on_max_abs=float((on - base_on).abs().max()), state_max_abs=float((st - base_st).abs().max()),
                        us=us)  # fmt: skip
    print("REC_SPREAD", s, res[f"s{s}"], flush=True)
RESULT = res
