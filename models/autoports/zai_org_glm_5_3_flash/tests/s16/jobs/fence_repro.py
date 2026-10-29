# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; program 2026-10-26 20:06:53Z + 14's spec 20:09:45Z): THE ROW FENCE's CLASS REPRODUCER on any mesh with
a line of >= 3 chips (d0 3x2: the 3-chip axis; the TP8 2x4: the 4-chip rows). Per arm (fence mode none / hop1 / mcast):
one trace of per_trace x [FenceBurst (position 0 multicasts npkt 4-KiB packets over the whole line + a LAND +1 that
nothing waits for) -> RowFence (check: LAND >= g after the fence passed, else LEAK += 1)], replayed `replays` times.
Expected (14): none leaks, hop1 still leaks (a relay's 1-hop token can overtake its forwarding channel), mcast never.
The fence's own time = T2 - T0 cycles of its last launch per chip and lane.

ARGS: modes (["none", "hop1", "mcast"]), npkt (256), per_trace (50), replays (200), axis (None = the long axis), tag."""
import os
import sys
import time

import ttnn

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TREE = os.environ["TT_METAL_HOME"]
MD = os.path.join(TREE, "models/autoports/zai_org_glm_5_3_flash")
sys.path.insert(0, os.path.join(MD, "tests/s16"))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import row_fence as RF  # noqa: E402

MODES = list(args.get("modes", ["none", "hop1", "mcast"]))
NPKT, PER, REPLAYS = int(args.get("npkt", 256)), int(args.get("per_trace", 50)), int(args.get("replays", 200))
AXIS = args.get("axis")
TAG = args.get("tag", "fence-repro")
MOCK = (
    os.environ.get("GLM53_MOCK", "") == "1"
)  # a host mock cannot capture: the compile launches only (host build + JIT)

# every arm's fence + burst first (the sync tensors before any capture), then the arms in order
arms = {m: RF.RowFence(mesh, axis=None if AXIS is None else int(AXIS), mode=m, check=True) for m in MODES}
bursts = {m: RF.FenceBurst(mesh, arms[m], npkt=NPKT) for m in MODES}
f0 = arms[MODES[0]]
S.log(f"{TAG}: mesh {tuple(mesh.shape)}, line axis {f0.axis} ({f0.nc} chips x {f0.nlines} lines), lanes {f0.lanes}, "
      f"npkt {NPKT} x 4 KiB, {PER} pairs per trace x {REPLAYS} replays per arm")  # fmt: skip
out = {}
for m in MODES:
    f, b = arms[m], bursts[m]
    t0 = time.time()
    b()  # the compile launch of the pair (once; the capture below compiles nothing)
    f()
    ttnn.synchronize_device(mesh)
    if MOCK:
        out[m] = dict(mock=True, compile_s=round(time.time() - t0, 1))
        S.log(f"{TAG}: arm {m}: MOCK compile launch done ({out[m]['compile_s']} s)")
        continue
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        for _ in range(PER):
            b()
            f()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    t1 = time.time()
    for r in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        if (r + 1) % 20 == 0:
            ttnn.synchronize_device(mesh)
    ttnn.synchronize_device(mesh)
    t2 = time.time()
    ttnn.release_trace(mesh, tid)
    w = f.words()
    chips = {}
    for (line, p), lanes in w.items():
        l0 = lanes[0]
        chips[f"{line},{p}"] = dict(gen=[l_[RF.F["GEN"]] for l_ in lanes], leak=l0[RF.F["LEAK"]], checks=l0[RF.F["CHECKS"]],
                                    max_short=l0[RF.F["SEEN"]], land=l0[RF.F["LAND"]],
                                    cyc=[(l_[RF.F["T2"]] - l_[RF.F["T0"]]) & 0xFFFFFFFF for l_ in lanes])  # fmt: skip
    leaks = sum(c_["leak"] for c_ in chips.values())
    checks = sum(c_["checks"] for c_ in chips.values())
    pairs = 1 + PER * REPLAYS
    out[m] = dict(leaks=leaks, checks=checks, pairs=pairs, us_per_pair=round((t2 - t1) * 1e6 / (PER * REPLAYS), 1),
                  capture_s=round(t1 - t0, 1), chips=chips)  # fmt: skip
    S.log(f"{TAG}: arm {m}: LEAK {leaks} of {checks} checks ({pairs} burst+fence pairs); {out[m]['us_per_pair']} us per pair; "
          f"chips {chips}")  # fmt: skip
ok = MOCK or ("mcast" not in out or out["mcast"]["leaks"] == 0) and ("none" not in out or out["none"]["leaks"] > 0)
RESULT = dict(test="the row fence class reproducer (relayed late packet: burst -> fence + LAND check), traced", tag=TAG,
              mesh=list(mesh.shape), axis=f0.axis, nc=f0.nc, npkt=NPKT, per_trace=PER, replays=REPLAYS,
              arms={m: {k_: v_ for k_, v_ in o_.items() if k_ != "chips"} for m, o_ in out.items()}, detail=out,
              status="mock" if MOCK else ("pass" if ok else "fail"))  # fmt: skip
print("FENCE_REPRO", RESULT["arms"], RESULT["status"], flush=True)
for f in arms.values():
    f.release()
for b in bursts.values():
    b.release()
