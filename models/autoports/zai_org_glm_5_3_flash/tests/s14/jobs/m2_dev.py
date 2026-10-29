# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): M2 — the in-program fabric all-reduce phase (tt/fabric_ar.py) on the opened mesh, B = 1 / 8 / 32,
device-vs-device against ttnn.all_reduce on the SAME inputs (and a host emulation of the phase's fixed fp32 order).
Every number is a traced replay (the program's first launch is its compile / warm-up only):
  * one-iteration program, traced replay, host wall (blocking) -> median of N;
  * K-iteration program (the phase looped K times inside ONE launch, the end-of-iteration barrier included): traced
    replay wall / K and the in-program lane / producer cycle probes;
  * ttnn.all_reduce traced on the same input (the dispatched CCL it replaces).
ARGS: bs (list), reps (int), k (int), pkt (pseudo-tiles per packet), tag (str), record (bool)."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.fabric_ar as FAR  # noqa: E402

importlib.reload(FAR)  # long-lived server: pick up edits

gc.collect()  # tensors of an earlier failed job (traceback cycles) free their L1 now
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
mgd = CTX["mgd"]  # noqa: F821
ndev = mesh.get_num_devices()
reps, K = int(args.get("reps", 50)), int(args.get("k", 50))
rec = bool(args.get("record", True))
tag = args.get("tag", "m2")
out = {}


def host_dev(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


for B in args.get("bs", [1, 8, 32]):
    g = torch.Generator().manual_seed(4242 + B)
    xh = (torch.randn(ndev, 1, B, 4096, generator=g) * 2.0).to(torch.bfloat16)
    t0 = time.time()
    far = FAR.FabricAllReduce(mesh, B, pkt_tiles=int(args.get("pkt", 2)), iters=1, grp=int(args.get("grp", 8)),
                              mode=args.get("mode", "tensor"))  # fmt: skip
    x, y = far.input_from_torch(xh), far.output()
    # --- one-iteration program: compile launch, then traced replays
    tr = C.Traced(ttnn, mesh, lambda: far(x, y))
    walls = tr.replays(reps)
    batch_us = tr.batch(reps)
    ys = [v.reshape(1, 1, -1, 4096)[:, :, :B] for v in host_dev(y)]
    pr1 = far.probes()
    tr.release()
    # --- ttnn.all_reduce on the same input (device-vs-device)
    # (a baseline only: in a long-lived server its static CBs can clash with L1 buffers of earlier jobs -> the
    # fabric numbers are still recorded, the comparison fields say why they are missing)
    ar_err = None
    try:
        tra = C.Traced(ttnn, mesh, lambda: ttnn.all_reduce(x, memory_config=ttnn.L1_MEMORY_CONFIG,
                                                             topology=ttnn.Topology.Linear))  # fmt: skip
        ar_walls = tra.replays(reps)
        ars = [v.reshape(1, 1, -1, 4096)[:, :, :B] for v in host_dev(tra.out)]
        tra.release()
        for t in tra.out if isinstance(tra.out, (list, tuple)) else [tra.out]:
            ttnn.deallocate(t)
    except RuntimeError as e:
        ar_err = str(e).splitlines()[0][:200] + " | " + next((l for l in str(e).splitlines() if "clash" in l), "")[:200]
        ar_walls, ars = None, None
    # --- K iterations in one launch (in-program steady state)
    yk = far.output()
    trk = C.Traced(ttnn, mesh, lambda: far(x, yk, iters=K))
    kw = trk.replays(5)
    prk = far.probes()
    yks = [v.reshape(1, 1, -1, 4096)[:, :, :B] for v in host_dev(yk)]
    trk.release()
    ref = far.reference(xh)
    if args.get("save"):
        torch.save(dict(x=xh, ys=torch.stack(ys), ars=torch.stack(ars) if ars else None, ref=ref),
                   os.path.join(os.environ.get("GLM53_WORK", "/opt/glm53/work-s14"), f"m2_{mgd}_B{B}_{tag}.pt"))
    exact32 = xh.float().sum(0, keepdim=True)
    same = all(torch.equal(ys[0], v) for v in ys[1:])
    same_k = all(torch.equal(ys[0], v) for v in yks)
    bit_ar = all(torch.equal(a, b) for a, b in zip(ys, ars)) if ars else None
    n_diff_ar = int((ys[0] != ars[0]).sum()) if ars else None
    bit_ref = bool(torch.equal(ys[0], ref))
    pcc_ar, mae_ar = (C.pcc(ys[0], ars[0]), float((ys[0] - ars[0]).abs().max())) if ars else (None, None)
    pcc_32, mae_32 = C.pcc(ys[0], exact32), float((ys[0] - exact32).abs().max())
    pcc_ar32, mae_ar32 = (C.pcc(ars[0], exact32), float((ars[0] - exact32).abs().max())) if ars else (None, None)
    one = C.stats(walls)
    per_iter = [w / K for w in kw]
    # in-program per-iteration span from the probes: lane-0 BRISC first-iteration own-ready -> end of the K loop
    d0 = prk[0]
    span_in_prog = [round((p["lane0_brisc"][5] - p["lane0_brisc"][1]) / K, 3) for p in prk.values()]
    phase1 = [round(p["prod_done_max"] - p["prod_start_min"], 3) for p in pr1.values()]
    ok = same and same_k and pcc_32 >= 0.99999
    row = dict(test=f"fabric_ar {mgd} B={B}", status="pass" if ok else "fail", milestone="M2" if mgd == "2x4" else "M2-pair",
               traced=True, mgd=mgd, ndev=ndev, B=B, bytes=B * 4096 * 2, packed_bytes_per_lane=far.pt * 2048,
               us=round(min(C.stats(per_iter)["median_us"], one["median_us"]), 2),
               one_launch_replay=one, batch_replay_mean_us=round(batch_us, 2) if batch_us else None,
               k=K, k_launch_replay_per_iter_us=C.stats(per_iter), in_program_per_iter_us=span_in_prog,
               single_launch_phase_us=phase1, ttnn_all_reduce_replay=C.stats(ar_walls) if ar_walls else ar_err, pkt_tiles=far.pkt_tiles,
               probes_first=pr1, probes_k=prk, tag=tag, grp=far.grp, mode=far.mode, cb_addr=far.addr, payload=CTX.get("payload"),  # noqa: F821
               note="us = best of (one-launch traced replay wall, K-iteration launch replay wall / K); in_program = lane "
                    "BRISC cycle probes over the K-iteration loop / K (incl. the end-of-iteration barrier)")  # fmt: skip
    prow = dict(test=f"fabric_ar {mgd} B={B} numerics", status="pass" if ok else "fail", mgd=mgd, B=B,
                replicas_identical=same, k_iteration_identical=same_k, bitwise_vs_ttnn_all_reduce=bit_ar,
                n_elems_diff_vs_ttnn_all_reduce=n_diff_ar, pcc_vs_ttnn_all_reduce=pcc_ar,
                max_abs_err_vs_ttnn_all_reduce=mae_ar, bitwise_vs_host_emulation=bit_ref, pcc_vs_fp32_sum=pcc_32,
                max_abs_err_vs_fp32_sum=mae_32, ttnn_all_reduce_pcc_vs_fp32_sum=pcc_ar32,
                ttnn_all_reduce_max_abs_err_vs_fp32_sum=mae_ar32, tag=tag,
                order="round 1 (axis 0): bf16(s0 + s1) [fp32]; round 2 (axis 1): bf16((r0 + r1) + (r2 + r3)) [fp32 DST]")  # fmt: skip
    watcher = bool(os.environ.get("TT_METAL_WATCHER"))
    if rec and watcher:  # watcher runs: numerics + clean exit are the evidence, timings are not
        C.record("pcc", {**prow, "test": f"fabric_ar {mgd} B={B} numerics under the watcher"})
        C.record("tests", dict(test=f"fabric_ar {mgd} B={B} watcher run", status=prow["status"], k=K, reps=reps,
                               watcher_log=os.environ.get("TT_METAL_LOGS_PATH")))
    elif rec:
        C.record("perf", row)
        C.record("pcc", prow)
    else:  # dev loop: rows to the work dir (not evidence)
        import json

        with open(os.path.join(os.environ.get("GLM53_WORK", "/opt/glm53/work-s14"), "dev_rows.jsonl"), "a") as f:
            f.write(json.dumps({"time_utc": C.now(), **row}, default=str) + "\n")
            f.write(json.dumps({"time_utc": C.now(), **prow}, default=str) + "\n")
        print("ROW", row, "\nPROW", prow, flush=True)
    out[B] = dict(us=row["us"], one=one["median_us"], per_iter=C.stats(per_iter)["median_us"], in_prog=span_in_prog,
                  ar=C.stats(ar_walls)["median_us"] if ar_walls else ar_err, same=same, bit_ar=bit_ar, n_diff=n_diff_ar, bit_ref=bit_ref,
                  pcc32=pcc_32, mae_ar=mae_ar, wall_s=round(time.time() - t0, 1))  # fmt: skip
    for t in (x, y, yk):
        ttnn.deallocate(t)
    far.release()
RESULT = out
