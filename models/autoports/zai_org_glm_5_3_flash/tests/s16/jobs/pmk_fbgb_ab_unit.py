# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair or 2x4): the f_b / g_b DRAM reclaim (program 2026-10-25 22:46:05Z) -- program A
(tt/pmk_assembly.PmkProgramA, the SERVED knobs: A_INPLACE 1, A_LR_EXACT 1, A34_MERGE 1, TP8_RESIDENT 0) on layer L with
GLM53_PMK_FBGB_JOINT=0 (the v1 f_b / g_b stream copies: one stream each at TP8, bank-padded) vs =1 (ONE joint f_b | g_b
copy, every bank whole streams), same golden rows, both TRACED. Gate: per chip BITWISE u / lr / f / gate_out (A2's
stores), y (the reduced attention output), x1 (the residual after P10) and x-hat_f between the arms; the joint arm
deterministic over `det` replays; the f_b / g_b DRAM bytes per chip of each arm (report dram_weights) and the traced
replay us of each. ARGS: layer (4), rows (2048), replays (20), det (20), tag, record."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
NDEV = int(mesh.get_num_devices())
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 20))
DET = int(args.get("det", 20))
tag = args.get("tag", f"pmk-fbgb-ab-L{L}")
# ab_env (cf3h-k4d8a L3): the env knob the two arms flip ("0" then "1"; default the joint-copy A/B, GLM53_PMK_A_ZSKIP for
# the zero-tile skip); lr_cols (0 = all): compare the low-rank store over its first lr_cols columns only (a diagnostic split
# if the dead pad columns >= 288 ever differ: ZSKIP packs them from the cleared DST instead of x @ 0)
AB_ENV = str(args.get("ab_env", "GLM53_PMK_FBGB_JOINT"))
LR_COLS = int(args.get("lr_cols", 0) or 0)
SERVED = {"PMK_A_INPLACE": "1", "GLM53_PMK_A_LR_EXACT": "1", "PMK_A34_MERGE": "1", "PMK_TP8_RESIDENT": "0",
          "GLM53_PF_MHC": "comp", "GLM53_PF_MHC_POST": "fused", "GLM53_PF_PACKED": "", "GLM53_PF_PROBE_LAYERS": ""}  # fmt: skip
saved = {k_: os.environ.get(k_) for k_ in list(SERVED) + ["GLM53_PMK_FBGB_JOINT", AB_ENV]}
os.environ.update(SERVED)
row = dict(test=f"f_b / g_b joint stream copy (GLM53_PMK_FBGB_JOINT 1) vs the v1 copies (0): program A L{L} tp{NDEV}, served "
           f"knobs, traced; bitwise u / lr / f / gate_out / y / x1 / x-hat_f + det", milestone="DRAM-FBGB", layer=L, tp=NDEV,
           tag=tag)  # fmt: skip
per_chip = lambda t_: [ttnn.to_torch(d).float().clone() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731
try:
    S.set_attn(args)
    ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
    pf = ent["pf"]
    pl = pf.layers[0]
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    assert sp and c == N, (sp, c, N)
    ci = pf.chunk_inputs(c)
    pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
    rows, _ = S.golden_layer(L, N)
    rows = rows.reshape(N, 4, 4096).float()
    x0 = pf.streams_to_device(rows, sp)
    zeros = {}

    def zero(t_):
        key = (tuple(t_.shape), t_.dtype, t_.layout)
        if key not in zeros:
            zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        ttnn.copy(zeros[key], t_)

    outs, arms = {}, {}
    for joint in [str(v_) for v_ in args.get("order", ["0", "1"])]:  # order: the arm order (an A/B ordering check)
        os.environ[AB_ENV] = joint
        x_p = pf.streams_to_device(rows, sp)
        state_p = pl.kda.allocate_state(1)
        ttnn.copy(ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh),
                  state_p.slot_mask)  # fmt: skip
        t1 = time.time()
        prog = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end)
        dw = prog.report.get("dram_weights") or {}
        arms[joint] = dict(build_s=round(time.time() - t1, 1), fb_gb_bytes=int((dw.get("copy_bytes") or {}).get("f_b", 0))
                           + int((dw.get("copy_bytes") or {}).get("g_b", 0)), copy_total=dw.get("copy_total"),
                           joint=getattr(prog, "fbgb_joint", None) is not None,
                           streams={k_: v_.get("nst") for k_, v_ in (prog.report.get("streams") or {}).items()})  # fmt: skip
        S.log(f"{tag}: arm JOINT={joint} built; {arms[joint]}")

        def fn():
            ttnn.copy(x0, x_p)
            zero(state_p.S)
            zero(state_p.C)
            prog(x_p, ci)

        fn()  # the compile launch
        ttnn.synchronize_device(mesh)

        def grab():
            st = prog.stores()
            lr_ = per_chip(st["lr"])
            if LR_COLS:
                lr_ = [t_[..., :LR_COLS] for t_ in lr_]
            return dict(u=per_chip(st["u"]), lr=lr_, f=per_chip(st["f"]), gate_out=per_chip(st["gate_out"]),
                        y=per_chip(prog.ys[0]), x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]))  # fmt: skip

        if C.MOCK:
            fn()
            outs[joint] = None
            continue
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        outs[joint] = grab()
        if joint == "1" and DET:
            same = True
            for _ in range(DET):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            g2 = grab()
            same = all(torch.equal(a_, b_) for k_ in g2 for a_, b_ in zip(g2[k_], outs["1"][k_]))
            arms[joint]["determinism"] = {str(DET): bool(same)}
        rs = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} JOINT={joint}")
        arms[joint]["median_us"] = rs.get("median_us")
        ttnn.release_trace(mesh, tid)
    row["arms"] = arms
    if C.MOCK:
        row["status"] = "mock"
    else:
        cmp = {k_: [dict(bitwise=bool(torch.equal(a_, b_)), maxabs=float((a_ - b_).abs().max()))
                    for a_, b_ in zip(outs["1"][k_], outs["0"][k_])] for k_ in outs["0"]}  # fmt: skip
        row["joint_vs_v1"] = cmp
        row["bitwise"] = all(ch["bitwise"] for v_ in cmp.values() for ch in v_)
        row["status"] = "pass" if (row["bitwise"] and all(arms["1"].get("determinism", {"0": True}).values())) else "fail"
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    row.update(status="fail", error=f"{type(e).__name__}: {e}"[:2000])
for k_, v_ in saved.items():
    if v_ is None:
        os.environ.pop(k_, None)
    else:
        os.environ[k_] = v_
S.log(f"{tag}: {row}")
if bool(args.get("record", False)) and not C.MOCK:
    C.record("pcc", row)
RESULT = [row]
print("PMK_FBGB_AB_UNIT", row, flush=True)
