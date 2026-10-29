# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair; also the 2x4 mock): the indexer-score + running top-512 phase (tt/ix_phase.py) vs the op path
(indexer_score_dsa + the visibility mask + topk_large_indices, prefill_tp.DsaIndexer.select's sdpa branch) on the same
inputs, both captured in a trace and replayed. Per chip: R query rows (the last R rows of a window of 4 P tokens: query
t sees pools < (4 P - R + t + 1) / 4), 32 heads x 128, P pooled keys; synthetic N(0, 1) bf16 q / keys and N(0, 1) head
weights (the model's scale folded in), seeded. Compared per query: the overlap of the two 512-pool sets, and on
`ref_rows` sampled rows both against the host fp32 top-512 (float64 scores of the same bf16 inputs). One pcc row (set
overlap) and one perf row (both traces' steady-state replay). Eager guard: the phase launches exactly twice (compile +
capture). ARGS: pools (16384 = a 64K window), rows (1024), kc (4), fid ("HiFi4"), replays (100), ref_rows (16), seed,
tag, record, diag (0; IxPhase timing diagnostics)."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.ix_phase import TOPK, IxPhase  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import NEG, _ckc_fp32  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
P = int(args.get("pools", 16384))
R = int(args.get("rows", 1024))
KC = int(args.get("kc", 4))
FID = str(args.get("fid", "HiFi4"))
REPLAYS = int(args.get("replays", 100))
NREF = int(args.get("ref_rows", 16))
tag = args.get("tag", f"ix-phase-P{P}-R{R}")
TPN = int(mesh.get_num_devices())
H, D = 32, 128
CS = 4 * P - R  # the chunk = the window's last R rows
DRAM = ttnn.DRAM_MEMORY_CONFIG
rep = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None

g = torch.Generator().manual_seed(int(args.get("seed", 1234)))
qh = torch.randn(R, H, D, generator=g).to(torch.bfloat16)
keys_h = torch.randn(P, D, generator=g).to(torch.bfloat16)
w_h = (torch.randn(R, H, generator=g) * (H**-0.5) * (D**-0.5)).float()
dev = lambda t, dt, lay: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, mesh_mapper=rep, memory_config=DRAM)  # noqa: E731
q_qm = dev(qh.reshape(1, 1, R * H, D), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # the phase: query-major
q_hm = dev(qh.permute(1, 0, 2).reshape(1, H, R, D).contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT)  # the op: head-major
w_rm = dev(w_h.reshape(1, 1, R, H), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
w_op = dev(w_h.reshape(1, 1, R, H), ttnn.bfloat16, ttnn.TILE_LAYOUT)
kk = dev(keys_h.reshape(1, 1, P, D), ttnn.bfloat16, ttnn.TILE_LAYOUT)
nvis = torch.clamp((CS + torch.arange(R) + 1) // 4, max=P)
vis_h = (torch.arange(P).reshape(1, P) < nvis.reshape(R, 1)).to(torch.bfloat16)
vis = dev(vis_h.reshape(1, 1, R, P), ttnn.bfloat16, ttnn.TILE_LAYOUT)
sel_ph = dev(torch.zeros(1, 1, R, TOPK, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
sel_op = dev(torch.zeros(1, 1, R, TOPK, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
DIAG = int(args.get("diag", 0))  # IxPhase timing diagnostics (the selections are then wrong)
phase = IxPhase(mesh, R, P, kc=KC, fidelity=FID, diag=DIAG)
ckc = _ckc_fp32(mesh)
ckc_score = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                   math_approx_mode=False, fp32_dest_acc_en=False)  # fmt: skip
S.log(f"{tag}: R {R} P {P} (window {4 * P}) members {phase.nm} nq {phase.nq} kc {KC} blocks {phase.nblk} fid {FID}; model estimate "
      f"{phase.estimate_us():.0f} us")  # fmt: skip


def op_fn():
    s = ttnn.experimental.indexer_score_dsa(q_hm, kk, w_op, key_compression_ratio=4, compute_kernel_config=ckc_score,
                                            chunk_start_idx=CS)  # fmt: skip
    if s.layout != ttnn.TILE_LAYOUT:
        s = ttnn.to_layout(s, ttnn.TILE_LAYOUT)
    if s.dtype != ttnn.bfloat16:
        s = ttnn.typecast(s, ttnn.bfloat16)
    sm = ttnn.where(vis, s, NEG)
    ttnn.deallocate(s)
    pools = ttnn.experimental.topk_large_indices(ttnn.to_layout(sm, ttnn.ROW_MAJOR_LAYOUT), k=TOPK)
    ttnn.deallocate(sm)
    ttnn.copy(pools if pools.dtype == ttnn.uint32 else ttnn.typecast(pools, ttnn.uint32), sel_op)
    ttnn.deallocate(pools)


def ph_fn():
    phase(q_qm, w_rm, kk, sel_ph, CS)


def capture(fn, name):
    """the trace of fn (its compile launch ran before the first capture, below); None on the mock"""
    if C.MOCK:
        return None
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    return tid


# both compile launches before the first capture (a program first compiled after a capture keeps a live buffer)
t0 = time.time()
op_fn()
ph_fn()
ttnn.synchronize_device(mesh)
S.log(f"{tag}: op + phase compile launches in {time.time() - t0:.1f} s")
op_tid = capture(op_fn, "op")
ph_tid = capture(ph_fn, "phase")
assert phase.calls == (1 if C.MOCK else 2), ("eager guard", phase.calls)
row = dict(test=f"indexer score + top-512 phase vs the op path (P {P} = window {4 * P}, R {R} rows per chip)",
           milestone="P3", pools=P, window=4 * P, rows=R, kc=KC, fid=FID, nq=phase.nq, members=phase.nm, tp=TPN, tag=tag, diag=DIAG,
           status="mock" if C.MOCK else "info")  # fmt: skip
if not C.MOCK:
    ttnn.execute_trace(mesh, op_tid, cq_id=0, blocking=True)
    ttnn.execute_trace(mesh, ph_tid, cq_id=0, blocking=True)
    host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0] if TPN > 1 else t).reshape(R, TOPK).to(torch.int64) & 0xFFFFFFFF  # noqa: E731
    a, b = host(sel_ph), host(sel_op)
    ov, pad_ph, bad = [], 0, 0
    for t in range(R):
        sa = {int(v) for v in a[t].tolist() if v != 0xFFFFFFFF}
        sb = {int(v) for v in b[t].tolist()}
        pad_ph += TOPK - len(sa)
        bad += sum(1 for v in sa if v >= int(nvis[t]))
        ov.append(len(sa & sb) / TOPK)
    ovt = torch.tensor(ov)
    ref = {}
    idx = torch.linspace(0, R - 1, NREF).long().tolist() if NREF else []
    if idx:  # host fp32 / float64 top-512 of the same bf16 inputs on sampled rows
        kd = keys_h.double()
        ra, rb = [], []
        for t in idx:
            sc = (torch.relu(qh[t].double() @ kd.T) * w_h[t].double().reshape(H, 1)).sum(0)
            sc[int(nvis[t]):] = -1e300
            top = set(torch.topk(sc, TOPK).indices.tolist())
            ra.append(len(top & {int(v) for v in a[t].tolist()}) / TOPK)
            rb.append(len(top & {int(v) for v in b[t].tolist()}) / TOPK)
        ref = dict(rows=len(idx), phase_vs_host_mean=round(sum(ra) / len(ra), 5), phase_vs_host_min=round(min(ra), 5),
                   op_vs_host_mean=round(sum(rb) / len(rb), 5), op_vs_host_min=round(min(rb), 5))  # fmt: skip
    row.update(overlap_mean=round(float(ovt.mean()), 5), overlap_min=round(float(ovt.min()), 5),
               rows_exact=int((ovt == 1.0).sum()), phase_padding=pad_ph, phase_invisible=bad, host_ref=ref)  # fmt: skip
    perf_ph = S.replay_stats(mesh, ph_tid, replays=REPLAYS, label=f"{tag} phase")
    perf_op = S.replay_stats(mesh, op_tid, replays=REPLAYS, label=f"{tag} op")
    assert phase.calls == 2, ("eager guard: the phase launched during the replays", phase.calls)
    row.update(phase_us=perf_ph.get("median_us"), op_us=perf_op.get("median_us"), phase_p90_us=perf_ph.get("p90_us"),
               speedup=round(perf_op["median_us"] / perf_ph["median_us"], 2) if perf_ph and perf_op else None,
               model_us=round(phase.estimate_us(), 1), replays=perf_ph.get("replays"))  # fmt: skip
    for t_ in (op_tid, ph_tid):
        ttnn.release_trace(mesh, t_)
S.log(f"{tag}: {row}")
if args.get("record", False):
    C.record("pcc", row)
RESULT = [row]
print("IX_PHASE_UNIT", row, flush=True)
