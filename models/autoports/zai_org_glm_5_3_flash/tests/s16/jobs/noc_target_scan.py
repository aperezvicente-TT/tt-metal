# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 MOCK): HANG static check (program 2026-10-23 21:02:25Z). Build the served PMK chain
(L0-4 + program B), capture every per-rank program-A descriptor (PmkSparseKdaPlan.descriptor) and every
ttnn.generic_op ProgramDescriptor (B, state_io, ...) of one forward, and scan every kernel's compile-time and per-core
runtime args for NoC targets on the DISPATCH column: translated NoC x = 14 (logical column 11, the dispatch pair at
(11, 0) / (11, 1); noc_map_unit.py: workers sit at x 1..7, 10..13, y 2..11). Flags (a) a pair (14, y) with
2 <= y <= 11, (b) four args (x0, y0, x1, y1) forming a rectangle that spans x = 14. Every hit is a candidate to read in
the kernel; no hit = no arg-borne target on the dispatch column (coordinates built in-kernel are not covered)."""
import os
import sys
import time
import traceback

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PL  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = int(args.get("rows", 2048))
WINDOWS = str(args.get("windows", "65536"))
DX = int(args.get("dispatch_x", 14))
import pmk45_alias as A45  # noqa: E402

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PF_SPARSE_T"] = WINDOWS
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
caps = []  # (label, rank, ProgramDescriptor)


def scan_list(a):
    hits = []
    for i in range(len(a) - 1):
        if a[i] == DX and 2 <= a[i + 1] <= 11:
            hits.append(("pair", i, a[i : i + 2]))
    for i in range(len(a) - 3):
        x0, y0, x1, y1 = a[i : i + 4]
        if 1 <= x0 < x1 <= 16 and 0 <= y0 <= y1 <= 11 and x0 <= DX <= x1:
            hits.append(("rect", i, a[i : i + 4]))
    return hits


def cores_of(crs):
    out = []
    for r in crs.ranges():
        for x in range(int(r.start.x), int(r.end.x) + 1):
            for y in range(int(r.start.y), int(r.end.y) + 1):
                out.append((x, y))
    return out


row = {}
try:
    max_seq = max(int(w) for w in WINDOWS.split(",")) + N + 512
    ent = S.build(mesh, ctx, LAYERS, b=2, buckets=(N,), max_seq=max_seq)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    A45.publish_page_table(backend, kv, io, pf, rows=2)
    c = pf.bucket_for(N)
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
    opts.update({k: v for k, v in args.items() if k.startswith("b_")})
    os.environ["GLM53_PF_MODE"] = "pmk"
    stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    stack.attach()
    orig_desc = PL.PmkSparseKdaPlan.descriptor

    def desc_wrap(self, device, io_, *a, **k):
        d = orig_desc(self, device, io_, *a, **k)
        caps.append((f"A L{getattr(self, 'li', '?')} scope {sorted(getattr(self, 'scope', None) or [])}", int(k.get("rank", 0)), d))
        return d

    PL.PmkSparseKdaPlan.descriptor = desc_wrap
    orig_go = ttnn.generic_op

    def go_wrap(io_, prog, *a, **k):
        if hasattr(prog, "kernels"):
            caps.append(("generic_op " + ",".join(os.path.basename(str(kd.kernel_source)) for kd in prog.kernels)[:120], -1, prog))
        return orig_go(io_, prog, *a, **k)

    ttnn.generic_op = go_wrap
    toks = S.golden_prompt(32768).reshape(-1).long()[:N]
    pf.write_inputs(c, toks, slot=0, chunk_start=0, n_valid=N)
    try:
        pf.forward(c)
    finally:
        PL.PmkSparseKdaPlan.descriptor = orig_desc
        ttnn.generic_op = orig_go
    report, n_kern, n_lists, rt_err = {}, 0, 0, None
    for label, rank, d in caps:
        for kd in d.kernels:
            n_kern += 1
            src = os.path.basename(str(kd.kernel_source))
            key = f"{label.split(' scope')[0] if label.startswith('A') else 'op/B'}::{src}"
            ent_ = report.setdefault(key, dict(kernels=0, cores=0, ct_hits=[], rt_hits=[], ranks=set()))
            ent_["kernels"] += 1
            ent_["ranks"].add(rank)
            ct = [int(v) for v in kd.compile_time_args]
            for h in scan_list(ct)[:4]:
                ent_["ct_hits"].append(dict(rank=rank, kind=h[0], at=h[1], vals=h[2]))
            for (x, y) in cores_of(kd.core_ranges):
                ent_["cores"] += 1
                try:
                    a = [int(v) for v in kd.runtime_args[x][y]]
                except Exception as e:  # noqa: BLE001
                    rt_err = f"{type(e).__name__}: {e}"[:200]
                    continue
                n_lists += 1
                for h in scan_list(a)[:4]:
                    if len(ent_["rt_hits"]) < 12:
                        ent_["rt_hits"].append(dict(rank=rank, core=[x, y], kind=h[0], at=h[1], vals=h[2], n_args=len(a),
                                                    window=[max(0, h[1] - 10), a[max(0, h[1] - 10) : h[1] + 12]]))  # fmt: skip
                if ent_.get("tails", 0) < 40 and len(a) >= 3 and all(10 <= v <= 15 for v in (a[-1], a[-2])):
                    ent_["tails"] = ent_.get("tails", 0) + 1  # fabric connection triples at the arg tail
                    t_, j_ = [], len(a)
                    while j_ >= 3 and 10 <= a[j_ - 1] <= 15 and 10 <= a[j_ - 2] <= 15:
                        t_.insert(0, a[j_ - 3 : j_])
                        j_ -= 3
                    ent_.setdefault("conn", []).append(dict(rank=rank, core=[x, y], triples=t_))
    hits = {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in report.items()
            if v["ct_hits"] or v["rt_hits"]}  # fmt: skip
    for k, v in hits.items():
        H.log(f"NOCSCAN HIT {k}: ct {v['ct_hits'][:3]} rt {v['rt_hits'][:4]}")
    row = dict(status="mock" if C.MOCK else "device", descriptors=len(caps), kernels=n_kern, rt_lists=n_lists, rt_err=rt_err,
               kernel_files=sorted(report), hits=hits, dispatch_x=DX,
               conns={k: v["conn"] for k, v in report.items() if v.get("conn")})  # fmt: skip
    H.log(f"NOCSCAN: {len(caps)} descriptors, {n_kern} kernels, {n_lists} rt lists scanned, {len(hits)} kernel files with "
          f"candidate hits; rt_err {rt_err}")  # fmt: skip
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    row = dict(status="fail", error=f"{type(e).__name__}: {e}"[:1500])
RESULT = row
