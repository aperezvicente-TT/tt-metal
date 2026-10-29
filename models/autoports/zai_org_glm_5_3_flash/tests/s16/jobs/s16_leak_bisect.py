# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job for Stage 14b's SHARED MESH SERVICE (or any server with a resident m5_serving backend): localize the TP8
bucket-order trace-allocation leak (00:32Z: 3 untilize outputs alive after the 128-bucket capture) WITHOUT allocation
tracebacks: the TP prefill attached as in s16_service_prefill.py, then for each bucket an EAGER forward cut layer by
layer (embed, every layer, last row + head); after each piece (synchronize + gc) the buffers allocated since the
smallest bucket's capture and still alive (the tracker's get_unsafe_tracked_ids of that trace) are diffed: the piece
that leaves buffers alive, with each buffer's allocation context (the op that allocated it). Then (ep=True) the same
around EpMoe.__call__ of one MoE layer per bucket. The service is restored at the end (decode-only re-warm-up).

ARGS: buckets ([512, 128, 2048]), knobs ({env: value}), ep (True), restore (True)."""
import datetime as dt
import gc
import os
import sys

import torch

import ttnn
from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import svc_attach  # noqa: E402

args, ctx, mesh = ARGS, CTX, MESH  # noqa: F821
BUCKETS = [int(v) for v in args.get("buckets", [512, 128, 2048])]
MOCK = os.environ.get("GLM53_MOCK") == "1"


def log(*a):
    print(f"S16LEAK {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=sorted(set(BUCKETS)))
be, gen = att.be, att.gen
findings = []


def alive(tid):
    if MOCK:
        return {}
    ttnn.synchronize_device(mesh)
    gc.collect()
    return dict(get_unsafe_tracked_ids(mesh, tid))


try:
    pf = att.attach()
    warm_s = att.warm_s
    tid0 = pf.traces.get((min(BUCKETS), 0)) if not MOCK else None
    log(f"attached ({warm_s} s), traces {sorted(pf.traces)}, reference trace {(min(BUCKETS), 0)}")
    base0 = alive(tid0)
    findings.append(dict(after_warmup=sorted(set(base0.values()))))
    log(f"alive after the warm-up (allocated after the {min(BUCKETS)} capture): {len(base0)} {sorted(set(base0.values()))}")
    for c in BUCKETS:
        ci = pf.chunk_inputs(c)
        sp = pf.mode(c) == "sp"
        pf.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=0, n_valid=c)
        before = alive(tid0)
        x = pf.embed(ci)
        new = {k: v for k, v in alive(tid0).items() if k not in before}
        rows = [dict(bucket=c, piece="embed (x itself is 1)", new_alive=len(new), contexts=sorted(set(new.values()))[:6])]
        for li in range(len(pf.layers)):
            b_ = alive(tid0)
            x = pf.layer(li, x, ci, sp)
            n_ = {k: v for k, v in alive(tid0).items() if k not in b_}
            rows.append(dict(bucket=c, piece=f"layer {li} ({pf.layers[li].kind})", new_alive=len(n_),
                             contexts=sorted(set(n_.values()))[:6]))  # fmt: skip
        row = pf.last_row(x, ci, sp)
        ttnn.deallocate(x)
        ttnn.deallocate(row)
        left = {k: v for k, v in alive(tid0).items() if k not in before}
        rows.append(dict(bucket=c, piece="after the forward (x / row freed)", new_alive=len(left),
                         contexts=sorted(set(left.values()))[:12]))  # fmt: skip
        for r in rows:
            log(r)
        findings += rows
        if args.get("ep", True):  # EpMoe alone on one MoE layer's real router output
            from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, EpMoe

            pl = next(p for p in pf.layers if getattr(p, "moe", None) is not None and hasattr(p.moe, "router"))
            x32 = ttnn.from_torch(torch.randn(1, 1, c, 4096) * 0.5, dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=pf.repl)  # fmt: skip
            idx, scores = pl.moe.router(x32, decode=False)
            sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
            hb = ttnn.typecast(x32, BF16)
            ep = getattr(pl, "epmoe", None) or EpMoe(pf, pl.moe)
            b_ = alive(tid0)
            y = ep(hb, idx, sel)
            ttnn.deallocate(y)
            n_ = {k: v for k, v in alive(tid0).items() if k not in b_}
            r = dict(bucket=c, piece=f"EpMoe.__call__ (v2 {ep.v2_dispatch}/{ep.v2_combine}) on layer {pl.L}",
                     new_alive=len(n_), contexts=sorted(set(n_.values()))[:12])  # fmt: skip
            log(r)
            findings.append(r)
            for t_ in (x32, idx, scores, sel, hb):
                ttnn.deallocate(t_)
finally:
    pf = ep = None  # the job's own references: restore() must be able to free every prefill buffer
    if args.get("restore", True):
        log(f"restored the decode-only service config (warm-up {att.restore()} s)")
RESULT = findings
