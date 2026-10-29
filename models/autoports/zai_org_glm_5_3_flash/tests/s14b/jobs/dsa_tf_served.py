# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; outside the tree: runs on the server's tree): Stage 14b context ladder rung (program SCOPE
2026-10-21 02:57Z, item 3): the Stage 6 teacher-forcing harness (readiness_check.run_teacher_forcing, unchanged) on
the SERVED construction (m5common.serving: native B = 32 + the width / narrow buckets of the env, shared weights) with
Stage 14's DSA decode (GLM53_STACK_DSA=1: the sparse_dsa layers on MkMla(dsa) + the per-layer pooled-key store) past
the dense-exact 2,051 rows, against Stage 16's full-model CPU reference (prefill4096_full.refpt: a 4,064-token prompt +
32 forced tokens). The harness drives one user: the generator runs it on the smallest bucket (narrow w1 / native w8).
Pass: traced decode (no eager step), top-5 >= 0.98, top-100 = 1.0. The knobs are set for this job only and restored;
the backend is closed after it (its max_seq_len / DSA state must not serve later jobs).
ARGS: b (32), layers ("all"), ref, max_seq_len (8192), native (true), env (extra knobs), rev, tag, record."""
import gc
import os
import sys
import time
from pathlib import Path

gc.collect()
ARGS_ = ARGS  # noqa: F821
ENV = {"GLM53_STACK_DSA": "1", **{k: str(v) for k, v in (ARGS_.get("env") or {}).items()}}
old = {k: os.environ.get(k) for k in ENV}
sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

print("DSA-TF closed before:", M5.prune_ctx(CTX, []), flush=True)  # noqa: F821
os.environ.update(ENV)
try:
    import importlib

    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.mk_mla as MK
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config

    WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
    sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
    from readiness_check.run_teacher_forcing import run_teacher_forcing

    MODEL_DIR = Path(os.environ["TT_METAL_HOME"]) / "models/autoports/zai_org_glm_5_3_flash"
    mesh, args, ctx = MESH, ARGS_, CTX  # noqa: F821
    B = int(args.get("b", 32))
    cfg = hf_config()
    _L = args.get("layers", "all")
    ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
    ref = args.get("ref", "/opt/glm53/work-s16/refpt/prefill4096_full.refpt")
    ms = int(args.get("max_seq_len", 8192))
    tag = args.get("tag", f"m5-ctx-ladder-4k-B{B}")
    TPN = int(mesh.get_num_devices())

    def log(*a):
        print(f"DSA-TF {C.now()}", *a, flush=True)

    t0 = time.time()
    key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), max_seq_len=ms,
                     rev=("dsa", ms, args.get("rev")), log=log)  # fmt: skip
    be, gen = ctx[key]["backend"], ctx[key]["gen"]
    log(f"backend dsa_decode={be.dsa_decode} supported_context={be.supported_context} widths={be.native_widths} "
        f"narrow={getattr(be, 'narrow_widths', ())} ready in {time.time() - t0:.1f}s dram={M5.dram_state(mesh)}")  # fmt: skip
    c0, r0 = dict(gen.counters), dict(be.trace_report())
    t1 = time.time()
    per_entry = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(ref), mesh_device=mesh,
                                    build_kwargs=dict(generator=gen))  # fmt: skip
    wall = time.time() - t1
    c1, r1 = dict(gen.counters), be.trace_report()
    d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    s = per_entry[0]
    traced = d.get("eager_decode_steps", 0) == 0 and r1.get("decode_eager", 0) == r0.get("decode_eager", 0) and \
        d.get("decode_trace_replays", 0) > 0  # fmt: skip
    ok = traced and s["top5"] >= 0.98 and s["top100"] >= 1.0
    r = dict(test=f"m5 context ladder: Stage 6 TF harness on the served StackBackend (B={B} server, one user) with "
                  f"Stage 14's DSA decode past 2,051 rows ({len(ids)} layers, tp{TPN}, max_seq_len {ms}) vs Stage 16's "
                  f"full-model CPU reference {os.path.basename(ref)}", status="pass" if ok else "fail",
             milestone="M5-ctx", B=B, tp=TPN, layers=len(ids), reference=ref, max_seq_len=ms, mla_rev=MK.MLA_REV,
             dsa_decode=be.dsa_decode, supported_context=be.supported_context, top1=s["top1"], top5=s["top5"],
             top100=s["top100"], matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]],
             decode_traced_no_eager=traced, counters_delta=d, wall_s=round(wall, 1),
             us_per_token=round(1e6 / s["decode_t/s/u"], 1) if s.get("decode_t/s/u") else None,
             ttft_ms=s.get("ttft_ms"), dram=M5.dram_state(mesh), knob_env=ENV, tag=tag,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")))  # fmt: skip
    log("ROW", r)
    if args.get("record") and not SM.MOCK:
        C.record("pcc", r)
    RESULT = dict(ok=ok or SM.MOCK, top1=s["top1"], top5=s["top5"], top100=s["top100"], mock=SM.MOCK)
finally:
    print("DSA-TF closed after:", M5.prune_ctx(CTX, []), flush=True)  # noqa: F821
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
