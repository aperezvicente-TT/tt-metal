# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, chip 3): Stage 14b — the Stage 6 teacher-forcing harness (the plugin's
readiness_check.run_teacher_forcing, unchanged) driving tt/generator.py on tt/stack_model.py's StackBackend: prefill as
decode steps, ONE execute_trace per decode step with the device top-K / sampled token, the forced token written by the
generator. On a MODEL PREFIX (default dense L0-L2 + the head) against a reference of the SAME prefix (CPU fp32,
doc/ttnn/14b-persistent-stack/scripts/make_prefix_refpt.py): top-1 / top-5 / top-100 = how often the device's token is
in the prefix reference's top-k (the harness metric), so the full-model run only swaps the layer list and the
reference. ARGS: b (1), dense ([0, 1, 2]), sparse ([]), ref (refpt path), tag, record, rev."""
import gc
import importlib
import os
import sys
import time
from pathlib import Path

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, MKS, MKH, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.run_teacher_forcing import run_teacher_forcing  # noqa: E402

MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 1))
DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [])]
DSA = [int(v) for v in args.get("dsa", [])]  # sparse_dsa layers (Stage 7b MkMla + MkMoE; one chip)
REF = args.get("ref", "/opt/glm53/work-s14b/refpt/aime24_prefix_L0-2.refpt")
tag = args.get("tag", f"m5c-tf-B{B}")
TPN = int(mesh.get_num_devices())
cfg = hf_config()
if args.get("layers") is not None:  # the model's own layer ids ("all" = the 45-layer model) -> the three kinds
    ids = list(range(cfg.num_hidden_layers)) if args["layers"] == "all" else [int(v) for v in args["layers"]]
    DENSE = [L for L in ids if cfg.layer_kind(L) == "kda_dense"]
    SPARSE = [L for L in ids if cfg.layer_kind(L) == "kda_moe"]
    DSA = [L for L in ids if cfg.layer_kind(L) == "mla_moe"]


def log(*a):
    print(f"M5C-TF {C.now()}", *a, flush=True)


key = ("m5c_tf", bool(args.get("native", False)), B, tuple(DENSE), tuple(SPARSE), tuple(DSA), TPN, args.get("rev"))
if args.get("serving") and TPN > 1:  # the gate batch's shared backend (m5common.serving)
    key = M5.serving(ctx, mesh, SM, G, B=B, ids=sorted(DENSE + SPARSE + DSA),
                     cfg=cfg, native=bool(args.get("native", False)),
                     rev=args.get("rev"), log=log)  # fmt: skip
else:
    M5.prune_ctx(ctx, [key])
if key not in ctx:
    t0 = time.time()
    M5.set_env({"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "700"})
    if TPN == 1:
        M5.keep_decoders(ctx, DENSE + SPARSE + DSA)
        # the backend's stacks run 4-row chunks: the sparse layers' T=4 expert ring (3 slots, 2 in flight)
        env_s = {**M5.SPARSE_KNOBS, "GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"} if DSA else M5.SPARSE_KNOBS
        seq = M5.build_layers(mesh, cfg, ctx, DENSE, SPARSE, M5.DENSE_KNOBS, env_s, dsa=DSA)
        backend = SM.build_model_backend(
            None, mesh, max_batch=B, seq=seq, hf_config=cfg, native_t=bool(args.get("native", False))
        )
    else:
        M5.keep_decoders(ctx, [])
        backend = SM.build_model_backend(
            None,
            mesh,
            max_batch=B,
            layers=sorted(DENSE + SPARSE + DSA),
            hf_config=cfg,
            native_t=bool(args.get("native", False)),
            weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights") else None,
        )
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
    t1 = time.time()
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
    log(f"backend {backend.trace_report()} built in {t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s")
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
c0 = dict(gen.counters)
r0 = dict(backend.trace_report())
t0 = time.time()
per_entry = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(REF), mesh_device=mesh,
                                build_kwargs=dict(generator=gen))  # fmt: skip
wall = time.time() - t0
c1 = dict(gen.counters)
d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
r1 = backend.trace_report()
s = per_entry[0]
traced = d.get("eager_decode_steps", 0) == 0 and r1.get("decode_eager", 0) == r0.get("decode_eager", 0) and \
    d.get("decode_trace_replays", 0) > 0  # fmt: skip
names = [f"L{e['L']}" for e in backend.seq]
ok = traced and s["top5"] >= 0.98 and s["top100"] >= 1.0
full = len(names) == cfg.num_hidden_layers
if full:  # the whole model vs the 45-layer GPU fp32 reference: the goal's accuracy criterion
    what = f"the full model ({len(names)} layers) + head, AIME24 prompt, vs the GPU fp32 reference {os.path.basename(REF)}"
    note = "full model vs the GPU fp32 reference: the model's TF accuracy on the device (Stage 6 TokenAccuracy rule)"
else:
    what = f"prefix {names} + head, AIME24 prompt, vs the same prefix's CPU fp32 reference"
    note = "prefix model: scores the device numerics + the harness path, not the model's accuracy"
r = dict(test=f"m5c Stage 6 TF harness (plugin run_teacher_forcing) on StackBackend, {what}, B={B} tp{TPN}",
         status="pass" if ok else "fail",
         milestone="M5c", B=B, tp=TPN, prefix=names, reference=REF, top1=s["top1"], top5=s["top5"],
         top100=s["top100"], matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]],
         decode_traced_no_eager=traced, counters_delta=d, trace_report=r1, dram=M5.dram_state(mesh), tag=tag,
         full_model=full, note=note)  # fmt: skip
prow = dict(test=f"m5c TF harness decode B={B} tp{TPN} {'full model' if full else 'prefix'} {len(names)} layers + head",
            status="info",
            milestone="M5c", traced=True, B=B, tp=TPN, us=round(1e6 / s["decode_t/s/u"], 1) if s.get("decode_t/s/u")
            else None, ttft_ms=s.get("ttft_ms"), wall_s=round(wall, 1), tag=tag,
            note="harness wall per decode token (one execute_trace + the generator's token handling per step)")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, top1=s["top1"], top5=s["top5"], top100=s["top100"])
