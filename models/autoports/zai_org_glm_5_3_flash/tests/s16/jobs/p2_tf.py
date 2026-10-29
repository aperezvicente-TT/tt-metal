# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P2 acceptance — PREFILL-THEN-DECODE teacher forcing through the real stack: the
Stage 6 harness (plugin readiness_check.run_teacher_forcing, unchanged) drives tt/generator.py on tt/stack_model.py's
StackBackend with GLM53_STACK_PREFILL=tp, so the prompt (2,016 tokens of the golden 2,048-token prompt) is prefilled by
the batched TP prefill (ONE traced replay per chunk: tt/prefill_tp.py) and the forced tokens run through the decode
stack's one-trace step, which reads the KDA states / latent pages the prefill wrote. Score: the device's token in the
reference's top-1 / top-5 / top-k (scripts/make_prefill_tf_refpt.py: the full model's = the CPU fp32 golden's top-32;
a prefix's = that prefix's CPU fp32 top-100). Bar (full model): top-5 >= 0.98.

ARGS: layers ("all" or ids), ref (refpt), b (1), max_seq (2560), tag, record, buckets ("128,512,2048")."""
import gc
import os
import sys
import time
from pathlib import Path

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

C = S.C
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.run_teacher_forcing import run_teacher_forcing  # noqa: E402

MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
B = int(args.get("b", 1))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_full.refpt")
tag = args.get("tag", "p2-tf")
TPN = int(mesh.get_num_devices())
os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
if args.get("moe"):
    os.environ["GLM53_PF_MOE"] = args["moe"]
if args.get("kda_scan"):
    os.environ["GLM53_PF_KDA_SCAN"] = args["kda_scan"]

key = ("s16-tf", tuple(LAYERS), B, TPN, os.environ["GLM53_PREFILL_BUCKETS"], int(args.get("max_seq", 2560)))
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "s16-tf" and k != key]:
    old = ctx.pop(k)
    old["gen"].teardown() if hasattr(old["gen"], "teardown") else None
    old["backend"].close()
if key not in ctx:
    t0 = time.time()
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg,
                                     weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                            max_seq_len=int(args.get("max_seq", 2560)), stop_token_ids=())  # fmt: skip
    t1 = time.time()
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
    S.log(f"tf backend layers={len(LAYERS)} built in {t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s "
          f"(prefill traces {sorted(backend.tpp.traces) if getattr(backend, 'tpp', None) else None})")  # fmt: skip
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
c0, r0 = dict(gen.counters), dict(backend.trace_report())
t0 = time.time()
per_entry = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(REF), mesh_device=mesh,
                                build_kwargs=dict(generator=gen))  # fmt: skip
wall = time.time() - t0
c1, r1 = dict(gen.counters), backend.trace_report()
d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
s = per_entry[0]
tp_chunks = r1.get("prefill_chunks_tp", 0) - r0.get("prefill_chunks_tp", 0)
traced = d.get("eager_decode_steps", 0) == 0 and r1.get("decode_eager", 0) == r0.get("decode_eager", 0) and \
    d.get("decode_trace_replays", 0) > 0 and tp_chunks > 0 and backend.tpp.counters["replays"] > 0  # fmt: skip
full = len(LAYERS) == cfg.num_hidden_layers
ok = traced and s["top5"] >= 0.98
r = dict(test=f"p2 prefill-then-decode TF (Stage 6 harness) {'full model' if full else f'prefix {LAYERS}'}: batched TP "
              f"prefill of {int(s.get('prompt_len', 0)) or 2016} tokens then forced decode, B={B} tp{TPN}",
         status="pass" if ok else "fail", milestone="P2" if full else "P2-prefix", B=B, tp=TPN, layers=len(LAYERS),
         reference=REF, top1=s["top1"], top5=s["top5"], top100=s["top100"],
         matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]], ttft_ms=s.get("ttft_ms"),
         prefill_chunks_tp=tp_chunks, decode_traced_no_eager=traced, counters_delta=d, tpp=dict(backend.tpp.counters),
         wall_s=round(wall, 1), tag=tag)  # fmt: skip
S.log(r)
if args.get("record", True):
    C.record("pcc", r)
RESULT = dict(ok=ok, top1=s["top1"], top5=s["top5"], ttft_ms=s.get("ttft_ms"))
