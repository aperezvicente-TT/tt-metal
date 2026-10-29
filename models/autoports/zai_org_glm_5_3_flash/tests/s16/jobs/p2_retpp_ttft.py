# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 TTFT (+ TF) of a prefill-knob variant on the SAME process's generator build (no
second full-model build): adopt the CTX "s16-full" TP-prefill build of these layers (p2_tf_ab / p2_pad_leak / p2_full),
release its TPPrefill, set the knob env (args env), build + compile + capture a new TPPrefill on the same backend and KV
cache, then TTFT through Glm53Generator.generate(prompt, 1) per n (median of reps; the first token sampled on device)
and the last chunk's trace replayed alone; then (args tf) the Stage 6 TF harness (its forced decode replays the decode
traces captured before this rebuild -- run last).

ARGS: layers ("all"), ns ([2048, 512, 128]), reps (3), tf (False: the decode replay is refused after the rebuild),
ref, env (the variant's knobs), tag, record (True)."""
import os
import sys
import time
from pathlib import Path

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

C = S.C
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.run_teacher_forcing import run_teacher_forcing  # noqa: E402

MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
NS = [int(v) for v in args.get("ns", [2048, 512, 128])]
REPS = int(args.get("reps", 3))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_full.refpt")
tag = args.get("tag", "p2-retpp-ttft")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

full = [k for k in ctx if isinstance(k, tuple) and k and ((k[0] == "s16-full" and k[1] == "tp" and list(k[2]) == LAYERS)
                                                            or (k[0] == "m5_serving" and list(k[2]) == LAYERS))]  # fmt: skip
assert full, "needs a same-process TP-prefill generator build of these layers (p2_tf_ab / p2_pad_leak / M5.serving)"
backend, gen = ctx[full[-1]]["backend"], ctx[full[-1]]["gen"]
kv = backend.tpp.kv
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})
t0 = time.time()
backend.tpp.release()
backend.tpp = None
backend.prepare_prefill(kv)  # GLM53_STACK_PREFILL=tp: a new TPPrefill (the variant's knobs) + every bucket compiled
pf = backend.tpp
pf.capture()
gen.reset()  # the earlier jobs' requests (prefill_forward leaves its slot live)
io = gen._binding(kv).io  # the binding of the TPPrefill's cache (own or caller-owned)
ttnn.synchronize_device(mesh)
S.log(f"TPPrefill rebuilt + captured in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}")
greedy = SamplingParams()
knobs = {k: v for k, v in (args.get("env") or {}).items()}
rows = []
for n in NS:
    toks = S.golden_prompt(n)
    eager0 = pf.counters["eager"]
    walls, first = [], None
    for rep in range(REPS + 1):
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        # the production prefill entry (StackBackend.prefill_chunk: one TP-prefill replay, the first token sampled on
        # device and read back) -- not generate(): its decode-trace replay is refused after this rebuild (buffers
        # allocated after the decode capture: TT_METAL_TRACE_ALLOC_TRACKING)
        first = int(backend.prefill_chunk(toks.reshape(-1).long(), kv_cache=kv, slot=0, chunk_start=0, n_valid=n,
                                          logits_rows="last", sample=greedy, io=io))  # fmt: skip
        if rep:
            walls.append((time.perf_counter() - t1) * 1e3)
    c, t_win = pf.plan(0, n)
    rs = []
    for _ in range(REPS):
        t1 = time.perf_counter()
        pf.replay(c, blocking=True, t=t_win)
        rs.append((time.perf_counter() - t1) * 1e6)
    st, rst = C.stats([w * 1e3 for w in walls]), C.stats(rs)
    ms = "P2" if (n == 2048 and len(LAYERS) == cfg.num_hidden_layers) else f"P2-n{n}"
    row = dict(test=f"TTFT B=1 prompt {n} tp{TPN} layers {len(LAYERS)} (StackBackend.prefill_chunk, TP prefill rebuilt with "
                    f"{knobs}, traced, first token on device)", milestone=ms, status="info",
               traced=pf.counters["eager"] == eager0, tp=TPN, n=n, layers=len(LAYERS), ms=round(st["median_us"] / 1e3, 2),
               ms_min=round(st["min_us"] / 1e3, 2), last_replay_ms=round(rst["median_us"] / 1e3, 2), first_token=first,
               knobs=knobs, tag=tag)  # fmt: skip
    S.log(row)
    rows.append(row)
    if rec:
        C.record("perf", row)
if args.get("tf", False):  # the forced decode replays the decode traces: refused after this rebuild (see above)
    s = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(REF), mesh_device=mesh,
                            build_kwargs=dict(generator=gen))[0]  # fmt: skip
    r = dict(test=f"p2 prefill-then-decode TF (Stage 6 harness), TP prefill rebuilt with {knobs}, tp{TPN} layers "
                  f"{len(LAYERS)}", milestone="P2" if len(LAYERS) == cfg.num_hidden_layers else "P2-prefix",
             status="pass" if s["top5"] >= 0.98 else "fail", tp=TPN, layers=len(LAYERS), top1=s["top1"], top5=s["top5"],
             top100=s["top100"], matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]],
             reference=REF, knobs=knobs, tag=tag)  # fmt: skip
    S.log(r)
    rows.append(r)
    if rec:
        C.record("pcc", r)
RESULT = rows
