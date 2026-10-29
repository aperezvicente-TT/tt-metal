# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P2 prefill-then-decode TF on 14b's SERVED construction (M5.serving: B decode rows,
native widths, the served env -- weights from the TP image when GLM53_WEIGHT_SOURCE=image): the Stage 6 TF harness
(readiness_check.run_teacher_forcing) through Glm53Generator, whose prompt prefill is the TP prefill
(GLM53_STACK_PREFILL=tp at build), per reference (default: the program's official AIME24 readiness refpt, 161 prompt
ids + 100 forced decode steps; and the 2,048-token prompt refpt, 32 steps). Run FIRST in a process: a TPPrefill rebuilt
later (p2_layer_golden / p4_packed adopt) makes the decode replays refuse.

ARGS: layers ("all"), b (32), native (true), rev ("t25b"), refs ([official AIME24, prefill2048_full]), env (the served
env, applied before the build), tag, record (True)."""
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

C = S.C
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.run_teacher_forcing import run_teacher_forcing  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s14b"))
import m5common as M5  # noqa: E402

MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
REFS = list(args.get("refs", [f"{WS}/goldens_gpu_fp32_aime24/readiness_aime24_chat.refpt",
                                "/opt/glm53/work-s16/refpt/prefill2048_full.refpt"]))  # fmt: skip
tag = args.get("tag", "p2-tf-m5")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
key = M5.serving(ctx, mesh, SM, G, B=int(args.get("b", 32)), ids=LAYERS, cfg=cfg, native=bool(args.get("native", True)),
                 rev=args.get("rev", "t25b"), log=S.log)  # fmt: skip
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
assert getattr(backend, "tpp", None) is not None, "the TF prompt prefill must be the TP prefill (GLM53_STACK_PREFILL=tp)"
full = len(LAYERS) == cfg.num_hidden_layers
rows = []
for ref in REFS:
    gen.reset()
    s = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(ref), mesh_device=mesh,
                            build_kwargs=dict(generator=gen))[0]  # fmt: skip
    r = dict(test=f"p2 prefill-then-decode TF (Stage 6 harness) on 14b's SERVED construction (M5.serving B "
                  f"{args.get('b', 32)}, TP prefill), reference {os.path.basename(ref)}, tp{TPN} layers {len(LAYERS)}",
             milestone="P2" if full else "P2-prefix", status="pass" if s["top5"] >= 0.98 else "fail", tp=TPN,
             layers=len(LAYERS), top1=s["top1"], top5=s["top5"], top100=s["top100"],
             matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]], reference=ref,
             construction="m5", tag=tag)  # fmt: skip
    S.log(r)
    rows.append(r)
    if rec:
        C.record("pcc", r)
gen.reset()
RESULT = rows
