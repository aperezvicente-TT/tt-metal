# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job module (tests/s7a/serve.py --mgd pair, job "mesh": "pair"): Stage 14's TP harness (tests/s14/jobs/m3_tp.py) on
7a's card1 — the TP / virtual-TP8 sparse layer with this tree's MoE (routing, placement), per-phase probes. ARGS = the
JSON in GLM53_S14_ARGS; rows -> GLM53_OUT_DIR (s14common, default 7a's wm_tp/); references: $GLM53_WORK/m3_ref (a link
to Stage 14's, read only). m3_tp sets its knobs in os.environ: the whole environment is restored afterwards."""

import json
import os
import runpy

E7 = "/opt/glm53/kit/doc/ttnn/07a-fused-streaming-layer"
M3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "s14", "jobs", "m3_tp.py")


def main(mesh=None, ctx=None):
    saved = dict(os.environ)
    os.environ.setdefault("GLM53_OUT_DIR", os.path.join(E7, "wm_tp"))
    os.makedirs(os.environ["GLM53_OUT_DIR"], exist_ok=True)
    tests = os.path.join(os.environ["GLM53_OUT_DIR"], "tests.jsonl")
    n0 = sum(1 for _ in open(tests)) if os.path.exists(tests) else 0
    try:
        args = json.loads(os.environ.get("GLM53_S14_ARGS", "{}"))
        ns = runpy.run_path(M3, init_globals={"MESH": mesh, "ARGS": args, "CTX": ctx}, run_name="__job__")
        res = ns.get("RESULT")
        print(f"S14M3_RESULT {json.dumps(res, default=str)[:4000]}", flush=True)
        # the job fails (serve.py: .failed) when any row it recorded failed (virtual-TP rows are status info)
        new = (
            [json.loads(line) for i, line in enumerate(open(tests)) if i >= n0 and line.strip()]
            if os.path.exists(tests)
            else []
        )
        bad = [r.get("test") for r in new if r.get("status") == "fail"]
        if bad:
            print(f"S14M3_FAIL rows: {bad}", flush=True)
        return 1 if bad else 0
    finally:
        os.environ.clear()
        os.environ.update(saved)
        # m3_tp reloads the model modules while ITS knobs are set (module-level constants read the environment at
        # import): after it, every later job of the process ran with those constants (card1 08:06Z: the sparse layer
        # gave NaN in every job after the m3_tp jobs). Reload every tt module with the restored environment (two
        # passes: the second rebinds `from x import y`) and drop m3_tp's resident objects (built from the old classes).
        # GLM53_S14M3_NO_RELOAD=1 (diagnosis only) keeps the old behaviour.
        if saved.get("GLM53_S14M3_NO_RELOAD") != "1":
            import gc
            import importlib
            import sys

            pkg = "models.autoports.zai_org_glm_5_3_flash.tt."
            names = [n for n in list(sys.modules) if n.startswith(pkg) and sys.modules[n] is not None]
            for _ in range(2):
                for n in names:
                    importlib.reload(sys.modules[n])
            for k in [k for k in ctx if isinstance(k, tuple) or k == "m3_rev"]:
                ctx.pop(k)
            gc.collect()
            print(f"S14M3_RELOADED {len(names)} tt modules, m3_tp resident objects dropped", flush=True)
