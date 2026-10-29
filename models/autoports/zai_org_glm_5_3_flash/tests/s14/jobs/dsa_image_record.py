# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; a MOCK or a device session): DSA-in-image, step 2 (program 04:57:16Z, Stage 14's plan 05:02Z).
The served StackBackend construction (tests/s14b/m5common.serving's build: SM.build_model_backend, the generator's
weight cache) recorded under tt/tp_image.TPWeightIO("record") -- exactly what an image build records -- with the
sparse_dsa layers' DSA halves built at construction (GLM53_STACK_DSA_DEFER=0, defer=0) or deferred past the image
(defer=1: GLM53_STACK_DSA_DEFER=1, the attach_dsa after the record is NOT recorded). No image is written (build_image
is not called), no weight file is written.

Writes one JSON (args out): map_sha (tp_image.map_sha: kind / address / spec of every upload, call order), the upload
count and per-bank bytes, the op-row count and an op signature (name + output addresses), the DRAM view. Two sessions
with defer=0 must give the SAME map_sha and op signature (the construction is deterministic, so an image built with
the DSA halves replays); defer=1 vs defer=0 gives what the DSA halves add to the recorded construction.
ARGS: layers ("0,1,2,3,4" | "all"), b (32), native (1), defer (0), out (path), env ({knob: value})."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ttnn  # noqa: E402

mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
for k_, v_ in (args.get("env") or {}).items():
    os.environ[k_] = str(v_)
defer = int(args.get("defer", 0))
os.environ["GLM53_STACK_DSA_DEFER"] = str(defer)
os.environ.setdefault("GLM53_STACK_DSA", "1")

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI  # noqa: E402

cfg = hf_config()
L = str(args.get("layers", "0,1,2,3,4"))
ids = list(range(cfg.num_hidden_layers)) if L == "all" else [int(v) for v in L.split(",")]
B = int(args.get("b", 32))
wc: dict = {}
io = TI.TPWeightIO("record")
with io:
    be = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=bool(int(args.get("native", 1))),
                                weight_cache=wc)  # fmt: skip
rows = io.rows()
ups = rows["uploads"]
ops = [(r["name"], tuple(o["address"] for o in r["outs"])) for r in io.op_rows]
dsa_seq = [e["L"] for e in be.seq if e["sparse"] == "dsa"]
pending = [e["L"] for e in be.seq if e["sparse"] == "dsa" and getattr(e.get("tp"), "dsa_pending", False)]
try:
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    dram = dict(alloc_per_bank=int(mv.total_bytes_allocated_per_bank), banks=int(mv.num_banks))
except Exception as e:  # noqa: BLE001
    dram = dict(error=str(e)[:120])
out = dict(
    defer=defer, layers=ids, b=B, map_sha=TI.map_sha(rows), uploads=len(ups),
    upload_bytes_per_bank=int(sum(int(r.get("extent", 0)) for r in ups)),
    kinds={k: sum(1 for r in ups if r["kind"] == k) for k in sorted({r["kind"] for r in ups})},
    op_rows=len(ops), op_sig=hashlib.sha256(json.dumps(ops).encode()).hexdigest()[:16],
    dsa_layers=dsa_seq, dsa_pending=pending, dram=dram, mock=os.environ.get("GLM53_MOCK") == "1",
    env={k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_STACK_DSA", "GLM53_DSA_SHARD", "GLM53_MLA_DSA"))},
)
print("DSA_IMAGE_RECORD", json.dumps(out), flush=True)
if args.get("out"):
    with open(args["out"], "w") as f:
        json.dump(out, f, indent=1)
be.close()
RESULT = dict(ok=True, **{k: out[k] for k in ("defer", "map_sha", "uploads", "op_rows", "op_sig")})
