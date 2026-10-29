# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, long-context rungs): the decode step time of a CTX-resident StackBackend (the one a previous
TF job of this process left: dsa_stack_tf / m5c_tf_lag keys) at the context its TF run filled, as traced replays of its
ONE decode trace: rows at ARGS pos (an int = every row at pos - u for row u, so B rows sit at B distinct positions; or a
list), token 0. Per step: the median of `replays` blocking replays (execute + sync) and the back-to-back batch mean
(launch overlap amortised = the device-bound step). Row u's KV / latent / pooled-key rows below its position are the TF
run's; each replay rewrites row u's own position only.
ARGS: key (CTX key prefix, "dsa_stack_tf" | "m5c_tflag"), pos, replays (500), tag, record (-> GLM53_STAGE_EVIDENCE)."""

import json
import os
import time

import torch

import ttnn

prefix = ARGS.get("key", "dsa_stack_tf")  # noqa: F821
keys = [k for k in CTX if isinstance(k, tuple) and k and str(k[0]) == prefix]  # noqa: F821
assert keys, f"no resident {prefix} backend in this process (run the TF job first, same server)"
ent = CTX[keys[-1]]  # noqa: F821
backend = ent["backend"]
kv = backend._kv
io, trace = kv.extra["io"], kv.extra["decode_trace"]
B = int(backend.max_batch)
p = ARGS.get("pos", 32767)  # noqa: F821
pos = [int(v) for v in p] if isinstance(p, list) else [int(p) - u for u in range(B)]
assert len(pos) == B, (len(pos), B)
n = int(ARGS.get("replays", 500))  # noqa: F821
backend.write(io.tokens, torch.zeros(B, dtype=torch.int64))
backend.write(io.cur_pos, torch.tensor(pos, dtype=torch.int64))
mesh = MESH  # noqa: F821
backend.replay(trace)
ttnn.synchronize_device(mesh)
d = []
for _ in range(n):
    t0 = time.perf_counter()
    backend.replay(trace)
    ttnn.synchronize_device(mesh)
    d.append((time.perf_counter() - t0) * 1e6)
med = round(sorted(d)[len(d) // 2], 1)
t0 = time.perf_counter()
for _ in range(n):
    backend.replay(trace)
ttnn.synchronize_device(mesh)
batch = round((time.perf_counter() - t0) * 1e6 / n, 1)
rep = backend.trace_report()
r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test=os.environ.get("GLM53_ROW_PREFIX", "") +
         "stack_step", status="info", execution="traced", B=B, tp=int(mesh.get_num_devices()),
         layers=[f"L{e['L']}" for e in backend.seq], pos_min=min(pos), pos_max=max(pos), replays=n,
         timing_us={"step": med, "step_batch": batch}, dsa_decode=getattr(backend, "dsa_decode", None),
         dram_gib_per_chip=rep.get("dram_gib_per_chip"), key=str(keys[-1][:3]), tag=ARGS.get("tag", "stack-step"),  # noqa: F821
         watcher=bool(os.environ.get("TT_METAL_WATCHER")))  # fmt: skip
print("STACK_STEP ROW", json.dumps(r, default=str), flush=True)
ev = os.environ.get("GLM53_STAGE_EVIDENCE")
if ARGS.get("record") and ev:  # noqa: F821
    for name in ("pcc.jsonl", "perf.jsonl"):
        with open(os.path.join(ev, name), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
RESULT = dict(ok=True, step_us=med, step_batch_us=batch)
