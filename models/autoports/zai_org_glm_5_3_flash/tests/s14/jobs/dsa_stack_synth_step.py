# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 long-context rungs on the mesh): the decode step of a CTX-resident DSA StackBackend
(GLM53_STACK_DSA=1; the backend a dsa_stack_tf job of this process built, e.g. warm_only) at context N WITHOUT feeding
N tokens: every sparse_dsa layer's paged latent cache, pooled-key store and ring are filled with synthetic bf16 values
(seeded normal, latent scaled like RMS-normed rows) and the KDA states are the backend's (reset). Valid at B = 1 (within
+-3 % of real fills, 8K..128K, d0 2026-10-22); NOT at B = 32: the radix select runs on compacted candidates whose count
depends on the score distribution (random keys keep far more: 2.4x the real-context step at 128K). Then the traced step at rows u = N - 1 - u: blocking median + back-to-back batch mean.
Timing rows only (numerics: dsa_stack_tf / stack_step on real context).
ARGS: key ("dsa_stack_tf"), n (131072), replays (500), seed (0), tag, record (-> GLM53_STAGE_EVIDENCE)."""

import json
import os
import time

import torch

import ttnn

prefix = ARGS.get("key", "dsa_stack_tf")  # noqa: F821
keys = [k for k in CTX if isinstance(k, tuple) and k and str(k[0]) == prefix]  # noqa: F821
assert keys, f"no resident {prefix} backend in this process (build it first: dsa_stack_tf warm_only)"
backend = CTX[keys[-1]]["backend"]  # noqa: F821
assert getattr(backend, "dsa_decode", False), "a GLM53_STACK_DSA=1 backend"
mesh = MESH  # noqa: F821
kv = backend._kv
io, trace = kv.extra["io"], kv.extra["decode_trace"]
B = int(backend.max_batch)
N = int(ARGS.get("n", 131072))  # noqa: F821
assert backend.supported_context >= N + B, (backend.supported_context, N)
n_rep = int(ARGS.get("replays", 500))  # noqa: F821
gen = torch.Generator().manual_seed(int(ARGS.get("seed", 0)))  # noqa: F821
repl = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None


def fill(dst, scale):
    """dst (a replicated DRAM tensor) <- seeded normal * scale, written through a same-shape device copy"""
    shape = tuple(int(v) for v in dst.shape)
    host = (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)
    tmp = ttnn.from_torch(host, dtype=dst.dtype, layout=dst.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                          mesh_mapper=repl)  # fmt: skip
    ttnn.copy(tmp, dst)
    ttnn.deallocate(tmp)
    return int(host.numel())


t0 = time.time()
backend.reset_all(kv)
n_el = 0
for i, lat in kv.extra["latent"].items():
    n_el += fill(lat, 1.0)  # the latent rows are RMS-normed (kv_a_layernorm): unit scale
    st = kv.extra.get("dsa", {}).get(i)
    if st is not None:
        n_el += fill(st["pools"], 1.0) + fill(st["ring"], 1.0)
ttnn.synchronize_device(mesh)
fill_s = round(time.time() - t0, 1)
pos = [N - 1 - u for u in range(B)]
backend.write(io.tokens, torch.zeros(B, dtype=torch.int64))
backend.write(io.cur_pos, torch.tensor(pos, dtype=torch.int64))
backend.replay(trace)
ttnn.synchronize_device(mesh)
d = []
for _ in range(n_rep):
    t1 = time.perf_counter()
    backend.replay(trace)
    ttnn.synchronize_device(mesh)
    d.append((time.perf_counter() - t1) * 1e6)
med = round(sorted(d)[len(d) // 2], 1)
t1 = time.perf_counter()
for _ in range(n_rep):
    backend.replay(trace)
ttnn.synchronize_device(mesh)
batch = round((time.perf_counter() - t1) * 1e6 / n_rep, 1)
rep = backend.trace_report()
r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test=os.environ.get("GLM53_ROW_PREFIX", "") +
         "dsa_stack_synth_step", status="info", execution="traced", B=B, tp=int(mesh.get_num_devices()),
         layers=len(backend.seq), n=N, pos_min=min(pos), pos_max=max(pos), synthetic_state=True, fill_s=fill_s,
         filled_elements=n_el, replays=n_rep, timing_us={"step": med, "step_batch": batch},
         dram_gib_per_chip=rep.get("dram_gib_per_chip"), key=str(keys[-1][:3]), tag=ARGS.get("tag", f"dsa-synth-step-{N}"),  # noqa: F821
         watcher=bool(os.environ.get("TT_METAL_WATCHER")),
         note="synthetic latent / pooled keys / ring (values do not change the DSA step's work): a timing row")  # fmt: skip
print("DSA_SYNTH_STEP ROW", json.dumps(r, default=str), flush=True)
ev = os.environ.get("GLM53_STAGE_EVIDENCE")
if ARGS.get("record") and ev:  # noqa: F821
    for name in ("pcc.jsonl", "perf.jsonl"):
        with open(os.path.join(ev, name), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
RESULT = dict(ok=True, step_us=med, step_batch_us=batch)
