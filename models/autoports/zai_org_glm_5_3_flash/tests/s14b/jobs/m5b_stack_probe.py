# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the mesh): Stage 14b — per-layer timeline of the one-trace decode step from the stacks' in-program
probes (program 16:15Z Q3: program start -> first weight byte, DRAM-busy vs span). The backend is built with
GLM53_STACK_PROBE=1 (every KDA stack program carries the probe table: layer l's marks at rows [256 l, 256 (l + 1)),
kernels/mk/probe.hpp; the sparse_dsa programs carry none: their time is the gap between two stacks). One step is
replayed; per KDA layer (chip 0, absolute wall clock): start (earliest mark), the weight streamer's first and last
mark (readers' rows 160 + 2n: start + one mark per streamed phase), end (latest mark) -> us from the step's first mark.
ARGS: b (1), layers ("all"), shared_weights, tag, record."""
import gc
import importlib
import os
import sys

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 1))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
MHZ = 1350.0


def log(*a):
    print(f"M5B-PROBE {C.now()}", *a, flush=True)


os.environ["GLM53_STACK_PROBE"] = "1"
key = ("m5b_probe", B, tuple(ids), int(mesh.get_num_devices()), args.get("rev"))
M5.prune_ctx(ctx, [key])
try:
    if key not in ctx:
        M5.keep_decoders(ctx, [])
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg,
                                         weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                         else None)  # fmt: skip
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                                max_seq_len=512, stop_token_ids=())  # fmt: skip
        gen.warmup()
        ctx[key] = dict(backend=backend, gen=gen)
finally:
    os.environ.pop("GLM53_STACK_PROBE", None)
backend = ctx[key]["backend"]
kv = backend._kv
io = kv.extra["io"]
backend.write(io.tokens, torch.tensor([1000 + 7 * u for u in range(B)]))
backend.write(io.cur_pos, torch.tensor([256 + u for u in range(B)]))
backend.replay(kv.extra["decode_trace"])
ttnn.synchronize_device(mesh)
rows = []
for r, stk in kv.stacks[0]:
    if stk is None or stk.probe is None:  # sparse_dsa programs / sparse_kda stacks (no probe table in MkSparseLayer)
        rows.append(dict(layers=[backend.seq[i]["L"] for i in r],
                         kind="sparse_dsa" if stk is None else "sparse_kda stack", probe=None))  # fmt: skip
        continue
    d = ttnn.get_device_tensors(stk.probe)[0] if backend.tp > 1 else stk.probe
    p = ttnn.to_torch(d).to(torch.int64).reshape(stk.n, -1, 16) & 0xFFFFFFFF
    for j in range(stk.n):
        q = p[j]
        nz = q[q > 0]
        if not nz.numel():
            continue
        rd = q[160::2][:, :]  # weight streamer rows (160 + 2n)
        rdn = rd[rd > 0]
        rows.append(dict(L=backend.seq[r[j]]["L"], kind="sparse_kda" if backend.seq[r[j]]["sparse"] else "dense_kda",
                         start=int(nz.min()), end=int(nz.max()),
                         w_first=int(rdn.min()) if rdn.numel() else None,
                         w_last=int(rdn.max()) if rdn.numel() else None))  # fmt: skip
abs_rows = [x for x in rows if x.get("start") is not None and "start" in x]
t0 = min(x["start"] for x in abs_rows) if abs_rows else 0
us = lambda v: None if v is None else round((v - t0) / MHZ, 1)  # noqa: E731
table = []
prev_end = None
for x in rows:
    if x.get("probe", 1) is None:
        table.append(dict(L=x["layers"], kind=x["kind"] + " (no probe)"))
        continue
    e = dict(L=x["L"], kind=x["kind"], start_us=us(x["start"]), first_weight_us=us(x["w_first"]),
             stream_end_us=us(x["w_last"]), end_us=us(x["end"]),
             span_us=round((x["end"] - x["start"]) / MHZ, 1),
             start_to_first_weight_us=None if x["w_first"] is None else round((x["w_first"] - x["start"]) / MHZ, 1),
             gap_from_prev_us=None if prev_end is None else round((x["start"] - prev_end) / MHZ, 1))  # fmt: skip
    prev_end = x["end"]
    table.append(e)
    log("LAYER", e)
kda = [e for e in table if "span_us" in e]
summ = dict(B=B, layers=len(ids), kda_layers=len(kda),
            kda_span_sum_us=round(sum(e["span_us"] for e in kda), 1),
            gaps_sum_us=round(sum(e["gap_from_prev_us"] or 0 for e in kda), 1),
            start_to_first_weight_median_us=sorted(e["start_to_first_weight_us"] or 0 for e in kda)[len(kda) // 2]
            if kda else None, step_us_probe=us(max(x["end"] for x in abs_rows)) if abs_rows else None)  # fmt: skip
log("SUMMARY", summ)
r = dict(test=f"m5b per-layer probe timeline of the one-trace decode step ({len(ids)} layers, B={B}, tp"
              f"{backend.tp}): program start -> first weight byte, span, inter-program gaps", status="info",
         milestone="M5b", **summ, tag=args.get("tag", "m5b-probe"))  # fmt: skip
if args.get("record") and not os.environ.get("TT_METAL_WATCHER"):
    C.record("perf", r)
RESULT = dict(ok=True, **summ)
