# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; mock or the mesh): Stage 14b — which persistent L1 tensors a StackBackend holds (the static-CB clash of
native B=32 without XSL, 2026-10-20 17:15Z: "L1 buffer allocated at 1,245,312, static CB region ends at 1,245,696",
cores (0,0)-(2,0), full model only). Builds the backend of ARGS layers (no warmup), then lists every live L1 device
tensor (gc): address, bytes per shard / bank, shard cores, owner type; FabricPhase instances by lane cores; the lowest
L1 address per core of the first grid row. ARGS: b (32), layers, native, shared_weights, rev."""
import collections
import gc
import importlib
import os
import sys

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.fabric_ar as FAR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", False))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5B-L1 {C.now()}", *a, flush=True)


key = ("m5b_l1", B, tuple(ids), NATIVE, os.environ.get("GLM53_STACK_NATIVE_WIDTHS"), args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    M5.keep_decoders(ctx, [])
    be = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=NATIVE,
                                weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                else None)  # fmt: skip
    ctx[key] = dict(backend=be)
    if args.get("warmup"):  # compile + capture every width (the stacks' pooled L1 intermediates exist after it)
        import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
        from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

        gen = G.build_generator(None, mesh, backend=be, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                                stop_token_ids=())  # fmt: skip
        gen.warmup()
        ctx[key]["gen"] = gen
gc.collect()
phases = [o for o in gc.get_objects() if isinstance(o, FAR.FabricPhase)]
by_lanes = collections.Counter(tuple(tuple(int(v) for v in ln) for ln in p.lanes) for p in phases)
l1 = {}
for o in gc.get_objects():
    if not isinstance(o, ttnn.Tensor):
        continue
    try:
        if not o.is_allocated():
            continue
        mc = o.memory_config()
        if mc.buffer_type != ttnn.BufferType.L1:
            continue
        addr = o.buffer_address()  # raises for a host tensor
    except Exception:  # noqa: BLE001 -- host / deallocated tensors
        continue
    if addr in l1:
        continue
    ss = mc.shard_spec
    cores = str(ss.grid) if ss is not None else "interleaved"
    l1[addr] = dict(addr=addr, shape=tuple(o.shape), dtype=str(o.dtype).split(".")[-1], layout=str(mc.memory_layout),
                    cores=cores[:120], shard=list(ss.shape) if ss is not None else None)  # fmt: skip
rows = sorted(l1.values(), key=lambda r: r["addr"])
for r in rows[:12]:
    log("L1", r)
log("SUMMARY", dict(B=B, native_t=NATIVE, layers=len(ids), fabric_phases=len(phases),
                    phases_by_lanes={str(k): v for k, v in by_lanes.items()}, l1_tensors=len(rows),
                    lowest_l1=rows[0]["addr"] if rows else None, l1_top=int(args.get("worker_l1", 1264512))))  # fmt: skip
RESULT = dict(ok=True, fabric_phases=len(phases), l1_tensors=len(rows), lowest_l1=rows[0]["addr"] if rows else None)
