# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the mesh; process started with TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_TRACEBACKS=1):
Stage 14b — the "a later capture leaves one buffer per sparse_dsa layer alive" defect (blocks every multi-trace scheme:
native-t width buckets, per-chunk prefill traces). Two backends that share the cached layers (weight cache: the
sparse_dsa layers' TPDsaLayer objects are SHARED): A = native B, captured first; B = B=1, captured second (keeping A
alive); then A is replayed -> the tracker logs the buffers allocated during B's capture that are still alive (with the
allocation tracebacks); GLM53_TRACE_LEAK_DEBUG=1 adds the Python-side holders. Finally both replay and the live rows of
A are compared with A's first replay (bitwise: a leaked buffer under A's intermediates corrupts them).
ARGS: b (32), layers ([0, 1, 2, 3, 4]), native (true), shared_weights, rev."""
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
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", True))
cfg = hf_config()
_L = args.get("layers", [0, 1, 2, 3, 4])
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5B-LEAK {C.now()}", *a, flush=True)


M5.prune_ctx(ctx, [])
M5.keep_decoders(ctx, [])
wc = ctx.setdefault("s14b_weights", {}) if args.get("shared_weights", True) else None
os.environ["GLM53_TRACE_LEAK_DEBUG"] = "1"
try:
    beA = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=NATIVE, weight_cache=wc)
    genA = G.build_generator(None, mesh, backend=beA, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                             stop_token_ids=())  # fmt: skip
    genA.warmup()
    log("A captured", beA.trace_report())
    kvA = beA._kv
    ioA = kvA.extra["io"]

    def stepA(n):
        beA._kv = kvA
        beA.reset_all(kvA)
        beA.write(ioA.tokens, torch.tensor([1000 + 13 * u for u in range(B)]))
        beA.write(ioA.cur_pos, torch.tensor([7 * u for u in range(B)]))
        out = []
        for _ in range(n):
            beA.replay(kvA.extra["decode_trace"])
            ttnn.synchronize_device(mesh)
            out.append(beA.read(ioA.logits)[:B].clone())
        return out

    ref = stepA(3)
    if args.get("mode") == "same":  # the second capture in the SAME backend (no second L1 set): the dsa leak alone
        beA._kv = kvA
        tid2 = beA._capture(kvA, beA._width_chunks(B))
        log("A captured twice", beA.trace_report())
        again = stepA(3)
        same = all(torch.equal(x, y) for x, y in zip(ref, again))
        beA.replay(tid2)
        ttnn.synchronize_device(mesh)
        res = dict(B=B, layers=len(ids), native_t=NATIVE, mode="same", A_bitwise_after_second_capture=same,
                   dsa_layers=sum(1 for e in beA.seq if e["sparse"] == "dsa"))  # fmt: skip
        log("RESULT", res)
        ctx[("m5b_leak", "A")] = dict(backend=beA, gen=genA)
        raise StopIteration
    beB = SM.build_model_backend(None, mesh, max_batch=1, layers=ids, hf_config=cfg, native_t=False, weight_cache=wc)
    genB = G.build_generator(None, mesh, backend=beB, tokenizer=load_tokenizer(), max_batch=1, max_seq_len=512,
                             stop_token_ids=())  # fmt: skip
    genB.warmup()  # the SECOND capture (A alive)
    log("B captured", beB.trace_report())
    again = stepA(3)  # the tracker reports here if B's capture left buffers alive under A's trace
    same = all(torch.equal(x, y) for x, y in zip(ref, again))
    kvB = beB._kv
    beB._kv = kvB
    beB.replay(kvB.extra["decode_trace"])
    ttnn.synchronize_device(mesh)
    res = dict(B=B, layers=len(ids), native_t=NATIVE, A_bitwise_after_B_capture=same,
               dsa_layers=sum(1 for e in beA.seq if e["sparse"] == "dsa"))  # fmt: skip
    log("RESULT", res)
    ctx[("m5b_leak", "A")] = dict(backend=beA, gen=genA)
    ctx[("m5b_leak", "B")] = dict(backend=beB, gen=genB)
except StopIteration:
    pass
finally:
    os.environ.pop("GLM53_TRACE_LEAK_DEBUG", None)
RESULT = dict(ok=True, **res)
