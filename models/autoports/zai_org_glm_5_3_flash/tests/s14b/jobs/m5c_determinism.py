# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the mesh / pair): Stage 14b — run-to-run determinism and batch invariance of the bare decode trace
(the B=32 generator nondeterminism of 2026-10-20 16:27Z / 16:38Z, isolated from the generator).

From the same device state (reset_all, the same token / position rows written) the decode trace is replayed `steps`
times, twice (A1, A2): the head's rows (the K best ids + keys of every row) and the logits of the active rows must be
BITWISE equal between A1 and A2 (run-to-run determinism). Then run B: the same active rows, but the INACTIVE rows hold
other tokens (garbage that differs): the active rows' outputs must still be bitwise equal to A (batch invariance: an
inactive row must not influence an active one). Reports the first step / row / layer kind that differs.
ARGS: b (32), layers ("all" or ids), active (16: rows 0..active-1 decode, the rest inactive), mixed (even rows active),
native, topk, steps (6), shared_weights."""
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
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, MKH, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NA = int(args.get("active", 16))
STEPS = int(args.get("steps", 6))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5C-DET {C.now()}", *a, flush=True)


TOPK = int(args.get("topk", 32))  # 32 at B=32 = the head's own fabric phase (rows in whole tiles); 5 = the layers'
NATIVE = bool(args.get("native", False))
key = ("m5c_det", B, tuple(ids), int(mesh.get_num_devices()), TOPK, NATIVE, args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    M5.keep_decoders(ctx, [])
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, topk=TOPK, native_t=NATIVE,
                                     weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                     else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
backend = ctx[key]["backend"]
kv = backend._kv
io = kv.extra["io"]
trace = kv.extra["decode_trace"]


# mixed (2026-10-20 17:50Z): the active rows interleaved with inactive ones (even rows active) -> every row chunk holds
# both; without it the inactive rows were whole chunks (rows NA..B-1) and an intra-chunk coupling stayed invisible
MIXED = bool(args.get("mixed", False))
ACT = list(range(0, 2 * NA, 2)) if MIXED else list(range(NA))
assert max(ACT) < B, (NA, B)


def run(inactive_tok: int):
    backend.reset_all(kv)
    tok = torch.full((B,), inactive_tok, dtype=torch.int64)
    tok[ACT] = torch.tensor([1000 + 13 * u for u in range(NA)])
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[ACT] = torch.tensor([7 * u for u in range(NA)])
    backend.write(io.tokens, tok)
    backend.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    rows, logits = [], []
    for _ in range(STEPS):
        backend.replay(trace)
        ttnn.synchronize_device(mesh)
        d = ttnn.get_device_tensors(backend.head.topk_out)[0] if backend.tp > 1 else backend.head.topk_out
        rows.append((ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF)[ACT].clone())
        logits.append(backend.read(io.logits)[ACT].clone())
    return rows, logits


def first_diff(x, y):
    for s, (a, b) in enumerate(zip(x, y)):
        if not torch.equal(a, b):
            r = int((a != b).reshape(a.shape[0], -1).any(-1).nonzero()[0])
            return dict(step=s, row=r, n_rows=int((a != b).reshape(a.shape[0], -1).any(-1).sum()))
    return None


a1 = run(0)
if args.get("probe_res"):  # the persistent residual after the step (the last layer's output, every row; chip 0)
    res = kv.res[0] if isinstance(kv.res, (list, tuple)) else kv.res
    d = ttnn.get_device_tensors(res)[0] if backend.tp > 1 else res
    x = ttnn.to_torch(d).float()
    x = x.reshape(-1, x.shape[-1]) if x.dim() > 1 else x
    rows = x.shape[0]
    act = set(ACT)
    fin = torch.isfinite(x).all(-1)
    mx = x.abs().nan_to_num(posinf=float("inf")).amax(-1)
    log("RES", dict(shape=list(ttnn.to_torch(d).shape), rows=rows,
                    active_nonfinite=[r for r in range(min(rows, B)) if r in act and not bool(fin[r])][:8],
                    inactive_nonfinite=[r for r in range(min(rows, B)) if r not in act and not bool(fin[r])][:8],
                    active_max=round(float(max([float(mx[r]) for r in range(min(rows, B)) if r in act] or [0])), 3),
                    inactive_max=float(max([float(mx[r]) for r in range(min(rows, B)) if r not in act] or [0]))))  # fmt: skip
a2 = run(0)
b1 = run(77777)
res = dict(rows_A1_vs_A2=first_diff(a1[0], a2[0]), logits_A1_vs_A2=first_diff(a1[1], a2[1]),
           rows_A_vs_B=first_diff(a1[0], b1[0]), logits_A_vs_B=first_diff(a1[1], b1[1]))  # fmt: skip
ok = all(v is None for v in res.values())
log("RESULT", dict(B=B, active=NA, mixed=MIXED, steps=STEPS, layers=len(ids), topk=TOPK, native_t=NATIVE,
                   head_own_fabric=backend.head.fabric is not getattr(backend.fl, "fabric", None), **res))
r = dict(test=f"m5c decode trace determinism + batch invariance ({len(ids)} layers, B={B}, {NA} active rows, "
              f"{STEPS} steps{', interleaved (mixed chunks)' if MIXED else ''}{' native-t' if NATIVE else ''}; "
              f"bitwise rows + logits)", status="pass" if ok else "record" if args.get("accepted") else "fail",
             outcome="pass" if ok else "fail", accepted=args.get("accepted"), milestone="M5c", mixed=MIXED,
         native_t=NATIVE, layers=len(ids), **res)  # fmt: skip
if args.get("record"):
    C.record("tests", r)
RESULT = dict(ok=ok, **res)
