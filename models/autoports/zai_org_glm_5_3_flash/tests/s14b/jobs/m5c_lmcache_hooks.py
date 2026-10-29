# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; mock or device): Stage 14b — the two stack-side LMCache hooks (program 16:48Z, contract
doc/ttnn/15-lmcache/page_contract.md): kv_cache_connector_views() (per DSA layer the [num_blocks, 64, 512] bf16 view that
aliases the latent cache) and export_kda_state / import_kda_state (a slot's KDA checkpoint). Device checks: a slot's
state after a few decode steps, exported, imported into another (zeroed) slot and exported again is bitwise equal; the
neighbour rows of the target chunk are unchanged; a view written through is read back from the cache tensor.
ARGS: b (8), layers ([0, 1, 2, 3, 4]), src (5), dst (2)."""
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
B = int(args.get("b", 8))
ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
SRC, DST = int(args.get("src", 5)), int(args.get("dst", 2))


def log(*a):
    print(f"M5C-LMC {C.now()}", *a, flush=True)


NATIVE = bool(args.get("native", False))
key = ("m5c_lmc", B, tuple(ids), int(mesh.get_num_devices()), NATIVE, args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(), native_t=NATIVE,
                                     weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                     else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
backend = ctx[key]["backend"]
kv = backend._kv
io = kv.extra["io"]
views = backend.kv_cache_connector_views(kv)
vinfo = {k: (tuple(v.shape), str(v.dtype).split(".")[-1]) for k, v in views.items()}
# a few decode steps so the states are non-trivial
backend.reset_all(kv)
backend.write(io.tokens, torch.tensor([1000 + 11 * u for u in range(B)]))
backend.write(io.cur_pos, torch.tensor([3 * u for u in range(B)]))
for _ in range(4):
    backend.replay(kv.extra["decode_trace"])
ttnn.synchronize_device(mesh)
src = backend.export_kda_state(kv, SRC)
backend.reset_slot(kv, DST)
nb_before = [d for d in backend.export_kda_state(kv, DST ^ 1)]  # a neighbour row of DST's chunk (DST ^ 1: same chunk)
backend.import_kda_state(kv, DST, src)
dst = backend.export_kda_state(kv, DST)
nb_after = backend.export_kda_state(kv, DST ^ 1)
same = all(torch.equal(a["S"], b["S"]) and torch.equal(a["C"], b["C"]) for a, b in zip(src, dst))
nb_same = all(torch.equal(a["S"], b["S"]) and torch.equal(a["C"], b["C"]) for a, b in zip(nb_before, nb_after))
nonzero = any(bool(d["S"].abs().sum() > 0) for d in src)
ok = same and nb_same and len(src) == sum(1 for e in backend.seq if e["sparse"] != "dsa")
r = dict(test=f"LMCache hooks: connector views ({len(views)} DSA layers) + KDA checkpoint export / import round trip "
              f"(slot {SRC} -> {DST}, neighbour rows unchanged), B={B}, tp{int(mesh.get_num_devices())}",
         status="pass" if ok else "fail", milestone="M5c", views=vinfo, layers_exported=len(src), round_trip_bitwise=same,
         neighbour_unchanged=nb_same, state_nonzero=nonzero)  # fmt: skip
log("ROW", r)
if args.get("record"):
    C.record("tests", r)
RESULT = dict(ok=ok)
