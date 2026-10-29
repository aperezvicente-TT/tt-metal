# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 multi-user correctness of the batched TP prefill in the generator: B users of
different prompt lengths through Glm53Generator.generate_batch (every user prefilled into its slot by ONE traced
replay per chunk, then the batched decode) vs the same prompts one at a time through generate (slot 0) -- greedy
tokens must match per user (a prefill replay rewrites every row of the head's outputs; the other slots' state, tokens,
positions and sampler rows must come back). Rows: per-user match, the batch's wall time.

ARGS: layers ([0..4]), b (32), native (True), lens ([2048, 512, 128, 1000]: one user each), new (8), pf_batch (0: > 1 =
GLM53_PF_BATCH, the generator admits groups of that many fresh users in ONE replay of the batched-prefill trace),
knobs (the s16common.set_attn args + mhc_post / moe_v2), tag, record (True)."""
import os
import sys
import time

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
B = int(args.get("b", 32))  # the serving configuration (Stage 9: B = 32 native); B = 4 non-native: the absorbed head
NATIVE = bool(args.get("native", True))  # matmul's CBs clash with that decode stack's lower L1 floor (01:55Z mock)
LENS = [int(v) for v in args.get("lens", [2048, 512, 128, 1000])]
NEW = int(args.get("new", 8))
tag = args.get("tag", "p2-multiuser")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = "ep"
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MOE_V2"] = "1" if args.get("moe_v2", True) else "0"
os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_BATCH"] = str(int(args.get("pf_batch", 0)))
os.environ["GLM53_PF_PACKED"] = str(args.get("pf_packed", ""))  # "T:KMAX": packed admission of mixed lengths
S.set_attn(args)
key = ("s16-multi", tuple(LAYERS), B, NATIVE, TPN, os.environ["GLM53_PF_MHC_POST"], os.environ["GLM53_PF_MOE_V2"], S._attn_knobs(),
       os.environ["GLM53_PF_BATCH"], os.environ["GLM53_PF_PACKED"])  # fmt: skip
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-multi")
          and k != key]:  # fmt: skip
    old = ctx.pop(k)
    if "pf" in old:
        old["pf"].release()
    old["backend"].close()
if key not in ctx:
    cfg = hf_config()
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg, native_t=NATIVE,
                                     weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
    # max_seq_len >= the 2,048 bucket: its chunked SDPA reads the slot's whole page row (kv length >= the chunk)
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                            max_seq_len=max(max(LENS) + NEW + 64, 2048), stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
gen, backend = ctx[key]["gen"], ctx[key]["backend"]
prompt = S.golden_prompt(4096)
prompts = [[int(v) for v in prompt[i * 37 : i * 37 + n]] for i, n in enumerate(LENS)]  # distinct prompts

eager0 = backend.tpp.counters["eager"]
singles = []
for p in prompts:
    singles.append([int(t) for t in gen.generate(p, NEW, stop_on_eos=False)])
ttnn.synchronize_device(mesh)
nbatch0 = gen.counters.get("prefill_batches", 0)
npacked0 = gen.counters.get("prefill_packed", 0)
t0 = time.perf_counter()
batch = gen.generate_batch(prompts, NEW, stop_on_eos=False)
wall = time.perf_counter() - t0
batch = [[int(t) for t in o] for o in batch]
match = [b_ == s_ for b_, s_ in zip(batch, singles)]
row = dict(test=f"multi-user batched TP prefill: generate_batch of {len(LENS)} users in a B={B} {'native ' if NATIVE else ''}"
                f"backend (prompt lengths {LENS}) vs one at a time, "
                f"{NEW} greedy tokens each, tp{TPN} layers {len(LAYERS)}", milestone="P2-prefix" if len(LAYERS) < 45 else "P2",
           status="pass" if all(match) else "fail", traced=backend.tpp.counters["eager"] == eager0, B=B, lens=LENS, new=NEW, match=match,
           first_diff=[next((j for j, (a, b_) in enumerate(zip(x, y)) if a != b_), None) for x, y in zip(batch, singles)],
           batch_wall_s=round(wall, 3), tp=TPN, pf_batch=int(os.environ["GLM53_PF_BATCH"]),
           batched_admissions=gen.counters.get("prefill_batches", 0) - nbatch0,
           packed_admissions=gen.counters.get("prefill_packed", 0) - npacked0, pf_packed=os.environ["GLM53_PF_PACKED"],
           tag=tag)  # fmt: skip
S.log(row)
if rec and not C.MOCK:
    C.record("pcc", row)
RESULT = row
