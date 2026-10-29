# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 service prefill with STAGGERED arrivals (the vLLM plugin's mixed steps: a new
request's prefill replay runs while other requests are mid-decode). Through Glm53Generator.prefill_forward /
decode_forward on a caller-owned cache with vLLM page rows (blocks held only, -1 padding):
  solo      -- each request alone: prefill, then D teacher-forced decode steps (the reference logits);
  staggered -- A prefills and decodes; B arrives (prefill replay with A live) and both decode; C arrives (prefill with A
               and B live) and all three decode; each request's per-position logits vs its solo run (PCC, argmax).
The TP prefill's head rewrites every decode row (tokens / sampled / positions / sampler rows are restored by
StackBackend._prefill_chunk_tp): a lost or stale row shows as a split in the live requests' logits right after the
arrival. Rows: pcc.jsonl.

ARGS: layers ([0..4]), lens ([150, 700, 22]: A, B, C), steps ([4, 4, 8]: decode steps after each arrival), max_seq
(2560), ref, the prefill knobs (default the service knobs), tag, record (True)."""
import gc
import os
import sys
import time

import torch

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
LENS = [int(v) for v in args.get("lens", [150, 700, 22])]
STEPS = [int(v) for v in args.get("steps", [4, 4, 8])]
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
tag = args.get("tag", "p2-stagger")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
B, MAXSEQ, BLK = 4, int(args.get("max_seq", 2560)), 64
BPS = MAXSEQ // BLK
K = len(LENS)
assert K == len(STEPS) and K < B

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = "ep"
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_MOE_V2"] = "1"
os.environ["GLM53_PF_KDA_SCAN"] = "op"
os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_SPARSE_T"] = ""
os.environ["GLM53_PF_BATCH"] = "0"
os.environ["GLM53_PF_PACKED"] = ""
for k_ in ("GLM53_PF_MOE_FFN", "GLM53_PF_MOE_XFFN_XSPLIT", "GLM53_PF_MOE_XFFN_SNAKE", "GLM53_PF_MOE_XFFN_VALID"):
    os.environ.pop(k_, None)
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})

prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
OFFS = [0, 300, 1100][:K]
TOTAL = sum(STEPS)
assert all(o + n + TOTAL <= prompt.numel() for o, n in zip(OFFS, LENS))
toks = [prompt[o : o + n + TOTAL] for o, n in zip(OFFS, LENS)]

for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-chunks")]:
    old = ctx.pop(k)  # one model per process
    if old.get("pf") is not None:
        old["pf"].release()
    old["backend"].close()
gc.collect()
t0 = time.time()
backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg,
                                 weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=MAXSEQ,
                        stop_token_ids=())  # fmt: skip
kv = gen.allocate_kv_cache(max_batch=B, max_seq_len=MAXSEQ, num_blocks=B * BPS + 1)  # + vLLM's null block 0
gen.warmup(kv)
ctx[("s16-chunks", tuple(LAYERS), TPN, "stagger")] = dict(backend=backend, gen=gen, pf=backend.tpp)
pf = backend.tpp
assert pf is not None and pf.kv is kv
S.log(f"built + warmed in {time.time() - t0:.1f} s")
eager0, dec0 = pf.counters["eager"], gen.counters.get("eager_decode_steps", 0)
POOL = {s_: [BPS * s_ + 1 + (7 * j) % BPS for j in range(BPS)] for s_ in range(B)}  # scattered blocks per slot


def vrow(slot, upto):
    r = torch.full((BPS,), -1, dtype=torch.int64)
    blk = POOL[slot][: -(-upto // BLK)]
    r[: len(blk)] = torch.tensor(blk)
    return r


def prefill(r):
    n = LENS[r]
    return gen.prefill_forward(toks[r][:n].reshape(1, -1), page_table=vrow(r, n)[None], kv_cache=kv, prompt_lens=[n],
                               empty_slots=[r]).float().reshape(-1)  # fmt: skip


def decode(live, done):
    """one teacher-forced step of every live request r (slot r) at position LENS[r] + done[r] -> {r: logits}"""
    pos = torch.tensor([LENS[r] + done[r] for r in live])
    tk = torch.stack([toks[r][LENS[r] + done[r]] for r in live]).reshape(-1, 1)
    out = gen.decode_forward(tk, pos, page_table=torch.stack([vrow(r, LENS[r] + done[r] + 1) for r in live]),
                             kv_cache=kv, slots=list(live), enable_trace=True).float()  # fmt: skip
    for r in live:
        done[r] += 1
    return {r: out[i] for i, r in enumerate(live)}


# solo references: each request alone, every decode step it takes in the staggered run
solo = {}
for r in range(K):
    first = prefill(r)
    done = {r: 0}
    seq = [decode([r], done)[r] for _ in range(sum(STEPS[r:]))]
    solo[r] = dict(first=first, dec=torch.stack(seq))
# staggered: request r arrives after the steps of the earlier arrivals
stag = {r: dict(dec=[]) for r in range(K)}
done = {}
for r in range(K):
    stag[r]["first"] = prefill(r)
    done[r] = 0
    for _ in range(STEPS[r]):
        out = decode(sorted(done), done)
        for q, lg in out.items():
            stag[q]["dec"].append(lg)
rows = []
traced = pf.counters["eager"] == eager0 and gen.counters.get("eager_decode_steps", 0) == dec0
for r in range(K):
    a, b = solo[r]["dec"], torch.stack(stag[r]["dec"])
    pccs = [C.pcc(a[j], b[j]) for j in range(a.shape[0])]
    am = (a.argmax(-1) == b.argmax(-1)).float()
    first_ok = int(solo[r]["first"].argmax()) == int(stag[r]["first"].argmax())
    ok = first_ok and float(am.mean()) == 1.0 and min(pccs) >= 0.99
    row = dict(test=f"service prefill, staggered arrivals: request {r} ({LENS[r]} tokens) arriving after "
                    f"{sum(STEPS[:r])} decode steps of the earlier requests, its prefill replay with {r} request(s) live, "
                    f"vLLM page rows; per-position decode logits vs the request alone, tp{TPN} layers {len(LAYERS)}",
               milestone="P2", status="pass" if ok else "fail", traced=traced, req=r, n=LENS[r], lens=LENS,
               steps=STEPS, first_same=first_ok, decode_top1_vs_solo=round(float(am.mean()), 4),
               decode_pcc_min=round(min(pccs), 6), decode_pcc=[round(p, 6) for p in pccs],
               bitwise=bool(torch.equal(a, b)), tp=TPN, layers=len(LAYERS), mock=C.MOCK, tag=tag)  # fmt: skip
    S.log(row)
    rows.append(row)
    if rec:
        C.record("pcc", row)
RESULT = rows
