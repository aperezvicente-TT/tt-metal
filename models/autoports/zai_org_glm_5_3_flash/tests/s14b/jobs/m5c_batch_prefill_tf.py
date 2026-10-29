# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; pair / mesh): Stage 14b -- Stage 16's BATCHED prefill handing off to the stack's decode trace
(the int23 validation). nb requests of the reference prompt P (distinct slots, any permutation) are admitted by
ONE replay of the batched prefill trace (StackBackend.prefill_batch_tp -> TPPrefill.prefill_batch), then every request
decodes the reference continuation G teacher-forced through the decode trace (ONE execute_trace per step, all nb rows
together): row b feeds G[i] at position P + i and predicts G[i + 1]. Every prediction (the batch's first token included)
is scored with the Stage 6 TokenAccuracy rule against the reference top-k of the same layer prefix. The same run with nb
SERIAL single-request prefills (StackBackend.prefill_chunk) must give bitwise the same tokens.

Built like Stage 16's p4_batch_ttft (s16common.build: StackBackend b = nb + kv cache + decode io + TPPrefill, P2 / P3
knobs); every program of every key compiles before the first capture (prefill single + batched, the decode step).
ARGS: layers ([0..4]), ref (work-s14b/refpt/aime24_prefix_L0-4.refpt), nb (4), slots (a permutation of range(nb)),
bucket (2048: the prefill bucket; at TP2 the 2,048 bucket's attention program and the b = 4 decode step's L1 do not
fit together: a prefill program's static buffers end at 1,356,928, the decode's lowest L1 buffer is 1,335,296), env ({}),
moe ("ep"), moe_v2 (true), kda_scan ("grouped"), tag, record."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
S16 = os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s16")
sys.path.insert(0, S16)
import s16common as S  # noqa: E402

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
REF = args.get("ref", "/opt/glm53/work-s14b/refpt/aime24_prefix_L0-4.refpt")
NB = int(args.get("nb", 4))
SLOTS = [int(v) for v in args.get("slots", range(NB))]
assert sorted(SLOTS) == list(range(NB)), SLOTS
BUCKET = int(args.get("bucket", 2048))
tag = args.get("tag", "m5c-batch-prefill-tf")
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()


def log(*a):
    print(f"M5C-BPTF {C.now()}", *a, flush=True)


ref = torch.load(REF, weights_only=True)["entries"][0]
prompt = ref["prompt_tokens"][0].long()
Gt = ref["generated_tokens"][0].tolist()
topk = ref["topk_tokens"]  # [len(G), 100]: row i = the reference's top-k for G[i] (after P + G[:i])
P, NG = int(prompt.numel()), len(Gt)

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_DSA"] = "0"  # fresh prompts <= 2,048: exact-dense attention
S.set_attn(args)
# a clean process state: no other backend's captured traces before this one compiles (compile-after-capture /
# post-capture L1 overlap); a Stage 16 entry of the same key built by another job is rebuilt (its traces predate ours)
M5.prune_ctx(ctx, [])
_k = S.backend_key(LAYERS, NB, TPN, False)
if _k in ctx and ctx[_k].get("bptf_captured") != BUCKET:  # another job's entry or another bucket: rebuild
    _old = ctx.pop(_k)
    _old["pf"].release()
    _old["backend"].close()
ent = S.build(mesh, ctx, LAYERS, b=NB, buckets=(BUCKET,), max_seq=max(2560, P + NG + 8))
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "grouped")
backend.tpp = pf
s = pf.bucket_for(P)
assert s == BUCKET, (s, BUCKET, P)
reqs = [(prompt, SLOTS[b], P) for b in range(NB)]

if ent.get("bptf_captured") != BUCKET:
    # every program of every key compiles BEFORE the first capture of the process: the single-request prefill, the
    # batched prefill, the decode step (a program first compiled after a capture keeps a live buffer in that trace)
    t0 = time.time()
    pf.compile((s,))
    backend.prefill_chunk(torch.zeros(s, dtype=torch.int64), kv_cache=kv, slot=0, chunk_start=0, n_valid=s,
                          logits_rows="last", io=io, sample=GREEDY)  # fmt: skip
    backend.prefill_batch_tp(reqs, kv_cache=kv, io=io, sample=GREEDY)
    backend.prepare_decode(io, kv, ())
    ttnn.synchronize_device(mesh)
    pf._warmed = True
    pf.capture((s,))
    pf.capture_batch(s, NB)
    backend.capture_decode(io, kv, NB)
    ent["bptf_captured"] = BUCKET
    log(f"P {P}, G {NG}, bucket {s}, nb {NB}: compiled + captured prefill {sorted(map(str, pf.traces))} + decode in "
        f"{time.time() - t0:.1f} s")
trace = kv.extra["decode_trace"]


def run(batched: bool):
    """admit the nb requests (batched: ONE replay; else nb serial prefills), then the teacher-forced decode"""
    backend.reset_all(kv)
    ttnn.synchronize_device(mesh)
    eager0, rep0 = pf.counters["eager"], backend.counters.get("decode_replays", 0)
    t0 = time.perf_counter()
    if batched:
        first = backend.prefill_batch_tp(reqs, kv_cache=kv, io=io, sample=GREEDY)
    else:
        first = [int(backend.prefill_chunk(tk, kv_cache=kv, slot=sl, chunk_start=0, n_valid=nv, logits_rows="last",
                                           io=io, sample=GREEDY)) for tk, sl, nv in reqs]  # fmt: skip
    ttnn.synchronize_device(mesh)
    admit_ms = (time.perf_counter() - t0) * 1e3
    preds = {sl: [int(first[b])] for b, sl in enumerate(SLOTS)}
    t0 = time.perf_counter()
    for i in range(NG - 1):  # row of slot sl feeds G[i] at P + i, predicts G[i + 1]
        tok = torch.zeros(NB, dtype=torch.int64)
        pos = torch.full((NB,), -1, dtype=torch.int64)
        for sl in SLOTS:
            tok[sl], pos[sl] = Gt[i], P + i
        backend.write(io.tokens, tok)
        backend.write(io.cur_pos, pos)
        backend.replay(trace)
        out = backend.read(io.sampled)
        for sl in SLOTS:
            preds[sl].append(int(out[sl]))
    decode_s = time.perf_counter() - t0
    traced = pf.counters["eager"] == eager0
    return dict(first=[int(v) for v in first], preds=preds, admit_ms=round(admit_ms, 1),
                decode_ms_per_step=round(decode_s * 1e3 / max(NG - 1, 1), 2), prefill_traced=traced,
                decode_replays=backend.counters.get("decode_replays", 0) - rep0)  # fmt: skip


def score(preds):
    m1 = m5 = m100 = tot = 0
    worst = 1.0
    for sl, p in preds.items():
        hits = [(t == int(topk[i, 0]), t in topk[i, :5].tolist(), t in topk[i, :100].tolist()) for i, t in enumerate(p)]
        a1, a5, a100 = (sum(h[j] for h in hits) for j in range(3))
        m1, m5, m100, tot = m1 + a1, m5 + a5, m100 + a100, tot + len(hits)
        worst = min(worst, a5 / max(len(hits), 1))
    return dict(top1=round(m1 / tot, 4), top5=round(m5 / tot, 4), top100=round(m100 / tot, 4), total=tot,
                worst_row_top5=round(worst, 4))  # fmt: skip


rb = run(True)
rs = run(False)
acc = score(rb["preds"])
rows_agree = all(rb["preds"][sl] == rb["preds"][SLOTS[0]] for sl in SLOTS)
same = rb["preds"] == rs["preds"]
ok = rb["prefill_traced"] and rs["prefill_traced"] and acc["top5"] >= 0.98 and acc["top100"] >= 1.0 and same \
    and rows_agree  # fmt: skip
r = dict(test=f"m5c batched prefill ({NB} requests x {P} tokens, ONE replay of Stage 16's batched trace, slots {SLOTS})"
              f" then teacher-forced decode of {NG} tokens through the stack's decode trace (all {NB} rows per step), "
              f"L{LAYERS[0]}-L{LAYERS[-1]} prefix vs {os.path.basename(REF)}, tp{TPN}; == {NB} serial prefills bitwise",
         status="pass" if ok else "fail", milestone="M5c", layers=LAYERS, nb=NB, slots=SLOTS, prompt_len=P, bucket=s,
         tp=TPN, reference=REF, **acc, rows_agree=rows_agree, batched_eq_serial=same, first_batched=rb["first"],
         first_serial=rs["first"], admit_ms_batched=rb["admit_ms"], admit_ms_serial=rs["admit_ms"],
         decode_ms_per_step=rb["decode_ms_per_step"], prefill_traced=rb["prefill_traced"] and rs["prefill_traced"],
         counters=dict(pf.counters), tag=tag)  # fmt: skip
log("ROW", r)
if args.get("record"):
    C.record("pcc", r)
RESULT = dict(ok=ok, **acc, batched_eq_serial=same, rows_agree=rows_agree)
