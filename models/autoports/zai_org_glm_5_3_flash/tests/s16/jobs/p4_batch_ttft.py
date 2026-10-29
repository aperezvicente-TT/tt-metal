# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P4 (throughput) -- BATCHED prefill end to end through the stack: nb fresh requests
(the reference prompt, slots 0 .. nb-1) admitted by ONE replay of the batched trace (StackBackend.prefill_batch_tp ->
TPPrefill.forward_batch: every layer's row-wise work over nb x 2,048 rows, attention per request, each request's first
token sampled on device by the head) vs nb serial single-request prefills (StackBackend.prefill_chunk, the production
P2 / P3 path). Both traced; the first tokens are scored against the reference top-5 (a Stage 6 readiness .refpt of the
same layer prefix) and against each other.

Rows: wall ms per admission of nb requests (host + device, median of reps), the steady-state replay of each trace
(s16common.replay_stats), tokens/s, first-token checks.

ARGS: layers ([0..4]), ref (refpt/prefill2048_L0-4.refpt), nb (4), slots (range(nb): the batch's decode slots, any
distinct permutation -- the head rows go through the device slot table), lens ([P] x nb: request b = the reference
prompt's first lens[b] tokens -- distinct lengths give distinct first tokens, so a slot / row mix-up shows as a batched
vs serial mismatch per request), reps (5), replays_single (1000), replays_batch
(300), the P2 / P3 knobs (moe, moe_v2, kda_scan, mhc_post, s16common.set_attn: mla_form, mm_nq, mm_kb, ...), env,
tag, record."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
NB = int(args.get("nb", 4))
REPS = int(args.get("reps", 5))
tag = args.get("tag", "p4-batch-ttft")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()

entry = load_reference(REF).entries[0]
prompt = entry.prompt_tokens.reshape(-1).long()
topk = entry.topk_tokens.long()
P = int(prompt.numel())
ref_top5 = topk[0, :5].tolist()

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_DSA"] = "0"  # fresh prompts <= 2,048: exact-dense attention
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=NB, buckets=(2048,), max_seq=2560)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "grouped")
backend.tpp = pf
s = pf.bucket_for(P)
SLOTS = [int(v) for v in args.get("slots", range(NB))]
assert sorted(SLOTS) == list(range(NB)), SLOTS
LENS = [int(v) for v in args.get("lens", [P] * NB)]
assert len(LENS) == NB and all(0 < v <= P for v in LENS), LENS
reqs = [(prompt[: LENS[b]], SLOTS[b], LENS[b]) for b in range(NB)]

# every program of every key compiles BEFORE the first capture of the process (single keys, the batched forward, and
# the request paths' lazily allocated buffers)
t0 = time.time()
pf.compile((s,))
backend.prefill_chunk(torch.zeros(s, dtype=torch.int64), kv_cache=kv, slot=0, chunk_start=0, n_valid=s,
                      logits_rows="last", io=io, sample=GREEDY)  # fmt: skip
backend.prefill_batch_tp(reqs, kv_cache=kv, io=io, sample=GREEDY)  # eager: the batched programs compile here
ttnn.synchronize_device(mesh)
pf._warmed = True
pf.capture((s,))
pf.capture_batch(s, NB)
t_prep = time.time() - t0
S.log(f"P {P}, bucket {s}, nb {NB}: compiled + captured {sorted(map(str, pf.traces))} in {t_prep:.1f} s")
eager0 = pf.counters["eager"]

serial_ms, batch_ms, serial_first, batch_first = [], [], [], []
for rep in range(REPS):
    t1 = time.perf_counter()
    fs = [int(backend.prefill_chunk(prompt[: LENS[b]], kv_cache=kv, slot=b, chunk_start=0, n_valid=LENS[b],
                                    logits_rows="last", io=io, sample=GREEDY)) for b in range(NB)]  # fmt: skip
    serial_ms.append((time.perf_counter() - t1) * 1e3)
    t1 = time.perf_counter()
    fb = backend.prefill_batch_tp(reqs, kv_cache=kv, io=io, sample=GREEDY)
    batch_ms.append((time.perf_counter() - t1) * 1e3)
    serial_first.append(fs)
    batch_first.append(fb)
traced = pf.counters["eager"] == eager0
med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
# the reference top-5 covers position P only: score the requests of full length; the others vs their serial result
all_first = [fs[b] for fs in serial_first + batch_first for b in range(NB) if LENS[b] == P]
in5 = sum(int(t in ref_top5) for t in all_first)
row = dict(test=f"p4 batched prefill admission: {NB} requests x {P} tokens (L{LAYERS[0]}-L{LAYERS[-1]} prefix) in ONE "
                f"traced replay vs {NB} serial single-request prefills, first token on device, tp{TPN}",
           milestone="P4", layers=LAYERS, nb=NB, prompt_len=P, bucket=s, tp=TPN, traced=traced,
           serial_ms=round(med(serial_ms), 1), batch_ms=round(med(batch_ms), 1),
           batch_vs_serial=round(med(serial_ms) / med(batch_ms), 3),
           serial_tok_s=round(NB * P / (med(serial_ms) * 1e-3), 1), batch_tok_s=round(NB * P / (med(batch_ms) * 1e-3), 1),
           slots=SLOTS, lens=LENS, first_serial=serial_first[-1], first_batch=batch_first[-1], ref_top5=ref_top5,
           first_in_top5=f"{in5}/{len(all_first)}", same_first=serial_first == batch_first, reps=REPS,
           counters=dict(pf.counters), tag=tag)  # fmt: skip
row["status"] = "pass" if traced and in5 == len(all_first) and row["same_first"] else "fail"
if traced:  # the steady-state device time of each trace (timing only: the KDA states / pages it writes repeat)
    key1 = pf.plan(0, P)
    ts = S.replay_stats(mesh, pf.traces[key1], replays=int(args.get("replays_single", 1000)), label="single")
    tb = S.replay_stats(mesh, pf.traces[("batch", s, NB)], replays=int(args.get("replays_batch", 300)), label="batch")
    # (timing only: these replays reuse the last request's inputs; replays 0 = that soak skipped -> {})
    if ts:
        row.update(single_key=str(key1), single_us=ts["median_us"], single_replays=ts["replays"])
    if tb:
        row.update(batch_us=tb["median_us"], batch_replays=tb["replays"], batch_soak_s=tb.get("soak_s"),
                   device_batch_tok_s=round(NB * P / (tb["median_us"] * 1e-6), 1))  # fmt: skip
    if ts and tb:
        row["device_batch_vs_serial"] = round(NB * ts["median_us"] / tb["median_us"], 3)
S.log(row)
if rec:
    C.record("perf", row)
RESULT = [row]
