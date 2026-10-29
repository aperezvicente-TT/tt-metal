# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; pair / mesh): Stage 9 -- the GENERATOR's batched prefill admission on device. Glm53Generator.
generate_batch (the code path of a vLLM prefill step: _admit_batch -> StackBackend.prefill_batch_tp ->
TPPrefill.prefill_batch, then the decode trace) on a layer prefix, with the served decode construction (native_t,
max_batch 32: launch_resident.sh's backend kw) and the served prefill-in-service construction (GLM53_STACK_PREFILL=tp,
GLM53_PREFILL_BUCKETS, GLM53_PF_BATCH=nb), built and captured by the generator's own warmup; args.env carries the
int's served stack env (14b's env-job keys; at TP2 its pair-served env: XSL, the knob set, NKMAX), else the process's
standing env applies; at TP2 ";s:GLM53_REC_UGROUPS=2" is appended to the knob set when it has no such item.

What it adds to Stage 16's generator check (4 users, idle other rows): `groups` x nb requests, so every group after
the first is admitted while the earlier groups' rows hold the request state their own admission wrote (first tokens in
the decode token / sampled rows, sampler rows: prefill_batch_tp's restore of the other rows) -- rows prefilled earlier
in the same call, not rows mid-decode (generate_batch admits every group before its decode loop writes positions;
vLLM's admission between decode steps is covered host-side by the engine mock test's wave B); a PERMUTED slot set
(request i -> slot SLOTS[i] != i: a row / slot mix-up changes a token); distinct prompts that are not prefixes of each
other (request 0 = the reference prompt, request i > 0 = a seeded permutation of its tokens, shorter: a cross-row read
changes a token); even requests greedy, odd ones seeded top-k sampled (per-request params into the batched trace's
sampler).
Request 0 is teacher-forced with the reference continuation G and scored against the reference top-k of the same layer
prefix (Stage 6 TokenAccuracy bars: top-5 >= 0.98, top-100 = 1.0); the others run free. The same requests with the
batch key hidden (backend.prefill_batch_size -> (0, 0): each request prefilled per user) must give bitwise the same
greedy tokens; the seeded sampled rows are reported (equal or not). Traced: TPPrefill's eager counter must not move
in either run (the batched run: one replay per group).

ARGS: layers ([0..4]), ref (work-s14b/refpt/aime24_prefix_L0-4.refpt), nb (4), groups (2), slots (a derangement of
range(groups x nb), default [5, 2, 7, 0, 3, 6, 1, 4]), cut (tokens cut per request, default [0, 11, 28, 41, 64, 73, 97,
120]), max_batch (32), native (true), bucket (2048), new_tokens (24), env ({}: the served env of the int under test,
e.g. 14b's env-job keys; applied before the build, restored after the job), tag, record.
Needs a tree whose TPPrefill static L1 ends below the decode stack's lowest buffer (Stage 16's L1 cut 2a0a6b0eb8b and
its C++ rebuild). TPPrefill.compile (inside the warmup) runs every key once before the decode stack allocates; without
the cut the batched program clashes at its next run, the first admission."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
TESTS = os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests")
sys.path.insert(0, os.path.join(TESTS, "s16"))
import s16common as S  # noqa: E402

sys.path.insert(0, os.path.join(TESTS, "s14b"))
import m5common as M5  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
REF = args.get("ref", "/opt/glm53/work-s14b/refpt/aime24_prefix_L0-4.refpt")
NB = int(args.get("nb", 4))
NREQ = NB * int(args.get("groups", 2))
SLOTS = [int(v) for v in args.get("slots", [5, 2, 7, 0, 3, 6, 1, 4])]
CUT = [int(v) for v in args.get("cut", [0, 11, 28, 41, 64, 73, 97, 120])]
assert len(SLOTS) == len(CUT) == NREQ and sorted(SLOTS) == list(range(NREQ)), (SLOTS, CUT, NREQ)
assert all(s != i for i, s in enumerate(SLOTS)), f"slots {SLOTS}: not a derangement"
MAXB = int(args.get("max_batch", 32))
NATIVE = bool(args.get("native", True))
assert NREQ <= MAXB, (NREQ, MAXB)
BUCKET = int(args.get("bucket", 2048))
NEW = int(args.get("new_tokens", 24))
tag = args.get("tag", "s9-gen-batch-admit")
TPN = int(mesh.get_num_devices())


def log(*a):
    print(f"S9-GBA {C.now()}", *a, flush=True)


ref = torch.load(REF, weights_only=True)["entries"][0]
prompt = ref["prompt_tokens"][0].long().tolist()
Gt = ref["generated_tokens"][0].tolist()
topk = ref["topk_tokens"]  # [len(G), 100]: row i = the reference's top-k for G[i] (after P + G[:i])
assert NEW <= len(Gt), (NEW, len(Gt))
P = len(prompt)
prompts = [prompt]
for i in range(1, NREQ):  # request i: a seeded permutation of the reference prompt's tokens, P - CUT[i] long
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(7 + i))[: P - CUT[i]].tolist()
    prompts.append([prompt[j] for j in perm])
assert all(1 <= len(p) <= BUCKET for p in prompts), [len(p) for p in prompts]
params = [SamplingParams() if i % 2 == 0 else SamplingParams(temperature=0.8, top_k=20, seed=1000 + i)
          for i in range(NREQ)]  # fmt: skip

# The env, by precedence: args.env (the int's served env) > the process's standing env (a served env an earlier env
# job set) > the PF defaults below (Stage 14b's batched-prefill validation, m5c_batch_prefill_tf: fresh prompts
# <= 2,048 exact-dense, MoE v2, grouped KDA scan); the prefill-in-service keys are this job's.
defaults = {"GLM53_PF_MHC": "comp", "GLM53_PF_MHC_POST": "fused", "GLM53_PF_DSA": "0", "GLM53_PF_MOE_V2": "1",
            "GLM53_PF_KDA_SCAN": "grouped"}  # fmt: skip
env = {k: v for k, v in defaults.items() if k not in os.environ}
env.update({str(k): str(v) for k, v in (args.get("env") or {}).items()})
# At TP2 the native step's KDA user groups (s7a283, t = 32: 8) exceed tp8_sparse's 64 head-core owners with 32 heads
# per chip: like Stage 14b's TP2 served profile (the int's knob set + ";s:GLM53_REC_UGROUPS=2"), append that item to
# the knob set in effect when it has none
if TPN == 2 and NATIVE:
    ks = env.get("GLM53_STACK_KNOB_SET", os.environ.get("GLM53_STACK_KNOB_SET", ""))
    if "GLM53_REC_UGROUPS=" not in ks:
        env["GLM53_STACK_KNOB_SET"] = ";".join(v for v in (ks, "s:GLM53_REC_UGROUPS=2") if v)
env.update({"GLM53_STACK_PREFILL": "tp", "GLM53_PREFILL_BUCKETS": str(BUCKET), "GLM53_PF_BATCH": str(NB)})
# the served construction keys the row records as they were in effect (env + the standing env)
SERVED_KEYS = {"GLM53_STACK_KNOBS", "GLM53_STACK_KNOB_SET", "GLM53_TP_XSL", "GLM53_TP_XSL_SLICED",
               "GLM53_MLA_ATT_NKMAX", "GLM53_STACK_NATIVE_WIDTHS", "GLM53_STACK_NARROW_WIDTHS",
               "GLM53_STACK_SHARE_FABRIC"}
saved = {k: os.environ.get(k) for k in env}
backend = None
try:
    os.environ.update(env)
    # a clean process state: no other job's backend or captured traces before this one compiles (compile-after-capture
    # / post-capture L1 overlap)
    M5.prune_ctx(ctx, [])
    for k in [k for k in ctx if isinstance(k, tuple) and k and str(k[0]).startswith(("s16-", "s9-"))]:
        old = ctx.pop(k)
        try:
            old["backend"].close()
        except Exception as e:  # noqa: BLE001
            log(f"close {k}: {e!r}")
    t0 = time.time()
    backend = SM.build_model_backend(None, mesh, max_batch=MAXB, layers=LAYERS, hf_config=hf_config(),
                                     weight_cache=ctx.setdefault("weight_cache", {}), native_t=NATIVE)  # fmt: skip
    gen = Glm53Generator(backend, max_batch=MAXB, max_seq_len=max(2560, P + NEW + 8))
    gen.warmup()  # phase 1: every prefill key incl. the batched one + decode compiled; phase 2: the captures
    assert backend.prefill_batch_size() == (BUCKET, NB), backend.prefill_batch_size()
    tpp = backend.tpp
    log(f"built + warmed in {time.time() - t0:.1f} s: layers {LAYERS}, nb {NB} x {NREQ // NB} groups, "
        f"max_batch {MAXB}, native_t {NATIVE}, bucket {BUCKET}, tp{TPN}, prompts {[len(p) for p in prompts]}, "
        f"slots {SLOTS}, tpp {dict(tpp.counters)}")

    def run(batched: bool):
        if not batched:
            backend.prefill_batch_size = lambda: (0, 0)  # hide the batch key: every request per user
        try:
            gen.reset()
            b0, c0 = backend.counters.get("prefill_batches_tp", 0), dict(tpp.counters)
            t = time.perf_counter()
            out = gen.generate_batch(prompts, NEW, sampling=params, slots=SLOTS, stop_on_eos=False,
                                     next_input=lambda u, step, p: Gt[step] if u == 0 else p)  # fmt: skip
            return dict(out=out, s=round(time.perf_counter() - t, 2),
                        batches=backend.counters.get("prefill_batches_tp", 0) - b0,
                        tpp={k: v - c0.get(k, 0) for k, v in tpp.counters.items() if isinstance(v, int)})  # fmt: skip
        finally:
            backend.__dict__.pop("prefill_batch_size", None)

    rb, rs = run(True), run(False)
    hits = [(t == int(topk[i, 0]), t in topk[i, :5].tolist(), t in topk[i, :100].tolist())
            for i, t in enumerate(rb["out"][0])]  # fmt: skip
    acc = {f"top{n}": round(sum(h[j] for h in hits) / len(hits), 4) for j, n in enumerate((1, 5, 100))}
    greedy_eq = all(rb["out"][i] == rs["out"][i] for i in range(0, NREQ, 2))
    sampled_eq = all(rb["out"][i] == rs["out"][i] for i in range(1, NREQ, 2))
    traced = rb["tpp"]["eager"] == 0 and rs["tpp"]["eager"] == 0
    eager_decode = gen.counters.get("eager_decode_steps", 0)
    ok = (greedy_eq and rb["batches"] == NREQ // NB and rs["batches"] == 0 and traced and eager_decode == 0
          and acc["top5"] >= 0.98 and acc["top100"] >= 1.0)  # fmt: skip
    r = dict(test=f"s9 generator batched admission: {NREQ} requests in {NREQ // NB} groups of {NB} (prompts "
                  f"{[len(p) for p in prompts]}, slots {SLOTS}, greedy / seeded top-k alternating) through "
                  f"Glm53Generator.generate_batch vs per-user, then {NEW} decode steps; "
                  f"L{LAYERS[0]}-L{LAYERS[-1]} prefix, tp{TPN}",
             status="pass" if ok else "fail", layers=LAYERS, nb=NB, groups=NREQ // NB, slots=SLOTS, max_batch=MAXB,
             native_t=NATIVE,
             prompt_lens=[len(p) for p in prompts], bucket=BUCKET, tp=TPN, reference=REF, tf_row0=acc,
             greedy_batched_eq_serial=greedy_eq, sampled_batched_eq_serial=sampled_eq,
             batches=(rb["batches"], rs["batches"]), tpp_batched=rb["tpp"], tpp_serial=rs["tpp"], traced=traced,
             run_s=(rb["s"], rs["s"]), eager_decode_steps=eager_decode, env=env,
             effective_env={k: os.environ.get(k) for k in sorted(set(env) | SERVED_KEYS)}, tag=tag)  # fmt: skip
    log("ROW", r)
    if args.get("record"):
        C.record("tests", r)
    RESULT = dict(ok=ok, **acc, greedy_batched_eq_serial=greedy_eq, sampled_batched_eq_serial=sampled_eq,
                  traced=traced, batches=[rb["batches"], rs["batches"]])  # fmt: skip
finally:
    try:
        if backend is not None:
            ttnn.synchronize_device(mesh)
            backend.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
