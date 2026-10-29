# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, chip 3 or the TP mesh): Stage 14b M5c — tt/stack_model.py StackBackend driven by the generator of the
serving contract (tt/generator.py, Stage 9 M6: samples_in_decode_trace = one execute_trace per decode step), on a
model PREFIX (chip 3: dense L0-L2 + sparse_kda L4-L6; mesh: dense L0-L2), so the tokens are not the model's:

  1. warmup (prepare = one eager step, capture = ONE decode trace incl. sampling);
  2. greedy generate of prompt P (prefill = decode steps), traced; trace counters: decode replays == decode steps,
     fused sampling steps == replays, no eager decode, no sampler trace;
  3. stale-input / batch independence: generate_batch([P, Q]) == [generate(P) alone, generate(Q) alone] (bitwise
     token sequences: rows of one step are independent, every slot's state / position / token row is device-resident);
  4. teacher forcing: generate(P, next_input=ref) returns the model's own predictions at every forced step;
  5. inactive slots: a slot that does not decode keeps its KDA state bit-identical across steps;
  6. timing: decode replays per second.
ARGS: b (4), new (8), dense ([0, 1, 2]), sparse ([4, 5, 6]), tag, record."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for mod in (M5, MKS, TPS, MKH, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 4))
NEW = int(args.get("new", 8))
DENSE = [int(v) for v in args.get("dense", [0, 1, 2])]
SPARSE = [int(v) for v in args.get("sparse", [4, 5, 6])]
DSA = [int(v) for v in args.get("dsa", [])]  # sparse_dsa layers (Stage 7b MkMla + MkMoE; one chip)
tag = args.get("tag", "m5c-backend")
TPN = int(mesh.get_num_devices())
cfg = hf_config()


def log(*a):
    print(f"M5C {C.now()}", *a, flush=True)


key = ("m5c", bool(args.get("native", False)), B, tuple(DENSE), tuple(SPARSE), tuple(DSA), TPN, args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    t0 = time.time()
    M5.set_env({"GLM53_MK_OPT": "O3", "GLM53_MK_READERS": "3", "GLM53_MK_RING_KB_QKV": "700"})
    if TPN == 1:
        M5.keep_decoders(ctx, DENSE + SPARSE + DSA)
        # the backend's stacks run 4-row chunks: the sparse layers' T=4 expert ring (3 slots, 2 in flight)
        env_s = {**M5.SPARSE_KNOBS, "GLM53_MKX_RING": "3", "GLM53_MKX_INFLIGHT": "2"} if DSA else M5.SPARSE_KNOBS
        seq = M5.build_layers(mesh, cfg, ctx, DENSE, SPARSE, M5.DENSE_KNOBS, env_s, dsa=DSA)
        backend = SM.build_model_backend(
            None, mesh, max_batch=B, seq=seq, hf_config=cfg, native_t=bool(args.get("native", False))
        )
    else:
        M5.keep_decoders(ctx, [])
        backend = SM.build_model_backend(
            None,
            mesh,
            max_batch=B,
            layers=sorted(DENSE + SPARSE + DSA),
            hf_config=cfg,
            native_t=bool(args.get("native", False)),
        )
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
    t1 = time.time()
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
    log(f"backend {backend.trace_report()} built in {t1 - t0:.1f}s, warmup {time.time() - t1:.1f}s")
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
tok = load_tokenizer()
P = tok.encode("Find the number of ordered pairs of integers (a, b) such that a + b = 10.")[:16]
Q = tok.encode("Let x be a real number with x^2 = 2. Compute x^4 + 1.")[:12]
res = {}
c0 = dict(gen.counters)
gen.reset()
t0 = time.time()
a_alone = gen.generate(P, NEW, enable_trace=True)
t_gen = time.time() - t0
gen.reset()
b_alone = gen.generate(Q, NEW, enable_trace=True)
gen.reset()
both = gen.generate_batch([P, Q], NEW, enable_trace=True)
batch_equal = both[0] == a_alone and both[1] == b_alone
gen.reset()
ref = list(a_alone)
forced = gen.generate(P, NEW, next_input=lambda step, pred: ref[step], enable_trace=True)
tf_consistent = forced == a_alone  # forcing the model's own greedy tokens reproduces them
c1 = dict(gen.counters)
d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
one_replay_per_step = d.get("eager_decode_steps", 0) == 0 and d.get("sampling_trace_replays", 0) == 0 and \
    d.get("fused_sampling_steps", 0) == d.get("decode_trace_replays", 0) > 0  # fmt: skip
# inactive slot: slot 1 never decodes while slot 0 does -> its KDA state stays bit-identical
gen.reset()
kv = gen._own.kv_cache if gen._own is not None else None
inactive_same = None
if kv is not None:
    kda_i = [i for i in range(len(backend.seq)) if kv.states[0][i] is not None]  # sparse_dsa layers: no KDA state
    snap = [ttnn.to_torch(ttnn.get_device_tensors(kv.states[0][i][0])[0])[1].clone() for i in kda_i]
    gen.generate(P, 4, enable_trace=True)
    after = [ttnn.to_torch(ttnn.get_device_tensors(kv.states[0][i][0])[0])[1] for i in kda_i]
    inactive_same = all(torch.equal(a, b) for a, b in zip(snap, after))
# the vLLM async split: read_async (non-blocking device -> host copy + event) then finish_read == the blocking read
io_ = backend._kv.extra["io"]
pend, evs = backend.read_async(io_.sampled)
for ev in evs:
    ttnn.event_synchronize(ev)
async_read_ok = bool(torch.equal(backend.finish_read(io_.sampled, pend), backend.read(io_.sampled)))
ok = batch_equal and tf_consistent and one_replay_per_step and inactive_same is not False and async_read_ok
names = [f"L{e['L']}" for e in backend.seq]
r = dict(test=f"m5c StackBackend under the serving generator (prefix {names}, B={B}, tp{TPN})", status="pass" if ok
         else "fail", milestone="M5c", B=B, tp=TPN, prefix=names, generate_alone=[a_alone, b_alone],
         batch_equal_to_alone=batch_equal, teacher_forcing_consistent=tf_consistent,
         one_execute_trace_per_step=one_replay_per_step, async_read_ok=async_read_ok, counters_delta=d, inactive_slot_state_bitwise=inactive_same,
         trace_report=backend.trace_report(), dram=M5.dram_state(mesh), tag=tag)  # fmt: skip
prow = dict(test=f"m5c generator decode B={B} tp{TPN} prefix {len(names)} layers", status="info", milestone="M5c",
            traced=True, us=round(t_gen * 1e6 / (len(P) + NEW), 1), B=B, tp=TPN, tag=tag,
            note="wall per token of generate(P, NEW) incl. prefill-as-decode and the host reads of the generator")  # fmt: skip
log("ROW", r)
log("PERF", prow)
if args.get("record"):
    C.record("pcc", r)
    C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
RESULT = dict(ok=ok, a=a_alone, b=b_alone, batch_equal=batch_equal)
