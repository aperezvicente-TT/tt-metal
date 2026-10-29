# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; chip 3 / pair / 2x4): Stage 14b — device sampling in the decode trace (the serving contract's
samples_in_decode_trace with k / p / temperature / seeds rows; kernels/head/topk_select.cpp TOPK_SAMPLE).

  A  exact oracle: the decode trace replayed `steps` times with mixed rows (greedy, top-k, top-p, temperatures);
     per step the device rows (the drawn token swapped into word 0, the K candidates with their bf16 keys) and the
     device RNG page before / after are checked against a host re-implementation of the select core's draw
     (xoshiro128**, the same exp polynomial in float32): drawn candidate, RNG advance (exactly one draw per sampled
     row, none for greedy rows), greedy rows = the argmax (ties: the lower id);
  B  the generator path: generate_batch with per-user SamplingParams — greedy users of a mixed batch == the all-greedy
     run (row independence), seeded users reproducible across runs, two seeds differ, one execute_trace per step.

ARGS: b (8), layers ([0, 1, 2, 4]), steps (24), new (16), tag, record, rev."""
import gc
import importlib
import os
import struct
import sys

import numpy as np
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
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY, SamplingParams  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 8))
_L = args.get("layers", [0, 1, 2, 4])
ids = list(range(hf_config().num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
STEPS, NEW = int(args.get("steps", 24)), int(args.get("new", 16))
tag = args.get("tag", "m5c-sample")
TPN = int(mesh.get_num_devices())
f32 = np.float32


def log(*a):
    print(f"M5C-SAMPLE {C.now()}", *a, flush=True)


# ------------------------------------------------------------------ host re-implementation of the select core's draw
def key_value(key):
    b = ((key & 0x7FFF) if (key & 0x8000) else (~key & 0xFFFF)) << 16
    return np.frombuffer(struct.pack("<I", b), dtype=np.float32)[0]


def exp_neg(x):
    x = f32(x)
    if x < f32(-87.0):
        return f32(0.0)
    y = f32(x * f32(1.44269504088896341))
    n = int(y)  # C truncation
    if f32(n) > y:
        n -= 1
    f = f32(y - f32(n))
    q = f32(1.0) + f * (f32(0.693147180) + f * (f32(0.240226507) + f * (f32(0.0555041087) + f * (
        f32(0.00961812911) + f * (f32(0.00133335581) + f * f32(0.000154035304))))))
    b = struct.unpack("<I", struct.pack("<f", float(f32(q))))[0]
    b = (b + ((n << 23) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return np.frombuffer(struct.pack("<I", b), dtype=np.float32)[0]


def rotl(x, k):
    return ((x << k) | (x >> (32 - k))) & 0xFFFFFFFF


def xoshiro_next(s):
    s = list(s)
    result = (rotl((s[1] * 5) & 0xFFFFFFFF, 7) * 9) & 0xFFFFFFFF
    t = (s[1] << 9) & 0xFFFFFFFF
    s[2] ^= s[0]
    s[3] ^= s[1]
    s[1] ^= s[2]
    s[0] ^= s[3]
    s[2] ^= t
    s[3] = rotl(s[3], 11)
    return result, s


def host_draw(k, p, invt, keys, state):
    """-> (index, new state, near_boundary)"""
    n = len(keys)
    kk = n if (k == 0 or k > n) else k
    v0 = key_value(keys[0])
    w = [exp_neg(f32(key_value(keys[i]) - v0) * f32(invt)) for i in range(kk)]
    total = f32(0.0)
    for x in w:
        total = f32(total + x)
    m, kept = kk, total
    if p < 1.0:
        thr = f32(f32(p) * total)
        pre, m = f32(0.0), 0
        for i in range(kk):
            if i > 0 and pre >= thr:
                break
            pre = f32(pre + w[i])
            m = i + 1
        kept = pre
    r, state = xoshiro_next(state)
    target = f32(f32(f32(r >> 8) * f32(1.0 / 16777216.0)) * kept)
    acc, idx, near = f32(0.0), m - 1, False
    for i in range(m):
        acc = f32(acc + w[i])
        near |= abs(float(target) - float(acc)) <= 1e-5 * max(float(kept), 1e-30)
        if target < acc:
            idx = i
            break
    return idx, state, near


# ------------------------------------------------------------------ backend + generator
NATIVE = bool(args.get("native", False))  # one launch of B rows per kind run (the served image at B=32)
key = ("m5c_sample", B, tuple(ids), TPN, NATIVE, args.get("rev"))
M5.prune_ctx(ctx, [key])
if key not in ctx:
    M5.keep_decoders(ctx, [])
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(), topk=32, sample=True,
                                     native_t=NATIVE,
                                     weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                     else None)  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=2048,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[key] = dict(backend=backend, gen=gen)
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
assert backend.sample and backend.head.sample, "device sampling not enabled"
K = backend.head.topk

# ---- A: exact oracle on the bare decode trace
kv = backend._kv
io = kv.extra["io"]
trace = kv.extra["decode_trace"]
rows_k = [1, 0, 5, 32, 0, 3, 1, 0] * 4
rows_p = [0.0, 0.95, 1.0, 0.9, 0.5, 1.0, 0.0, 0.8] * 4
rows_t = [1.0, 1.0, 0.7, 1.3, 1.0, 0.5, 1.0, 2.0] * 4
backend.write(io.k, torch.tensor(rows_k[:B]))
backend.write(io.p, torch.tensor(rows_p[:B]))
backend.write(io.temp, torch.tensor(rows_t[:B]))
backend.write(io.seeds, torch.tensor([1000 + 17 * u for u in range(B)]))
backend.write(io.tokens, torch.tensor([1000 + 7 * u for u in range(B)]))
backend.write(io.cur_pos, torch.tensor([300 + u for u in range(B)]))
ttnn.synchronize_device(mesh)
checks = dict(draws=0, draw_match=0, near_boundary_miss=0, rng_ok=0, rng_rows=0, greedy_ok=0, greedy_rows=0,
              chips_identical=True)  # fmt: skip
for step in range(STEPS):
    rng0 = backend._read_rng()
    backend.replay(trace)
    ttnn.synchronize_device(mesh)
    dev = ttnn.get_device_tensors(backend.head.topk_out) if TPN > 1 else [backend.head.topk_out]
    rows = [ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF for d in dev]
    checks["chips_identical"] &= all(torch.equal(rows[0][:B], r[:B]) for r in rows[1:])
    rows = rows[0]
    rng1 = backend._read_rng()
    for r in range(B):
        cand = [(int(rows[r, 32 + i]), int(rows[r, i])) for i in range(K) if int(rows[r, 32 + i]) != 0]
        order = sorted(cand, key=lambda c: (-c[0], c[1]))  # the merge order: key descending, ties the lower id
        draw = rows_k[r] != 1 and rows_t[r] > 0
        if not order:  # no candidates read back (mock: kernels do not run)
            checks["greedy_rows" if not draw else "draws"] += 1
            continue
        if not draw:
            checks["greedy_rows"] += 1
            checks["greedy_ok"] += int(int(rows[r, 0]) == order[0][1] and torch.equal(rng0[r], rng1[r]))
            continue
        idx, st, near = host_draw(rows_k[r], rows_p[r], 1.0 / rows_t[r], [c[0] for c in order],
                                  [int(v) for v in rng0[r]])  # fmt: skip
        checks["draws"] += 1
        ok = int(rows[r, 0]) == order[idx][1]
        checks["draw_match"] += int(ok)
        checks["near_boundary_miss"] += int((not ok) and near)
        checks["rng_rows"] += 1
        checks["rng_ok"] += int([int(v) for v in rng1[r]] == st)
okA = (checks["greedy_ok"] == checks["greedy_rows"] and checks["rng_ok"] == checks["rng_rows"] and
       checks["draw_match"] + checks["near_boundary_miss"] == checks["draws"] and checks["chips_identical"])  # fmt: skip
log("A", checks)

# ---- B: the generator path with per-user SamplingParams (A wrote the params page behind the generator's shadow:
# back to all-greedy, the generator's initial view)
backend.write(io.k, torch.ones(32, dtype=torch.int32))
backend.write(io.p, torch.zeros(32))
backend.write(io.temp, torch.ones(32))
tok = load_tokenizer()
P = tok.encode("Find the number of ordered pairs of positive integers (a, b) with a + b = 100.")
prompts = [P + [1000 + u] for u in range(B)]
mixed = [GREEDY if u % 2 == 0 else SamplingParams(temperature=0.8, top_p=0.95, seed=4242 + u) for u in range(B)]
_reset = gen.reset
if args.get("zero_tokens"):  # diagnostic: every run starts from the same device token rows (no stale inactive rows)
    _orig_reset = gen.reset

    def _reset():
        _orig_reset()
        backend.write(io.tokens, torch.zeros(B, dtype=torch.int64))
        ttnn.synchronize_device(mesh)

    gen.reset = _reset
gen.reset()
c0 = dict(gen.counters)
greedy_all = gen.generate_batch(prompts, NEW, sampling=[GREEDY] * B)
gen.reset()
greedy_all2 = gen.generate_batch(prompts, NEW, sampling=[GREEDY] * B)
gen.reset()
run1 = gen.generate_batch(prompts, NEW, sampling=mixed)
gen.reset()
run2 = gen.generate_batch(prompts, NEW, sampling=mixed)
gen.reset()
other = gen.generate_batch(prompts, NEW, sampling=[GREEDY if u % 2 == 0 else
                                                   SamplingParams(temperature=0.8, top_p=0.95, seed=99 + u)
                                                   for u in range(B)])  # fmt: skip
c1 = dict(gen.counters)
d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
greedy_same = all(run1[u] == greedy_all[u] for u in range(0, B, 2))
log("B-detail", dict(greedy_deterministic=greedy_all == greedy_all2,
                     per_user_run1_eq_run2=[run1[u] == run2[u] for u in range(B)],
                     per_user_run1_eq_greedy=[run1[u] == greedy_all[u] for u in range(B)],
                     first_div=[next((i for i, (a, b_) in enumerate(zip(run1[u], run2[u])) if a != b_), None)
                                for u in range(B)]))  # fmt: skip
reproducible = run1 == run2
seeds_differ = any(run1[u] != other[u] for u in range(1, B, 2))
one_trace = d.get("eager_decode_steps", 0) == 0 and d.get("sampling_trace_replays", 0) == 0
okB = greedy_same and reproducible and seeds_differ and one_trace
log("B", dict(greedy_rows_equal_all_greedy=greedy_same, seeded_reproducible=reproducible, seeds_differ=seeds_differ,
              one_execute_trace_per_step=one_trace, counters=d,
              sampled_text_u1=tok.decode(run1[1][:NEW]) if B > 1 else None))  # fmt: skip
ok = okA and okB
r = dict(test=f"m5c device sampling in the decode trace (head select core: top-{K} candidates, k / p / temperature / "
              f"seeded xoshiro128**), B={B} tp{TPN}, prefix {['L%d' % L for L in ids]}", status="pass" if ok else "fail",
         milestone="M5c", B=B, tp=TPN, K=K, oracle=checks, greedy_rows_equal_all_greedy=greedy_same,
         seeded_reproducible=reproducible, seeds_differ=seeds_differ, one_execute_trace_per_step=one_trace, tag=tag)  # fmt: skip
log("ROW", r)
if args.get("record"):
    C.record("tests", r)
RESULT = dict(ok=ok, oracle=checks)
