# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the mesh): Stage 14b — where the B=32 generator nondeterminism (#67, Stage 9's page-growth
divergence) comes from, below the generator. m5c_determinism showed the bare decode trace bitwise deterministic and
invariant to INACTIVE rows (synchronize after every replay). Here, on the same backend:

  G  garbage in the DSA latent caches past each active row's run end (sync per step) == A (zeros): rows past cur_pos
     are never read (the "a reset needs nothing" assumption of reset_slot);
  C  rows NA.. ACTIVE (other tokens / positions) == A for rows 0..NA-1: batch invariance w.r.t. active neighbours;
  P1 / P2 / Ps  the generator's own backend path (prefill_chunk per user = token / position writes + replays with no
     synchronize, then back-to-back decode replays with the chip-0 sampled-token read the generator does): P1 == P2
     (run-to-run) and == Ps (the same with a synchronize after every replay: cross-chip trace-boundary drift).

ARGS: b (32), layers ("all"), native (false), active (16), steps (6), users (8), plen (16), dec (8), shared_weights,
record, rev, serving (the shared serving backend: m5common.serving)."""
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
NA = int(args.get("active", 16))
STEPS = int(args.get("steps", 6))
NU, PLEN, DEC = int(args.get("users", 8)), int(args.get("plen", 16)), int(args.get("dec", 8))
NATIVE = bool(args.get("native", False))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5C-BINV {C.now()}", *a, flush=True)


key = ("m5c_binv", B, tuple(ids), int(mesh.get_num_devices()), NATIVE, args.get("rev"))
if args.get("serving"):  # the gate batch's shared backend (m5common.serving)
    key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=args.get("rev"), log=log)
else:
    M5.prune_ctx(ctx, [key])
if key not in ctx:
    M5.keep_decoders(ctx, [])
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=NATIVE,
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
repl = ttnn.ReplicateTensorToMesh(mesh) if backend.tp > 1 else None


def fill_latent(scale: float):
    g = torch.Generator().manual_seed(1234)
    for t in kv.extra.get("latent", {}).values():
        x = torch.randn(tuple(int(v) for v in t.shape), generator=g) * scale if scale else torch.zeros(
            tuple(int(v) for v in t.shape))  # fmt: skip
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=t.dtype, layout=t.layout, mesh_mapper=repl), t)
    ttnn.synchronize_device(mesh)


def head_rows(n):
    d = ttnn.get_device_tensors(backend.head.topk_out)[0] if backend.tp > 1 else backend.head.topk_out
    return (ttnn.to_torch(d).to(torch.int64).reshape(32, 64) & 0xFFFFFFFF)[:n].clone()


def run(latent_scale: float, neighbours: bool, fill=None):
    backend.reset_all(kv)
    fill() if fill is not None else fill_latent(latent_scale)
    tok = torch.zeros((B,), dtype=torch.int64)
    tok[:NA] = torch.tensor([1000 + 13 * u for u in range(NA)])
    pos = torch.full((B,), -1, dtype=torch.int64)
    pos[:NA] = torch.tensor([7 * u for u in range(NA)])
    if neighbours:
        tok[NA:] = torch.tensor([3000 + 17 * u for u in range(B - NA)])
        pos[NA:] = torch.tensor([300 + 5 * u for u in range(B - NA)])
    backend.write(io.tokens, tok)
    backend.write(io.cur_pos, pos)
    ttnn.synchronize_device(mesh)
    rows, logits = [], []
    for _ in range(STEPS):
        backend.replay(trace)
        ttnn.synchronize_device(mesh)
        rows.append(head_rows(NA))
        logits.append(backend.read(io.logits)[:NA].clone())
    return rows, logits


def first_diff(x, y):
    for s, (a, b) in enumerate(zip(x, y)):
        if not torch.equal(a, b):
            m = (a != b).reshape(a.shape[0], -1).any(-1)
            return dict(step=s, row=int(m.nonzero()[0]), n_rows=int(m.sum()))
    return None


tk = load_tokenizer()
P = tk.encode("Find the number of ordered pairs of positive integers (a, b) with a + b = 100. Show the steps.")
prompts = [(P + [1000 + u] * PLEN)[:PLEN - 1] + [1000 + u] for u in range(NU)]
if args.get("gendet_prompts"):  # m5c_gen_det.py's prompt set (the generator-level reproduction)
    _q = tk.encode("Solve: a + b = 100.")
    prompts = [_q + [1000 + u] for u in range(NU)]


def gen_path(sync: bool):
    """the generator's backend calls (no synchronize unless `sync`): per-user prefill_chunk, then DEC decode replays
    with the chip-0 sampled-token read per step"""
    backend.reset_all(kv)
    fill_latent(0.0)
    backend.write(io.tokens, torch.zeros(B, dtype=torch.int64))
    backend.write(io.cur_pos, torch.full((B,), -1, dtype=torch.int64))
    ttnn.synchronize_device(mesh)
    orig = backend.replay
    if sync:
        def rp(t):
            orig(t)
            ttnn.synchronize_device(mesh)

        backend.replay = rp
    try:
        last = []
        for u, p in enumerate(prompts):
            lg = backend.prefill_chunk(torch.tensor(p), kv_cache=kv, slot=u, chunk_start=0, n_valid=len(p),
                                       logits_rows="last")  # fmt: skip
            last.append(lg.clone())
        first = torch.stack([lg.reshape(-1).float().argmax() for lg in last]).to(torch.int64)
        tok = torch.zeros(B, dtype=torch.int64)
        tok[:NU] = first
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[:NU] = torch.tensor([len(p) for p in prompts])
        backend.write(io.tokens, tok)
        backend.write(io.cur_pos, pos)
        sampled = []
        for _ in range(DEC):
            backend.replay(trace)
            sampled.append(backend.read(io.sampled)[:NU].clone())
        ttnn.synchronize_device(mesh)
        fin = backend.read(io.logits)[:NU].clone()
    finally:
        backend.replay = orig
    return [torch.cat(last)], sampled, [fin]




def fill_past(scale: float, where: str):
    """zeros at every slot's positions < its run's end (history + the rows the run writes); garbage (scale) past the end:
    "near" = the rest of the block holding the end position, "far" = the slot's later blocks"""
    g = torch.Generator().manual_seed(4321)
    bps = kv.blocks_per_seq
    for t in kv.extra.get("latent", {}).values():
        shp = tuple(int(v) for v in t.shape)
        x = torch.zeros(shp)
        for u in range(B):
            end = 7 * u + STEPS if u < NA else 0
            blk, r0 = divmod(end, shp[-2])
            base = (u * bps) % shp[0]
            if where == "near" and blk < bps:
                x[base + blk, :, r0:, :] = torch.randn((shp[1], shp[-2] - r0, shp[-1]), generator=g) * scale
            if where == "far" and blk + 1 < bps:
                x[base + blk + 1 : base + bps] = torch.randn((bps - blk - 1,) + shp[1:], generator=g) * scale
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=t.dtype, layout=t.layout, mesh_mapper=repl), t)
    ttnn.synchronize_device(mesh)


if args.get("stale"):
    def run_past(scale, where):
        backend.reset_all(kv)
        fill_past(scale, where)
        tok = torch.zeros((B,), dtype=torch.int64)
        tok[:NA] = torch.tensor([1000 + 13 * u for u in range(NA)])
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[:NA] = torch.tensor([7 * u for u in range(NA)])
        backend.write(io.tokens, tok)
        backend.write(io.cur_pos, pos)
        ttnn.synchronize_device(mesh)
        rows, logits = [], []
        for _ in range(STEPS):
            backend.replay(trace)
            ttnn.synchronize_device(mesh)
            rows.append(head_rows(NA))
            logits.append(backend.read(io.logits)[:NA].clone())
        return rows, logits

    a0 = run_past(0.0, "near")
    a0b = run_past(0.0, "near")
    res = dict(zeros_rerun=first_diff(a0[1], a0b[1]))
    for sc in args.get("scales", [37.0, 1.0, 0.05]):
        for wh in ("near", "far"):
            v = run_past(float(sc), wh)
            fd = first_diff(v[1], a0[1])
            mx = max(float((x.float() - y.float()).abs().max()) for x, y in zip(v[1], a0[1]))
            res[f"{wh}_x{sc}"] = dict(logits=fd, rows=first_diff(v[0], a0[0]), max_abs_logit_diff=round(mx, 4),
                                      top1_changed=sum(int((x.reshape(NA, -1)[:, 0] != y.reshape(NA, -1)[:, 0]).sum())
                                                       for x, y in zip(v[0], a0[0])))  # fmt: skip
    ok = all(v is None or (isinstance(v, dict) and v.get("logits") is None) for v in res.values())
    log("STALE", dict(B=B, native_t=NATIVE, chunk=backend.chunk, layers=len(ids), active=NA, steps=STEPS, **res))
    r = dict(test=f"m5c DSA latent rows past cur_pos never read ({len(ids)} layers, B={B}{' native-t' if NATIVE else ''},"
                  f" {NA} active rows, {STEPS} steps): garbage past each row's end (same block / later blocks) vs zeros",
             status="pass" if ok else "fail", milestone="M5c", **res)  # fmt: skip
    if args.get("record"):
        C.record("tests", r)
    RESULT = dict(ok=ok, **res)
elif args.get("only_gen"):
    p1 = gen_path(False)
    p2 = gen_path(False)
    res = dict(prefill_last_logits_P1_vs_P2=first_diff(p1[0], p2[0]), decode_tokens_P1_vs_P2=first_diff(p1[1], p2[1]),
               final_logits_P1_vs_P2=first_diff(p1[2], p2[2]))  # fmt: skip
    pl = p1[0][0].reshape(len(prompts), -1).float()
    res["prefill_rows_differing"] = int((p1[0][0].reshape(len(prompts), -1) != p2[0][0].reshape(len(prompts), -1))
                                        .any(-1).sum())  # fmt: skip
    ok = all(v is None for k, v in res.items() if k != "prefill_rows_differing")
    log("ONLYGEN", dict(B=B, users=NU, plen=len(prompts[0]), **res))
    r = dict(test=f"m5c backend generator path (prefill_chunk per user + decode, no sync) run-to-run, {NU} users x "
                  f"{len(prompts[0])}-token prompts ({len(ids)} layers, B={B})", status="pass" if ok else "record" if args.get("accepted") else "fail",
             outcome="pass" if ok else "fail", accepted=args.get("accepted"),
             milestone="M5c", **res)  # fmt: skip
    if args.get("record"):
        C.record("tests", r)
    RESULT = dict(ok=ok, **res)
else:
    a = run(0.0, False)
    # G: garbage only PAST each active row's end (a row at position p reads its history 0..p-1: garbage there is
    # read by design, so a whole-cache fill would differ on every row with p > 0)
    g = run(37.0, False, fill=lambda: fill_past(37.0, "near"))
    c = run(0.0, True)
    p1 = gen_path(False)
    p2 = gen_path(False)
    ps = gen_path(True)
    res = dict(
        rows_A_vs_garbage_latent=first_diff(a[0], g[0]), logits_A_vs_garbage_latent=first_diff(a[1], g[1]),
        rows_A_vs_active_neighbours=first_diff(a[0], c[0]), logits_A_vs_active_neighbours=first_diff(a[1], c[1]),
        prefill_last_logits_P1_vs_P2=first_diff(p1[0], p2[0]), decode_tokens_P1_vs_P2=first_diff(p1[1], p2[1]),
        final_logits_P1_vs_P2=first_diff(p1[2], p2[2]),
        prefill_last_logits_P1_vs_Psync=first_diff(p1[0], ps[0]), decode_tokens_P1_vs_Psync=first_diff(p1[1], ps[1]),
        final_logits_P1_vs_Psync=first_diff(p1[2], ps[2]),
    )  # fmt: skip
    ok = all(v is None for v in res.values())
    info = dict(B=B, native_t=NATIVE, chunk=backend.chunk, layers=len(ids), active=NA, steps=STEPS, users=NU, plen=PLEN,
                dec=DEC, P1_tokens=[[int(v) for v in s] for s in p1[1]][:3])  # fmt: skip
    log("RESULT", dict(**info, **res))
    r = dict(test=f"m5c generator-path determinism below the generator ({len(ids)} layers, B={B}"
                  f"{' native-t' if NATIVE else ''}): garbage latent / active neighbours / prefill_chunk + decode without "
                  f"sync run-to-run and vs synced", status="pass" if ok else "record" if args.get("accepted") else "fail",
             outcome="pass" if ok else "fail", accepted=args.get("accepted"), milestone="M5c", **res)  # fmt: skip
    if args.get("record"):
        C.record("tests", r)
    RESULT = dict(ok=ok, **res)
