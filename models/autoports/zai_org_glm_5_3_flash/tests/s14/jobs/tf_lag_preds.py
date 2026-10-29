# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 long-context TF, GLM53_STACK_DSA=1): 14b's m5c_tf_lag loop (B rows lagged one step each on
the ONE decode trace; row u at position s - u in step s; the decode program writes every latent / pooled-key / ring row
itself) that SAVES each row's argmax at the continuation positions, so the device run does not wait for the CPU
reference: the Stage 6 TokenAccuracy rule needs only the predicted token per forced position (top-k = "argmax in the
reference's top-k"). host_score_preds.py scores the saved predictions once the reference exists. With `ref` the job
takes the tokens from the reference and scores inline (as m5c_tf_lag); with `prompt` (a token-id .pt) + `gen` it
forces the last `gen` tokens of the prompt (the chunked reference's split: prompt = ids[:n - gen], generated = the rest).
Prompt-only steps skip the device read (the fill runs at the replay rate).
key (reuse the CTX backend whose key starts with this prefix, e.g. "dsa_stack_tf" built warm_only: no build here);
deadline (ISO UTC, default env GLM53_DEADLINE = the mesh turn's hard stop) + reserve_s (240): after 512 steps and every
4,096 the run projects its end and ABORTS (job fails, the server goes on) if it would end later than deadline - reserve_s;
replays (0): after the TF, the traced step at the last fed positions (row u at n - 2 - u): blocking median + batch mean.
ARGS: b (1), layers ([0..4] or "all"), ref | prompt + gen (32), max_seq_len (n + b + 64), native (b > 4), env (set
before the build; GLM53_STACK_DSA defaults to 1), out (predictions .json), tag, record (-> GLM53_STAGE_EVIDENCE)."""

import gc
import hashlib
import json
import os
import time
from datetime import datetime

import torch

gc.collect()
ENV = {"GLM53_STACK_DSA": "1", **{k: str(v) for k, v in (ARGS.get("env") or {}).items()}}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    import importlib

    import ttnn

    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.mk_mla as MK
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    for m in (MK, SM, G):
        importlib.reload(m)
    mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
    cfg = hf_config()
    ids = list(range(cfg.num_hidden_layers)) if args.get("layers") == "all" else [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
    B = int(args.get("b", 1))
    topk = None
    if args.get("ref"):
        e = torch.load(args["ref"], weights_only=True)["entries"][0]
        seq = e["prompt_tokens"][0].tolist() + e["generated_tokens"][0].tolist()
        NG = int(e["generated_tokens"].shape[-1])
        topk = e["topk_tokens"]
        src = args["ref"]
    else:
        seq = torch.load(args["prompt"], weights_only=True).reshape(-1).long().tolist()
        NG = int(args.get("gen", 32))
        src = args["prompt"]
    NP = len(seq) - NG
    ms = int(args.get("max_seq_len", len(seq) + B + 64))
    native = bool(args.get("native", B > 4))
    tag = args.get("tag", f"tf-lag-preds-{len(seq)}-B{B}")

    def log(*a):
        print(f"TF-LAG-PREDS {time.strftime('%H:%M:%S')}", *a, flush=True)

    key = ("tf_lag_preds", B, tuple(ids), ms, native, tuple(sorted(ENV.items())))
    if args.get("key"):  # a resident backend of another job (its own build env); B from the backend
        hits = [k for k in ctx if isinstance(k, tuple) and k and str(k[0]) == args["key"]]
        assert hits, f"no resident {args['key']} backend in this process"
        key = hits[-1]
        B = int(ctx[key]["backend"].max_batch)
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "tf_lag_preds" and k != key]:
        be_old = ctx.pop(k)["backend"]  # one backend per process
        be_old.close() if hasattr(be_old, "close") else None
    gc.collect()
    if key not in ctx and not args.get("key"):
        t0 = time.time()
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=native)
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=ms,
                                stop_token_ids=())  # fmt: skip
        t1 = time.time()
        gen.warmup()
        ctx[key] = dict(backend=backend, gen=gen)
        log(f"backend built {t1 - t0:.1f}s warmup {time.time() - t1:.1f}s dsa_decode={backend.dsa_decode} "
            f"supported_context={backend.supported_context} {backend.trace_report()}")  # fmt: skip
    backend = ctx[key]["backend"]
    assert backend.dsa_decode and backend.supported_context >= len(seq) + B, (backend.dsa_decode, backend.supported_context)
    kv = backend._kv
    io, trace = kv.extra["io"], kv.extra["decode_trace"]
    r0 = dict(backend.trace_report())
    backend.reset_all(kv)
    ttnn.synchronize_device(mesh)
    preds = [[] for _ in range(B)]
    n_steps = NP + NG + B - 1
    dl = args.get("deadline") or os.environ.get("GLM53_DEADLINE")
    t_end = datetime.fromisoformat(dl.replace("Z", "+00:00")).timestamp() - float(args.get("reserve_s", 240)) if dl else None
    t0 = time.perf_counter()
    for s in range(n_steps):
        tok = torch.zeros(B, dtype=torch.int64)
        pos = torch.full((B,), -1, dtype=torch.int64)
        scored = False
        for u in range(B):
            q = s - u  # row u's position this step: feeds seq[q], predicts seq[q + 1]
            if 0 <= q < NP + NG - 1:
                tok[u], pos[u] = seq[q], q
                scored |= q >= NP - 1
        backend.write(io.tokens, tok)
        backend.write(io.cur_pos, pos)
        backend.replay(trace)
        if scored:
            out = backend.read(io.sampled)
            for u in range(B):
                if NP - 1 <= s - u < NP + NG - 1:  # the prediction of seq[s - u + 1] = G[s - u - NP + 1]
                    preds[u].append(int(out[u]))
        if s % 8192 == 0 or (t_end is not None and (s == 512 or s % 4096 == 0)):
            ttnn.synchronize_device(mesh)
            el = time.perf_counter() - t0
            log(f"step {s} / {n_steps} {el:.1f}s")
            if t_end is not None and s > 0 and time.time() + el / s * (n_steps - s) > t_end:
                raise RuntimeError(f"tf_lag_preds: projected end {el / s * n_steps:.0f}s of steps passes the deadline "
                                   f"{dl} - {args.get('reserve_s', 240)}s: aborted at step {s} (no overrun)")  # fmt: skip
    ttnn.synchronize_device(mesh)
    wall = time.perf_counter() - t0
    r1 = backend.trace_report()
    traced = r1.get("decode_eager", 0) == r0.get("decode_eager", 0)
    complete = all(len(p) == NG for p in preds)
    rows_agree = all(p == preds[0] for p in preds)  # same context per position -> same argmax (device vs device)
    tok_sha = hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]
    names = [f"L{e['L']}" for e in backend.seq]
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             test=os.environ.get("GLM53_ROW_PREFIX", "") + "dsa_stack_tf_lag", B=B, tp=int(mesh.get_num_devices()),
             prefix=names, source=src, n=len(seq), prompt_len=NP, generated=NG, tokens_sha256_16=tok_sha,
             steps=n_steps, wall_s=round(wall, 1), us_per_step=round(wall * 1e6 / n_steps, 1), rows_complete=complete,
             rows_agree=rows_agree, decode_traced_no_eager=traced, rev=MK.MLA_REV, dsa_decode=backend.dsa_decode,
             max_seq_len=ms, native=native, dram_gib_per_chip=r1.get("dram_gib_per_chip"),
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), knob_env=ENV, backend_key=str(key[:4]),
             supported_context=backend.supported_context, tag=tag)  # fmt: skip
    if topk is not None:
        m = [0, 0, 0]
        for p in preds:
            for i, t in enumerate(p[:NG]):
                m[0] += int(t == int(topk[i, 0]))
                m[1] += int(t in topk[i, :5].tolist())
                m[2] += int(t in topk[i, :100].tolist())
        tot = sum(min(len(p), NG) for p in preds)
        r.update(top1=round(m[0] / tot, 4), top5=round(m[1] / tot, 4), top100=round(m[2] / tot, 4),
                 matches=m + [tot])  # fmt: skip
        ok = traced and complete and r["top5"] >= 0.98 and r["top100"] >= 1.0
        r["status"] = "pass" if ok else "fail"
    else:
        r["status"] = "pending-reference"  # host_score_preds.py writes the scored row
        ok = traced and complete
    n_rep = int(args.get("replays", 0))
    if n_rep:  # the traced step on the TF-filled context (each replay rewrites row u's own last position only)
        last = [max(len(seq) - 2 - u, 0) for u in range(B)]
        backend.write(io.tokens, torch.zeros(B, dtype=torch.int64))
        backend.write(io.cur_pos, torch.tensor(last, dtype=torch.int64))
        backend.replay(trace)
        ttnn.synchronize_device(mesh)
        d = []
        for _ in range(n_rep):
            t1 = time.perf_counter()
            backend.replay(trace)
            ttnn.synchronize_device(mesh)
            d.append((time.perf_counter() - t1) * 1e6)
        t1 = time.perf_counter()
        for _ in range(n_rep):
            backend.replay(trace)
        ttnn.synchronize_device(mesh)
        r["step"] = dict(pos_min=min(last), pos_max=max(last), replays=n_rep, step_us=round(sorted(d)[len(d) // 2], 1),
                         step_batch_us=round((time.perf_counter() - t1) * 1e6 / n_rep, 1))  # fmt: skip
    out_path = args.get("out")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dict(row=r, preds=preds), f)
    log("ROW", json.dumps(r, default=str))
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if args.get("record") and ev:
        for name in (["pcc.jsonl"] if topk is not None else []) + (["perf.jsonl"] if n_rep else []):
            with open(os.path.join(ev, name), "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
    RESULT = dict(ok=ok, status=r["status"], rows_agree=rows_agree, wall_s=r["wall_s"])
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
