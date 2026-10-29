# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 long-context rungs, GLM53_STACK_DSA=1): fill a StackBackend's context to N positions through
its ONE decode trace (the decode program writes every latent row, pooled key and ring row itself, as dsa_stack_tf's
"decode" prefill) and time the decode step there - the step-time rung where no TF reference exists yet (128K: Stage
16 has no 131072-token prompt / reference). Tokens: the 32,768-token golden prompt cycled; B rows lagged one step each
(m5c_tf_lag's rule), so the last step has row u at N - 1 - u (B distinct positions). Then `replays` blocking replays
(median) + the back-to-back batch mean at those positions (each replay rewrites row u's own position only).
ARGS: layers ([0..4] or "all"), b (1), n (positions per row, 131072), max_seq_len (n + b + 64), native (b > 4),
replays (500), tag, record (-> GLM53_STAGE_EVIDENCE), env (set before the build; GLM53_STACK_DSA defaults to 1)."""

import gc
import json
import os
import time

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
    N = int(args.get("n", 131072))
    ms = int(args.get("max_seq_len", N + B + 64))
    native = bool(args.get("native", B > 4))
    n_rep = int(args.get("replays", 500))
    tag = args.get("tag", f"dsa-stack-fill-{N}-B{B}")

    def log(*a):
        print(f"DSA-STACK-FILL {time.strftime('%H:%M:%S')}", *a, flush=True)

    key = ("dsa_stack_fill", B, tuple(ids), ms, native, tuple(sorted(ENV.items())))
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "dsa_stack_fill" and k != key]:
        be_old = ctx.pop(k)["backend"]
        be_old.close() if hasattr(be_old, "close") else None
    gc.collect()
    if key not in ctx:
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
    assert backend.dsa_decode and backend.supported_context >= N + B, (backend.dsa_decode, backend.supported_context)
    kv = backend._kv
    io, trace = kv.extra["io"], kv.extra["decode_trace"]
    ws = os.environ.get("GLM53_WS", "/opt/glm53/kit")
    prompt = torch.load(os.path.join(ws, "goldens", "prompts", "32768.pt")).reshape(-1).long()
    backend.reset_all(kv)
    ttnn.synchronize_device(mesh)
    n_steps = N + B - 1
    t0 = time.perf_counter()
    for s in range(n_steps):
        tok = torch.zeros(B, dtype=torch.int64)
        pos = torch.full((B,), -1, dtype=torch.int64)
        for u in range(B):
            q = s - u
            if 0 <= q < N:
                tok[u], pos[u] = prompt[q % prompt.numel()], q
        backend.write(io.tokens, tok)
        backend.write(io.cur_pos, pos)
        backend.replay(trace)
        if s % 4096 == 0:
            ttnn.synchronize_device(mesh)
            log(f"step {s} / {n_steps} {time.perf_counter() - t0:.1f}s")
    ttnn.synchronize_device(mesh)
    fill_s = time.perf_counter() - t0
    last = [N - 1 - u for u in range(B)]  # each row's last filled position: its step rewrites it
    backend.write(io.tokens, torch.zeros(B, dtype=torch.int64))
    backend.write(io.cur_pos, torch.tensor(last, dtype=torch.int64))
    d = []
    for _ in range(n_rep):
        t1 = time.perf_counter()
        backend.replay(trace)
        ttnn.synchronize_device(mesh)
        d.append((time.perf_counter() - t1) * 1e6)
    med = round(sorted(d)[len(d) // 2], 1)
    t1 = time.perf_counter()
    for _ in range(n_rep):
        backend.replay(trace)
    ttnn.synchronize_device(mesh)
    batch = round((time.perf_counter() - t1) * 1e6 / n_rep, 1)
    rep = backend.trace_report()
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             test=os.environ.get("GLM53_ROW_PREFIX", "") + "dsa_stack_fill_step", status="info", execution="traced",
             B=B, tp=int(mesh.get_num_devices()), layers=[f"L{e['L']}" for e in backend.seq], n=N, native=native,
             pos_min=min(last), pos_max=max(last), fill_steps=n_steps, fill_s=round(fill_s, 1),
             fill_us_per_step=round(fill_s * 1e6 / n_steps, 1), replays=n_rep,
             timing_us={"step": med, "step_batch": batch}, rev=MK.MLA_REV, dsa_decode=backend.dsa_decode,
             supported_context=backend.supported_context, dram_gib_per_chip=rep.get("dram_gib_per_chip"),
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), knob_env=ENV, tag=tag)  # fmt: skip
    log("ROW", json.dumps(r, default=str))
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if args.get("record") and ev:
        for name in ("pcc.jsonl", "perf.jsonl"):
            with open(os.path.join(ev, name), "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
    RESULT = dict(ok=True, step_us=med, step_batch_us=batch, fill_s=round(fill_s, 1))
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
