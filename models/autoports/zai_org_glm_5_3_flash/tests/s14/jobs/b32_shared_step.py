# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 served long context, GLM53_STACK_DSA=1 + GLM53_STACK_PREFILL=tp + GLM53_PF_DSA=1): the B-row
traced decode step at context N on REAL context through the production fill, within the DRAM budget: ONE user's
N-token prompt runs the chunked TP prefill (backend.prefill_chunk, plan_prefill chunks = the served path; the prefill
writes the latent pages and the DSA pooled-key store / ring of slot 0), then every other row SHARES slot 0's latent
pages (page-table rows = row 0's: 32 x 32K x 11 layers of private replicated latent = 11 GiB per chip does not fit the
KV budget at TP8) and gets device copies of slot 0's KDA states (copy_slot) and pooled keys / ring. Row u steps at
position N - 1 - u (B distinct positions on the real 32K context): blocking median + back-to-back batch mean of the ONE
decode trace. A timing row (rows past 0 hold slot 0's state at a shifted position); numerics: the B=1 TF.
ARGS: key (CTX key prefix of a resident backend, e.g. "m5_serving" / "dsa_stack_fill"; else built here: layers, b
(32), max_seq_len, native (b > 4), env), prompt (.pt token ids) | ref (.refpt: prompt + generated), n (prompt tokens
to prefill, default all), shared_pages (True: a built backend's generator gets a caller-owned page pool of ONE row's
blocks, num_blocks = blocks_per_seq -> the default page-table rows wrap onto row 0's blocks: 1 / B of the latent DRAM),
tf (True with a ref: the SERVED teacher forcing first - the reference prompt through the chunked TP prefill of slot 0
(its device sample = the prediction of position P), then the forced tokens decoded on row 0 alone (other rows inactive)
on the same decode trace, each row-0 sample scored against the reference top-k (the Stage 6 TokenAccuracy rule) -> a
pcc row; one build serves the TF and the B-row step), replays (500), tag, record (-> GLM53_STAGE_EVIDENCE)."""

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
    import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PT
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821

    def log(*a):
        print(f"B32-SHARED {time.strftime('%H:%M:%S')}", *a, flush=True)

    gen_tok = topk = None
    if args.get("ref"):
        e = torch.load(args["ref"], weights_only=True)["entries"][0]
        seq = e["prompt_tokens"][0].long()
        gen_tok, topk = e["generated_tokens"][0].long(), e["topk_tokens"]
    else:
        seq = torch.load(args["prompt"], weights_only=True).reshape(-1).long()
    N = int(args.get("n", seq.numel()))
    seq = seq[:N]
    if args.get("key"):
        hits = [k for k in ctx if isinstance(k, tuple) and k and str(k[0]) == args["key"]]
        assert hits, f"no resident {args['key']} backend in this process"
        key = hits[-1]
    else:
        for m in (MK, PT, SM, G):
            importlib.reload(m)
        cfg = hf_config()
        ids = list(range(cfg.num_hidden_layers)) if args.get("layers") == "all" else [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
        B = int(args.get("b", 32))
        ms = int(args.get("max_seq_len", -(-(N + 64) // 2048) * 2048))
        native = bool(args.get("native", B > 4))
        key = ("b32_shared", B, tuple(ids), ms, native, tuple(sorted(ENV.items())))
        for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "b32_shared" and k != key]:
            be_old = ctx.pop(k)["backend"]
            be_old.close() if hasattr(be_old, "close") else None
        gc.collect()
        if key not in ctx:
            t0 = time.time()
            build = lambda: SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=native)  # noqa: E731
            if os.environ.get("GLM53_WEIGHT_SOURCE") == "image":  # Stage 12 fast start, as m5common.serving (with
                # GLM53_STACK_DSA_DEFER=1 the DSA halves attach after the restore, at the generator's KV allocation)
                from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI

                backend, info = TI.build_or_restore(build, mesh, signature=dict(B=B, layers=list(ids), native=native),
                                                    teardown=lambda b_: b_.close(), timeline={})  # fmt: skip
                log(f"fast start: weights from {info.get('source')} {info.get('image') or ''} restore "
                    f"{info.get('restore_s')} tried {info.get('tried')}")  # fmt: skip
            else:
                backend = build()
            gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                                    max_seq_len=ms, stop_token_ids=())  # fmt: skip
            t1 = time.time()
            kvc = None
            if args.get("shared_pages", True):  # one row's blocks: rows u > 0 wrap onto row 0's pages
                kvc = gen.allocate_kv_cache(max_batch=B, max_seq_len=ms, num_blocks=G.blocks_per_seq(ms, backend.block_size))
                gen.warmup(kvc)
            else:
                gen.warmup()
            ctx[key] = dict(backend=backend, gen=gen, kv=kvc)
            log(f"backend built {t1 - t0:.1f}s warmup {time.time() - t1:.1f}s {backend.trace_report()}")
    backend = ctx[key]["backend"]
    assert backend.dsa_decode, "GLM53_STACK_DSA=1 backend"
    assert getattr(backend, "tpp", None) is not None, "GLM53_STACK_PREFILL=tp backend (the served chunked TP prefill)"
    kv = ctx[key].get("kv") or backend._kv
    backend._kv = kv
    io, trace = kv.extra["io"], kv.extra["decode_trace"]
    B = int(backend.max_batch)
    r0 = dict(backend.trace_report())
    # 1. the production fill of slot 0: the chunked TP prefill (the served chunk plan), first token sampled on device
    pf0 = dict(backend.tpp.counters)
    t0 = time.perf_counter()
    plans = G.plan_prefill(N, 0, backend.prefill_chunk_tokens)
    first = None
    for p in plans:
        last = p is plans[-1]
        out = backend.prefill_chunk(seq[p.offset : p.offset + p.n_valid], kv_cache=kv, slot=0, chunk_start=p.start,
                                    n_valid=p.n_valid, logits_rows="last", io=io,
                                    sample=SamplingParams() if last else None)  # fmt: skip
        if last:
            first = int(out)
    ttnn.synchronize_device(mesh)
    prefill_s = time.perf_counter() - t0
    pf_delta = {k: int(backend.tpp.counters.get(k, 0)) - int(pf0.get(k, 0)) for k in ("replays", "eager", "captures")}
    log(f"prefill {N} tokens in {len(plans)} chunks: {prefill_s:.2f}s, first token {first}, prefill traces {pf_delta}")
    tf_row = None
    if args.get("tf") and topk is not None:  # the served TF: row 0 decodes the forced tokens, other rows inactive
        NG = int(gen_tok.numel())
        preds = [first]
        t1 = time.perf_counter()
        for i in range(1, NG):
            tok = torch.zeros(B, dtype=torch.int64)
            pos = torch.full((B,), -1, dtype=torch.int64)
            tok[0], pos[0] = int(gen_tok[i - 1]), N + i - 1
            backend.write(io.tokens, tok)
            backend.write(io.cur_pos, pos)
            backend.replay(trace)
            preds.append(int(backend.read(io.sampled)[0]))
        ttnn.synchronize_device(mesh)
        m = [sum(int(p == int(topk[i, 0])) for i, p in enumerate(preds)),
             sum(int(p in topk[i, :5].tolist()) for i, p in enumerate(preds)),
             sum(int(p in topk[i].tolist()) for i, p in enumerate(preds)), NG]  # fmt: skip
        traced = backend.trace_report().get("decode_eager", 0) == r0.get("decode_eager", 0)
        ok = traced and pf_delta["eager"] == 0 and m[1] / NG >= 0.98 and m[2] == NG
        tf_row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      test=os.environ.get("GLM53_ROW_PREFIX", "") + "served_tf", status="pass" if ok else "fail",
                      B_backend=B, tp=int(mesh.get_num_devices()), layers=len(backend.seq), prompt_len=N, generated=NG,
                      reference=args["ref"], top1=round(m[0] / NG, 4), top5=round(m[1] / NG, 4),
                      top100=round(m[2] / NG, 4), matches=m, decode_traced_no_eager=traced,
                      fill="chunked TP prefill (served plan) + row-0 decode on the B-row trace",
                      prefill_chunks=len(plans), prefill_trace_counters=pf_delta, ttft_s=round(prefill_s, 2),
                      decode_us_per_token=round((time.perf_counter() - t1) * 1e6 / max(NG - 1, 1), 1), rev=MK.MLA_REV,
                      knob_env=ENV, tag=args.get("tag", "served-tf") + "-tf")  # fmt: skip
        log("TF ROW", json.dumps(tf_row, default=str))
    # 2. rows 1 .. B-1 share slot 0's latent pages and get copies of its KDA states / pooled keys / ring
    t1 = time.perf_counter()
    bps = int(kv.blocks_per_seq)  # every row -> row 0's blocks 0 .. bps-1 (the generator's identity / wrapped row 0)
    backend.set_page_table_host(kv, torch.arange(bps, dtype=torch.int32).repeat(B, 1))
    for u in range(1, B):
        backend.copy_slot(kv, 0, u)
    repl = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None

    def tile_slot0(t, slots: int):
        d0 = ttnn.get_device_tensors(t)[0] if int(mesh.get_num_devices()) > 1 else t
        h = ttnn.to_torch(ttnn.from_device(d0)).to(torch.bfloat16)
        per = h.shape[0] // slots
        full = h[:per].repeat(slots, *([1] * (h.dim() - 1)))
        tmp = ttnn.from_torch(full, dtype=t.dtype, layout=t.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              mesh_mapper=repl)  # fmt: skip
        ttnn.copy(tmp, t)
        ttnn.deallocate(tmp)

    for i, st in kv.extra["dsa"].items():
        tile_slot0(st["pools"], int(st["slots"]))
        tile_slot0(st["ring"], int(st["slots"]))
    ttnn.synchronize_device(mesh)
    share_s = time.perf_counter() - t1
    # 3. the traced step: row u at position N - 1 - u (its own position only is rewritten by each replay)
    pos = [N - 1 - u for u in range(B)]
    backend.write(io.tokens, torch.tensor([int(seq[q]) for q in pos], dtype=torch.int64))
    backend.write(io.cur_pos, torch.tensor(pos, dtype=torch.int64))
    backend.replay(trace)
    ttnn.synchronize_device(mesh)
    n_rep = int(args.get("replays", 500))
    d = []
    for _ in range(n_rep):
        t2 = time.perf_counter()
        backend.replay(trace)
        ttnn.synchronize_device(mesh)
        d.append((time.perf_counter() - t2) * 1e6)
    t2 = time.perf_counter()
    for _ in range(n_rep):
        backend.replay(trace)
    ttnn.synchronize_device(mesh)
    batch = round((time.perf_counter() - t2) * 1e6 / n_rep, 1)
    med = round(sorted(d)[len(d) // 2], 1)
    r1 = backend.trace_report()
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             test=os.environ.get("GLM53_ROW_PREFIX", "") + "b32_shared_step", status="info", execution="traced", B=B,
             tp=int(mesh.get_num_devices()), layers=len(backend.seq), n=N, pos_min=min(pos), pos_max=max(pos),
             fill="chunked TP prefill of slot 0 (served plan) + shared-prefix pages + slot copies",
             prefill_chunks=len(plans), prefill_trace_counters=pf_delta, prefill_s=round(prefill_s, 2), first_token=first,
             share_s=round(share_s, 2),
             replays=n_rep, timing_us={"step": med, "step_batch": batch},
             decode_traced_no_eager=r1.get("decode_eager", 0) == r0.get("decode_eager", 0),
             dram_gib_per_chip=r1.get("dram_gib_per_chip"), num_blocks=int(getattr(kv, "num_blocks", 0) or 0), rev=MK.MLA_REV, backend_key=str(key[:4]),
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), knob_env=ENV, tag=args.get("tag", f"b32-shared-{N}"))  # fmt: skip
    log("ROW", json.dumps(r, default=str))
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if args.get("record") and ev:
        with open(os.path.join(ev, "perf.jsonl"), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        if tf_row is not None:
            with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
                f.write(json.dumps(tf_row, default=str) + "\n")
    RESULT = dict(ok=tf_row is None or tf_row["status"] == "pass", step_us=med, step_batch_us=batch,
                  prefill_s=round(prefill_s, 2), tf=None if tf_row is None else tf_row["matches"])
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
