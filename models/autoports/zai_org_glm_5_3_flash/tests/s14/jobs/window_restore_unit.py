# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, the dense window under restores: doc/ttnn/14-production-tp8-megakernel/dsa/
dense_window_restore_design.md W1 / W2 gate). GLM53_DSA_SHARD + GLM53_DSA_DENSE_WINDOW on a caller-owned (serving)
cache with shared prefix pages, through the generator API the vLLM adapter uses:
- REF  slot 0: a fresh prompt prefilled [0, N) + [N, N + S) (the hit's split), its slot state exported at N;
- HIT  slot 2: a NEW request whose first N tokens are "loaded by the connector" (its row lists slot 0's first N / 64
  logical blocks: the same physical pages) -> prefill_forward(prefix_states=, start_pos=N): import_kda_state + the
  W2 hook's rebuild_window + the [N, N + S) chunk;
- CTRL slot 3: the same hit with rebuild_window disabled (the pre-fix behaviour: a stale / zero window).
Then K traced decode steps of slots 0 / 2 / 3 together (an all-dense launch: every row < DENSE_MAX reads its window).
Criterion: HIT window rows [0, N + S) bitwise == REF's on every chip and every sparse_dsa layer; prefill last-row logits
bitwise REF vs HIT; every decode step's logits bitwise REF vs HIT; the CTRL differs (the test sees a missing rebuild);
decode traced, prefill traced, 0 eager.
ARGS: layers ([0..4]), b (4), ref (a .refpt whose prompt supplies the tokens), n (1,024), s (384), k (8), max_seq_len
(4,096), env, tag, record."""

import gc
import json
import os
import time

import torch

gc.collect()
ENV = {"GLM53_STACK_DSA": "1", "GLM53_DSA_SHARD": "1", "GLM53_DSA_DENSE_WINDOW": "1", "GLM53_PREFIX_DEVICE_ENTRIES": "0",
       **{k: str(v) for k, v in (ARGS.get("env") or {}).items()}}  # noqa: F821  # fmt: skip
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

    def log(*a):
        print(f"WINDOW-RESTORE {time.strftime('%H:%M:%S')}", *a, flush=True)

    e = torch.load(args["ref"], weights_only=True)["entries"][0]
    N, S_, K = int(args.get("n", 1024)), int(args.get("s", 384)), int(args.get("k", 8))
    toks = e["prompt_tokens"][0].long()
    assert toks.numel() >= N + S_ + K and N % 128 == 0 and N + S_ + K < MK.DSA_DENSE_MAX, (toks.numel(), N, S_, K)
    cfg = hf_config()
    ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
    B, ms = int(args.get("b", 4)), int(args.get("max_seq_len", 4096))
    assert B >= 4, "slots 0 / 2 / 3"
    key = ("window_restore", B, tuple(ids), ms, tuple(sorted(ENV.items())))
    for k_ in [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "window_restore" and k_ != key]:
        be_old = ctx.pop(k_)["backend"]
        be_old.close() if hasattr(be_old, "close") else None
    gc.collect()
    if key not in ctx:
        t0 = time.time()
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=B > 4)
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=ms,
                                stop_token_ids=())  # fmt: skip
        gen.shared_pages = True  # prefix caching: live rows share a cached prefix's pages read-only
        kvc = gen.allocate_kv_cache(max_batch=B, max_seq_len=ms, null_block=0)
        t1 = time.time()
        gen.warmup(kvc)
        ctx[key] = dict(backend=backend, gen=gen, kvc=kvc)
        log(f"backend built {t1 - t0:.1f}s warmup {time.time() - t1:.1f}s {backend.trace_report()}")
    backend, gen, kvc = ctx[key]["backend"], ctx[key]["gen"], ctx[key]["kvc"]
    kv = kvc  # the caller-owned StackCache (its extra holds the window / staging)
    assert kv.extra.get("win_stage") is not None, "GLM53_DSA_DENSE_WINDOW=1 with the window staging (W1)"
    tpp = getattr(backend, "tpp", None)
    pf0 = dict(tpp.counters) if tpp is not None else {}
    c0 = dict(backend.counters)
    bs, bps = backend.block_size, G.blocks_per_seq(ms, backend.block_size)
    nbt, nbp = -(-(N + S_ + K) // bs), N // bs

    def row(blocks):
        r = torch.full((bps,), -1, dtype=torch.int64)
        r[: len(blocks)] = torch.tensor(blocks, dtype=torch.int64)
        return r

    rows = {0: row(list(range(1, 1 + nbt))),  # REF: its own blocks
            2: row(list(range(1, 1 + nbp)) + list(range(100, 100 + nbt - nbp))),  # HIT: the REF's prefix pages
            3: row(list(range(1, 1 + nbp)) + list(range(200, 200 + nbt - nbp)))}  # CTRL: the same, no rebuild
    pf = lambda slot, lo, hi, **kw: gen.prefill_forward(  # noqa: E731
        toks[lo:hi].reshape(1, -1), page_table=rows[slot].reshape(1, -1), kv_cache=kvc, prompt_lens=[hi - lo],
        empty_slots=[slot], start_pos=[lo], **kw)  # fmt: skip
    pf(0, 0, N)
    st = gen.export_slot_state(kvc, 0, n_tokens=N)
    ref_l = torch.as_tensor(pf(0, N, N + S_)).float()
    rebuild_ms = []
    real = backend.rebuild_window

    def timed(*a, **kw):
        t_ = time.perf_counter()
        n_ = real(*a, **kw)
        rebuild_ms.append(round((time.perf_counter() - t_) * 1e3, 2))
        return n_

    backend.rebuild_window = timed
    hit_l = torch.as_tensor(pf(2, N, N + S_, prefix_states={0: st})).float()
    backend.rebuild_window = lambda *a, **kw: 0  # CTRL: the pre-fix behaviour
    ctl_l = torch.as_tensor(pf(3, N, N + S_, prefix_states={0: st})).float()
    backend.rebuild_window = real
    # the window rows [0, N + S) of every sparse_dsa layer, every chip: HIT / CTRL vs REF
    nwb = -(-(N + S_) // 64)
    dev = lambda t: ttnn.get_device_tensors(t) if int(mesh.get_num_devices()) > 1 else [t]  # noqa: E731
    win = dict(layers=0, hit_bitwise=True, ctrl_bitwise=True, hit_maxabs=0.0)
    for li, st_ in sorted(kv.extra.get("dsa", {}).items()):
        w = st_.get("window")
        if w is None:
            continue
        for d in dev(w):
            wt = ttnn.to_torch(d).float()
            blk = lambda s_: wt[s_ * MK.DSA_WIN_BLOCKS : s_ * MK.DSA_WIN_BLOCKS + nwb].reshape(-1, 512)[: N + S_]  # noqa: E731
            win["hit_bitwise"] &= bool(torch.equal(blk(0), blk(2)))
            win["ctrl_bitwise"] &= bool(torch.equal(blk(0), blk(3)))
            win["hit_maxabs"] = max(win["hit_maxabs"], float((blk(0) - blk(2)).abs().max()))
        win["layers"] += 1
    # K traced decode steps of REF / HIT / CTRL together (an all-dense launch), forced inputs from the prompt
    dec = dict(steps=0, hit_bitwise=True, ctrl_differs=False, hit_maxabs=0.0, ctrl_maxabs=0.0)
    pt3 = torch.stack([rows[0], rows[2], rows[3]])
    for k in range(K):
        p = N + S_ + k
        t = int(toks[p])
        lg = torch.as_tensor(gen.decode_forward(torch.tensor([[t], [t], [t]]), torch.tensor([p, p, p]), page_table=pt3,
                                                kv_cache=kvc, slots=[0, 2, 3])).float().reshape(3, -1)  # fmt: skip
        dec["hit_bitwise"] &= bool(torch.equal(lg[0], lg[1]))
        dec["ctrl_differs"] |= not bool(torch.equal(lg[0], lg[2]))
        dec["hit_maxabs"] = max(dec["hit_maxabs"], float((lg[0] - lg[1]).abs().max()))
        dec["ctrl_maxabs"] = max(dec["ctrl_maxabs"], float((lg[0] - lg[2]).abs().max()))
        dec["steps"] += 1
    pfc = {k: int(tpp.counters.get(k, 0)) - int(pf0.get(k, 0)) for k in ("replays", "eager", "captures")} if tpp else {}
    cd = {k: int(backend.counters.get(k, 0)) - int(c0.get(k, 0)) for k in ("window_rebuilds", "window_rebuild_blocks",
                                                                           "slot_ops", "decode_trace_replays",
                                                                           "eager_decode_steps")}  # fmt: skip
    logits_bitwise = bool(torch.equal(ref_l, hit_l))
    ok = (win["layers"] > 0 and win["hit_bitwise"] and logits_bitwise and dec["hit_bitwise"] and dec["ctrl_differs"]
          and cd["window_rebuilds"] == 1 and pfc.get("eager", 0) == 0 and cd.get("eager_decode_steps", 0) == 0)  # fmt: skip
    row_ = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                test=os.environ.get("GLM53_ROW_PREFIX", "") + "window_restore_unit", status="pass" if ok else "fail",
                B=B, tp=int(mesh.get_num_devices()), layers=ids, n=N, s=S_, k=K, window=win, decode=dec,
                prefill_logits_bitwise=logits_bitwise, prefill_logits_maxabs=float((ref_l - hit_l).abs().max()),
                ctrl_prefill_logits_bitwise=bool(torch.equal(ref_l, ctl_l)), rebuild_ms=rebuild_ms, counters=cd,
                prefill_trace_counters=pfc, reference=args["ref"], rev=MK.MLA_REV, knob_env=ENV,
                tag=args.get("tag", "window-restore-unit"))  # fmt: skip
    log("ROW", json.dumps(row_, default=str))
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if args.get("record") and ev:
        with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
            f.write(json.dumps(row_, default=str) + "\n")
    RESULT = dict(ok=ok, window=win, decode=dec, logits_bitwise=logits_bitwise)
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
