# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (program 12:47:57Z / 18:09:24Z: the prefix cache's G snapshots for int28; design
doc/ttnn/14-production-tp8-megakernel/dsa/g_snapshot_design.md): the decode trace TEES a crossing slot's post-step state
into its snapshot-pool entry (GLM53_DECODE_TEE=1: tt/stack_model.py DecodeTee, the last program of the decode trace).
1. row 0 = the served chunked TP prefill of n0 prompt tokens (slot 0), then forced decode steps (the prompt's own next
   tokens) up to the 128 boundary C = ceil128(n0 + 1); the step consuming position C - 1 runs with the tee pair
   (slot 0 -> entry, DSA pages [0, C)) and the next step with none;
2. right after the crossing step: the entry's KDA S / conv C (every KDA layer, every chip) and its DSA pooled keys of
   the C / 4 closed pools + ring vs row 0's -> BITWISE;
3. (a) row 0 continues k forced steps (row 1 idle), logits read every step; (b) restore_slot(entry -> slot 1) (row 1's
   page-table row = row 0's pages: the latent cache is shared, as vLLM's page hit), then row 1 decodes the same k forced
   tokens from position C (row 0 idle) -> logits of every step BITWISE vs (a);
4. control (ARGS tee False): the same without the tee pair -> the entry holds nothing: 2 and 3 must FAIL.
Decode traced (one replay per step, 0 eager). ARGS: layers ([0..4]), b (32), ref (.refpt whose prompt supplies the
tokens), n0 (2,900: the sparse regime past 2,051), k (64), entry (0), max_seq_len (4,096), tee (True), env, tag,
record."""

import gc
import json
import os
import time

import torch

gc.collect()
ENV = {"GLM53_STACK_DSA": "1", "GLM53_DECODE_TEE": "1",
       **{k: str(v) for k, v in (ARGS.get("env") or {}).items()}}  # noqa: F821  # fmt: skip
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
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821

    def log(*a):
        print(f"G-TEE {time.strftime('%H:%M:%S')}", *a, flush=True)

    e = torch.load(args["ref"], weights_only=True)["entries"][0]
    toks = e["prompt_tokens"][0].long()
    N0, K = int(args.get("n0", 2900)), int(args.get("k", 64))
    ENTRY = int(args.get("entry", 0))
    TEE = bool(args.get("tee", True))
    C = -(-(N0 + 1) // 128) * 128  # the first 128 boundary a decode step reaches
    assert C + K <= toks.numel(), (C, K, toks.numel())
    cfg = hf_config()
    ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
    B = int(args.get("b", 32))
    ms = int(args.get("max_seq_len", 4096))
    key = ("g_tee", B, tuple(ids), ms, tuple(sorted(ENV.items())))
    for k_ in [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "g_tee" and k_ != key]:
        be_old = ctx.pop(k_)["backend"]
        be_old.close() if hasattr(be_old, "close") else None
    gc.collect()
    if key not in ctx:  # a fresh build loads the tree's modules; a reused backend keeps the classes it was built with
        for m in (MK, PT, SM, G):  # (a reload under a live backend breaks isinstance checks of its objects)
            importlib.reload(m)
        t0 = time.time()
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=B > 4)
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=ms,
                                stop_token_ids=())  # fmt: skip
        t1 = time.time()
        gen.warmup()
        ctx[key] = dict(backend=backend, gen=gen)
        log(f"backend built {t1 - t0:.1f}s warmup {time.time() - t1:.1f}s {backend.trace_report()}")
    backend = ctx[key]["backend"]
    kv = backend._kv
    assert backend.decode_tee(kv), "GLM53_DECODE_TEE=1 with the snapshot pool (GLM53_PREFIX_DEVICE_ENTRIES)"
    tpp = backend.tpp
    io, trace = kv.extra["io"], kv.extra["decode_trace"]
    tee = kv.extra["tee"]
    bps = G.blocks_per_seq(ms, backend.block_size)
    pt = (torch.arange(B).reshape(-1, 1) * bps + torch.arange(bps).reshape(1, -1)).to(torch.int32)
    pt[1] = pt[0]  # row 1 reads / writes row 0's latent pages (the restored row continues the same request)
    backend.set_page_table_host(kv, pt)
    r0, pf0 = dict(backend.trace_report()), dict(tpp.counters)
    dev = lambda t: ttnn.get_device_tensors(t) if int(mesh.get_num_devices()) > 1 else [t]  # noqa: E731

    def step(row: int, tok: int, pos: int):
        tt = torch.zeros(B, dtype=torch.int64)
        pp = torch.full((B,), -1, dtype=torch.int64)
        tt[row], pp[row] = tok, pos
        backend.write(io.tokens, tt)
        backend.write(io.cur_pos, pp)
        backend.replay(trace)
        return backend.read(io.logits)[row : row + 1].float().clone()

    def kda_rows(slot_a: int, slot_b: int) -> dict:
        ca, ra = divmod(slot_a, backend.chunk)
        cb, rb = divmod(slot_b, backend.chunk)
        out = dict(layers=0, S_bitwise=True, C_bitwise=True, S_maxabs=0.0, C_maxabs=0.0)
        for li in range(len(backend.seq)):
            if backend.seq[li]["sparse"] == "dsa":
                continue
            Sa, Ca = kv.states[ca][li]
            Sb, Cb = kv.states[cb][li]
            for d_sa, d_ca, d_sb, d_cb in zip(dev(Sa), dev(Ca), dev(Sb), dev(Cb)):
                s_a, s_b = ttnn.to_torch(d_sa).float()[ra], ttnn.to_torch(d_sb).float()[rb]
                c_a, c_b = ttnn.to_torch(d_ca).float()[:, :, ra], ttnn.to_torch(d_cb).float()[:, :, rb]
                out["S_bitwise"] &= bool(torch.equal(s_a, s_b))
                out["C_bitwise"] &= bool(torch.equal(c_a, c_b))
                out["S_maxabs"] = max(out["S_maxabs"], float((s_a - s_b).abs().max()))
                out["C_maxabs"] = max(out["C_maxabs"], float((c_a - c_b).abs().max()))
            out["layers"] += 1
        return out

    def dsa_rows(slot_a: int, slot_b: int, pools: int) -> dict:
        out = dict(layers=0, pools=pools, store_bitwise=True, ring_bitwise=True, store_maxabs=0.0)
        for li, st_ in kv.extra.get("dsa", {}).items():
            for dp, dr in zip(dev(st_["pools"]), dev(st_["ring"])):
                pl = ttnn.to_torch(dp).float().reshape(int(st_["slots"]), int(st_["tp"]), -1)
                rg = ttnn.to_torch(dr).float()
                out["store_bitwise"] &= bool(torch.equal(pl[slot_a, :pools], pl[slot_b, :pools]))
                out["ring_bitwise"] &= bool(torch.equal(rg[slot_a], rg[slot_b]))
                out["store_maxabs"] = max(out["store_maxabs"], float((pl[slot_a, :pools] - pl[slot_b, :pools]).abs().max()))
            out["layers"] += 1
        return out

    # 1. row 0: the served prefill of n0 tokens, then forced steps through the crossing (tee on its last step)
    t0 = time.perf_counter()
    for p in G.plan_prefill(N0, 0, backend.prefill_chunk_tokens):
        backend.prefill_chunk(toks[p.start : p.start + p.n_valid], kv_cache=kv, slot=0, chunk_start=p.start,
                              n_valid=p.n_valid, logits_rows="last", io=io)  # fmt: skip
    ttnn.synchronize_device(mesh)
    prefill_s = time.perf_counter() - t0
    tee_writes0 = tee.counters["writes"]
    for pos in range(N0, C):
        if pos == C - 1 and TEE:
            backend.set_decode_tee(kv, [(0, ENTRY, 0, C)])
        step(0, int(toks[pos]), pos)
        if pos == C - 1:
            backend.set_decode_tee(kv, [])
    ttnn.synchronize_device(mesh)
    # 2. the entry vs row 0 right after the crossing step
    vslot = int(kv.max_batch) + ENTRY
    snap_kda = kda_rows(0, vslot)
    snap_dsa = dsa_rows(0, vslot, C // 4)
    # 3a. row 0 continues k forced steps; 3b. the entry restored into slot 1 decodes the same steps
    la = [step(0, int(toks[pos]), pos) for pos in range(C, C + K)]
    backend.restore_slot(kv, ENTRY, 1, upto=C)
    lb = [step(1, int(toks[pos]), pos) for pos in range(C, C + K)]
    ttnn.synchronize_device(mesh)
    steps_bitwise = [bool(torch.equal(x, y)) for x, y in zip(la, lb)]
    maxabs = [float((x - y).abs().max()) for x, y in zip(la, lb)]
    end_kda = kda_rows(0, 1)
    traced = backend.trace_report().get("decode_eager", 0) == r0.get("decode_eager", 0)
    pf = {k_: int(tpp.counters.get(k_, 0)) - int(pf0.get(k_, 0)) for k_ in ("replays", "eager", "captures")}
    ok = (snap_kda["S_bitwise"] and snap_kda["C_bitwise"] and (snap_dsa["layers"] == 0 or (snap_dsa["store_bitwise"]
          and snap_dsa["ring_bitwise"])) and all(steps_bitwise) and traced and pf["eager"] == 0)  # fmt: skip
    row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               test=os.environ.get("GLM53_ROW_PREFIX", "") + "g_tee_unit", status="pass" if ok else "fail",
               control=not TEE, B=B, tp=int(mesh.get_num_devices()), layers=len(backend.seq), n0=N0, crossing=C, k=K,
               entry=ENTRY, entry_vs_row0_after_crossing=dict(kda=snap_kda, dsa=snap_dsa),
               restored_vs_continuing=dict(steps=K, bitwise_steps=sum(steps_bitwise), maxabs=max(maxabs),
                                           first_diverging=next((i for i, b_ in enumerate(steps_bitwise) if not b_), None),
                                           argmax_equal=sum(int(x.argmax() == y.argmax()) for x, y in zip(la, lb))),
               kda_after_k=end_kda, tee_ctl_writes=tee.counters["writes"] - tee_writes0, decode_traced_no_eager=traced,
               prefill_trace_counters=pf, prefill_s=round(prefill_s, 2), reference=args["ref"], rev=MK.MLA_REV,
               knob_env=ENV, tag=args.get("tag", "g-tee-unit"))  # fmt: skip
    log("ROW", json.dumps(row, default=str))
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if args.get("record") and ev:
        with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    RESULT = dict(ok=ok if TEE else not ok, row_status=row["status"], control=not TEE,
                  snap=dict(kda=snap_kda, dsa=snap_dsa), steps_bitwise=sum(steps_bitwise))  # fmt: skip
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
