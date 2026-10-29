# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (program 12:50Z: the P-snapshot proof the prefix-cache serving cell relies on; tree = Stage 14's int25
+ DSA tree with Stage 9's srv/prefixcache d1c991b332 merged): the served chunked TP prefill of N + S tokens UNCACHED vs
the prefill of N tokens with a P snapshot (state_snap = entry e, the TP prefill's STATE_OUT also writes the snapshot
pool's virtual slot) + an ATTACH of the S-token suffix (state_src = e: the chunk's initial KDA state is read from the
entry inside the prefill trace; the suffix row's page-table row shares the prefix's latent pages, as vLLM's page hit
does). The REFERENCE is the uncached prefill as the serving scheduler runs it: split at the prompt's last 128 boundary
(prefixcache's _align_prefill_chunk_end: [0, N) + [N, N + S) for N = floor128(N + S - 1)), the SAME chunks and traces as
the snapshot + attach path, so the only difference is where the suffix's initial state comes from. Criterion: BITWISE
logits of the last position AND bitwise KDA state S / conv carry C of every KDA layer on every chip (reference vs
attached), prefill traced (0 eager). Info: the same prompt prefilled in ONE chunk (a different chunking: PCC-equal,
not bitwise; the device run of 13:27Z compared against it: logits max |diff| 0.094, same argmax).
Slots: ref (split) = 0, the snapshot donor = 1, the attached row = 2, the one-chunk prefill = 3 (B rows, others idle).
Sparse regime (N + S > 2,051, GLM53_STACK_DSA): also the attached slot's DSA pooled keys of every closed pool and its
ring vs the uncached slot's, bitwise (the snapshot / attach move the entry's store pages, 2026-10-22).
ARGS: layers ([0..4]), b (32), ref (a .refpt whose prompt supplies the tokens), n (1,024), s (128), entry (0),
max_seq_len (4,096), dsa_copy (True; False = the pre-fix control), env, tag, record."""

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
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    for m in (MK, PT, SM, G):
        importlib.reload(m)
    mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821

    def log(*a):
        print(f"PREFIX-ATTACH {time.strftime('%H:%M:%S')}", *a, flush=True)

    e = torch.load(args["ref"], weights_only=True)["entries"][0]
    N, S_ = int(args.get("n", 1024)), int(args.get("s", 128))
    ENTRY = int(args.get("entry", 0))
    toks = e["prompt_tokens"][0].long()[: N + S_]
    assert toks.numel() == N + S_ and N % 128 == 0, (toks.numel(), N, S_)
    cfg = hf_config()
    ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
    B = int(args.get("b", 32))
    ms = int(args.get("max_seq_len", 4096))
    key = ("prefix_attach", B, tuple(ids), ms, tuple(sorted(ENV.items())))
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "prefix_attach" and k != key]:
        be_old = ctx.pop(k)["backend"]
        be_old.close() if hasattr(be_old, "close") else None
    gc.collect()
    if key not in ctx:
        t0 = time.time()
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=B > 4)
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=ms,
                                stop_token_ids=())  # fmt: skip
        t1 = time.time()
        gen.warmup()
        ctx[key] = dict(backend=backend, gen=gen)
        log(f"backend built {t1 - t0:.1f}s warmup {time.time() - t1:.1f}s {backend.trace_report()}")
    backend = ctx[key]["backend"]
    tpp = getattr(backend, "tpp", None)
    assert tpp is not None and tpp.sio is not None, "GLM53_STACK_PREFILL=tp with the TP prefill's StateIO"
    kv = backend._kv
    io = kv.extra["io"]
    assert int(kv.extra.get("snapshot_chunks", 0)) > 0 and ENTRY < backend.snapshot_entries, (
        "the snapshot pool (GLM53_PREFIX_DEVICE_ENTRIES)", kv.extra.get("snapshot_chunks"), backend.snapshot_entries)
    tpp.prefix_dsa_copy = bool(args.get("dsa_copy", True))  # False: the pre-fix control (no DSA state moves)
    bps = G.blocks_per_seq(ms, backend.block_size)
    pf0 = dict(tpp.counters)
    # the slots' page-table rows: the identity pool; the attached row 2 shares slot 1's prefix pages [0, N)
    pt = (torch.arange(B).reshape(-1, 1) * bps + torch.arange(bps).reshape(1, -1)).to(torch.int32)
    nblk = N // backend.block_size
    pt[2, :nblk] = pt[1, :nblk]
    backend.set_page_table_host(kv, pt)

    def run_plan(slot, lo, hi, *, src=None, snap=None):
        plans = G.plan_prefill(hi - lo, lo, backend.prefill_chunk_tokens)
        out = None  # p.start = the absolute position: toks is indexed by position
        for i, p in enumerate(plans):
            first, last = i == 0, i == len(plans) - 1
            out = backend.prefill_chunk(toks[p.start : p.start + p.n_valid],
                                        kv_cache=kv, slot=slot, chunk_start=p.start, n_valid=p.n_valid, logits_rows="last",
                                        io=io, state_src=src if first else None, state_snap=snap if last else None)  # fmt: skip
        return out, len(plans)

    t0 = time.perf_counter()
    run_plan(0, 0, N)
    ref_logits, n_ref = run_plan(0, N, N + S_)  # the scheduler's split: [0, N) + [N, N + S)
    ttnn.synchronize_device(mesh)
    t_ref = time.perf_counter() - t0
    one_logits, _ = run_plan(3, 0, N + S_)  # info: one chunk
    t0 = time.perf_counter()
    _, n_pre = run_plan(1, 0, N, snap=ENTRY)
    ttnn.synchronize_device(mesh)
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter()
    att_logits, n_att = run_plan(2, N, N + S_, src=ENTRY)
    ttnn.synchronize_device(mesh)
    t_att = time.perf_counter() - t0
    pf = {k: int(tpp.counters.get(k, 0)) - int(pf0.get(k, 0)) for k in ("replays", "eager", "captures")}
    ref_l, att_l = torch.as_tensor(ref_logits).float(), torch.as_tensor(att_logits).float()
    one_l = torch.as_tensor(one_logits).float()
    logits_bitwise = bool(torch.equal(ref_l, att_l))
    # KDA state S / conv carry C of every KDA layer on every chip: slot 0 (chunk 0 row 0) vs slot 2 (chunk 0 row 2)
    ch = backend.chunk
    cr, rr = divmod(0, ch)
    ca, ra = divmod(2, ch)
    state = dict(layers=0, S_bitwise=True, C_bitwise=True, S_maxabs=0.0, C_maxabs=0.0)
    for li in range(len(backend.seq)):
        if backend.seq[li]["sparse"] == "dsa":
            continue
        Sr, Cr = kv.states[cr][li]
        Sa, Ca = kv.states[ca][li]
        dev = lambda t: ttnn.get_device_tensors(t) if int(mesh.get_num_devices()) > 1 else [t]  # noqa: E731
        for d_sr, d_cr, d_sa, d_ca in zip(dev(Sr), dev(Cr), dev(Sa), dev(Ca)):
            s_r, s_a = ttnn.to_torch(d_sr).float()[rr], ttnn.to_torch(d_sa).float()[ra]
            c_r, c_a = ttnn.to_torch(d_cr).float()[:, :, rr], ttnn.to_torch(d_ca).float()[:, :, ra]
            state["S_bitwise"] &= bool(torch.equal(s_r, s_a))
            state["C_bitwise"] &= bool(torch.equal(c_r, c_a))
            state["S_maxabs"] = max(state["S_maxabs"], float((s_r - s_a).abs().max()))
            state["C_maxabs"] = max(state["C_maxabs"], float((c_r - c_a).abs().max()))
        state["layers"] += 1
    # the DSA indexer state (sparse regime, 2026-10-22): the attached slot's pooled keys of every closed pool and its
    # ring vs the uncached slot's, every sparse_dsa layer, every chip (the attach moves the entry's pages into the slot)
    dsa = dict(layers=0, pools=(N + S_) // 4, store_bitwise=True, ring_bitwise=True, store_maxabs=0.0)
    for li, st_ in kv.extra.get("dsa", {}).items():
        dev = lambda t: ttnn.get_device_tensors(t) if int(mesh.get_num_devices()) > 1 else [t]  # noqa: E731
        for dp, dr in zip(dev(st_["pools"]), dev(st_["ring"])):
            pools = ttnn.to_torch(dp).float().reshape(int(st_["slots"]), int(st_["tp"]), -1)
            ring = ttnn.to_torch(dr).float()
            p_r, p_a = pools[0, : dsa["pools"]], pools[2, : dsa["pools"]]
            dsa["store_bitwise"] &= bool(torch.equal(p_r, p_a))
            dsa["ring_bitwise"] &= bool(torch.equal(ring[0], ring[2]))
            dsa["store_maxabs"] = max(dsa["store_maxabs"], float((p_r - p_a).abs().max()))
        dsa["layers"] += 1
    ok = logits_bitwise and state["S_bitwise"] and state["C_bitwise"] and pf["eager"] == 0
    ok = ok and (dsa["layers"] == 0 or (dsa["store_bitwise"] and dsa["ring_bitwise"]))
    row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               test=os.environ.get("GLM53_ROW_PREFIX", "") + "prefix_attach_unit", status="pass" if ok else "fail",
               B=B, tp=int(mesh.get_num_devices()), layers=len(backend.seq), n=N, s=S_, entry=ENTRY,
               logits_bitwise=logits_bitwise, logits_maxabs=float((ref_l - att_l).abs().max()),
               argmax_ref=int(ref_l.argmax()), argmax_attach=int(att_l.argmax()), kda_state=state, dsa_state=dsa,
               one_chunk_info=dict(logits_maxabs_vs_attach=float((one_l - att_l).abs().max()), argmax=int(one_l.argmax()),
                                   bitwise=bool(torch.equal(one_l, att_l))),
               prefill_trace_counters=pf, chunks=dict(ref=f"split {N} + {S_}", prefix=n_pre, attach=n_att),
               prefill_s=dict(uncached=round(t_ref, 3), prefix_with_snapshot=round(t_pre, 3), attach_suffix=round(t_att, 3)),
               dsa_copy=tpp.prefix_dsa_copy, reference=args["ref"], rev=MK.MLA_REV, knob_env=ENV,
               tag=args.get("tag", "prefix-attach-unit"))  # fmt: skip
    log("ROW", json.dumps(row, default=str))
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if args.get("record") and ev:
        with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    RESULT = dict(ok=ok, logits_bitwise=logits_bitwise, kda_state=state, dsa_state=dsa, prefill=pf)
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
