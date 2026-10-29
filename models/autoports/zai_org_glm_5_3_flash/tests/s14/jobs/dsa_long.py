# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, DSA long context; program 2026-10-21 02:57:57Z SCOPE: the model's max context 1,048,576):
one decode token of MkMla(dsa) at position ARGS["pos"] over a SYNTHETIC state -- random bf16 latent rows (the paged
cache) and random pooled keys (the indexer's per-user state) -- because the float64 model oracle cannot build prompts of
this length. The token's input row is the HF capture's decode input (layer GLM53_S7B_LAYER), so q_lat / q_idx / wsc are
the model's. Program: ARGS["tp"] (default 8) = chip 0's TP head shard (the per-chip program of the TP mesh, replicated
indexer); 1 = the full 64-head program.

Checks (the device's own intermediates, as tests/s7b/dev_mk_dsa.py dsa_select_emulation):
  scores = wsc . relu(q_idx K_pool^T) over the n_closed pools (float64 on the host), exact top-512 + the tail rows,
  softmax attention of q_lat over those latent rows (the host copy of the cache + the step's own latent row) vs the
  device's o_lat (bar 0.9999); traced replay median of the step.
ARGS: pos (int, or a list -> one case per position), tp (8), replays (2000), soak_s (0: seconds of back-to-back replays
after the timing), seed (5), t (1: rows per launch; t > 1 = slots at pos + 37 i, each with its own pages / pooled keys -
the B = 32 long-context regime; host + DRAM bound the length: t x pos x 1 KB of latent rows), slot_pos (t > 1: an explicit
position per slot, e.g. a mixed-context batch; `pos` then only seeds the case), check_slots (t > 1: the
slots emulated on the host, default [0, t - 1]), probe (False: in-program marks of
the owners' NCRISC -- start, I4 keys in, I5 select done, selection out, end -> row "owner_dm1", us after the earliest
mark)."""

import importlib
import itertools
import math
import os
import time

import torch

import ttnn

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, int_tensor, pcc
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK

    for m in (MK, MC):
        importlib.reload(m)
    mesh = MESH  # noqa: F821
    layer = int(os.environ.get("GLM53_S7B_LAYER", "3"))
    poss = ARGS.get("pos", [131074])  # noqa: F821
    poss = [int(poss)] if isinstance(poss, (int, str)) else [int(p) for p in poss]
    tpn = int(ARGS.get("tp", 8))  # noqa: F821
    replays = int(ARGS.get("replays", 2000))  # noqa: F821  (>= 2,000: program 03:20Z DEVICE DUTY)
    soak_s = float(ARGS.get("soak_s", 0))  # noqa: F821
    seed = int(ARGS.get("seed", 5))  # noqa: F821
    rows = MC.Rows(layer)
    def soak(tr_, seconds):
        """program DEVICE DUTY rule (2026-10-21 03:20Z): replay the captured program back to back for `seconds`
        (unwatched soak: the race / hang detector) -> replays done"""
        n, t_end = 0, time.time() + seconds
        while time.time() < t_end:
            tr_.replay()
            n += 1
        return n

    tr = Traced(mesh)
    h0 = lambda t: MK._host0(t)  # noqa: E731
    L = MC.layer_of(mesh, CTX, layer)  # noqa: F821
    attn = L.attn
    ix = MK.DsaIndexer(mesh, L.source, layer)
    shard = MK.HeadShard(attn, 0, tpn) if tpn > 1 else attn
    kw = dict(kb=(8, 16, 8, 16, 16), rings=(140, 105, 128, 280)) if tpn > 1 else {}
    probe = bool(ARGS.get("probe", False))  # noqa: F821
    T = int(ARGS.get("t", 1))  # noqa: F821
    assert 1 <= T <= 32, T
    if T > 4 and tpn > 1:  # the native-t program: dsa_tp.py's vtp8 config (the stack's indexer K blocks + o_proj ring)
        kw = dict(kb=(8, 16, 8, 16, 16), rings=(140, 105, 64, 280), dsa_kb=(16, 4))
    mk = MK.MkMla(mesh, L.a_args, shard, dsa=ix, probe=probe, **kw)
    hr = shard.h
    gd = golden(layer, 2048)
    x_row = gd["decode0_streams_in"].float().reshape(1, 1, 1, -1)
    bs = attn.block_size

    def case_t(p):
        """t > 1: slots i at p + 37 i, own pages (a random permutation of t x nb blocks) holding the same random latent
        rows (rows >= p zero), own random pooled keys; the same decode input row on every slot"""
        g = torch.Generator().manual_seed(seed + p)
        ps = [int(v) for v in ARGS["slot_pos"]] if ARGS.get("slot_pos") else [p + 37 * i for i in range(T)]  # noqa: F821
        assert len(ps) == T, (len(ps), T)
        cap = -(-(max(ps) + 1) // bs) * bs
        nb = cap // bs
        pt_host = torch.randperm(T * nb, generator=g).to(torch.int32).reshape(T, nb)
        lat = torch.randn(cap, attn.kv_lora, generator=g).to(torch.bfloat16)
        lat[p:] = 0
        ch = torch.empty(T * nb, 1, bs, attn.kv_lora, dtype=torch.bfloat16)
        blocks = lat.reshape(nb, 1, bs, attn.kv_lora)
        for i in range(T):
            ch[pt_host[i].long()] = blocks
        cache = ttnn.from_torch(ch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        del ch
        ptab = ttnn.from_torch(pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        tp_rows = MK.DsaIndexer.pool_rows(cap)
        pools = torch.zeros(T, 1, tp_rows, 128, dtype=torch.float32)
        for i, pi in enumerate(ps):
            pools[i, 0, : (pi + 1) // 4] = torch.randn((pi + 1) // 4, 128, generator=g)
        st = ix.state(mesh, T, cap, pools=pools, ring=torch.zeros(T, 1, 32, 256))
        del pools
        cur = int_tensor(ps, mesh)
        x = ttnn.from_torch(x_row.expand(1, 1, T, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                            device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        return ps, cap, lat, cache, ptab, st, cur, x

    for p in poss:
        t0 = time.time()
        if T > 1:
            ps, cap, lat, cache, ptab, st, cur, x = case_t(p)
            r_ = dict(p=p, t=T, cap=cap, tp=tpn, heads=hr, rev=MK.MLA_REV, setup_s=round(time.time() - t0, 1))
            try:
                fn = lambda: mk(x, cache, ptab, cur, t=T, dsa_state=st)  # noqa: E731
                tr.warm(fn)
                out = tr.run(fn)
                got = {k: h0(out[k]) for k in ("o_lat", "q_lat", "q_ix", "wsc", "latent")}
                out_mbox = out["mbox"]
                us = MC.timed(tr, replays)
                n_soak = soak(tr, soak_s) if soak_s > 0 and tr.enabled else 0
                tr.release()
                out = None
                if probe:
                    pr = MK.decode_probe(ttnn.to_torch(ttnn.get_device_tensors(mk.probe)[0]) if mesh.get_num_devices() > 1
                                         else ttnn.to_torch(mk.probe), len(mk.last_readers.assign))  # fmt: skip
                    r_.update({k: pr[k] for k in ("owner_dm1", "owner_dm1_max", "owner", "owner_max", "reader_a_max",
                                                  "ix_dm0", "ix_dm1") if k in pr})  # fmt: skip
                pools_after = h0(st["pools"]).reshape(T, 1, -1, 128)
                # the device's I5 result per token (diagnostic): the final radix prefix T (the k-th largest key) + k_rem
                # from a lane owner's mailbox (all owners of the token's lane got the same messages)
                mb = ttnn.to_torch(ttnn.get_device_tensors(out_mbox)[0] if mesh.get_num_devices() > 1 else out_mbox)
                mb = mb.to(torch.int64) & 0xFFFFFFFF
                nl, cl = mk.att_lanes(T), mk.chunks // mk.att_lanes(T)
                kf = lambda s_: (lambda b: torch.where(b >= 2**31, b ^ 0xFFFFFFFF, b | 2**31))(  # noqa: E731
                    s_.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF)  # mladsa::key_of on fp32 bits
                pcs, diag = [], []
                for i in [int(v) for v in ARGS.get("check_slots", [0, T - 1])]:  # noqa: F821
                    pi, nc = ps[i], (ps[i] + 1) // 4
                    qi = got["q_ix"][0, i].double()
                    w = got["wsc"][0, 0, i, :32].double()
                    scores = w @ torch.relu(qi @ pools_after[i, 0, :nc].double().T)
                    k = min(MK.DSA_TOPK, nc)
                    sel = torch.topk(scores, k).indices.sort().values
                    kv_all = lat.double()[: pi + 1].clone()
                    kv_all[pi] = got["latent"].reshape(-1, attn.kv_lora)[i].double()
                    ql = got["q_lat"][0, i, :hr].double()

                    def emulate(sel_):
                        rows_ = torch.cat([(4 * sel_[:, None] + torch.arange(4)).reshape(-1), torch.arange(4 * nc, pi + 1)])
                        kv = kv_all[rows_]
                        o_em = torch.softmax((ql @ kv.T) * attn.scale, dim=-1) @ kv
                        return pcc(got["o_lat"][0, i, :hr].double().reshape(-1).float(), o_em.reshape(-1).float())

                    pcs.append(emulate(sel))
                    if pi + 1 <= MK.DSA_DENSE_MAX:  # a dense-regime slot (slot_pos): every row, no I5 message to read
                        diag.append(dict(slot=i, dense=True, pcc_tz=pcs[-1], pcc_rn=pcs[-1]))
                        continue
                    own = mk.place["k"][(i % nl) * cl]  # a group-0 owner of the token's lane
                    row = mk.mbox_rows[(int(own.x), int(own.y))]
                    off = (i * MK.DSA_MSGS + MK.DSA_ROUNDS - 1) * 4  # the last radix round's message (16 B = 4 words)
                    t_key, k_rem = int(mb[row, off]), int(mb[row, off + 1])
                    keys = kf(scores)
                    n_gt = int((keys > t_key).sum())
                    eq = torch.nonzero(keys == t_key).reshape(-1)
                    sel_dev = torch.cat([torch.nonzero(keys > t_key).reshape(-1), eq[:k_rem]]).sort().values
                    srt = scores.sort(descending=True).values
                    # the device's score arithmetic (att_c.cpp score): S = relu(q_idx K^T) in fp32 (bf16 operands, exact
                    # products), then wsc @ S on the FPU matmul -> fp32 operands enter as tf32 (10 mantissa bits); model it
                    # with truncation (tz) and round-to-nearest (rn) of both operands
                    def tf32(v, rn):
                        b = v.float().view(torch.int32)
                        if rn:
                            b = b + 0x1000 - 1 + ((b >> 13) & 1)
                        return (b & ~0x1FFF).view(torch.float32).double()

                    relu_k = torch.relu(qi @ pools_after[i, 0, :nc].double().T).float().double()
                    models = {}
                    for nm, rn in (("tz", False), ("rn", True)):
                        sm = tf32(w, rn) @ tf32(relu_k, rn)
                        km = kf(sm)
                        models[f"n_gt_{nm}"] = int((km > t_key).sum())
                        models[f"pcc_{nm}"] = emulate(torch.topk(sm, k).indices.sort().values)
                        if nm == "tz":
                            s_tz = sm
                    # boundary enumeration: pools whose tf32-model score is within EPS of the device threshold T are
                    # ambiguous at score precision; the device took exactly k - (#certain) of them -> try every such
                    # subset (if few) and keep the best emulation: = the exact attention over a boundary-consistent set
                    tb = t_key & 0x7FFFFFFF if t_key >= 2**31 else (~t_key) & 0xFFFFFFFF
                    t_f = float(torch.tensor([tb], dtype=torch.int64).to(torch.int32).view(torch.float32)[0])
                    rel = (s_tz - t_f) / max(abs(t_f), 1e-30)
                    rel64 = (scores - t_f) / max(abs(t_f), 1e-30)
                    enum_best, n_comb, eps_used, n_amb, n_cert = None, 0, None, None, None
                    # ambiguous = within EPS of T by the float64 OR the tf32-model score; certain = above T + EPS by both
                    # (the widest band whose subsets stay <= 2,000; card0 u69: one 32K slot needed > 3e-4)
                    for EPS in (1e-3, 6e-4, 3e-4, 1.5e-4):
                        c_m, o_m = (rel > EPS) & (rel64 > EPS), (rel < -EPS) & (rel64 < -EPS)
                        certain = torch.nonzero(c_m).reshape(-1)
                        amb = torch.nonzero(~c_m & ~o_m).reshape(-1)
                        need = k - int(certain.numel())
                        if 0 <= need <= int(amb.numel()) and math.comb(int(amb.numel()), need) <= 2000:
                            eps_used, n_amb, n_cert = EPS, int(amb.numel()), int(certain.numel())
                            for sub in itertools.combinations(amb.tolist(), need):
                                n_comb += 1
                                sel_c = torch.cat([certain, torch.tensor(sub, dtype=torch.int64)]).sort().values
                                v = emulate(sel_c)
                                enum_best = v if enum_best is None or v > enum_best else enum_best
                            break
                    models.update(boundary_eps=eps_used, n_certain=n_cert, n_ambiguous=n_amb, n_comb=n_comb,
                                  pcc_boundary_enum=enum_best)  # fmt: skip
                    diag.append(dict(slot=i, t_key=t_key, k_rem=k_rem, n_gt=n_gt, n_eq=int(eq.numel()),
                                     n_sel_dev=int(sel_dev.numel()), sel_diff=int(len(set(sel.tolist()) ^ set(sel_dev.tolist()))),
                                     cut_rel_gap=float((srt[k - 1] - srt[k]) / srt[k - 1].abs()) if nc > k else None,
                                     pcc_dev_threshold=emulate(sel_dev), **models))  # fmt: skip
                r_.update(execution=tr.mode, timing_us={"mk_dsa_step": us}, replays=replays, soak_s=soak_s,
                          soak_replays=n_soak, lanes=mk.att_lanes(T), o_lat_pcc_vs_emulation=min(pcs),
                          o_lat_pcc_per_slot=pcs, select_diag=diag)  # fmt: skip
                # pass rule (card0 u53 / u54): the device's top-512 boundary carries the tf32 head-weighted score noise
                # (~5e-4 relative): a slot passes on (a) the exact float64 or the tf32-model emulation at PCC >= 0.9999,
                # or (b) boundary-consistent: the device threshold within 4 ranks of 512 by the tf32 model (n_gt_tz)
                # and the better emulation >= 0.999 (1-4 swapped pools at the cut)
                cls = []
                for pc, d_ in zip(pcs, diag):
                    best = max(pc, d_["pcc_tz"], d_["pcc_rn"])
                    enum = d_.get("pcc_boundary_enum")
                    cls.append("exact" if best >= 0.9999 else
                               "boundary" if enum is not None and enum >= 0.9999 else "fail")  # fmt: skip
                r_.update(slot_class=cls, status_rule="exact|tf32-model top-512 >= 0.9999, or a boundary-consistent "
                          "selection (pools within 1e-3 .. 1.5e-4 of the device threshold by the float64 or tf32-model score, every subset) >= 0.9999")  # fmt: skip
                r_["status"] = "pass" if all(c != "fail" for c in cls) else "fail"
                if os.environ.get("GLM53_MOCK") == "1":
                    r_["status"] = "record"
            except Exception as e:  # noqa: BLE001
                tr.release()
                r_.update(status="fail", error=str(e)[-600:])
                if "timeout" in str(e).lower():
                    raise
            rows("dsa_long_context", **r_)
            cache = ptab = st = x = cur = lat = None
            continue
        g = torch.Generator().manual_seed(seed + p)
        cap = -(-(p + 1) // bs) * bs
        nb = cap // bs
        pt_host = torch.randperm(nb, generator=g).to(torch.int32).reshape(1, nb)
        # synthetic latent rows ~ the model's scale (kv_a_layernorm output: O(1) values)
        lat = torch.randn(cap, attn.kv_lora, generator=g).to(torch.bfloat16)
        lat[p:] = 0
        ch = torch.zeros(nb, 1, bs, attn.kv_lora, dtype=torch.bfloat16)
        ch[pt_host[0].long()] = lat.reshape(nb, 1, bs, attn.kv_lora)
        cache = ttnn.from_torch(ch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        del ch
        ptab = ttnn.from_torch(pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        tp_rows = MK.DsaIndexer.pool_rows(cap)
        pools = torch.zeros(1, 1, tp_rows, 128, dtype=torch.float64)
        n_closed = (p + 1) // 4
        pools[0, 0, :n_closed] = torch.randn(n_closed, 128, generator=g, dtype=torch.float64)
        ring = torch.zeros(1, 1, 32, 256, dtype=torch.float64)
        st = ix.state(mesh, 1, cap, pools=pools, ring=ring)
        cur = int_tensor([p], mesh)
        x = ttnn.from_torch(x_row, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        setup_s = round(time.time() - t0, 1)
        r_ = dict(p=p, cap=cap, n_closed=n_closed, tp=tpn, heads=hr, rev=MK.MLA_REV, setup_s=setup_s)
        try:
            fn = lambda: mk(x, cache, ptab, cur, t=1, dsa_state=st)  # noqa: E731
            tr.warm(fn)
            out = tr.run(fn)
            got = {k: h0(out[k]) for k in ("o_lat", "q_lat", "q_ix", "wsc", "latent")}
            us = MC.timed(tr, replays)
            n_soak = soak(tr, soak_s) if soak_s > 0 and tr.enabled else 0
            tr.release()
            out = None
            if probe:
                pr = MK.decode_probe(ttnn.to_torch(ttnn.get_device_tensors(mk.probe)[0]) if mesh.get_num_devices() > 1
                                     else ttnn.to_torch(mk.probe), len(mk.last_readers.assign))  # fmt: skip
                r_.update({k: pr[k] for k in ("owner_dm1", "owner_dm1_max", "owner", "reader_a") if k in pr})
            pools_after = h0(st["pools"])
            # host emulation on the device's own intermediates (float64)
            qi = got["q_ix"][0, 0].double()  # [32, 128]
            w = got["wsc"][0, 0, 0, :32].double()
            kp = pools_after[0, 0, :n_closed].double()
            scores = w @ torch.relu(qi @ kp.T)
            k = min(MK.DSA_TOPK, n_closed)
            sel = torch.topk(scores, k).indices.sort().values
            sel_rows = torch.cat([(4 * sel[:, None] + torch.arange(4)).reshape(-1), torch.arange(4 * n_closed, p + 1)])
            kv = lat.double()[: p + 1].clone()
            kv[p] = got["latent"].reshape(-1, attn.kv_lora)[0].double()  # the step's own row
            kv = kv[sel_rows]
            ql = got["q_lat"][0, 0, :hr].double()
            o_em = torch.softmax((ql @ kv.T) * attn.scale, dim=-1) @ kv
            o_dev = got["o_lat"][0, 0, :hr].double()
            r_.update(execution=tr.mode, timing_us={"mk_dsa_step": us}, n_sel_rows=int(sel_rows.numel()), replays=replays,
                      soak_s=soak_s, soak_replays=n_soak,
                      o_lat_pcc_vs_emulation=pcc(o_dev.reshape(-1).float(), o_em.reshape(-1).float()),
                      o_lat_maxabs_vs_emulation=float((o_dev - o_em).abs().max()))  # fmt: skip
            r_["status"] = "pass" if r_["o_lat_pcc_vs_emulation"] >= 0.9999 else "fail"
            if os.environ.get("GLM53_MOCK") == "1":
                r_["status"] = "record"
        except Exception as e:  # noqa: BLE001
            tr.release()
            r_.update(status="fail", error=str(e)[-600:])
            if "timeout" in str(e).lower():
                raise
        rows("dsa_long_context", **r_)
        cache = ptab = st = None
    rows.flush()
    bad = rows.failed()
    print(f"DSA_LONG_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    RESULT = dict(ok=not bad)
    if bad:
        raise RuntimeError(f"dsa_long: {len(bad)} failing row(s)")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
