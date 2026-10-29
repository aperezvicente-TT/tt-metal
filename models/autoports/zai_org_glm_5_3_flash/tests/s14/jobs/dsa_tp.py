# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, DSA sparse decode on the TP mesh; program 2026-10-21 02:57:57Z SCOPE): TPMla(dsa=
DsaIndexer) on the open mesh -- every chip runs its head shard's MkMla(dsa) program with the REPLICATED indexer (all 32
indexer heads, the same top-512 selection on every chip) + the in-program fabric all-reduce of o_proj's partial -- vs
the replicated one-chip MkMla(dsa) on the same state (Stage 7b's validated program) and vs the fp32 oracle
(tests/s7b/golden_mla.dsa_step). The decode slots sit at ARGS["pos"] (default [4098]: past the dense-exact bound 2,051),
over Stage 7b's synthetic prompt (tests/s7b/dev_mk_dsa._prompt); latent cache, pooled keys and ring rows prefilled
from the CPU oracle (golden_mla.dsa_state: the contract Stage 16's prefill writes).

Rows (GLM53_STAGE_EVIDENCE pcc.jsonl / tests.jsonl):
  dsa_tp_vs_full: s1 bitwise across chips, s1 / o_lat (this chip's heads) vs the replicated program, the step's index
      writes (ring row p % 4, pooled key p // 4) bitwise across chips and equal to the replicated program's;
  dsa_tp_golden: s1 vs the fp32 oracle (bar 0.995, as dev_mk_dsa's golden_fp32_dsa);
  dsa_tp_timing: traced replay median of the TP step (info).
ARGS: pos (list of positions, <= 32 slots: t > 4 = Stage 14's native t), replays (2000), soak_s (0), vtp (8 on a pair: the
TP8 per-chip shard -- timing of the production geometry; the numerics rows are skipped as in dev_mk_tp_mesh's vtp mode),
dsa_kb / rings (the indexer readers' K blocks / MkMla ring budgets; default MkMla's (32, 8) and dev_mk_dsa's shard rings;
the stack's: [16, 4] / [140, 105, 64, 280]), dense_ab (vtp: also time the DENSE TPMla on the same rows -> program "dense"),
ix_ab (the reference program on the pre-rev-46 full-tile index writes), ref_env (knobs set while the reference program
builds, e.g. {"GLM53_MLA_DSA_IX_LANES": "0"}: rev 47's lane index cores vs one index core)."""

import importlib
import os
import time

import torch

import ttnn

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    from models.autoports.zai_org_glm_5_3_flash.tests.common import pcc
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import dev_mk_dsa as DSA
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

    for m in (MK, TPM, MC, G, DSA):
        importlib.reload(m)

    mesh = MESH  # noqa: F821
    pos = [int(v) for v in ARGS.get("pos", [4098])]  # noqa: F821
    replays = int(ARGS.get("replays", 2000))  # noqa: F821  (>= 2,000: program 03:20Z DEVICE DUTY)
    soak_s = float(ARGS.get("soak_s", 0))  # noqa: F821
    vtp = ARGS.get("vtp")  # noqa: F821
    assert 1 <= len(pos) <= 32, pos
    ndev = int(mesh.get_num_devices())
    rows = MC.Rows(DSA.LAYER)
    def soak(tr_, seconds):
        """program DEVICE DUTY rule (2026-10-21 03:20Z): replay the captured program back to back for `seconds`
        (unwatched soak: the race / hang detector) -> replays done"""
        n, t_end = 0, time.time() + seconds
        while time.time() < t_end:
            tr_.replay()
            n += 1
        return n

    tr = Traced(mesh)
    per_chip = lambda t: [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]  # noqa: E731
    h0 = lambda t: MK._host0(t).double()  # noqa: E731
    knobs = dict(rev=MK.MLA_REV, pos=pos, tp=ndev, mesh=list(mesh.shape), vtp=vtp)

    L = MC.layer_of(mesh, CTX, DSA.LAYER)  # noqa: F821
    attn = L.attn
    W = dict(
        qa_kva=h0(attn.ds_qa_kva.weight).reshape(-1, attn.ds_qa_kva.n_pad)[:, : attn.ds_qa_kva.n],
        qa_norm=h0(attn.qa_norm).reshape(-1)[: attn.q_lora],
        kva_norm=h0(attn.kva_norm).reshape(-1)[: attn.kv_lora],
        qb=h0(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n],
        w_uk=h0(attn.w_uk).reshape(attn.h, attn.dqk, attn.kv_lora),
        w_uv=h0(attn.w_uv).reshape(attn.h, attn.kv_lora, attn.dv),
        o=h0(attn.ds_o.weight).reshape(attn.h * attn.dv, -1)[:, : attn.ds_o.n],
    )
    ix = MK.DsaIndexer(mesh, L.source, DSA.LAYER)
    t0 = time.time()
    gd, hn_all, lat_all = DSA._prompt(L, W, attn, max(pos) + 1)
    ir = G.indexer_rows(hn_all, ix.host)
    g_in = L.g.reshape(-1, 16384).double()
    xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, len(pos))]
    prompt_s = round(time.time() - t0, 1)
    t = len(pos)

    rings = tuple(int(v) for v in ARGS.get("rings", DSA.RINGS_SHARD))  # noqa: F821
    dkb = {"dsa_kb": tuple(int(v) for v in ARGS["dsa_kb"])} if ARGS.get("dsa_kb") else {}  # noqa: F821
    knobs.update(rings=list(rings), dsa_kb=list(dkb.get("dsa_kb", (32, 8))))
    probe = bool(ARGS.get("probe", False)) and bool(vtp)  # noqa: F821  (rev 45 marks incl. the index core; timing runs)
    if ndev == 1 and not vtp:  # one chip, all 64 heads: the standard one-chip program (the TP shard's rings do not fit 64
        # heads' readers: 1.82 MB of CBs) -- with ix_ab this is the rev 46 index-write A/B on the same program
        tp = MK.MkMla(mesh, L.a_args, attn, dsa=ix, **dkb)
    else:
        tp = TPM.TPMla(mesh, L.a_args, attn, kb=DSA.KB_SHARD, rings=rings, ts=(t,), dsa=ix, vtp=vtp, probe=probe, **dkb)
    mkp = getattr(tp, "mk", tp)  # the MkMla of the TP program (or the one-chip program itself)
    knobs.update(ix_lanes=mkp.ix_lanes(t) if hasattr(mkp, "ix_lanes") else 1)
    ran = {}  # what the programs actually built (rev 53, review S2: GLM53_MLA_DSA_MCAST drops silently past 16 rectangles)
    # the one-chip reference streams the indexer phases with the SAME K blocks (I1's fp32 accumulation order -> the
    # bitwise index writes; u37 08:06Z: TP (16, 4) vs the reference's default (32, 8) differed at t = 32)
    full = MK.MkMla(mesh, L.a_args, attn, dsa=ix, **dkb) if not vtp else None
    ix_ab = bool(ARGS.get("ix_ab", False))  # noqa: F821  (the reference program on the pre-rev-46 full-tile index writes)
    ref_env = {k: str(v) for k, v in (ARGS.get("ref_env") or {}).items()}  # noqa: F821
    if ix_ab:
        ref_env.setdefault("GLM53_MLA_DSA_IX_ROWS", "0")
    knobs.update(ix_ab=ix_ab, ref_env=ref_env or None)
    keys = ("s1", "o_lat", "latent")
    got = {}
    for name, prog in (("tp", tp), ("full", full)) if full is not None else (("tp", tp),):
        c = DSA.Case(mesh, L, ix, pos, lat_all, ir, xs, dsa=True)
        fn = lambda prog=prog, c=c: {k: v for k, v in prog(c.x, c.cache, c.pt, c.cur, t=c.t, dsa_state=c.st).items()
                                     if k in keys}  # fmt: skip
        env_b = {k: os.environ.get(k) for k in ref_env} if name == "full" else {}
        os.environ.update(ref_env if name == "full" else {})  # read at the program build (each call builds its descriptor)
        try:
            tr.warm(fn)
            out = tr.run(fn)
        finally:
            for k, v in env_b.items():
                os.environ.pop(k, None) if v is None else os.environ.update({k: v})
        got[name] = {k: per_chip(v) for k, v in out.items()}
        ran[name] = dict(mcast_rects=getattr(getattr(prog, "mk", prog), "last_mcast", None))
        got[name]["pools"], got[name]["ring"] = per_chip(c.st["pools"]), per_chip(c.st["ring"])
        if name == "tp":
            us = MC.timed(tr, replays)
            if probe:
                pr = MK.decode_probe(ttnn.to_torch(ttnn.get_device_tensors(mkp.probe)[0]), len(mkp.last_readers.assign))
                rows("dsa_tp_probe", t=t, chip=0, status="info", timing_us={"tp_step": us}, program="dsa",
                     lanes=mkp.att_lanes(t), ot=mkp.ot(t), **{k: pr[k] for k in pr}, **knobs)  # fmt: skip
            n_soak = soak(tr, soak_s) if soak_s > 0 and tr.enabled else 0
            ref_state = c.ref_state
        tr.release()
        out = c = None
    rows("dsa_tp_timing", t=t, execution=tr.mode, status="info", timing_us={"tp_step": us}, prompt_s=prompt_s,
         replays=replays, soak_s=soak_s, soak_replays=n_soak, program="dsa", lanes=mkp.att_lanes(t), ot=mkp.ot(t),
         mcast_rects=getattr(mkp, "last_mcast", None), **knobs)  # fmt: skip
    if vtp and ARGS.get("dense_ab"):  # noqa: F821  (the dense TPMla on the same rows: what DSA costs at this t / regime)
        tpd = TPM.TPMla(mesh, L.a_args, attn, kb=DSA.KB_SHARD, rings=DSA.RINGS_SHARD, ts=(t,), vtp=vtp,
                        shard=tp.shard, probe=probe)  # fmt: skip  (its own fabric phase: the dense program's lanes differ)
        c = DSA.Case(mesh, L, ix, pos, lat_all, ir, xs, dsa=False)
        fn = lambda c=c: {k: v for k, v in tpd(c.x, c.cache, c.pt, c.cur, t=c.t).items() if k == "s1"}  # noqa: E731
        tr.warm(fn)
        tr.run(fn)
        us_d = MC.timed(tr, replays)
        if probe:
            pr = MK.decode_probe(ttnn.to_torch(ttnn.get_device_tensors(tpd.mk.probe)[0]), len(tpd.mk.last_readers.assign))
            rows("dsa_tp_probe", t=t, chip=0, status="info", timing_us={"tp_step": us_d}, program="dense",
                 lanes=tpd.mk.att_lanes(t), ot=tpd.mk.ot(t), **{k: pr[k] for k in pr}, **knobs)  # fmt: skip
        tr.release()
        c = None
        rows("dsa_tp_timing", t=t, execution=tr.mode, status="info", timing_us={"tp_step": us_d}, prompt_s=prompt_s,
             replays=replays, program="dense", lanes=tpd.mk.att_lanes(t), ot=tpd.mk.ot(t), **knobs)  # fmt: skip
    if not vtp:  # (vtp: the TP8 shard on a pair, 2 of 8 partials reduced -> timing only)

        hr = attn.h // ndev
        r_ = dict(t=t, **knobs)
        s1 = got["tp"]["s1"]
        r_["s1_bitwise_across_chips"] = all(torch.equal(s1[0], s) for s in s1[1:])
        r_["s1_pcc_vs_full"] = min(pcc(s1[d].reshape(-1), got["full"]["s1"][d].reshape(-1)) for d in range(ndev))
        o_p = []
        for d in range(ndev):
            o_f = got["full"]["o_lat"][d].reshape(1, t, attn.h, -1)[:, :, d * hr : (d + 1) * hr]
            o_p.append(pcc(got["tp"]["o_lat"][d][:, :t, :hr].reshape(-1), o_f.reshape(-1)))
        r_["o_lat_pcc_vs_full_min"] = min(o_p)
        if ndev == 1:  # rev 53 (review S7): both programs are the one-chip 64-head MkMla -> exact equality is meaningful
            r_["s1_equal_vs_full"] = bool(torch.equal(s1[0], got["full"]["s1"][0]))
            r_["o_lat_equal_vs_full"] = bool(torch.equal(got["tp"]["o_lat"][0], got["full"]["o_lat"][0]))
            r_["latent_equal_vs_full"] = bool(torch.equal(got["tp"]["latent"][0], got["full"]["latent"][0]))
        idx_eq, idx_cross, idx_dmax = [], [], []
        for i, p in enumerate(pos):
            for name in ("pools", "ring"):
                sl = (lambda a: a[i, 0, p // 4]) if name == "pools" else (lambda a: a[i, 0, p % 4])
                v = [sl(a) for a in got["tp"][name]]
                idx_cross.append(all(torch.equal(v[0], w) for w in v[1:]))
                idx_eq.append(torch.equal(v[0], sl(got["full"][name][0])))
                idx_dmax.append(float((v[0].double() - sl(got["full"][name][0]).double()).abs().max()))
        r_["index_writes_bitwise_across_chips"] = all(idx_cross)
        r_["index_writes_eq_full"] = all(idx_eq)
        r_["index_writes_maxabs_vs_full"] = max(idx_dmax) if idx_dmax else None
        r_["index_writes_mismatch_slots"] = sum(1 for e in idx_eq if not e)
        r_["status"] = ("pass" if r_["s1_bitwise_across_chips"] and r_["s1_pcc_vs_full"] >= 0.9999
                        and r_["o_lat_pcc_vs_full_min"] >= 0.999 and all(idx_cross) and all(idx_eq) else "fail")  # fmt: skip
        if os.environ.get("GLM53_MOCK") == "1":
            r_["status"] = "record"
        r_["ran"] = ran
        rows("dsa_tp_vs_full", **r_)

        sc = dict(eps=attn.eps, scale=attn.scale, heads=attn.h, q_lora=attn.q_lora)
        for i, p in enumerate(pos):
            pl, rg = ref_state[i]
            ref = G.dsa_step(xs[i].reshape(1, -1), p, L.a_args, W, ix.host, lat_all[:p], pl, rg, **sc)
            s1_i = s1[0][i, 0] if t <= 4 else s1[0][0, 0, i].reshape(4, -1)
            a_, b_ = s1_i.reshape(-1), ref["s1"].reshape(-1)[: s1_i.numel()].float()
            r_ = dict(t=t, slot=i, p=p, sparse=p + 1 > MK.DSA_DENSE_MAX, s1_pcc_vs_fp32=pcc(a_, b_),
                      s1_maxabs_vs_fp32=float((a_ - b_).abs().max()), **knobs)  # fmt: skip
            r_["status"] = "pass" if r_["s1_pcc_vs_fp32"] >= DSA.BAR else "fail"
            if os.environ.get("GLM53_MOCK") == "1":
                r_["status"] = "record"
            rows("dsa_tp_golden", **r_)
    rows.flush()
    bad = rows.failed()
    print(f"DSA_TP_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    RESULT = dict(ok=not bad, rows=len(rows.rows))
    if bad:
        raise RuntimeError(f"dsa_tp: {len(bad)} failing row(s)")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
