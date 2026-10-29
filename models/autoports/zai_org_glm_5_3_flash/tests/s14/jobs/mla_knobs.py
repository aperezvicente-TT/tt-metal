# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, M5 glue on the sparse_dsa attention half): the served DENSE TPMla (14b's construction:
tt/tp_dsa.py MLA_KB_SHARD / MLA_RINGS_SHARD, in-contract positions <= 2,050) under mHC glue knob variants, device vs
device. 14b's StackBackend pops every mHC knob for the sparse_dsa kind (stack_model.set_knobs("dsa")), so its MkMla runs
mhc_pre / mhc_post on their defaults while the KDA kinds take the TP trims (PRE_EARLY, HN_FIRST, dz, ...). Each variant
sets its env, builds its own TPMla on the SAME resident head shard (only the program objects differ), replays the same
rows, and compares s1 / o_lat / latent with the first variant (the base) on the device outputs.

Rows (GLM53_STAGE_EVIDENCE pcc.jsonl / tests.jsonl / perf.jsonl):
  mla_knob_timing: traced replay median (blocking) + the back-to-back batch mean of the variant's program (info);
  mla_knob_vs_base: s1 bitwise across chips, s1 / o_lat / latent equal / PCC / maxabs / rel_l2 vs the base variant
      (status: "bitwise" variants must be equal, "pcc" variants PCC >= 0.99999 per output; rel_l2 recorded).
ARGS: pos (slots, <= 32; default [1500]), vtp (8 on a pair: the TP8 per-chip program), replays (2000), batch (2000),
soak_s (0, per variant), variants ([[name, {env}, "bitwise" | "pcc"], ...]; the first is the base)."""

import importlib
import os
import time

import torch

import ttnn

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
# every knob a variant may set: cleared before each variant's build (a variant sets only its own)
KEYS = ("GLM53_MHC_PRE_EARLY", "GLM53_MHC_HN_FIRST", "GLM53_MHC_MODE", "GLM53_MHC_FPU_COLLAPSE", "GLM53_MHC_ROWREAD",
        "GLM53_MHC_POST_ROWWRITE", "GLM53_MHC_PRE_SPLIT", "GLM53_MK_W_DELAY_US", "GLM53_MLA_DEFER")  # fmt: skip
old = {k: os.environ.get(k) for k in set(ENV) | set(KEYS)}
os.environ.update(ENV)
try:
    from models.autoports.zai_org_glm_5_3_flash.tests.common import pcc
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import dev_mk_dsa as DSA
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa as TPD
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

    for m in (MK, TPM, MC, G, DSA, TPD):
        importlib.reload(m)

    mesh = MESH  # noqa: F821
    pos = [int(v) for v in ARGS.get("pos", [1500])]  # noqa: F821
    replays = int(ARGS.get("replays", 2000))  # noqa: F821
    n_batch = int(ARGS.get("batch", 2000))  # noqa: F821
    soak_s = float(ARGS.get("soak_s", 0))  # noqa: F821
    vtp = ARGS.get("vtp")  # noqa: F821
    variants = ARGS.get("variants") or [["base", {}, "bitwise"]]  # noqa: F821
    assert 1 <= len(pos) <= 32 and max(pos) + 1 <= MK.DSA_DENSE_MAX, pos  # the served (dense) program's contract
    t = len(pos)
    ndev = int(mesh.get_num_devices())
    rows = MC.Rows(DSA.LAYER)
    tr = Traced(mesh)
    per_chip = lambda v: [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(v)]  # noqa: E731
    h0 = lambda v: MK._host0(v).double()  # noqa: E731
    knobs = dict(rev=MK.MLA_REV, pos=pos, tp=ndev, mesh=list(mesh.shape), vtp=vtp, kb=list(TPD.MLA_KB_SHARD),
                 rings=list(TPD.MLA_RINGS_SHARD))  # fmt: skip

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
    t0 = time.time()
    gd, hn_all, lat_all = DSA._prompt(L, W, attn, max(pos) + 1)
    g_in = L.g.reshape(-1, 16384).double()
    xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, t)]
    prompt_s = round(time.time() - t0, 1)

    def soak(seconds):
        n, t_end = 0, time.time() + seconds
        while time.time() < t_end:
            tr.replay()
            n += 1
        return n

    def batch_us(n):
        """n replays back to back (non-blocking), one sync: mean us per replay"""
        if not tr.enabled or tr.tid is None:
            return None
        t1 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh, tr.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        return round((time.perf_counter() - t1) * 1e6 / n, 1)

    shard, fabrics, base = None, None, None
    for name, env, expect in variants:
        for k in KEYS:
            os.environ.pop(k, None)
        os.environ.update(ENV)
        os.environ.update({k: str(v) for k, v in env.items()})
        tp = TPM.TPMla(mesh, L.a_args, attn, kb=TPD.MLA_KB_SHARD, rings=TPD.MLA_RINGS_SHARD, ts=(t,), vtp=vtp,
                       shard=shard, fabrics=fabrics)  # fmt: skip
        shard, fabrics = tp.shard, (tp.fabrics or None)  # one set of lane buffers / counters for every variant
        c = DSA.Case(mesh, L, None, pos, lat_all, None, xs, dsa=False)
        keys = ("s1", "o_lat", "latent")
        fn = lambda tp=tp, c=c: {k: v for k, v in tp(c.x, c.cache, c.pt, c.cur, t=c.t).items() if k in keys}  # noqa: E731
        tr.warm(fn)
        out = tr.run(fn)
        got = {k: per_chip(v) for k, v in out.items() if v is not None}
        us = MC.timed(tr, replays)
        us_b = batch_us(n_batch)
        n_soak = soak(soak_s) if soak_s > 0 and tr.enabled else 0
        tr.release()
        out = c = tp = None
        envk = {k: os.environ.get(k) for k in KEYS if os.environ.get(k) is not None}
        rows("mla_knob_timing", t=t, variant=name, execution=tr.mode, status="info",
             timing_us={"tp_step": us, "tp_step_batch": us_b}, replays=replays, batch=n_batch, soak_s=soak_s,
             soak_replays=n_soak, variant_env=envk, prompt_s=prompt_s, **knobs)  # fmt: skip
        if base is None:
            base = got
            continue
        r_ = dict(t=t, variant=name, expect=expect, variant_env=envk, **knobs)
        s1 = got["s1"]
        r_["s1_bitwise_across_chips"] = all(torch.equal(s1[0], s) for s in s1[1:])
        ok = r_["s1_bitwise_across_chips"]
        for k in keys:
            if k not in got or k not in base:
                continue
            a = torch.cat([v.reshape(-1) for v in got[k]])
            b = torch.cat([v.reshape(-1) for v in base[k]])
            r_[f"{k}_equal_vs_base"] = bool(torch.equal(a, b))
            r_[f"{k}_pcc_vs_base"] = pcc(a, b)
            r_[f"{k}_maxabs_vs_base"] = float((a - b).abs().max())
            r_[f"{k}_rel_l2_vs_base"] = float((a.double() - b.double()).norm() / max(b.double().norm(), 1e-30))
            if expect == "bitwise":
                ok = ok and r_[f"{k}_equal_vs_base"]
            else:
                ok = ok and r_[f"{k}_pcc_vs_base"] >= 0.99999  # (rel_l2 recorded: bf16 one-ulp flips alone give ~3e-3)
        r_["status"] = "record" if os.environ.get("GLM53_MOCK") == "1" else ("pass" if ok else "fail")
        rows("mla_knob_vs_base", **r_)
    rows.flush()
    bad = rows.failed()
    print(f"MLA_KNOBS_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    RESULT = dict(ok=not bad, rows=len(rows.rows))
    if bad:
        raise RuntimeError(f"mla_knobs: {len(bad)} failing row(s)")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
