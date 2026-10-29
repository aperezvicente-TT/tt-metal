# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 MkMla rev 38, the persistent stack's DSA store; GLM53_STACK_DSA): one sparse_dsa decode
launch of t rows (TPMla(dsa) on a mesh, MkMla(dsa) on one chip) whose dsa_state is ONE store of `slots` slots in
mk_mla.DsaIndexer.stack_state's layout (pools [S Tp / 32, 1, 32, 128] = Stage 16's pooled-key store, ring [S, 1, 32,
256]) with the launch's rows on slots s0 .. s0 + t - 1 (dsa_slot_addrs: the base address moved by s0 pps / banks pages),
vs the same launch on its own per-launch state (S = t, s0 = 0: Stage 7b's validated program, as dsa_tp.py). The other
slots of the store hold a random sentinel. State prefilled from the CPU oracle over Stage 7b's synthetic prompt
(golden_mla.dsa_state: the contract Stage 16's prefill writes).

Rows (GLM53_STAGE_EVIDENCE pcc.jsonl / tests.jsonl):
  dsa_slot_base: s1 / o_lat bitwise vs the per-launch state (every chip), the store's slots s0.. after the step bitwise
      equal to the per-launch state after its step (pools + ring), every other slot unchanged (the sentinel), a
      misaligned s0 (s0 pps not a multiple of the bank count) rejected on the host;
  dsa_slot_timing: traced replay median of the store launch (+ the soak's replay count).
ARGS: pos (list, <= 32 rows; default [4098, 2050, 8194, 6146]: sparse + dense rows), s0 (4), slots (12), replays (2000),
soak_s (0)."""

import importlib
import os
import time

import torch

import ttnn

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import dev_mk_dsa as DSA
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

    for m in (MK, TPM, MC, G, DSA):
        importlib.reload(m)

    mesh = MESH  # noqa: F821
    pos = [int(v) for v in ARGS.get("pos", [4098, 2050, 8194, 6146])]  # noqa: F821
    s0 = int(ARGS.get("s0", 4))  # noqa: F821
    slots = int(ARGS.get("slots", 12))  # noqa: F821
    replays = int(ARGS.get("replays", 2000))  # noqa: F821  (>= 2,000: program 03:20Z DEVICE DUTY)
    soak_s = float(ARGS.get("soak_s", 0))  # noqa: F821
    t = len(pos)
    assert 1 <= t <= 32 and 0 < s0 and s0 + t <= slots, (pos, s0, slots)  # (t > 4: the native-t program, token lanes)
    ndev = int(mesh.get_num_devices())
    rows = MC.Rows(DSA.LAYER)
    tr = Traced(mesh)
    per_chip = lambda x: [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(x)]  # noqa: E731
    h0 = lambda x: MK._host0(x).double()  # noqa: E731
    knobs = dict(rev=MK.MLA_REV, pos=pos, s0=s0, slots=slots, tp=ndev, mesh=list(mesh.shape))

    def soak(tr_, seconds):
        """program DEVICE DUTY rule (2026-10-21 03:20Z): back-to-back replays for `seconds` -> replays done"""
        n, t_end = 0, time.time() + seconds
        while time.time() < t_end:
            tr_.replay()
            n += 1
        return n

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
    gd, hn_all, lat_all = DSA._prompt(L, W, attn, max(pos) + 1)
    ir = G.indexer_rows(hn_all, ix.host)
    g_in = L.g.reshape(-1, 16384).double()
    xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, t)]
    if ndev > 1:
        prog = TPM.TPMla(mesh, L.a_args, attn, kb=DSA.KB_SHARD, rings=DSA.RINGS_SHARD, ts=(t,), dsa=ix)
    else:
        prog = MK.MkMla(mesh, L.a_args, attn, dsa=ix)
    keys = ("s1", "o_lat")

    def run(c, st, timing: bool):
        fn = lambda: {k: v for k, v in prog(c.x, c.cache, c.pt, c.cur, t=c.t, dsa_state=st).items()  # noqa: E731
                      if k in keys}  # fmt: skip
        tr.warm(fn)
        out = tr.run(fn)
        got = {k: per_chip(v) for k, v in out.items()}
        us = MC.timed(tr, replays) if timing else None
        n_soak = soak(tr, soak_s) if timing and soak_s > 0 and tr.enabled else 0
        tr.release()
        return got, us, n_soak

    # A: the per-launch state (S = t, s0 = 0)
    ca = DSA.Case(mesh, L, ix, pos, lat_all, ir, xs, dsa=True)
    got_a, _, _ = run(ca, ca.st, False)
    tp_rows = ca.st["tp"]
    pools_a = [p.reshape(t, 1, tp_rows, 128) for p in per_chip(ca.st["pools"])]
    ring_a = per_chip(ca.st["ring"])
    # B: the same inputs on slots s0.. of a store (stack_state's layout), the other slots a sentinel
    cb = DSA.Case(mesh, L, ix, pos, lat_all, ir, xs, dsa=True)
    g = torch.Generator().manual_seed(7)
    sp = torch.randn(slots, 1, tp_rows, 128, generator=g).to(torch.bfloat16)
    sr = torch.randn(slots, 1, 32, 256, generator=g).to(torch.bfloat16)
    sp[s0 : s0 + t] = MK._host0(cb.st["pools"]).reshape(t, 1, tp_rows, 128).to(torch.bfloat16)
    sr[s0 : s0 + t] = MK._host0(cb.st["ring"]).to(torch.bfloat16)
    ttnn.deallocate(cb.st["pools"])
    ttnn.deallocate(cb.st["ring"])
    put = lambda v: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    store = dict(pools=put(sp.reshape(slots * tp_rows // 32, 1, 32, 128)), ring=put(sr), tp=tp_rows, slots=slots, s0=s0)
    got_b, us, n_soak = run(cb, store, True)
    pools_b = [p.reshape(slots, 1, tp_rows, 128) for p in per_chip(store["pools"])]
    ring_b = per_chip(store["ring"])
    rows("dsa_slot_timing", t=t, execution=tr.mode, status="info", timing_us={"store_step": us}, replays=replays,
         soak_s=soak_s, soak_replays=n_soak, **knobs)  # fmt: skip

    others = [s for s in range(slots) if not s0 <= s < s0 + t]
    r_ = dict(t=t, **knobs)
    r_["outputs_bitwise"] = all(torch.equal(got_a[k][d], got_b[k][d]) for k in keys for d in range(ndev))
    r_["store_slots_eq_launch_state"] = all(
        torch.equal(pools_b[d][s0 : s0 + t], pools_a[d]) and torch.equal(ring_b[d][s0 : s0 + t], ring_a[d])
        for d in range(ndev)
    )
    r_["other_slots_untouched"] = all(
        torch.equal(pools_b[d][others], sp[others].float()) and torch.equal(ring_b[d][others], sr[others].float())
        for d in range(ndev)
    )
    r_["state_written"] = all(  # the step wrote each row's ring row / pooled key into ITS slot
        not (torch.equal(pools_b[0][s0 + i], sp[s0 + i].float()) and torch.equal(ring_b[0][s0 + i], sr[s0 + i].float()))
        for i in range(t)
    )
    pps = tp_rows // 32 * 4
    banks = int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).num_banks)
    bad_s0 = next((s for s in range(1, slots - t + 1) if s * pps % banks), None)
    if bad_s0 is None:
        r_["misaligned_s0_rejected"] = None
    else:
        try:
            MK.dsa_slot_addrs(dict(store, s0=bad_s0), t)
            r_["misaligned_s0_rejected"] = False
        except ValueError:
            r_["misaligned_s0_rejected"] = True
    r_.update(pps=pps, banks=banks, bad_s0=bad_s0)
    ok = r_["outputs_bitwise"] and r_["store_slots_eq_launch_state"] and r_["other_slots_untouched"]
    r_["status"] = "pass" if ok and r_["state_written"] and r_["misaligned_s0_rejected"] is not False else "fail"
    if os.environ.get("GLM53_MOCK") == "1":
        r_["status"] = "record"
    rows("dsa_slot_base", **r_)
    rows.flush()
    bad = rows.failed()
    print(f"DSA_SLOTS_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    RESULT = dict(ok=not bad, rows=len(rows.rows))
    if bad:
        raise RuntimeError(f"dsa_slots: {len(bad)} failing row(s)")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
