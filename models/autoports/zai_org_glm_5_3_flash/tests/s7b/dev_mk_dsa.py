# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b DSA in the one-program attention half: MkMla(dsa=DsaIndexer) (5d m4_phase_spec I1..I5 + S1), traced, layer
GLM53_S7B_LAYER (default 3).

State: slot i decodes at position GLM53_DSA_POS[i] (default "4098,2050": one sparse token, one dense) over a synthetic
prompt = the HF capture's 2048 prompt rows (mHC pre reference -> hn) followed by the same rows with 5 % seeded
multiplicative noise; latent cache, pooled keys and ring rows are prefilled from the CPU float64 oracle
(tests/s7b/golden_mla.py dsa_state, the device's stored dtypes). Checks per token:
  * index core: ring row p % 4 and pooled key p // 4 vs the oracle (bf16 values);
  * fp32 oracle (golden_mla.dsa_step: its own exact top-512): o_lat, a, s1 (bar 0.995);
  * sparse tokens, device intermediates (q_idx, wsc, the pooled keys after the step, q_lat, the cache) -> the CPU
    emulation of the select (fp32 scores, exact top-512) and attention: o_lat (bar 0.9999), selection overlap with the
    oracle's;
  * GLM53_DSA_DENSE=1 (default): a dense-only case (positions <= 2050) MkMla(dsa) vs MkMla: o_lat / s1 bitwise (the D1
    path is unchanged), both traced -> the indexer's cost in the dense regime.
Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import os
import sys
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, int_tensor, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import MlaLayer, Rows, host, timed
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
POS = [int(v) for v in os.environ.get("GLM53_DSA_POS", "4098,2050").split(",")]
DENSE = os.environ.get("GLM53_DSA_DENSE", "1") == "1"
REPLAYS = int(os.environ.get("GLM53_MK_REPLAYS", "30"))
BAR, BAR_DEV = 0.995, 0.9999
# GLM53_DSA_TP=8: the TP8 geometry with DSA (one head group, 64 row chunks per token), emulated on this chip: the 8
# HeadShard programs on the same state -> sum of the partial a vs the full program's a, o_lat of each rank's heads
TP = int(os.environ.get("GLM53_DSA_TP", "0"))
KB_SHARD = tuple(int(v) for v in os.environ.get("GLM53_TP_KB", "8,16,8,16,16").split(","))
RINGS_SHARD = tuple(int(v) for v in os.environ.get("GLM53_TP_RINGS", "140,105,128,280").split(","))


def _prompt(L, W, attn, n: int):
    """hn / latents / indexer rows of the synthetic prompt (n rows)"""
    gd = golden(LAYER, 2048)
    hn = G.mhc_pre_ref(gd["streams_in"].double().reshape(2048, -1), L.a_args)["hn"]
    if n > hn.shape[0]:
        gen = torch.Generator().manual_seed(7)
        reps = [hn]
        while sum(r.shape[0] for r in reps) < n:
            m = min(2048, n - sum(r.shape[0] for r in reps))
            reps.append(hn[:m] * (1 + 0.05 * torch.randn(m, hn.shape[1], generator=gen, dtype=torch.float64)))
        hn = torch.cat(reps)
    hn = hn[:n]
    return gd, hn, G.latents(hn, W, attn.eps, attn.q_lora)


class Case:
    """one decode call's device state: slots at `pos`, the cache / page table / DSA state prefilled"""

    def __init__(self, mesh, L, ix, pos, lat, ir, xs, *, dsa: bool):
        attn = L.attn
        t = len(pos)
        bs = attn.block_size
        cap = -(-(max(pos) + 1) // bs) * bs
        self.bps = cap // bs
        n = t * self.bps
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(11 + t))
        self.pt_host = perm.reshape(t, self.bps).to(torch.int32)
        self.pt = ttnn.from_torch(self.pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        ch = torch.zeros(n, 1, bs, attn.kv_lora)
        for i, p in enumerate(pos):
            for r in range(p):
                ch[int(self.pt_host[i, r // bs]), 0, r % bs] = lat[r].float()
        self.cache = ttnn.from_torch(ch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        self.cur = int_tensor(pos, mesh)
        self.x = ttnn.from_torch(torch.stack(xs).float().reshape(1, 1, t, -1), dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        self.st, self.ref_state = None, []
        if dsa:
            tp = MK.DsaIndexer.pool_rows(cap)
            pools = torch.zeros(t, 1, tp, 128, dtype=torch.float64)
            ring = torch.zeros(t, 1, 32, 256, dtype=torch.float64)
            for i, p in enumerate(pos):
                pl, rg = G.dsa_state(p, ir["k"], ir["g"], ix.host["ape"], tp)
                pools[i, 0], ring[i, 0] = pl, rg
                self.ref_state.append((pl, rg))
            self.st = ix.state(mesh, t, cap, pools=pools, ring=ring)
        self.t, self.pos = t, pos

    def cache_rows(self, i: int, rows):
        """latent rows (device cache after the step) of slot i"""
        c = host(self.cache)
        bs = c.shape[2]
        return torch.stack([c[int(self.pt_host[i, r // bs]), 0, r % bs] for r in rows.tolist()]).double()


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = (ttnn.open_device(device_id=0, worker_l1_size=wl1) if os.environ.get("GLM53_MOCK") == "1" else
                ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, worker_l1_size=wl1))  # fmt: skip
    rows = Rows(LAYER)
    tr = Traced(mesh)
    knobs = dict(rev=MK.MLA_REV, pos=POS)
    try:
        L = MlaLayer(mesh, ctx, LAYER)
        attn = L.attn
        W = G.host_weights(attn)
        ix = MK.DsaIndexer(mesh, L.source, LAYER)
        gd, hn_all, lat_all = _prompt(L, W, attn, max(POS) + 1)
        ir = G.indexer_rows(hn_all, ix.host)
        g_in = L.g.reshape(-1, 16384).double()
        xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, len(POS))]
        sc = dict(eps=attn.eps, scale=attn.scale, heads=attn.h, q_lora=attn.q_lora)
        mk = MK.MkMla(mesh, L.a_args, attn, dsa=ix)
        c = Case(mesh, L, ix, POS, lat_all, ir, xs, dsa=True)
        fn = lambda: mk(c.x, c.cache, c.pt, c.cur, t=c.t, dsa_state=c.st)
        tr.warm(fn)
        out = tr.run(fn)
        got = {k: host(out[k]) for k in ("o_lat", "a", "s1", "q_lat", "q_ix", "wsc", "latent")}
        us = timed(tr, REPLAYS)
        tr.release()
        out = None  # the call's L1 outputs (~8 MB at T = 4) freed before the dense-only programs run
        pools_d, ring_d = host(c.st["pools"]), host(c.st["ring"])
        rows("mk_mla_dsa_timing", t=c.t, execution="traced", timing_us={"mk_one_dsa": us}, variant="mk_mla_dsa",
             status="info", **knobs)  # fmt: skip
        for i, p in enumerate(POS):
            pl, rg = c.ref_state[i]
            ref = G.dsa_step(xs[i].reshape(1, -1), p, L.a_args, W, ix.host, lat_all[:p], pl, rg, **sc)
            r_ = dict(t=c.t, slot=i, p=p, sparse=p + 1 > MK.DSA_DENSE_MAX, **knobs)
            ring_row, pooled = ring_d[i, 0, p % 4], pools_d[i, 0, p // 4]
            r_["ring_bitwise_share"] = float((ring_row.double() == ref["ring_row"]).double().mean())
            r_["ring_pcc"] = pcc(ring_row, ref["ring_row"].float())
            r_["pooled_bitwise_share"] = float((pooled.double() == ref["pooled"]).double().mean())
            r_["pooled_pcc"] = pcc(pooled, ref["pooled"].float())
            r_["status"] = "pass" if min(r_["ring_pcc"], r_["pooled_pcc"]) >= 0.999 else "fail"
            rows("dsa_index_core", **r_)
            r_ = dict(t=c.t, slot=i, p=p, sparse=p + 1 > MK.DSA_DENSE_MAX, program="mk_mla_one_dsa", **knobs)
            # s1 row i: t <= 4 stacked [t,1,4,4096]; t > 4 flat [1,1,t,16384] (stream-major columns: the same [4, 4096])
            s1_i = got["s1"][i, 0] if c.t <= 4 else got["s1"][0, 0, i].reshape(4, -1)
            dev = dict(o_lat=got["o_lat"][0, i], a=got["a"][0, 0, i, : ref["a"].shape[-1]], s1=s1_i)
            for k, v in dev.items():
                a_, b_ = v.reshape(-1), ref[k].reshape(-1)[: v.numel()].float()
                r_[f"{k}_pcc_vs_fp32"] = pcc(a_, b_)
                r_[f"{k}_maxabs_vs_fp32"] = float((a_ - b_).abs().max())
            r_["status"] = "pass" if min(r_[f"{k}_pcc_vs_fp32"] for k in dev) >= BAR else "fail"
            rows("golden_fp32_dsa", **r_)
            if p + 1 > MK.DSA_DENSE_MAX:  # the select / attention emulated on the device's own intermediates
                n_closed = (p + 1) // 4
                qi = got["q_ix"][0, i].double()  # [32, 128]
                w = got["wsc"][0, 0, i, :32].double()
                s_ = torch.relu(qi @ pools_d[i, 0, :n_closed].double().T)
                scores = w @ s_
                sel = torch.topk(scores, MK.DSA_TOPK).indices.sort().values
                sel_rows = torch.cat([(4 * sel[:, None] + torch.arange(4)).reshape(-1),
                                      torch.arange(4 * n_closed, p + 1)])  # fmt: skip
                kv = c.cache_rows(i, sel_rows)
                ql = got["q_lat"][0, i].double()
                o_em = torch.softmax((ql @ kv.T) * attn.scale, dim=-1) @ kv
                ov = len(set(sel.tolist()) & set(ref["sel"].tolist())) / MK.DSA_TOPK
                r_ = dict(t=c.t, slot=i, p=p, program="mk_mla_one_dsa", sel_overlap_vs_fp32=ov,
                          scores_pcc_vs_fp32=pcc(scores.float(), ref["scores"].float()),
                          o_lat_pcc_vs_emulation=pcc(got["o_lat"][0, i].reshape(-1), o_em.reshape(-1).float()),
                          o_lat_maxabs_vs_emulation=float((got["o_lat"][0, i].double() - o_em).abs().max()),
                          **knobs)  # fmt: skip
                r_["status"] = "pass" if r_["o_lat_pcc_vs_emulation"] >= BAR_DEV else "fail"
                rows("dsa_select_emulation", **r_)
        if TP:  # the same state: every step write is idempotent, so the shards see the full program's inputs
            hr = attn.h // TP
            a_sum, o_pcc, us_r0 = None, [], None
            for r in range(TP):
                mr = MK.MkMla(mesh, L.a_args, MK.HeadShard(attn, r, TP), dsa=ix, kb=KB_SHARD, rings=RINGS_SHARD)
                fr = lambda mr=mr: mr(c.x, c.cache, c.pt, c.cur, t=c.t, dsa_state=c.st, keep=("a", "o_lat"))
                tr.warm(fr)
                o_ = tr.run(fr)
                a_r, ol_r = host(o_["a"]), host(o_["o_lat"])
                if r == 0:
                    us_r0 = timed(tr, REPLAYS)
                tr.release()
                o_ = None
                a_sum = a_r if a_sum is None else a_sum + a_r
                full_o = got["o_lat"].reshape(1, c.t, attn.h, -1)[:, :, r * hr : (r + 1) * hr]
                o_pcc.append(pcc(ol_r.reshape(1, c.t, -1, full_o.shape[-1])[:, :, :hr].reshape(-1), full_o.reshape(-1)))
            r_ = dict(t=c.t, tp=TP, execution=tr.mode, timing_us={"mk_one_dsa_tp_r0": us_r0}, kb_shard=list(KB_SHARD),
                      rings_shard=list(RINGS_SHARD), **knobs)  # fmt: skip
            r_["a_sum_pcc"] = pcc(a_sum.reshape(-1), got["a"].reshape(-1))
            r_["o_lat_pcc_min"] = min(o_pcc)
            r_["status"] = "pass" if r_["a_sum_pcc"] >= BAR_DEV and r_["o_lat_pcc_min"] >= 0.999 else "fail"
            rows("dsa_tp_emulation", **r_)
        if DENSE:  # dense-only positions: MkMla(dsa) vs MkMla on the same state, bitwise, + timing of both
            pos_d = [min(p, 2050) - 13 * i for i, p in enumerate(POS)]
            mk0 = MK.MkMla(mesh, L.a_args, attn)
            res, times = {}, {}
            for name, use in (("mk_one", False), ("mk_one_dsa", True)):
                cd = Case(mesh, L, ix, pos_d, lat_all, ir, xs, dsa=use)
                m_ = mk if use else mk0
                f_ = (lambda m_=m_, cd=cd: m_(cd.x, cd.cache, cd.pt, cd.cur, t=cd.t, dsa_state=cd.st)) if use else (
                    lambda m_=m_, cd=cd: m_(cd.x, cd.cache, cd.pt, cd.cur, t=cd.t))  # fmt: skip
                tr.warm(f_)
                o_ = tr.run(f_)
                res[name] = {k: host(o_[k]) for k in ("o_lat", "s1", "q_lat", "latent")}
                times[name] = timed(tr, REPLAYS)
                tr.release()
                o_ = cd = None
            r_ = dict(t=len(pos_d), pos_dense=pos_d, **knobs)
            for k in res["mk_one"]:
                r_[f"{k}_bitwise"] = bool(torch.equal(res["mk_one"][k], res["mk_one_dsa"][k]))
                r_[f"{k}_pcc"] = pcc(res["mk_one"][k].reshape(-1), res["mk_one_dsa"][k].reshape(-1))
            r_["status"] = "pass" if all(r_[f"{k}_bitwise"] for k in res["mk_one"]) else "fail"
            rows("dsa_dense_vs_mk_one", **r_)
            rows("mk_mla_dsa_dense_timing", t=len(pos_d), execution="traced", timing_us=times, variant="mk_mla_dsa",
                 status="info", pos_dense=pos_d, rev=MK.MLA_REV)  # fmt: skip
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rows("dsa_mk_mla", status="fail", error=traceback.format_exc()[-800:], **knobs)
    finally:
        rows.flush()
        if own:
            ttnn.close_device(mesh) if os.environ.get("GLM53_MOCK") == "1" else ttnn.close_mesh_device(mesh)
    bad = rows.failed()
    print(f"DEV_MK_DSA_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
