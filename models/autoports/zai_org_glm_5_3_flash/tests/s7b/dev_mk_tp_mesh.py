# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b TP on the fabric mesh (tt/tp_mla.py TPMla: MkMla per chip on its head shard + Stage 14's in-program fabric
all-reduce of o_proj's partial) vs the one-chip program replicated on every chip (MkMla with all 64 heads, the same
inputs and starting caches), layer GLM53_S7B_LAYER (default 3), T = GLM53_MK_TS (default 1,2,4), both traced:
  * s1 (the stacked residual, after mhc_post on the all-reduced a): TP vs the full program on every chip (PCC, bar
    0.9999), and BITWISE identical on every chip (a replicated residual must not drift between chips);
  * the latent cache rows the step writes: TP vs the full program bitwise, every chip;
  * o_lat of chip r (its heads) vs the float64 attention on the chip's own q_lat and cache (bar 0.9999), and vs the full
    program's rows of those heads (PCC, info);
  * traced time of the TP step and of the replicated full step (rows -> perf.jsonl).
Runs in a fabric-mesh server (serve.py --mgd 2x4) or standalone (GLM53_MOCK=1: the mock 8-chip cluster)."""

from __future__ import annotations

import os
import sys
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import Rows, compare, layer_of, resident, timed
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,2,4").split(",")]
REPLAYS = int(os.environ.get("GLM53_MK_REPLAYS", "50"))
FULL = os.environ.get("GLM53_TP_FULL", "1") == "1"
# GLM53_TP_VTP=8 on a pair: the TP8 per-chip shard (8 heads per chip; MeshHeadShard vtp) -- timing of the production
# geometry only (2 of 8 partials are all-reduced: the numerics checks are skipped, as on the mock)
VTP = int(os.environ.get("GLM53_TP_VTP", "0")) or None
POS_LAST = int(os.environ.get("GLM53_MK_LAST", "0")) or None  # positions LAST - STEP * i (default: the case's)
POS_STEP = int(os.environ.get("GLM53_MK_STEP", "13"))  # the replicated one-chip program as the reference
BAR = 0.9999
# the head shard's reader phases (8 heads per chip): K blocks / ring budgets of dev_mk_tp8.py (the full program's
# defaults overflow a reader's L1 with the shard's geometry)
KB_SHARD = tuple(int(v) for v in os.environ.get("GLM53_TP_KB", "8,16,8,16,16").split(","))
RINGS_SHARD = tuple(int(v) for v in os.environ.get("GLM53_TP_RINGS", "140,105,128,280").split(","))


def per_chip(t) -> list:
    """host copies of a mesh tensor, one per chip (mesh device order)"""
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def written(c, cache_host):
    """the cache rows the step writes (cur_pos through the page table) -> [T, 512]"""
    bs = cache_host.shape[2]
    pt = torch.as_tensor(c["page_table_host"])
    return torch.stack([cache_host[int(pt[i, p // bs]), 0, p % bs] for i, p in enumerate(c["pos"])])


def o_lat_f64(c, cache_host, q_lat, scale: float):
    """float64 attention of q_lat [1, T, H, 512] over cache_host (rows 0..p of row t's sequence via the page table)"""
    bs = cache_host.shape[2]
    pt = torch.as_tensor(c["page_table_host"])
    out = []
    for t_, p in enumerate(c["pos"]):
        r = torch.arange(p + 1)
        kv = cache_host[pt[t_, r // bs].long(), 0, r % bs].double()
        out.append(torch.softmax((q_lat[0, t_].double() @ kv.T) * scale, dim=-1) @ kv)
    return torch.stack(out)[None]


def check(t, ct, cf, got, *, heads: int, ndev: int, scale: float, cache_tp: list, cache_f: list | None, w_uv=None,
          **knobs):
    """the TP step's acceptance row from host copies: got[name][key] = per-chip tensors ("tp", optionally "full")"""
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import compare

    hr = heads // ndev
    r_ = dict(t=t, **knobs)
    s1 = got["tp"]["s1"]
    r_["s1_bitwise_across_chips"] = all(torch.equal(s1[0], s) for s in s1[1:])
    w_tp = [written(ct, ch) for ch in cache_tp]
    r_["cache_rows_bitwise_across_chips"] = all(torch.equal(w_tp[0], w) for w in w_tp[1:])
    # independent of the reference program (which shares the latent core's code): the rows the step wrote into the
    # cache ARE the program's latent rows, bitwise, on every chip
    lat = got["tp"]["latent"]
    r_["cache_rows_eq_latent"] = all(torch.equal(w_tp[d], lat[d].reshape(-1, lat[d].shape[-1])[:t]) for d in range(ndev))
    f64, f64_rel = [], []
    for d in range(ndev):
        em = o_lat_f64(ct, cache_tp[d], got["tp"]["q_lat"][d], scale)
        rr = {}
        compare(rr, {"o": got["tp"]["o_lat"][d][:, :t, :hr]}, {"o": em[:, :, :hr].float()})
        f64.append(rr["o_pcc"])
        f64_rel.append(rr["o_rel_l2"])
    r_["o_lat_vs_f64_pcc_min"] = min(f64)
    r_["o_lat_vs_f64_rel_l2_max"] = max(f64_rel)  # PCC is scale-blind: the relative L2 bounds a scale error too
    ok = (r_["s1_bitwise_across_chips"] and r_["cache_rows_bitwise_across_chips"] and r_["cache_rows_eq_latent"]
          and min(f64) >= BAR
          and max(f64_rel) <= 0.01)  # fmt: skip
    if w_uv is not None and "o" in got["tp"]:
        # W_uv on its own (independent of the reference program, which runs the same W_uv path): o rows of chip d's
        # heads vs the host product of the chip's own o_lat rows (bf16 values) and its W_uv slice [hr, 512, dv]
        o_pcc, o_rel = [], []
        for d in range(ndev):
            ol = got["tp"]["o_lat"][d][0, :t, :hr].double()  # [t, hr, 512]
            ref = torch.einsum("thk,hkv->thv", ol, w_uv[d].double()).reshape(t, -1)
            rr = {}
            compare(rr, {"o": got["tp"]["o"][d].reshape(-1, ref.shape[-1])[:t]}, {"o": ref.float()})
            o_pcc.append(rr["o_pcc"])
            o_rel.append(rr["o_rel_l2"])
        r_["o_vs_host_w_uv_pcc_min"], r_["o_vs_host_w_uv_rel_l2_max"] = min(o_pcc), max(o_rel)
        ok = ok and min(o_pcc) >= BAR and max(o_rel) <= 0.01
    if cache_f is not None:
        pcc_s1, pcc_o, cache_eq, rel_o = [], [], [], []
        for d in range(ndev):
            rr = {}
            compare(rr, {"s1": s1[d]}, {"s1": got["full"]["s1"][d]})
            o_f = got["full"]["o_lat"][d].reshape(1, t, heads, -1)[:, :, d * hr : (d + 1) * hr]
            compare(rr, {"o": got["tp"]["o_lat"][d][:, :t, :hr]}, {"o": o_f})
            pcc_s1.append(rr["s1_pcc"])
            pcc_o.append(rr["o_pcc"])
            rel_o.append(rr["o_rel_l2"])
            cache_eq.append(torch.equal(w_tp[d], written(cf, cache_f[d])))
        r_.update(s1_pcc_vs_full_min=min(pcc_s1), o_lat_pcc_vs_full_min=min(pcc_o), o_lat_rel_l2_vs_full_max=max(rel_o),
                  cache_rows_bitwise_vs_full=all(cache_eq),
                  full_s1_bitwise_across_chips=all(torch.equal(got["full"]["s1"][0], s) for s in got["full"]["s1"][1:]))
        ok = ok and min(pcc_s1) >= BAR and all(cache_eq) and max(rel_o) <= 0.02
    r_["status"] = "pass" if ok else "fail"
    return r_


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:  # standalone: the mock 8-chip cluster (TT_METAL_MOCK_CLUSTER_DESC_PATH) or real hardware with fabric
        os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mesh",
            "glm8_2x4_mesh_graph_descriptor.textproto"))  # fmt: skip
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4), trace_region_size=200_000_000, worker_l1_size=wl1)
    rows = Rows(LAYER)
    tr = Traced(mesh)
    ndev = int(mesh.get_num_devices())
    knobs = dict(rev=MK.MLA_REV, mesh=list(mesh.shape), tp=ndev, vtp=VTP, mk_readers=os.environ.get("GLM53_MK_READERS"),
                 kb_shard=list(KB_SHARD), rings_shard=list(RINGS_SHARD))  # fmt: skip
    try:
        L = layer_of(mesh, ctx, LAYER)
        attn = L.attn
        ts = tuple(sorted(set(TS)))  # one FabricPhase per row count (t <= 32: Stage 14 native t)
        tp = resident(ctx, ("tp_mla", LAYER, ndev, KB_SHARD, RINGS_SHARD, ts, VTP),
                      lambda: TPM.TPMla(mesh, L.a_args, attn, kb=KB_SHARD, rings=RINGS_SHARD, ts=ts, vtp=VTP))  # fmt: skip
        full = (resident(ctx, ("mk_one_mesh", LAYER, ndev), lambda: MK.MkMla(mesh, L.a_args, attn))
                if FULL and not VTP else None)  # fmt: skip
        hr = attn.h // ndev
        for t in TS:
            try:
                pk = {} if POS_LAST is None else dict(last=POS_LAST, step=POS_STEP)
                sfx = "" if POS_LAST is None else f"@{POS_LAST}/{POS_STEP}"
                ct, cf = L.case(t, tag="tpm" + sfx, **pk), L.case(t, tag="tpm_full" + sfx, **pk)
                keys = ("s1", "o_lat", "q_lat", "latent", "o")
                f_tp = lambda c=ct, t=t: {k: v for k, v in tp(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t).items()
                                          if k in keys}  # fmt: skip
                f_full = lambda c=cf, t=t: {k: v for k, v in full(c["x"], c["cache"].latent, c["pt"], c["cur"],
                                                                     t=t).items() if k in keys}  # fmt: skip
                fns = [("tp", f_tp)] + ([("full", f_full)] if full is not None else [])
                tr.warm(*[f for _, f in fns])
                got, us = {}, {}
                for name, fn in fns:
                    out = tr.run(fn)
                    got[name] = {k: per_chip(v) for k, v in out.items()}
                    us[f"{name}_step"] = timed(tr, REPLAYS)
                    tr.release()
                rows("tp_mla_mesh_timing", t=t, execution=tr.mode, status="info", timing_us=us,
                     variant="tp_mla_mesh", ot=tp.mk.ot(t), **knobs)  # fmt: skip
                if os.environ.get("GLM53_MOCK") == "1" or VTP:
                    continue
                r_ = check(t, ct, cf, got, heads=attn.h, ndev=ndev, scale=attn.scale,
                           cache_tp=per_chip(ct["cache"].latent),
                           cache_f=per_chip(cf["cache"].latent) if full is not None else None,
                           w_uv=[w.float() for w in tp.shard.host_weights["w_uv"]], **knobs)  # fmt: skip
                rows("tp_mla_mesh_vs_full", ot=tp.mk.ot(t), **r_)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rows("tp_mla_mesh_case", t=t, status="fail", error=traceback.format_exc()[-800:], **knobs)
                tr.release()
                if "timeout" in str(e).lower():
                    raise
    finally:
        rows.flush()
        if own:
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    bad = rows.failed()
    print(f"DEV_MK_TP_MESH_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
