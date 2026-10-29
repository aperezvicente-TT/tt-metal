# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b: the sparse_dsa attention half of layer GLM53_S7B_LAYER (default 3) as megakernel programs (tt/mk_mla.py):
MkMlaA (mhc_pre -> q_a|kv_a -> norms -> latent cache write -> q_b -> W_uk) -> the flash-MLA op -> MkMlaB (W_uv ->
o_proj -> mhc_post), device vs the op chain they replace, traced (every case warmed before any capture), T =
GLM53_MK_TS (default 1,2,4):
  * program A vs the chain's first half on identical inputs and identical starting caches: hn / x_out / coef (mhc_pre:
    bitwise), qk, latent, q, q_lat, the cache after the step (rows written: PCC; every other row bitwise);
  * program B in isolation (the same o_lat / x_out / coef as the chain's): a, s1;
  * the attention half end to end: s1 (and a, o_lat, q_lat) vs the chain's;
  * traced wall clock: A, the flash-MLA op, B, the half (A + flash + B), the chain half; the layer with 7a's MkMoE
    (alone and two layers chained in one trace = steady state per layer);
  * GLM53_MK_PROBE=1: per-phase in-program cycle probes of A and B (kernels/mk/probe.hpp marks).
Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import os
import sys
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import (
    Rows,
    aiclk_mhz,
    compare,
    host,
    layer_of,
    resident,
    timed,
)
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,2,4").split(",")]
PROBE = os.environ.get("GLM53_MK_PROBE") == "1"
REPLAYS = int(os.environ.get("GLM53_MK_REPLAYS", "50"))
FLASH_HEADS = int(os.environ.get("GLM53_MLA_FLASH_HEADS", "0"))  # 0 = the decoder's rule (32 per call below 16 rows)
WITH_LAYER = os.environ.get("GLM53_MK_LAYER", "1") == "1"
SOAK_S = float(os.environ.get("GLM53_MK_SOAK_S", "0"))  # > 0: back-to-back replays of the mk half for this long
ONE = os.environ.get("GLM53_MK_ONE", "1") == "1"  # the one-program half (tt/mk_mla.py MkMla)
# 7a's MkMoE has an open OOB read at T=1 (program 22:32Z): the layer cases run at these T only
LAYER_TS = [int(v) for v in os.environ.get("GLM53_MK_LAYER_TS", "2,4").split(",") if v]
TWO = os.environ.get("GLM53_MK_TWO", "1") == "1"  # the two-program variants (A + flash op + B, A + B-with-attention)
# positions of the T rows: LAST - STEP * i (default the case's 2049 - 13 i); early / chunk-edge positions cover the
# attention's empty-chunk partials and the last partial tile
POS_LAST = int(os.environ.get("GLM53_MK_LAST", "0")) or None
POS_STEP = int(os.environ.get("GLM53_MK_STEP", "13"))


class Hung(Exception):
    """a device TIMEOUT: the command queue is not trusted any more -> end the job, main returns 3 (serve.py closes)"""


def _check_hang(e):
    if "TIMEOUT" in str(e) or "hang" in str(e):
        raise Hung(str(e)[:300]) from e


def _l1_reject(e) -> bool:
    """a configuration that does not fit L1 next to the live tensors / the kernel-config ring (recorded, not a failure)"""
    return any(k in str(e) for k in ("clash with L1 buffers", "beyond max L1 size", "grow to"))


# the op-chain reference programs: an L1 reject of one of THEM is a limit of the reference next to the case's tensors
# (info); a reject of a megakernel program, or a case left without a comparison, fails its row
REFERENCE = ("chain_a", "chain_b", "flash", "chain_att", "chain_att_mkmoe", "chain_att_mkmoe_x2")


def _info_reject(name: str, e) -> bool:
    return name in REFERENCE and _l1_reject(e)


def chain_a(L, c):
    """the chain's first half with its intermediates (OptimizedMLA._project_decode + the cache write of decode)"""
    attn, t = L.attn, c["t"]
    r = L.mhc_pre(c)
    qk = attn._lin("ds_qa_kva", r.hn, attn.w_qa_kva)
    q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, attn.q_lora))
    cp = ttnn.slice(qk, (0, 0, 0, attn.q_lora), (1, 1, t, attn.q_lora + attn.kv_lora))
    q_r = ttnn.rms_norm(q_r, weight=attn.qa_norm, epsilon=attn.eps, compute_kernel_config=attn.ckc_fp32)
    latent = ttnn.rms_norm(cp, weight=attn.kva_norm, epsilon=attn.eps, compute_kernel_config=attn.ckc_fp32)
    q = attn._lin("ds_qb", q_r, attn.w_qb)
    qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=attn.h, num_kv_heads=0, transpose_k_heads=False,
                                                       memory_config=attn.mem)  # fmt: skip
    q_lat = ttnn.permute(attn._head_matmul(qh, attn.w_uk), (0, 2, 1, 3))  # [1,t,64,512]
    g = attn.device.compute_with_storage_grid_size()
    shard = ttnn.create_sharded_memory_config(shape=(ttnn.TILE_SIZE, attn.kv_lora),
                                              core_grid=ttnn.num_cores_to_corerangeset(t, g, row_wise=True),
                                              strategy=ttnn.ShardStrategy.HEIGHT,
                                              use_height_and_width_as_shard_shape=True)  # fmt: skip
    lat = ttnn.to_memory_config(ttnn.reshape(latent, (1, t, 1, attn.kv_lora)), shard)
    ttnn.experimental.paged_update_cache(c["cache"].latent, lat, update_idxs_tensor=c["cur"], page_table=c["pt"])
    return dict(hn=r.hn, x_out=r.x_out, coef=r.coef, qk=qk, latent=latent, q=q, q_lat=q_lat)


def chain_b(L, c, o_lat, x_out, coef):
    """the chain's second half: W_uv + concat + o_proj (OptimizedMLA._output_decode) + mhc_post"""
    attn = L.attn
    a = attn._output_decode(ttnn.permute(o_lat, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG))
    s1 = L.mpost(a, x_out, coef, t=c["t"], x_mode=MK._mpost.X_STACKED, out_mode=MK._mpost.OUT_STACKED)
    return dict(a=a, s1=s1)


def written_rows(c, cache_host):
    """bool mask [num_blocks, 1, 64, 1] of the cache rows the step writes (cur_pos through the page table)"""
    m = torch.zeros(cache_host.shape[0], 1, cache_host.shape[2], 1, dtype=torch.bool)
    bs = cache_host.shape[2]
    for t_, p in enumerate(c["pos"]):
        if p >= 0:
            m[int(c["page_table_host"][t_, p // bs]), 0, p % bs, 0] = True
    return m


def o_lat_f64(c, q_lat, scale: float):
    """float64 attention of a program's own q_lat [1, T, H, 512] over its own cache after the step (rows 0..p of row
    t's sequence through the page table): the acceptance of the in-program attention at any position (bar 0.9999)"""
    ch = host(c["cache"].latent)
    bs = ch.shape[2]
    out = []
    for t_, p in enumerate(c["pos"]):
        r = torch.arange(p + 1)
        kv = ch[torch.as_tensor(c["page_table_host"])[t_, r // bs].long(), 0, r % bs].double()
        out.append(torch.softmax((q_lat[0, t_].double() @ kv.T) * scale, dim=-1) @ kv)
    return torch.stack(out)[None]


def _vs_golden(r: dict, keys):
    """programs with the in-program attention phase: the fp32 golden showed it is MORE accurate than the flash-MLA op
    (a vs fp32: one-program 0.999983, op chain 0.999344), so their acceptance is vs the fp32 golden (dev_mk_golden.py,
    PCC >= 0.9999; program ruling 2026-10-19T23:39:08Z) and this op-chain comparison is informational unless a
    value is outright wrong (PCC < 0.999)"""
    worst = min(r[f"{k}_pcc"] for k in keys)
    r["acceptance"] = "fp32_golden"
    r["status"] = "info" if worst >= 0.999 else "fail"


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:  # standalone / mock (GLM53_MOCK=1): the serve processes' worker_l1_size (GLM53_WORKER_L1, default 1264512)
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = (ttnn.open_device(device_id=0, worker_l1_size=wl1) if os.environ.get("GLM53_MOCK") == "1" else
                ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, worker_l1_size=wl1))  # fmt: skip
    rows = Rows(LAYER)
    tr = Traced(mesh)
    knobs = dict(rpb=os.environ.get("GLM53_MLA_RPB", "4"), rings_a=os.environ.get("GLM53_MLA_RINGS_A"),
                 rings_one=os.environ.get("GLM53_MLA_RINGS"), kb_one=os.environ.get("GLM53_MLA_KB"),
                 mk_readers=os.environ.get("GLM53_MK_READERS"),
                 rings_b=os.environ.get("GLM53_MLA_RINGS_B"), flash_grid=os.environ.get("GLM53_MLA_GRID", "8x4"),
                 kchunk=os.environ.get("GLM53_MLA_KCHUNK", "32"), rev=MK.MLA_REV,
                 defer=os.environ.get("GLM53_MLA_DEFER", "0"), ukv_bf8=os.environ.get("GLM53_MLA_UKV_BF8", "0"),
                 att_ob=os.environ.get("GLM53_MLA_ATT_OB", "4"), placement=os.environ.get("GLM53_MLA_PLACEMENT"),
                 pos_last=POS_LAST, pos_step=POS_STEP if POS_LAST else None)  # fmt: skip
    try:
        # resident state (JOB PIPELINE RULE 2026-10-19T23:52:53Z): the layer and every program object (with its
        # re-laid-out weights) live in CTX across jobs, keyed by their knobs and the code version
        L = layer_of(mesh, ctx, LAYER)
        attn = L.attn
        env = lambda *ks: tuple(os.environ.get(k) for k in ks)
        ka = env("GLM53_MLA_RINGS_A", "GLM53_MLA_RPB")
        kb_ = env("GLM53_MLA_RINGS_B", "GLM53_MLA_RPB")
        # every knob MkMla reads at construction (a resident object built under other values must not be reused)
        k1 = env("GLM53_MLA_RINGS", "GLM53_MLA_KB", "GLM53_MLA_LOOKAHEAD", "GLM53_MLA_QN", "GLM53_MLA_RPB",
                 "GLM53_MLA_QFOLD", "GLM53_MLA_DEFER", "GLM53_MLA_UKV_BF8", "GLM53_MLA_PLACEMENT", "GLM53_MLA_KORDER")
        mka = resident(ctx, ("MkMlaA", LAYER, ka), lambda: MK.MkMlaA(mesh, L.a_args, attn))
        mkb = resident(ctx, ("MkMlaB", LAYER, kb_), lambda: MK.MkMlaB(mesh, attn))
        mkb2 = resident(ctx, ("MkMlaB2", LAYER, kb_), lambda: MK.MkMlaB(mesh, attn, attention=True))
        kb1 = tuple(int(v) for v in os.environ["GLM53_MLA_KB"].split(",")) if os.environ.get("GLM53_MLA_KB") else None
        kw1 = dict(kb=kb1) if kb1 else {}
        mk1 = resident(ctx, ("MkMla", LAYER, k1), lambda: MK.MkMla(mesh, L.a_args, attn, **kw1)) if ONE else None
        mk1_p = (resident(ctx, ("MkMlaP", LAYER, k1), lambda: MK.MkMla(mesh, L.a_args, attn, probe=True, **kw1))
                 if ONE and PROBE else None)  # fmt: skip
        mka_p = (
            resident(ctx, ("MkMlaAP", LAYER, ka), lambda: MK.MkMlaA(mesh, L.a_args, attn, probe=True))
            if PROBE
            else None
        )
        mkb_p = resident(ctx, ("MkMlaBP", LAYER, kb_), lambda: MK.MkMlaB(mesh, attn, probe=True)) if PROBE else None
        mkmoe = None
        if WITH_LAYER:
            from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE

            # NOT resident: 7a's MkMoE keeps its per-T shared-expert intermediate in L1 (`_persistent`), which would
            # lower every later program's L1 ceiling in this process (a CB clash in the later jobs)
            mkmoe = MkMoE(mesh, L.f_args, L.mlp)
        heads = lambda t: FLASH_HEADS or (attn.h if t >= attn.fcfg.mla_single_call_min_batch else attn.decode_heads)
        cases = {}
        for t in TS:
            pk = {} if POS_LAST is None else dict(last=POS_LAST, step=POS_STEP)
            sfx = "" if POS_LAST is None else f"@{POS_LAST}/{POS_STEP}"
            cm, cc = L.case(t, tag="mk" + sfx, **pk), L.case(t, tag="chain" + sfx, **pk)
            # fixed inputs of program B in isolation: the chain's own o_lat / x_out / coef (eager, before any capture)
            ra0 = chain_a(L, cc)
            o_fix = L.flash(cc, ra0["q_lat"], heads=heads(t))
            # in DRAM: the per-T fixed inputs live through the whole job (every case's), and L1 copies of them lowered
            # the next programs' L1 ceiling (7a's MkMoE tail cores ended 2.5 KB short)
            fix = dict(o_lat=ttnn.clone(o_fix, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                       x_out=ttnn.clone(ra0["x_out"], memory_config=ttnn.DRAM_MEMORY_CONFIG),
                       coef=ttnn.clone(ra0["coef"], memory_config=ttnn.DRAM_MEMORY_CONFIG),
                       q_lat=ttnn.clone(ra0["q_lat"], memory_config=ttnn.DRAM_MEMORY_CONFIG))  # fmt: skip
            ra0 = o_fix = None  # the chain's L1 outputs: not kept for the whole job

            def mk_att(c=cm, t=t, a_=mka, b_=mkb):
                ra = a_(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t)
                o_lat = L.flash(c, ra["q_lat"], heads=heads(t))
                rb = b_(o_lat, ra["x_out"], ra["coef"], t=t)
                return dict(s1=rb["s1"], a=rb["a"], o_lat=o_lat, q_lat=ra["q_lat"])

            def chain_att(c=cc, t=t):
                ra = chain_a(L, c)
                o_lat = L.flash(c, ra["q_lat"], heads=heads(t))
                rb = chain_b(L, c, o_lat, ra["x_out"], ra["coef"])
                return dict(s1=rb["s1"], a=rb["a"], o_lat=o_lat, q_lat=ra["q_lat"])

            def mk_att2(c=cm, t=t, a_=mka, b_=mkb2):
                ra = a_(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t)
                rb = b_(None, ra["x_out"], ra["coef"], t=t, cache=c["cache"].latent, page_table=c["pt"],
                        cur_pos=c["cur"], q_lat=ra["q_lat"])  # fmt: skip
                return dict(s1=rb["s1"], a=rb["a"], o_lat=rb["o_lat"], q_lat=ra["q_lat"])

            fns = dict(
                mk_a=lambda c=cm, t=t: mka(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t),
                chain_a=lambda c=cc: chain_a(L, c),
                mk_b=lambda t=t, f=fix: mkb(f["o_lat"], f["x_out"], f["coef"], t=t),
                chain_b=lambda c=cc, f=fix: chain_b(L, c, f["o_lat"], f["x_out"], f["coef"]),
                flash=lambda c=cc, q=fix["q_lat"], t=t: L.flash(c, q, heads=heads(t)),
                mk_att=mk_att,
                chain_att=chain_att,
                # program B with the attention phase on the chain's q_lat (isolates the attention phase + B)
                mk_b2=lambda c=cc, t=t, f=fix: mkb2(
                    None,
                    f["x_out"],
                    f["coef"],
                    t=t,
                    cache=c["cache"].latent,
                    page_table=c["pt"],
                    cur_pos=c["cur"],
                    q_lat=f["q_lat"],
                ),
                mk_att2=mk_att2,
            )
            if mk1 is not None:
                fns["mk_one"] = lambda c=cm, t=t: {
                    k: v
                    for k, v in mk1(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t).items()
                    if k in ("s1", "a", "o_lat", "q_lat", "latent", "hn", "x_out")
                }
            if mk1_p is not None:
                fns["probe_one"] = lambda c=cm, t=t: mk1_p(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t)["s1"]
            if not TWO:
                for k_ in ("mk_a", "mk_b", "mk_att", "mk_b2", "mk_att2"):
                    fns.pop(k_, None)
            if mkmoe is not None and t in LAYER_TS:
                if TWO:
                    fns["mk_layer"] = lambda f=mk_att, t=t: mkmoe(f()["s1"], t=t)["out"]
                    fns["mk_layer2"] = lambda f=mk_att2, t=t: mkmoe(f()["s1"], t=t)["out"]
                if mk1 is not None:
                    fns["mk_layer_one"] = lambda c=cm, t=t: mkmoe(
                        mk1(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t, keep=("s1",))["s1"], t=t
                    )["out"]
                fns["chain_att_mkmoe"] = lambda f=chain_att, t=t: mkmoe(f()["s1"], t=t)["out"]
            if PROBE:
                fns["probe_a"] = lambda c=cm, t=t: mka_p(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t)
                fns["probe_b"] = lambda t=t, f=fix: mkb_p(f["o_lat"], f["x_out"], f["coef"], t=t)
            cases[t] = (cm, cc, fns)
        for t, (cm, cc, fns) in cases.items():  # compile every case before any capture
            for name in list(fns):
                try:
                    tr.warm(fns[name])
                except Exception as e:  # noqa: BLE001 -- recorded, the case dropped
                    traceback.print_exc()
                    _check_hang(e)
                    rows(f"warm_{name}", t=t, status="info" if _info_reject(name, e) else "fail", error=str(e)[:600],
                         l1_reject=_l1_reject(e), **knobs)  # fmt: skip
                    del fns[name]
        for t, (cm, cc, fns) in cases.items():
            name = None
            try:
                us, got = {}, {}
                for name, fn in fns.items():
                    out = tr.run(fn)
                    if name.startswith("probe_"):
                        prog = dict(probe_a=mka_p, probe_b=mkb_p, probe_one=mk1_p)[name]
                        raw = ttnn.to_torch(prog.probe)
                        dump = os.path.join(os.environ.get("GLM53_WORK", "/tmp"), "dumps",
                                            f"{os.environ.get('GLM53_RUN_ID', 'run')}_{name}_t{t}.pt")  # fmt: skip
                        os.makedirs(os.path.dirname(dump), exist_ok=True)
                        torch.save(raw, dump)
                        p = MK.decode_probe(raw, len(prog.last_readers.assign), chunks=getattr(prog, "chunks", 0),
                                            groups=getattr(prog, "groups", 0),
                                            mhz=(lambda f: f if f and f >= 1000 else 1350.0)(aiclk_mhz()))  # fmt: skip
                        rows(name, t=t, status="info", probe_dump=dump, aiclk_mhz=aiclk_mhz(), **p, **knobs)
                        tr.release()
                        continue
                    got[name] = {k: host(v) for k, v in out.items()} if isinstance(out, dict) else host(out)
                    us[name] = timed(tr, REPLAYS)
                    tr.release()
                if os.environ.get("GLM53_MK_DUMP"):  # debugging: every compared tensor, host side
                    got["weights"] = dict(w_uk=host(attn.w_uk), wv_uk=host(mka.w_uk.weight), w_uv=host(attn.w_uv),
                                          wv_uv=host(mkb.w_uv.weight))  # fmt: skip
                    torch.save(got, f"{os.environ['GLM53_MK_DUMP']}_t{t}.pt")
                r = dict(t=t, execution=tr.mode, timing_us=us, variant="mk_mla_v1", **knobs)
                if us.get("mk_att") and us.get("chain_att"):
                    r["speedup"] = round(us["chain_att"] / us["mk_att"], 3)
                if us.get("mk_att2") and us.get("chain_att"):
                    r["speedup2"] = round(us["chain_att"] / us["mk_att2"], 3)
                rows("mk_mla_timing", status="info", **r)
                # numerics, program A (inputs and starting caches identical)
                if "mk_a" in got and "chain_a" in got:
                    ra = dict(t=t, **knobs)
                    keys = ("hn", "x_out", "coef", "qk", "latent", "q", "q_lat")
                    compare(ra, {k: got["mk_a"][k] for k in keys}, {k: got["chain_a"][k] for k in keys})
                    cm_h, cc_h = host(cm["cache"].latent), host(cc["cache"].latent)
                    wr = written_rows(cm, cm_h).expand_as(cm_h)
                    ra["cache_rest_bitwise"] = bool(torch.equal(cm_h[~wr], cc_h[~wr]))
                    compare(ra, {"cache_rows": cm_h[wr]}, {"cache_rows": cc_h[wr]})
                    ok = (ra["hn_bitwise"] and ra["x_out_bitwise"] and ra["coef_bitwise"] and ra["cache_rest_bitwise"]
                          and min(ra[f"{k}_pcc"] for k in keys + ("cache_rows",)) >= 0.9999
                          and max(ra[f"{k}_rel_l2"] for k in ("latent", "q_lat", "cache_rows")) <= 0.02)  # fmt: skip
                    ra["status"] = "pass" if ok else "fail"
                    rows("mk_mla_a_vs_chain", **ra)
                if "mk_b" in got and "chain_b" in got:
                    rb = dict(t=t, **knobs)
                    compare(rb, {k: got["mk_b"][k] for k in ("a", "s1")}, {k: got["chain_b"][k] for k in ("a", "s1")})
                    rb["status"] = "pass" if min(rb["a_pcc"], rb["s1_pcc"]) >= 0.9999 else "fail"
                    rows("mk_mla_b_vs_chain", **rb)
                if "mk_b2" in got and "chain_b" in got:  # the attention phase vs the flash-MLA op (same q_lat)
                    rb2 = dict(t=t, **knobs)
                    keys = ("o_lat", "a", "s1")
                    compare(rb2, {k: got["mk_b2"][k] for k in keys},
                            dict(o_lat=got["flash"], a=got["chain_b"]["a"], s1=got["chain_b"]["s1"]))  # fmt: skip
                    _vs_golden(rb2, keys)
                    rows("mk_mla_b2_vs_chain", **rb2)
                if "mk_one" in got:  # the one-program half: vs the chain (when it ran) + its attention vs float64
                    r1 = dict(t=t, **knobs)
                    keys = ("q_lat", "o_lat", "a", "s1")
                    r1["status"] = "fail"  # no chain comparison: the float64 check below is not enough for A / B
                    if "chain_att" in got:
                        compare(r1, {k: got["mk_one"][k] for k in keys}, {k: got["chain_att"][k] for k in keys})
                        _vs_golden(r1, keys)
                    if "chain_a" in got:
                        compare(r1, {k: got["mk_one"][k] for k in ("hn", "x_out", "latent")},
                                {k: got["chain_a"][k] for k in ("hn", "x_out", "latent")})  # fmt: skip
                    em = o_lat_f64(cm, got["mk_one"]["q_lat"], attn.scale)
                    compare(r1, {"o_lat_vs_f64": got["mk_one"]["o_lat"][:, : t]}, {"o_lat_vs_f64": em.float()})
                    if "chain_att" in got:  # the flash op on its own q_lat: info
                        em_c = o_lat_f64(cc, got["chain_att"]["q_lat"], attn.scale)
                        compare(r1, {"chain_o_lat_vs_f64": got["chain_att"]["o_lat"][:, :t]},
                                {"chain_o_lat_vs_f64": em_c.float()})  # fmt: skip
                    r1["pos"] = cm["pos"]
                    # gates: PCC and relative L2 (a scale error passes PCC: the deferred-norm bug, latent x0.04 at PCC
                    # 0.99999); hn is not compared under the deferred norm (it is h * w by design)
                    rl = dict(o_lat_vs_f64=0.01, q_lat=0.02, latent=0.02)
                    r1["rel_l2_bad"] = [k for k, lim in rl.items() if r1.get(f"{k}_rel_l2", 0.0) > lim]
                    if r1["o_lat_vs_f64_pcc"] < 0.9999 or r1["rel_l2_bad"]:
                        r1["status"] = "fail"
                    if os.environ.get("GLM53_MLA_DEFER") == "1":
                        r1["hn_note"] = "deferred norm: hn = h * w (not the chain's normalised hn)"
                    rows("mk_mla_one_vs_chain", **r1)
                if us.get("mk_one") and us.get("chain_att"):
                    rows("mk_mla_one_timing", t=t, execution=tr.mode, status="info", variant="mk_mla_one",
                         timing_us={"mk_one": us["mk_one"], "chain_att": us["chain_att"]},
                         speedup=round(us["chain_att"] / us["mk_one"], 3), **knobs)  # fmt: skip
                if "mk_att2" in got and "chain_att" in got:
                    rh2 = dict(t=t, **knobs)
                    keys = ("q_lat", "o_lat", "a", "s1")
                    compare(rh2, {k: got["mk_att2"][k] for k in keys}, {k: got["chain_att"][k] for k in keys})
                    _vs_golden(rh2, keys)
                    rows("mk_mla_half2_vs_chain", **rh2)
                if "mk_att" in got and "chain_att" in got:
                    rh = dict(t=t, **knobs)
                    keys = ("q_lat", "o_lat", "a", "s1")
                    compare(rh, {k: got["mk_att"][k] for k in keys}, {k: got["chain_att"][k] for k in keys})
                    # the bar is per program (A's q_lat, B's a / s1 above); end to end the residual s1 and the attention
                    # output a carry it (o_lat is the flash-MLA op's output on A's q_lat: recorded)
                    # end to end (A -> flash op -> B on A's own q_lat): informational; the bars are per program (the A
                    # and B rows above) and A's q_lat differs from the chain's in the last bf16 bits, which the softmax
                    # of the flash op amplifies at T = 4
                    rh["acceptance"] = "per_program_rows"
                    rh["status"] = "info" if min(rh["s1_pcc"], rh["a_pcc"]) >= 0.999 else "fail"
                    rows("mk_mla_half_vs_chain", **rh)
                soak_fn = "mk_one" if "mk_one" in fns else "mk_att"
                if SOAK_S > 0 and soak_fn in fns:  # steady replays (stability + per-replay wall clock)
                    import time as _time

                    tr.run(fns[soak_fn])
                    n, t0 = 0, _time.perf_counter()
                    while _time.perf_counter() - t0 < SOAK_S:
                        for _ in range(100):
                            ttnn.execute_trace(mesh, tr.tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        n += 100
                    el = _time.perf_counter() - t0
                    tr.release()
                    rows("mk_mla_soak", t=t, execution=tr.mode, status="info", replays=n, seconds=round(el, 1),
                         timing_us={f"{soak_fn}_soak_per_replay": round(el * 1e6 / n, 1)}, **knobs)  # fmt: skip
            except Hung:
                raise
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                err = traceback.format_exc()
                # an L1 reject of the op-chain baseline (the flash-MLA op at T = 4 next to the case's tensors) is a
                # configuration limit of the reference, recorded as info; anything else fails the row
                rows("mk_mla_case", t=t, status="info" if _info_reject(name, e) else "fail", l1_reject=_l1_reject(e),
                     program=name, error=(err[:400] + " ... " + err[-800:]) if len(err) > 1200 else err, **knobs)  # fmt: skip
                _check_hang(e)
                tr.release()
        if mkmoe is not None:  # steady state: two layers chained in one trace
            for t, (cm, cc, fns) in cases.items():
                if t not in LAYER_TS:
                    continue
                try:
                    us = {}

                    def two_mk(c=cm, t=t):
                        y = c["x"]
                        for _ in range(2):
                            ra = mka(y, c["cache"].latent, c["pt"], c["cur"], t=t)
                            o_lat = L.flash(c, ra["q_lat"], heads=heads(t))
                            y = mkmoe(mkb(o_lat, ra["x_out"], ra["coef"], t=t)["s1"], t=t)["out"]
                        return y

                    def two_chain(c=cc, t=t):
                        y = c["x"]
                        for _ in range(2):
                            y = mkmoe(L.chain_att(dict(c, x=y)), t=t)["out"]
                        return y

                    def two_one(c=cm, t=t):
                        y = c["x"]
                        for _ in range(2):
                            y = mkmoe(mk1(y, c["cache"].latent, c["pt"], c["cur"], t=t, keep=("s1",))["s1"], t=t)["out"]
                        return y

                    steady = [("chain_att_mkmoe_x2", two_chain)]
                    if TWO:
                        steady.append(("mk_layer_x2", two_mk))
                    if mk1 is not None:
                        steady.append(("mk_one_layer_x2", two_one))
                    rejects = {}
                    for name, fn in steady:  # each case on its own: an L1 reject of one (the op chain's flash-MLA op
                        try:  # next to the layer's tensors) must not drop the others' numbers
                            tr.warm(fn)
                            tr.run(fn)
                            us[name] = timed(tr, REPLAYS)
                        except Exception as e:  # noqa: BLE001
                            if not _l1_reject(e):
                                raise
                            rejects[name] = str(e)[:300]
                            tr.release()
                            continue
                        tr.release()
                    ok = all(k in REFERENCE for k in rejects)
                    rows("mk_mla_layer_steady", t=t, execution=tr.mode, timing_us=us, variant="mk_mla_v1",
                         steady_per_layer_us={k: round(v / 2, 1) for k, v in us.items() if v},
                         status="info" if ok else "fail", l1_rejects=rejects, **knobs)  # fmt: skip
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    rows("mk_mla_layer_steady", t=t, status="fail", error=traceback.format_exc()[-800:], **knobs)
                    _check_hang(e)
                    tr.release()
    except Hung as h:
        rows("device_hang", status="fail", error=str(h), **knobs)
        rows.flush()
        print(f"DEV_MK_MLA_HUNG {h}", flush=True)
        return 3
    finally:
        rows.flush()
        if own:
            ttnn.close_device(mesh) if os.environ.get("GLM53_MOCK") == "1" else ttnn.close_mesh_device(mesh)
    bad = rows.failed()
    print(f"DEV_MK_MLA_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
