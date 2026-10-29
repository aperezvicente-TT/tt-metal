# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 ACCURACY (program 2026-10-24 19:58:04Z, the top item): the served op prefill's
scale error enters at L3 (the first MoE + DSA layer: norm ratio vs fp32 0.992 -> 1.027). One prefill LAYER of
tt/prefill_tp.py (the served layer sequence: TPPrefill.layer + TPPrefill.ffn, comp mHC, ops post) fed the fp32 HF
reference input of that layer (hf_ref/8be48d885fce L<L>/streams_in = goldens/layer_io), with TAPS on every sublayer
boundary, each compared with the fp32 reference (hf_ref L<L>/sub.pt, mid.pt, router.pt, L<L+1>/streams_in):
  attn_in (mHC pre + input norm), attn_out (attention after the reduce-scatter), mid (the residual after the attention
  mHC post), ffn_in (mHC pre + post-attention norm, fp32), the router (top-8 sets, the renormalised x 2.5 weights),
  routed / shared (the chips' partials summed on the host), ffn_out (after the reduce-scatter), out (the layer output).
Per tap: PCC, norm ratio |tt| / |ref|, projection scale <tt, ref> / <ref, ref>, rel L2, max abs. TRACED: the taps are
copies made inside ONE capture of the tapped layer (the eager call is the compile); the served pf.layer runs as its own
capture too and must match the tapped replica's output (the taps follow the served path).

ARGS: layer (3), layers (the backend's, default [2, layer]), n (2048), hf_ref (the hf_ref root), env ({}), tag, record (True)."""
import gc
import os
import sys

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, EpMoe, swiglu_prefill  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
L = int(args.get("layer", 3))
N = int(args.get("n", 2048))
HF = args.get("hf_ref", "/opt/glm53/data/hf_ref/8be48d885fce")
if not os.path.isdir(HF):
    HF = HF.replace("/opt/glm53/share", "/opt/glm53/home/mpi-shfs")  # host-p300's path of the share
tag = args.get("tag", f"acc-l{L}-taps")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)

LAYERS = sorted({int(v) for v in args.get("layers", [2, L])} | {L})  # the backend needs a non-DSA layer (L2: KDA)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
pf, kv = ent["pf"], ent["kv"]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
LI = LAYERS.index(L)
pl = pf.layers[LI]
assert pl.L == L and pl.kind != DENSE_KDA, (pl.L, pl.kind)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
try:
    x_in, x_out = S.golden_layer(L, N)  # goldens/layer_io: L0 .. L4
except FileNotFoundError:  # beyond: the hf_ref chain itself (L<L>/streams_in -> L<L+1>/streams_in)
    x_in = torch.load(os.path.join(HF, f"L{L:02d}", "streams_in.pt"), map_location="cpu")
    x_out = torch.load(os.path.join(HF, f"L{L + 1:02d}", "streams_in.pt"), map_location="cpu")
pad = torch.zeros(c, 4, 4096)
pad[:N] = x_in.reshape(N, 4, 4096).float()
X = pf.streams_to_device(pad, sp)
Y = pf.streams_to_device(torch.zeros(c, 4, 4096), sp)


def tapped(taps: dict):
    """TPPrefill.layer + TPPrefill.ffn (comp mHC, ops post, moe ep), each sublayer boundary copied into `taps`"""
    cp = lambda t: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # noqa: E731
    h, post, comb, xs = pl.hc["attn"].pre_streams(X)
    hn = pl.rms(h, "attn", BF16)
    taps["attn_in"] = cp(hn)
    hb = pf.gather(hn, sp)
    sparse = pf._sparse_now
    a = pf.reduce(pf.attention(pl, hb, ci, LI, hn, sparse), sp)
    taps["attn_out"] = cp(a)
    s1 = pl.hc["attn"].post_streams(a, xs, post, comb)
    x1 = ttnn.concat(s1, dim=-1)
    taps["mid"] = cp(x1)
    h2, post2, comb2, _ = pl.hc["ffn"].pre_streams(x1, s1)
    x32 = pl.rms(h2, "ffn", FP32)
    taps["ffn_in"] = cp(x32)
    idx, scores = pl.moe.router(x32, decode=False)
    hb2 = pf.gather(ttnn.typecast(x32, BF16), sp)
    taps["hb2"] = cp(hb2)  # the experts' input (all rows, bf16)
    k = int(idx.shape[-1])
    sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
    pack = ttnn.concat([ttnn.typecast(idx, FP32), sel], dim=-1)
    taps["router_pack"] = cp(pack)  # local rows: [ids | the selected unbiased scores]
    pack = pf.gather(pack, sp)
    rows = int(pack.shape[-2])
    ids_f = ttnn.slice(pack, (0, 0, 0, 0), (1, 1, rows, k))
    sel_g = ttnn.slice(pack, (0, 0, 0, k), (1, 1, rows, 2 * k))
    idx16 = ttnn.typecast(ids_f, ttnn.uint16)
    # EpMoe's on-device routing weights, as EpMoe computes them (the host check above rebuilds them from sel)
    wsum = ttnn.sum(sel_g, dim=-1, keepdim=True)
    taps["wsum_dev"] = cp(wsum)
    taps["w_dev"] = cp(ttnn.multiply(ttnn.multiply(sel_g, ttnn.reciprocal(ttnn.add(wsum, 1e-20))), pl.moe.scaling))
    taps["sel_g"] = cp(sel_g)
    if getattr(pl, "epmoe", None) is None:
        pl.epmoe = EpMoe(pf, pl.moe, xsl=getattr(pl, "xsl", None))
    pl.epmoe.tap_on = bool(args.get("expert_tap", False))
    routed = pl.epmoe(hb2, idx16, sel_g)
    if pl.epmoe.tap_on:
        taps["x_out"], taps["x_dest"] = pl.epmoe.tap["out"], pl.epmoe.tap["dest"]
    taps["routed"] = cp(routed)  # per-chip partials (all rows)
    shared = swiglu_prefill(pl.shared, hb2)
    taps["shared"] = cp(shared)
    m = pf.reduce(ttnn.add(routed, shared), sp)
    taps["ffn_out"] = cp(m)
    s2 = pl.hc["ffn"].post_streams(m, s1, post2, comb2)
    out = ttnn.concat(s2, dim=-1)
    taps["out"] = cp(out)
    dead = [h, hn, hb, a, x1, h2, x32, idx, scores, hb2, sel, pack, ids_f, sel_g, idx16, routed, shared, m, out]
    for t_ in dead + list(xs) + list(s1) + list(s2):
        try:
            ttnn.deallocate(t_)
        except Exception:  # noqa: BLE001  (aliases already freed)
            pass


def served():
    y = pf.layer(LI, X, ci, sp, keep_input=True)
    ttnn.copy(y, Y)
    ttnn.deallocate(y)


def captured(fn):
    fn()  # the compile launch (the only non-traced execution)
    ttnn.synchronize_device(mesh)
    if C.MOCK:
        return None
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    return tid


def rows_host(t) -> torch.Tensor:
    """local-row tensor (SP: rows sharded over the chips in order) -> [rows, cols] fp32"""
    parts = ttnn.get_device_tensors(t) if TPN > 1 else [t]
    hs = [ttnn.to_torch(p_).float().reshape(-1, int(p_.shape[-1])) for p_ in parts]
    return torch.cat(hs, dim=0) if sp else hs[0]


def partial_sum(t) -> torch.Tensor:
    parts = ttnn.get_device_tensors(t) if TPN > 1 else [t]
    return sum(ttnn.to_torch(p_).float().reshape(-1, int(p_.shape[-1])) for p_ in parts)


def metrics(tt: torch.Tensor, ref: torch.Tensor) -> dict:
    tt, ref = tt.reshape(-1).double(), ref.reshape(-1).double()
    return dict(pcc=round(C.pcc(tt, ref), 6), norm_ratio=round(float(tt.norm() / ref.norm()), 5),
                proj_scale=round(float((tt @ ref) / (ref @ ref)), 5), rel_l2=round(float((tt - ref).norm() / ref.norm()), 5),
                max_abs=round(float((tt - ref).abs().max()), 5))  # fmt: skip


taps = {}
tid_t = captured(lambda: (taps.clear(), tapped(taps)))
row = dict(test=f"accuracy L{L} sublayer taps vs fp32 HF reference (op prefill, served sequence, n={N}) tp{TPN}",
           milestone="ACC-L3", layer=L, kind=pl.kind, n=N, tp=TPN, mode="sp" if sp else "rep", moe=pf.moe_impl,
           mhc=pf.mhc_mode, mhc_post=pf.mhc_post, mla_form=pf.mla_form, xsl=getattr(pl, "xsl", None) is not None,
           env={k_: os.environ.get(k_) for k_ in ("GLM53_TP_XSL", "GLM53_TP_XSL_SLICED")}, traced=tid_t is not None,
           tag=tag, status="mock" if C.MOCK else "info")  # fmt: skip
if not C.MOCK:
    sub = torch.load(os.path.join(HF, f"L{L:02d}", "sub.pt"), map_location="cpu", weights_only=False)
    mid = torch.load(os.path.join(HF, f"L{L:02d}", "mid.pt"), map_location="cpu", weights_only=False)
    rt = torch.load(os.path.join(HF, f"L{L:02d}", "router.pt"), map_location="cpu", weights_only=False)
    T = {}
    T["attn_in"] = metrics(rows_host(taps["attn_in"])[:N], sub["attn_in"])
    T["attn_out"] = metrics(rows_host(taps["attn_out"])[:N], sub["attn_out"])
    T["mid"] = metrics(rows_host(taps["mid"])[:N], mid.reshape(N, -1))
    T["ffn_in"] = metrics(rows_host(taps["ffn_in"])[:N], sub["ffn_in"])
    T["ffn_out"] = metrics(rows_host(taps["ffn_out"])[:N], sub["ffn_out"])
    T["out"] = metrics(rows_host(taps["out"])[:N], x_out.reshape(N, -1))
    routed, shared = partial_sum(taps["routed"])[:N], partial_sum(taps["shared"])[:N]
    T["routed_plus_shared"] = metrics(routed + shared, sub["ffn_out"])
    T["routed_share_of_norm"] = round(float(routed.norm() / (routed + shared).norm()), 5)
    T["shared_share_of_norm"] = round(float(shared.norm() / (routed + shared).norm()), 5)
    pk = rows_host(taps["router_pack"])[:N]
    k = pk.shape[1] // 2
    ids_tt, sel_tt = pk[:, :k].round().long(), pk[:, k:]
    w_tt = sel_tt / (sel_tt.sum(-1, keepdim=True) + 1e-20) * float(pl.moe.scaling)
    ids_hf, w_hf = rt["indices"].reshape(N, -1).long(), rt["weights"].reshape(N, -1).float()
    same = (ids_tt.sort(-1)[0] == ids_hf.sort(-1)[0]).all(-1)
    o_tt, o_hf = ids_tt.argsort(-1), ids_hf.argsort(-1)
    wt_s, wh_s = w_tt.gather(1, o_tt), w_hf.gather(1, o_hf)
    T["router"] = dict(top8_same_rows=round(float(same.float().mean()), 5), w_sum_tt=round(float(w_tt.sum(-1).mean()), 5),
                       w_sum_hf=round(float(w_hf.sum(-1).mean()), 5),
                       w_rel_l2_same_rows=round(float((wt_s[same] - wh_s[same]).norm() / wh_s[same].norm()), 5),
                       w_proj_scale_same_rows=round(float((wt_s[same] * wh_s[same]).sum() / (wh_s[same] ** 2).sum()), 5),
                       scaling=float(pl.moe.scaling))  # fmt: skip
    d0_ = lambda t: (ttnn.get_device_tensors(t)[0] if TPN > 1 else t)  # noqa: E731  (gathered: chip 0 = all rows)
    sel_d = ttnn.to_torch(d0_(taps["sel_g"])).float().reshape(-1, k)[:N]
    w_d = ttnn.to_torch(d0_(taps["w_dev"])).float().reshape(-1, k)[:N]
    ws_d = ttnn.to_torch(d0_(taps["wsum_dev"])).float().reshape(-1)[:N]
    w_h = sel_d / (sel_d.sum(-1, keepdim=True) + 1e-20) * float(pl.moe.scaling)
    T["device_weights"] = dict(w_sum_dev=round(float(w_d.sum(-1).mean()), 5), w_sum_host=round(float(w_h.sum(-1).mean()), 5),
                               proj_scale=round(float((w_d * w_h).sum() / (w_h ** 2).sum()), 5),
                               wsum_dev_over_host=round(float((ws_d / sel_d.sum(-1)).mean()), 5),
                               rel_l2=round(float((w_d - w_h).norm() / w_h.norm()), 5))  # fmt: skip
    out_taps = rows_host(taps["out"])[:N]
    if args.get("dump_dir"):  # the routed path's own inputs / outputs for the host replay (hf_routed_quant --tt)
        os.makedirs(args["dump_dir"], exist_ok=True)
        hb2_h = (ttnn.get_device_tensors(taps["hb2"])[0] if TPN > 1 else taps["hb2"])
        extra = {}
        if "x_out" in taps:  # per chip: the expert rows [cap, 4096] and the pair -> row map [K C]
            dv = lambda t: ttnn.get_device_tensors(t) if TPN > 1 else [t]  # noqa: E731
            extra = dict(x_out=[ttnn.to_torch(t_).float().reshape(-1, 4096) for t_ in dv(taps["x_out"])],
                         x_dest=[ttnn.to_torch(t_).reshape(-1).long() for t_ in dv(taps["x_dest"])],
                         ffn=pl.epmoe.ffn, combine=pl.epmoe.combine, e_loc=int(getattr(pl.epmoe, "e_loc", 0)))
        torch.save({"hb2": ttnn.to_torch(hb2_h).float().reshape(-1, 4096)[:N], "pack": pk, "routed": routed,
                    "shared": shared, "ffn_out": rows_host(taps["ffn_out"])[:N], "out": out_taps, **extra},
                   os.path.join(args["dump_dir"], f"tt_l{L}_routed.pt"))  # fmt: skip
    # one live trace at a time (traced-unit harness rules): the taps are on the host -> release, free, then the served
    # layer's own compile + capture
    ttnn.release_trace(mesh, tid_t)
    for t_ in taps.values():
        ttnn.deallocate(t_)
    taps.clear()
    tid_s = captured(served)
    ys = pf.streams_from_device(Y, sp).reshape(c, -1)[:N]
    T["served_vs_taps_out"] = metrics(ys, out_taps)
    T["served_out"] = metrics(ys, x_out.reshape(N, -1))
    row["taps"] = T
    first = next((n_ for n_ in ("attn_in", "attn_out", "mid", "ffn_in", "ffn_out", "out")
                  if abs(T[n_]["norm_ratio"] - 1.0) > 0.01), None)  # fmt: skip
    row["first_norm_departure"] = first
    if tid_s is not None and int(args.get("replays", 0)) > 0:  # the served layer's traced time (the TTFT cost of a fix)
        T["served_layer_timing"] = S.replay_stats(mesh, tid_s, replays=int(args["replays"]), label=f"L{L}")
    if tid_s is not None:
        ttnn.release_trace(mesh, tid_s)
if args.get("weight_check") and not C.MOCK:  # EpMoe's OWN device weight tensors (what moe_fused_swiglu reads) for experts
    ep_ = pl.epmoe  # chip k's slot i = expert k e_loc + i (EP)
    wc = {}
    for e_ in [int(v) for v in args["weight_check"]]:
        chip_, slot_ = (e_ // ep_.e_loc, e_ % ep_.e_loc) if not ep_.xsl else (0, e_)
        g_ = ttnn.to_torch(ttnn.get_device_tensors(ep_.gate_up[slot_])[chip_] if TPN > 1 else ep_.gate_up[slot_]).float()
        d_ = ttnn.to_torch(ttnn.get_device_tensors(ep_.down[slot_])[chip_] if TPN > 1 else ep_.down[slot_]).float()
        wc[e_] = dict(gate_up_shape=list(g_.shape), gate_up_norm=round(float(g_.norm()), 6), gate_up_abs=round(float(g_.abs().sum()), 4),
                      down_shape=list(d_.shape), down_norm=round(float(d_.norm()), 6), down_abs=round(float(d_.abs().sum()), 4))
        if args.get("dump_dir"):
            torch.save({"gate_up": g_, "down": d_}, os.path.join(args["dump_dir"], f"epmoe_w_l{L}_e{e_}.pt"))
    row["epmoe_weights"] = wc
S.log(row)
if rec:
    C.record("pcc", row)
RESULT = [row]
