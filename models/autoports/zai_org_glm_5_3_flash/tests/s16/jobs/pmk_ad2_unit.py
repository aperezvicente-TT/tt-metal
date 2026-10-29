# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 TP2 pair / the TP8 2x4; watched once then unwatched): the device unit of AD-2a-DENSE (the
attention-half merge, program 2026-10-27 00:2xZ; doc/ttnn/16-prefill-production/pmk/merge/README.md "AD-2a"): the PMK
attention kernel (tt/pmk_attn.py rev 12.2, dense geometry, chunk_start 0) as the chunk-0 attention core of a sparse_dsa
layer (prefill_tp.MlaPrefill._ad2_dense, served behind GLM53_PMK_DSA_AD2=dense) vs the op path's SDPA on the SAME
q_lat / latent: the served chunk-0 call of the sharded cache (causal over the chunk's own latents, K = V, absorbed).

Inputs: the real L-layer chunk-0 tensors, built once before any capture by the op path's own steps (the golden residual ->
mHC pre -> gather -> q_a | kv_a -> RMSNorms -> q_b -> heads -> W_uk). Each arm is TRACED on its own (one live trace at a
time: the op arm first, captured, replayed, read and released, then the PMK arm), its output poisoned (NaN, device copy
from a source allocated before the first capture) before every replay.

Criteria: the PMK o_lat finite and BITWISE run to run (replay i vs replay 0) on every replay; reported: PCC / max |d| of the
PMK o_lat vs the op SDPA per chip, both vs a host fp32 reference (pmk_attn.attention_host) on chip 0, the replay us of each.
ARGS: layer (3), rows (2048), det (5: poisoned compared replays), replays (50: timing), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_attn as PA  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 3))
N = int(args.get("rows", 2048))
DET = int(args.get("det", 5))
REPLAYS = int(args.get("replays", 50))
tag = args.get("tag", f"pmk-ad2-unit-L{L}")
NDEV = int(mesh.get_num_devices())
DRAM = ttnn.DRAM_MEMORY_CONFIG
os.environ["GLM53_PF_DSA"] = "1"
os.environ.setdefault("GLM53_PF_MHC", "comp")
os.environ.setdefault("GLM53_PF_MHC_POST", "fused")
for k_, v_ in (("GLM53_PF_MHC_PRE", "fused7a"), ("GLM53_PF_MHC_FPU", "1"), ("GLM53_PF_MHC_POST7A", "1")):
    os.environ.setdefault(k_, v_)
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
t0 = time.time()
LAYERS = [int(v) for v in args.get("layers", [2, L])]  # the stack backend needs one non-DSA layer (as p3_long_check)
assert L in LAYERS, (L, LAYERS)
LI = LAYERS.index(L)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
assert pf.dsa, "TPPrefill built without GLM53_PF_DSA"
pl = pf.layers[LI]
assert pl.kind == PT.SPARSE_DSA, (L, pl.kind, "AD-2a is the sparse_dsa layers' attention core")
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
assert c == N, (c, N)
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
try:
    rows, _ = S.golden_layer(L, N)
except FileNotFoundError:  # the model is causal: the first N rows of the 32K golden are the N-token golden's rows
    big, _ = S.golden_layer(L, 32768)
    rows = big.reshape(32768, -1)[:N].clone()
    del big
rows = rows.reshape(N, 4, 4096).float()
mla = pl.mla
cache = pf.kv.extra["latent"][LI]

# ---- the chunk-0 inputs, by the op path's own steps (MlaPrefill.__call__ up to q_lat), built ONCE, outside every trace
x0 = pf.streams_to_device(rows, sp)
h, post, comb = pf._mhc_pre(pl, "attn", x0)
hn = pl.rms(h, "attn", BF16)
if hn is not h:
    ttnn.deallocate(h)
for t_ in (post, comb):
    ttnn.deallocate(t_)
hb = pf.gather(hn, sp)
qk = PT.pf_mm(mla.attn.ds_qa_kva, hb)
t = int(qk.shape[-2])
q_r = ttnn.slice(qk, (0, 0, 0, 0), (1, 1, t, mla.q_lora))
c_ = ttnn.slice(qk, (0, 0, 0, mla.q_lora), (1, 1, t, mla.q_lora + mla.kv_lora))
q_r = ttnn.rms_norm(q_r, weight=mla.attn.qa_norm, epsilon=mla.eps, compute_kernel_config=mla.ckc_fp32)
latent = ttnn.rms_norm(c_, weight=mla.attn.kva_norm, epsilon=mla.eps, compute_kernel_config=mla.ckc_fp32)
q = PT.pf_mm(mla.shard.ds_qb, q_r)
q, _, _ = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=mla.h, num_kv_heads=0, transpose_k_heads=False,
                                                 memory_config=DRAM)  # fmt: skip
q_lat = ttnn.matmul(q, mla.w_uk, compute_kernel_config=mla.ckc_head, dtype=BF16, memory_config=DRAM)
for t_ in (qk, q_r, c_, q, hb, hn):
    try:
        ttnn.deallocate(t_)
    except Exception:  # noqa: BLE001 -- a view / an alias already freed
        pass
assert tuple(q_lat.shape) == (1, mla.h, N, 512) and tuple(latent.shape) == (1, 1, N, 512), (q_lat.shape, latent.shape)
S.log(f"{tag}: pf + inputs ready in {time.time() - t0:.1f} s (tp{NDEV}, c={c}, H_r {mla.h}, scale {mla.scale})")

prog = mla.sdpa_prog if mla.sdpa_prog.q_chunk_size <= N else mla.attn.prefill_program


def fn_op():
    """the served chunk-0 attention (MlaPrefill.__call__, keep_lat: the sharded cache's fresh chunk)"""
    return ttnn.transformer.scaled_dot_product_attention(q_lat, latent, latent, is_causal=True, scale=mla.scale,
                                                         program_config=prog, compute_kernel_config=mla.sdpa_ckc,
                                                         memory_config=DRAM)  # fmt: skip


def fn_pmk():
    return mla._ad2_dense(q_lat, latent, cache)


per_chip = lambda t_: [
    ttnn.to_torch(d).float() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])
]  # noqa: E731


def cmp(a_list, b_list):
    out = []
    for a, b in zip(a_list, b_list):
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        fin = torch.isfinite(a) & torch.isfinite(b)
        out.append(dict(bitwise=bool(torch.equal(a, b)), finite=bool(torch.isfinite(a).all()),
                        pcc=round(C.pcc(a[fin], b[fin]), 7) if fin.sum() > 1 else None,
                        maxabs=float((a - b)[fin].abs().max()) if fin.any() else None))  # fmt: skip
    return out


# every program compiled before the FIRST capture (the op SDPA, the PMK arm incl. its state / latent-store copy, the
# poison copies); the NaN poison source allocated here
S.log(f"{tag}: compile launches (op SDPA, PMK attention)")
o_ref = fn_op()
o_pm = fn_pmk()
ttnn.synchronize_device(mesh)
st = next(iter(PT._AD2.values()))
nan = ttnn.from_torch(torch.full(tuple(o_ref.shape), float("nan")), dtype=o_ref.dtype, layout=o_ref.layout, device=mesh,
                      memory_config=DRAM, mesh_mapper=(ttnn.ReplicateTensorToMesh(mesh) if NDEV > 1 else None))  # fmt: skip
ttnn.copy(nan, o_ref)
ttnn.copy(nan, o_pm)
ttnn.synchronize_device(mesh)
for t_ in (o_ref, o_pm):
    ttnn.deallocate(t_)
geo = st["geo"]
row = dict(test=f"PMK AD-2a-dense L{L} tp{NDEV} (bucket {c}): the PMK attention kernel rev {PA.ATT_REV} (dense, chunk 0) vs the "
                f"op SDPA (causal over the chunk's latents), traced, {DET} poisoned replays each",
           milestone="PMK-AD2", layer=L, tp=NDEV, bucket=c, heads=mla.h, geo=dict(nk=geo.nk, qg=geo.qg, cores=geo.cores,
           items=geo.items, lat_l1=geo.lat_l1, dual_noc=geo.dual_noc, kslots=geo.kslots), l1=PA.l1_plan(geo),
           cbs=[(k_, dt_, page_, n_) for k_, dt_, page_, n_ in PA.cb_spec(geo)], mock=C.MOCK, tag=tag)  # fmt: skip
S.log(f"{tag}: geometry {row['geo']}; L1 plan {row['l1']}")
if C.MOCK:
    row["status"] = "mock"
else:
    res = {}
    for nm_, fn_ in (("op", fn_op), ("pmk", fn_pmk)):  # one live trace at a time
        calls0 = st["calls"]
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            o_t = fn_()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        det, first = [], None
        for i in range(DET):
            ttnn.copy(nan, o_t)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            cur = per_chip(o_t)
            if first is None:
                first = cur
            d = dict(finite=all(bool(torch.isfinite(x).all()) for x in cur),
                     bitwise=all(torch.equal(a, b) for a, b in zip(cur, first)))  # fmt: skip
            det.append(d)
        perf = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} {nm_}")
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        out = per_chip(o_t)
        ttnn.release_trace(mesh, tid)
        ttnn.deallocate(o_t)
        res[nm_] = dict(det=det, perf=perf, out=out, launches=st["calls"] - calls0)
        S.log(f"{tag}: {nm_}: det {det}; replay median {perf['median_us']:.0f} us")
    vs_op = cmp(res["pmk"]["out"], res["op"]["out"])
    # host fp32 reference on chip 0 (q_lat / latent are replicated per chip only in their latent; q_lat is the chip's heads)
    q0 = per_chip(q_lat)[0][0]  # [H_r, C, 512]
    lat0 = per_chip(latent)[0].reshape(-1, 512)  # [C, 512]
    ref = PA.attention_host(q0, lat0, 0, scale=mla.scale)  # [H_r, C, 512] fp32
    vs_host = dict(pmk=cmp([res["pmk"]["out"][0][0]], [ref])[0], op=cmp([res["op"]["out"][0][0]], [ref])[0])
    ok = all(d["finite"] and d["bitwise"] for d in res["pmk"]["det"])
    pmin = min(ch["pcc"] for ch in vs_op if ch["pcc"] is not None)
    row.update(det_pmk=res["pmk"]["det"], det_op=res["op"]["det"], vs_op=vs_op, pcc_vs_op=pmin, vs_host=vs_host,
               us_pmk=res["pmk"]["perf"]["median_us"], us_op=res["op"]["perf"]["median_us"],
               perf_pmk=res["pmk"]["perf"], perf_op=res["op"]["perf"], pmk_launches_in_capture=res["pmk"]["launches"],
               status="pass" if ok else "fail")  # fmt: skip
    S.log(f"{tag}: PMK vs op {[(ch['pcc'], ch['maxabs']) for ch in vs_op]}; vs host fp32 pmk {vs_host['pmk']['pcc']} "
          f"op {vs_host['op']['pcc']}; PMK {row['us_pmk']:.0f} us vs op SDPA {row['us_op']:.0f} us; deterministic + "
          f"finite {ok}; status {row['status']}")  # fmt: skip
    if bool(args.get("record", True)):
        C.record("pcc", dict(test=row["test"], milestone="PMK-AD2", layer=L, tp=NDEV, bucket=c, pcc_vs_op=pmin,
                             pcc_pmk_vs_host=vs_host["pmk"]["pcc"], pcc_op_vs_host=vs_host["op"]["pcc"],
                             deterministic=ok, status=row["status"], tag=tag))  # fmt: skip
        C.record("perf", dict(test=f"PMK AD-2a-dense L{L} tp{NDEV} traced replay: PMK attention (chunk 0) vs op SDPA",
                              milestone="PMK-AD2", layer=L, tp=NDEV, bucket=c, us=row["us_pmk"], us_op=row["us_op"],
                              replays=res["pmk"]["perf"]["replays"], status=row["status"], tag=tag))  # fmt: skip
RESULT = row
print("PMK_AD2_UNIT", {k_: v_ for k_, v_ in row.items() if k_ not in ("perf_pmk", "perf_op", "cbs")}, flush=True)
