# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair; 2026-10-24 01:25Z, the 32K miss: the B-q split-K path costs ~0.0023 PCC at L3 after the
m-mask fix): WHICH step of split-K attention loses the accuracy, on a synthetic case at the L3 shapes, each path ONE
traced replay. q [1, H, C, D] bf16 RM, kv [1, 1, T, D] bf16 RM, every row's ids = `nsel` distinct keys (sentinel suffix);
the ids split into two disjoint owner halves by unit ((t // 128) % 2), as two chips would hold them.
  full     ttnn.transformer.sparse_sdpa over all ids (the BQ=0 path)
  split    sparse_sdpa_stats over each half -> (o_c bf16, m_c, l_c) + the device merge of MlaPrefill._bq_block (fixed m mask)
  host     the same merge in fp64 on the host from the DEVICE o_c / m_c / l_c (separates merge arithmetic from the partials)
vs an fp64 host reference over all ids; plus each half's m / l vs the host's (the stats' accuracy) and o_c vs the host
partial (the partials' accuracy). Metrics: PCC, max |diff|, rel L2. ARGS: H (64), C (128), T (4096), D (512), nsel (2048),
scale (0.0417), seed, tag."""
import os
import time
import traceback

import torch

import ttnn

mesh, args = MESH, ARGS  # noqa: F821
TPN = int(mesh.get_num_devices())
H, Cr, T, D = int(args.get("H", 64)), int(args.get("C", 128)), int(args.get("T", 4096)), int(args.get("D", 512))
NSEL, W = int(args.get("nsel", 2048)), int(args.get("W", 2304))
SCALE = float(args.get("scale", 0.0417))
MOCK = os.environ.get("GLM53_MOCK", "") == "1"
tag = args.get("tag", "bq-merge-op")
SENT = 0xFFFFFFFF
NEG = -1e30
t0 = time.time()
row, err = dict(H=H, C=Cr, T=T, D=D, nsel=NSEL, W=W, scale=SCALE, tp=TPN), None


def log(m):
    print(f"BQOP {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {tag}: {m}", flush=True)


def one(t_):
    return ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_)


def stat(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    am, bm = a - a.mean(), b - b.mean()
    return dict(pcc=round(float((am @ bm) / (am.norm() * bm.norm() + 1e-30)), 7), maxabs=float("%.4g" % float((a - b).abs().max())),
                rel_l2=float("%.4g" % float((a - b).norm() / (b.norm() + 1e-30))))  # fmt: skip


try:
    g = torch.Generator().manual_seed(int(args.get("seed", 0)))
    q = (torch.randn(1, H, Cr, D, generator=g) * float(args.get("q_std", 1.0))).to(torch.bfloat16)
    kv = (torch.randn(1, 1, T, D, generator=g) * float(args.get("k_std", 1.0))).to(torch.bfloat16)
    ids = torch.full((1, 1, Cr, W), SENT, dtype=torch.int64)
    halves = [torch.full((1, 1, Cr, W), SENT, dtype=torch.int64) for _ in range(2)]
    sel = []
    for r in range(Cr):
        s_ = torch.randperm(T, generator=g)[:NSEL]
        sel.append(s_)
        ids[0, 0, r, :NSEL] = s_
        for h_ in range(2):
            part = s_[(s_ // 128) % 2 == h_]
            halves[h_][0, 0, r, : part.numel()] = part
    rep = ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None
    dev = lambda x, dt: ttnn.from_torch(x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep,  # noqa: E731
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    q_d, kv_d = dev(q, ttnn.bfloat16), dev(kv, ttnn.bfloat16)
    ids_d = dev(ids.to(torch.int64).bitwise_and(SENT).to(torch.int32), ttnn.uint32)
    hv_d = [dev(h_.to(torch.int64).bitwise_and(SENT).to(torch.int32), ttnn.uint32) for h_ in halves]
    from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: PLC0415

    ckc = PT._ckc_fp32(mesh)  # MlaPrefill.ckc_fp32: the B-q / sparse attention config
    KC = PT.sparse_kchunk()
    KF = ttnn.transformer.SparseKVFormat.BF16

    def full():
        return ttnn.transformer.sparse_sdpa(q_d, kv_d, ids_d, D, kv_format=KF, scale=SCALE, k_chunk_size=KC, compute_kernel_config=ckc)

    MSC = float(args.get("merge_scale", SCALE))  # the merge's factor on (m_c - M): SCALE = MlaPrefill._bq_block today

    def split():
        parts = [ttnn.transformer.sparse_sdpa_stats(q_d, kv_d, hv_d[h_], D, kv_format=KF, scale=SCALE, k_chunk_size=KC,
                                                    compute_kernel_config=ckc) for h_ in range(2)]  # fmt: skip
        o_t = [ttnn.to_layout(o_, ttnn.TILE_LAYOUT) for o_, _ in parts]
        st_t = [ttnn.to_layout(s_, ttnn.TILE_LAYOUT) for _, s_ in parts]
        m_ = [ttnn.slice(s_, (0, 0, 0, 0), (1, 1, Cr, H)) for s_ in st_t]
        l_ = [ttnn.slice(s_, (0, 0, 0, H), (1, 1, Cr, 2 * H)) for s_ in st_t]
        M = ttnn.maximum(m_[0], m_[1])
        e_ = [ttnn.multiply(ttnn.exp(ttnn.multiply(ttnn.subtract(m_[h_], M), MSC)), l_[h_]) for h_ in range(2)]
        Z = ttnn.add(e_[0], e_[1])
        w_ = [ttnn.permute(ttnn.divide(e_[h_], Z), (0, 3, 2, 1)) for h_ in range(2)]  # [1, H, C, 1]
        acc = ttnn.add(ttnn.multiply(ttnn.typecast(o_t[0], ttnn.float32), w_[0]), ttnn.multiply(ttnn.typecast(o_t[1], ttnn.float32), w_[1]))
        out = ttnn.typecast(acc, ttnn.bfloat16)
        return out, parts

    outs = {}
    if MOCK:
        f_ = full()
        s_, parts = split()
        outs = dict(full=f_, split=s_, parts=parts)
    else:
        f0 = full()  # compile passes
        s0, p0 = split()
        for t_ in [f0, s0] + [x for pr in p0 for x in pr]:
            ttnn.deallocate(t_)
        keep = {}

        def body_full():
            keep["full"] = full()

        def body_split():
            keep["split"], keep["parts"] = split()

        for nm_, fn in (("full", body_full), ("split", body_split)):
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                fn()
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            outs[nm_] = keep[nm_]
            if nm_ == "split":
                outs["parts"] = keep["parts"]
            ttnn.release_trace(mesh, tid)
    # host fp64 reference + the host merge from the device partials
    qd, kd = q.double()[0], kv.double()[0, 0]  # [H, C, D], [T, D]
    ref = torch.zeros(H, Cr, D, dtype=torch.float64)
    host_parts = [dict(o=torch.zeros(H, Cr, D, dtype=torch.float64), m=torch.zeros(Cr, H, dtype=torch.float64),
                       l=torch.zeros(Cr, H, dtype=torch.float64)) for _ in range(2)]  # fmt: skip
    for r in range(Cr):
        s_ = sel[r]
        sc = torch.einsum("hd,td->ht", qd[:, r], kd[s_])  # [H, nsel] unscaled
        p = torch.softmax(sc * SCALE, dim=-1)
        ref[:, r] = p @ kd[s_]
        for h_ in range(2):
            msk = (s_ // 128) % 2 == h_
            sch = sc[:, msk]
            mh = sch.max(-1).values
            eh = torch.exp((sch - mh[:, None]) * SCALE)
            host_parts[h_]["m"][r], host_parts[h_]["l"][r] = mh, eh.sum(-1)
            host_parts[h_]["o"][:, r] = (eh / eh.sum(-1, keepdim=True)) @ kd[s_][msk]
    f_dev = one(outs["full"]).double().reshape(H, Cr, D)
    s_dev = one(outs["split"]).double().reshape(H, Cr, D)
    dparts = []
    for o_, st_ in outs["parts"]:
        o_h = one(o_).double().reshape(H, Cr, D)
        st_h = one(st_).double().reshape(Cr, 2 * H)
        dparts.append(dict(o=o_h, m=st_h[:, :H], l=st_h[:, H:]))
    Mh = torch.maximum(dparts[0]["m"], dparts[1]["m"])
    eh = [torch.exp((dp["m"] - Mh) * SCALE) * dp["l"] for dp in dparts]
    wz = [e / (eh[0] + eh[1]) for e in eh]  # [C, H]
    host_merge = sum(wz[h_].T[:, :, None] * dparts[h_]["o"] for h_ in range(2))
    eh1 = [torch.exp(dp["m"] - Mh) * dp["l"] for dp in dparts]  # m taken as ALREADY scaled (no x scale)
    wz1 = [e / (eh1[0] + eh1[1]) for e in eh1]
    host_merge_m_scaled = sum(wz1[h_].T[:, :, None] * dparts[h_]["o"] for h_ in range(2))
    row["m_dev_over_host_m"] = float("%.5g" % float((dparts[0]["m"] / host_parts[0]["m"]).median()))
    if args.get("dump"):  # raw device stats + the host's per-half m / l for an offline layout check
        torch.save(dict(dev_st=[one(st_).clone() for _, st_ in outs["parts"]], host=[{k: v.clone() for k, v in hp.items() if k != "o"} for hp in host_parts],
                        dev_o=[one(o_)[:, :4].clone() for o_, _ in outs["parts"]], host_o=[hp["o"][:4].clone() for hp in host_parts],
                        H=H, C=Cr, scale=SCALE), args["dump"])  # fmt: skip
    row["hostmerge_m_scaled_vs_ref"] = stat(host_merge_m_scaled, ref)
    row.update(full_vs_ref=stat(f_dev, ref), split_vs_ref=stat(s_dev, ref), hostmerge_vs_ref=stat(host_merge, ref),
               split_vs_full=stat(s_dev, f_dev), hostmerge_vs_split=stat(host_merge, s_dev),
               stats=[dict(m=stat(dparts[h_]["m"], host_parts[h_]["m"]), l=stat(dparts[h_]["l"], host_parts[h_]["l"]),
                           l_rel_maxerr=float("%.4g" % float(((dparts[h_]["l"] - host_parts[h_]["l"]).abs() / host_parts[h_]["l"]).max())),
                           o=stat(dparts[h_]["o"], host_parts[h_]["o"])) for h_ in range(2)],
               status="mock" if MOCK else "pass")  # fmt: skip
    log({k: v for k, v in row.items()})
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-4000:]
    log(f"FAILED:\n{err}")
    row.update(status="fail", error=err)
row["merge_scale"] = MSC
RESULT = dict(test="split-K sparse attention: full vs stats halves + device merge vs host merge vs fp64", tag=tag,
              wall_s=round(time.time() - t0, 1), **row)  # fmt: skip
