# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P2 — where a prefill layer's time goes, per phase, without Tracy: the layer's
op sequence (tt/prefill_tp.py TPPrefill.layer) cut after each phase; every PREFIX is captured as its own trace and
replayed (blocking, median of `reps`); a phase's time = prefix(k) - prefix(k - 1). Phases: attn mHC pre (+ norm),
gather, attention, reduce-scatter, attn mHC post, ffn mHC pre (+ norm, router), gather(s), FFN (experts / MLP),
reduce-scatter, ffn mHC post. Inputs: the CPU golden's streams_in of each checked layer (real activations, real routing).

ARGS: layers ([0..4]), check ([0, 3, 4]), n (2048), reps (5), moe, moe_v2, kda_scan, mhc, mhc_post, the P3 attention
knobs (s16common.set_attn: mla_form, mha_sdpa, sdpa, mm_nq), tag, record."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16, FP32, kda_prefill_tp  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
CHECK = [int(v) for v in args.get("check", [0, 3, 4])]
N = int(args.get("n", 2048))
REPS = int(args.get("reps", 5))
tag = args.get("tag", "p2-phases")
TPN = int(mesh.get_num_devices())
PHASES = ["mhc_pre_attn", "gather_attn", "attention", "rs_attn", "mhc_post_attn", "mhc_pre_ffn", "gather_ffn", "ffn",
          "rs_ffn", "mhc_post_ffn"]  # fmt: skip

os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
S.set_attn(args)
att, out0 = None, C.OUT
if args.get("serving"):  # inside 14b's shared mesh service: the resident backend, rows into Stage 16's evidence
    import svc_attach

    knobs = {k: os.environ[k] for k in ("GLM53_PF_MHC", "GLM53_PF_MHC_POST", "GLM53_PF_MLA_FORM", "GLM53_PF_MHA_SDPA",
                                        "GLM53_PF_SDPA", "GLM53_PF_MM_NQ") if os.environ.get(k)}  # fmt: skip
    knobs.update(dict(args.get("knobs", {})))
    att = svc_attach.Attached(ctx, knobs=knobs, buckets=str(N))
    pf = att.attach()
    kv = pf.kv
    LAYERS = [int(e["L"]) for e in pf.b.seq]
    C.OUT = os.environ.get("GLM53_S16_EVIDENCE", S.EVIDENCE)
else:
    ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,))
    pf, kv = ent["pf"], ent["kv"]
pf.moe_impl = args.get("moe", pf.moe_impl)
S.set_moe_v2(pf, bool(args.get("moe_v2", False)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)


def run_prefix(li, X, k):
    """the layer's first k phases (outputs dropped)"""
    pl = pf.layers[li]
    sm = pf.mhc_mode == "sm"
    live = []
    h, ctx = pf.mhc_pre(pl, "attn", X)
    hn = pl.rms(h, "attn", BF16)
    ttnn.deallocate(h)
    live += [ctx[0], ctx[1]] + ([] if sm else list(ctx[2]))
    out = [hn]
    if k >= 2:
        out = [pf.gather(hn, sp)]
    if k >= 3:
        a = pl.mla(out[0], kv.extra["latent"][li], ci, fresh=pf.mla_form == "mha") if pl.kind == SPARSE_DSA else kda_prefill_tp(
            pl.kda, out[0], pl.scratch, ci.valid_end, ci.valid_end_f, scan=pf.kda_scan)  # fmt: skip
        out = [a]
    if k >= 4:
        out = [pf.reduce(out[0], sp)]
    if k >= 5:
        s1 = pf.mhc_post_streams(pl, "attn", out[0], ctx)
        x1 = s1 if sm else ttnn.concat(s1, dim=-1)
        live += [s1] if sm else list(s1)
        out = [] if sm else [x1]
    if k >= 6:
        h2, ctx2 = pf.mhc_pre(pl, "ffn", x1, None if sm else s1)
        live += [ctx2[0], ctx2[1]]
        if pl.kind == DENSE_KDA:
            parts = [pl.rms(h2, "ffn", BF16)]
        else:
            x32 = pl.rms(h2, "ffn", FP32)
            idx, scores = pl.moe.router(x32, decode=False)
            parts = [ttnn.typecast(x32, BF16), idx, scores]
            if pf.moe_impl == "ep":
                kk = int(idx.shape[-1])
                sel = ttnn.typecast(ttnn.gather(scores, dim=-1, index=idx), FP32)
                parts = [parts[0], ttnn.concat([ttnn.typecast(idx, FP32), sel], dim=-1)]
        out += parts
    if k >= 7:
        g = [pf.gather(parts[0], sp)]
        if pl.kind != DENSE_KDA and pf.moe_impl == "ep":
            pk = pf.gather(parts[1], sp)
            rows = int(pk.shape[-2])
            g += [ttnn.typecast(ttnn.slice(pk, (0, 0, 0, 0), (1, 1, rows, kk)), ttnn.uint16),
                  ttnn.slice(pk, (0, 0, 0, kk), (1, 1, rows, 2 * kk))]  # fmt: skip
        elif pl.kind != DENSE_KDA and sp:
            g += [ttnn.typecast(pf.gather(ttnn.typecast(parts[1], FP32), sp), ttnn.uint16), pf.gather(parts[2], sp)]
        elif pl.kind != DENSE_KDA:
            g += parts[1:]
        out += g
        fin = g
    if k >= 8:
        if pl.kind == DENSE_KDA:
            m = PF.swiglu_prefill(pl.mlp, fin[0])
        else:
            if pf.moe_impl == "ep":
                if getattr(pl, "epmoe", None) is None:
                    pl.epmoe = PF.EpMoe(pf, pl.moe)
                routed = pl.epmoe(*fin)
            else:
                routed = pf.moe_local(pl, *fin)
            m = ttnn.add(routed, PF.swiglu_prefill(pl.shared, fin[0]))
        out += [m]
    if k >= 9:
        m = pf.reduce(m, sp)
        out += [m]
    if k >= 10:
        s2 = pf.mhc_post_streams(pl, "ffn", m, ctx2)
        out += [s2] if sm else list(s2)
    for t_ in out + live:  # every tensor listed once
        ttnn.deallocate(t_)


results = []
try:
  for L in CHECK:
      li = LAYERS.index(L)
      pl = pf.layers[li]
      x_in, _ = S.golden_layer(L, N)
      pad = torch.zeros(c, 4, 4096)
      pad[:N] = x_in.reshape(N, 4, 4096).float()
      X = pf.streams_to_device(pad, sp)
      prefix_us, prefix_replays = [0.0], []
      for k in range(1, len(PHASES) + 1):
          run_prefix(li, X, k)  # compile launch
          ttnn.synchronize_device(mesh)
          if C.MOCK:
              prefix_us.append(0.0)
              continue
          tid = ttnn.begin_trace_capture(mesh, cq_id=0)
          try:
              run_prefix(li, X, k)
          finally:
              ttnn.end_trace_capture(mesh, tid, cq_id=0)
          st = S.replay_stats(mesh, tid, replays=args.get("replays"))  # >= 2,000 replays (program 03:20Z)
          ttnn.release_trace(mesh, tid)
          prefix_us.append(st["median_us"])
          prefix_replays.append(st["replays"])
      phases = {PHASES[i]: round(prefix_us[i + 1] - prefix_us[i], 1) for i in range(len(PHASES))}
      row = dict(test=f"p2 prefill layer phases L{L} {pl.kind} (traced prefixes, n={N}) tp{TPN}", milestone=S.MILESTONE,
                 status="info", traced=True, layer=L, kind=pl.kind, n=N, tp=TPN, mode="sp" if sp else "rep",
                 layer_us=prefix_us[-1], phases_us=phases, prefix_us=prefix_us[1:], replays=prefix_replays,
                 moe=pf.moe_impl,
                 kda_scan=pf.kda_scan, mla_form=pf.mla_form, mha_sdpa=os.environ.get("GLM53_PF_MHA_SDPA", ""),
                 mm_nq=os.environ.get("GLM53_PF_MM_NQ", "4"), moe_v2=os.environ.get("GLM53_PF_MOE_V2", "0"), tag=tag)  # fmt: skip
      S.log(row)
      if args.get("record", True):
          C.record("perf", row)
      results.append(row)
      ttnn.deallocate(X)
finally:
    C.OUT = out0
    if att is not None:
        pf = kv = pl = None  # the job's own references: restore() must be able to free every prefill buffer
        S.log(f"restored the decode-only service config (warm-up {att.restore()} s)")
RESULT = results

