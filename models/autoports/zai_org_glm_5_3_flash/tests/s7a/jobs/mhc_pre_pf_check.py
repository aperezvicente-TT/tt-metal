# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a lever A: the fused prefill mHC pre (tt/mhc_pre_pf.py) on one chip, real layer weights and golden residual
rows, against
  * the CPU oracle (tt/mhc_pre.reference_step, float64): hn, pre-derived post / comb;
  * Stage 16's prefill chain on the same device and rows (tt/prefill_tp.py mhc_coeffs + mhc_pre_prefill + the
    sublayer's rms_norm + typecast, the GLM53_PF_MHC=comp / MHC_POST=fused path);
then traced timing of both (one trace each, `replays` timed replays) and determinism (two replays bitwise equal).
serve.py script job (unit mesh). ARGS: layer (4), ts ([256, 1024]), subs (["attn", "ffn"]), replays (200), p / g /
xbuf (program knobs, default the module's), record (rows -> $GLM53_OUT_DIR/{pcc,perf,tests}.jsonl), tag.
GLM53_MOCK=1: logic dry run (no asserts; the mock never runs kernels)."""

from __future__ import annotations

import json
import os
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
import importlib  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf as _mpf  # noqa: E402

MhcPrePf = importlib.reload(_mpf).MhcPrePf  # a standing server without a git tree never reloads: the module on disk
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
    FdTileCacheSource,
    OptimizedCompositeHyperConnection,
    OptimizedDecoderConfig,
)
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import CompositeHyperConnection
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import _ckc_fp32, mhc_coeffs, mhc_pre_prefill

A = globals().get("ARGS", {}) or {}
mesh = globals().get("MESH")
MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(A.get("layer", 4))
TS = [int(t) for t in A.get("ts", [256, 1024])]
SUBS = list(A.get("subs", ["attn", "ffn"]))
REPLAYS = int(A.get("replays", 200))
# variants: program knob sets {p, g, xbuf, fpu} (each one program per sublayer; the chain is timed once per (sub, t))
VARIANTS = [dict(members=v.get("p"), max_groups=v.get("g"), xbuf=v.get("xbuf"), fpu=v.get("fpu"), skip=int(v.get("skip", 0)))
            for v in A.get("variants", [{}])]  # fmt: skip
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
TAG = A.get("tag", "mhc-pre-pf-check")
REC = bool(A.get("record", False))
NORM = {"attn": "input_layernorm.weight", "ffn": "post_attention_layernorm.weight"}
ROWS = []


def row(test, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test=test, layer=LAYER, tag=TAG,
             chip=os.environ.get("GLM53_CHIP"), mock=MOCK, **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r, default=str), flush=True)
    return r


def host(t) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def maxabs(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


class Chain:
    """Stage 16's prefill pre chain of one sublayer (PrefillLayer.hc / .norm / .rms, tt/prefill_tp.py)."""

    def __init__(self, mesh, source, cfg, which):
        dev = lambda n: source.device(LAYER, n, mesh)  # noqa: E731
        hst = lambda n: source.host(LAYER, n)  # noqa: E731
        kw = dict(hc_mult=cfg.hc_mult, hidden_size=cfg.hidden_size, sinkhorn_iters=cfg.hc_sinkhorn_iters,
                  hc_eps=cfg.hc_eps, norm_eps=cfg.rms_norm_eps)  # fmt: skip
        hc = CompositeHyperConnection(mesh, dev(f"hc_{which}_fn"), hst(f"hc_{which}_base"), hst(f"hc_{which}_scale"),
                                      **kw)  # fmt: skip
        hc.__class__ = OptimizedCompositeHyperConnection
        hc.ocfg = OptimizedDecoderConfig()
        self.hc = hc
        w = dev(NORM[which])
        self.norm = ttnn.reshape(w, (1, 1, 1, w.shape[-1]))
        self.eps = cfg.rms_norm_eps
        self.ckc = _ckc_fp32(mesh)

    def __call__(self, x):
        pre, post, comb = mhc_coeffs(self.hc, x)
        h = mhc_pre_prefill(x, pre)
        ttnn.deallocate(pre)
        hn = ttnn.rms_norm(h, weight=self.norm, epsilon=self.eps, compute_kernel_config=self.ckc)
        ttnn.deallocate(h)
        return ttnn.typecast(hn, ttnn.bfloat16), post, comb


def timed(tr, fn, n):
    out = tr.run(fn)
    if not tr.enabled:
        return out, None
    ttnn.synchronize_device(tr.mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(tr.mesh, tr.tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(tr.mesh)
    return out, round((time.perf_counter() - t0) / n * 1e6, 1)


CHAIN_US = {}


def case(cfg, prog, chain, sub, t, g):
    x_host = g[0, :t].reshape(t, 16384).float()
    x = ttnn.from_torch(x_host.reshape(1, 1, t, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    tr = Traced(mesh)
    tr.warm(lambda: prog(x), lambda: chain(x))
    (hn, post, comb), us_f = timed(tr, lambda: prog(x), REPLAYS)
    hn_f, post_f, comb_f = host(hn)[0, 0, :t], host(post)[0, 0, :t, :4], host(comb)[0, 0, :t, :16]
    det = None
    if tr.enabled:
        tr.replay()
        det = bool(torch.equal(host(hn)[0, 0, :t], hn_f) and torch.equal(host(post)[0, 0, :t, :4], post_f))
    tr.release()
    (hn_c, post_c, comb_c), us_c = timed(tr, lambda: chain(x), REPLAYS if (sub, t) not in CHAIN_US else 1)
    us_c = CHAIN_US.setdefault((sub, t), us_c)
    hn_c, post_c, comb_c = host(hn_c)[0, 0, :t], host(post_c)[0, 0, :t, :4], host(comb_c)[0, 0, :t, :16]
    tr.release()
    ttnn.deallocate(x)
    ref = M.reference_step(x_host, prog.host["fn_t"], prog.host["scale"], prog.host["bias"], prog.host["w"],
                           hc_eps=prog.hc_eps, x_eps=prog.x_eps, h_eps=prog.h_eps, iters=prog.iters)  # fmt: skip
    r_hn, r_post, r_comb = ref["hn"], ref["post"], ref["comb"].reshape(t, 16)
    res = dict(sublayer=sub, t=t, p=prog.p, cpc=prog.cpc, g=min(prog.max_groups, t // 32), xbuf=prog.xbuf, fpu=prog.fpu,
               skip=prog.skip,
               pcc_hn_oracle=pcc(hn_f, r_hn), pcc_hn_chain=pcc(hn_f, hn_c), pcc_chain_oracle=pcc(hn_c, r_hn),
               maxabs_hn_oracle=maxabs(hn_f, r_hn), maxabs_chain_oracle=maxabs(hn_c, r_hn),
               maxabs_post_oracle=maxabs(post_f, r_post), maxabs_post_chain=maxabs(post_f, post_c),
               maxabs_comb_oracle=maxabs(comb_f, r_comb), maxabs_comb_chain=maxabs(comb_f, comb_c),
               maxabs_chain_post_oracle=maxabs(post_c, r_post), maxabs_chain_comb_oracle=maxabs(comb_c, r_comb),
               deterministic=det, us_fused=us_f, us_chain=us_c, replays=REPLAYS, mode=tr.mode)  # fmt: skip
    if not MOCK and not prog.skip:
        # the gate: PCC >= 0.9999 vs the oracle, hn maxabs no worse than 2x the chain's (scale-blind PCC), post / comb
        # within 1e-3 of the oracle or no worse than the chain's own error (the PF_FPU tf32 sums), deterministic
        ok = (res["pcc_hn_oracle"] >= 0.9999 and res["maxabs_hn_oracle"] <= 2 * res["maxabs_chain_oracle"] + 1e-2
              and res["maxabs_post_oracle"] <= max(1e-3, res["maxabs_chain_post_oracle"])
              and res["maxabs_comb_oracle"] <= max(1e-3, res["maxabs_chain_comb_oracle"]) and det is not False)  # fmt: skip
        res["status"] = "pass" if ok else "fail"
    return row("mhc_pre_pf", **res)


def main():
    cfg = hf_config()
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    g = golden(LAYER, 2048)["streams_in"].float()  # [1, 2048, 4, 4096]
    for sub in SUBS:
        chain = Chain(mesh, source, cfg, sub)
        for kw in VARIANTS:
            prog = MhcPrePf.from_source(mesh, source, cfg, LAYER, sub, NORM[sub], **kw)
            row("mhc_pre_pf_l1", sublayer=sub, fpu=prog.fpu, p=prog.p, xbuf=prog.xbuf, **prog.l1_bytes())
            for t in TS:
                try:
                    case(cfg, prog, chain, sub, t, g)
                except Exception:
                    traceback.print_exc()
                    row("mhc_pre_pf", sublayer=sub, t=t, fpu=prog.fpu, status="fail",
                        error=traceback.format_exc()[-600:])  # fmt: skip
    if REC and not MOCK:
        for name in ("pcc.jsonl", "tests.jsonl"):
            with open(os.path.join(OUT, name), "a") as f:
                for r in ROWS:
                    f.write(json.dumps(r, default=str) + "\n")
    return [r for r in ROWS if r.get("status") == "fail"]


fails = main()
RESULT = None if not fails else [{k: r.get(k) for k in ("sublayer", "t", "status", "error")} for r in fails]
