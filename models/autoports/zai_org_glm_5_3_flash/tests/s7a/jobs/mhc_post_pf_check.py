# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a lever A, post side: tt/mhc_pre_pf.mhc_post_prefill_7a (one SFPU pass per output tile) against Stage 16's
mhc_post_prefill (tt/prefill_tp.py) and the float64 host mix, on one chip: x = golden L4 residual rows, post / comb =
the fused pre program's coefficients of those rows (real weights), y = a bf16 sublayer output stand-in (seeded normal
at the residual's scale). Then traced timing of both and determinism.
serve.py script job (unit mesh). ARGS: layer (4), ts ([256, 1024, 2048]), replays (200), record, tag.
GLM53_MOCK=1: logic dry run."""

from __future__ import annotations

import json
import os
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
import importlib  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre_pf as _mpf  # noqa: E402

_mpf = importlib.reload(_mpf)  # a standing server without a git tree never reloads: take the module on disk (MPF7_REV)
MhcPrePf, mhc_post_prefill_7a = _mpf.MhcPrePf, _mpf.mhc_post_prefill_7a
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import mhc_post_prefill

A = globals().get("ARGS", {}) or {}
mesh = globals().get("MESH")
MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(A.get("layer", 4))
TS = [int(t) for t in A.get("ts", [256, 1024, 2048])]
REPLAYS = int(A.get("replays", 200))
OUT = os.environ.get("GLM53_OUT_DIR", "/tmp")
TAG = A.get("tag", "mhc-post-pf7a-check")
REC = bool(A.get("record", False))
ROWS = []


def row(test, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), test=test, layer=LAYER, tag=TAG,
             chip=os.environ.get("GLM53_CHIP"), mock=MOCK, **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r, default=str), flush=True)
    return r


def host(t) -> torch.Tensor:
    return ttnn.to_torch(t).float()


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


BATCHES = [int(v) for v in A.get("batches", [])]
DBG = [int(v) for v in A.get("dbg", [])]  # bisect variants (mhc_post_prefill_7a dbg): 1 out = x_0, 2 / 3 out_s = post_s * y


def bisect(pre, t, g):
    x_host = g[0, :t].reshape(t, 16384).float()
    put = lambda v, dt: ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    x = put(x_host.reshape(1, 1, t, 16384), ttnn.float32)
    gen = torch.Generator().manual_seed(1234 + t)
    y_host = (torch.randn(t, 4096, generator=gen) * float(x_host.std())).to(torch.bfloat16).float()
    y = put(y_host.reshape(1, 1, t, 4096), ttnn.bfloat16)
    _, post, comb = pre(x)
    p_h = host(post)[0, 0, :t, :4].double()
    tr = Traced(mesh)
    tr.warm(*[(lambda d_=d: mhc_post_prefill_7a(y, x, post, comb, dbg=d_)) for d in DBG],
            *[(lambda b_=b: mhc_post_prefill_7a(y, x, post, comb, batch=b_)) for b in BATCHES],
            lambda: mhc_post_prefill(y, x, post, comb))  # fmt: skip (every program compiled before the first capture)
    us = {}
    for d in DBG:
        _, us[d] = timed(tr, lambda d_=d: mhc_post_prefill_7a(y, x, post, comb, dbg=d_), REPLAYS)
        tr.release()
    for b_ in BATCHES:  # the full mix at other reader / writer batch sizes (+ its numerics)
        o_b, us[f"full_b{b_}"] = timed(tr, lambda: mhc_post_prefill_7a(y, x, post, comb, batch=b_), REPLAYS)
        us[f"full_b{b_}_equal_b_default"] = bool(torch.equal(host(o_b), host(mhc_post_prefill_7a(y, x, post, comb))))
        tr.release()
    _, us16 = timed(tr, lambda: mhc_post_prefill(y, x, post, comb), REPLAYS)
    tr.release()
    row("mhc_post_pf7a_timing", t=t, us={str(k): v for k, v in us.items()}, us_16=us16, replays=REPLAYS)
    for d in [d for d in DBG if d != 5]:
        o = host(mhc_post_prefill_7a(y, x, post, comb, dbg=d))[0, 0, :t].double().reshape(t, 4, 4096)
        if d == 1:
            want = x_host.double().reshape(t, 4, 4096)[:, 0:1].expand(t, 4, 4096)
        elif d == 0:  # the full mix
            c_h = host(comb)[0, 0, :t, :16].double()
            xs = x_host.double().reshape(t, 4, 4096)
            want = torch.stack([p_h[:, s : s + 1] * y_host.double() + sum(c_h[:, 4 * i + s : 4 * i + s + 1] * xs[:, i]
                                                                          for i in range(4)) for s in range(4)], 1)  # fmt: skip
        else:
            want = torch.stack([p_h[:, s : s + 1] * y_host.double() for s in range(4)], 1)
        err = (o - want).abs()
        bad = (err > 1e-5 * float(want.abs().max()) + 1e-30)
        # where the wrong values sit: per stream / row / column tile (first 32 rows) of the first wrong row block
        rows = bad.any(-1).any(-1).nonzero().flatten()[:8].tolist()
        cols = bad[:32].any(0).any(0).nonzero().flatten()
        row("mhc_post_pf7a_bisect", t=t, dbg=d, maxabs=float(err.max()), want_maxabs=float(want.abs().max()),
            frac_bad=float(bad.double().mean()), bad_per_stream=[float(bad[:, s].double().mean()) for s in range(4)],
            first_bad_rows=rows, bad_cols_in_tile=sorted({int(c) % 32 for c in cols.tolist()})[:40],
            bad_col_tiles=sorted({int(c) // 32 for c in cols.tolist()})[:10],
            sample_out=o[0, 0, :6].tolist(), sample_want=want[0, 0, :6].tolist())  # fmt: skip
    for t_ in (x, y, post, comb):
        ttnn.deallocate(t_)


def case(pre, t, g):
    x_host = g[0, :t].reshape(t, 16384).float()
    put = lambda v, dt: ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    x = put(x_host.reshape(1, 1, t, 16384), ttnn.float32)
    gen = torch.Generator().manual_seed(1234 + t)
    y_host = (torch.randn(t, 4096, generator=gen) * float(x_host.std())).to(torch.bfloat16).float()
    y = put(y_host.reshape(1, 1, t, 4096), ttnn.bfloat16)
    _, post, comb = pre(x)
    p_h, c_h = host(post)[0, 0, :t, :4].double(), host(comb)[0, 0, :t, :16].double()
    xs = x_host.double().reshape(t, 4, 4096)
    ref = torch.stack([p_h[:, s : s + 1] * y_host.double() + sum(c_h[:, 4 * i + s : 4 * i + s + 1] * xs[:, i]
                                                                 for i in range(4)) for s in range(4)], 1)  # fmt: skip
    ref = ref.reshape(t, 16384)
    tr = Traced(mesh)
    tr.warm(lambda: mhc_post_prefill_7a(y, x, post, comb), lambda: mhc_post_prefill(y, x, post, comb))
    o7, us7 = timed(tr, lambda: mhc_post_prefill_7a(y, x, post, comb), REPLAYS)
    a7 = host(o7)[0, 0, :t]
    det = None
    if tr.enabled:
        tr.replay()
        det = bool(torch.equal(host(o7)[0, 0, :t], a7))
    tr.release()
    o16, us16 = timed(tr, lambda: mhc_post_prefill(y, x, post, comb), REPLAYS)
    a16 = host(o16)[0, 0, :t]
    tr.release()
    for t_ in (x, y, post, comb):
        ttnn.deallocate(t_)
    e7, e16 = (a7.double() - ref).abs(), (a16.double() - ref).abs()
    scale = float(ref.abs().max())
    res = dict(t=t, pcc_7a_oracle=pcc(a7, ref.float()), pcc_7a_16=pcc(a7, a16),
               maxabs_7a_oracle=float(e7.max()), maxabs_16_oracle=float(e16.max()),
               mean_abs_7a_oracle=float(e7.mean()), mean_abs_16_oracle=float(e16.mean()), ref_maxabs=scale,
               maxabs_7a_16=float((a7 - a16).abs().max()), bitwise_7a_16=bool(torch.equal(a7, a16)),
               deterministic=det, us_7a=us7, us_16=us16, replays=REPLAYS, mode=tr.mode)  # fmt: skip
    if not MOCK:
        # the gate: no worse than Stage 16's exact-SFPU program vs the float64 mix (FMA: one rounding instead of two)
        ok = (res["maxabs_7a_oracle"] <= 2 * res["maxabs_16_oracle"] + 1e-6 * scale and res["pcc_7a_oracle"] >= 0.99999
              and det is not False)  # fmt: skip
        res["status"] = "pass" if ok else "fail"
    return row("mhc_post_pf7a", **res)


def main():
    cfg = hf_config()
    source = FdTileCacheSource(CACHE_DIR, MANIFEST)
    g = golden(LAYER, 2048)["streams_in"].float()  # [1, 2048, 4, 4096]
    pre = MhcPrePf.from_source(mesh, source, cfg, LAYER, "attn", "input_layernorm.weight", fpu=True)
    for t in TS:
        if DBG:
            bisect(pre, t, g)
            continue
        try:
            case(pre, t, g)
        except Exception:
            traceback.print_exc()
            row("mhc_post_pf7a", t=t, status="fail", error=traceback.format_exc()[-600:])
    if REC and not MOCK:
        for name in ("pcc.jsonl", "tests.jsonl"):
            with open(os.path.join(OUT, name), "a") as f:
                for r in ROWS:
                    f.write(json.dumps(r, default=str) + "\n")
    return [r for r in ROWS if r.get("status") == "fail"]


fails = main()
RESULT = None if not fails else [{k: r.get(k) for k in ("t", "status", "error")} for r in fails]
