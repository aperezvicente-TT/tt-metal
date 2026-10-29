# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a kda_rec check (tt/kda_rec.py: the KDA decode recurrence + sigmoid-gated o RMSNorm as ONE program, the
megakernel's P4 phase) vs the op chain it replaces (fused_recurrent_kda in place + the decoder's gated norm, i.e.
OptimizedKDA._recurrence_and_output without o_proj) and vs the fp64 oracle (kda_rec.reference_step), dense_kda L0's
o_norm, B = 1 / 2 / 4 / 8 / 32, random q / k / v / g / beta / gate_out / state; traced (every case warmed before any
capture; the state is restored inside the traced call). State bitwise vs fused_recurrent_kda, on PCC vs the chain and
vs the oracle, determinism, traced time of each. Rows -> pcc.jsonl / tests.jsonl (watcher-tagged)."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt.kda_rec import KdaRec, reference_step

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "0"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
BATCHES = [int(b) for b in os.environ.get("GLM53_REC_BATCHES", "1,2,4,8,32").split(",")]
REPLAYS = 20
ROWS = []
_FLUSHED = 0


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    global _FLUSHED
    new, _FLUSHED = ROWS[_FLUSHED:], len(ROWS)
    with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
        for r in new:
            f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "b") if k in r}) + "\n")


def host(t):
    return ttnn.to_torch(t).float()


def timed(tr):
    if not tr.enabled:
        return None
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return sorted(d)[len(d) // 2]


def main():
    cfg = hf_config()
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    )
    tr = Traced(mesh)
    try:
        dec = load_optimized_decoder(cfg, LAYER, mesh)
        kda = dec.layer.attn
        DEBUG = os.environ.get("GLM53_REC_DEBUG") == "1"  # debug taps: the pre-norm o rows vs the oracle's o
        rec = KdaRec(kda, opt_size=os.environ.get("GLM53_REC_OPT_SIZE") == "1", debug=DEBUG)
        # the reference is the OP CHAIN (fused_recurrent_kda + the gated-norm ops): the decoder's own kda_rec (default
        # ON since 17:12Z for B <= 2) is unbound, or the chain would route through the program under test
        kda.kda_rec = None
        kda._lin = lambda role, x, *a, **k: x  # the chain's _recurrence_and_output up to (not incl.) o_proj
        row("kda_rec_setup", l1_bytes=rec.l1_bytes(), cores=len(rec.core_list), opt_size=("MK_OPT_SIZE", "1") in rec.defines,
            status="info")  # fmt: skip
        dev = lambda t, mem=ttnn.L1_MEMORY_CONFIG: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                                                   device=mesh, memory_config=mem)  # fmt: skip
        g0 = torch.Generator().manual_seed(7)
        cases = {}
        for b in BATCHES:
            r_ = lambda *s: torch.randn(*s, generator=g0)
            h = dict(q=r_(b, 8192), k=r_(b, 8192), v=r_(b, 8192) * 0.5,
                     g=-torch.rand(b, 8192, generator=g0) * 2.0, beta=torch.sigmoid(r_(b, 64)), go=r_(b, 8192),
                     s=r_(b, 64, 128, 128) * 0.1)  # fmt: skip
            t = {n: dev(h[n].reshape(1, 1, b, -1)) for n in ("q", "k", "v", "g", "beta", "go")}
            s0 = dev(h["s"], ttnn.DRAM_MEMORY_CONFIG)
            st_a, st_b = kda.allocate_state(b), kda.allocate_state(b)

            def run_rec(t=t, s0=s0, st=st_a, b=b):
                ttnn.copy(s0, st.S)
                return rec(t["q"], t["k"], t["v"], t["g"], t["beta"], t["go"], st.S, rows=b)

            def run_chain(t=t, s0=s0, st=st_b, b=b):
                ttnn.copy(s0, st.S)
                return kda._recurrence_and_output(t["q"], t["k"], t["v"], t["g"], t["beta"], t["go"], st, b)

            cases[b] = (h, run_rec, run_chain, st_a, st_b)
        tr.warm(*[f for c in cases.values() for f in c[1:3]])  # every program compiled before any capture
        for b, (h, run_rec, run_chain, st_a, st_b) in cases.items():
            try:
                on = tr.run(run_rec)
                got_on, got_s = host(on).reshape(b, 64, 128), host(st_a.S)
                got_o = host(rec.last_tap).reshape(b, 64, 128) if DEBUG else None
                us_rec = timed(tr)
                tr.replay()
                det = torch.equal(host(on).reshape(b, 64, 128), got_on) and torch.equal(host(st_a.S), got_s)
                tr.release()
                on2 = tr.run(run_chain)
                want_on, want_s = host(on2).reshape(b, 64, 128), host(st_b.S)
                us_chain = timed(tr)
                tr.release()
                ref = reference_step(h["q"].reshape(b, 64, 128), h["k"].reshape(b, 64, 128),
                                     h["v"].reshape(b, 64, 128), h["g"].reshape(b, 64, 128), h["beta"],
                                     h["go"].reshape(b, 64, 128), h["s"], rec.host_o_norm, kda.eps)  # fmt: skip
                r = dict(b=b, execution=tr.mode, rec_us=us_rec, chain_us=us_chain, deterministic=det,
                         opt_size=("MK_OPT_SIZE", "1") in rec.defines,
                         state_bitwise_vs_op=bool(torch.equal(got_s, want_s)),
                         state_max_abs_vs_op=float((got_s - want_s).abs().max()),
                         state_pcc_vs_oracle=pcc(got_s.reshape(-1), ref["state"].reshape(-1)),
                         on_pcc_vs_chain=pcc(got_on.reshape(-1), want_on.reshape(-1)),
                         on_pcc_vs_oracle=pcc(got_on.reshape(-1), ref["on"].reshape(-1)),
                         chain_on_pcc_vs_oracle=pcc(want_on.reshape(-1), ref["on"].reshape(-1)),
                         on_max_abs_vs_chain=float((got_on - want_on).abs().max()))  # fmt: skip
                if DEBUG:  # tap: the pre-norm o (the recurrence output) vs the fp64 oracle
                    r["tap_o_pcc_vs_oracle"] = pcc(got_o.reshape(-1), ref["o"].reshape(-1))
                    r["tap_o_max_rel_vs_oracle"] = float((got_o.double() - ref["o"]).abs().max() / ref["o"].abs().max())
                ok = (
                    det
                    and r["state_bitwise_vs_op"]
                    and r["on_pcc_vs_chain"] >= 0.9999
                    and r["on_pcc_vs_oracle"] >= 0.9999
                    and (not DEBUG or r["tap_o_pcc_vs_oracle"] >= 0.99999)
                )
                r["status"] = "record" if MOCK else ("pass" if ok else "fail")
                row("kda_rec_vs_chain", **r)
            except Exception:
                traceback.print_exc()
                err = traceback.format_exc()
                row("kda_rec_vs_chain", b=b, status="fail", error=err[-400:])
                if "TIMEOUT" in err or "hang" in err:
                    flush()
                    print("DEV_KDA_REC_ABORT device timeout", flush=True)
                    os._exit(3)
                tr.release()
    finally:
        flush()
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_KDA_REC_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
