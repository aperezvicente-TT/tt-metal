# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 14b's pair or mesh server): Stage 14b TTFT ATTRIBUTION (program 2026-10-22 03:14Z) -- the
generator path of a B=1 prompt on the resident serving backend (m5common.serving: built here when the server holds
none) with Stage 16's batched TP prefill attached (svc_attach), split per chunk:
  plan      plan_prefill -> chunks (start, live rows) and the bucket each replays (rows computed), head / head-less trace;
  replay    each chunk key's trace replayed ALONE (pinned blocking replays: what one request waits for);
  host      StackBackend._prefill_chunk_tp + TPPrefill.prefill host steps (GLM53_PF_HOST_TIMING: decode-row reads,
            sampler params, cur_pos, KDA slot state ops, write_inputs, replay enqueue, first-token wait (holds the
            device replay), stale-input restores), per request;
  decode    the first decode step + its sample readback = generate(n, 2) - generate(n, 1);
  split     TTFT - sum of the chunks' replays = the generator's host time.
No soak (pinned replays). The service is restored afterwards (svc_attach.restore).
ARGS: b (32), layers ([0..7]; "all"), native (true), rev ("sv24"), ns ([2048]), reps (3), replays (20), knobs ({}),
buckets ("128,512,2048"), record (true), tag."""
import datetime as dt
import gc
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
S16 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(S16), "s14b"))
sys.path.insert(0, S16)
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402
import svc_attach  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.generator import plan_prefill  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
cfg = hf_config()
_L = args.get("layers", list(range(8)))
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, NATIVE, REV = int(args.get("b", 32)), bool(args.get("native", True)), args.get("rev", "sv24")
NS = [int(v) for v in args.get("ns", [2048])]
REPS, NREP = int(args.get("reps", 3)), int(args.get("replays", 20))
TP = int(mesh.get_num_devices())


def log(*a):
    print(f"S14B-TTFT-ATTR {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


def prompt(n):
    src = n if n in (128, 512, 2048, 4096) else 4096
    p = os.path.join(GOLDENS, "prompts", f"{src}.pt")
    return [int(v) for v in torch.load(p, map_location="cpu", weights_only=True).reshape(-1)[:n]]


def med(v):
    v = sorted(v)
    return round(v[len(v) // 2], 2) if v else None


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=REV, log=log)
att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=str(args.get("buckets", "128,512,2048")))
be, gen = att.be, att.gen
greedy = SamplingParams()
results = []
pf = None
os.environ["GLM53_PF_HOST_TIMING"] = "1"
try:
    pf = att.attach()
    log(f"attached: key {key[:4]}, prefill traces {sorted(map(str, pf.traces))}, warm-up {att.warm_s} s")
    for n in NS:
        if n > gen.supported_context:
            log(f"n {n}: past the served context {gen.supported_context}, skipped")
            results.append(dict(n=n, skipped=f"supported_context {gen.supported_context}"))
            continue
        toks = prompt(n)
        plans = plan_prefill(n, 0, be.prefill_chunk_tokens, be.prefill_align)
        chunks = []
        for i, pl in enumerate(plans):
            c, t = pf.plan(pl.start, pl.n_valid)
            last = i == len(plans) - 1
            nohead = (not last) and bool(getattr(pf, "nohead", False)) and (c, t) in getattr(pf, "traces_nh", {})
            chunks.append(dict(start=pl.start, live_rows=pl.n_valid, physical=pl.physical, bucket_rows=c, t_win=t,
                               key=[c, t], trace="head-less" if nohead else "head"))  # fmt: skip
        gen.generate(toks, 1, sampling=greedy, stop_on_eos=False)  # warm (untimed)
        be.counters.pop("pf_host_ms", None)
        pf.counters.pop("host_ms", None)
        c0 = {k: v for k, v in pf.counters.items() if isinstance(v, (int, float))}
        g0 = {k: v for k, v in gen.counters.items() if isinstance(v, (int, float))}
        walls, walls2, first = [], [], None
        for _ in range(REPS):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            out = gen.generate(toks, 1, sampling=greedy, stop_on_eos=False)
            walls.append((time.perf_counter() - t0) * 1e3)
            first = int(out[0])
        host = {k: round(v / REPS, 3) for k, v in be.counters.get("pf_host_ms", {}).items() if k != "n"}
        host.update({f"tpp_{k}": round(v / REPS, 3) for k, v in pf.counters.get("host_ms", {}).items()})
        dc = {k: (v - c0.get(k, 0)) / REPS for k, v in pf.counters.items() if isinstance(v, (int, float))}
        dg = {k: (v - g0.get(k, 0)) / REPS for k, v in gen.counters.items() if isinstance(v, (int, float))}
        # the first decode step + its readback: generate(m, 2) - generate(m, 1) on m = n - 1 tokens when n fills the
        # served context (n + 1 cache positions needed; same chunk plan / buckets for n - 1)
        m = n if n + 1 <= gen.supported_context else n - 1
        walls1m = walls if m == n else []
        for _ in range(REPS if m != n else 0):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            gen.generate(toks[:m], 1, sampling=greedy, stop_on_eos=False)
            walls1m.append((time.perf_counter() - t0) * 1e3)
        for _ in range(REPS):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            gen.generate(toks[:m], 2, sampling=greedy, stop_on_eos=False)
            walls2.append((time.perf_counter() - t0) * 1e3)
        # each chunk key's trace alone: pinned blocking replays (no soak)
        rep = {}
        for ch in chunks:
            k = tuple(ch["key"])
            if k in rep:
                continue
            rs = []
            for _ in range(NREP if not C.MOCK else 1):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                pf.replay(k[0], blocking=True, t=k[1], head=ch["trace"] == "head")
                rs.append((time.perf_counter() - t0) * 1e3)
            rep[k] = med(rs)
        for ch in chunks:
            ch["replay_ms"] = rep[tuple(ch["key"])]
        ttft = med(walls)
        sum_rep = round(sum(ch["replay_ms"] or 0 for ch in chunks), 2)
        row = dict(
            test=f"TTFT attribution B=1 prompt {n} tp{TP} layers {len(ids)} on 14b's resident serving backend "
            f"{key[:2] + key[3:4]} + Stage 16 TP prefill (Glm53Generator.generate; per chunk: bucket, replay alone, "
            f"host steps; first decode step)",
            milestone="M5c-ttft-attr" if len(ids) == cfg.num_hidden_layers else "M5c-ttft-attr-prefix",
            status="info", traced=dc.get("eager", 0) == 0, n=n, chunks=chunks, ttft_ms=ttft, ttft_min_ms=round(min(walls), 2),
            replays_sum_ms=sum_rep, host_ms=round(ttft - sum_rep, 2) if ttft is not None else None,
            host_steps_ms=host, decode_probe_prompt=m, gen1_ms=med(walls1m), gen2_ms=med(walls2),
            first_decode_step_ms=round(med(walls2) - med(walls1m), 2),
            per_request_counters=dict(tpp={k: round(v, 3) for k, v in dc.items() if v},
                                      gen={k: round(v, 3) for k, v in dg.items() if v}),
            first_token=first, reps=REPS, replays=NREP, knobs=att.knobs, buckets=att.buckets, rev=REV,
            tag=args.get("tag", f"m5c-ttft-attr-tp{TP}"),
        )  # fmt: skip
        log(row)
        if args.get("record", True) and not C.MOCK:
            C.record("perf", row)
        results.append(row)
finally:
    os.environ.pop("GLM53_PF_HOST_TIMING", None)
    pf = None
    gc.collect()
    log(f"restored: decode-only warm-up {att.restore()} s")
RESULT = results  # noqa: F841
