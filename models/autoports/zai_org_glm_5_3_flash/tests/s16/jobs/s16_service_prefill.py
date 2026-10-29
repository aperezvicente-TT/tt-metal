# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job for Stage 14b's SHARED MESH SERVICE (q_mesh, 14b #120; runs inside the 14b server: MESH, ARGS, CTX): Stage 16's
batched TP prefill rows on the RESIDENT serving backend CTX[('m5_serving', B, ids, ndev, native, rev)] -- no device open,
no weight load:
  1. attach: the generator's traces released, GLM53_STACK_PREFILL=tp + the Stage 16 knobs (ARGS knobs) set, the
     binding's preparation flags cleared, then the generator's own warm-up again (prepare_prefill: TPPrefill + one
     eager forward per bucket, BEFORE any capture; prepare_decode; capture_decode: the decode trace, then every prefill
     trace) -- the prefill's persistent buffers are allocated before every capture (no trace overlaps them);
  2. rows: TTFT through Glm53Generator.generate(prompt, 1) per prompt length (rep 0 untimed), the last chunk's replay,
     throughput (tput_n back-to-back 2,048-token requests), every request checked for an eager forward (traced flag);
  3. restore (restore=True, default): the traces released, TPPrefill dropped, GLM53_STACK_PREFILL back, the decode-only
     warm-up again -- the service is left as it was found (its decode traces re-captured).
Rows are appended to Stage 16's evidence (perf.jsonl), not the server's.

ARGS: ns ([2048, 128, 512]), reps (3), tput_n (4), knobs ({env: value}: GLM53_PF_MOE_V2, GLM53_PF_MHC_POST,
GLM53_PF_MLA_FORM, ...), buckets ("128,512,2048"), tf_ref (a .refpt: the prefill-then-decode TF after the rows),
restore (True), milestone ("P2"), tag, record (True)."""
import datetime as dt
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402
import svc_attach  # noqa: E402

C = S.C

args, ctx, mesh = ARGS, CTX, MESH  # noqa: F821
EVID = os.environ.get("GLM53_S16_EVIDENCE",
                      "/opt/glm53/kit/doc/ttnn/16-prefill-production")  # fmt: skip
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
NS = [int(v) for v in args.get("ns", [2048, 128, 512])]
REPS = int(args.get("reps", 3))
TN = int(args.get("tput_n", 4))
tag = args.get("tag", "svc-p2")
rec = bool(args.get("record", True))
def log(*a):
    print(f"S16SVC {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


def record(kind, row):
    row = dict(time_utc=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), **row)
    if rec and os.environ.get("GLM53_MOCK") != "1":
        with open(os.path.join(EVID, f"{kind}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    log(row)
    return row


def prompt(n):
    p = os.path.join(GOLDENS, "prompts", f"{n if n in (128, 512, 2048, 4096) else 4096}.pt")
    return [int(v) for v in torch.load(p, map_location="cpu", weights_only=True).reshape(-1)[:n]]


att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=str(args.get("buckets", "128,512,2048")))
KNOBS, key = att.knobs, att.key
be, gen = att.be, att.gen
results = []


try:
    pf = att.attach()
    warm_s = att.warm_s
    log(f"attached: key {key[:4]}, prefill traces {sorted(map(str, pf.traces))}, warm-up {warm_s} s, knobs {KNOBS}")
    FULL = len(be.seq) == int(be.cfg.num_hidden_layers) if hasattr(be, "cfg") else len(be.seq) == 45
    ms = args.get("milestone", "P2")  # a layer-subset backend never writes a milestone row (gate: the full model)
    # the bucket-order leak (TP8 00:32Z): is any prefill trace already unsafe right after the warm-up, before any request?
    if os.environ.get("GLM53_MOCK") != "1":
        from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

        unsafe = {}
        for k_, tid in sorted(pf.traces.items(), key=lambda kv: str(kv[0])):
            try:
                TraceAllocationTracker.verify_before_replay(mesh, tid)
                unsafe[str(k_)] = "clean"
            except RuntimeError as e:
                unsafe[str(k_)] = str(e)[:1500]
        log(f"trace safety after the warm-up: {unsafe}")
        results.append(dict(trace_safety_after_warmup=unsafe))
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import plan_prefill
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

    greedy = SamplingParams()
    soaks = {}
    for n in NS:
        toks = prompt(n)
        walls, first = [], None
        eager0 = pf.counters["eager"]
        for rep in range(REPS + 1):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            out = gen.generate(toks, 1, sampling=greedy, stop_on_eos=False)
            wall = (time.perf_counter() - t0) * 1e3
            first = int(out[0])
            if rep:
                walls.append(wall)
        last = plan_prefill(n, 0, be.prefill_chunk_tokens)[-1]
        c, t_win = pf.plan(last.start, last.n_valid)
        # the last chunk's trace alone: >= 2,000 steady-state replays (program 03:20Z device duty), once per key
        if (c, t_win) in pf.traces and not C.MOCK:
            if (c, t_win) not in soaks:
                soaks[(c, t_win)] = S.replay_stats(mesh, pf.traces[(c, t_win)], replays=args.get("replays"))
            rst = soaks[(c, t_win)]
        else:
            rs = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                pf.replay(c, blocking=True, t=t_win)
                rs.append((time.perf_counter() - t0) * 1e6)
            rst = C.stats(rs)
        g = torch.load(os.path.join(GOLDENS, "logits", f"{n}.prefill.pt"), map_location="cpu", weights_only=True) \
            if n in (128, 512, 2048, 4096) else None  # fmt: skip
        row = dict(test=f"TTFT B=1 prompt {n} tp{int(mesh.get_num_devices())} layers {len(be.seq)} on 14b's resident "
                        f"serving backend {key[:2] + key[3:4]} (Glm53Generator.generate, batched TP prefill traced per bucket)",
                   milestone=(ms if n == 2048 else f"{ms}-n{n}") if FULL else f"{ms}-prefix",
                   status="info", traced=pf.counters["eager"] == eager0, n=n, ms=round(sorted(walls)[len(walls) // 2], 2),
                   ms_min=round(min(walls), 2), last_replay_ms=round(rst["median_us"] / 1e3, 2),
                   replays=rst.get("replays", REPS), replay_key=[c, t_win],
                   tok_per_s=round(n / (sorted(walls)[len(walls) // 2] / 1e3), 1), first_token=first, requests=REPS,
                   knobs=KNOBS, warmup_s=warm_s, tag=tag)  # fmt: skip
        if g is not None:
            row["golden_argmax"] = int(torch.argmax(g["last_logits"]))
            row["first_in_golden_top5"] = first in [int(v) for v in g["prompt_top32_indices"][n - 1, :5]]
        if not row["traced"]:
            row["status"] = "fail"
        results.append(record("perf", row))
    if TN and 2048 in NS:
        toks = prompt(2048)
        eager_t = pf.counters["eager"]
        t0 = time.perf_counter()
        for _ in range(TN):
            gen.generate(toks, 1, sampling=greedy, stop_on_eos=False)
        wall = time.perf_counter() - t0
        results.append(record("perf", dict(
            test=f"prefill throughput: {TN} x 2,048-token requests back to back on the resident serving backend",
            milestone=f"{ms}-tput" if FULL else f"{ms}-prefix", status="info", traced=pf.counters["eager"] == eager_t,
            requests=TN, wall_s=round(wall, 3), tok_per_s=round(TN * 2048 / wall, 1), knobs=KNOBS, tag=tag)))  # fmt: skip
    if args.get("tf_ref"):  # prefill-then-decode teacher forcing (the Stage 6 harness) on the resident generator
        from pathlib import Path

        ws = os.environ.get("GLM53_WS", "/opt/glm53/kit")
        sys.path.insert(0, os.path.join(ws, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
        from readiness_check.run_teacher_forcing import run_teacher_forcing

        model_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        c0, r0, e0 = dict(gen.counters), dict(be.trace_report()), pf.counters["eager"]
        t0 = time.time()
        per = run_teacher_forcing(model_dir=model_dir, reference_path=Path(args["tf_ref"]), mesh_device=mesh,
                                  build_kwargs=dict(generator=gen))  # fmt: skip
        c1, r1 = dict(gen.counters), be.trace_report()
        d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
        s_ = per[0]
        tp_chunks = r1.get("prefill_chunks_tp", 0) - r0.get("prefill_chunks_tp", 0)
        traced = d.get("eager_decode_steps", 0) == 0 and d.get("decode_trace_replays", 0) > 0 and tp_chunks > 0 \
            and pf.counters["eager"] == e0  # fmt: skip
        results.append(record("pcc", dict(
            test=f"prefill-then-decode TF (Stage 6 harness) on 14b's resident serving backend, layers {len(be.seq)}, "
                 f"tp{int(mesh.get_num_devices())}", milestone=ms if FULL else f"{ms}-prefix",
            status="pass" if traced and s_["top5"] >= 0.98 else "fail", traced=traced, top1=s_["top1"], top5=s_["top5"],
            top100=s_["top100"], matches=[s_["matches_top1"], s_["matches_top5"], s_["matches_top100"], s_["total"]],
            reference=args["tf_ref"], prefill_chunks_tp=tp_chunks, wall_s=round(time.time() - t0, 1), knobs=KNOBS,
            tag=tag)))  # fmt: skip
finally:
    pf = None  # the job's own reference: restore() must be able to free every prefill buffer
    if args.get("restore", True):
        log(f"restored the decode-only service config (warm-up {att.restore()} s)")
RESULT = results
