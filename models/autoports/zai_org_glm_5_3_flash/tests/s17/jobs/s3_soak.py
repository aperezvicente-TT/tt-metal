# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; an UNWATCHED process after the watched one — the stage's 30-s soak): Stage 17 — the speculative step
trace replayed back-to-back for >= `seconds` in rounds: per round the same prompt is teacher-forced (plain trace), the
bootstrap replay drafts, `per_round` spec-step replays run with no host work (the device feeds tokens / positions), then
one record read. Every round must end on the SAME record (n, c, a, drafts): the speculative loop is deterministic
across replays. ARGS: layers, b, k, narrow, prompt ("aime24_00" | "g4096:N"), seconds (30), per_round (200), profile, env
({knob: value}), max_seq_len (2048), tag."""
import importlib
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 4)), int(args.get("k", 3))
_L = str(args.get("layers", "0,1,2,3,4"))
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}  # knobs for this backend's build (e.g. the served set)
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")),
                         extra_env=XENV or None, max_seq_len=int(args.get("max_seq_len", 2048)))  # fmt: skip
sp, kv = be.spec, be._kv
io = kv.extra["io"]
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))
prompt = S.prompt_ids(str(args.get("prompt", "aime24_00")))
tr = sp.capture()["step"]
recs, replays, t_end = [], 0, time.time() + float(args.get("seconds", 30))
while (time.time() < t_end or not recs) and not C.MOCK:
    S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)  # (the MTP cache complete over the prompt)
    sp.enter(pt[0])
    for _ in range(int(args.get("per_round", 200))):
        ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    replays += int(args.get("per_round", 200))
    r = sp.read()
    recs.append((r["n"], r["c"], tuple(r["a"]), tuple(r["drafts"])))
    sp.leave()
same = len(set(recs)) == 1
S.record("tests", dict(test=f"s3-soak-k{K}", milestone="S3", status="pass" if (same or C.MOCK) else "fail", K=K, B=B,
                       layers=_L, rounds=len(recs), replays=replays, deterministic=same, last=list(recs[-1]) if recs else None,
                       seconds=float(args.get("seconds", 30)), env=XENV, tag=args.get("tag")))  # fmt: skip
RESULT = dict(rounds=len(recs), replays=replays, deterministic=same)
