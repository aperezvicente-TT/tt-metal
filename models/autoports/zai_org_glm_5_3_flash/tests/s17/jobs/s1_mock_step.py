# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 MOCK cluster — S1): Stage 17 — the whole speculative step built and "replayed" (mock: the
trace is the eager call sequence; no kernel runs) on the served construction: native B = 32 (+ widths 8, 16), the
speculative step of K drafts (tt/spec_mtp.py: the verify width K+1 in verify mode, accept, the MTP catch-up + chain,
gather), the bootstrap, enter / step / read / leave, then a plain step. Checks: every program builds (CB / L1 floor,
the program-cache key vs runtime-arg counts when probe_hooks "install" ran first), the traces capture, the host API
round-trips. ARGS: layers ("0,1,2,3,4" or "all"), b (32), k (3), narrow ("none" | e.g. "1,4"), steps (2), tag."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)  # a long-lived server caches modules: this job's helpers

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 32)), int(args.get("k", 3))
_L = str(args.get("layers", "0,1,2,3,4"))


def log(*a):
    print(f"S17-MOCK {C.now()}", *a, flush=True)


from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
t0 = time.time()
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")), log=log)
t1 = time.time()
sp = be.spec
assert sp is not None, "GLM53_SPEC_K: no SpecMTP"
kv = be._kv
io = kv.extra["io"]
log("built", be.trace_report(), f"{t1 - t0:.1f}s; vstacks", len(sp.vstacks), "heads", sorted(sp.heads))
# a plain user in slot 0 at position 10 (its page-table row = the identity row of slot 0)
tok = torch.zeros(B, dtype=torch.int64)
tok[0] = 1234
pos = torch.full((B,), -1, dtype=torch.int64)
pos[0] = 10
be.write(io.tokens, tok)
be.write(io.cur_pos, pos)
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
sp.capture()
sp.enter(pt[0])
for _ in range(int(args.get("steps", 2))):
    sp.step()
rec = sp.read()
out = sp.leave()
# back to plain decode: the narrowest trace covering slot 0
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
be.replay(wts.get(w0, kv.extra.get("decode_trace")))
row = dict(test="s1-mock-spec-step", milestone="S1", status="pass", K=K, B=B, layers=_L,
           narrow=str(args.get("narrow", "none")), counters=sp.counters, rec_keys=sorted(rec), trace_report=be.trace_report(),
           build_s=round(t1 - t0, 1), tag=args.get("tag"))  # fmt: skip
S.record("tests", row)
RESULT = dict(counters=sp.counters, rec=rec, leave=out)
