# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, unwatched; S3 perf): where the speculative step's time goes. After the spec traces exist (enter),
each piece of the step is captured as its OWN trace and replayed back-to-back: verify (embed + every stack + the
verify head), accept (SpecCtl ACCEPT), the MTP catch-up (eh norm / proj / expand + the MTP layer at t = K+1 + its
head), each chained row (t = 1), gather; then the whole step, the bootstrap and the plain step for reference. Every
row: >= `reps` replays in rounds of `per_round`, the speculative state (positions, selector, record, token rows)
restored by host writes between rounds (ACCEPT advances the positions on every replay). Rows: perf
`s3-breakdown-k<K>` (us per piece, sum vs the whole step). fine=1 also splits chained row 0 into eh / the MTP layer /
its head. ARGS: layers, b, k, narrow, prompt, reps (2000), per_round (200), fine (0), profile, max_seq_len (2048), rev
(profile "asis" + rev: the integrator's backend as its env job built it), tag."""
import importlib
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 4)), int(args.get("k", 3))
REPS, PR = int(args.get("reps", 2000)), int(args.get("per_round", 200))
_L = str(args.get("layers", "0,1,2,3,4"))
WS = "/opt/glm53/kit"  # (host-8chip and host-p300)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows, write_rows  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")),
                         max_seq_len=int(args.get("max_seq_len", 2048)), rev=args.get("rev"))  # fmt: skip
sp, kv = be.spec, be._kv
io = kv.extra["io"]
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))
PROMPTS = {q["id"]: q for q in json.load(open(os.path.join(WS, "doc/prework/p5_mtp_acceptance/prompts.json")))["prompts"]}
prompt = [int(v) for v in PROMPTS[str(args.get("prompt", "aime24_04"))]["ids"]]
max_pos = int(kv.blocks_per_seq) * int(be.block_size)
assert len(prompt) + (K + 1) * PR + 8 < max_pos, ("rounds would run past the page table", len(prompt), PR, max_pos)

S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)
sp.enter(pt[0])
ttnn.synchronize_device(mesh)
STATE = [sp.pos_v, sp.mpos_cu, sp.sel, sp.rec, sp.tin, be.pos] + list(sp.mpos_ch)
saved = [(t, read_rows(mesh, t).clone(), t.dtype) for t in STATE]


def restore():
    ttnn.synchronize_device(mesh)
    for t, v, dt in saved:
        write_rows(mesh, t, v, dt)


def timed(tr) -> float:
    us = []
    for _ in range(max(1, REPS // PR)):
        restore()
        ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)  # (one warm replay per round)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(PR):
            ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us.append((time.perf_counter() - t0) / PR * 1e6)
    return sorted(us)[len(us) // 2]


pieces = [("verify", sp._verify), ("accept", lambda: sp._ctl_accept(False)), ("catchup", sp._catchup)]
pieces += [(f"chain{i}", (lambda i=i: sp._chain(i))) for i in range(K - 1)] + [("gather", sp._gather)]
pieces += [("boot_ctl", lambda: sp._ctl_accept(True))]
if int(args.get("fine", 0)) and K >= 2:  # chained row 0 split: eh (enorm / hnorm + eh_proj + expand), the MTP layer, head
    def _eh0():
        sp.eh.norm_e(sp.heads["mtp_cu"].topk_out, be.embed.table, t=1, sel=sp.sel)
        sp.eh.expand(sp.eh.norm_h_proj(sp.R_m, mode=3, t=1, sel=sp.sel), sp.R_c[0], t=1)

    pieces += [("chain0_eh", _eh0),
               ("chain0_layer", lambda: sp._dsa(dict(mla=sp.mtp.mla, moe=sp.mtp.moe), sp.R_c[0], sp.mtp_cache, sp.pt_v,
                                                sp.mpos_ch[0], 1, False)),
               ("chain0_head", lambda: sp.heads["mtp_ch"][0]([sp.R_c[0]], None))]  # fmt: skip
res, tids = {}, []
try:
    for name, fn in pieces:
        restore()
        if C.MOCK:  # (the mock cluster cannot capture a trace: each piece runs eagerly once)
            fn()
            res[name] = 0.0
            continue
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tids.append(tid)
        res[name] = round(timed(tid), 1)
        print(f"S17-BD {name} {res[name]} us", flush=True)
    if not C.MOCK:
        tr = sp.capture()
        res["step"], res["boot"] = round(timed(tr["step"]), 1), round(timed(tr["boot"]), 1)
        res["plain"] = round(timed(plain), 1)
    else:
        res.update(step=0.0, boot=0.0, plain=0.0)
finally:
    for tid in tids:
        ttnn.release_trace(mesh, tid)
    restore()
parts = sum(v for k, v in res.items() if k not in ("step", "boot", "plain", "boot_ctl") and not k.startswith("chain0_"))
S.record("perf", dict(test=f"s3-breakdown-k{K}", milestone="S3", status="info", K=K, B=B, layers=_L, tp=be.tp,
                      reps=REPS, per_round=PR, us=res, pieces_sum_us=round(parts, 1),
                      residual_us=round(res["step"] - parts, 1), plain_width=w0, tag=args.get("tag")))  # fmt: skip
print(f"S17-BD {res} sum {parts:.1f}", flush=True)
sp.leave()
RESULT = dict(us=res)
