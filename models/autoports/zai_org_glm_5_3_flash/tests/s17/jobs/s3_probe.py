# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair; diagnosis): what the speculative step's ACCEPT reads, per chip. After a prefill + the
bootstrap, around each step: the verify input rows (tin = the main head's token rows 0..K, word 0), the verify head's
argmax rows, the MTP heads' rows, the record, the selector and the positions -- on EVERY chip, before the step (after
an optional host write of the draft rows) and after it. Steps: one unforced, then forced drafts (= plain's
continuation) with the target accepted counts in ARGS["targets"]. Rows: tests `s3-probe` (info). Host reads between
synchronized replays only. ARGS: layers, b, k, narrow, prompt, targets ("3,1,0"), tag."""
import importlib
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 4)), int(args.get("k", 3))
_L = str(args.get("layers", "0,1,2,3,4"))
WS = "/opt/glm53/kit"  # (host-8chip and host-p300)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")))
sp, kv = be.spec, be._kv
io = kv.extra["io"]
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import write_rows  # noqa: E402

pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))
PROMPTS = {q["id"]: q for q in json.load(open(os.path.join(WS, "doc/prework/p5_mtp_acceptance/prompts.json")))["prompts"]}
prompt = [int(v) for v in PROMPTS[str(args.get("prompt", "aime24_04"))]["ids"]]


def chips(t, rows: int, words: int = 1):
    return [ttnn.to_torch(d).to(torch.int64).reshape(-1, int(d.shape[-1]))[:rows, :words].reshape(-1).tolist()
            for d in ttnn.get_device_tensors(t)]


def snap(tag: str) -> dict:
    ttnn.synchronize_device(mesh)
    s = dict(tag=tag, tin=chips(sp.tin, K + 1), verify=chips(sp.heads["verify"].topk_out, K + 1),
             mtp_cu=chips(sp.heads["mtp_cu"].topk_out, K + 1),
             mtp_ch=[chips(h.topk_out, 1) for h in sp.heads["mtp_ch"]],
             rec=[r[:8] + r[16:16 + K] + r[31:32] for r in chips(sp.rec, 1, 64)], sel=chips(sp.sel, 1),
             pos_v=chips(sp.pos_v, 1, K + 1), pos=chips(be.pos, 1, 4))  # fmt: skip
    print("S17-PROBE", json.dumps(s), flush=True)
    return s


def prefill() -> int:
    return S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)


ref = [prefill()]
for _ in range(16):
    be.replay(plain)
    ttnn.synchronize_device(mesh)
    ref.append(int(be.read(io.tokens)[0]))
print("S17-PROBE ref", ref, flush=True)
prefill()
sp.enter(pt[0])
snaps = [snap("after-boot")]
toks = [ref[0]]
sp.step()
snaps.append(snap("after-step-unforced"))
r = sp.read()
toks += r["committed"]
for tgt in [int(v) for v in str(args.get("targets", "3,1,0")).split(",")]:
    L = len(toks)
    host = ttnn.to_torch(ttnn.get_device_tensors(sp.tin)[0]).to(torch.int64).reshape(32, 64).clone()
    for j in range(1, K + 1):
        t = ref[min(L - 1 + j, len(ref) - 1)]
        host[j, 0] = (t + 1) % int(cfg.vocab_size) if j == tgt + 1 else t
    write_rows(mesh, sp.tin, host)
    snaps.append(snap(f"after-write-target{tgt}"))
    sp.step()
    snaps.append(snap(f"after-step-target{tgt}"))
    r = sp.read()
    toks += r["committed"]
ok = toks == ref[: len(toks)]
S.record("tests", dict(test=f"s3-probe-k{K}", milestone="S3", status="info", K=K, B=B, layers=_L, identity=ok,
                       toks=toks, ref=ref, snaps=snaps, tag=args.get("tag")))  # fmt: skip
sp.leave()
RESULT = dict(identity=ok, n=[s["rec"][0][0] for s in snaps])
