# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair): u1 of doc/ttnn/17-mtp-spec-decode/dflash2_intrace.md §5 -- the in-trace DFlash2 drafter's
drafts on the prototype's 16 golden anchors. Backend: the served construction with GLM53_SPEC_K=7,
GLM53_SPEC_DRAFTER=dflash2, GLM53_SPEC_VERIFY_W=plain (w8), GLM53_DF2_DIR, GLM53_DF2_TAP_MODE=mean (the prototype's tap
convention). Every device row is a trace replay: the drafter's ctx program is captured ONCE (after prepare() compiled it)
and replayed per 32-row chunk of the golden taps (host writes of the tap slots / positions between replays: the
prototype's ctx trace); each anchor is one replay of SpecMTP's captured BOOT trace (ACCEPT boot + PREP + the block) with
token row 0 = the anchor's bonus at position L + 1. Anchors run in DESCENDING L: a boot writes its block rows' K / V at
positions L+1 .. L+1+K of the drafter cache, above every smaller anchor's context. Gates: the 7 drafts (token rows 1..7)
equal the bfp8 golden's (= the TP2 prototype's, tests d0:dflash2-proto-golden tag d0-df2-tp2-*) on >= 14 / 16 anchors;
hidden PCC >= 0.999 (rows 1..7 of the block's final hidden).
Rung 3 (env GLM53_DF2_LAYER=mk, tt/dflash2_mk.py): the ctx program's all-reduce carries rows 0..7, so the context
is seeded 8 rows per replay (a chunk's garbage rows land on the next chunk's positions and are rewritten), and the
rows' cos / sin tiles the device PREP writes in a step are written by the host per chunk (as DFlash2Kind.seed_ctx).
ARGS: layers ("0,1,2,3,4"), profile ("served24"), b (32), taps, golden, max_seq_len (4096), env (the dflash2 knobs;
the hidden is read from the kind's persistent `last` copy: GLM53_DF2_KEEP_LAST=1, the default), tag."""
import importlib
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows  # noqa: E402

cfg = hf_config()
ids = [int(v) for v in str(args.get("layers", "0,1,2,3,4")).split(",")]
B, K = int(args.get("b", 32)), 7
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}
XENV.setdefault("GLM53_SPEC_DRAFTER", "dflash2")
XENV.setdefault("GLM53_SPEC_VERIFY_W", "plain")
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow="none", profile=str(args.get("profile", "served24")),
                         extra_env=XENV, max_seq_len=int(args.get("max_seq_len", 4096)))  # fmt: skip
sp, kv = be.spec, be._kv
io = kv.extra["io"]
df2 = sp.df2
assert df2 is not None and sp.K == K and sp.wv == 8, (sp.drafter, sp.K, sp.wv)
assert C.MOCK or df2.last is not None, "u1 reads the block's hidden: build with GLM53_DF2_KEEP_LAST=1 (the default)"


def log(*a):
    print(f"S17-DF2I {C.now()}", *a, flush=True)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


T = torch.load(args["taps"], weights_only=False)
G = torch.load(args["golden"], weights_only=False)
taps = T["taps"].permute(1, 0, 2).contiguous()  # [n, 9, H] (stream means)
n = int(taps.shape[0])
rows_out = []
if not C.MOCK:
    sp.capture()  # the step / boot traces (captured once per backend; a no-op when they exist)
    if "df2_ctx_trace" not in ctx[key]:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            df2.ctx_program()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ctx[key]["df2_ctx_trace"] = tid
    tid = ctx[key]["df2_ctx_trace"]
    TILE = 32
    step = 8 if df2.mk is not None else TILE  # rung 3: 8 valid rows per ctx replay (the all-reduce's rows)
    for r0 in range(0, n, step):  # the prompt context: the prototype's taps, `step` rows per replay
        part = taps[r0 : r0 + step]
        m = int(part.shape[0])
        for s_ in range(df2.d.n_taps):
            slot = torch.zeros(TILE, df2.tap_w)
            slot[:m] = part[:, s_].float()
            df2.ops.write(df2.taps[s_], slot.reshape(1, 1, TILE, -1))
        pos = torch.arange(r0, r0 + TILE, dtype=torch.int32)
        df2.ops.write(df2.st.cpos, pos)
        df2.ops.write(df2.st.cposu, pos.reshape(1, TILE))
        if df2.mk is not None:  # the rows' cos / sin tiles (the device PREP writes them in a step)
            from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_ref import rope_cos_sin

            cos_t, sin_t = rope_cos_sin(pos.clamp(0, df2.cap - 1), df2.d.theta, df2.d.hd)
            df2.ops.write(df2.rope_tiles[2], cos_t.to(torch.bfloat16).reshape(1, 1, TILE, -1))
            df2.ops.write(df2.rope_tiles[3], sin_t.to(torch.bfloat16).reshape(1, 1, TILE, -1))
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    for a in sorted(G["anchors"], key=lambda a_: -int(a_["L"])):
        L = int(a["L"])
        tk = torch.zeros(B, dtype=torch.int64)
        pos = torch.full((B,), -1, dtype=torch.int64)
        tk[0], pos[0] = int(a["bonus"]), L + 1
        be.write(io.tokens, tk)
        be.write(io.cur_pos, pos)
        sp._replay("boot")
        ttnn.synchronize_device(mesh)
        tin = read_rows(mesh, sp.tin).reshape(TILE, -1)
        drafts = [int(v) for v in tin[1 : K + 1, 0]]
        hid = ttnn.to_torch(ttnn.get_device_tensors(df2.last["hid"])[0]).float().reshape(TILE, -1)[1 : K + 1]
        rows_out.append(dict(L=L, drafts=drafts, ref=[int(v) for v in a["drafts"]], draft_eq=drafts == [int(v) for v in a["drafts"]],
                             hid_pcc=round(pcc(hid, a["hid"]), 6)))  # fmt: skip
        log("anchor", L, "drafts", drafts, "ref", a["drafts"], "hid pcc", rows_out[-1]["hid_pcc"])
eq = sum(r["draft_eq"] for r in rows_out)
ok = C.MOCK or (eq >= len(rows_out) - 2 and min(r["hid_pcc"] for r in rows_out) >= 0.999)
S.record("tests", dict(test="dflash2-intrace-u1", milestone="DF2", status="pass" if ok else "fail", K=K, B=B, layers=str(args.get("layers")),
                       tp=int(mesh.get_num_devices()), anchors=len(rows_out), draft_eq=eq, layer=df2.layer_mode,
                       drafter_launches_boot=int(getattr(sp, "counters", {}).get("boots", 0)),
                       min_hid_pcc=min([r["hid_pcc"] for r in rows_out] or [None]) if rows_out else None,
                       golden_weights=G.get("weights", "bfloat16"), rows=rows_out, env=XENV, tag=args.get("tag")))  # fmt: skip
log("draft_eq", eq, "/", len(rows_out))
RESULT = dict(ok=ok, draft_eq=eq, anchors=len(rows_out))
