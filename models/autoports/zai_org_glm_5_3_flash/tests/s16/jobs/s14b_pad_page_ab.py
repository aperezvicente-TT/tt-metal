# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 14b's pair or mesh server): Stage 14b triage of the served int24svc accuracy failure (program
2026-10-22 04:15Z) -- a PADDED TP prefill chunk under a vLLM-style page row. The resident serving backend
(m5common.serving) with Stage 16's TP prefill attached (svc_attach); one prompt of n tokens (default 161 = the AIME
prompt: bucket 512, 3 valid pages) then `new` greedy decode steps, three ways:
  I    generator-owned identity row (Glm53Generator.generate: the harness path, distinct tail pages);
  off  a caller row like vllm_page_rows makes it (non-contiguous blocks, -1 past the allocation -> the generator repeats
       the last valid block in the tail) with GLM53_PF_PAD_NULL=0 (the old chunk tables: padding rows land on the last
       valid page);
  on   the same row with GLM53_PF_PAD_NULL=1 (pure-padding pages -> the null block 0).
Row: first prefill token and the first index where each run's tokens differ from I. Expected: off diverges after the
first token, on == I. No soak.
ARGS: b (32), layers ([0..7]), native (true), rev, n (161), new (16), slot (3), knobs, buckets, record (true), tag."""
import datetime as dt
import os
import sys

import torch

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
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
GOLDENS = os.environ.get("GLM53_GOLDENS", "/opt/glm53/kit/goldens")
cfg = hf_config()
_L = args.get("layers", list(range(8)))
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, NATIVE, REV = int(args.get("b", 32)), bool(args.get("native", True)), args.get("rev", "p25")
N, NEW, SLOT = int(args.get("n", 161)), int(args.get("new", 16)), int(args.get("slot", 3))
TP = int(mesh.get_num_devices())


def log(*a):
    print(f"S14B-PAD-AB {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


toks = [int(v) for v in torch.load(os.path.join(GOLDENS, "prompts", "512.pt"), map_location="cpu",
                                   weights_only=True).reshape(-1)[:N]]  # fmt: skip
key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=REV, log=log)
att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=str(args.get("buckets", "128,512,2048")))
be, gen = att.be, att.gen
greedy = SamplingParams()
saved = os.environ.get("GLM53_PF_PAD_NULL")
pf = None
try:
    pf = att.attach()
    b = gen._own_binding()
    kv = b.kv_cache
    # the identity path (the harness): prefill + NEW decode steps
    gen.reset()
    ref = [int(v) for v in gen.generate(toks, NEW + 1, sampling=greedy, stop_on_eos=False)]
    # a vLLM-style row: non-contiguous blocks from the top of the pool (no live slot owns them), -1 past the allocation
    need = -(-(N + NEW) // be.block_size)
    blocks = [int(b.num_blocks) - 3 - 5 * j for j in range(need)]
    row = torch.full((1, int(b.blocks_per_seq)), -1, dtype=torch.int64)
    row[0, :need] = torch.tensor(blocks)
    c, t_win = pf.plan(0, N)
    out = {}
    for mode in ("off", "on"):
        os.environ["GLM53_PF_PAD_NULL"] = "0" if mode == "off" else "1"
        gen.reset()
        tok = torch.zeros(1, N, dtype=torch.int64)
        tok[0, :] = torch.tensor(toks)
        first = gen.prefill_forward(tok, page_table=row, kv_cache=kv, prompt_lens=[N], empty_slots=[SLOT],
                                    sampling_params=[greedy])  # fmt: skip
        seq = [int(first[0])]
        for i in range(NEW):
            o = gen.decode_forward(torch.tensor([-1]), torch.tensor([N + i]), page_table=row, kv_cache=kv, slots=[SLOT],
                                   sampling_params=[greedy])  # fmt: skip
            seq.append(int(torch.as_tensor(o).reshape(-1)[0]))
        out[mode] = seq
        log(mode, seq)
    log("identity", ref)
    d_off, d_on = first_diff(out["off"], ref), first_diff(out["on"], ref)
    row_ = dict(test=f"served-prefill triage: a PADDED TP prefill chunk under a vLLM-style page row (n {N}, bucket {c}, "
                     f"{need} pages, tail -1) vs the identity row, greedy {NEW} decode steps, tp{TP} layers {len(ids)}",
                milestone="M5c-triage", status="pass" if d_on is None else "fail", n=N, bucket=c, t_win=t_win,
                pages=blocks, first_token={"identity": ref[0], "off": out["off"][0], "on": out["on"][0]},
                first_divergence={"off (old tables)": d_off, "on (GLM53_PF_PAD_NULL)": d_on},
                tokens={"identity": ref, "off": out["off"], "on": out["on"]}, rev=REV, knobs=att.knobs,
                tag=args.get("tag", f"m5c-pad-page-ab-tp{TP}"))  # fmt: skip
    log(row_)
    if args.get("record", True) and not C.MOCK:
        C.record("pcc", row_)
    RESULT = dict(ok=d_on is None, off=d_off, on=d_on)
finally:
    if saved is None:
        os.environ.pop("GLM53_PF_PAD_NULL", None)
    else:
        os.environ["GLM53_PF_PAD_NULL"] = saved
    try:
        gen.reset()
        gen._ensure_identity_rows(gen._own_binding(), list(range(B)))
    except Exception as e:  # noqa: BLE001
        log("identity rows restore failed:", repr(e))
    pf = None
    log(f"restored: decode-only warm-up {att.restore()} s")
