# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P4 gate "a decode step with an interleaved prefill chunk <= 2x a plain decode step"
(chunked prefill interleaved with decode). The served generator (B decode rows, native widths, TP prefill traces of
every bucket) is warmed and captured. Then three steady states are timed as traced replays:
  decode  -- the full-width decode trace alone (every row live at position `pos`);
  chunk   -- one prefill chunk trace alone (bucket c, a fresh chunk at position 0, or a continuation at `cont_start`);
  step    -- the decode trace followed by the chunk trace, back to back on the queue (the interleaved serving step).
Ratio = step / decode per chunk size. No sampling-path or host work is timed; KDA states / pages of slot 0 are
overwritten (timing only).

ARGS: layers ([0..4]), b (32), native (True), pos (1024: every decode row's position), chunks ([128, 512]: the
interleaved chunk sizes, each a captured bucket), buckets ("128,512,2048"), replays (2000), env, knobs (s16common.set_attn
+ mhc_post / moe_v2), tag, record (True)."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", True))
POS = int(args.get("pos", 1024))
CHUNKS = [int(v) for v in args.get("chunks", [128, 512])]
REPLAYS = int(args.get("replays", 2000))
tag = args.get("tag", "p4-interleave")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = "ep"
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MOE_V2"] = "1" if args.get("moe_v2", True) else "0"
os.environ["GLM53_PF_DSA"] = "0"
S.set_attn(args)
for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-multi")]:
    old = ctx.pop(k)
    if "pf" in old:
        old["pf"].release()
    old["backend"].close()
cfg = hf_config()
backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg, native_t=NATIVE,
                                 weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                        max_seq_len=POS + 2048 + 64, stop_token_ids=())  # fmt: skip
gen.warmup()
ctx[("s16-multi", "interleave")] = dict(backend=backend, gen=gen)
b = gen._own_binding()
pf = backend.tpp
width = max(b.decode_traces)
dec = b.decode_traces[width]
gen._ensure_identity_rows(b, list(range(b.max_batch)), {s_: POS + 1 for s_ in range(b.max_batch)})  # pages to POS
backend.write(b.io.cur_pos, torch.full((b.max_batch,), POS, dtype=torch.int64))  # every row live at POS
ttnn.synchronize_device(mesh)

row = dict(test=f"P4 interleave: decode step (B={B}, width {width}, every row at {POS}) + one prefill chunk vs the plain "
                f"decode step, traced, tp{TPN} L{LAYERS[0]}-L{LAYERS[-1]}", milestone="P4", layers=LAYERS, B=B,
           width=width, pos=POS, tp=TPN, tag=tag)  # fmt: skip
d = S.replay_stats(mesh, dec, replays=REPLAYS, label="decode")
row.update(decode_us=d.get("median_us"), decode_replays=d.get("replays"))
per = {}
for c in CHUNKS:
    key = pf.plan(0, c)
    if key not in pf.traces:
        per[c] = dict(error=f"no captured prefill trace for {key}")
        continue
    ch = S.replay_stats(mesh, pf.traces[key], replays=REPLAYS, label=f"chunk {c}")
    st = S.replay_stats(mesh, [dec, pf.traces[key]], replays=REPLAYS, label=f"decode + chunk {c}")
    per[c] = dict(key=str(key), chunk_us=ch.get("median_us"), step_us=st.get("median_us"), replays=st.get("replays"),
                  ratio=round(st["median_us"] / d["median_us"], 3) if st and d else None,
                  overlap_us=round(d["median_us"] + ch["median_us"] - st["median_us"], 1) if st and d and ch else None)  # fmt: skip
row["chunks"] = per
ok = [v.get("ratio") for v in per.values() if v.get("ratio") is not None]
row["status"] = "pass" if ok and all(r_ <= 2.0 for r_ in ok) else ("fail" if ok else "n/a")
S.log(row)
if rec and not C.MOCK:
    C.record("perf", row)
RESULT = row
