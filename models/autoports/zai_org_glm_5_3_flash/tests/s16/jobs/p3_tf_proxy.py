# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P3 long-context TF PROXY -- teacher forcing through the PREFILL alone, for
contexts the decode stack cannot hold yet (StackBackend.supported_context = 2,051 while tp_dsa.TPDsaLayer runs MkMla
without its DSA indexer; program request #28). For each forced position i < gen of a Stage 6 reference (readiness
format: prompt_tokens [1, P], generated_tokens [1, G], topk_tokens [G, K]), the batched traced TP prefill of the first P
+ i tokens samples its first token on device (greedy, as serving does) -> the prediction for position P + i, scored
against the reference's top-1 / top-5 / top-K at that position (the readiness harness's metrics). Every request runs
the production chunk plan (backend.prefill_chunk_tokens = the largest bucket) through the captured traces.

ARGS: layers ([0..4]), ref (a .refpt), gen (32: forced positions), buckets ([2048]), max_seq (>= P + gen), sparse_ts
(windows), the P3 knobs (s16common.set_attn: sparse_attn, mla_form, ...), moe ("ep"), moe_v2, kda_scan, mhc,
mhc_post, logits_check (False: True = the final chunk returns its last-row logits instead of the device sample; the
prediction is their host argmax, and each position reports the non-finite count / max |logit| / the device-sampled
token of a second, sampled request at the same position -- the 32K diagnosis: hidden state vs the head's sampler),
tag, record."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.generator import plan_prefill  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

GREEDY = SamplingParams()
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
REF = args["ref"]
BUCKETS = tuple(int(v) for v in args.get("buckets", [2048]))
tag = args.get("tag", "p3-tf-proxy")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

entry = load_reference(REF).entries[0]
prompt = entry.prompt_tokens.reshape(-1).long()
forced = entry.generated_tokens.reshape(-1).long()
topk = entry.topk_tokens.long()
P, G = int(prompt.numel()), min(int(args.get("gen", 32)), int(forced.numel()))
MAXSEQ = int(args.get("max_seq", -(-(P + G) // 2048) * 2048))

os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_DSA"] = "1"
os.environ["GLM53_PF_SPARSE_T"] = ",".join(str(int(v)) for v in args.get("sparse_ts", [MAXSEQ]))
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=BUCKETS, max_seq=MAXSEQ)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
pf.sparse_ts = tuple(int(v) for v in args.get("sparse_ts", [MAXSEQ]))
backend.tpp = pf
t0 = time.time()
pf.compile(BUCKETS)
for c_ in BUCKETS:  # the request path's lazily allocated buffers before any capture (as p2_ttft's prefill warmup)
    backend.prefill_chunk(torch.zeros(c_, dtype=torch.int64), kv_cache=kv, slot=0, chunk_start=0, n_valid=c_,
                          logits_rows="last", io=io, sample=GREEDY)  # fmt: skip
ttnn.synchronize_device(mesh)
pf.capture(BUCKETS)
S.log(f"P {P}, G {G}, max_seq {MAXSEQ}: compiled + captured {sorted(pf.traces)} in {time.time() - t0:.1f} s")

eager0 = pf.counters["eager"]
toks = torch.cat([prompt, forced])
LCHK = bool(args.get("logits_check", False))
preds, walls, ldbg = [], [], []


def request(n: int, sampled: bool):
    plans = plan_prefill(n, 0, backend.prefill_chunk_tokens)
    for p in plans:
        out = backend.prefill_chunk(toks[p.offset : p.offset + p.n_valid], kv_cache=kv, slot=0, chunk_start=p.start,
                                    n_valid=p.n_valid, logits_rows="last", io=io,
                                    sample=GREEDY if sampled and p is plans[-1] else None)  # fmt: skip
    return out


for i in range(G):
    n = P + i
    t1 = time.perf_counter()
    out = request(n, not LCHK)
    if LCHK:
        lg = torch.as_tensor(out).float().reshape(-1)
        bad = int((~torch.isfinite(lg)).sum())
        lg_ = torch.nan_to_num(lg, nan=-1e30, posinf=-1e30, neginf=-1e30)
        preds.append(int(lg_.argmax()))
        walls.append((time.perf_counter() - t1) * 1e3)
        # what the head received: the persistent row_buf (the last valid row's final residual, all reduced) and the
        # last chunk's last_mask on every chip (the row the masked sum picks)
        d0 = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()  # noqa: E731
        rb = d0(pf.row_buf()).reshape(-1)
        nv_last = int(plan_prefill(n, 0, backend.prefill_chunk_tokens)[-1].n_valid)
        ci_ = pf.chunk_inputs(pf.bucket_for(nv_last, sparse=pf.has_dsa() and n > 2051))
        lm = [torch.nonzero(ttnn.to_torch(t_).float().reshape(-1)).reshape(-1).tolist()[:4]
              for t_ in (ttnn.get_device_tensors(ci_.last_mask) if TPN > 1 else [ci_.last_mask])]  # fmt: skip
        dev_tok = int(request(n, True))  # the same position through the device sampler
        ldbg.append(dict(n=n, nonfinite=bad, max_abs=round(float(lg_.abs().max()), 3), host_argmax=preds[-1],
                         device_sampled=dev_tok, ref_top1=int(topk[i, 0]),
                         row_buf_max_abs=round(float(rb.abs().max()), 4), row_buf_nonfinite=int((~torch.isfinite(rb)).sum()),
                         row_buf_stream_max=[round(float(rb[k * 4096:(k + 1) * 4096].abs().max()), 3) for k in range(4)],
                         last_mask_ones_per_chip=lm, n_valid_last=nv_last))  # fmt: skip
        S.log(ldbg[-1])
    else:
        preds.append(int(out))
        walls.append((time.perf_counter() - t1) * 1e3)
k = int(topk.shape[1])
m1 = sum(int(preds[i] == int(topk[i, 0])) for i in range(G))
m5 = sum(int(preds[i] in topk[i, :5].tolist()) for i in range(G))
mk = sum(int(preds[i] in topk[i].tolist()) for i in range(G))
traced = pf.counters["eager"] == eager0
miss = [(P + i, preds[i], topk[i, :5].tolist()) for i in range(G) if preds[i] not in topk[i, :5]]
row = dict(test=f"p3 TF proxy (prefill-only teacher forcing, {G} positions after a {P}-token prompt) layers "
                f"{LAYERS if len(LAYERS) < 45 else 'all'} tp{TPN}",
           milestone="P3" if len(LAYERS) == 45 else "P3-prefix",
           status="pass" if traced and m5 / G >= 0.98 else "fail", traced=traced, tp=TPN, prompt_len=P, positions=G,
           reference=REF, k=k, top1=round(m1 / G, 4), top5=round(m5 / G, 4), topk=round(mk / G, 4),
           matches=[m1, m5, mk, G], mismatches_top5=miss[:8],
           buckets=list(BUCKETS), chunk=backend.prefill_chunk_tokens, sparse_attn=pf.sparse_attn,
           request_ms_median=round(sorted(walls)[len(walls) // 2], 1), counters=dict(pf.counters), tag=tag)  # fmt: skip
if LCHK:
    row.update(logits_check=ldbg, prediction="host argmax of the device logits")
S.log(row)
if rec:
    C.record("pcc", row)
RESULT = [row]
