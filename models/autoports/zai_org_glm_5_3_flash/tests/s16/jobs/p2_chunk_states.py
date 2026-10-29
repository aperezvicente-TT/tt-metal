# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 service-prefill triage (API accuracy failure of the int24 service): the prefill as
the vLLM plugin issues it, through Glm53Generator.prefill_forward / decode_forward on a caller-owned cache, vs the
full-coverage page rows the readiness harness uses.

vLLM page-table rows cover only the blocks a request holds (generator_vllm.vllm_page_rows marks the rest -1; the
generator pads the row's tail with its last block), and every vLLM prefill step is its own prefill_forward call (a
continuation chunk = start_pos at the previous chunk's end). Cases (args cases), each request in its own slot:
  ref   - the prompt in one chunk, a full-coverage row (40 distinct blocks): the harness condition (pair TF 1.0)
  v1    - the same prompt in one chunk, a vLLM row (ceil(n / 64) blocks)
  v2    - the same prompt as the chunks `chunks` (e.g. 128 + 22), vLLM rows growing per step
then D teacher-forced decode steps per slot (tokens = the prompt's continuation, page rows grown as vLLM would).
Compared with ref: the prefill's last logits, the decode logits per step (argmax / PCC / ref top-1 in top-5), every KDA
layer's S / C, every sparse_dsa layer's latent rows < n (max |diff|, bitwise). `mappings` runs the cases with the
pre-fix chunk page mapping (legacy: pages past the row's coverage alias its last page) and the current one
(prefill_tp.chunk_pages). Rows: pcc.jsonl.

ARGS: layers ([0..4]), prompts ([[150, [128]], [22, []]]: [n, first chunks]), d (16), mappings (["legacy", "fixed"]),
max_seq (2560), ref (its prompt repeats past its end), the prefill knobs (default the service knobs), tag, record."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PT  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
PROMPTS = [(int(n), [int(c) for c in ch]) for n, ch in args.get("prompts", [[150, [128]], [22, []]])]
D = int(args.get("d", 16))
MAPPINGS = list(args.get("mappings", ["legacy", "fixed"]))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
tag = args.get("tag", "p2-chunk-states")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
B, MAXSEQ, BLK = 4, int(args.get("max_seq", 2560)), 64
BPS = MAXSEQ // BLK

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = args.get("moe", "ep")
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MOE_V2"] = "1"
os.environ["GLM53_PF_KDA_SCAN"] = args.get("kda_scan", "op")
os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_SPARSE_T"] = ""
os.environ["GLM53_PF_BATCH"] = "0"
for k_ in ("GLM53_PF_MOE_FFN", "GLM53_PF_MOE_XFFN_XSPLIT", "GLM53_PF_MOE_XFFN_SNAKE", "GLM53_PF_MOE_XFFN_VALID"):
    os.environ.pop(k_, None)  # the code defaults (the service profile pins none of these)
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})

prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
prompt = prompt.repeat(-(-max(n + D for n, _ in PROMPTS) // prompt.numel()))

for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-chunks")]:
    old = ctx.pop(k)  # one model per process
    if "pf" in old and old["pf"] is not None:
        old["pf"].release()
    old["backend"].close()
gc.collect()
t0 = time.time()
backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg,
                                 weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=MAXSEQ,
                        stop_token_ids=())  # fmt: skip
kv = gen.allocate_kv_cache(max_batch=B, max_seq_len=MAXSEQ, num_blocks=B * BPS + 1)  # + vLLM's null block 0
gen.warmup(kv)
ctx[("s16-chunks", tuple(LAYERS), TPN)] = dict(backend=backend, gen=gen, pf=backend.tpp)
pf = backend.tpp
assert pf is not None and pf.kv is kv, "the TP prefill must be bound to the serving cache"
S.log(f"backend + generator built and warmed in {time.time() - t0:.1f} s; prefill traces {sorted(map(str, pf.traces))}")
eager0, dec_eager0 = pf.counters["eager"], gen.counters.get("eager_decode_steps", 0)


def legacy_pages(row, chunk_start, n_valid, c):  # the pre-fix mapping (write_inputs up to 6178a3252bf)
    blocks = row[chunk_start // BLK : chunk_start // BLK + c // BLK]
    if blocks.numel() < c // BLK:
        blocks = torch.cat([blocks, blocks[-1:].expand(c // BLK - blocks.numel())])
    return torch.where(blocks < 0, torch.zeros_like(blocks), blocks)


FIXED_PAGES = PT.chunk_pages
FULL_ROW = torch.arange(1, 1 + BPS)  # the harness's full-coverage row (blocks 1 .. BPS: block 0 = the null block)
POOL = {s_: [BPS * s_ + 1 + (7 * j) % BPS for j in range(BPS)] for s_ in (1, 2, 3)}  # vLLM-style scattered blocks


def vrow(slot, upto):
    """slot's vLLM page-table row covering positions < upto (blocks it holds, then -1)"""
    r = torch.full((BPS,), -1, dtype=torch.int64)
    if slot == 0:
        return FULL_ROW.clone()
    blk = POOL[slot][: -(-upto // BLK)]
    r[: len(blk)] = torch.tensor(blk)
    return r


def prefill(slot, n, chunks):
    bounds, s = [], 0
    for c in chunks + [n - sum(chunks)]:
        bounds.append((s, c))
        s += c
    out = None
    for s_, c in bounds:
        out = gen.prefill_forward(prompt[s_ : s_ + c].reshape(1, -1), page_table=vrow(slot, s_ + c)[None], kv_cache=kv,
                                  prompt_lens=[c], empty_slots=[slot], start_pos=[s_])  # fmt: skip
    return out.float().reshape(-1), bounds


def decode_tf(slot, n):
    logits = []
    for j in range(D):
        p = n + j
        logits.append(gen.decode_forward(prompt[p : p + 1].reshape(1, 1), torch.tensor([p]), page_table=vrow(slot, p + 1)[None],
                                         kv_cache=kv, slots=[slot], enable_trace=True).float().reshape(-1))  # fmt: skip
    return torch.stack(logits)


def snapshot(slot, n):
    blocks = gen._binding(kv).pt_shadow[slot]
    lat = {L: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()
           for L, t_ in backend.latent_pages(kv).items()}  # fmt: skip
    lr = {L: torch.cat([c_[int(blocks[p]), 0] for p in range(-(-n // BLK))], dim=0)[:n] for L, c_ in lat.items()}
    kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, slot)}
    return kda, lr


def diff(u, v):
    return [float((u - v).abs().max()), round(C.pcc(u.flatten(), v.flatten()), 6)]


rows = []
for mapping in MAPPINGS:
    PT.chunk_pages = legacy_pages if mapping == "legacy" else FIXED_PAGES
    for n, chunks in PROMPTS:
        cases = [("ref", 0, [])] + [("v1", 1, [])] + ([("v2", 2, chunks)] if chunks else [])
        res = {}
        for name, slot, ch in cases:
            last, bounds = prefill(slot, n, ch)
            res[name] = dict(slot=slot, bounds=bounds, last=last)
        for name, slot, _ in cases:  # decode after every prefill: the slots' prefill states are final
            res[name]["dec"] = decode_tf(slot, n)
            res[name]["kda"], res[name]["lat"] = snapshot(slot, n)
        ref = res["ref"]
        ref_top5 = ref["dec"].topk(5, dim=-1).indices
        for name, _, _ in cases[1:]:
            r = res[name]
            per = {}
            for L in ref["kda"]:
                per[f"L{L}.S"] = diff(ref["kda"][L][0], r["kda"][L][0])
                per[f"L{L}.C"] = diff(ref["kda"][L][1], r["kda"][L][1])
            for L in ref["lat"]:
                per[f"L{L}.lat"] = diff(ref["lat"][L], r["lat"][L])
                bad = (ref["lat"][L] - r["lat"][L]).abs().amax(dim=-1) > 0
                per[f"L{L}.lat_rows_differ"] = [int(bad.sum()), int(torch.nonzero(bad)[0]) if bad.any() else -1]
            per["prefill_last"] = diff(ref["last"], r["last"])
            am_ref, am = ref["dec"].argmax(-1), r["dec"].argmax(-1)
            top1 = float((am_ref == am).float().mean())
            in5 = float(
                torch.tensor([int(am_ref[j]) in r["dec"][j].topk(5).indices.tolist() for j in range(D)]).float().mean()
            )
            dec_pcc = min(C.pcc(ref["dec"][j], r["dec"][j]) for j in range(D))
            same_first = int(ref["last"].argmax()) == int(r["last"].argmax())
            ok = same_first and top1 == 1.0 and dec_pcc >= 0.99
            row = dict(test=f"service prefill triage ({mapping} chunk pages): {n}-token prompt as {[c for _, c in r['bounds']]} "
                            f"with vLLM page rows (blocks held only) vs one chunk with a full-coverage row, prefill_forward "
                            f"then {D} teacher-forced decode steps, tp{TPN} layers {len(LAYERS)}",
                       milestone="P2", status=("pass" if ok else "fail") if mapping != "legacy" else "info", result="pass" if ok else "fail", mapping=mapping, case=name, n=n,
                       chunks=[c for _, c in r["bounds"]], plans=[list(pf.plan(s_, c_)) for s_, c_ in r["bounds"]],
                       prefill_first_same=same_first, decode_top1_vs_ref=round(top1, 4),
                       decode_ref_in_top5=round(in5, 4), decode_pcc_min=round(dec_pcc, 6),
                       bitwise_states=all(v[0] == 0.0 for k_, v in per.items() if not k_.endswith("rows_differ")),
                       per=per, tp=TPN, layers=len(LAYERS), mock=C.MOCK, tag=tag)  # fmt: skip
            S.log({k_: v for k_, v in row.items() if k_ != "per"})
            S.log(f"  per: {per}")
            rows.append(row)
PT.chunk_pages = FIXED_PAGES
traced = pf.counters["eager"] == eager0 and gen.counters.get("eager_decode_steps", 0) == dec_eager0
for row in rows:
    row["traced"] = traced
    if rec:
        C.record("pcc", row)
RESULT = rows
