# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 TF A/B in ONE build (the TP8 TF top-5 0.969 vs the pair's 1.0, program
2026-10-22 03:34Z): the same prompt prefilled into slot 0 (a) by the TP prefill (StackBackend.prefill_chunk ->
TPPrefill, one traced replay) and (b) by the decode stack (the same call with backend.tpp unset: one decode-trace
replay per token, the plain int24 path), compared per layer -- every KDA layer's final S / conv carry C, every
sparse_dsa layer's latent rows < n, the last position's logits (max |diff|, PCC, argmax) -- then the Stage 6 TF harness
with each prefill (top-1 / 5 / 100 vs the CPU fp32 golden). A layer whose states split first is where the TP prefill
diverges; TF (b) >= 0.98 with (a) < 0.98 puts the miss in the prefill, equal TFs put it elsewhere.

Reuses p2_full's backend in the same process (CTX "s16-full" with the TP prefill and these layers), else builds it with
the same knobs. ARGS: layers ("all"), n (2048), ref (refpt/prefill2048_full.refpt), tf (True), b (1), max_seq (2560),
the prefill knobs as p2_full (default the service knobs), tag, record (True)."""
import gc
import os
import sys
import time
from pathlib import Path

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

C = S.C
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.run_teacher_forcing import run_teacher_forcing  # noqa: E402
from readiness_check.schema import load_reference  # noqa: E402

MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
N = int(args.get("n", 2048))
B = int(args.get("b", 1))
MAXSEQ = int(args.get("max_seq", 2560))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_full.refpt")
tag = args.get("tag", "p2-tf-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = args.get("moe", "ep")
os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MOE_V2"] = "1" if args.get("moe_v2", True) else "0"
os.environ["GLM53_PF_KDA_SCAN"] = args.get("kda_scan", "op")
os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_SPARSE_T"] = ""
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})

full = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "s16-full" and k[1] == "tp" and list(k[2]) == LAYERS]
if full:
    ent = ctx[full[-1]]
    backend, gen = ent["backend"], ent["gen"]
    S.log(f"adopted p2_full's backend {full[-1][:3]}")
else:
    for k in [
        k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-chunks")
    ]:
        old = ctx.pop(k)  # one model per process
        if old.get("pf") is not None:
            old["pf"].release()
        old["backend"].close()
    gc.collect()
    t0 = time.time()
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg,
                                     weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=MAXSEQ,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[("s16-full", "tp", tuple(LAYERS), "tf-ab")] = dict(
        backend=backend, gen=gen, eager0=backend.tpp.counters["eager"]
    )
    S.log(f"built + warmed in {time.time() - t0:.1f} s")
pf = backend.tpp
assert pf is not None, "the TP prefill backend (GLM53_STACK_PREFILL=tp)"
b = gen._own_binding()
kv = b.kv_cache
prompt_all = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
N = min(N, int(prompt_all.numel()))  # the reference prompt may be shorter than n (2,016 tokens)
# ns: a sweep of prompt lengths (state A/B only, no TF): e.g. the 32-row KDA chunk edges 1, 31..33, 63..65 and the
# 128-bucket in the replicated mode TP8 uses for it (args rep_buckets [128] at TP2)
NS = [min(int(v), int(prompt_all.numel())) for v in args.get("ns", [N])]
row0 = b.identity_row(0)


def prefill(tp: bool, n: int):
    prompt = prompt_all[:n]
    backend.tpp = pf if tp else None  # unset: StackBackend.prefill_chunk runs the decode-step prefill
    try:
        out = gen.prefill_forward(prompt.reshape(1, -1), page_table=row0[None], kv_cache=kv, prompt_lens=[n],
                                  empty_slots=[0])  # fmt: skip
    finally:
        backend.tpp = pf
    ttnn.synchronize_device(mesh)
    lat = {L: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()
           for L, t_ in backend.latent_pages(kv).items()}  # fmt: skip
    blocks = b.pt_shadow[0]
    lr = {L: torch.cat([c_[int(blocks[p]), 0] for p in range(-(-n // 64))], dim=0)[:n] for L, c_ in lat.items()}
    kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, 0)}
    return out.float().reshape(-1), kda, lr


# golden (args golden_file): the prefix model's CPU fp32 logits at position n - 1 of the same prompt (causal: = the
# logits of the prompt truncated to n tokens), work-s16/tmp/prefix_golden_logits.py
GOLD = torch.load(args["golden_file"], map_location="cpu", weights_only=True) if args.get("golden_file") else None
rows, prev = [], None
for n in NS if args.get("ab", True) else []:  # ab False: the TF rows only
    t0 = time.time()
    tp_out = prefill(True, n)
    t1 = time.time()
    dec_out = prefill(False, n)
    t2 = time.time()
    per = {}
    gold = {}
    if GOLD is not None and n in GOLD["ns"]:
        assert torch.equal(GOLD["prompt"][:n].long(), prompt_all[:n]), "golden prompt != the reference prompt"
        gl = GOLD["logits"][GOLD["ns"].index(n)].float()
        for nm, lg in (("tp", tp_out[0]), ("decode", dec_out[0])):
            lg = lg[: gl.numel()]
            gold[nm] = dict(pcc=round(C.pcc(lg, gl), 6), argmax=int(lg.argmax()), golden_argmax=int(gl.argmax()),
                            top5_overlap=len(set(lg.topk(5).indices.tolist()) & set(gl.topk(5).indices.tolist())))  # fmt: skip
    delta = {}
    if prev is not None and prev[0] == n - 1:  # the one extra token's state update, TP vs decode stack
        for L in tp_out[1]:
            dt = tp_out[1][L][0] - prev[1][1][L][0]
            dd = dec_out[1][L][0] - prev[2][1][L][0]
            delta[f"L{L}.dS"] = [
                round(float(dt.norm()), 4),
                round(float(dd.norm()), 4),
                round(C.pcc(dt.flatten(), dd.flatten()), 6),
            ]
    prev = (n, tp_out, dec_out)
    for L in sorted(set(tp_out[1]) | set(tp_out[2])):
        if L in tp_out[1]:
            for nm, i in (("S", 0), ("C", 1)):
                u, v = tp_out[1][L][i], dec_out[1][L][i]
                per[f"L{L}.{nm}"] = [round(float((u - v).abs().max()), 6), round(C.pcc(u.flatten(), v.flatten()), 6)]
        if L in tp_out[2]:
            u, v = tp_out[2][L], dec_out[2][L]
            per[f"L{L}.lat"] = [round(float((u - v).abs().max()), 6), round(C.pcc(u.flatten(), v.flatten()), 6),
                                round(C.pcc(u[-64:].flatten(), v[-64:].flatten()), 6)]  # fmt: skip
    lt, ld = tp_out[0], dec_out[0]
    worst = sorted(per.items(), key=lambda kv_: kv_[1][1])[:6]
    first_bad = next((k_ for k_, v_ in per.items() if v_[1] < 0.999), None)
    c_, t_ = pf.plan(0, n)
    row = dict(test=f"p2 TF A/B: {n}-token prompt prefilled by the TP prefill vs the decode stack (decode-trace replay per "
                    f"token) in ONE build, per-layer state PCC + the last logits, tp{TPN} layers {len(LAYERS)}",
               milestone="P2" if len(LAYERS) == cfg.num_hidden_layers else "P2-prefix", status="info", tp=TPN,
               layers=len(LAYERS), n=n, bucket=c_, mode=pf.mode(c_), logits_pcc=round(C.pcc(lt, ld), 6),
               logits_max_abs=round(float((lt - ld).abs().max()), 4), argmax=[int(lt.argmax()), int(ld.argmax())],
               top5_overlap=len(set(lt.topk(5).indices.tolist()) & set(ld.topk(5).indices.tolist())),
               first_layer_below_0999=first_bad, worst=worst, prefill_s=[round(t1 - t0, 1), round(t2 - t1, 1)], per=per,
               golden=gold, state_delta_vs_prev_n=delta, tag=tag)  # fmt: skip
    S.log({k_: v_ for k_, v_ in row.items() if k_ != "per"})
    S.log(f"  per: {per}")
    rows.append(row)
if args.get("tf", True) and "ns" not in args:
    if args.get("padmask"):  # the pad-row MoE mask (GLM53_PF_MOE_PADMASK) as the TP prefill, eager (its own programs)
        pf.moe_padmask = True
        pf.traces = {}
    for tp in (True, False):
        backend.tpp = pf if tp else None
        try:
            t3 = time.time()
            s = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(REF), mesh_device=mesh,
                                    build_kwargs=dict(generator=gen))[0]  # fmt: skip
        finally:
            backend.tpp = pf
        r = dict(test=f"p2 TF A/B (Stage 6 harness, same build): prefill by the {'TP prefill' if tp else 'decode stack'} "
                      f"then forced decode, tp{TPN} layers {len(LAYERS)}", milestone="P2" if len(LAYERS) ==
                 cfg.num_hidden_layers else "P2-prefix", status="pass" if s["top5"] >= 0.98 else "fail", tp=TPN,
                 layers=len(LAYERS), prefill="tp" if tp else "decode", top1=s["top1"], top5=s["top5"],
                 top100=s["top100"], matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]],
                 wall_s=round(time.time() - t3, 1), reference=REF, tag=tag)  # fmt: skip
        S.log(r)
        rows.append(r)
for r in rows:
    if rec:
        C.record("pcc", r)
RESULT = rows
