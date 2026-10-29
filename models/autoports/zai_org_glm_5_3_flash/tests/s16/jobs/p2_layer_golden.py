# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 TP8 numerics (program 2026-10-22 07:23Z: TP8 misses golden tokens at
large-margin positions -> name the first layer / op where TP8 diverges). Prompts through the TP prefill with in-program
probes (GLM53_PF_PROBE_LAYERS: each probed layer's output residual copied inside the trace) -> per probed layer, the
prompt's rows vs the CPU fp32 golden layer_io (causal: a longer prompt's first rows = the shorter prompt's rows): PCC,
max |diff|, relative L2 error and the norm ratio (PCC is scale-blind); plus the first token's golden rank.

Cases: "g<n>" = the golden 2,048-token prompt's first n tokens (goldens/layer_io/L<L>/2048 rows < n, golden =
goldens/logits/2048.prefill.pt top-32 at n - 1); "aime161" / "aime162" / "aime164" = the AIME24 readiness prompt (161
ids) + its first 0 / 1 / 3 golden continuation tokens (goldens_gpu_fp32_aime24/layer_io/L<L>/161 rows < 161, golden =
the GPU fp32 continuation logits: exact rank and margin).

construction: "s16" (Stage 16's own build: SM.build_model_backend, B = 1, weight cache) | "m5" (14b's SERVED
construction, M5.serving: B (32) decode rows, native widths, weights from the TP image when GLM53_WEIGHT_SOURCE=image;
the env = args env, e.g. work-s14b/int25_svc_env.done's -- job zero's build). The built TPPrefill is replaced by one
with the probes (p2_retpp_ttft's rebuild on the same backend / KV cache). entry: "chunk" (StackBackend.prefill_chunk:
last-row logits) | "generator" (job zero's: Glm53Generator.prefill_forward on identity rows, slot 0, device greedy).
Variants (env overlays, e.g. {"GLM53_PF_REDUCE_FP32": "1"}) rebuild the TPPrefill again and repeat.

ARGS: layers ("all" | list), cases (["g96", "g130"]), probe ([0, 1, 2, 3, 4, 7, 43, 44] within layers), variants ([{}]),
construction ("s16"), b (32, m5), native (true, m5), rev ("t25b", m5), entry ("chunk"), env, max_seq (2560), tag,
record (True)."""
import gc
import os
import sys
import time

import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

C = S.C
WS = "/opt/glm53/kit"
GOLD, AIME = f"{WS}/goldens", f"{WS}/goldens_gpu_fp32_aime24"
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = list(range(cfg.num_hidden_layers)) if args.get("layers", "all") == "all" else [int(v) for v in args["layers"]]
CASES = list(args.get("cases", ["g96", "g130"]))
PROBE = [L for L in (int(v) for v in args.get("probe", [0, 1, 2, 3, 4, 7, 43, 44])) if L in LAYERS]
VARIANTS = list(args.get("variants", [{}]))
CONS = args.get("construction", "s16")
ENTRY = args.get("entry", "chunk")
MAXSEQ = int(args.get("max_seq", 2560))
tag = args.get("tag", "p2-layer-golden")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
VOCAB = int(cfg.vocab_size)
GREEDY = SamplingParams()

if CONS == "s16":  # Stage 16's own build knobs (the m5 construction takes args env as is: job zero's served env)
    os.environ["GLM53_STACK_PREFILL"] = "tp"
    os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
    os.environ["GLM53_PF_MOE"] = "ep"
    os.environ["GLM53_PF_MHC"] = "comp"
    os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
    os.environ["GLM53_PF_MOE_V2"] = "1"
    os.environ["GLM53_PF_KDA_SCAN"] = "op"
    os.environ["GLM53_PF_DSA"] = "0"
    os.environ["GLM53_PF_SPARSE_T"] = ""
    os.environ["GLM53_PF_BATCH"] = "0"
    os.environ["GLM53_PF_PACKED"] = ""


def set_env(variant: dict, probes: bool):
    if CONS == "s16":
        S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, "rep_buckets": [128], **args})
    else:  # job zero's construction: its env only (S.set_attn would reset the P3 knobs to Stage 16's defaults)
        for k, v in (args.get("env") or {}).items():
            os.environ[str(k)] = str(v)
    os.environ["GLM53_PF_REDUCE_FP32"] = "0"  # every variant names its own (else a leak from the previous one)
    os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(L) for L in PROBE) if probes else ""
    for k, v in variant.items():
        os.environ[str(k)] = str(v)


t0 = time.time()
if CONS == "m5":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s14b"))
    import m5common as M5  # noqa: E402

    set_env(VARIANTS[0], probes=False)  # the served build as job zero had it (its traces without probes)
    key = M5.serving(ctx, mesh, SM, G, B=int(args.get("b", 32)), ids=LAYERS, cfg=cfg,
                     native=bool(args.get("native", True)), rev=args.get("rev", "t25b"), log=S.log)  # fmt: skip
    backend, gen = ctx[key]["backend"], ctx[key]["gen"]
    rebuild = True
else:
    set_env(VARIANTS[0], probes=True)
    full = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "s16-full" and k[1] == "tp" and list(k[2]) == LAYERS]
    if full:  # a same-process build of these layers: rebuild only its TPPrefill (with the probes) below
        backend, gen = ctx[full[-1]]["backend"], ctx[full[-1]]["gen"]
        rebuild = True
        S.log(f"adopted {full[-1][:3]}")
    else:
        for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend",
                                                                              "s16-chunks")]:  # fmt: skip
            old = ctx.pop(k)
            if old.get("pf") is not None:
                old["pf"].release()
            old["backend"].close()
        gc.collect()
        backend = SM.build_model_backend(None, mesh, max_batch=1, layers=LAYERS, hf_config=cfg,
                                         weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=1,
                                max_seq_len=MAXSEQ, stop_token_ids=())  # fmt: skip
        gen.warmup()
        ctx[("s16-full", "tp", tuple(LAYERS), "layer-golden")] = dict(backend=backend, gen=gen)
        rebuild = False
assert getattr(backend, "tpp", None) is not None, "needs the TP prefill attached (GLM53_STACK_PREFILL=tp at build)"
S.log(f"backend ({CONS}) ready in {time.time() - t0:.1f} s (rebuild TPPrefill: {rebuild})")
b = gen._own_binding()  # job zero's binding (its identity rows / KV cache); the TPPrefill is attached to it
kv = b.kv_cache
assert backend.tpp.kv is kv, "the TPPrefill is not on the generator's own cache"

# ---------------------------------------------------------------------------------------------------- the cases
g2048 = torch.load(f"{GOLD}/prompts/2048.pt", map_location="cpu", weights_only=True).reshape(-1).long()
gtop = torch.load(f"{GOLD}/logits/2048.prefill.pt", map_location="cpu", weights_only=True)
aref = torch.load(f"{AIME}/readiness_aime24_chat.refpt", map_location="cpu", weights_only=False)["entries"][0]
aime = aref["prompt_tokens"].reshape(-1).long()
agen = aref["generated_tokens"].reshape(-1).long()
alog = torch.load(f"{AIME}/logits/161.pt", map_location="cpu", weights_only=True, mmap=True)["continuation_logits"]


def case(name: str) -> dict:
    """tokens, the per-layer golden source (dir, rows compared), the golden of the next token"""
    if name.startswith("g"):
        n = int(name[1:])
        vals, ids = gtop["prompt_top32_values"][n - 1], gtop["prompt_top32_indices"][n - 1]
        return dict(toks=g2048[:n], lio=(f"{GOLD}/layer_io", 2048, n), top_ids=ids, top_vals=vals, full=None)
    j = int(name[4:]) - 161  # aime16x: the prompt + its first j golden continuation tokens
    full = alog[j].float()
    v, i = full.topk(32)
    return dict(toks=torch.cat([aime, agen[:j]]), lio=(f"{AIME}/layer_io", 161, 161), top_ids=i, top_vals=v, full=full)


_lio: dict = {}


def layer_golden(src: str, length: int, L: int) -> torch.Tensor:
    if (src, L) not in _lio:
        _lio[(src, L)] = torch.load(f"{src}/L{L:02d}/{length}/streams_out.pt", map_location="cpu", weights_only=True,
                                    mmap=True)[0].float()  # fmt: skip
    return _lio[(src, L)]


def stats(dev: torch.Tensor, ref: torch.Tensor) -> dict:
    d, r = dev.reshape(-1).double(), ref.reshape(-1).double()
    return dict(pcc=round(float(C.pcc(dev, ref)), 6), max_abs=round(float((d - r).abs().max()), 5),
                rel_l2=round(float((d - r).norm() / r.norm().clamp_min(1e-30)), 6),
                norm_ratio=round(float(d.norm() / r.norm().clamp_min(1e-30)), 6))  # fmt: skip


def ranks(cs: dict, tok: int, logits=None) -> dict:
    ids, vals = cs["top_ids"], cs["top_vals"]
    hit = (ids == tok).nonzero()
    out = dict(first=tok, golden_top1=int(ids[0]), golden_margin=round(float(vals[0] - vals[1]), 4),
               dev_rank_in_golden=int(hit[0, 0]) + 1 if hit.numel() else None)  # fmt: skip
    if cs["full"] is not None:  # the full golden logits (AIME): the device token's exact golden rank / logit gap
        out["dev_rank_in_golden_full"] = int((cs["full"] > cs["full"][tok]).sum()) + 1
        out["dev_gap_to_golden_top1"] = round(float(cs["full"].max() - cs["full"][tok]), 4)
    if logits is not None:  # the golden token's rank in the device logits
        lg = logits.reshape(-1)[:VOCAB].float()
        out["golden_rank_in_dev"] = int((lg > lg[int(ids[0])]).sum()) + 1
    return out


def run(cs: dict):
    toks = cs["toks"]
    n = int(toks.numel())
    if ENTRY == "generator":  # job zero's entry
        gen.reset()
        first = gen.prefill_forward(toks.reshape(1, -1), page_table=b.identity_row(0).reshape(1, -1),
                                    kv_cache=b.kv_cache, prompt_lens=[n], empty_slots=[0],
                                    sampling_params=[GREEDY])  # fmt: skip
        ttnn.synchronize_device(mesh)
        gen.reset()
        return int(first[0]), None
    logits = backend.prefill_chunk(toks, kv_cache=kv, slot=0, chunk_start=0, n_valid=n, logits_rows="last",
                                   io=b.io).float()  # fmt: skip
    ttnn.synchronize_device(mesh)
    return int(logits.reshape(-1)[:VOCAB].argmax()), logits


rows = []
for vi, variant in enumerate(VARIANTS):
    if vi or rebuild:  # this variant's TPPrefill (with the probes) on the same backend / KV cache
        set_env(variant, probes=True)
        t1 = time.time()
        backend.tpp.release()
        backend.tpp = None
        backend.prepare_prefill(kv)
        backend.tpp.capture()
        gen.reset()
        ttnn.synchronize_device(mesh)
        b = gen._own_binding()
        S.log(f"TPPrefill rebuilt for {variant} in {time.time() - t1:.1f} s; traces {sorted(map(str, backend.tpp.traces))}")
    pf = backend.tpp
    assert pf.probe_layers == set(PROBE), (pf.probe_layers, PROBE)
    for name in CASES:
        cs = case(name)
        n = int(cs["toks"].numel())
        eager0 = pf.counters["eager"]
        tok, logits = run(cs)
        c, _ = pf.plan(0, n)
        src, length, nr = cs["lio"]
        per_layer = {}
        for L in PROBE:
            dev = pf.probe(c, L)[:nr]
            ref = layer_golden(src, length, L)[:nr]
            per_layer[f"L{L}"] = stats(dev, ref)
            per_layer[f"L{L}"]["last_row"] = stats(dev[nr - 1], ref[nr - 1])
        row = dict(test=f"per-layer hidden states vs the CPU fp32 golden (layer_io, the prompt's first {nr} rows) + the "
                        f"first token's golden rank, {name} ({n} ids), {CONS} construction, entry {ENTRY}, traced with "
                        f"in-program probes, tp{TPN} layers {len(LAYERS)}", milestone="P2", status="info", case=name,
                   n=n, rows_compared=nr, bucket=c, mode=pf.mode(c), construction=CONS, entry=ENTRY, variant=variant,
                   traced=pf.counters["eager"] == eager0, layers=len(LAYERS), tp=TPN, rank=ranks(cs, tok, logits),
                   per_layer=per_layer, tag=tag)  # fmt: skip
        S.log({k: v for k, v in row.items() if k not in ("per_layer", "test")})
        S.log({k: (v["pcc"], v["rel_l2"], v["last_row"]["rel_l2"]) for k, v in per_layer.items()})
        rows.append(row)
        if rec:
            C.record("pcc", row)
RESULT = rows
