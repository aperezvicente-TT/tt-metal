# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 pad-row leak probe (program 2026-10-22 05:26Z: the TP prefill's last valid
rows are wrong in a pad-count-dependent way at TP8 -- do the bucket's PAD rows leak into the valid rows?). The first n
tokens of the reference prompt through the production entry (StackBackend.prefill_chunk -> TPPrefill, traced) with the
bucket's pad tokens = 0 (production), = random ids (two seeds), = the previous request's tokens; per mode: the last
position's logits, every KDA layer's S / C, every sparse_dsa layer's latent rows < n -- vs the zeros run (any change =
a leak: causal attention and per-row ops cannot see the pads) and the zeros run vs the prefix golden (args golden_file).
Then (args flips) the same with one knob flipped at a time, run EAGER (the captured traces hold the served programs):
"moe_v1" (GLM53_PF_MOE_V2=0), "ep_rm" (GLM53_PF_MOE_EP_V2C=0), "xffn_valid" (pad rows skipped by xffn).

ARGS: layers ([0..4] | "all"), ns ([97, 161]), modes (["zeros", "rand1", "rand2", "prev"]), flips ([]: moe_v1, ep_rm,
xffn_valid, mfs, mhc_ops (eager), bucket512 (traced)), rep_buckets ([128]), golden_file (prefix logits) | golden_full
(the full model's top-32 per position), states (True: KDA / latent snapshots; False: logits only), ref, max_seq (2560),
tag, record (True)."""
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
LAYERS = (
    list(range(cfg.num_hidden_layers))
    if args.get("layers") == "all"
    else [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
)
NS = [int(v) for v in args.get("ns", [97, 161])]
MODES = list(args.get("modes", ["zeros", "rand1", "rand2", "prev"]))
FLIPS = list(args.get("flips", []))
FLIP_NS = [int(v) for v in args.get("flip_ns", [])]  # the flips' lengths / pad modes (default: ns / modes)
FLIP_MODES = list(args.get("flip_modes", []))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
MAXSEQ = int(args.get("max_seq", 2560))
STATES = bool(args.get("states", True))
tag = args.get("tag", "p2-pad-leak")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

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
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, "rep_buckets": [128], **args})

full = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "s16-full" and k[1] == "tp" and list(k[2]) == LAYERS]
t0 = time.time()
if full:  # a same-process TP-prefill build of these layers (p2_tf_ab / p2_full): no second backend (L1 floor)
    backend, gen = ctx[full[-1]]["backend"], ctx[full[-1]]["gen"]
    S.log(f"adopted {full[-1][:3]}")
else:
    for k in [
        k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-chunks")
    ]:
        old = ctx.pop(k)
        if old.get("pf") is not None:
            old["pf"].release()
        old["backend"].close()
    gc.collect()
    backend = SM.build_model_backend(None, mesh, max_batch=1, layers=LAYERS, hf_config=cfg,
                                     weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=1, max_seq_len=MAXSEQ,
                            stop_token_ids=())  # fmt: skip
    gen.warmup()
    ctx[("s16-full", "tp", tuple(LAYERS), "pad-leak")] = dict(backend=backend, gen=gen,
                                                               eager0=backend.tpp.counters["eager"])  # fmt: skip
pf = backend.tpp
b = gen._own_binding()
kv, io = b.kv_cache, b.io
S.log(f"built + warmed in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}")
prompt_all = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
GOLD = torch.load(args["golden_file"], map_location="cpu", weights_only=True) if args.get("golden_file") else None
# golden_full: the full model's CPU fp32 golden top-32 per prompt position (goldens/logits/2048.prefill.pt; its prompt
# = goldens/prompts/2048.pt, whose first 2,016 tokens are the full refpt's prompt)
GFULL = torch.load(args["golden_full"], map_location="cpu", weights_only=True) if args.get("golden_full") else None
VOCAB = int(cfg.vocab_size)

PAD = {"mode": "zeros", "prev": None}
_orig_write = PT.TPPrefill.write_inputs


def _write_inputs(self, c, tokens, *, slot, chunk_start, n_valid, ci=None, **kw):
    _orig_write(self, c, tokens, slot=slot, chunk_start=chunk_start, n_valid=n_valid, ci=ci, **kw)
    if PAD["mode"] == "zeros" or ci is not None:
        return
    tok = torch.zeros(1, c, dtype=torch.int32)
    tok[0, :n_valid] = tokens.reshape(-1)[:n_valid].to(torch.int32)
    if PAD["mode"] == "prev" and PAD["prev"] is not None:
        tok[0, n_valid:] = PAD["prev"][n_valid:c].to(torch.int32)
    elif PAD["mode"].startswith("rand"):
        g = torch.Generator().manual_seed(int(PAD["mode"][4:] or 1))
        tok[0, n_valid:] = torch.randint(1, VOCAB, (c - n_valid,), generator=g, dtype=torch.int32)
    sp = self.mode(c) == "sp"
    mp = ttnn.ShardTensorToMesh(self.mesh, dim=-1) if sp else self.repl
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                                    mesh_mapper=mp), self.chunk_inputs(c).tokens)  # fmt: skip


PT.TPPrefill.write_inputs = _write_inputs


def run(n):
    out = backend.prefill_chunk(prompt_all[:n], kv_cache=kv, slot=0, chunk_start=0, n_valid=n, logits_rows="last",
                                io=io).float().reshape(-1)[:VOCAB]  # fmt: skip
    ttnn.synchronize_device(mesh)
    if not STATES:  # logits only (the full model at TP8: ~45 layer snapshots per run cost seconds each)
        return out, {}, {}
    lat = {L: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()
           for L, t_ in backend.latent_pages(kv).items()}  # fmt: skip
    blocks = b.pt_shadow[0]
    lr = {L: torch.cat([c_[int(blocks[p]), 0] for p in range(-(-n // 64))], dim=0)[:n] for L, c_ in lat.items()}
    kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, 0)}
    return out, kda, lr


def diff(a, z):
    d = {"logits": [round(float((a[0] - z[0]).abs().max()), 5), round(C.pcc(a[0], z[0]), 6)]}
    for L in z[1]:
        for nm, i in (("S", 0), ("C", 1)):
            d[f"L{L}.{nm}"] = round(float((a[1][L][i] - z[1][L][i]).abs().max()), 6)
    for L in z[2]:
        d[f"L{L}.lat"] = round(float((a[2][L] - z[2][L]).abs().max()), 6)
    return d


def vs_golden(n, lg):
    if GFULL is not None:
        ix = GFULL["prompt_top32_indices"][n - 1]
        am = int(lg.argmax())
        g1 = int(ix[0])
        rank_of_golden = int((lg > lg[g1]).sum()) + 1  # 1 = the device's top-1 is the golden's
        top32 = ix[:32].tolist()
        return dict(argmax=am, golden_argmax=g1, golden_rank=rank_of_golden, in_golden_top5=am in top32[:5],
                    dev_rank_in_golden=(top32.index(am) + 1) if am in top32 else None,
                    top5_overlap=len(set(lg.topk(5).indices.tolist()) & set(top32[:5])))  # fmt: skip
    if GOLD is None or n not in GOLD["ns"]:
        return None
    gl = GOLD["logits"][GOLD["ns"].index(n)].float()[:VOCAB]
    g1 = int(gl.argmax())
    gtop = gl.topk(2).values
    return dict(pcc=round(C.pcc(lg, gl), 6), argmax=int(lg.argmax()), golden_argmax=g1,
                golden_rank=int((lg > lg[g1]).sum()) + 1,  # 1 = the device's top-1 is the golden's
                golden_margin=round(float(gtop[0] - gtop[1]), 4),  # the fp32 golden's own top-1 - top-2
                dev_margin_vs_golden=round(float(lg.max() - lg[g1]), 4),  # how far the device's top-1 is above it
                top5_overlap=len(set(lg.topk(5).indices.tolist()) & set(gl.topk(5).indices.tolist())))  # fmt: skip


rows = []
for flip in [None] + FLIPS:
    if flip is not None:  # one knob flipped, eager forwards (the traces keep the served programs)
        parts = flip.split("+")  # compound flips: "nopad+mfs" = the padding mask off and xffn off together
        table = {"moe_v1": {"GLM53_PF_MOE_V2": "0"}, "ep_rm": {"GLM53_PF_MOE_EP_V2C": "0"},
                 "xffn_valid": {"GLM53_PF_MOE_XFFN_VALID": "1"}, "mfs": {"GLM53_PF_MOE_FFN": "mfs"},
                 "xffn": {"GLM53_PF_MOE_FFN": "xffn"}}  # fmt: skip
        env = {k_: v_ for p_ in parts for k_, v_ in table.get(p_, {}).items()}
        keep = {k_: os.environ.get(k_) for k_ in env}
        os.environ.update(env)
        S.set_moe_v2(pf, os.environ.get("GLM53_PF_MOE_V2") == "1")
        keep_attr = dict(mhc_post=pf.mhc_post, buckets=pf.buckets, padmask=pf.moe_padmask)
        if "mhc_ops" in parts:  # the per-op mHC post path (TPPrefill.layer reads the attribute per call)
            pf.mhc_post = "ops"
        if "padmask" in parts:  # the fix candidate: padding rows out of the routed MoE (eager; its inputs allocate now)
            pf.moe_padmask = True
        if "nopad" in parts:  # a PADMASK=1 build without the mask (the unfixed path, eager)
            pf.moe_padmask = False
        traces = pf.traces
        if parts == ["bucket512"]:  # chunks <= 128 in the 512 bucket (sequence-parallel at TP8): its captured trace
            pf.buckets = tuple(b_ for b_ in pf.buckets if b_ != 128)
        elif len(parts) == 1 and parts[0].startswith("drop"):  # "drop256": the same build without that bucket, so the
            # prompt runs in the next captured one (traced; the 06:34Z short-prompt defect: 256 vs 512 in one process)
            pf.buckets = tuple(b_ for b_ in pf.buckets if b_ != int(parts[0][4:]))
        else:  # eager forwards: the traces hold the served programs
            pf.traces = {}
    ns_, modes_ = (NS, MODES) if flip is None else (FLIP_NS or NS, FLIP_MODES or MODES)
    for n in ns_:
        res = {}
        for mode in modes_:
            PAD["mode"] = mode
            res[mode] = run(n)
            PAD["prev"] = torch.cat([prompt_all[n:], prompt_all[:n]])[:2048]  # a different request's tokens
        z = res[modes_[0]]
        leak = {m: diff(res[m], z) for m in modes_[1:]}
        leaked = any(v_ != 0.0 for d_ in leak.values() for k_, v_ in d_.items() if k_ != "logits") or \
            any(d_["logits"][0] != 0.0 for d_ in leak.values())  # fmt: skip
        c_, t_ = pf.plan(0, n)
        row = dict(test=f"pad-row leak probe: {n}-token prompt in the {c_} bucket ({pf.mode(c_)}), pad tokens "
                        f"{modes_} -> valid-row logits / KDA states / latent rows vs the zeros run"
                        f"{'' if flip is None else f' [{flip}, eager]'}, tp{TPN} layers {len(LAYERS)}",
                   milestone="P2", status="info", flip=flip or "served", n=n, bucket=c_, mode=pf.mode(c_),
                   reduce_form=getattr(pf, "reduce_form", "rs"),
                   leak=leaked, leak_detail=leak, golden={m: vs_golden(n, res[m][0]) for m in modes_}, tp=TPN,
                   layers=len(LAYERS), tag=tag)  # fmt: skip
        S.log(row)
        rows.append(row)
    if flip is not None:
        pf.traces = traces
        pf.mhc_post, pf.buckets = keep_attr["mhc_post"], keep_attr["buckets"]
        pf.moe_padmask = keep_attr["padmask"]
        for k_, v_ in keep.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
        S.set_moe_v2(pf, True)
PT.TPPrefill.write_inputs = _orig_write
PAD["mode"] = "zeros"
# conts ([[128, 227]]: the 2026-10-23 06:34Z short-prompt defect's second form): the prompt as a cached HIT of h tokens
# (a head-less chunk [0, h)) + a continuation chunk [h, n) with the last-row logits, vs the cold n-token run and the golden
CONT_A = {}  # (h, n) -> pass A's continuation logits (ab_env compares against them)
for h, n in [tuple(int(v) for v in hc) for hc in args.get("conts", [])]:
    cold = run(n)[0] if n <= max(pf.buckets) else None  # (a prompt longer than the largest bucket has no cold form)
    backend.prefill_chunk(prompt_all[:h], kv_cache=kv, slot=0, chunk_start=0, n_valid=h, logits_rows="none", io=io)
    out = backend.prefill_chunk(prompt_all[h:n], kv_cache=kv, slot=0, chunk_start=h, n_valid=n - h, logits_rows="last",
                                io=io).float().reshape(-1)[:VOCAB]  # fmt: skip
    ttnn.synchronize_device(mesh)
    CONT_A[(h, n)] = out.clone()
    if cold is None:
        cold = out
    b_cold = pf.plan(0, n)[0] if n <= max(pf.buckets) else None
    row = dict(test=f"short-prompt continuation probe: {h}-token hit + {n - h}-token continuation (buckets "
                    f"{pf.plan(0, h)[0]} / {pf.plan(h, n - h)[0]}) vs the cold {n}-token prompt ({b_cold} bucket) "
                    f"and the fp32 golden, tp{TPN} layers {len(LAYERS)}", milestone="P2", status="info", flip="cont",
               n=n, hit=h, bucket_hit=pf.plan(0, h)[0], bucket_cont=pf.plan(h, n - h)[0], bucket_cold=b_cold,
               cont_vs_cold=[round(float((out - cold).abs().max()), 5), round(C.pcc(out, cold), 6)],
               reduce_form=getattr(pf, "reduce_form", "rs"),
               golden_cont=vs_golden(n, out), golden_cold=vs_golden(n, cold), tp=TPN, layers=len(LAYERS), tag=tag)  # fmt: skip
    S.log(row)
    rows.append(row)
# form2 ("a2a"): the SAME build re-captured under another cross-chip reduce form (GLM53_PF_REDUCE): every trace
# released, the form switched, every key compiled (eager, the one-time compile) and captured again, then the same
# prompts TRACED -- cold in the served bucket set, cold with each dropNNN flip, and the conts (2026-10-23 06:54Z: rs first,
# a2a as the diagnostic twin of the short-prompt defect)
FORM2 = args.get("form2")
if FORM2:
    t2 = time.time()
    pf.release()
    pf.reduce_form = str(FORM2)
    pf._warmed = False
    pf.compile(pf.buckets)
    pf.capture(pf.buckets)
    S.log(f"{tag}: re-captured under GLM53_PF_REDUCE={FORM2} in {time.time() - t2:.1f} s; traces {sorted(map(str, pf.traces))}")
    eager2 = pf.counters["eager"]
    for drop in [None] + [int(f_[4:]) for f_ in FLIPS if f_.startswith("drop")]:
        keep_b = pf.buckets
        if drop:
            pf.buckets = tuple(b_ for b_ in pf.buckets if b_ != drop)
        for n in (FLIP_NS or NS):
            PAD["mode"] = "zeros"
            out = run(n)[0]
            c_, t_ = pf.plan(0, n)
            row = dict(test=f"short-prompt probe under GLM53_PF_REDUCE={FORM2}: {n}-token prompt in the {c_} bucket "
                            f"({pf.mode(c_)}), traced, vs the golden, tp{TPN} layers {len(LAYERS)}", milestone="P2",
                       status="info", flip=f"form2{'' if not drop else f'+drop{drop}'}", n=n, bucket=c_, mode=pf.mode(c_),
                       reduce_form=pf.reduce_form, golden={"zeros": vs_golden(n, out)}, tp=TPN, layers=len(LAYERS),
                       tag=tag)  # fmt: skip
            S.log(row)
            rows.append(row)
        pf.buckets = keep_b
    for h, n in [tuple(int(v) for v in hc) for hc in args.get("conts", [])]:
        cold = run(n)[0]
        backend.prefill_chunk(prompt_all[:h], kv_cache=kv, slot=0, chunk_start=0, n_valid=h, logits_rows="none", io=io)
        out = backend.prefill_chunk(prompt_all[h:n], kv_cache=kv, slot=0, chunk_start=h, n_valid=n - h,
                                    logits_rows="last", io=io).float().reshape(-1)[:VOCAB]  # fmt: skip
        ttnn.synchronize_device(mesh)
        row = dict(test=f"short-prompt continuation probe under GLM53_PF_REDUCE={FORM2}: {h}-token hit + {n - h} vs cold "
                        f"{n}, tp{TPN} layers {len(LAYERS)}", milestone="P2", status="info", flip="form2+cont", n=n, hit=h,
                   bucket_hit=pf.plan(0, h)[0], bucket_cont=pf.plan(h, n - h)[0], bucket_cold=pf.plan(0, n)[0],
                   cont_vs_cold=[round(float((out - cold).abs().max()), 5), round(C.pcc(out, cold), 6)],
                   reduce_form=pf.reduce_form, golden_cont=vs_golden(n, out), golden_cold=vs_golden(n, cold), tp=TPN,
                   layers=len(LAYERS), tag=tag)  # fmt: skip
        S.log(row)
        rows.append(row)
    S.log(f"{tag}: form2 pass eager forwards after its capture: {pf.counters['eager'] - eager2} (0 = all traced)")
# ab_env ({"GLM53_PF_BQ_HEAD_BLOCK": "16"}): the SAME build re-captured with the env changed (release -> env -> compile
# -> capture, traced), the conts replayed and their logits compared BITWISE with pass A (a knob that must not change bits)
AB_ENV = {str(k_): str(v_) for k_, v_ in (args.get("ab_env") or {}).items()}
if AB_ENV:
    t3 = time.time()
    pf.release()
    os.environ.update(AB_ENV)
    pf._warmed = False
    pf.compile(pf.buckets)
    pf.capture(pf.buckets)
    S.log(f"{tag}: re-captured under {AB_ENV} in {time.time() - t3:.1f} s")
    eager3 = pf.counters["eager"]
    for (h, n), out_a in CONT_A.items():
        backend.prefill_chunk(prompt_all[:h], kv_cache=kv, slot=0, chunk_start=0, n_valid=h, logits_rows="none", io=io)
        out = backend.prefill_chunk(prompt_all[h:n], kv_cache=kv, slot=0, chunk_start=h, n_valid=n - h,
                                    logits_rows="last", io=io).float().reshape(-1)[:VOCAB]  # fmt: skip
        ttnn.synchronize_device(mesh)
        d_ = (out - out_a).abs()
        row = dict(test=f"env A/B in one build: {AB_ENV} vs pass A, {h}-token chunk + {n - h}-token continuation "
                        f"(buckets {pf.plan(0, h)[0]} / {pf.plan(h, n - h)[0]}), last-row logits, traced, tp{TPN} layers "
                        f"{len(LAYERS)}", milestone="P2", flip="ab_env", ab_env=AB_ENV, n=n, hit=h,
                   bitwise=bool(d_.max() == 0), max_abs=round(float(d_.max()), 6), pcc=round(C.pcc(out, out_a), 7),
                   argmax_a=int(out_a.argmax()), argmax_b=int(out.argmax()), tp=TPN, layers=len(LAYERS), tag=tag)  # fmt: skip
        row["status"] = "pass" if row["bitwise"] else "fail"
        S.log(row)
        rows.append(row)
    S.log(f"{tag}: ab_env pass eager forwards after its capture: {pf.counters['eager'] - eager3} (0 = all traced)")
for r in rows:
    if rec:
        C.record("pcc", r)
RESULT = rows
