# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 batched admission (GLM53_PF_BATCH=4, the service knob) with vLLM page rows: four
fresh requests of different lengths in ONE Glm53Generator.prefill_forward call with device sampling (-> _admit_batch ->
one replay of the batched trace: every request in its own 2,048-row bucket) vs the same requests one call each
(the per-user TP prefill), both with vLLM rows (blocks held only, generator_vllm.vllm_page_rows), and a reference pass
with full-coverage rows. Per request: the first token, D teacher-forced decode steps (argmax / PCC vs the reference),
the KDA S / C and the latent rows < n (max |diff|). `mappings` as in p2_chunk_states (legacy = the pre-fix chunk page
mapping). Rows: pcc.jsonl.

ARGS: layers ([0..4]), env ({}: knob overlay after the defaults), lens ([150, 22, 300, 90]), d (8), mappings (["legacy", "fixed"]), admit ("batch": the equal-
segment batched trace, GLM53_PF_BATCH=4; "packed": the mixed-length packed trace, GLM53_PF_PACKED=2048:4), ref, tag,
record (True)."""
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
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
LENS = [int(v) for v in args.get("lens", [150, 22, 300, 90])]
D = int(args.get("d", 8))
MAPPINGS = list(args.get("mappings", ["legacy", "fixed"]))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
tag = args.get("tag", "p4-batch-rows")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
NB, MAXSEQ, BLK = 4, 2560, 64
BPS = MAXSEQ // BLK
GREEDY = SamplingParams()
assert len(LENS) == NB

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = "ep"
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_MOE_V2"] = "1"
os.environ["GLM53_PF_KDA_SCAN"] = "op"
os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_SPARSE_T"] = ""
ADMIT = args.get("admit", "batch")
os.environ["GLM53_PF_BATCH"] = str(NB) if ADMIT == "batch" else "0"
os.environ["GLM53_PF_PACKED"] = f"2048:{NB}" if ADMIT == "packed" else ""
for k_ in ("GLM53_PF_MOE_FFN", "GLM53_PF_MOE_XFFN_XSPLIT", "GLM53_PF_MOE_XFFN_SNAKE", "GLM53_PF_MOE_XFFN_VALID"):
    os.environ.pop(k_, None)
ENV = {k_: str(v_) for k_, v_ in (args.get("env") or {}).items()}  # the variant's knobs (e.g. GLM53_PF_MOE_PADMASK=1)
os.environ.update(ENV)
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, **args})

prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
OFFS = [0, 400, 800, 1200]  # each request a different slice of the reference prompt
assert all(o + n + D <= prompt.numel() for o, n in zip(OFFS, LENS))
toks = [prompt[o : o + n + D] for o, n in zip(OFFS, LENS)]

for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-chunks")]:
    old = ctx.pop(k)
    if old.get("pf") is not None:
        old["pf"].release()
    old["backend"].close()
gc.collect()
t0 = time.time()
backend = SM.build_model_backend(None, mesh, max_batch=NB, layers=LAYERS, hf_config=cfg,
                                 weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=NB, max_seq_len=MAXSEQ,
                        stop_token_ids=())  # fmt: skip
kv = gen.allocate_kv_cache(max_batch=NB, max_seq_len=MAXSEQ)
gen.warmup(kv)
ctx[("s16-chunks", tuple(LAYERS), TPN, "batch")] = dict(backend=backend, gen=gen, pf=backend.tpp)
pf = backend.tpp
assert pf is not None and pf.kv is kv, "the TP prefill on the serving cache"
assert pf.batch_keys if ADMIT == "batch" else pf.packed_keys, f"no {ADMIT} trace keys"
S.log(f"built + warmed in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}, admit {ADMIT}")
COUNTER = "prefill_batches" if ADMIT == "batch" else "prefill_packed"
eager0 = pf.counters["eager"]


def legacy_pages(row, chunk_start, n_valid, c):  # the pre-fix mapping (write_inputs up to 6178a3252bf)
    blocks = row[chunk_start // BLK : chunk_start // BLK + c // BLK]
    if blocks.numel() < c // BLK:
        blocks = torch.cat([blocks, blocks[-1:].expand(c // BLK - blocks.numel())])
    return torch.where(blocks < 0, torch.zeros_like(blocks), blocks)


FIXED_PAGES = PT.chunk_pages
VBLK = {k: [10 + k, 20 + k, 30 + k, 15 + k, 25 + k, 35 + k] for k in range(NB)}  # vLLM-style scattered blocks


def row_of(k, upto, full):
    if full:  # full coverage: slot k's identity row shifted by one block (block 0 stays out of every used range)
        return (torch.arange(k * BPS + 1, (k + 1) * BPS + 1) % (NB * BPS)).to(torch.int64)
    r = torch.full((BPS,), -1, dtype=torch.int64)
    blk = VBLK[k][: -(-upto // BLK)]
    r[: len(blk)] = torch.tensor(blk)
    return r


def run(mode, full):
    """mode 'batch': one prefill_forward of the 4 requests (batched admission); 'serial': one call per request"""
    if mode == "batch":
        w = max(LENS)
        tk = torch.zeros(NB, w, dtype=torch.int64)
        for k, n in enumerate(LENS):
            tk[k, :n] = toks[k][:n]
        c0 = gen.counters.get(COUNTER, 0)
        first = gen.prefill_forward(tk, page_table=torch.stack([row_of(k, n, full) for k, n in enumerate(LENS)]),
                                    kv_cache=kv, prompt_lens=LENS, empty_slots=list(range(NB)),
                                    sampling_params=[GREEDY] * NB).tolist()  # fmt: skip
        assert gen.counters.get(COUNTER, 0) == c0 + 1, f"the {ADMIT} admission did not fire"
    else:
        first = [int(gen.prefill_forward(toks[k][:n].reshape(1, -1), page_table=row_of(k, n, full)[None], kv_cache=kv,
                                         prompt_lens=[n], empty_slots=[k], sampling_params=[GREEDY])[0])
                 for k, n in enumerate(LENS)]  # fmt: skip
    blocks = gen._binding(kv).pt_shadow
    lat = {L: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()
           for L, t_ in backend.latent_pages(kv).items()}  # fmt: skip
    snap = []
    for k, n in enumerate(LENS):
        lr = {L: torch.cat([c_[int(blocks[k, p]), 0] for p in range(-(-n // BLK))], dim=0)[:n] for L, c_ in lat.items()}
        kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, k)}
        snap.append(dict(lat=lr, kda=kda))
    dec = []
    for j in range(D):  # all four slots decode together (teacher-forced)
        pos = torch.tensor([n + j for n in LENS])
        dec.append(gen.decode_forward(torch.stack([toks[k][n + j] for k, n in enumerate(LENS)]).reshape(NB, 1), pos,
                                      page_table=torch.stack([row_of(k, n + j + 1, full) for k, n in enumerate(LENS)]),
                                      kv_cache=kv, slots=list(range(NB)), enable_trace=True).float())  # fmt: skip
    return first, snap, torch.stack(dec, dim=1)  # [NB, D, V]


rows = []
for mapping in MAPPINGS:
    PT.chunk_pages = legacy_pages if mapping == "legacy" else FIXED_PAGES
    ref = run("serial", full=True)
    for mode in ("serial", "batch"):
        first, snap, dec = run(mode, full=False)
        for k, n in enumerate(LENS):
            md = 0.0
            for L in ref[1][k]["kda"]:
                for i in (0, 1):
                    md = max(md, float((ref[1][k]["kda"][L][i] - snap[k]["kda"][L][i]).abs().max()))
            ml = max([float((ref[1][k]["lat"][L] - snap[k]["lat"][L]).abs().max()) for L in snap[k]["lat"]] or [0.0])
            am_r, am = ref[2][k].argmax(-1), dec[k].argmax(-1)
            top1 = float((am_r == am).float().mean())
            pcc = min(C.pcc(ref[2][k][j], dec[k][j]) for j in range(D))
            ok = first[k] == ref[0][k] and top1 == 1.0 and pcc >= 0.99
            row = dict(test=f"batched admission triage ({mapping} chunk pages): request {k} ({n} tokens) of {LENS}, "
                            f"{f'ONE prefill_forward ({ADMIT} admission)' if mode == 'batch' else 'one prefill_forward each'} with vLLM "
                            f"page rows vs one call each with full-coverage rows, then {D} teacher-forced decode steps, "
                            f"tp{TPN} layers {len(LAYERS)}",
                       milestone="P4", status=("pass" if ok else "fail") if mapping != "legacy" else "info", result="pass" if ok else "fail", mapping=mapping, mode=mode, admit=ADMIT, req=k, n=n,
                       first=first[k], first_ref=ref[0][k], decode_top1_vs_ref=round(top1, 4),
                       decode_pcc_min=round(pcc, 6), kda_max_abs=md, lat_max_abs=ml, tp=TPN, layers=len(LAYERS),
                       env=ENV, mock=C.MOCK, tag=tag)  # fmt: skip
            S.log(row)
            rows.append(row)
PT.chunk_pages = FIXED_PAGES
traced = pf.counters["eager"] == eager0
for row in rows:
    row["traced"] = traced
    row["padmask_sites"] = dict(pf.counters.get("padmask", {}))  # MoE layers masked per program build (PADMASK)
    row["gen_admission"] = {k_: v_ for k_, v_ in gen.counters.items() if k_.startswith("prefill_")}  # packed bucket used
    if rec:
        C.record("pcc", row)
RESULT = rows
