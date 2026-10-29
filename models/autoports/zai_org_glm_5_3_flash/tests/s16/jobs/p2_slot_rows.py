# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 adapter-path pre-checks (program 2026-10-22 05:33Z: the 161-id prompt passes
through the generator at TP8, fails through vLLM -> the adapter path). Through Glm53Generator.prefill_forward on a
caller-owned cache (the adapter's entry), the 161-id prompt's last logits, every KDA layer's S / C and the L3 latent
rows < n, bitwise:
  (a) page rows: vLLM's descending row [3, 2, 1] + -1 tail on slot 0 vs the identity row of a fresh slot (slot 1);
  (d) slot reuse: slot 0 serves a 512-id request, then a 128-id request (each on its own vLLM blocks), then the 161-id
      prompt on [3, 2, 1] -- vs (a)'s first-use slot-0 run and vs the fresh slot.
ARGS: layers ([0..4]), n (161), reuse_lens ([512, 128]), rep_buckets ([128]), ref, max_seq (2560), tag, record."""
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
N = int(args.get("n", 161))
REUSE = [int(v) for v in args.get("reuse_lens", [512, 128])]
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
MAXSEQ, BLK, B = int(args.get("max_seq", 2560)), 64, 2
BPS = MAXSEQ // BLK
tag = args.get("tag", "p2-slot-rows")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())

os.environ["GLM53_STACK_PREFILL"] = "tp"
os.environ["GLM53_PREFILL_BUCKETS"] = args.get("buckets", "128,512,2048")
os.environ["GLM53_PF_MOE"] = "ep"
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_MOE_V2"] = "1"
os.environ["GLM53_PF_KDA_SCAN"] = "op"
os.environ["GLM53_PF_DSA"] = "0"
os.environ["GLM53_PF_SPARSE_T"] = ""
os.environ["GLM53_PF_BATCH"] = "0"
os.environ["GLM53_PF_PACKED"] = ""
S.set_attn({"mla_form": "absorbed", "mm_nq": 4, "mm_kb": 640, "rep_buckets": [128], **args})

for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "s16-tf", "s16-backend", "s16-chunks")]:
    old = ctx.pop(k)
    if old.get("pf") is not None:
        old["pf"].release()
    old["backend"].close()
gc.collect()
t0 = time.time()
backend = SM.build_model_backend(None, mesh, max_batch=B, layers=LAYERS, hf_config=cfg,
                                 weight_cache=ctx.setdefault("weight_cache", {}))  # fmt: skip
gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=MAXSEQ,
                        stop_token_ids=())  # fmt: skip
kv = gen.allocate_kv_cache(max_batch=B, max_seq_len=MAXSEQ, num_blocks=B * BPS + 1)
gen.warmup(kv)
ctx[("s16-chunks", tuple(LAYERS), TPN, "slot-rows")] = dict(backend=backend, gen=gen, pf=backend.tpp)
S.log(f"built + warmed in {time.time() - t0:.1f} s")
prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()


def row_of(blocks):
    r = torch.full((BPS,), -1, dtype=torch.int64)
    r[: len(blocks)] = torch.tensor(blocks)
    return r


IDENT1 = torch.arange(BPS + 1, 2 * BPS + 1)  # slot 1's full-coverage identity row (block 0 = vLLM's null block)


def prefill(slot, toks, row):
    out = gen.prefill_forward(toks.reshape(1, -1), page_table=row[None], kv_cache=kv, prompt_lens=[int(toks.numel())],
                              empty_slots=[slot]).float().reshape(-1)  # fmt: skip
    ttnn.synchronize_device(mesh)
    return out


def snapshot(slot, n):
    blocks = gen._binding(kv).pt_shadow[slot]
    lat = {L: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()
           for L, t_ in backend.latent_pages(kv).items()}  # fmt: skip
    lr = {L: torch.cat([c_[int(blocks[p]), 0] for p in range(-(-n // BLK))], dim=0)[:n] for L, c_ in lat.items()}
    kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, slot)}
    return kda, lr


def compare(x, y):
    """x, y = (logits, kda, latent): max |diff| per piece; bitwise = all zero"""
    d = {"logits": float((x[0] - y[0]).abs().max())}
    for L in x[1]:
        d[f"L{L}.S"] = float((x[1][L][0] - y[1][L][0]).abs().max())
        d[f"L{L}.C"] = float((x[1][L][1] - y[1][L][1]).abs().max())
    for L in x[2]:
        d[f"L{L}.lat"] = float((x[2][L] - y[2][L]).abs().max())
    return d, all(v == 0.0 for v in d.values())


toks = prompt[:N]
desc = row_of([3, 2, 1][: -(-N // BLK)] + list(range(4, 4 + max(0, -(-N // BLK) - 3))))
run_a = (prefill(0, toks, desc), *snapshot(0, N))  # (a) slot 0, first use after warmup, descending row
run_f = (prefill(1, toks, IDENT1), *snapshot(1, N))  # fresh slot 1, identity row
blk = 10  # free blocks: slot 0 holds 1..3, slot 1 holds 41..80
for n_ in REUSE:  # (d) slot 0 serves other requests on other blocks first
    nb = -(-n_ // BLK)
    prefill(0, prompt[100 : 100 + n_], row_of(list(range(blk, blk + nb))))
    blk += nb
run_d = (prefill(0, toks, desc), *snapshot(0, N))
rows = []
for name, x, y in (("(a) descending [3,2,1] slot 0 vs identity fresh slot 1", run_a, run_f),
                   ("(d) slot 0 after 512 + 128 requests vs slot 0 first use", run_d, run_a),
                   ("(d) slot 0 after 512 + 128 requests vs identity fresh slot 1", run_d, run_f)):  # fmt: skip
    d, bitwise = compare(x, y)
    row = dict(test=f"adapter pre-check {name}: {N}-id prompt through Glm53Generator.prefill_forward, last logits + "
                    f"KDA S / C + latent rows, tp{TPN} layers {len(LAYERS)}", milestone="P2",
               status="pass" if bitwise else "fail", bitwise=bitwise, max_abs=d, n=N, desc_row=desc[: -(-N // BLK)].tolist(),
               argmax=[int(x[0].argmax()), int(y[0].argmax())], tp=TPN, layers=len(LAYERS), tag=tag)  # fmt: skip
    S.log(row)
    rows.append(row)
    if rec:
        C.record("pcc", row)
RESULT = rows
