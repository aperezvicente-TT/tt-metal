# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 L1 cut, BITWISE A/B (program ruling 2026-10-21 20:23Z: TPPrefill's static CB
region must end <= 1,330,000, numerics bitwise). The cut sizes CBs only: nlp_concat_heads' interleaved input CB = 4
head blocks instead of a double-buffered row of every head; recurrent_chunk_scan's summary-only CBs = 1 tile in the
recurrent scan. TT_NLP_CONCAT_HEADS_FULL_CB=1 / TT_KDA_SCAN_LEGACY_DFBS=1 restore the old sizes. Both configurations run
in ONE process: the program cache is cleared between them (the factories read the switches when a program is built),
and each configuration compiles, captures and REPLAYS the prefill of the same prompt into slot 0. The job then
compares SHA-256 of the last-row logits, every KDA layer's slot state and the MLA latent pages.

ARGS: layers ([0..4]), ref (refpt/prefill2048_L0-4.refpt), mla_form ("absorbed": the path through nlp_concat_heads),
kda_scan ("grouped"), order (["new", "legacy", "new"]), knobs (s16common.set_attn), tag, record (True)."""
import hashlib
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
ORDER = list(args.get("order", ["new", "legacy", "new"]))
tag = args.get("tag", "l1-cut-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
SWITCHES = ("TT_NLP_CONCAT_HEADS_FULL_CB", "TT_KDA_SCAN_LEGACY_DFBS")

prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
P = int(prompt.numel())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_DSA"] = "0"
args = dict(args)
args.setdefault("mla_form", "absorbed")
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(2048,), max_seq=2560)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "grouped")
backend.tpp = pf
s = pf.bucket_for(P)


def sha(t_: torch.Tensor) -> str:
    return hashlib.sha256(t_.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def chips(t_):
    return [ttnn.to_torch(x) for x in (ttnn.get_device_tensors(t_) if TPN > 1 else [t_])]


def run(cfg: str) -> dict:
    for k in SWITCHES:
        os.environ.pop(k, None)
    if cfg == "legacy":
        for k in SWITCHES:
            os.environ[k] = "1"
    pf.release()
    ttnn.synchronize_device(mesh)
    mesh.clear_program_cache()  # every program is rebuilt under this configuration's CB sizes
    pf.compile((s,))
    backend.prefill_chunk(torch.zeros(s, dtype=torch.int64), kv_cache=kv, slot=0, chunk_start=0, n_valid=s,
                          logits_rows="last", io=io, sample=None)  # fmt: skip
    ttnn.synchronize_device(mesh)
    pf._warmed = True
    pf.capture((s,))
    eager0 = pf.counters["eager"]
    lg = backend.prefill_chunk(prompt, kv_cache=kv, slot=0, chunk_start=0, n_valid=P, logits_rows="last", io=io,
                               sample=None)  # fmt: skip
    ttnn.synchronize_device(mesh)
    out = dict(cfg=cfg, traced=pf.counters["eager"] == eager0, argmax=int(torch.as_tensor(lg).float().reshape(-1).argmax()),
               logits=sha(torch.as_tensor(lg)))  # fmt: skip
    st = {}
    for i, e in enumerate(backend.seq):
        s_ = kv.states[0][i]
        if s_ is not None:
            st[f"L{e['L']}"] = sha(torch.cat([x.reshape(-1).float() for t_ in s_ for x in chips(t_)]))
    out["kda_state"] = st
    lat = kv.extra.get("latent", {})
    out["latent"] = {f"L{backend.seq[i]['L']}": sha(torch.cat([x.reshape(-1).float() for x in chips(t_)]))
                     for i, t_ in lat.items()}  # fmt: skip
    S.log(out)
    return out


runs = [run(c_) for c_ in ORDER]
ref = runs[0]
same = all(r_[k] == ref[k] for r_ in runs[1:] for k in ("logits", "kda_state", "latent"))
row = dict(test=f"L1 cut bitwise A/B: TP2 L{LAYERS[0]}-L{LAYERS[-1]} traced prefill of the {P}-token prompt, new vs legacy CB "
                f"sizes (nlp_concat_heads input depth, recurrent_chunk_scan summary-only CBs) in one process, tp{TPN}",
           milestone="P2", layers=LAYERS, prompt_len=P, mla_form=pf.mla_form, kda_scan=pf.kda_scan, order=ORDER,
           runs=runs, bitwise=same, traced=all(r_["traced"] for r_ in runs),
           status="pass" if same and all(r_["traced"] for r_ in runs) else "fail", tag=tag)  # fmt: skip
for k in SWITCHES:
    os.environ.pop(k, None)
S.log(row)
if rec and not C.MOCK:
    C.record("pcc", row)
RESULT = row
