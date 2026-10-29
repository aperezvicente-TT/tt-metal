# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (host mock or device; CPU only): the split-K payload precision on REAL data -- the anchor prompt's
q_lat / latents at ARGS["pos"] (tests/s7b/golden_mla.dsa_step, float64 oracle: its exact top-512 selection and the
attended rows). Per token: the exact attention over the selected rows (fp32, all 64 heads) vs the split-K merge of
the two chips' partials (tt/dsa_shard.py ShardMap owners; attention_partial / merge_partials) with the peer's partial
crossing as fp32, 1 / 2 / 3 bf16 terms, and with the device's tf32 operand rounding of O before the split. Rows:
dsa_shard_payload (PCC per variant, per chip's heads). ARGS: pos ([32767, 32730]), layer (DSA.LAYER)."""

import importlib
import os

import torch

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
os.environ.update(ENV)
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import dev_mk_dsa as DSA  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK  # noqa: E402

for m in (DS, MC, G, DSA):
    importlib.reload(m)
mesh = MESH  # noqa: F821
pos = [int(v) for v in ARGS.get("pos", [32767, 32730])]  # noqa: F821
rows = MC.Rows(DSA.LAYER)
h0 = lambda t: MK._host0(t).double()  # noqa: E731
L = MC.layer_of(mesh, CTX, DSA.LAYER)  # noqa: F821
attn = L.attn
W = dict(
    qa_kva=h0(attn.ds_qa_kva.weight).reshape(-1, attn.ds_qa_kva.n_pad)[:, : attn.ds_qa_kva.n],
    qa_norm=h0(attn.qa_norm).reshape(-1)[: attn.q_lora],
    kva_norm=h0(attn.kva_norm).reshape(-1)[: attn.kv_lora],
    qb=h0(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n],
    w_uk=h0(attn.w_uk).reshape(attn.h, attn.dqk, attn.kv_lora),
    w_uv=h0(attn.w_uv).reshape(attn.h, attn.kv_lora, attn.dv),
    o=h0(attn.ds_o.weight).reshape(attn.h * attn.dv, -1)[:, : attn.ds_o.n],
)
ix = MK.DsaIndexer(mesh, L.source, DSA.LAYER)
gd, hn_all, lat_all = DSA._prompt(L, W, attn, max(pos) + 1)
ir = G.indexer_rows(hn_all, ix.host)
g_in = L.g.reshape(-1, 16384).double()
xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, len(pos))]
sc = dict(eps=attn.eps, scale=attn.scale, heads=attn.h, q_lora=attn.q_lora)
tp_g = MK.DsaIndexer.pool_rows(-(-(max(pos) + 1) // attn.block_size) * attn.block_size)
sm = DS.ShardMap(2, attn.block_size)
hr = attn.h // 2


def tf32(x):
    """the FPU operand rounding of an fp32 tile (10 mantissa bits)"""
    b = x.float().contiguous().view(torch.int32)
    return ((b + (1 << 12)) & ~((1 << 13) - 1)).view(torch.float32)


for i, p in enumerate(pos):
    pl, rg = G.dsa_state(p, ir["k"], ir["g"], ix.host["ape"], tp_g)
    ref = G.dsa_step(xs[i].reshape(1, -1), p, L.a_args, W, ix.host, lat_all[:p], pl, rg, **sc)
    q = ref["q_lat"].float()  # [H, 512]
    kv_all = torch.cat([lat_all[:p], ref["latent"].reshape(1, -1)], dim=0)
    rws = ref["rows"].reshape(-1).to(torch.int64)
    K = kv_all[rws].float()
    m, l, O = DS.attention_partial(q, K, attn.scale)
    o_ref = O / l[:, None]
    owner = torch.tensor([sm.owner(int(r)) for r in rws.tolist()])
    parts = [DS.attention_partial(q, K[owner == r], attn.scale) for r in range(2)]
    r_ = dict(slot=i, p=p, rows=int(rws.numel()), rows_per_chip=[int((owner == r).sum()) for r in range(2)],
              status="info", program="split_k_cpu")  # fmt: skip
    for label, prep in (("fp32", lambda x: x), ("tf32", tf32)):
        for terms in (0, 1, 2, 3):
            pcc_min = 1.0
            for own in range(2):  # the head owner keeps its own partial exact, the peer's crosses as the payload
                pay = []
                for r, (m_, l_, O_) in enumerate(parts):
                    if r == own or terms == 0:
                        pay.append((m_, l_, prep(O_)))
                    else:
                        pay.append((DS.bf16_split(m_, terms), DS.bf16_split(l_, terms), DS.bf16_split(prep(O_), terms)))
                mg, lg, Og = DS.merge_partials(pay)
                o = Og / lg[:, None]
                hs = slice(own * hr, (own + 1) * hr)
                pcc_min = min(pcc_min, DS.pcc(o[hs], o_ref[hs]))
            r_[f"pcc_{label}_{terms}terms"] = round(pcc_min, 9)
    rows("dsa_shard_payload", **r_)
    print("PAYLOAD", {k: v for k, v in r_.items() if k.startswith("pcc") or k in ("p", "rows_per_chip")}, flush=True)
rows.flush()
RESULT = dict(ok=True, rows=len(rows.rows))
