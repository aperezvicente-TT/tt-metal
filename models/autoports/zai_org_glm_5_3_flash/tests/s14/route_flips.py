# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (Stage 14, program 2026-10-24 20:56:53Z): the MoE top-8 ROUTING of a layer from each path's x1, in
fp64, vs the CPU fp32 reference chain (hf_ref). In the served chain the PMK and the op share the router OP (op fillers);
the paths differ only in the x1 that the served FFN pre (mhc pre -> RMSNorm, fp32) and the router read. Here:
  router input  hn = RMSNorm(sum_s pre_s(x1) x1_s) * w_post_attention_layernorm   (mhc pre with hc_ffn_*, fp64)
  router        s = sigmoid(hn @ W_gate); choice = s + e_score_correction_bias; top-8 of choice (n_group 1);
                w = s[top-8] / sum * routed_scaling_factor (2.5)
validated on the reference's own x1 (hf_ref L{L}/mid.pt) against its router.pt (indices) and sub.pt (ffn_in), then
applied to the PMK's and the op's device x1 (a pmk_asm_a_unit dump_x1 file). Counts: rows whose top-8 SET differs,
PMK vs fp32, op vs fp32, PMK vs op; the reference's 8th - 9th choice margin on those rows (near-ties); the rows' x1
relative error. Usage: route_flips.py <dump.pt> <hf_ref layer dir> <layer> [out.json]"""
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("MODEL_ROOT", "/opt/glm53/tt-metal-s14v2"))
from models.autoports.zai_org_glm_5_3_flash.tt.weights import TileCacheSource  # noqa: E402

dump, ref_dir, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else None
CACHE = os.environ.get("GLM53_CACHE_DIR", "/opt/glm53/tt-cache/glm53-flash")
MANIFEST = os.environ.get("GLM53_MANIFEST", "/opt/glm53/kit/weights/manifest.json")
src = TileCacheSource(CACHE, MANIFEST)
n, H, eps, hc_eps, K, scale_r = 4, 4096, 1e-5, 1e-6, 8, 2.5
fn = src.host(L, "hc_ffn_fn").reshape(n * H, 24).double()
sc = src.host(L, "hc_ffn_scale").double().reshape(-1)
sc = torch.cat([sc[0].repeat(n), sc[1].repeat(n), sc[2].repeat(n * n)])
base = src.host(L, "hc_ffn_base").double().reshape(-1)
w_norm = src.host(L, "post_attention_layernorm.weight").double().reshape(-1)
w_gate = src.host(L, "mlp.gate.weight").double().reshape(H, -1)  # [4096, E]
bias = src.host(L, "mlp.gate.e_score_correction_bias").double().reshape(-1)


def ffn_in(x1):  # x1 [N, 4, 4096] -> hn [N, 4096] fp64
    x = x1.double().reshape(x1.shape[0], -1)
    z = (x @ fn) * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * sc + base
    pre = torch.sigmoid(z[:, :4]) + hc_eps
    h = (pre[:, :, None] * x1.double()).sum(1)
    return h * torch.rsqrt((h * h).mean(-1, keepdim=True) + eps) * w_norm


def route(hn):
    s = torch.sigmoid(hn @ w_gate)
    choice = s + bias
    top = torch.topk(choice, K + 1, dim=-1)
    idx = top.indices[:, :K]
    wts = s.gather(1, idx)
    return dict(idx=idx, w=wts / wts.sum(-1, keepdim=True) * scale_r, margin=(top.values[:, K - 1] - top.values[:, K]))


def sets(i):
    return [frozenset(r) for r in i.tolist()]


mid = torch.load(os.path.join(ref_dir, "mid.pt"), map_location="cpu", weights_only=True).double()
sub = torch.load(os.path.join(ref_dir, "sub.pt"), map_location="cpu", weights_only=True)
rr = torch.load(os.path.join(ref_dir, "router.pt"), map_location="cpu", weights_only=True)
hn_ref = ffn_in(mid)
val = dict(ffn_in_maxabs_vs_sub=float((hn_ref - sub["ffn_in"].double()).abs().max()),
           ffn_in_ref_absmax=float(sub["ffn_in"].abs().max()))
r_ref = route(hn_ref)
s_ref, s_pt = sets(r_ref["idx"]), sets(rr["indices"])
val["host_router_set_equal_router_pt"] = sum(a == b for a, b in zip(s_ref, s_pt)) / len(s_pt)
d = torch.load(dump, map_location="cpu", weights_only=False)
N = int(d["n"])


def gather_rows(parts):  # per chip [.., r, 16384] in mesh order -> [N, 4, 4096] (sp: rows sharded over the chips)
    t = torch.cat([p.reshape(-1, n * H) for p in parts], 0) if d["sp"] else parts[0].reshape(-1, n * H)
    return t[:N].reshape(N, n, H)


res = dict(layer=L, rows=N, validation=val)
paths = {}
for name in ("pmk", "op"):
    x1 = gather_rows(d[name])
    r = route(ffn_in(x1))
    paths[name] = dict(sets=sets(r["idx"]), x1_rel=((x1.double() - mid[:N]).norm(dim=(1, 2)) / mid[:N].norm(dim=(1, 2))))
ref_sets = s_pt[:N]
margin = r_ref["margin"][:N]
for a, b in (("pmk", "fp32"), ("op", "fp32"), ("pmk", "op")):
    A = paths[a]["sets"]
    B = ref_sets if b == "fp32" else paths[b]["sets"]
    rows = [i for i in range(N) if A[i] != B[i]]
    res[f"{a}_vs_{b}"] = dict(rows_flipped=len(rows), frac=round(len(rows) / N, 5),
                              experts_swapped=sum(len(A[i] - B[i]) for i in rows),
                              ref_margin_median_on_flips=float(margin[rows].median()) if rows else None,
                              ref_margin_median_all=float(margin.median()),
                              first_rows=rows[:12])  # fmt: skip
for name in ("pmk", "op"):
    rel = paths[name]["x1_rel"]
    res[f"{name}_x1_rel_vs_fp32"] = dict(median=float(rel.median()), p95=float(rel.quantile(0.95)), max=float(rel.max()))
print(json.dumps(res, indent=1))
if out_path:
    json.dump(res, open(out_path, "w"), indent=1)
