# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (a serve.py MOCK job: the host bfp unpack needs an open (mock) mesh): the CPU MTP oracle (tests/s17/mtp_oracle.py, the tile cache's
exact weight values; routed experts in float32) on the SAME golden inputs as jobs/s2_golden_accept.py (golden layer-44
states normed, rounded to bf16 as the device head writes them; x_{p+1}; positions in order; chain = the MTP's own raw
output + its draft) -> per-depth acceptance on the prompt region and the drafts per position (json), to compare with
the device drafter row by row. ARGS: T, out (json path), n (positions; default T - 2)."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_oracle import MtpOracle  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402

WS = "/opt/glm53/kit"
args = ARGS  # noqa: F821
T, OUT = int(args["T"]), str(args["out"])
N = int(args.get("n", T - 2))
K = 3
cfg = hf_config()
src = FdTileCacheSource(CACHE_DIR, MANIFEST)
orc = MtpOracle(src, cfg, MKH.lm_head_host(src), MKH.embed_host(src), expert_dtype=torch.float32)
w_norm = MKH.norm_host(src).double()
eps = float(cfg.rms_norm_eps)
X = torch.load(os.path.join(WS, "goldens/layer_io/L44", str(T), "streams_out.pt"), map_location="cpu", weights_only=False)[0]
toks = [int(v) for v in torch.load(os.path.join(WS, "goldens/prompts", f"{T}.pt"), map_location="cpu",
                                    weights_only=False).reshape(-1)[:T]]  # fmt: skip


def normed(x4):
    m = x4.double().mean(0)
    return (m * torch.rsqrt((m * m).mean() + eps) * w_norm).to(torch.bfloat16).double()


acc, rows, t0 = [[0, 0] for _ in range(K)], [], time.time()
for p in range(min(N, T - 2)):
    o = orc.row(toks[p + 1], normed(X[p]), p)
    d = [o["token"]]
    x, margins = o["x"], [o["margin"]]
    for j in range(1, K):
        oc = orc.row(d[-1], x, p + j)
        d.append(oc["token"])
        x = oc["x"]
        margins.append(oc["margin"])
    ok = True
    for j in range(K):
        if p + 2 + j >= T or not ok:
            break
        acc[j][1] += 1
        ok = d[j] == toks[p + 2 + j]
        acc[j][0] += int(ok)
    rows.append(dict(p=p, drafts=d, margins=[round(m, 4) for m in margins]))
    if p % 16 == 0:
        print(f"p {p} drafts {d} next {toks[p + 2: p + 2 + K]} acc {acc} {time.time() - t0:.0f}s experts {len(orc.experts)}",
              flush=True)
per = [round(a / n, 4) if n else None for a, n in acc]
json.dump(dict(T=T, per_depth=per, counts=acc, rows=rows, expert_dtype="float32"), open(OUT, "w"))
print("ORACLE", T, per, acc, flush=True)
