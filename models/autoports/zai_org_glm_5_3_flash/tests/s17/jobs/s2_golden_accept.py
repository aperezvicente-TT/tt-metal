# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a pair; S2/S3 acceptance input): the DEVICE MTP drafter's acceptance on the FULL model's states,
without the full model on device. The golden layer-44 output of a prompt (goldens/layer_io/L44/<T>/streams_out.pt,
the fp32 CPU reference) gives h_p = RMSNorm(mean of the 4 streams) * norm.weight -- exactly what the device head
writes into its h rows (kernels/head/head_common.hpp) and what P5 calls `normed`. Per prompt position p (in order, so
the MTP's own latent cache fills like decode): the host writes h_p into the head's h row 0, x_{p+1} into the verify
input row 0 and position p + 1, then the BOOTSTRAP trace replays (the MTP catch-up row 0 = (h_p, E[x_{p+1}]) at MTP
position p, K - 1 chained rows) -> drafts d1..dK, compared with the prompt's x_{p+2}..x_{p+1+K}.
Acceptance as verify accepts and as P5 measures it: depth 1 = d1 == the MAIN MODEL's greedy argmax at position p + 1
(goldens/logits prompt_top32_indices[p + 1, 0]); P5's offline depth-1 numbers on the same prompts' prompt region:
tf128 0.452 (n 126), tf2048 0.373 (n 2,046). Also recorded: d_j vs the prompt's own tokens x_{p+1+j} (depth j
conditional on depths < j; on prompt positions the main model's argmax and the prompt token differ, so this is NOT the
speculative acceptance; with the HF checkpoint's weights the oracle reproduces P5's 0.452 only on the argmax metric). Host writes
between synchronized replays only. A host check first: the golden h through the cache's lm_head reproduces the golden
prompt top-1 ids (goldens/logits/<T>.prefill.pt) at a few positions.
start = P (default 0): no bootstrap before position P -> the MTP cache keeps whatever an earlier job left at positions
< P (e.g. another prompt's latents): the drafter's acceptance without an MTP pass over the earlier positions (what a
prefill that skips the MTP layer gives), counted on positions >= P only.
ARGS: layers, b, k, narrow, T ("128,2048"), n (max positions per prompt, 0 = all), start (0), tag."""
import importlib
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 4)), int(args.get("k", 3))
_L = str(args.get("layers", "0,1,2,3,4"))
WS = "/opt/glm53/kit"  # (host-8chip and host-p300)
P5 = {128: (0.45238, 126), 2048: (0.37292, 2046), 512: (None, None), 4096: (0.34758, 4094)}
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import mk_head as MKH  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows, write_rows  # noqa: E402


def log(*a):
    print(f"S17-GA {C.now()}", *a, flush=True)


cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")), log=log)
sp, kv = be.spec, be._kv
io = kv.extra["io"]
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
max_pos = int(kv.blocks_per_seq) * int(be.block_size) - K - 2
src = FdTileCacheSource(CACHE_DIR, MANIFEST)
w_norm = MKH.norm_host(src).double()
eps = float(cfg.rms_norm_eps)
t_h = int(be.head.h.shape[-2])
repl = ttnn.ReplicateTensorToMesh(mesh) if be.tp > 1 else None
lm = None if C.MOCK else MKH.lm_head_host(src)


def normed(x4: torch.Tensor) -> torch.Tensor:
    m = x4.double().mean(0)
    return (m * torch.rsqrt((m * m).mean() + eps) * w_norm).float()


rows_out = []
for T in [int(v) for v in str(args.get("T", "128,2048")).split(",")]:
    X = torch.load(os.path.join(WS, "goldens/layer_io/L44", str(T), "streams_out.pt"), map_location="cpu",
                   weights_only=False)[0]  # [T, 4, 4096] fp32  # fmt: skip
    toks = torch.load(os.path.join(WS, "goldens/prompts", f"{T}.pt"), map_location="cpu", weights_only=False)
    toks = [int(v) for v in toks.reshape(-1)[:T]]
    gl = torch.load(os.path.join(WS, "goldens/logits", f"{T}.prefill.pt"), map_location="cpu", weights_only=False)
    chk = []
    if lm is not None:  # host: golden h -> lm_head -> the golden prompt top-1
        for p in (0, T // 3, T - 1):
            lg = normed(X[p]) @ lm[:, : int(cfg.vocab_size)]
            chk.append(int(torch.argmax(lg)) == int(gl["prompt_top32_indices"][p, 0]))
    log(f"T {T}: golden h check (lm_head top-1 == golden) {chk}")
    n_pos = min(T - 2, max_pos)
    if int(args.get("n", 0)):
        n_pos = min(n_pos, int(args.get("n")))
    acc = [[0, 0] for _ in range(K)]  # per depth vs the prompt's tokens: [accepted, reached]
    top1 = [int(v) for v in gl["prompt_top32_indices"][:, 0]]
    acc_arg = [0, 0]  # depth 1 vs the main model's argmax at p + 1 (the speculative acceptance)
    drafts_all = []  # per position (the CPU oracle's golden_oracle_accept.py json compares row by row)
    t0 = time.time()
    start = int(args.get("start", 0))
    for p in range(start if not C.MOCK else n_pos - 1, n_pos):
        hh = torch.zeros(1, 1, t_h, 4096, dtype=torch.bfloat16)
        hh[0, 0, 0] = normed(X[p]).to(torch.bfloat16)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(hh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                                        mesh_mapper=repl), be.head.h)  # fmt: skip
        tin = read_rows(mesh, sp.tin).reshape(32, -1).clone()
        tin[0, 0] = toks[p + 1]
        write_rows(mesh, sp.tin, tin)
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[0] = p + 1
        be.write(io.cur_pos, pos)
        sp.enter(pt[0])  # the bootstrap replay: c = p
        ttnn.synchronize_device(mesh)
        r = sp.read()
        if C.MOCK:
            break
        assert r["c"] == p, (r["c"], p)
        drafts_all.append(r["drafts"])
        if p + 1 < len(top1):
            acc_arg[1] += 1
            acc_arg[0] += int(r["drafts"][0] == top1[p + 1])
        ok = True
        for j in range(K):
            if p + 2 + j >= T or not ok:
                break
            acc[j][1] += 1
            ok = r["drafts"][j] == toks[p + 2 + j]
            acc[j][0] += int(ok)
        if p % 256 == 0:
            log(f"T {T} p {p} drafts {r['drafts']} next {toks[p + 2: p + 2 + K]} acc {acc}")
    per = [round(a / n, 4) if n else None for a, n in acc]
    if not C.MOCK:
        import json

        dj = os.path.join(C.OUT, "golden", f"device_golden_drafts_{T}_{args.get('tag')}.json")
        os.makedirs(os.path.dirname(dj), exist_ok=True)
        json.dump(dict(T=T, K=K, start=start, drafts=drafts_all, code=os.environ.get("GLM53_CODE_SHA")), open(dj, "w"))
    row = dict(test=f"s2-mtp-accept-golden-k{K}", milestone="S2", status="info", K=K, T=T, positions=n_pos, start=start,
               region="prompt", depth1_accept_vs_main_argmax=round(acc_arg[0] / acc_arg[1], 4) if acc_arg[1] else None,
               counts_vs_main_argmax=acc_arg, per_depth_vs_prompt_tokens=per, counts=acc,
               p5_depth1_prompt=P5.get(T, (None, None))[0],
               p5_n=P5.get(T, (None, None))[1], golden_h_check=chk, hidden="golden L44 normed (fp32 CPU reference)",
               wall_s=round(time.time() - t0, 1), tp=be.tp, layers=_L, tag=args.get("tag"))  # fmt: skip
    S.record("perf", row)
    rows_out.append(row)
    log(f"T {T}: depth-1 acceptance vs the main model's argmax {acc_arg} (P5 depth 1 {P5.get(T)}); "
        f"vs prompt tokens per depth {per} counts {acc}")
sp.leave()
RESULT = dict(rows=[(r["T"], r["depth1_accept_vs_main_argmax"]) for r in rows_out])
