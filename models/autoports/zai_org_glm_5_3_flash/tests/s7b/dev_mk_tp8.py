# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b TP8 emulation on one chip (shard_contract.md): the one-program attention half of each rank r's head shard
(tt/mk_mla.py HeadShard: 8 heads, q_b columns / W_uk / W_uv / o_proj rows of those heads; q_a | kv_a, norms, latent
cache replicated) vs the single-chip MkMla on the same inputs and starting caches, layer GLM53_S7B_LAYER (default 3),
T = GLM53_MK_TS (default 1,2), traced:
  * sum_r a_r (the partials Stage 14's fabric phase reduces) vs the single chip's a: PCC;
  * q_lat_r / o_lat_r vs the single chip's rows of heads(r): PCC (different head grouping / row chunking of the
    attention -> not bitwise);
  * traced time of rank 0's program (the per-chip TP8 attention half before the fabric phase) and of the single chip's.
Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import os
import sys
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import MlaLayer, Rows, compare, host, timed
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,2").split(",")]
TP = int(os.environ.get("GLM53_TP", "8"))
REPLAYS = int(os.environ.get("GLM53_MK_REPLAYS", "50"))
KB_SHARD = tuple(int(v) for v in os.environ.get("GLM53_TP_KB", "8,16,8,16,16").split(","))
RINGS_SHARD = tuple(int(v) for v in os.environ.get("GLM53_TP_RINGS", "140,105,128,280").split(","))


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = (ttnn.open_device(device_id=0, worker_l1_size=wl1) if os.environ.get("GLM53_MOCK") == "1" else
                ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, worker_l1_size=wl1))  # fmt: skip
    rows = Rows(LAYER)
    tr = Traced(mesh)
    knobs = dict(tp=TP, kb_shard=list(KB_SHARD), rings_shard=list(RINGS_SHARD), rev=MK.MLA_REV)
    hung = None
    try:
        L = MlaLayer(mesh, ctx, LAYER)
        attn = L.attn
        full = MK.MkMla(mesh, L.a_args, attn)
        shards = [MK.MkMla(mesh, L.a_args, MK.HeadShard(attn, r, TP), kb=KB_SHARD, rings=RINGS_SHARD)
                  for r in range(TP)]  # fmt: skip
        keys = ("a", "o_lat", "q_lat", "latent")
        for t in TS:
            c = L.case(t, tag="tp8")
            run = lambda m, c=c, t=t: {k: v for k, v in m(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t).items()
                                       if k in keys}  # fmt: skip
            try:
                for m in [full] + shards:
                    tr.warm(lambda m=m: run(m))
                got = {}
                us = {}
                for name, m in [("full", full)] + [(f"r{r}", s) for r, s in enumerate(shards)]:
                    out = tr.run(lambda m=m: run(m))
                    got[name] = {k: host(v) for k, v in out.items()}
                    if name in ("full", "r0"):
                        us[f"mk_one_{name}"] = timed(tr, REPLAYS)
                    tr.release()
                hr = attn.h // TP
                a_sum = sum(got[f"r{r}"]["a"] for r in range(TP))
                r_ = dict(t=t, execution=tr.mode, timing_us=us, variant="mk_mla_one_tp8_emulation", **knobs)
                compare(r_, {"a_sum": a_sum}, {"a_sum": got["full"]["a"]})
                worst = {}
                for k in ("q_lat", "o_lat"):
                    f = got["full"][k].reshape(t, attn.h, -1)
                    pccs = []
                    for r in range(TP):
                        part = {k: got[f"r{r}"][k].reshape(t, -1, f.shape[-1])[:, :hr]}
                        rr = {}
                        compare(rr, part, {k: f[:, r * hr : (r + 1) * hr]})
                        pccs.append(rr[f"{k}_pcc"])
                    worst[f"{k}_min_pcc"] = min(pccs)
                lat_ok = all(torch.equal(got[f"r{r}"]["latent"], got["full"]["latent"]) for r in range(TP))
                r_.update(worst, latent_replicated_bitwise=lat_ok)
                r_["status"] = ("pass" if r_["a_sum_pcc"] >= 0.9999 and min(worst.values()) >= 0.9999 and lat_ok
                                else "fail")  # fmt: skip
                rows("mk_mla_tp8_emulation", **r_)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rows("mk_mla_tp8_emulation", t=t, status="fail", error=traceback.format_exc()[-800:], **knobs)
                tr.release()
                if "TIMEOUT" in str(e) or "hang" in str(e):
                    hung = str(e)[:300]
                    break
    finally:
        rows.flush()
        if own:
            ttnn.close_device(mesh) if os.environ.get("GLM53_MOCK") == "1" else ttnn.close_mesh_device(mesh)
    bad = rows.failed()
    print(f"DEV_MK_TP8_DONE rows={len(rows.rows)} fail={len(bad)} hung={hung}", flush=True)
    return 3 if hung else (1 if bad else 0)


if __name__ == "__main__":
    sys.exit(main())
