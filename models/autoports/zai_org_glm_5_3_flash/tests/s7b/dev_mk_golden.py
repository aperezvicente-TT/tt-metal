# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b acceptance: the fp32 golden check of every MLA program, once (layer GLM53_S7B_LAYER, default 3), traced.

The HF capture's first decode step after the 2048-token prompt (goldens/layer_io/L03/2048: decode0_streams_in / _out at
position 2048, dense-exact): the latent cache is prefilled with the prompt's latents computed by the CPU fp32 oracle
(tests/s7b/golden_mla.py: mHC pre reference -> q_a | kv_a -> RMSNorm_512, no device prefill), then
  * MkMlaA:  latent, q_lat vs the oracle;  MkMlaA -> the flash-MLA op -> MkMlaB: a, s1 vs the oracle;
  * MkMla (the whole half, one program): latent, q_lat, o_lat, a, s1 vs the oracle;
  * the op chain on the same cache: a, s1 vs the oracle (control);
  * the layer: MkMla -> 7a's MkMoE (GLM53_MK_READERS=3) vs the HF fp32 golden decode0_streams_out; the chain + MkMoE
    as control.
Bar: PCC >= 0.995 vs the fp32 oracle per program (goal), the layer vs the HF golden recorded with the same bar.
Rows -> pcc.jsonl / tests.jsonl."""

from __future__ import annotations

import os
import sys
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.common import int_tensor, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import MlaLayer, Rows, host
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
PROMPT = 2048
BAR = 0.995
BAR_ONE = 0.9999


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = (ttnn.open_device(device_id=0, worker_l1_size=wl1) if os.environ.get("GLM53_MOCK") == "1" else
                ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, worker_l1_size=wl1))  # fmt: skip
    rows = Rows(LAYER)
    tr = Traced(mesh)
    knobs = dict(rev=MK.MLA_REV, mk_readers=os.environ.get("GLM53_MK_READERS"), prompt=PROMPT, position=PROMPT)
    try:
        L = MlaLayer(mesh, ctx, LAYER)
        attn, dec = L.attn, L.dec
        W = G.host_weights(attn)
        from models.autoports.zai_org_glm_5_3_flash.tests.common import golden

        gd = golden(LAYER, PROMPT)
        x_prompt = gd["streams_in"].double().reshape(PROMPT, -1)
        lat_prompt = G.latents(G.mhc_pre_ref(x_prompt, L.a_args)["hn"], W, attn.eps, attn.q_lora)  # [2048, 512]
        x0 = gd["decode0_streams_in"].double().reshape(1, -1)
        ref = G.attention_half(x0, L.a_args, W, [lat_prompt], eps=attn.eps, scale=attn.scale, heads=attn.h,
                               q_lora=attn.q_lora)  # fmt: skip
        # the device cache: the prompt latents (bf16) at positions 0..2047 through the page table
        cache = dec.allocate_cache(max_batch=1, max_seq_len=DSA_DENSE_EXACT_MAX_KV)
        pt = dec.make_page_table(cache)
        shape = tuple(cache.latent.shape)
        ch = torch.zeros(shape, dtype=torch.float32)
        bs = shape[2]
        for p in range(PROMPT):
            ch[int(cache.page_table_host[0, p // bs]), 0, p % bs] = lat_prompt[p].float()
        ttnn.copy(ttnn.from_torch(ch.to(torch.bfloat16), dtype=cache.latent.dtype, layout=ttnn.TILE_LAYOUT,
                                  device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG), cache.latent)  # fmt: skip
        cur = int_tensor([PROMPT], mesh)
        x = ttnn.from_torch(x0.float().reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        c = dict(t=1, cache=cache, pt=pt, cur=cur, x=x, pos=[PROMPT])
        mka, mkb, mk1 = MK.MkMlaA(mesh, L.a_args, attn), MK.MkMlaB(mesh, attn), MK.MkMla(mesh, L.a_args, attn)
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE

        mkmoe = MkMoE(mesh, L.f_args, L.mlp)

        def two():
            ra = mka(x, cache.latent, pt, cur, t=1)
            o_lat = L.flash(c, ra["q_lat"], heads=attn.decode_heads)
            rb = mkb(o_lat, ra["x_out"], ra["coef"], t=1)
            return dict(latent=ra["latent"], q_lat=ra["q_lat"], a=rb["a"], s1=rb["s1"])

        def one():
            r_ = mk1(x, cache.latent, pt, cur, t=1)
            return {k: r_[k] for k in ("latent", "q_lat", "o_lat", "a", "s1")}

        def chain():
            r_ = L.mhc_pre(c)
            a_ = L.mla(c, r_.hn)
            return dict(a=a_, s1=L.mhc_post(c, a_, r_))

        fns = dict(two=two, one=one, chain=chain, layer_one=lambda: mkmoe(one()["s1"], t=1)["out"],
                   layer_chain=lambda: mkmoe(chain()["s1"], t=1)["out"])  # fmt: skip
        for f in fns.values():
            tr.warm(f)
        got = {}
        for name, f in fns.items():
            out = tr.run(f)
            got[name] = {k: host(v) for k, v in out.items()} if isinstance(out, dict) else host(out)
            tr.release()
        refs = dict(latent=ref["latent"], q_lat=ref["q_lat"], o_lat=ref["o_lat"], a=ref["a"], s1=ref["s1"])
        for prog, keys in (("mk_mla_a+b", ("latent", "q_lat", "a", "s1")),
                           ("mk_mla_one", ("latent", "q_lat", "o_lat", "a", "s1")), ("op_chain", ("a", "s1"))):  # fmt: skip
            src = dict(**{"mk_mla_a+b": got["two"], "mk_mla_one": got["one"], "op_chain": got["chain"]})[prog]
            r_ = dict(t=1, program=prog, **knobs)
            for k in keys:
                a_, b_ = src[k].reshape(-1), refs[k].reshape(-1)[: src[k].numel()].float()
                r_[f"{k}_pcc_vs_fp32"] = pcc(a_, b_)
                r_[f"{k}_maxabs_vs_fp32"] = float((a_ - b_).abs().max())
            # the one-program half (in-program attention phase) is accepted vs this golden at 0.9999 (program
            # ruling 2026-10-19T23:39:08Z); programs with the flash-MLA op keep the goal's 0.995
            bar = BAR_ONE if prog == "mk_mla_one" else BAR
            r_["bar"] = bar
            r_["status"] = "pass" if min(r_[f"{k}_pcc_vs_fp32"] for k in keys) >= bar else "fail"
            rows("golden_fp32_attention_half", **r_)
        gold = gd["decode0_streams_out"].float().reshape(-1)
        for name in ("layer_one", "layer_chain"):
            v = got[name].reshape(-1)[: gold.numel()]
            r_ = dict(t=1, program=name, layer_pcc_vs_hf_golden=pcc(v, gold),
                      layer_maxabs_vs_hf_golden=float((v - gold).abs().max()), **knobs)  # fmt: skip
            r_["status"] = "pass" if r_["layer_pcc_vs_hf_golden"] >= BAR else "fail"
            rows("golden_hf_layer_decode0", **r_)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rows("golden_fp32_attention_half", status="fail", error=traceback.format_exc()[-800:], **knobs)
    finally:
        rows.flush()
        if own:
            ttnn.close_device(mesh) if os.environ.get("GLM53_MOCK") == "1" else ttnn.close_mesh_device(mesh)
    bad = rows.failed()
    print(f"DEV_MK_GOLDEN_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
