# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 S2 oracle: the MTP layer (layer 45) on the CPU in float64 from the tile cache's EXACT weight values (the
bfloat8_b / bfloat4_b values the device streams), the semantics of reference/glm5_next_text/mtp.py
(doc/architecture.md §6): x = eh_proj(cat[enorm(E[tok]), hnorm(h)]); x += MLA(input_layernorm(x)) (dense causal over
this MTP cache: context <= 2,051, the DSA indexer selects every visible token); x += MoE(post_attention_layernorm(x));
logits = lm_head(shared_head.norm(x)) -> the draft = argmax (lowest id on ties, the device's merge). Rows are fed
one at a time with their MTP position; the latent cache is a dict position -> latent (a later row at the same
position overwrites, as the device's paged cache does).

Host bfp unpack touches the device runtime: build it inside the device process (its devices are open) or under
.agents/scripts/no-tt-devices."""
from __future__ import annotations

import torch

MTP_L = 45
VOCAB = 154880
D = torch.float64


class MtpOracle:
    def __init__(self, source, cfg, lm_head: torch.Tensor, embed: torch.Tensor, expert_dtype=D):
        """source: tt.optimized_decoder.FdTileCacheSource; lm_head [4096, 154880] (the cache's bfloat8_b values, e.g.
        tt.mk_head.lm_head_host); embed [154880, 4096] (bf16 values)"""
        h = lambda n: source.host(MTP_L, n).to(D)  # noqa: E731
        self.cfg, self.eps = cfg, cfg.rms_norm_eps
        self.w_eh = h("eh_proj.weight")  # [8192, 4096] = [in, out]
        self.w_en, self.w_hn = h("enorm.weight").reshape(-1), h("hnorm.weight").reshape(-1)
        self.w_in, self.w_post = h("input_layernorm.weight").reshape(-1), h("post_attention_layernorm.weight").reshape(-1)
        self.w_sh = h("shared_head.norm.weight").reshape(-1)
        self.qa, self.qa_n = h("self_attn.q_a_proj.weight"), h("self_attn.q_a_layernorm.weight").reshape(-1)
        self.qb = h("self_attn.q_b_proj.weight")
        self.kva, self.kva_n = h("self_attn.kv_a_proj_with_mqa.weight"), h("self_attn.kv_a_layernorm.weight").reshape(-1)
        self.w_uk = h("self_attn.kv_b_proj.W_uk")  # [64, 256, 512]: k_h = W_uk[h] @ c
        self.w_uv = h("self_attn.kv_b_proj.W_uv")  # [64, 512, 256]: v_h = c @ W_uv[h]
        self.wo = h("self_attn.o_proj.weight")  # [16384, 4096]
        g = h("mlp.gate.weight")
        self.gate = g if g.shape[0] == cfg.hidden_size else g.t()  # [4096, 288]
        self.bias = h("mlp.gate.e_score_correction_bias").reshape(-1)
        self.sg, self.su, self.sd = (h(f"mlp.shared_experts.{n}_proj.weight") for n in ("gate", "up", "down"))
        self.source = source
        self.experts = {}
        self.edt = expert_dtype  # routed experts' storage / matmul dtype (float32 halves the host memory of long runs)
        self.lm = lm_head.to(torch.float32)
        self.embed = embed
        self.cache = {}  # MTP position -> latent [512]

    def _rms(self, x, w):
        return x * torch.rsqrt((x * x).mean() + self.eps) * w

    def _expert(self, e: int):
        if e not in self.experts:
            gu = self.source.host(MTP_L, f"mlp.experts.{e}.gate_up_proj").to(self.edt)  # [4096, 4096]: gate | up
            dn = self.source.host(MTP_L, f"mlp.experts.{e}.down_proj").to(self.edt)  # [2048, 4096]
            self.experts[e] = (gu, dn)
        return self.experts[e]

    def _swiglu(self, x, gate, up, down):
        lim = self.cfg.swiglu_limit
        a = (x @ gate).clamp(max=lim)
        b = (x @ up).clamp(min=-lim, max=lim)
        return (a * torch.sigmoid(a) * b) @ down

    def row(self, tok: int, hidden: torch.Tensor, pos: int, force_experts: list | None = None) -> dict:
        """one MTP row at MTP position `pos`: (E[tok], hidden) -> dict(token, logits (fp32), margin, raw x (the chain
        input), top2)"""
        cfg = self.cfg
        e = self.embed[int(tok)].to(D)
        hcat = torch.cat([self._rms(e, self.w_en), self._rms(hidden.to(D).reshape(-1), self.w_hn)])
        x = hcat @ self.w_eh
        # attention (dense causal over the MTP cache; NoPE; scale 256^-0.5)
        hn = self._rms(x, self.w_in)
        q_r = self._rms(hn @ self.qa, self.qa_n)
        q = (q_r @ self.qb).reshape(cfg.num_attention_heads, -1)  # [64, 256]
        c = self._rms((hn @ self.kva)[: cfg.kv_lora_rank], self.kva_n)
        self.cache[int(pos)] = c
        keys = sorted(p for p in self.cache if p <= pos)
        lat = torch.stack([self.cache[p] for p in keys])  # [T, 512]
        q_lat = torch.einsum("hd,hdc->hc", q, self.w_uk)  # [64, 512]
        s = q_lat @ lat.t() * (cfg.qk_head_dim ** -0.5)  # [64, T]
        p = torch.softmax(s, dim=-1)
        o_lat = p @ lat  # [64, 512]
        o = torch.einsum("hc,hcv->hv", o_lat, self.w_uv).reshape(-1)  # [16384]
        x = x + o @ self.wo
        # MoE (noaux_tc router in fp32 semantics, ascending expert accumulation)
        hn2 = self._rms(x, self.w_post)
        logits_r = hn2 @ self.gate
        sc = torch.sigmoid(logits_r)
        biased = sc + self.bias
        top9 = torch.topk(biased, cfg.num_experts_per_tok + 1).values
        router_gap = float(top9[-2] - top9[-1])  # the k-th vs the (k+1)-th biased score: a near tie flips the routing
        sel = torch.topk(biased, cfg.num_experts_per_tok).indices
        if force_experts is not None:  # (diagnosis: the device's routing)
            sel = torch.tensor(sorted(int(v) for v in force_experts))
        wts = sc[sel] / (sc[sel].sum() + 1e-20) * cfg.routed_scaling_factor
        y = torch.zeros_like(x)
        for i in torch.argsort(sel):
            gu, dn = self._expert(int(sel[i]))
            half = gu.shape[1] // 2
            y = y + self._swiglu(hn2.to(gu.dtype), gu[:, :half], gu[:, half:], dn).to(D) * wts[i]
        x = x + y + self._swiglu(hn2, self.sg, self.su, self.sd)
        hs = self._rms(x, self.w_sh).to(torch.float32)
        lg = hs @ self.lm[:, :VOCAB]
        top = torch.topk(lg, 2)
        mx = top.values[0]
        token = int(torch.nonzero(lg == mx)[0])  # the lowest id among the maxima
        return dict(token=token, logits=lg, margin=float(top.values[0] - top.values[1]), x=x,
                    experts=sorted(int(v) for v in sel), router_gap=router_gap)  # fmt: skip
