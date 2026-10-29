# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only control for the golden-state acceptance: the CPU MTP oracle (tests/s17/mtp_oracle.py) with
the HF checkpoint's OWN layer-45 / lm_head / embedding weights (fp8 blocks dequantized exactly, bf16 as stored) instead
of the tile cache's block-float values, on the same golden inputs as jobs/s2_golden_accept.py. Two hidden variants:
h in fp32 (as P5's reference) and h rounded to bf16 (as the device head writes it). Pure torch + safetensors (no
ttnn): run under .agents/scripts/no-tt-devices.
usage: golden_oracle_hf.py <T> <out.json> [n positions]"""
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../../../..")))  # the tt-metal tree root (models.*)
from mtp_oracle import MtpOracle  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.reference.weights import ShardReader, _read_maybe_fp8  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config  # noqa: E402

SNAP = "/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
WS = "/opt/glm53/kit"
P = "model.language_model.layers.45."


class HFSource:
    """source.host(layer, name) in the tile cache's conventions ([in, out] matrices, fused expert gate|up, W_uk / W_uv)
    from the HF checkpoint (fp32 values of the stored fp8 / bf16 weights)"""

    T_IN_OUT = ("eh_proj.weight", "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight", "self_attn.o_proj.weight", "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight", "mlp.shared_experts.down_proj.weight")  # fmt: skip

    def __init__(self, cfg):
        self.r, self.cfg = ShardReader(SNAP), cfg

    def _w(self, name: str) -> torch.Tensor:
        return _read_maybe_fp8(self.r, P + name, torch.float32).float()

    def host(self, layer: int, name: str) -> torch.Tensor:
        assert layer == 45, layer
        if name in self.T_IN_OUT:
            return self._w(name).t().contiguous()
        if name.startswith("self_attn.kv_b_proj.W_u"):
            h, kn = int(self.cfg.num_attention_heads), int(self.cfg.qk_nope_head_dim)
            kv = self._w("self_attn.kv_b_proj.weight").reshape(h, -1, int(self.cfg.kv_lora_rank))  # [64, 512, 512]
            return kv[:, :kn, :].contiguous() if name.endswith("W_uk") else kv[:, kn:, :].transpose(1, 2).contiguous()
        m = re.fullmatch(r"mlp\.experts\.(\d+)\.(gate_up_proj|down_proj)", name)
        if m:
            e = m.group(1)
            if m.group(2) == "down_proj":
                return self._w(f"mlp.experts.{e}.down_proj.weight").t().contiguous()  # [2048, 4096]
            g, u = self._w(f"mlp.experts.{e}.gate_proj.weight"), self._w(f"mlp.experts.{e}.up_proj.weight")
            return torch.cat([g.t(), u.t()], dim=1).contiguous()  # [4096, 4096] = gate | up
        return self._w(name)


def main():
    T, out = int(sys.argv[1]), sys.argv[2]
    cfg = load_hf_text_config(SNAP)
    src = HFSource(cfg)
    lm = src.r.get("lm_head.weight").float().t().contiguous()  # [4096, V]
    emb = src.r.get("model.language_model.embed_tokens.weight")
    w_norm = src.r.get("model.language_model.norm.weight").double()
    eps = cfg.rms_norm_eps
    orcs = {v: MtpOracle(src, cfg, lm, emb, expert_dtype=torch.float32) for v in ("h_fp32", "h_bf16")}
    orcs["h_bf16"].experts = orcs["h_fp32"].experts  # one expert store
    X = torch.load(os.path.join(WS, "goldens/layer_io/L44", str(T), "streams_out.pt"), map_location="cpu",
                   weights_only=False)[0]  # fmt: skip
    toks = [int(v) for v in torch.load(os.path.join(WS, "goldens/prompts", f"{T}.pt"), map_location="cpu",
                                        weights_only=False).reshape(-1)[:T]]  # fmt: skip
    N = int(sys.argv[3]) if len(sys.argv) > 3 else T - 2
    K, t0 = 3, time.time()
    res = {}
    for v, orc in orcs.items():
        acc, rows = [[0, 0] for _ in range(K)], []
        for p in range(min(N, T - 2)):
            m = X[p].double().mean(0)
            h = m * torch.rsqrt((m * m).mean() + eps) * w_norm
            h = h.to(torch.bfloat16).double() if v == "h_bf16" else h
            o = orc.row(toks[p + 1], h, p)
            d, x = [o["token"]], o["x"]
            for j in range(1, K):
                oc = orc.row(d[-1], x, p + j)
                d.append(oc["token"])
                x = oc["x"]
            ok = True
            for j in range(K):
                if p + 2 + j >= T or not ok:
                    break
                acc[j][1] += 1
                ok = d[j] == toks[p + 2 + j]
                acc[j][0] += int(ok)
            rows.append(dict(p=p, drafts=d, margin1=round(o["margin"], 4)))
            if p % 32 == 0:
                print(f"{v} p {p} drafts {d} next {toks[p + 2: p + 2 + K]} acc {acc} {time.time() - t0:.0f}s", flush=True)
        res[v] = dict(per_depth=[round(a / n, 4) if n else None for a, n in acc], counts=acc, rows=rows)
        print("HF-ORACLE", T, v, res[v]["per_depth"], acc, flush=True)
    json.dump(dict(T=T, weights="HF checkpoint (fp8 blocks dequantized, bf16 as stored)", **res), open(out, "w"))


if __name__ == "__main__":
    main()
