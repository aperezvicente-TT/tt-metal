# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 DFlash2 drafter (Apache-2.0 drafter only) -> the TP8 tile cache the served image loads (host only, no
device: ttnn's host bfp8 pack + dump_tensor; run under no-tt-devices). Program 12:56:32Z: "the DFlash2 drafter
weight conversion for the image (bfp8 taps, < 1 GB/chip)".

Layout (tp ranks r = 0 .. tp-1, every matmul weight stored [in, out] for ttnn.linear, bfloat8_b TILE):
  column split (out / tp per rank): fc (the 9-tap fusion, [36864, 4096]), q / k / v (4 q heads + 1 kv head per rank at
      tp 8: GQA groups stay whole), gate / up, lm_head ([4096, 154880 / tp]: each rank its vocab slice, top-16 per rank)
  row split (in / tp per rank, the partial sums all-reduced): o_proj, down
  replicated: attention_conv / mlp_conv kernel_projection ([4096, 1024] bfp8: the conv deltas act on the replicated
      hidden), the RMS norm rows (bf16), the conv base kernels (fp32)
  host (not converted): embed_tokens + mask_embedding (the block's input rows), the candidate selector
One file per (tensor, rank) (replicated: one file), manifest.json = shapes / dtype / layout / split / bytes per rank.
usage: dflash2_convert.py <drafter dir> <out dir> [tp (8)] [dtype (bfloat8_b)]"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open

import ttnn

COL = ("fc.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
       "mlp.gate_proj.weight", "mlp.up_proj.weight", "lm_head.weight")  # fmt: skip
ROW = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
REP_MM = ("attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight")
REP_ROW = ("hidden_norm.weight", "norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
           "self_attn.q_norm.weight", "self_attn.k_norm.weight")  # fmt: skip
REP_BASE = ("attention_conv.base_kernel", "mlp_conv.base_kernel")
HOST = ("embed_tokens.weight", "candidate_selector.")


def kind(name: str) -> str:
    for k, tags in (("col", COL), ("row", ROW), ("rep_mm", REP_MM), ("rep_row", REP_ROW), ("rep_base", REP_BASE)):
        if any(name == t or name.endswith("." + t) for t in tags):
            return k
    if name.startswith(HOST):
        return "host"
    raise ValueError(f"dflash2_convert: no layout rule for {name}")


def main(src: str, out: str, tp: int = 8, dtype: str = "bfloat8_b"):
    os.makedirs(out, exist_ok=True)
    mm_dt = getattr(ttnn, dtype)
    f = safe_open(os.path.join(src, "model.safetensors"), framework="pt", device="cpu")
    cfg = json.load(open(os.path.join(src, "config.json")))
    man = dict(source=os.path.abspath(src), tp=tp, dtype=dtype, config=cfg, tensors={}, bytes_per_rank=[0] * tp,
               bytes_replicated=0, host=[])  # fmt: skip
    t0 = time.time()

    def dump(t: torch.Tensor, dt, path: str) -> int:
        h = ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(os.path.join(out, path), h)
        return os.path.getsize(os.path.join(out, path))

    for name in sorted(f.keys()):
        k = kind(name)
        if k == "host":
            man["host"].append(name)
            continue
        w = f.get_tensor(name)
        e = dict(kind=k, src_shape=list(w.shape))
        if k in ("col", "row"):
            wt = w.t().contiguous()  # [in, out]
            dim = 1 if k == "col" else 0
            assert wt.shape[dim] % (tp * 32) == 0, (name, wt.shape, tp)
            parts = torch.chunk(wt, tp, dim=dim)
            e.update(dtype=dtype, split_dim=dim, shape=list(parts[0].shape), paths=[], bytes=[])
            for r, p in enumerate(parts):
                path = f"{name}.r{r}.tensorbin"
                b = dump(p.to(torch.bfloat16), mm_dt, path)
                e["paths"].append(path)
                e["bytes"].append(b)
                man["bytes_per_rank"][r] += b
        else:
            if k == "rep_mm":
                t, dt, dts = w.t().contiguous().to(torch.bfloat16), mm_dt, dtype
            elif k == "rep_row":
                t, dt, dts = w.reshape(1, 1, 1, -1).to(torch.bfloat16), ttnn.bfloat16, "bfloat16"
            else:  # base kernel [2 (prepare / finish), 2 taps, C] -> fp32 rows [4, C] (row 2 s + t)
                t, dt, dts = w.reshape(4, -1).float(), ttnn.float32, "float32"
            path = f"{name}.tensorbin"
            b = dump(t, dt, path)
            e.update(dtype=dts, shape=list(t.shape), paths=[path], bytes=[b])
            man["bytes_replicated"] += b
        man["tensors"][name] = e
    per_chip = [b + man["bytes_replicated"] for b in man["bytes_per_rank"]]
    man["bytes_per_chip"] = per_chip
    man["wall_s"] = round(time.time() - t0, 1)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"dflash2_convert: {len(man['tensors'])} tensors, tp {tp} {dtype}: per chip {max(per_chip) / 2**20:.1f} MiB "
          f"(sharded {max(man['bytes_per_rank']) / 2**20:.1f} + replicated {man['bytes_replicated'] / 2**20:.1f}), "
          f"host-side {len(man['host'])} tensors, {man['wall_s']} s", flush=True)  # fmt: skip
    return man


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 8, a[3] if len(a) > 3 else "bfloat8_b")
