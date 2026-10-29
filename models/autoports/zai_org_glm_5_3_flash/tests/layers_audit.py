# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""45-layer weight-cache audit (stage 06a; program directive A5, doc/DECISIONS.md 2026-10-17 ~22:40Z).

For every decoder layer 0..44 the audit instantiates, on the host and from the pre-packed tile cache, exactly the
parameter set the Stage 1/2 decoder requests from its weight source (`tt/decoder_layer.py` and the identical list in
`tt/fused_decoder.py`: `source.device(L, name)`, `source.host(L, name)`, `source.experts(L, name, 288)`), and checks
  * presence in `weights/manifest.json` and on disk,
  * logical shape against the HF config (`tt/config.py`; linears stored [in, out]),
  * dtype against the decoder's dtype map (`tt/weights.py::cache_dtype`, parsed without importing ttnn) and TILE layout,
  * file size against shape + dtype (tile bytes), and sha256 against the manifest for EVERY file (routed experts too),
  * decodability: every non-expert tensor is decoded with the pure-numpy `weights/tile_codec.py` (never ttnn: unpacking a
    host bfp tensor through ttnn opens the UMD driver) and must be finite; routed experts 0 and 287 are decoded as a
    sample (all 288 x 2 are hashed); the per-layer router bias (`mlp.gate.e_score_correction_bias`, fp32 [288]) and the
    shared expert are reported explicitly.
Tensors present in the cache but not requested by the Stage 1/2 decoder are listed (`present_not_loaded`: the DSA
indexer weights, which only the Stage 5 sparse path uses). One JSON row per layer.

Host only, no ttnn import; run under the no-tt-devices sandbox with a CPU pin:
    no-tt-devices taskset -c 224-255 python models/autoports/zai_org_glm_5_3_flash/tests/layers_audit.py --out <jsonl>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
TILE_BYTES = {"bfloat8_b": 1088, "bfloat4_b": 576, "bfloat16": 2048, "float32": 4096}
WEIGHT_DTYPE_NAMES = {"routed_expert": None, "projection_bfp8": None, "bf16": None, "fp32": None}
# negative control: manifest entries corrupted in memory -> the audit must fail exactly these checks
TAMPER = {
    "layers.0.self_attn.q_proj.weight": ("sha256", "0" * 64, "sha_mismatch"),
    "layers.0.mlp.down_proj.weight": ("shape", [4096, 12288], "shape_mismatch"),
    "layers.0.self_attn.A_log": (None, None, "missing"),
    "layers.3.mlp.gate.e_score_correction_bias": ("dtype", "bfloat16", "dtype_mismatch"),
    "layers.3.mlp.experts.17.down_proj": ("sha256", "f" * 64, "sha_mismatch"),
}


def git_state() -> dict:
    import subprocess

    git = ["git", "-c", "safe.directory=*", "-C", HERE]
    sha = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain", "--", MODEL], capture_output=True, text=True).stdout
    return {"tt_metal_sha": sha or None, "tt_metal_dirty": bool(dirty.strip())}


def _load_config():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(MODEL))))  # tt-metal root (models package)
    from models.autoports.zai_org_glm_5_3_flash.tt.config import LayerSpec, load_hf_text_config

    snap = os.environ.get("GLM53_HF_SNAPSHOT") or (
        "/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
    )
    return load_hf_text_config(snap), LayerSpec


def _dtype_rule():
    """cache_dtype(suffix) of tt/weights.py with WEIGHT_DTYPES of tt/precision.py, parsed with ast (no ttnn import)."""
    consts = {}
    for node in ast.parse(open(os.path.join(MODEL, "tt", "weights.py")).read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("_BFP8_SUFFIXES", "_FP32_SUFFIXES"):
                consts[node.targets[0].id] = set(ast.literal_eval(node.value))
    names = {}
    for node in ast.parse(open(os.path.join(MODEL, "tt", "precision.py")).read()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "WEIGHT_DTYPES"
        ):
            for k, v in zip(node.value.keys, node.value.values):
                names[ast.literal_eval(k)] = v.attr  # ttnn.<dtype>
    assert set(names) == set(WEIGHT_DTYPE_NAMES), names

    def cache_dtype(suffix: str) -> str:
        if ".experts." in suffix:
            return names["routed_expert"]
        if suffix in consts["_BFP8_SUFFIXES"]:
            return names["projection_bfp8"]
        if suffix in consts["_FP32_SUFFIXES"]:
            return names["fp32"]
        return names["bf16"]

    return cache_dtype


def requested(cfg, spec) -> list[tuple[str, str, list[int]]]:
    """(name, mode, expected logical shape) of every tensor the Stage 1/2 decoder requests for this layer kind
    (tt/decoder_layer.py:38-128; tt/fused_decoder.py:1032-1160 requests the same names). mode: device | host | expert.
    """
    H, hc = cfg.hidden_size, cfg.hc_mult
    r = [
        ("hc_attn_fn", "device", [hc * H, hc * hc + 2 * hc]),
        ("hc_attn_base", "host", [hc * hc + 2 * hc]),
        ("hc_attn_scale", "host", [3]),
        ("hc_ffn_fn", "device", [hc * H, hc * hc + 2 * hc]),
        ("hc_ffn_base", "host", [hc * hc + 2 * hc]),
        ("hc_ffn_scale", "host", [3]),
        ("input_layernorm.weight", "device", [H]),
        ("post_attention_layernorm.weight", "device", [H]),
    ]
    if spec.attention == "kda":
        nh, hd, k = cfg.linear_num_heads, cfg.linear_head_dim, cfg.linear_conv_kernel_dim
        qkv = nh * hd
        r += [
            ("self_attn.q_proj.weight", "device", [H, qkv]),
            ("self_attn.k_proj.weight", "device", [H, qkv]),
            ("self_attn.v_proj.weight", "device", [H, qkv]),
            ("self_attn.f_a_proj.weight", "device", [H, hd]),
            ("self_attn.g_a_proj.weight", "device", [H, hd]),
            ("self_attn.b_proj.weight", "device", [H, nh]),
            ("self_attn.f_b_proj.weight", "device", [hd, qkv]),
            ("self_attn.g_b_proj.weight", "device", [hd, qkv]),
            ("self_attn.o_proj.weight", "device", [qkv, H]),
            ("self_attn.q_conv1d.weight", "device", [k, qkv]),
            ("self_attn.k_conv1d.weight", "device", [k, qkv]),
            ("self_attn.v_conv1d.weight", "device", [k, qkv]),
            ("self_attn.o_norm.weight", "device", [hd]),
            ("self_attn.dt_bias", "host", [qkv]),
            ("self_attn.A_log", "host", [nh]),
        ]
    else:
        nh = cfg.num_attention_heads
        r += [
            ("self_attn.q_a_proj.weight", "device", [H, cfg.q_lora_rank]),
            ("self_attn.kv_a_proj_with_mqa.weight", "device", [H, cfg.kv_lora_rank + cfg.qk_rope_head_dim]),
            ("self_attn.q_a_layernorm.weight", "device", [cfg.q_lora_rank]),
            ("self_attn.kv_a_layernorm.weight", "device", [cfg.kv_lora_rank]),
            ("self_attn.q_b_proj.weight", "device", [cfg.q_lora_rank, nh * cfg.qk_head_dim]),
            ("self_attn.kv_b_proj.W_uk", "device", [nh, cfg.qk_nope_head_dim, cfg.kv_lora_rank]),
            ("self_attn.kv_b_proj.W_uv", "device", [nh, cfg.kv_lora_rank, cfg.v_head_dim]),
            ("self_attn.o_proj.weight", "device", [nh * cfg.v_head_dim, H]),
        ]
    if spec.mlp == "dense":
        I = cfg.intermediate_size
        r += [
            ("mlp.gate_proj.weight", "device", [H, I]),
            ("mlp.up_proj.weight", "device", [H, I]),
            ("mlp.down_proj.weight", "device", [I, H]),
        ]
    else:
        Im, E = cfg.moe_intermediate_size, cfg.n_routed_experts
        r += [
            ("mlp.shared_experts.gate_proj.weight", "device", [H, Im * cfg.n_shared_experts]),
            ("mlp.shared_experts.up_proj.weight", "device", [H, Im * cfg.n_shared_experts]),
            ("mlp.shared_experts.down_proj.weight", "device", [Im * cfg.n_shared_experts, H]),
            ("mlp.gate.weight", "device", [H, E]),
            ("mlp.gate.e_score_correction_bias", "host", [E]),
        ]
        for e in range(E):
            r.append((f"mlp.experts.{e}.gate_up_proj", "expert", [H, 2 * Im]))
            r.append((f"mlp.experts.{e}.down_proj", "expert", [Im, H]))
    return r


def expected_data_bytes(shape, dtype, layout):
    if layout == "ROW_MAJOR":
        return math.prod(shape) * {"bfloat16": 2, "float32": 4}[dtype]
    h, w = (1, shape[0]) if len(shape) == 1 else (shape[-2], shape[-1])
    batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return batch * math.ceil(h / 32) * math.ceil(w / 32) * TILE_BYTES[dtype]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def data_bytes(path):
    import struct

    with open(path, "rb") as f:
        (hdr,) = struct.unpack("<Q", f.read(8))
    return os.path.getsize(path) - 8 - hdr


def audit_layer(L, cfg, LayerSpec, man, cache_dir, cache_dtype, codec, pool) -> dict:
    spec = LayerSpec.from_config(cfg, L)
    req = requested(cfg, spec)
    prefix = f"layers.{L}."
    in_cache = {k[len(prefix) :]: v for k, v in man.items() if k.startswith(prefix)}
    row = {
        "layer": L,
        "kind": spec.kind,
        "attention": spec.attention,
        "mlp": spec.mlp,
        "requested": len(req),
        "missing": [],
        "shape_mismatch": [],
        "dtype_mismatch": [],
        "layout_mismatch": [],
        "size_mismatch": [],
        "sha_mismatch": [],
        "decode_errors": [],
        "nonfinite": [],
    }
    files = []
    for name, mode, shape in req:
        e = in_cache.get(name)
        path = os.path.join(cache_dir, e["path"]) if e else None
        if e is None or not os.path.isfile(path):
            row["missing"].append(name)
            continue
        if list(e["shape"]) != list(shape):
            row["shape_mismatch"].append([name, e["shape"], shape])
        if e["dtype"] != cache_dtype(name):
            row["dtype_mismatch"].append([name, e["dtype"], cache_dtype(name)])
        if e["layout"] != "TILE":
            row["layout_mismatch"].append([name, e["layout"]])
        files.append((name, mode, e, path))

    # every file: size + sha256 (parallel, I/O bound)
    def check(item):
        name, mode, e, path = item
        ok_size = os.path.getsize(path) == e["bytes"] and data_bytes(path) == expected_data_bytes(
            e["shape"], e["dtype"], e["layout"]
        )
        return name, ok_size, sha256(path) == e["sha256"]

    for name, ok_size, ok_sha in pool.map(check, files):
        if not ok_size:
            row["size_mismatch"].append(name)
        if not ok_sha:
            row["sha_mismatch"].append(name)
    # decode: every non-expert tensor; experts 0 and E-1 as a sample
    decoded, sample = {}, {
        "mlp.experts.0.gate_up_proj",
        "mlp.experts.0.down_proj",
        f"mlp.experts.{cfg.n_routed_experts - 1}.gate_up_proj",
        f"mlp.experts.{cfg.n_routed_experts - 1}.down_proj",
    }
    for name, mode, e, path in files:
        if mode == "expert" and name not in sample:
            continue
        try:
            x = codec.decode(path, e["shape"], e["dtype"], e["layout"])
        except Exception as ex:  # noqa: BLE001
            row["decode_errors"].append([name, str(ex)[:200]])
            continue
        fin = bool(np.isfinite(x).all())
        if not fin:
            row["nonfinite"].append(name)
        if mode == "host" or name in ("mlp.gate.weight",) or mode == "expert":
            decoded[name] = {
                "shape": list(x.shape),
                "finite": fin,
                "min": float(np.nanmin(x)),
                "max": float(np.nanmax(x)),
                "mean_abs": float(np.nanmean(np.abs(x))),
            }
        del x
    row["host_vectors"] = {k: v for k, v in decoded.items() if not k.startswith("mlp.experts.")}
    row["expert_samples"] = {k: v for k, v in decoded.items() if k.startswith("mlp.experts.")}
    if spec.mlp == "moe":
        experts = [n for n, m, _ in req if m == "expert"]
        row["routed_experts"] = {
            "count": cfg.n_routed_experts,
            "files_present": sum(1 for n in experts if n not in row["missing"]),
            "gate_up_shape": [cfg.hidden_size, 2 * cfg.moe_intermediate_size],
            "down_shape": [cfg.moe_intermediate_size, cfg.hidden_size],
            "dtype": cache_dtype("mlp.experts.0.gate_up_proj"),
        }
        rb = decoded.get("mlp.gate.e_score_correction_bias")
        row["router_bias"] = rb and {**rb, "dtype": in_cache["mlp.gate.e_score_correction_bias"]["dtype"]}
        row["shared_expert"] = all(f"mlp.shared_experts.{p}_proj.weight" in in_cache for p in ("gate", "up", "down"))
    req_names = {n for n, _, _ in req}
    row["present_not_loaded"] = sorted(k for k in in_cache if k not in req_names)
    row["files_checked"] = len(files)
    row["bytes"] = sum(e["bytes"] for _, _, e, _ in files)
    bad = [
        k
        for k in (
            "missing",
            "shape_mismatch",
            "dtype_mismatch",
            "layout_mismatch",
            "size_mismatch",
            "sha_mismatch",
            "decode_errors",
            "nonfinite",
        )
        if row[k]
    ]
    row["status"] = "pass" if not bad else "fail"
    row["failed_checks"] = bad
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(WS, "weights", "manifest.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-44")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--negative-control", action="store_true", help="audit layers 0 and 3 against a tampered manifest")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(WS, "weights"))
    import tile_codec as codec

    cfg, LayerSpec = _load_config()
    cache_dtype = _dtype_rule()
    manifest = json.load(open(args.manifest))
    man, cache_dir = manifest["tensors"], manifest["cache_dir"]
    layers = range(*(lambda a, b: (a, b + 1))(*(int(x) for x in args.layers.split("-"))))
    if args.negative_control:
        man = {k: dict(v) for k, v in man.items() if k.startswith(("layers.0.", "layers.3."))}
        for key, (field, value, _) in TAMPER.items():
            if field is None:
                del man[key]
            else:
                man[key][field] = value
        layers = (0, 3)
    git = git_state()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool, open(args.out + ".tmp", "w") as f:
        for L in layers:
            t = time.time()
            row = audit_layer(L, cfg, LayerSpec, man, cache_dir, cache_dtype, codec, pool)
            row["seconds"] = round(time.time() - t, 1)
            row["manifest"] = os.path.relpath(args.manifest, WS)
            row["cache_tt_metal_sha"] = manifest.get("tt_metal_sha")
            row.update(git)
            if args.negative_control:
                want = sorted({c for k, (_, _, c) in TAMPER.items() if k.startswith(f"layers.{L}.")})
                row["negative_control"] = {
                    "tampered": [k for k in TAMPER if k.startswith(f"layers.{L}.")],
                    "expected_failed_checks": want,
                    "detected": row["status"] == "fail" and set(want) <= set(row["failed_checks"]),
                }
                assert row["negative_control"]["detected"], (L, row["failed_checks"], want)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(
                f"L{L:02d} {row['kind']:<10} {row['status']} files={row['files_checked']} "
                f"{row['bytes'] / 1e9:.2f} GB {row['seconds']}s {row['failed_checks']}",
                flush=True,
            )
    os.replace(args.out + ".tmp", args.out)
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
