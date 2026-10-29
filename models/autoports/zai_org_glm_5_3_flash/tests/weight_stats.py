# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record per-tensor stats (name, shape, dtype, mean, std) of the real weights of one layer per kind.

CPU only (run under `.agents/scripts/no-tt-devices`): tensors are decoded from the shared tile cache with the pure-numpy
`weights/tile_codec.py` (decoding a host bfp tensor through ttnn opens every chip). Routed experts are summarised over
a deterministic sample of experts (same shape for all 288). Output: tests/weight_stats.json, consumed by
test_synthetic.py to generate deterministic synthetic weights with the real shapes.

    python models/autoports/zai_org_glm_5_3_flash/tests/weight_stats.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

LAYERS = {"dense_kda": 0, "sparse_kda": 4, "sparse_dsa": 3}
EXPERT_SAMPLE = (0, 17, 101, 287)


def main():
    ws = os.environ["GLM53_WS"]
    sys.path.insert(0, os.path.join(ws, "weights"))
    import tile_codec  # noqa: E402

    cache = os.environ["GLM53_CACHE_DIR"]
    tensors = json.load(open(os.path.join(ws, "weights", "manifest.json")))["tensors"]
    out = {"source": "GLM53 tile cache (values as the device sees them)", "layers": {}}
    for kind, layer in LAYERS.items():
        prefix = f"layers.{layer}."
        stats = {}
        for key, e in sorted(tensors.items()):
            if not key.startswith(prefix) or ".indexer." in key:
                continue  # the DSA indexer is unused in the dense-exact regime
            name = key[len(prefix) :]
            if ".experts." in name:
                idx = int(name.split(".")[2])
                if idx not in EXPERT_SAMPLE:
                    continue
                name = "mlp.experts.*." + name.split(".")[-1]
            x = tile_codec.decode(os.path.join(cache, e["path"]), e["shape"], e["dtype"], e["layout"]).astype(
                np.float64
            )
            s = stats.setdefault(name, {"shape": e["shape"], "dtype": e["dtype"], "n": 0, "sum": 0.0, "sq": 0.0})
            s["n"] += x.size
            s["sum"] += float(x.sum())
            s["sq"] += float((x * x).sum())
        for s in stats.values():
            mean = s["sum"] / s["n"]
            s["mean"] = mean
            s["std"] = float(np.sqrt(max(s["sq"] / s["n"] - mean * mean, 0.0)))
            del s["n"], s["sum"], s["sq"]
        out["layers"][kind] = {"layer": layer, "tensors": stats}
        print(kind, layer, len(stats), "tensors", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
