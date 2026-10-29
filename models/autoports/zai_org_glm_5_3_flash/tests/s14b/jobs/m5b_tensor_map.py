# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the mesh or a mock): Stage 14b -> Stage 12 (device-resident fast start, program 20:00Z) — the
tensor map of a StackBackend: every device tensor reachable from the backend and the shared weight cache
(CTX["s14b_weights"]), with its owner path, per-chip shape / bytes, dtype, layout, memory config and DRAM / L1 address,
sorted by address (the lock-step allocator hands addresses out in allocation order). The LAYOUT KEY hashes the sorted
(path, per-chip shape, dtype, memory config) list — code changes that keep the weight layout keep the key; the ADDRESS
KEY adds the addresses (a reopen that rebuilds in the same order lands every tensor at the same address).
Writes doc/ttnn/14b-persistent-stack/tensor_map/<tag>.json. ARGS: b (32), layers ("all"), native (true), tag, rev."""
import gc
import hashlib
import importlib
import json
import os
import sys

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (M5, G, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NATIVE = bool(args.get("native", True))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
TAG = args.get("tag", f"stack_B{B}{'n' if NATIVE else ''}_{len(ids)}L")
# the stage's evidence dir under the workspace (GLM53_EVIDENCE of the device wrappers is the doc root, not the stage)
OUT = os.path.join(os.environ.get("GLM53_WS", "/opt/glm53/kit"), "doc", "ttnn",
                   "14b-persistent-stack", "tensor_map")  # fmt: skip
TILE_BYTES = {"BFLOAT4_B": 576, "BFLOAT8_B": 1088, "BFLOAT16": 2048, "FLOAT32": 4096, "UINT32": 4096, "INT32": 4096,
              "UINT16": 2048, "UINT8": 1024}  # fmt: skip
ELEM_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1}


def log(*a):
    print(f"M5B-TMAP {C.now()}", *a, flush=True)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=args.get("rev"), log=log)
backend = ctx[key]["backend"]


def chip_bytes(t) -> int:
    d = ttnn.get_device_tensors(t)[0] if int(mesh.get_num_devices()) > 1 else t
    shp = [int(v) for v in d.padded_shape] if hasattr(d, "padded_shape") else [int(v) for v in d.shape]
    dt = str(d.dtype).split(".")[-1]
    vol = 1
    for v in shp:
        vol *= v
    if str(d.layout).endswith("TILE"):
        return vol // 1024 * TILE_BYTES.get(dt, 2048)
    return vol * ELEM_BYTES.get(dt, 2)


seen, rows = set(), []


def visit(o, path, depth):
    if depth > 9 or id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, ttnn.Tensor):
        try:
            if o.storage_type() != ttnn.StorageType.DEVICE or not o.is_allocated():
                return
            mc = o.memory_config()
            d = ttnn.get_device_tensors(o)[0] if int(mesh.get_num_devices()) > 1 else o
            rows.append(dict(path=path, addr=int(o.buffer_address()), chip_bytes=chip_bytes(o),
                             chip_shape=[int(v) for v in d.shape], dtype=str(o.dtype).split(".")[-1],
                             layout=str(o.layout).split(".")[-1], buffer=str(mc.buffer_type).split(".")[-1],
                             mem=str(mc.memory_layout).split(".")[-1],
                             shard=None if mc.shard_spec is None else list(mc.shard_spec.shape)))  # fmt: skip
        except Exception as e:  # noqa: BLE001 -- a deallocated / host view
            log("skip", path, repr(e)[:80])
        return
    if isinstance(o, dict):
        for k, v in list(o.items()):
            visit(v, f"{path}[{k!r}]", depth + 1)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            visit(v, f"{path}[{i}]", depth + 1)
    elif hasattr(o, "__dict__") and not isinstance(o, type) and type(o).__module__.split(".")[0] in ("models", "tests",
                                                                                                    "m5common"):
        for k, v in list(vars(o).items()):
            if k.startswith("__"):
                continue
            visit(v, f"{path}.{k}", depth + 1)


visit(ctx.get("s14b_weights", {}), "weights", 0)
visit(backend, "backend", 0)
rows.sort(key=lambda r: (r["buffer"], r["addr"]))
lay = hashlib.sha256(json.dumps(sorted((r["path"], r["chip_shape"], r["dtype"], r["mem"], r["shard"]) for r in rows),
                                default=str).encode()).hexdigest()[:16]  # fmt: skip
adr = hashlib.sha256(json.dumps([(r["path"], r["addr"]) for r in rows], default=str).encode()).hexdigest()[:16]
dram = [r for r in rows if r["buffer"] == "DRAM"]
summ = dict(tag=TAG, B=B, native_t=NATIVE, layers=len(ids), tensors=len(rows), dram_tensors=len(dram),
            dram_gib_per_chip=round(sum(r["chip_bytes"] for r in dram) / 2**30, 3),
            weights_gib_per_chip=round(sum(r["chip_bytes"] for r in dram if r["path"].startswith("weights")) / 2**30, 3),
            layout_key=lay, address_key=adr, code=os.environ.get("GLM53_CODE_SHA"),
            report_dram_gib_per_chip=backend.trace_report().get("dram_gib_per_chip"))  # fmt: skip
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, f"{TAG}.json"), "w") as f:
    json.dump(dict(summary=summ, tensors=rows), f, indent=0, default=str)
log("SUMMARY", summ)
RESULT = dict(ok=True, **summ)
