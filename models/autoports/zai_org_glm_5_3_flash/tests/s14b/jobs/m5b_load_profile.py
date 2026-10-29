# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; mock or the mesh): Stage 14b — where the 45-layer weight load spends its time (program 15:16Z:
10 min for ~174 GB). Builds the TP layers of ARGS layers through StackBackend._build_tp with every load primitive
timed (FdTileCacheSource._load / .host, ttnn.load_tensor, the gate_up permutation, from_host_shards, to_device,
from_torch) and the bytes read. ARGS: layers ([4, 5]), cold (drop the page cache of the layers' files first: false)."""
import collections
import gc
import importlib
import os
import sys
import time

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.ep_experts as EPX  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder as FD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder as OD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402

for mod in (EPX, SM):
    importlib.reload(mod)
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [4, 5])]
acc = collections.defaultdict(float)
cnt = collections.Counter()


def timed(name, fn):
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[name] += time.perf_counter() - t0
            cnt[name] += 1

    return w


patches = [(ttnn, "load_tensor"), (ttnn, "from_host_shards"), (ttnn, "to_device"), (ttnn, "from_torch"),
           (FD, "_permuted_gate_up_file"), (OD.FdTileCacheSource, "_load"), (OD.FdTileCacheSource, "host")]  # fmt: skip
saved = [(o, n, getattr(o, n)) for o, n in patches]
for o, n, f in saved:
    setattr(o, n, timed(f"{getattr(o, '__name__', o)}.{n}", f))
try:
    cfg = hf_config()
    source = OD.FdTileCacheSource(CACHE_DIR, MANIFEST)
    t0 = time.perf_counter()
    be = SM.StackBackend.__new__(SM.StackBackend)  # only the layer construction (no head / embed / KV)
    be.mesh, be.tp, be.cfg, be.chunk, be.weight_cache, be.ep_weights = mesh, int(mesh.get_num_devices()), cfg, 1, None, True
    if args.get("prefetch") is not None:
        os.environ["GLM53_LOAD_PREFETCH"] = "1" if args["prefetch"] else "0"
    seq = be._build_tp(source, LAYERS)
    wall = time.perf_counter() - t0
finally:
    for o, n, f in saved:
        setattr(o, n, f)
rows = {k: dict(s=round(v, 2), n=cnt[k]) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
print(f"LOADPROF {C.now()} layers={LAYERS} prefetch={os.environ.get('GLM53_LOAD_PREFETCH', '1')} wall_s={wall:.1f} "
      f"load={getattr(be, 'load_info', None)} parts={rows}", flush=True)
os.environ.pop("GLM53_LOAD_PREFETCH", None)
del seq
gc.collect()
RESULT = dict(ok=True, wall_s=round(wall, 1), parts=rows)
