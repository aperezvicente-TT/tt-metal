# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; MOCK or device): the KV pool a served construction can give vLLM at max_model_len `max_seq`
(program 2026-10-22 08:30Z, int26 at 32K). On the resident serving backend (m5common.serving, the generator-owned
cache at its own max_seq_len) the per-chip DRAM (lock-step: chip 0 = every chip) is read, then probe caches are
allocated with StackBackend.allocate_kv_cache(max_batch, max_seq, num_blocks = n1 / n2) and released:
  per_block   = (DRAM(n2) - DRAM(n1)) / (n2 - n1)          (the latent pages of every sparse_dsa layer, replicated)
  fixed       = DRAM(n1) - n1 * per_block                   (per-cache state: KDA S / C rows, DSA stores / rings, IO)
  gen_cache   = the generator-owned cache's own cost (a third probe with its geometry)
  base        = DRAM(resident) - gen_cache                  (weights + buffers + traces)
  num_blocks  = floor((budget - margin - base - fixed) / per_block)   -> the vLLM KV pool at max_seq.
RUN IT LAST in a process: the probe caches rebind the backend's shared persistent hand-off buffers (one StackCache per
backend is the served shape: vLLM allocates one); after they are freed the generator's own cache is unusable (mock 229:
the next job's first op 'input_tensor.is_allocated()').
ARGS: b (32), layers ("all"), rev, max_seq (32768), n1 (64), n2 (128), budget_gib (28), margin_gib (0.5), record."""
import gc
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, MAXSEQ = int(args.get("b", 32)), int(args.get("max_seq", 32768))
N1, N2 = int(args.get("n1", 64)), int(args.get("n2", 128))
GIB = 2**30


def log(*a):
    print(f"KV-POOL {C.now()}", *a, flush=True)


def dram():
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=True, rev=args.get("rev"), log=log)
be, gen = ctx[key]["backend"], ctx[key]["gen"]
gc.collect()
d_res = dram()
gb = gen._own_binding()


def probe(max_seq, nb):
    d0 = dram()
    kv = be.allocate_kv_cache(max_batch=B, max_seq_len=max_seq, num_blocks=nb)
    d1 = dram()
    nb_eff = int(kv.num_blocks)
    del kv
    gc.collect()
    return d1 - d0, nb_eff


c1, n1 = probe(MAXSEQ, N1)
c2, n2 = probe(MAXSEQ, N2)
per_block = (c2 - c1) / max(1, n2 - n1)
fixed = c1 - n1 * per_block
g_cost, g_nb = probe(int(gb.max_seq_len), None)  # the generator-owned geometry (num_blocks = max_batch x bps)
base = d_res - g_cost
budget, margin = float(args.get("budget_gib", 28)) * GIB, float(args.get("margin_gib", 0.5)) * GIB
nb_max = int((budget - margin - base - fixed) // per_block) if per_block > 0 else None
row = dict(test=f"KV pool at max_seq {MAXSEQ}: per-chip DRAM of the served construction ({len(ids)} layers, B={B}, tp"
                f"{int(mesh.get_num_devices())}) and the latent blocks that fit {args.get('budget_gib', 28)} GiB / chip",
           status="info", milestone="M5c", resident_gib=round(d_res / GIB, 3), gen_cache_gib=round(g_cost / GIB, 3),
           gen_cache_blocks=g_nb, base_gib=round(base / GIB, 3), per_block_kib=round(per_block / 1024, 1),
           fixed_gib_at_max_seq=round(fixed / GIB, 3), num_blocks_max=nb_max,
           tokens_max=(nb_max * 64 if nb_max else None), users_at_max_seq=(nb_max * 64 // MAXSEQ if nb_max else None),
           mock=C.MOCK, tag=args.get("tag", "kv-pool-probe"))  # fmt: skip
log(row)
if args.get("record", True) and not C.MOCK:
    C.record("perf", row)
RESULT = row
