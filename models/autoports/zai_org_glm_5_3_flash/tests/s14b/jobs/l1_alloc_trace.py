# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; MOCK, diagnostics): which Python-allocated L1 tensors are created during the serving backend's
warm-up, by which step (the full width, a native width, a narrow width), and which are still alive after it — the
lowest addresses first (the buffers that bound the static-CB region of the programs compiled before them). Wraps
ttnn.empty / zeros / from_torch / allocate_tensor_on_device for the build + warm-up and tags each allocation with the
StackBackend._step it happened in. ARGS: b (32), layers ("all"), native (true), rev, top (25)."""
import collections
import gc
import os
import sys
import traceback

import ttnn

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
recs = []
phase = {"name": "build"}
names = [n for n in ("empty", "zeros", "from_torch", "allocate_tensor_on_device") if hasattr(ttnn, n)]
orig = {n: getattr(ttnn, n) for n in names}


def wrap(n, f):
    def g(*a, **k):
        t = f(*a, **k)
        try:
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                st = [fr for fr in traceback.extract_stack(limit=12)[:-1] if "zai_org_glm_5_3_flash/tt/" in fr.filename]
                # no reference to t is kept (that would keep transient buffers alive and change the L1 layout)
                recs.append((phase["name"], n, t.buffer_address(), tuple(t.shape), str(t.dtype).split(".")[-1],
                             " < ".join(f"{os.path.basename(fr.filename)}:{fr.lineno}" for fr in st[::-1][:5])))
        except Exception:  # noqa: BLE001 -- host tensors
            pass
        return t

    return g


orig_step = SM.StackBackend._step
blocks = {"seen": None, "new": []}  # the allocator's L1 block table (C++ allocations included), diffed per step


def l1_blocks():
    try:
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        out = set()
        for b in mv.block_table:
            d = dict(b)
            if str(d.get("allocated", "")).lower() in ("yes", "true", "1"):
                out.add((int(d.get("address", 0)), int(d.get("size", 0))))
        return out
    except Exception as e:  # noqa: BLE001
        print("L1TRACE block table unavailable", repr(e), flush=True)
        return None


def step(self, kv, chunks=None, width=None):
    name = f"step w={width if width is not None else 'full'}"
    if blocks["seen"] is None:
        blocks["seen"] = l1_blocks() or set()
        print("L1TRACE before the first step: lowest allocated L1 blocks", sorted(blocks["seen"])[:6], flush=True)
    phase["name"] = name
    r = orig_step(self, kv, chunks, width)
    now = l1_blocks()
    if now is not None:
        for blk in sorted(now - blocks["seen"]):
            blocks["new"].append((blk[0], blk[1], name))
        blocks["seen"] = now
    return r


# finer tags inside a step: the allocator blocks each composite call adds (C++ allocations of ops included)
import models.autoports.zai_org_glm_5_3_flash.tt.mk_head as MKH  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_dsa as TPD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp_mla as TPM  # noqa: E402

sub_orig = []


def tag_calls(cls, label):
    f = cls.__call__

    def g(self, *a, **k):
        before = l1_blocks() if blocks["seen"] is not None else None
        r = f(self, *a, **k)
        if before is not None:
            after = l1_blocks() or set()
            t = k.get("t", getattr(self, "t", None))
            for blk in sorted(after - before):
                blocks["calls"].append((blk[0], blk[1], phase["name"], f"{label} t={t}"))
        return r

    sub_orig.append((cls, f))
    cls.__call__ = g


blocks["calls"] = []
for cls, label in ((TPM.TPMla, "TPMla"), (TPD.TPMoE, "TPMoE"), (MKH.MkEmbed, "MkEmbed"), (MKH.MkHead, "MkHead")):
    tag_calls(cls, label)
for nm in dir(MKS):
    c = getattr(MKS, nm)
    if isinstance(c, type) and "Stack" in nm and hasattr(c, "__call__") and c.__module__ == MKS.__name__:
        tag_calls(c, nm)
import models.autoports.zai_org_glm_5_3_flash.tt.tp_stack as TPS  # noqa: E402

for cls in (TPS.TPStack, TPS.TPSparseStack):
    if "__call__" in cls.__dict__:
        tag_calls(cls, cls.__name__)
for n, f in orig.items():
    setattr(ttnn, n, wrap(n, f))
SM.StackBackend._step = step
try:
    cfg = hf_config()
    _L = args.get("layers", "all")
    ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
    key = M5.serving(ctx, mesh, SM, G, B=int(args.get("b", 32)), ids=ids, cfg=cfg, native=bool(args.get("native", True)),
                     rev=args.get("rev"), log=lambda *a: print("L1TRACE", *a, flush=True))  # fmt: skip
finally:
    for n, f in orig.items():
        setattr(ttnn, n, f)
    SM.StackBackend._step = orig_step
    for cls, f in sub_orig:
        cls.__call__ = f
gc.collect()
live = set()
for o in gc.get_objects():  # the L1 tensors Python still holds (ttnn.Tensor is not GC-tracked: scan the containers)
    vals = o.values() if isinstance(o, dict) else (o if isinstance(o, (list, tuple)) else ())
    for v in vals:
        if isinstance(v, ttnn.Tensor):
            try:
                if v.is_allocated() and v.memory_config().buffer_type == ttnn.BufferType.L1:
                    live.add(v.buffer_address())
            except Exception:  # noqa: BLE001
                pass
last = {}
for r in recs:  # the latest allocation at an address is the live one
    last[r[2]] = r
alive = sorted((addr, ph, n, shape, dt, where) for addr, (ph, n, _a, shape, dt, where) in last.items() if addr in live)
by_phase = collections.Counter(a[1] for a in alive)
print("L1TRACE alive L1 tensors by step:", dict(by_phase), "allocated total:", len(recs), flush=True)
for a in alive[: int(args.get("top", 25))]:
    print("L1TRACE", a, flush=True)
final = l1_blocks() or set()
newb = sorted(b for b in blocks["new"] if (b[0], b[1]) in final)
print("L1TRACE allocator blocks added by step and still allocated (lowest first):", flush=True)
for b in newb[:30]:
    print("L1TRACE block", b, flush=True)
print("L1TRACE lowest allocated L1 blocks at the end", sorted(final)[:8], flush=True)
for b in sorted(b for b in blocks["calls"] if (b[0], b[1]) in final)[:20]:
    print("L1TRACE call-block", b, flush=True)
RESULT = dict(ok=True, alive=len(alive), by_phase=dict(by_phase), lowest=[a[:5] for a in alive[:10]],
              blocks=newb[:20])
