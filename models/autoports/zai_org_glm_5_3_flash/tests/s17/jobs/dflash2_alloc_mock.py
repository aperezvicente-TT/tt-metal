# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the pair MOCK): the host-side equivalent of TT_METAL_TRACE_ALLOC_TRACKING for the speculative step
with the dflash2 drafter -- the mock never captures, so the functions SpecMTP.capture() would record (sp._boot,
sp._step) run eagerly here with ttnn's allocating entry points wrapped (allocate_tensor_on_device, empty, from_torch,
to_device, copy_host_to_device_tensor, zeros / full): every call is logged with its model-side call stack; after the
function returns (+ gc.collect()) the live ttnn.Tensor objects are enumerated (gc) and every logged allocation whose
buffer address AND shape still belong to a live tensor is reported = a buffer allocated inside the would-be capture
that outlives it (the guard's 'still alive before trace replay'). ARGS: layers ("0,1,2,3,4"), b (32), k (7), tag."""
import gc
import importlib
import os
import sys
import traceback

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 32)), int(args.get("k", 7))
ids = [int(v) for v in str(args.get("layers", "0,1,2,3,4")).split(",")]


def log(*a):
    print(f"S17-DF2A {C.now()}", *a, flush=True)


extra = dict(GLM53_SPEC_DRAFTER="dflash2", GLM53_SPEC_VERIFY_W="8", GLM53_DF2_WEIGHTS="random",
             GLM53_DF2_TAP_MODE=str(args.get("tap_mode", "mean")), GLM53_DF2_CONV="fused")  # fmt: skip
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow="none", profile="served24", extra_env=extra, log=log)
sp, kv = be.spec, be._kv
io = kv.extra["io"]
tok = torch.zeros(B, dtype=torch.int64)
tok[0] = 1234
pos = torch.full((B,), -1, dtype=torch.int64)
pos[0] = 10
be.write(io.tokens, tok)
be.write(io.cur_pos, pos)
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
sp.capture()  # (mock: the functions)
sp.enter(pt[0])
# probe=1: the guard's NEGATIVE check -- the kind's stage probes keep capture-time handles by design (a diagnosis
# mode); with them on, the guard MUST report leaks (the job then passes iff it does)
NEG = str(args.get("probe", "0")) == "1"
if NEG:
    sp.df2.probe = {}

NAMES = [
    "allocate_tensor_on_device",
    "empty",
    "from_torch",
    "to_device",
    "copy_host_to_device_tensor",
    "zeros",
    "full",
    "ones",
]
records = []
orig = {n: getattr(ttnn, n) for n in NAMES if hasattr(ttnn, n)}


def _site():
    out = []
    for fr in traceback.extract_stack()[:-2]:
        if "zai_org_glm_5_3_flash" in fr.filename:
            out.append(f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}")
    return " > ".join(out[-4:])


def wrap(name, fn):
    def w(*a, **kw):
        r = fn(*a, **kw)
        try:
            addr = (
                r.buffer_address()
                if hasattr(r, "buffer_address") and r.storage_type() == ttnn.StorageType.DEVICE
                else None
            )
            shape = tuple(r.shape)
        except Exception:
            addr, shape = None, None
        records.append(dict(fn=name, addr=addr, shape=shape, site=_site()))
        return r

    return w


def live_tensors():
    gc.collect()
    out = {}
    for o in gc.get_objects():
        if isinstance(o, ttnn.Tensor):
            try:
                if o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                    out[(o.buffer_address(), tuple(o.shape))] = o
            except Exception:
                pass
    return out


report = {}
# NEG: the boot only -- its probe handles (L1 head shards among them) would clash with the step's static CBs
for name, fn in (("boot", sp._boot),) if NEG else (("boot", sp._boot), ("step", sp._step)):
    for n, f in orig.items():
        setattr(ttnn, n, wrap(n, f))
    records.clear()
    before = S.live_device_tensors()
    try:
        fn()
        ttnn.synchronize_device(mesh)
    finally:
        for n, f in orig.items():
            setattr(ttnn, n, f)
    live = live_tensors()
    alive = [r for r in records if r["addr"] is not None and (r["addr"], r["shape"]) in live]
    leaked = S.new_live_tensors(before)  # EVERY tensor the function left alive (op outputs included)
    for k, _ in leaked:
        log("  LEAK (alive after the function, created inside it):", k)
    alive += [
        dict(fn="(op output)", addr=k[0], shape=k[1], site="")
        for k, _ in leaked
        if not any(r["addr"] == k[0] for r in alive)
    ]
    by_fn = {}
    for r in records:
        by_fn[r["fn"]] = by_fn.get(r["fn"], 0) + 1
    report[name] = dict(calls=by_fn, alive=alive, n_records=len(records))
    log(name, "allocating calls", by_fn, "alive after:", len(alive))
    for r in alive:
        log("  ALIVE", r)
    sites = {}
    for r in records:
        if r["fn"] in ("allocate_tensor_on_device", "from_torch", "to_device", "copy_host_to_device_tensor"):
            sites[(r["fn"], r["site"])] = sites.get((r["fn"], r["site"]), 0) + 1
    for (fn_, site), n in sorted(sites.items(), key=lambda kv: -kv[1])[:20]:
        log(f"  {fn_} x{n}: {site}")
if NEG:  # the negative check leaks on purpose: drop the handles (a later job in this server shares the backend)
    sp.df2.probe = None
    gc.collect()
any_alive = any(report[k]["alive"] for k in report)
ok = any_alive if NEG else not any_alive
S.record("tests", dict(test="dflash2-alloc-mock", milestone="DF2I", status="pass" if ok else "fail", negative_check=NEG,
                       K=K, B=B, layers=str(args.get("layers")), tp=int(mesh.get_num_devices()), tap_mode=extra["GLM53_DF2_TAP_MODE"],
                       report={k: dict(calls=v["calls"], alive=v["alive"]) for k, v in report.items()}, tag=args.get("tag")))  # fmt: skip
RESULT = dict(
    ok=ok, negative_check=NEG, report={k: dict(calls=v["calls"], n_alive=len(v["alive"])) for k, v in report.items()}
)
