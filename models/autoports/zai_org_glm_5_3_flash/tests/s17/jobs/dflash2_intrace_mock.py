# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 MOCK or the pair mock): the speculative step with the DFlash2 drafter INSIDE it (SpecMTP
drafter "dflash2", the verify at the plain w8 width) on the served construction: build (verify stacks split at the
tap layers, the kind resident), prepare (eager compile), capture (mock), boot / steps / read / leave, then a plain
step. Census: generic_op launches of the step by part (verify / ctl / drafter), the program-cache keys of a further
step (0 new), DRAM GiB per chip (trace_report), L1 floor (GLM53_SPEC_L1_DEBUG). ARGS: layers ("0,1,2,3,4" or "all"),
b (32), k (7), narrow ("none"), steps (2), weights ("random"), profile ("auto"), max_seq_len (2048), conv ("fused" |
"ops"), mm ("stream" | "ds"), mm_cores, conv_cores (rung 3 M1 knobs), tag."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib  # noqa: E402

import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 32)), int(args.get("k", 7))
_L = str(args.get("layers", "0,1,2,3,4"))


def log(*a):
    print(f"S17-DF2I {C.now()}", *a, flush=True)


from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
extra = dict(GLM53_SPEC_DRAFTER="dflash2", GLM53_SPEC_VERIFY_W="8", GLM53_DF2_WEIGHTS=str(args.get("weights", "random")),
             GLM53_DF2_TAP_MODE=str(args.get("tap_mode", "fold")), GLM53_DF2_CONV=str(args.get("conv", "fused")),
             GLM53_DF2_MM=str(args.get("mm", "stream")), GLM53_DF2_CONV_RES=str(args.get("conv_res", "0")),
             GLM53_DF2_HEADS_SHARDED=str(args.get("heads_sharded", "0")), GLM53_DF2_LAYER=str(args.get("layer", "ops")))  # fmt: skip
for k_, e_ in (("mm_cores", "GLM53_DF2_MM_CORES"), ("conv_cores", "GLM53_DF2_CONV_CORES")):
    if args.get(k_) is not None:
        extra[e_] = str(args[k_])
if str(args.get("stack_dsa", "0")) == "1":
    extra["GLM53_STACK_DSA"] = "1"
# launch census: every generic_op of the step by the caller's module (verify stacks / spec ctl / the drafter kind)
orig = ttnn.generic_op
hits = []


def hooked(io, pd):
    import inspect

    who = "other"
    for fr in inspect.stack()[1:12]:
        b = os.path.basename(fr.filename)
        if b in ("dflash2_kind.py", "mm_stream.py", "dflash2_mm.py"):
            who = "drafter"
            break
        if b == "spec_mtp.py" and fr.function in ("_ctl_accept", "_gather"):
            who = "ctl"
            break
        if b in (
            "tp_stack.py",
            "mk_stack.py",
            "mk_mla.py",
            "mk_moe.py",
            "tp_mla.py",
            "tp_dsa.py",
            "mk_head.py",
            "mk_layer.py",
        ):
            who = "verify"
            break
    hits.append(who)
    return orig(io, pd)


t0 = time.time()
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")),
                         extra_env=extra, max_seq_len=int(args.get("max_seq_len", 2048)), log=log)  # fmt: skip
t1 = time.time()
sp = be.spec
assert sp is not None and sp.drafter == "dflash2" and sp.df2 is not None, "GLM53_SPEC_DRAFTER=dflash2: no kind"
kv = be._kv
io = kv.extra["io"]
rep = be.trace_report()
log("built", rep, f"{t1 - t0:.1f}s; verify runs", [r for r, _ in sp.vstacks], "taps slots", len(sp.df2.taps))
tok = torch.zeros(B, dtype=torch.int64)
tok[0] = 1234
pos = torch.full((B,), -1, dtype=torch.int64)
pos[0] = 10
be.write(io.tokens, tok)
be.write(io.cur_pos, pos)
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
sp.capture()
sp.enter(pt[0])
ttnn.generic_op = hooked
ttnn_ops = {}
import collections  # noqa: E402

# the stock-op launches of the drafter are counted by the kind itself (ops.launches); generic_op hits by owner
leaked = {}
try:
    sp.df2.ops.launches = 0
    sp.df2.ops.census.clear()
    hits.clear()
    # the mock's trace-allocation guard (s17common.guard_capture_fn): the step and the boot as the capture records
    # them must leave no tensor of their own alive (the device guard refuses every later checked replay otherwise)
    leaked["step"] = S.guard_capture_fn(sp.step, log=log, name="step")
    ttnn.synchronize_device(mesh)
    gen_ops = dict(collections.Counter(hits))
    df2_launches = sp.df2.ops.launches  # the step's drafter launches (the boot below adds its own)
    leaked["boot"] = S.guard_capture_fn(sp.df2.boot, log=log, name="boot")
    n_keys = mesh.num_program_cache_entries()
    for _ in range(int(args.get("steps", 2)) - 1):
        sp.step()
    ttnn.synchronize_device(mesh)
    new_keys = mesh.num_program_cache_entries() - n_keys
finally:
    ttnn.generic_op = orig
rec = sp.read()
out = sp.leave()
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
be.replay(wts.get(w0, kv.extra.get("decode_trace")))
ttnn.synchronize_device(mesh)
clean = new_keys == 0 and not any(leaked.values())
row = dict(test="dflash2-intrace-mock", milestone="DF2I", status="pass" if clean else "fail", K=K, B=B, layers=_L,
           trace_alloc_leaks={k: [list(map(str, x)) for x in v] for k, v in leaked.items()},
           tp=int(mesh.get_num_devices()), stack_dsa=extra.get("GLM53_STACK_DSA", "0"), verify_runs=[r for r, _ in sp.vstacks],
           conv=extra["GLM53_DF2_CONV"], mm=extra["GLM53_DF2_MM"], mm_cores=args.get("mm_cores"), conv_cores=args.get("conv_cores"),
           conv_res=extra["GLM53_DF2_CONV_RES"], heads_sharded=extra["GLM53_DF2_HEADS_SHARDED"], layer=extra["GLM53_DF2_LAYER"],
           mk_l1=(sp.df2.mk.l1 if sp.df2.mk is not None else None), mm_plans=sp.df2.ops.mm_plans, drafter_census=dict(sp.df2.ops.census),
           step_generic_ops_by_owner=gen_ops, drafter_launches=df2_launches, new_keys_further_step=new_keys,
           program_cache_keys=n_keys, trace_report=rep, counters=sp.counters, rec_keys=sorted(rec), build_s=round(t1 - t0, 1),
           tag=args.get("tag"))  # fmt: skip
S.record("tests", row)
log(row)
RESULT = dict(
    ok=clean, leaks=leaked, gen_ops=gen_ops, drafter_launches=df2_launches, rec=rec, leave=out, trace_report=rep
)
