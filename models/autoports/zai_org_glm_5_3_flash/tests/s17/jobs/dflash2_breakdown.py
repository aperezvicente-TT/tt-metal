# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, unwatched; Stage 17 DFlash2 perf): where the in-trace DFlash2 speculative step's time goes (the rung-3
sizing). After the spec traces exist (enter), each piece is captured as its OWN trace and replayed back-to-back:
  verify     the main model's verify at w8, with the tap copies on the tap layers
  accept     SpecCtl ACCEPT
  df2_prep   the drafter's PREP (kernels/spec/df2_ctl.cpp)
  df2_ctx    the drafter's ctx program (this step's rows -> the K / V of every drafter layer)
  df2_block  the drafter's block program (8 layers + lm_head + top-k + the selector walk)
  mm_ctx     the ctx program's streaming matmuls alone (fc + 8 x qkv), on persistent dummy inputs
  mm_block   the block program's streaming matmuls alone (8 x (kp_attn, qkv, o, kp_mlp, ug, down) + lm_head + hproj)
  op_<kind>  (rung 3, `ops_bench=1`) the step's calls of ONE stock-op kind replayed alone: every call the drafter's
             ctx + block programs made of that kind (all_reduce, sdpa, head_norm, concat_heads, conv_fused[_res],
             rms, cache_write, rope, create_heads, rot_mats, all_gather, swiglu, typecast, add, topk, embedding,
             selector_scores, slice_pad) with its real arguments: persistent tensors (caches, pages, weights, tables)
             as they are, transient ones as dummies of the recorded TensorSpecs -> the 18 ms of block_other by op kind
then the whole step, the bootstrap and the plain step. ctx_other = df2_ctx - mm_ctx and block_other = df2_block -
mm_block are the stock ops' share (layout / norm / rope / conv / attention / collectives / top-k ...). The dummy inputs
copy the TensorSpecs of the real inputs, recorded while df2_ctx / df2_block are captured (the same program keys: no
compile after a capture), and are allocated in the trace-allocation tracker's corruptible scope (their contents do not
matter). Rounds restore the speculative state like s3_breakdown. ARGS: layers ("0,1,2,3,4"), b (32), k (7), narrow ("none"), prompt ("aime24_00"), reps (2000),
per_round (200), profile ("served24"), max_seq_len (4096), env (the dflash2 knobs), ops_bench (1), tag."""
import importlib
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
B, K = int(args.get("b", 32)), int(args.get("k", 7))
REPS, PR = int(args.get("reps", 2000)), int(args.get("per_round", 200))
_L = str(args.get("layers", "0,1,2,3,4"))
WS = "/opt/glm53/kit"  # (host-8chip and host-p300)
TILE = 32
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows, write_rows  # noqa: E402


def log(*a):
    print(f"S17-DF2BD {C.now()}", *a, flush=True)


cfg = hf_config()
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}
XENV.setdefault("GLM53_SPEC_DRAFTER", "dflash2")
XENV.setdefault("GLM53_SPEC_VERIFY_W", "plain")
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")),
                         profile=str(args.get("profile", "served24")), log=log, extra_env=XENV,
                         max_seq_len=int(args.get("max_seq_len", 4096)), rev=args.get("rev"))  # fmt: skip
sp, kv = be.spec, be._kv
df2 = sp.df2
assert df2 is not None and sp.K == K, (sp.drafter, sp.K)
io = kv.extra["io"]
pt = (torch.arange(B * kv.blocks_per_seq) % kv.num_blocks).reshape(B, -1)
be.write(io.page_table, pt)
wts = kv.extra.get("decode_traces", {})
w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [B])
plain = wts.get(w0, kv.extra.get("decode_trace"))
PROMPTS = {
    q["id"]: q for q in json.load(open(os.path.join(WS, "doc/prework/p5_mtp_acceptance/prompts.json")))["prompts"]
}
prompt = [int(v) for v in PROMPTS[str(args.get("prompt", "aime24_00"))]["ids"]]
max_pos = int(kv.blocks_per_seq) * int(be.block_size)
assert len(prompt) + (K + 1) * PR + 8 < max_pos, ("rounds would run past the page table", len(prompt), PR, max_pos)

# the drafter's matmul LAUNCH UNITS in program order (DFlash2Kind.mm_units: an mm_stream program each in
# GLM53_DF2_MM=stream -- 9 ctx + 50 block; in "ds" the same-input groups are one ReaderPhases program: 2 ctx + 49
# block, tt/dflash2_mm.py); the kind's probe (mm_probe) notes each unit's real input TensorSpec when df2_ctx /
# df2_block are captured (the ctx program runs the qkv group, the block program the per-layer units)
SPECS, DUMMY = {}, {}
df2.mm_probe = SPECS
MM_CTX, MM_BLOCK = df2.mm_units["ctx"], df2.mm_units["block"]


def mm_run(lst):
    for name, unit in lst:
        unit(DUMMY[name])  # (the outputs are trace-internal buffers: dropped)


# ---------------------------------------------------------------- the stock ops by kind (rung 3 sizing)
import collections  # noqa: E402

OP_KINDS = ["all_reduce", "sdpa", "head_norm", "concat_heads", "conv_fused", "rms", "cache_write", "rope", "create_heads",
            "rot_mats", "all_gather", "swiglu", "typecast", "add", "topk", "embedding", "selector_scores", "slice",
            "gather", "to_rm", "prev_ids", "add_offset", "concat", "pad_rows"]  # fmt: skip
CALLS = collections.defaultdict(list)  # kind -> [(args, kwargs)] with tensors replaced by their records
REC = [False]
OPS_BENCH = int(args.get("ops_bench", 1))


def _rec(v):
    if isinstance(v, ttnn.Tensor):
        return ("T", v.spec, int(v.buffer_address()), tuple(int(s) for s in v.shape), str(v.dtype))
    if isinstance(v, dict):
        return {k: _rec(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_rec(x) for x in v)
    return v


def _wrap(kind):
    f = getattr(df2.ops, kind)

    def w(*a, **kw):
        if REC[0]:
            CALLS[kind].append((_rec(a), _rec(kw)))
        return f(*a, **kw)

    setattr(df2.ops, kind, w)
    return f


if OPS_BENCH:
    ORIG = {k: _wrap(k) for k in OP_KINDS if hasattr(df2.ops, k)}  # (the residual fold goes through conv_fused)
LIVE, DUM2 = {}, {}


def _mat(v):
    """a recorded argument -> a usable one: a live persistent tensor with that (address, shape, dtype), else a dummy"""
    if isinstance(v, tuple) and v and v[0] == "T":
        _, spec, addr, shape, dt = v
        key = (addr, shape, dt)
        if key in LIVE:
            return LIVE[key]
        if key not in DUM2:
            DUM2[key] = ttnn.allocate_tensor_on_device(spec, mesh)
        return DUM2[key]
    if isinstance(v, dict):
        return {k: _mat(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_mat(x) for x in v)
    return v


def op_run(kind):
    f = ORIG[kind]
    for a, kw in CALLS[kind]:
        f(*_mat(a), **_mat(kw))


def make_op_dummies():
    from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

    LIVE.update(S.live_device_tensors())
    with corruptible_allocation_scope(mesh):
        for kind in CALLS:
            for a, kw in CALLS[kind]:
                _mat(a), _mat(kw)
    log("op kinds", {k: len(v) for k, v in CALLS.items()}, "dummies", len(DUM2), "live", len(LIVE))


def make_dummies():
    from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

    missing = [n for n, _ in MM_CTX + MM_BLOCK if n not in SPECS]
    assert not missing, ("matmul units not seen in df2_ctx / df2_block", missing)
    with corruptible_allocation_scope(mesh):
        for n, _ in MM_CTX + MM_BLOCK:
            DUMMY[n] = ttnn.allocate_tensor_on_device(SPECS[n], mesh)
    log("matmul units", len(MM_CTX), "ctx +", len(MM_BLOCK), "block; dummies", len(DUMMY))


S.mtp_prefill(be, sp, io, prompt, plain, pt[0], B)
sp.enter(pt[0])
ttnn.synchronize_device(mesh)
STATE = [t for t in [sp.pos_v, getattr(sp, "mpos_cu", None), sp.sel, sp.rec, sp.tin, be.pos]
         + list(getattr(sp, "mpos_ch", None) or []) if t is not None]  # fmt: skip
saved = [(t, read_rows(mesh, t).clone(), t.dtype) for t in STATE]


def restore():
    ttnn.synchronize_device(mesh)
    for t, v, dt in saved:
        write_rows(mesh, t, v, dt)


def timed(tr) -> float:
    us = []
    for _ in range(max(1, REPS // PR)):
        restore()
        ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)  # (one warm replay per round)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(PR):
            ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        us.append((time.perf_counter() - t0) / PR * 1e6)
    return sorted(us)[len(us) // 2]


MK = df2.mk is not None
pieces = [("verify", sp._verify), ("accept", lambda: sp._ctl_accept(False)), ("df2_prep", df2.prep_program),
          ("df2_ctx", df2.ctx_program), ("df2_block", df2.block_program)]  # fmt: skip
if (
    MK
):  # rung 3: the block's parts alone (every tensor persistent: the program kinds, the SDPA launches, the head chain)
    mk, st, Ls = df2.mk, df2.st, df2.L

    def mk_emb():
        e = df2.ops.all_gather(df2.ops.embedding(st.ids, df2.embed), 2)
        return df2.ops.reshape(e, (1, 1, 32, df2.d.H))

    ATT = [None]

    def mk_att_dummy():
        from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

        if ATT[0] is None:
            with corruptible_allocation_scope(mesh):
                ATT[0] = ttnn.allocate_tensor_on_device(mk.attention(Ls[0], df2.cur_g, df2.scale).spec, mesh)
        return ATT[0]

    pieces += [("mk_emb", mk_emb),
               ("mk_a0", lambda: mk.layer_a0(Ls[0], mk.res, df2.rope_tiles[0], df2.rope_tiles[1], st.bpos, df2.KC)),
               ("mk_sdpa", lambda: [mk.attention(Ly, df2.cur_g, df2.scale) for Ly in Ls]),
               ("mk_ba_one", lambda: mk.layer_ba(0, Ls[0], Ls[1], mk_att_dummy(), mk.res, df2.rope_tiles[0], df2.rope_tiles[1], st.bpos, df2.KC)),
               ("mk_ba_all", lambda: [mk.layer_ba(i, Ls[i], Ls[i + 1] if i + 1 < len(Ls) else None, mk_att_dummy(), mk.res,
                                                  df2.rope_tiles[0], df2.rope_tiles[1], st.bpos, df2.KC) for i in range(len(Ls))]),
               ("mk_head", lambda: df2._block_head(mk.hid, st, df2.K)),
               ("mk_head_mm", lambda: df2._head_mm(mk.hid)),
               ("mk_head_topk", lambda: df2._head_topk(HD[0])),
               ("mk_head_sel", lambda: df2._head_sel(HD[1], HD[2], HD[3], st, df2.K, mk.hid))]  # fmt: skip
    HD = [None, None, None, None]  # persistent dummies with the real specs: logits, vals, ids, hp (corruptible scope)

    def mk_head_dummies():
        from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

        if HD[0] is None:
            lg, hp = df2._head_mm(mk.hid)
            va, ii = df2._head_topk(lg)
            with corruptible_allocation_scope(mesh):
                HD[0], HD[3] = ttnn.allocate_tensor_on_device(lg.spec, mesh), ttnn.allocate_tensor_on_device(
                    hp.spec, mesh
                )
                HD[1], HD[2] = ttnn.allocate_tensor_on_device(va.spec, mesh), ttnn.allocate_tensor_on_device(
                    ii.spec, mesh
                )

else:
    pieces += [("mm_ctx", lambda: mm_run(MM_CTX)), ("mm_block", lambda: mm_run(MM_BLOCK))]
if OPS_BENCH and not MK:
    pieces += [(f"op_{k}", (lambda k=k: op_run(k))) for k in OP_KINDS if hasattr(df2.ops, k)]
res, tids = {}, []
try:
    for name, fn in pieces:
        if name == "mm_ctx":
            make_dummies()
            REC[0] = False
        if name == "mk_emb":
            REC[0] = False
        if name == "mk_ba_one":
            mk_att_dummy()
        if name == "mk_head_mm":
            mk_head_dummies()
        if name == "df2_ctx" and not MK:
            REC[0] = True  # record the ctx + block programs' stock-op calls (the one capture of each)
        if name.startswith("op_"):
            if not CALLS[name[3:]]:
                continue
            if not LIVE:
                make_op_dummies()
        restore()
        if C.MOCK:  # (the mock cluster cannot capture a trace: each piece runs eagerly once)
            fn()
            res[name] = 0.0
            continue
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tids.append(tid)
        res[name] = round(timed(tid), 1)
        log(name, res[name], "us")
    if not C.MOCK:
        tr = sp.capture()
        res["step"], res["boot"] = round(timed(tr["step"]), 1), round(timed(tr["boot"]), 1)
        res["plain"] = round(timed(plain), 1)
    else:
        res.update(step=0.0, boot=0.0, plain=0.0)
finally:
    for tid in tids:
        ttnn.release_trace(mesh, tid)
    restore()
drafter = res["df2_prep"] + res["df2_ctx"] + res["df2_block"]
parts = res["verify"] + res["accept"] + drafter
if MK:
    res.setdefault("mm_ctx", 0.0)
    res.setdefault("mm_block", 0.0)
derived = dict(ctx_other=round(res["df2_ctx"] - res["mm_ctx"], 1), block_other=round(res["df2_block"] - res["mm_block"], 1),
               drafter=round(drafter, 1), step_minus_plain=round(res["step"] - res["plain"], 1),
               residual=round(res["step"] - parts, 1),
               ops_sum=round(sum(v for k, v in res.items() if k.startswith("op_")), 1),
               mk_parts_sum=round(sum(res.get(k, 0.0) for k in ("mk_emb", "mk_a0", "mk_sdpa", "mk_ba_all", "mk_head")), 1) if MK else None,
               mk_head_sum=round(sum(res.get(k, 0.0) for k in ("mk_head_mm", "mk_head_topk", "mk_head_sel")), 1) if MK else None,
               topk=os.environ.get("GLM53_DF2_TOPK", "ttnn"),
               op_calls={k: len(v) for k, v in CALLS.items() if v})  # fmt: skip
census = dict(getattr(df2.ops, "census", {}) or {})  # (every launch the kind issued in this process, by op)
S.record("perf", dict(test=f"dflash2-breakdown-k{K}", milestone="DF2", status="info", K=K, B=B, layers=_L, tp=be.tp, layer=df2.layer_mode,
                      reps=REPS, per_round=PR, us=res, derived=derived, mm_ctx_n=len(MM_CTX), mm_block_n=len(MM_BLOCK),
                      conv=XENV.get("GLM53_DF2_CONV", "fused"), mm=XENV.get("GLM53_DF2_MM", "stream"),
                      mm_plans=getattr(df2.ops, "mm_plans", None), census=census, plain_width=w0, env=XENV,
                      tag=args.get("tag")))  # fmt: skip
log(res, derived)
sp.leave()
RESULT = dict(us=res, derived=derived)
