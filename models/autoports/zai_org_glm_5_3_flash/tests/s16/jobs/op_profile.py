# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 op-level profile of one prefill layer WITHOUT Tracy (traced replays only): the
layer runs once eagerly under a recorder that intercepts every ttnn op call (args kept alive: ttnn.deallocate is a no-op
while recording); then every recorded call is captured ALONE, K times in one trace, and replayed (blocking, median of
reps): µs per op = replay / K. Ops are grouped by the layer phase they belong to (the recorder's current label, set by
the wrappers of TPPrefill's phase methods) and by op name. Rows: perf.jsonl (status info) + a top-N table in the log.

ARGS: layers ([0, 1, 2, 3, 4]), check ([4]), n (2048), k (8 calls per trace), reps (3), top (25), moe ("ep"), mhc
("comp"), dsa (False), tag, record."""
import collections
import gc
import os
import sys

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
CHECK = [int(v) for v in args.get("check", [4])]
N = int(args.get("n", 2048))
K = int(args.get("k", 8))
REPS = int(args.get("reps", 3))
TOP = int(args.get("top", 25))
tag = args.get("tag", "op-profile")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = args.get("mhc", "comp")
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "ops")
os.environ["GLM53_PF_DSA"] = "1" if args.get("dsa", False) else "0"
S.set_attn(args)
S.adopt_full(ctx, LAYERS, TPN)  # p2_full's backend in this process, if any (no second full-model build)
ent = S.build(mesh, ctx, LAYERS, b=1, buckets=(N,), max_seq=int(args.get("max_seq", N + 512)))
pf = ent["pf"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", False)))
pf.kda_scan = args.get("kda_scan", pf.kda_scan)
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)

# ---------------------------------------------------------------- recorder
SKIP = {"deallocate", "to_torch", "from_torch", "get_device_tensors", "synchronize_device", "begin_trace_capture",
        "end_trace_capture", "execute_trace", "release_trace", "copy_host_to_device_tensor", "reshape", "Tensor",
        "CoreCoord", "CoreRange", "CoreRangeSet", "MemoryConfig", "ShardTensorToMesh", "ReplicateTensorToMesh",
        "init_device_compute_kernel_config", "get_num_devices"}  # fmt: skip
calls = []
label = ["-"]


def _wrap(mod, name, qual):
    fn = getattr(mod, name)

    def w(*a, **kw):
        calls.append((label[0], qual, fn, a, kw))
        return fn(*a, **kw)

    return fn, w


def _targets():
    out = []
    for mod, prefix in ((ttnn, "ttnn"), (ttnn.experimental, "ttnn.experimental"), (ttnn.transformer, "ttnn.transformer"),
                        (ttnn.experimental.kda, "ttnn.experimental.kda"),
                        (ttnn.experimental.deepseek_prefill, "ttnn.experimental.deepseek_prefill")):  # fmt: skip
        for name in dir(mod):
            if name.startswith("_") or name in SKIP:
                continue
            obj = getattr(mod, name, None)
            if callable(obj) and not isinstance(obj, type) and type(obj).__name__ != "module":
                out.append((mod, name, f"{prefix}.{name}"))
    return out


# the innermost phase method on the stack labels an op; unlabeled ops ("-") = mHC pre / post, norms, the stream concat
PHASES = ("gather", "reduce", "attention", "ffn")


def run_recorded(li, X):
    patched = []
    for mod, name, qual in _targets():
        try:
            orig, w = _wrap(mod, name, qual)
            setattr(mod, name, w)
            patched.append((mod, name, orig))
        except (AttributeError, TypeError):
            continue
    real_dealloc = ttnn.deallocate
    ttnn.deallocate = lambda *a, **k: None
    meth = {}
    for m in PHASES:
        orig = getattr(pf, m)
        meth[m] = orig

        def mk(o, ph):
            def f(*a, **k):
                prev = label[0]
                label[0] = ph
                try:
                    return o(*a, **k)
                finally:
                    label[0] = prev

            return f

        setattr(pf, m, mk(orig, m))
    pf._fresh_now = pf.mla_form == "mha"  # a chunk at position 0: the MHA-form path when it is enabled (as forward)
    try:
        y = pf.layer(li, X, ci, sp, keep_input=True)
    finally:
        pf._fresh_now = False
        for mod, name, orig in patched:
            setattr(mod, name, orig)
        ttnn.deallocate = real_dealloc
        for m in meth:
            delattr(pf, m)
    return y


def _addr(t_):
    try:
        return t_.buffer_address() if t_.is_allocated() else None
    except Exception:  # noqa: BLE001
        return None


def _free(out, keep=()):
    """free an op's output(s) unless it aliases an input (in-place ops and views return their input's buffer)"""
    if isinstance(out, ttnn.Tensor):
        a_ = _addr(out)
        if a_ is not None and a_ not in keep:
            try:
                ttnn.deallocate(out)
            except Exception:  # noqa: BLE001
                pass
    elif isinstance(out, (list, tuple)):
        for o in out:
            _free(o, keep)


# ---------------------------------------------------------------- per-phase model (operator 2026-10-22 02:48Z)
# phases: attention / moe_route_combine / expert_ffn / glue_relayout / collectives / cache_writes / mhc_norm. DRAM-busy
# = the op's DRAM bytes (DRAM-resident inputs read once + outputs written once; views 0) at DRAM_GBPS; compute-busy =
# the op's matmul / SDPA / xffn FLOPs at the grid's HiFi2 peak (tech_reports/GEMM_FLOPS: 2.7 TFLOPS per Tensix at
# 1.35 GHz). Both are lower bounds of the busy time (ideal rates): the gap to the span is the latency / overhead share.
DRAM_GBPS = 512.0
_grid = mesh.compute_with_storage_grid_size()
PEAK_TFLOPS = 2.7 * int(_grid.x) * int(_grid.y)
_ES = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0, "BFLOAT8_B": 1088 / 1024,
       "BFLOAT4_B": 576 / 1024}  # fmt: skip
GLUE = ("to_layout", "tilize", "untilize", "pad", "reshape", "typecast", "transpose", "permute", "concat", "slice", "copy",
        "clone", "fill", "zeros", "ones", "empty", "repeat", "interleaved_to_sharded", "sharded_to_interleaved",
        "to_memory_config", "move", "tilize_with_val_padding", "untilize_with_unpadding", "view", "squeeze",
        "unsqueeze", "nlp_create_qkv_heads", "nlp_concat_heads", "experimental.view")  # fmt: skip


def _tensors(a, kw):
    out = [t_ for t_ in list(a) + list(kw.values()) if isinstance(t_, ttnn.Tensor)]
    out += [t_ for v in list(a) + list(kw.values()) if isinstance(v, (list, tuple)) for t_ in v
            if isinstance(t_, ttnn.Tensor)]  # fmt: skip
    return out


def _dram_bytes(t_):
    try:
        if t_.memory_config().buffer_type != ttnn.BufferType.DRAM:
            return 0.0
        n = 1
        for d in t_.padded_shape:
            n *= int(d)
        return n * _ES.get(str(t_.dtype).split(".")[-1], 2.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _kernels(a):
    for v in a:
        if hasattr(v, "kernels"):
            try:
                return " ".join(str(k.kernel_source) for k in v.kernels)
            except Exception:  # noqa: BLE001
                return ""
    return ""


def _flops(name, a, kw):
    try:
        if name in ("matmul", "linear") and len(a) >= 2:
            sa, sb = [int(v) for v in a[0].shape], [int(v) for v in a[1].shape]
            batch = 1
            for d in sa[:-2]:
                batch *= d
            return 2.0 * batch * sa[-2] * sa[-1] * sb[-1]
        if name == "scaled_dot_product_attention":
            sq, sk = [int(v) for v in a[0].shape], [int(v) for v in a[1].shape]
            f = 4.0 * sq[0] * sq[1] * sq[2] * sk[2] * sq[3]
            return f / 2 if kw.get("is_causal", True) and kw.get("attn_mask") is None and sq[2] == sk[2] else f
        if name == "moe_fused_swiglu":  # (x, gate_up[], gate_up[], down[], ...; output=[1, 1, CAP, H])
            cap, h = [int(v) for v in kw["output"].shape][-2:]
            two_w = int(a[1][0].shape[-1])
            return 2.0 * cap * (h * two_w + (two_w // 2) * h)
        if name == "generic_op" and "moe_xffn" in _kernels(a):
            io = a[0]  # [x_ext, disp, cnt, start, tbl, out_rm, gate_up..., down...]
            cap, h = [int(v) for v in io[5].shape][-2:]
            two_w = int(io[6].shape[-1])
            return 2.0 * cap * (h * two_w + (two_w // 2) * h)
    except Exception:  # noqa: BLE001
        return 0.0
    return 0.0


def classify(ph, name, a):
    if any(w in name for w in ("all_gather", "reduce_scatter", "all_reduce", "all_to_all")):
        return "collectives"
    if name == "generic_op":
        ks = _kernels(a)
        if "moe_xffn" in ks:
            return "expert_ffn"
        if any(w in ks for w in ("moe_plan", "moe_combine", "moe_dispatch", "rows_to")):
            return "moe_route_combine"
        if "mhc_post_pf" in ks:
            return "mhc_norm"
        if "state_io" in ks or "cache" in ks:
            return "cache_writes"
    if "cache" in name:
        return "cache_writes"
    if name in GLUE or name.split(".")[-1] in GLUE:
        return "glue_relayout"
    if ph == "attention":
        return "attention"
    if ph == "ffn":
        if name in ("matmul", "linear", "moe_fused_swiglu", "clamped_silu_glu") or "swiglu" in name:
            return "expert_ffn"
        return "moe_route_combine"
    if ph in ("gather", "reduce"):
        return "collectives" if any(w in name for w in ("gather", "scatter", "reduce")) else "glue_relayout"
    return "mhc_norm"  # unlabeled: mHC pre / post mixes and coefficients, RMSNorms, the residual math


results = []
for L in CHECK:
    li = LAYERS.index(L)
    pl = pf.layers[li]
    x_in, _ = S.golden_layer(L, N)
    pad = torch.zeros(c, 4, 4096)
    pad[:N] = x_in.reshape(N, 4, 4096).float()
    X = pf.streams_to_device(pad, sp)
    calls.clear()
    y = run_recorded(li, X)
    ttnn.synchronize_device(mesh)
    S.log(f"L{L} {pl.kind}: {len(calls)} recorded op calls")
    per = []
    for i, (ph, qual, fn, a, kw) in enumerate(calls):
        us = None
        ins = [t_ for t_ in list(a) + list(kw.values()) if isinstance(t_, ttnn.Tensor)]
        ins += [t_ for v in list(a) + list(kw.values()) if isinstance(v, (list, tuple)) for t_ in v
                if isinstance(t_, ttnn.Tensor)]  # fmt: skip
        live = all(t_.is_allocated() for t_ in ins)
        keep = {_addr(t_) for t_ in ins}
        nbytes = sum(_dram_bytes(t_) for t_ in ins) if live else 0.0  # + the outputs at the warm launch
        if not C.MOCK and not live:
            S.log(f"  op {i} {qual} not timed: an input was freed outside the recorder")
        elif not C.MOCK:
            try:  # warm launch outside any trace: an op that cannot run alone fails HERE, never inside a capture
                out = fn(*a, **kw)
                ttnn.synchronize_device(mesh)
                outs = [out] if isinstance(out, ttnn.Tensor) else [o for o in (out or []) if isinstance(o, ttnn.Tensor)] \
                    if isinstance(out, (list, tuple)) else []  # fmt: skip
                nbytes += sum(_dram_bytes(o) for o in outs if _addr(o) not in keep)
                _free(out, keep)
            except Exception as e:  # noqa: BLE001
                S.log(f"  op {i} {qual} not timed: {str(e)[:160]}")
            else:
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                ok = False
                try:
                    for _ in range(K):
                        _free(fn(*a, **kw), keep)
                    ok = True
                except Exception as e:  # noqa: BLE001
                    S.log(f"  op {i} {qual} not timed (capture): {str(e)[:160]}")
                finally:
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    if not ok:
                        ttnn.release_trace(mesh, tid)
                if not ok:
                    per.append(dict(i=i, phase=ph, op=qual.split(".")[-1], us=None))
                    continue
                try:  # >= 2,000 steady-state replays of the K calls (program 03:20Z device duty)
                    us = round(S.replay_stats(mesh, tid, replays=args.get("replays"))["median_us"] / K, 1)
                finally:
                    ttnn.release_trace(mesh, tid)
        name = qual.split(".")[-1]
        per.append(dict(i=i, phase=ph, op=name, us=us, cls=classify(ph, name, a), mb=round(nbytes / 2**20, 3),
                        gflop=round(_flops(name, a, kw) / 1e9, 3)))  # fmt: skip
    timed = [p for p in per if p["us"] is not None or C.MOCK]  # (the mock: the classification / bytes, no times)
    for p in timed:
        p["us"] = p["us"] or 0.0
    model = collections.defaultdict(lambda: dict(span_us=0.0, mb=0.0, gflop=0.0, ops=0))
    for p in timed:
        m_ = model[p["cls"]]
        m_["span_us"] += p["us"]
        m_["mb"] += p["mb"]
        m_["gflop"] += p["gflop"]
        m_["ops"] += 1
    phases = {}
    for k_, m_ in sorted(model.items(), key=lambda kv: -kv[1]["span_us"]):
        dram_us = m_["mb"] * 2**20 / (DRAM_GBPS * 1e9) * 1e6
        comp_us = m_["gflop"] * 1e9 / (PEAK_TFLOPS * 1e12) * 1e6
        phases[k_] = dict(span_us=round(m_["span_us"], 1), ops=m_["ops"], dram_mb=round(m_["mb"], 1),
                          dram_busy_us=round(dram_us, 1), dram_frac=round(dram_us / max(m_["span_us"], 1e-9), 3),
                          gflop=round(m_["gflop"], 2), compute_busy_us=round(comp_us, 1),
                          compute_frac=round(comp_us / max(m_["span_us"], 1e-9), 3))  # fmt: skip
        S.log(f"  phase {k_:18s} span {m_['span_us']:9.1f} us  ops {m_['ops']:3d}  DRAM {m_['mb']:8.1f} MB -> "
              f"{dram_us:8.1f} us busy  compute {m_['gflop']:8.2f} GFLOP -> {comp_us:7.1f} us busy")  # fmt: skip
    by_phase = collections.defaultdict(float)
    by_op = collections.defaultdict(lambda: [0, 0.0])
    for p in timed:
        by_phase[p["phase"]] += p["us"]
        by_op[p["op"]][0] += 1
        by_op[p["op"]][1] += p["us"]
    top = sorted(timed, key=lambda p: -p["us"])[:TOP]
    for p in top:
        S.log(f"  {p['us']:9.1f} us  #{p['i']:3d} {p['phase']:10s} {p['op']}")
    row = dict(test=f"op profile L{L} {pl.kind} (each op traced alone x{K}, n={N}) tp{TPN}", milestone="P2",
               status="info", traced=True, layer=L, kind=pl.kind, n=N, tp=TPN, ops=len(per), timed=len(timed),
               sum_us=round(sum(p["us"] for p in timed), 1),
               by_phase={k: round(v, 1) for k, v in sorted(by_phase.items(), key=lambda kv: -kv[1])},
               by_op={k: [v[0], round(v[1], 1)] for k, v in sorted(by_op.items(), key=lambda kv: -kv[1][1])},
               top=[[p["i"], p["phase"], p["op"], p["us"]] for p in top], moe=pf.moe_impl, mhc=pf.mhc_mode,
               phases=phases, model=dict(dram_gbps=DRAM_GBPS, peak_tflops=round(PEAK_TFLOPS, 1),
                                         note="busy = bytes / peak BW, FLOPs / HiFi2 grid peak: ideal-rate lower bounds; moe_fused_swiglu "
                                              "(EP layers) FLOPs over its CAP rows: an upper bound"),
               ops_detail=[[p["i"], p["cls"], p["op"], p["us"], p["mb"], p["gflop"]] for p in per],
               tag=tag)  # fmt: skip
    S.log({k: v for k, v in row.items() if k != "top"})
    if rec and not C.MOCK:
        C.record("perf", row)
    results.append(row)
    calls.clear()
    gc.collect()
RESULT = results
