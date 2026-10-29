# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the d0 pair or the 2x4 mesh): W1's SERVING entry StackBackend.prefill_group (design/w1_grpmax_design.md
§8) vs the per-chunk StackBackend.prefill_chunk on the same 4-chunk request, through the backend's own flow (cur_pos,
sampler rows, reads / restores, the KDA state I/O, the head on the last valid row):
  P   4 x prefill_chunk (each chunk its key's captured trace; the last with the head);
  G4  ONE prefill_group of 4 chunks (head);
  G22 prefill_group of chunks 0-1 HEAD-LESS + prefill_group of chunks 2-3 (head): the (windows, head) keys.
Every program compiles before the first capture (TPPrefill.compile + each group key's eager forward + the backend's
host-row writes and state export), then every trace is captured (the groups' through GroupStack.capture, as
GroupStack.prepare does). Gate: G4 and G22 vs P BITWISE on the last row's logits and on every tensor of the slot's
exported checkpoint after the request (each KDA layer's state; with the export staging, the KDA pack and the DSA blob),
and no program compiled after the captures (post_capture_compiles 0). The chain is the served PMK stack of the job's env (the op arm: GLM53_PMK_SLOTS=none + GLM53_PF_GROUP_OP_FFN=1; the
hybrid: the attention slots + the op FFN grouped; the PMK arm: all slots). ARGS: layers ([0..4]), windows ("65536"), b
(2), tag, record."""
import gc
import os
import sys
import time
import traceback

import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = 2048
NC = 4
WINDOWS = str(args.get("windows", "65536"))
tag = args.get("tag", f"prefill-group-unit-tp{TPN}")
rec = bool(args.get("record", True))
import pmk45_alias as A45  # noqa: E402

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", os.environ.get("GLM53_PMK_SLOTS", "auto")))
os.environ["GLM53_PF_SPARSE_T"] = WINDOWS
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
t0 = time.time()
info, row = {}, {}
try:
    ent = S.build(mesh, ctx, LAYERS, b=int(args.get("b", 2)), buckets=(N,), max_seq=max(int(w) for w in WINDOWS.split(",")) + 512)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    cnt0 = {k: v for k, v in backend.counters.items() if isinstance(v, (int, float))}  # this job's deltas only
    info["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=int(args.get("b", 2)))
    c = pf.bucket_for(N)
    assert c == N and pf.mode(c) == "sp", (c, pf.mode(c))
    os.environ["GLM53_PF_MODE"] = "pmk"
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
    opts.update({k: v for k, v in args.items() if k.startswith("b_") or k.startswith("group_")})
    stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    stack.attach()
    backend.pmk = stack  # (prefill_group_obj drops groups built over an earlier job's chain)
    assert backend.prefill_group_ok(NC) and backend.prefill_group_ok(2), "prefill_group needs the served chain"
    g32 = S.golden_prompt(32768).reshape(-1).long()
    toks = g32[: NC * N].clone()
    g4, g2 = backend.prefill_group_obj(NC), backend.prefill_group_obj(2)
    keys = {g4: [(g4.windows(0, [N] * NC), True)],
            g2: [(g2.windows(0, [N] * 2), False), (g2.windows(2 * N, [N] * 2), True)]}  # fmt: skip
    info["group_keys"] = {str(g.n): [[list(ts), h] for ts, h in ks] for g, ks in keys.items()}

    def pc():
        try:
            return int(mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001
            return -1

    def state():
        """every tensor of the slot's exported checkpoint, by path: the per-layer {L, S, C} records, or (a cache with
        the export staging, e.g. GLM53_PF_KDA_SNAPSHOT) the {kda: pack} record, and the {dsa: blob} record"""
        flat = {}

        def walk(v, path):
            if isinstance(v, torch.Tensor):
                flat[path] = v.float().clone()
            elif isinstance(v, dict):
                for k_ in sorted(v, key=str):
                    walk(v[k_], f"{path}.{k_}" if path else str(k_))
            elif isinstance(v, (list, tuple)):
                for i_, x_ in enumerate(v):
                    walk(x_, f"{path}[{i_}]")

        for d in backend.export_kda_state(kv, 0):
            walk({"S": d["S"], "C": d["C"]} if "L" in d else d, f"L{int(d['L'])}" if "L" in d else "")
        return flat

    # ---- every program before the first capture: the per-chunk keys, then each group key's eager forward
    # use_prewarm: the SERVING path instead -- GLM53_PF_GROUP_PREWARM through StackBackend._compile_groups (after
    # tpp.compile, as prepare_prefill) and _capture_groups (after tpp.capture, as the capture sites)
    PREWARM = bool(args.get("use_prewarm", False))
    t1 = time.time()
    pf.compile((c,))
    if PREWARM:
        os.environ["GLM53_PF_GROUP_MAX"] = str(NC)
        os.environ["GLM53_PF_GROUP_PREWARM"] = f"{NC}@0,2@0,2@{2 * N}"
        backend._compile_groups()
        info["prewarm_keys"] = [[n_, s_, list(ts_), h_] for n_, s_, ts_, h_ in backend._group_keys]
    else:
        for g, ks in keys.items():
            for ts, h in ks:
                g.write(toks[: g.n * N], slot=0, chunk_start=0, n_valid=[N] * g.n)
                g.forward(tuple(ts), head=h)
    # the backend's own programs too (pgu8 2026-10-25: StackBackend.write(io.cur_pos)'s mask copies, first compiled
    # AFTER the captures, left two live program-cache buffers and the trace allocation tracker refused the first replay):
    # the host-row writes and the state export eagerly once, as the served warmup does
    backend.write(io.tokens, backend.read(io.tokens))
    backend.write(io.sampled, backend.read(io.sampled))
    backend.write(io.cur_pos, backend.read(io.cur_pos))
    state()
    ttnn.synchronize_device(mesh)
    info["compile_s"] = round(time.time() - t1, 1)
    n0 = pc()
    pf.capture((c,))
    if PREWARM:
        backend._capture_groups()
    else:
        for g, ks in keys.items():
            for ts, h in ks:
                g.capture(ts, head=h)
    info["capture_cache_growth"] = pc() - n0
    n1 = pc()  # nothing may compile from here on (a program first compiled after a capture leaks a live buffer)
    H.log(f"{tag}: compiled {info['compile_s']} s; captures added {info['capture_cache_growth']} programs; keys {info['group_keys']}")

    def form(name):
        ta = time.perf_counter()
        if name == "P":
            for k in range(NC):
                out = backend.prefill_chunk(toks[k * N : (k + 1) * N], kv_cache=kv, slot=0, chunk_start=k * N, n_valid=N,
                                            logits_rows="last" if k == NC - 1 else "none", io=io)  # fmt: skip
        elif name == "G4":
            out = backend.prefill_group(toks, kv_cache=kv, slot=0, chunk_start=0, n_valid=[N] * NC, logits_rows="last", io=io)
        else:
            o1 = backend.prefill_group(toks[: 2 * N], kv_cache=kv, slot=0, chunk_start=0, n_valid=[N, N], logits_rows="none",
                                       io=io)  # fmt: skip
            assert o1 is None, o1
            out = backend.prefill_group(toks[2 * N :], kv_cache=kv, slot=0, chunk_start=2 * N, n_valid=[N, N],
                                        logits_rows="last", io=io)  # fmt: skip
        assert out is not NotImplemented, f"{name}: prefill_group not served"
        us = (time.perf_counter() - ta) * 1e6
        ttnn.synchronize_device(mesh)
        return out.float().clone(), state(), us

    res = {}
    forms = {}
    for rep in range(int(args.get("reps", 2))):
        for name in ("P", "G4", "G22"):
            lg, st, us = form(name)
            forms[name] = (lg, st)
            res.setdefault(name, []).append(round(us, 1))
            if not C.MOCK:  # the TP8 rule (quiesce after replays, before eager device work); the mock's reset segfaults
                mesh.quiesce_devices()
    info["post_capture_compiles"] = pc() - n1
    lp, sp_ = forms["P"]
    cmp = {}
    for name in ("G4", "G22"):
        lg, st = forms[name]
        assert sorted(st) == sorted(sp_), (name, sorted(st), sorted(sp_))
        d = dict(logits=H.cmp(lg, lp), state={k_: H.cmp(st[k_], sp_[k_]).get("bitwise") for k_ in sorted(sp_)})
        d["bitwise"] = bool(d["logits"].get("bitwise")) and bool(d["state"]) and all(d["state"].values())
        cmp[name] = d
    bitwise = all(v["bitwise"] for v in cmp.values())
    status = "mock" if C.MOCK else ("pass" if bitwise and info["post_capture_compiles"] == 0 else "fail")
    row = dict(test=f"W1 serving: StackBackend.prefill_group (G4, G22 with a head-less group) vs 4 x prefill_chunk, L{LAYERS[0]}-L{LAYERS[-1]} tp{TPN}",
               milestone="W1-serving", status=status, tp=TPN, layers=LAYERS, bitwise=bitwise, compare=cmp,
               host_us_per_form=res, counters={k: v - cnt0.get(k, 0) for k, v in backend.counters.items()
                                               if "prefill" in str(k) and isinstance(v, (int, float))},
               group_keys=info["group_keys"], compile_s=info["compile_s"], capture_cache_growth=info["capture_cache_growth"],
               post_capture_compiles=info["post_capture_compiles"],
               slots=os.environ.get("GLM53_PMK_SLOTS"), group_op_ffn=os.environ.get("GLM53_PF_GROUP_OP_FFN", ""),
               use_prewarm=PREWARM, prewarm_keys=info.get("prewarm_keys"),
               wall_s=round(time.time() - t0, 1), tag=tag)  # fmt: skip
    H.log(f"{tag}: status {status} bitwise {bitwise}; {cmp}; host us {res}")
    if rec and not C.MOCK:
        C.record("pcc", row)
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    row = dict(test="W1 serving prefill_group", milestone="W1-serving", status="fail", error=f"{type(e).__name__}: {e}"[:2000],
               tp=TPN, layers=LAYERS, tag=tag, info={k: v for k, v in info.items() if k != "page_table"})  # fmt: skip
    if rec and not C.MOCK:
        C.record("pcc", row)
for g_ in list(getattr(globals().get("backend"), "_pf_groups", {}).values()):
    try:
        g_.release()
    except Exception:  # noqa: BLE001
        pass
if "backend" in globals():
    backend.__dict__.pop("_pf_groups", None)
    backend.__dict__.pop("_pf_groups_owner", None)
    backend.pmk = None
if "stack" in globals():
    try:
        stack.release()
    except Exception:  # noqa: BLE001
        pass
gc.collect()
RESULT = row
