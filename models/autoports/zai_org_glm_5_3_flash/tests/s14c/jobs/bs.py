# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c job (serve.py): batch scaling of the TP dense_kda / sparse_kda layer program (Stage 14's TPDenseKDALayer /
TPSparseKDALayer on 7a's MkLayer / MkSparseLayer, ONE program per chip with the 2 in-program fabric all-reduces) at
B decode rows in ONE launch (7a's native t <= 32; `native=False` = Stage 14's B / 4 launches of 4 rows), device vs
device and vs the Stage 3 single-chip layer + the fp32 goldens (Stage 14's references, work-s14/m3_ref; B=16 = the
first 16 rows of the B=32 reference: decode rows are independent for B >= 8, checked bitwise for B=8 by bs_refcheck.py).

JOB PIPELINE RULE (program 2026-10-19 23:52Z): CTX keeps the layer (weights + programs) per (layer, code rev,
knobs, spreads, vtp), the fabric phase / inputs / traces per B on it; one job = a sweep of configs (`sweep`: a list of
arg overrides; one row per config); rows go straight to perf.jsonl / pcc.jsonl (`record`).

Per config:
  numerics: ONE traced call with the state restored first -> the layer output on every chip (must be identical), PCC
      vs Stage 3 (0.9999) and vs golden (0.995), the KDA state / conv history vs Stage 3 (TP; virtual TP: "info");
      `save=<name>`: every phase output of that call (per chip) -> work-s14c/bs_out/<name>.pt; `cmp=<name>`: the same
      phases vs a saved run (bitwise flag + PCC per phase) — the device-vs-device check of a program change (pass =
      every phase >= 0.9999; virtual TP rows are pass / fail on it, "info" without a cmp; under GLM53_MK_DEFER_INV=1
      hn / hn2 are h * w and excluded);
  timing: a capture WITHOUT the restore replayed `reps` times (state drifts; timing only); `chain=N`: N launches in
      one trace -> steady state per layer (7a's M1 convention); NO eager launch is measured;
  probe: per-phase in-program marks (kernels/mk/probe.hpp) of the numerics call -> the phase table (us, per chip).
ARGS: layer (0), b (1), prompt (128), reps (20), chain (8), tag, probe, rec_spread (1), kda_spread, vtp, native (True),
 save, cmp, record (False), env {knob: value} (reset every config), ref_w (/opt/glm53/work-s14), sweep [{...}],
 build_only (construct the layer and its fabric phase, no launch)."""
import gc
import importlib
import json
import os
import sys
import time

import torch

import ttnn

gc.collect()


def _load(name, fname):
    """This stage's helper module by file path under a unique name: a job may run in another stage's server, whose
    own s14common / probe_table already sit in sys.modules under the plain names."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("s14c_common", "s14common.py")
PT = _load("s14c_probe_table", "probe_table.py")

import models.autoports.zai_org_glm_5_3_flash.tt.ds_mm as DSMM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder as FDEC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder as ODEC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.fabric_ar as FAR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_pre as KPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mhc_post as MPOST  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre as MPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_dense as MKD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_ffn as MKF  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8 as TP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_experts as MKX  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_moe as MKM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.router_tail as RTAIL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse as TPS  # noqa: E402

KNOBS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
         "GLM53_MK_READERS", "GLM53_MK_LR_FIDELITY", "GLM53_MK_DS_FIDELITY", "GLM53_MHC_ROWREAD", "GLM53_MK_DEFER_INV",
         "GLM53_MHC_POST_ROWWRITE", "GLM53_REC_EARLY_SIGNAL", "GLM53_MHC_SYNC_MARKS", "GLM53_REC_DEPTH",
         "GLM53_REC_ITEM_MARKS", "GLM53_REC_UGROUPS", "GLM53_MK_W_DELAY_US", "GLM53_MK_FOLD_Y2", "GLM53_MHC_X_ROWS",
         "GLM53_REC_MODE", "GLM53_KDA_MODE", "GLM53_MHC_MODE", "GLM53_MHC_PRE_EARLY", "GLM53_MK_PARK", "GLM53_CB_REPORT",
         "GLM53_REC_V2_UG", "GLM53_MK_STACKED_MAX",
         # merged Stage 14 / 7a knobs (d97439c8ebb): prefetch, hn-first, TP-shared expert, router / expert phase knobs
         "GLM53_KDA_PREFETCH", "GLM53_MHC_HN_FIRST", "GLM53_TP_SHARED", "GLM53_MK_ROUTER_KSPLIT", "GLM53_MK_ROUTER_FIDELITY",
         "GLM53_MK_KB_ROUTER", "GLM53_MK_KB_SGU", "GLM53_MK_FAR_CREDIT_NEXT", "GLM53_MKX_RING", "GLM53_MKX_INFLIGHT",
         "GLM53_MKX_TPT", "GLM53_MKX_MBUF", "GLM53_MKX_MMBLOCK")  # fmt: skip  (GLM53_REC_V2_UG: Stage 14 tree gate of v2 + user groups)
mesh = MESH  # noqa: F821
ctx = CTX  # noqa: F821
# rows / outputs of THIS stage even when the job runs in another stage's server (Stage 14's 8-chip process)
W = ARGS.get("work") or os.environ.get("GLM53_WORK", "/opt/glm53/work-s14c")  # noqa: F821
C.OUT = ARGS.get("evidence") or C.OUT  # noqa: F821  (C is this job's private copy of s14common)


def _tree_rev() -> str:
    """HEAD + a hash of uncommitted model-dir edits: the S14_REV define (no stale in-process FILE_PATH kernel binary)."""
    import hashlib
    import subprocess

    home = os.environ.get("TT_METAL_HOME", ".")
    md = "models/autoports/zai_org_glm_5_3_flash"
    head = subprocess.run(["git", "-C", home, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", home, "diff", "HEAD", "--", md], capture_output=True, text=True).stdout
    return head + ("_" + hashlib.sha1(diff.encode()).hexdigest()[:8] if diff else "")


REV0 = ARGS.get("rev") or _tree_rev()  # noqa: F821
if ctx.get("14c_bs_rev") != REV0:  # new code: reload the model modules (leaves first), drop everything built from the old
    for mod in (FDEC, ODEC, DSMM, MKR, MPRE, MPOST, KREC, KPRE, RTAIL, MKD, MKF, MKL, MKX, MKM, MKS, FAR, TP, TPL, TPS):
        importlib.reload(mod)
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] in ("14c_tp", "14c_trace", "14c_fab", "14c_inp")]:
        v = ctx.pop(k)
        if type(v).__name__ == "Traced":  # (a previous job's private s14common copy)
            v.release()
    ctx["14c_bs_rev"] = REV0
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config, pcc  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

# the server's tree must carry 7a's native t <= 32 layer code (Stage 14's server: after merging 14c's branch); refuse
# before any device work otherwise (an old MkLayer at t > 4 builds the t <= 4 layouts)
if not hasattr(MKL, "residual_modes") or not hasattr(TPL.TPDenseKDALayer, "_unhooked"):
    raise RuntimeError(f"bs.py needs the 14c model code (native t <= 32, nofab): {MKL.__file__}")
cfg = hf_config()
source = FdTileCacheSource(CACHE_DIR, MANIFEST)
PHASES = ("hn", "qkv", "lr", "on", "a", "s1", "hn2", "gu", "m", "y", "out")  # MkLayer / MkSparseLayer outputs dict


def log(*a):
    print(f"BS {C.now()}", *a, flush=True)


def load_ref(refw, layer, b, prompt):
    """Stage 3 reference rows (cached: the B=32 file once, sliced for 8 <= B < 32)."""
    key = ("14c_ref", layer, b, prompt)
    if key in ctx:
        return ctx[key]
    p = os.path.join(refw, "m3_ref", f"L{layer}_B{b}_P{prompt}.pt")
    p32 = os.path.join(refw, "m3_ref", f"L{layer}_B32_P{prompt}.pt")
    if b >= 8 and b < 32 and os.path.exists(p32):  # rows independent for B >= 8 (bs_refcheck.py: B=8 == B=32[:8])
        r = load_ref(refw, layer, 32, prompt)[0]
        out = dict(r)
        for k in ("x", "y_ref", "golden", "S0", "S1"):
            out[k] = r[k][:b]
        for k in ("C0", "C1"):
            out[k] = r[k][:, :, :b]
        out["positions"] = list(r["positions"])[:b]
        ctx[key] = (out, p32 + f"[:{b}]")
    elif os.path.exists(p):
        for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "14c_ref" and k[1:] != (layer, 32, prompt)]:
            ctx.pop(k)  # host RAM: keep the B=32 file of the layer and the current one
        ctx[key] = (torch.load(p), p)
    elif C.MOCK:  # mock cluster (compile / build check only): synthetic reference
        g = torch.Generator().manual_seed(0)
        z = lambda *sh: torch.randn(*sh, generator=g)  # noqa: E731
        ctx[key] = (dict(x=z(b, 16384), y_ref=z(b, 16384), golden=z(b, 16384), S0=z(b, 64, 128, 128) * 0.1,
                         C0=z(1, 3, b, 24576).to(torch.bfloat16), S1=z(b, 64, 128, 128),
                         C1=z(1, 3, b, 24576).to(torch.bfloat16), positions=[0] * b, prefill_mode="mock"), "mock")  # fmt: skip
    else:
        raise FileNotFoundError(p)
    log("reference", ctx[key][1])
    return ctx[key]


def get_layer(a, env, rev):
    """The resident TP layer for (layer, rev, knobs, spreads, vtp, probe) — weights and host objects built once."""
    layer = int(a.get("layer", 0))
    key = ("14c_tp", layer, a.get("kda_spread"), bool(a.get("probe", False)), int(a.get("rec_spread", 1)), rev,
           tuple(sorted(env.items())), a.get("vtp"))  # fmt: skip
    if key in ctx:
        return key, ctx[key]
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] in ("14c_tp", "14c_trace", "14c_fab", "14c_inp")]:  # one layer (L1/DRAM)
        v = ctx.pop(k)
        if type(v).__name__ == "Traced":  # (a previous job's private s14common copy)
            v.release()
    gc.collect()
    t0 = time.time()
    sparse = cfg.mlp_layer_types[layer] == "sparse"
    b0 = int(a.get("b", 1))
    if sparse:
        mkey = ("14c_moe", layer)
        if mkey not in ctx:
            for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "14c_moe"]:
                ctx.pop(k)
            gc.collect()
            ctx[mkey] = TPS.build_moe(mesh, cfg, source, layer, OptimizedDecoderConfig())
            log(f"MoE of layer {layer} built in {time.time() - t0:.1f}s")
        lay = TPS.TPSparseKDALayer(mesh, cfg, layer, source, OptimizedDecoderConfig(), b=b0, moe=ctx[mkey],
                                   kda_spread=a.get("kda_spread"), rec_spread=int(a.get("rec_spread", 1)), rev=rev,
                                   vtp=a.get("vtp"), probe=bool(a.get("probe", False)))  # fmt: skip
    else:
        lay = TPL.TPDenseKDALayer(mesh, cfg, layer, source, OptimizedDecoderConfig(), b=b0,
                                  probe=bool(a.get("probe", False)), kda_spread=a.get("kda_spread"),
                                  rec_spread=int(a.get("rec_spread", 1)), rev=rev, vtp=a.get("vtp"))  # fmt: skip
    if lay.fabric is not None:
        ctx[("14c_fab", key, b0)] = lay.fabric
    ctx[key] = lay
    log(f"TP layer {key[:5]} built in {time.time() - t0:.1f}s, lanes {lay.lanes}")
    return key, lay


def l1_buffers():
    """The device's live L1 buffers (address, bytes per bank, layout), lowest address first (the static CB regions of
    a program must end below the lowest)."""
    out = []
    for b in ttnn._ttnn.reports.get_buffers(mesh):
        if "L1" in str(b.buffer_type) and int(b.device_id) == int(mesh.get_device_ids()[0]):
            out.append((int(b.address), int(b.max_size_per_bank), str(b.buffer_layout).split(".")[-1]))
    return sorted(out)


# Stage 14c opt-ins and the code they need in the SERVING process's tree (the job also runs in Stage 14's 8-chip server):
# knob value -> (file under tt/, a text only the right version has); missing -> refuse the config before device work
TREE_NEEDS = {
    ("GLM53_REC_MODE", "v2"): ("kernels/kda_rec/kda_rec_v2_sfpu.hpp", "kdav2_spread4"),  # the SFPTRANSP fix 108e51dc35a
    ("GLM53_MHC_MODE", "cp"): ("mhc_pre.py", "MHC_COEF_PLANE"),
    ("GLM53_MHC_MODE", "pv2"): ("kernels/mhc_post/reader.cpp", "MK_POST_Y_FABRIC"),  # + the t > 4 fabric y fix
    ("GLM53_KDA_MODE", "batch"): ("kda_pre.py", "KDA_BATCH"),
    ("GLM53_MK_STACKED_MAX", "1"): ("mk_layer.py", "GLM53_MK_STACKED_MAX"),  # an older tree would ignore it silently
}


def require_tree(env):
    tt_dir = os.path.dirname(KREC.__file__)
    for (k, v), (rel, text) in TREE_NEEDS.items():
        if v in str(env.get(k, "")).split(","):
            path = os.path.join(tt_dir, rel)
            if not os.path.exists(path) or text not in open(path).read():
                raise RuntimeError(f"{k}={env[k]}: the serving tree {tt_dir} lacks it ({rel}: no '{text}')")


def run(a):
    if a.get("l1_report"):
        log("L1 before config", json.dumps(l1_buffers()))
    for k_ in KNOBS:
        os.environ.pop(k_, None)
    env = {k_: str(v_) for k_, v_ in (a.get("env") or {}).items()}
    assert set(env) <= set(KNOBS), env
    require_tree(env)
    os.environ.update(env)
    rev = REV0
    layer, B, prompt = int(a.get("layer", 0)), int(a.get("b", 1)), int(a.get("prompt", 128))
    reps, chain_n = int(a.get("reps", 20)), int(a.get("chain", 8))
    native, probe = bool(a.get("native", True)), bool(a.get("probe", False))
    assert cfg.layer_types[layer] == "linear_attention", f"layer {layer}: {cfg.layer_types[layer]} (KDA kinds only)"
    ref, ref_path = load_ref(a.get("ref_w", "/opt/glm53/work-s14"), layer, B, prompt)
    key, lay = get_layer(a, env, rev)
    if a.get("l1_report"):
        log("L1 after get_layer", json.dumps(l1_buffers()))
    CH = B if native else min(B, 4)
    assert B % CH == 0, B
    NCH = B // CH
    if not lay.nofab:  # the fabric phase is sized by the launch's rows (packed partial rows)
        fk = ("14c_fab", key, CH)
        if fk not in ctx:
            ctx[fk] = FAR.FabricPhase(mesh, CH, lanes=lay.lanes, mode="cb")
            ctx[fk].probe()
        lay.fabric = ctx[fk]
    lay.b = CH
    if a.get("build_only"):
        return dict(build_only=True, key=str(key[:5]))
    sh = lay.shards
    ik = ("14c_inp", key, B, CH, prompt)
    if ik not in ctx:
        act = lay.masks(torch.ones(CH, dtype=torch.bool))
        chunks = []
        for c in range(NCH):
            sl = slice(c * CH, (c + 1) * CH)
            S0c, C0c = ref["S0"][sl], ref["C0"][:, :, sl]
            chunks.append(dict(x=lay.replicated(ref["x"][sl].reshape(1, 1, CH, 16384)), snap=sh.shard_state(S0c, C0c),
                               live=sh.shard_state(S0c, C0c)))  # fmt: skip
        ctx[ik] = (act, chunks)
    act, chunks = ctx[ik]

    def step(restore=True, keep=False):
        outs = []
        for ch in chunks:
            (S0m, C0m), (Sm, Cm) = ch["snap"], ch["live"]
            if restore:
                ttnn.copy(S0m, Sm)
                ttnn.copy(C0m, Cm)
            r = lay(ch["x"], Cm, act, Sm, t=CH)
            if keep and NCH == 1:  # the phase outputs of the one launch (device-vs-device per phase)
                o = r[2]
                outs.append({k: o[k] for k in PHASES if o.get(k) is not None}
                            | {f"kda{i}": t for i, t in enumerate((o.get("kda") or ())[:6])} | {"out": r[1]})  # fmt: skip
            else:
                outs.append({"out": r[1]})  # the DRAM output only: chunk L1 temporaries freed before the next chunk
            del r
        return outs

    tk = ("14c_trace", key, B, CH, prompt)
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "14c_trace" and k[:5] != tk]:
        ctx.pop(k).release()  # another config's traces hold its L1 outputs: L1 of the resident layer stays one config's
    pers = getattr(getattr(lay, "mk", None), "persist", {})
    for k in [k for k in pers if isinstance(k, tuple) and len(k) == 2 and isinstance(k[1], int) and k[1] != CH]:
        del pers[k]  # MkLayer's per-t persistent L1 buffers (o_proj input, m) of other row counts
    gc.collect()
    if tk + ("num",) not in ctx:
        ctx[tk + ("num",)] = C.Traced(ttnn, mesh, lambda: step(True, keep=True))  # compile launch, then the capture
    tr = ctx[tk + ("num",)]
    pc0 = mesh.num_program_cache_entries()
    t_cfg = time.time()
    tr.replay()

    def per_chip(t):
        return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]

    ys_c = [[y.reshape(-1, 16384)[:CH] for y in per_chip(o["out"])] for o in tr.out]
    ys = [torch.cat([yc[d] for yc in ys_c]) for d in range(len(ys_c[0]))]
    st_c = [sh.gather_state(*ch["live"]) for ch in chunks]
    S1, C1 = torch.cat([v[0] for v in st_c]), torch.cat([v[1] for v in st_c], dim=2)
    soak = None
    if a.get("soak_s"):  # one soak per program version: the numerics trace replayed, every output vs the first bitwise
        t_end, n, bad = time.time() + float(a["soak_s"]), 0, 0
        first = [per_chip(o["out"]) for o in tr.out]
        while time.time() < t_end:
            tr.replay()
            n += 1
            bad += int(any(not torch.equal(u, v) for o, f in zip(tr.out, first) for u, v in zip(per_chip(o["out"]), f)))
        soak = dict(seconds=float(a["soak_s"]), replays=n, mismatches=bad)
        log("SOAK", soak)
    phase_cmp = None
    if NCH == 1 and (a.get("save") or a.get("cmp")):
        ph = {k: per_chip(t) for k, t in tr.out[0].items()}
        ph["S1"], ph["C1"] = [S1], [C1.float()]
        od = os.path.join(W, "bs_out")
        os.makedirs(od, exist_ok=True)
        if a.get("save"):
            tmp = os.path.join(od, f".{a['save']}.tmp")
            torch.save(ph, tmp)
            os.replace(tmp, os.path.join(od, f"{a['save']}.pt"))
        if a.get("cmp"):
            base = torch.load(os.path.join(od, f"{a['cmp']}.pt"))
            phase_cmp = {}
            # a residual layout switch (GLM53_MK_STACKED_MAX: s1 stacked [t,1,4,4096] vs flat [1,1,t,16384], both stream-major
            # per row) changes only the shape: same-size outputs are compared in the saved run's shape (a different
            # element order would show as a low PCC, never as a pass)
            shape_as = lambda x_, y_: y_ if x_.shape == y_.shape or x_.numel() != y_.numel() else y_.reshape(x_.shape)
            for k in sorted(set(base) & set(ph)):
                rr = [dict(bitwise=bool(torch.equal(x_, shape_as(x_, y_))), pcc=pcc(shape_as(x_, y_), x_),
                           max_abs=float((x_ - shape_as(x_, y_)).abs().max()) if x_.numel() else 0.0)
                      for x_, y_ in zip(base[k], ph[k])]  # fmt: skip
                phase_cmp[k] = dict(bitwise=all(r["bitwise"] for r in rr), pcc_min=min(r["pcc"] for r in rr),
                                    max_abs=max(r["max_abs"] for r in rr))  # fmt: skip
            log("PHASE_CMP vs", a["cmp"], json.dumps(phase_cmp))
    tables, raws = None, {}
    if getattr(lay, "mk", None) is not None and lay.mk.probe is not None:  # per-phase marks of the numerics call
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import decode_probe

        MHZ = 1350.0
        lanes = lay.fabric.lane_marks() if lay.fabric is not None else {}
        tables = {}
        for d, pt in enumerate(ttnn.get_device_tensors(lay.mk.probe)):
            phr = ttnn.to_torch(pt)
            p = phr.reshape(-1, 16).long() & 0xFFFFFFFF
            if not bool((p > 0).any()):  # mock cluster: nothing written
                continue
            t0 = int(p[p > 0].min())

            def us(v, t0=t0):  # signed 32-bit cycle difference vs the program's earliest mark
                dd = (int(v) - t0) & 0xFFFFFFFF
                return round((dd - (1 << 32) if dd >= (1 << 31) else dd) / MHZ, 1) if v else None

            nk = len(lay.mk.place["k"])
            pr = dict(mk=decode_probe(phr, lay.mk.readers.assign, k_cores=nk, mhz=MHZ),
                      lane_brisc=[[us(v) for v in ln[0]] for ln in lanes.get(d, [])],
                      lane_ncrisc=[[us(v) for v in ln[1]] for ln in lanes.get(d, [])],
                      owner_brisc_all=[[us(v) for v in p[k] if int(v) > 0] for k in range(nk)],
                      owner_ncrisc_all=[[us(v) for v in p[96 + k] if int(v) > 0] for k in range(nk)])  # fmt: skip
            raws[d] = pr
            try:
                if cfg.mlp_layer_types[layer] == "sparse":
                    tables[d] = PT.sparse_rows(pr)
                else:
                    tables[d] = PT.phase_rows(pr, rec_cores=int(lay.attn.kda_rec_spread) * int(sh.h)
                                              * int(getattr(lay.attn, "kda_rec_ugroups", 1)))  # fmt: skip
            except Exception as e:  # noqa: BLE001 -- a probe layout this decoder does not know: keep the raw marks
                tables[d] = dict(error=repr(e), raw=pr)
            log(f"PROBE chip {d}", json.dumps(tables[d]))
    same = all(torch.equal(ys[0], v) for v in ys[1:])
    p_ref, p_gold = pcc(ys[0], ref["y_ref"]), pcc(ys[0], ref["golden"])
    p_state = pcc(S1, ref["S1"]) if not sh.virtual else None
    hist_ok = bool(torch.equal(C1, ref["C1"])) if not sh.virtual else None
    rows_pcc = [pcc(ys[0][i], ref["y_ref"][i]) for i in range(ys[0].shape[0])] if ys[0].shape[0] > 1 else None
    rows = None
    if rows_pcc:
        worst = min(range(len(rows_pcc)), key=lambda i: rows_pcc[i])
        rows = dict(min=rows_pcc[worst], worst_row=worst, below_9999=sum(p < 0.9999 for p in rows_pcc))
    # timing: captures without the restores (state drifts; timing only), resident per config
    if tk + ("t",) not in ctx:
        ctx[tk + ("t",)] = C.Traced(ttnn, mesh, lambda: step(False), warm=False)
    st = C.stats(ctx[tk + ("t",)].replays(reps))
    chain = None
    if chain_n > 1:
        if tk + ("c", chain_n) not in ctx:
            ctx[tk + ("c", chain_n)] = C.Traced(ttnn, mesh, lambda: [step(False) for _ in range(chain_n)], warm=False)
        wc = C.stats(ctx[tk + ("c", chain_n)].replays(reps))
        chain = dict(n=chain_n, median_us=wc["median_us"], steady_per_layer_us=round(
            (wc["median_us"] - st["median_us"]) / (chain_n - 1), 2), mean_per_layer_us=round(wc["median_us"] / chain_n, 2))  # fmt: skip
    try:  # allocations made while a trace is alive (TT_METAL_TRACE_ALLOC_TRACKING=1): must be none at replay time
        from ttnn._ttnn.operations.trace import get_all_unsafe_tracked_ids

        unsafe = len(get_all_unsafe_tracked_ids())
    except Exception as e:  # noqa: BLE001
        unsafe = f"n/a: {e}"
    pc1 = mesh.num_program_cache_entries()
    virtual = bool(sh.virtual)
    kind = "sparse_kda" if cfg.mlp_layer_types[layer] == "sparse" else "dense_kda"
    ok_ref = (p_ref >= 0.9999 and p_gold >= 0.995) if not virtual else None
    # GLM53_MK_DEFER_INV=1 (Stage 14): both mhc_pre instances publish hn = h * w (1/rms deferred to the readers) -> hn /
    # hn2 differ from a non-deferred reference by a per-row scale; the phases they feed (qkv, gu, ...) are compared
    excl = [k for k in ("hn", "hn2") if str(env.get("GLM53_MK_DEFER_INV")) == "1"] if phase_cmp else []
    ok_dvd = all(v["pcc_min"] >= 0.9999 for k, v in phase_cmp.items() if k not in excl) if phase_cmp else None
    cfgs = dict(rec_spread=lay.attn.kda_rec_spread, kda_pre_spread=lay.attn.kda_pre_spread,
                rec_ugroups=getattr(lay.attn, "kda_rec_ugroups", 1), knobs=env, native=native, launches=NCH)  # fmt: skip
    name = (f"{kind} L{layer} B={B} tp{mesh.get_num_devices()}" + (f" vtp{sh.tp}" if virtual else "")
            + (" nofab" if lay.nofab else ""))
    tag = a.get("tag", "14c")
    # timing_only (reason string): a config whose numerics are known-invalid by construction (e.g. a code path with a
    # known defect) -> "info" with the reason, never a test result
    timing_only = a.get("timing_only")
    row = dict(test=f"14c {name} numerics", tag=tag, layer=layer, B=B, kind=kind,
               status=("fail" if soak and soak["mismatches"] else "info" if timing_only else
                       (("info" if ok_dvd is None else "pass" if ok_dvd and same else "fail") if virtual else
                        ("pass" if ok_ref and same and (ok_dvd is not False) else "fail"))),
               timing_only=timing_only, soak=soak,
               replicas_identical=same, pcc_vs_stage3=None if virtual else p_ref,
               pcc_vs_golden=None if virtual else p_gold, stage3_vs_golden=pcc(ref["y_ref"], ref["golden"]),
               rows_pcc_vs_stage3=None if virtual else rows, state_pcc_vs_stage3=p_state, conv_history_bitwise=hist_ok,
               reference=ref_path, phase_cmp_vs=a.get("cmp"), phase_cmp=phase_cmp, phase_cmp_pass=ok_dvd,
               phase_cmp_excluded=excl or None,
               program_cache_stable=pc0 == pc1, unsafe_allocations_live_at_replay=unsafe, rev=rev, **cfgs)  # fmt: skip
    prow = dict(test=f"14c {name} step", tag=tag, status="info", layer=layer, B=B, kind=kind, traced=True,
                us=chain["mean_per_layer_us"] if chain else st["median_us"],
                us_metric="steady_state_chain_mean" if chain else "single_launch_median", single_launch=st,
                # the timing replays do not restore the KDA state: a MoE layer's routed-expert set drifts with it
                timing_routing="uncontrolled (state not restored in timing replays)" if kind == "sparse_kda" else None,
                chain=chain, virtual_tp=sh.tp if virtual else None, tp=mesh.get_num_devices(), probe=probe, rev=rev,
                phase_table=tables.get(0) if tables else None, phase_table_chip1=tables.get(1) if tables else None,
                config_s=round(time.time() - t_cfg, 2), **cfgs)  # fmt: skip
    if a.get("record"):
        C.record("pcc", row)
        if not os.environ.get("TT_METAL_WATCHER"):  # a watcher process's timing is not a performance number
            C.record("perf", {**prow, "test": prow["test"] + (" probe" if probe else "")})
    else:
        with open(os.path.join(W, "dev_rows.jsonl"), "a") as f:
            f.write(json.dumps({"time_utc": C.now(), **row}, default=str) + "\n")
            f.write(json.dumps({"time_utc": C.now(), **prow}, default=str) + "\n")
    log("ROW", json.dumps(row, default=str))
    log("PERF", json.dumps(prow, default=str))
    return dict(B=B, pcc_ref=p_ref, pcc_gold=p_gold, same=same, us=st["median_us"], chain=chain, state=p_state,
                hist=hist_ok, tables=tables, phase_cmp=phase_cmp, config_s=prow["config_s"],
                raw=raws if a.get("raw") else None)  # fmt: skip


BASE = {k: v for k, v in ARGS.items() if k != "sweep"}  # noqa: F821
_env0 = {k_: os.environ.get(k_) for k_ in KNOBS}  # the serving process's knob env, restored after the job (shared servers)
results = []
try:
    for over in ARGS.get("sweep") or [{}]:  # noqa: F821
        results.append(run({**BASE, **over}))
finally:
    for k_, v_ in _env0.items():
        if v_ is None:
            os.environ.pop(k_, None)
        else:
            os.environ[k_] = v_
RESULT = results if len(results) > 1 else results[0]
