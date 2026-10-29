# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, the PMK accuracy blocker, program 2026-10-24 19:56:43Z): the served prefill's chunk 0 on a
model PREFIX (default L0-4: dense_kda L0-2, the first MoE + DSA layer L3, sparse_kda L4) on the PAIR, per MODE scored
against the CPU fp32 reference chain (hf_ref 8be48d885fce, L{L+1}/streams_in, rows 0..2041), with the op path as the
control -- realgate3's chunk-0 per-layer table (14b, jobs_fh/realgate3.py) for one candidate knob per arm.

Construction: M5.serving (B 4 by default, native) with the twin's env file (args env_file, a serve.py env job) minus the
TP image (the pair loads the tile cache), then per mode a TPPrefill on the same backend by backend.prepare_prefill
(GLM53_PF_MODE op / pmk) with in-program probes on every layer, captured, chunk 0 = golden32768[:2048] in ONE replay.
Modes: "op"; "pmk" (LR_EXACT 0); "pmk_lrx" (GLM53_PMK_A_LR_EXACT=1); "pmk_bfid4" (GLM53_PMK_B_FID=HiFi4);
"pmk_lrx_bfid4" (both). Each mode's knobs are set before its build and removed after it.
Row: per mode, per layer pcc_all / row median / p5 / min / last / max_abs / norm ratio vs fp32, and per PMK arm the
per-layer deltas vs op (p5, max_abs). ARGS: env_file, env, layers ([0..4]), modes, b (4), native (True), tag, record."""
import gc
import json
import os
import sys
import time

import torch

import ttnn

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TREE = os.environ["TT_METAL_HOME"]
MD = os.path.join(TREE, "models/autoports/zai_org_glm_5_3_flash")
sys.path.insert(0, os.path.join(MD, "tests/s16"))
sys.path.insert(0, os.path.join(MD, "tests/s14b"))
import s16common as S  # noqa: E402
import m5common as M5  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

C = S.C
cfg = hf_config()
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
MODES = list(args.get("modes", ["op", "pmk", "pmk_lrx", "pmk_bfid4", "pmk_lrx_bfid4"]))
MODE_ENV = {"op": {}, "pmk": {"GLM53_PMK_A_LR_EXACT": "0"}, "pmk_lrx": {"GLM53_PMK_A_LR_EXACT": "1"},
            "pmk_bfid4": {"GLM53_PMK_A_LR_EXACT": "0", "GLM53_PMK_B_FID": "HiFi4"},
            "pmk_lrx_bfid4": {"GLM53_PMK_A_LR_EXACT": "1", "GLM53_PMK_B_FID": "HiFi4"}}  # fmt: skip
B = int(args.get("b", 4))
NATIVE = bool(args.get("native", True))  # False: 4-row-chunk decode programs (fewer L1-resident buffers beside program B at TP2)
CH = 2048
REF_ROWS = 2042  # golden32768[:2048] == goldens/prompts/2048.pt on rows 0..2041 (realgate3)
CPU_REF = os.environ.get("GLM53_CPU_REF", "/opt/glm53/data/hf_ref/8be48d885fce")
tag = args.get("tag", "layer-fp32-pair")
rec = bool(args.get("record", True))

env = {}
if args.get("env_file"):
    env.update((json.load(open(args["env_file"])).get("args") or {}).get("env") or {})
env.update(args.get("env") or {})
for k_ in ("GLM53_WEIGHT_SOURCE", "GLM53_TP_IMAGE_DIR", "GLM53_TP_IMAGE_ROOT", "GLM53_IMAGE_PIN", "GLM53_LMCACHE"):
    env.pop(k_, None)  # the pair loads the tile cache (the TP image is TP8's); no connector here
for k, v in env.items():
    os.environ[str(k)] = str(v)
os.environ["GLM53_PREFILL_BUCKETS"] = str(CH)
os.environ["GLM53_PF_NOHEAD"] = "0"
os.environ["GLM53_PF_MODE"] = "op"
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
os.environ["GLM53_PF_SPARSE_T"] = ""

t0 = time.time()
key = M5.serving(ctx, mesh, SM, G, B=B, ids=LAYERS, cfg=cfg, native=NATIVE, log=S.log)
backend = ctx[key]["backend"]
S.log(f"{tag}: served backend L{LAYERS} B {B} ready in {time.time() - t0:.1f} s")
if getattr(backend, "tpp", None) is not None:
    backend.tpp.release()
_gen = ctx[key].pop("gen", None)  # its own kv / io go with it (realgate3: frees DRAM for the measured TPPrefills)
if _gen is not None:
    try:
        _b0 = _gen._own_binding()
        for _a in ("kv_cache", "io"):
            _t = getattr(_b0, _a, None)
            if _t is not None:
                for _attr in [k_ for k_, v_ in list(vars(backend).items()) if v_ is _t]:
                    setattr(backend, _attr, None)
    except Exception as e:  # noqa: BLE001
        S.log(f"{tag}: generator unlink: {e!r}")
    _b0 = _t = None
    del _gen
gc.collect()
kv = backend.allocate_kv_cache(max_batch=1, max_seq_len=CH + 512)
io = backend.allocate_decode_io(kv)
backend._pt_host = None
prompt = S.golden_prompt(32768).reshape(-1).long()[:CH]


def build(mode: str):
    """realgate3's build(): release the previous TPPrefill (and its probes), clear the program cache, then this mode's
    TPPrefill with probes on every layer, compiled before its capture"""
    if getattr(backend, "pmk", None) is not None:  # realgate3 19:54Z: the previous PMK stack held 3.7 GB/bank
        try:
            backend.pmk.detach()
        except Exception as e:  # noqa: BLE001
            S.log(f"{tag}: pmk detach: {e!r}")
        backend.pmk = None
        gc.collect()
    if getattr(backend, "tpp", None) is not None:
        old_ = backend.tpp
        old_.release()
        backend.tpp = None
        for name_ in ("_probe", "inputs", "pads", "_pads", "sparse"):
            d_ = getattr(old_, name_, None)
            if isinstance(d_, dict):
                if name_ == "_probe":
                    for v_ in list(d_.values()):
                        if isinstance(v_, ttnn.Tensor):
                            ttnn.deallocate(v_)
                d_.clear()
        if getattr(old_, "_row_buf", None) is not None:
            ttnn.deallocate(old_._row_buf)
            old_._row_buf = None
        del old_
    gc.collect()
    try:
        mesh.clear_program_cache()
    except Exception as e:  # noqa: BLE001
        S.log(f"{tag}: clear_program_cache: {e!r}")
    gc.collect()
    os.environ["GLM53_PF_MODE"] = "op" if mode == "op" else "pmk"
    os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(L) for L in LAYERS)
    os.environ["GLM53_PF_SPARSE_T"] = str(CH + 512)
    t1 = time.time()
    backend.prepare_prefill(kv)
    pf = backend.tpp
    pf.probe_layers = set(LAYERS)
    pf.capture()
    ttnn.synchronize_device(mesh)
    gc.collect()  # capture-time temporaries must die before the first replay (realgate3)
    S.log(f"{tag}: {mode} TPPrefill built + captured in {time.time() - t1:.1f} s")
    return pf


REF = {}


def fp32_table(pf, c: int) -> dict:
    """chunk 0, rows 0..2041: probe layer L's output vs the CPU fp32 reference (hf_ref L{L+1}/streams_in)"""
    out = {}
    for L in LAYERS:
        if L not in REF:
            REF[L] = torch.load(os.path.join(CPU_REF, f"L{L + 1:02d}", "streams_in.pt"), map_location="cpu",
                                weights_only=True).reshape(-1, 4 * 4096)[:REF_ROWS].float()  # fmt: skip
        ref = REF[L]
        dev = pf.probe(c, L).reshape(c, -1)[:REF_ROWS].float() if not C.MOCK else ref.clone()
        a_, b_ = ref.double(), dev.double()
        ac, bc = a_ - a_.mean(1, keepdim=True), b_ - b_.mean(1, keepdim=True)
        rp = (ac * bc).sum(1) / (ac.norm(dim=1) * bc.norm(dim=1)).clamp_min(1e-30)
        out[L] = dict(pcc_all=round(float(C.pcc(ref, dev)), 6), row_pcc_median=round(float(rp.median()), 6),
                      row_pcc_p5=round(float(rp.quantile(0.05)), 6), row_pcc_min=round(float(rp.min()), 6),
                      row_last=round(float(rp[-1]), 6), max_abs=round(float((a_ - b_).abs().max()), 5),
                      norm_ratio=round(float(b_.norm() / a_.norm().clamp_min(1e-30)), 6))  # fmt: skip
    return out


res = {}
for mode in MODES:
    menv = MODE_ENV[mode]
    old = {k_: os.environ.get(k_) for k_ in menv}
    os.environ.update(menv)
    try:
        pf = None
        gc.collect()
        pf = build(mode)
        e0 = pf.counters["eager"]
        backend._kv = kv
        pf.prefill(prompt, slot=0, chunk_start=0, n_valid=CH)  # the served chunk: ONE replay
        ttnn.synchronize_device(mesh)
        c = pf.plan(0, CH)[0]
        res[mode] = dict(fp32=fp32_table(pf, c), eager=pf.counters["eager"] - e0, env=menv)
        S.log(f"{tag}: {mode} vs fp32 {json.dumps(res[mode]['fp32'])} eager {res[mode]['eager']}")
    finally:
        for k_, v_ in old.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_

delta = {}
if "op" in res:
    for m in res:
        if m == "op":
            continue
        delta[m] = {L: dict(p5=round(res[m]["fp32"][L]["row_pcc_p5"] - res["op"]["fp32"][L]["row_pcc_p5"], 6),
                            median=round(res[m]["fp32"][L]["row_pcc_median"] - res["op"]["fp32"][L]["row_pcc_median"], 6),
                            max_abs_ratio=round(res[m]["fp32"][L]["max_abs"] / max(res["op"]["fp32"][L]["max_abs"], 1e-9), 3),
                            norm=round(res[m]["fp32"][L]["norm_ratio"] - res["op"]["fp32"][L]["norm_ratio"], 6))
                    for L in LAYERS}  # fmt: skip
row = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           test=f"{os.environ.get('GLM53_ROW_PREFIX', '')}layer_fp32_pair", status="mock" if C.MOCK else "record",
           tp=int(mesh.get_num_devices()), layers=LAYERS, chunk=CH, ref_rows=REF_ROWS, cpu_ref=CPU_REF,
           modes={m: dict(fp32=res[m]["fp32"], eager=res[m]["eager"], env=res[m]["env"]) for m in res},
           delta_vs_op=delta, env_file=args.get("env_file"), env_overrides=args.get("env") or {}, tag=tag)  # fmt: skip
S.log(f"{tag}: ROW {json.dumps(row)}")
if rec:
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")
    if ev:
        with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
RESULT = dict(ok=all(res[m]["eager"] == 0 for m in res), delta=delta)
