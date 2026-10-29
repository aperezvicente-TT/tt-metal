# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a device numerics of the fused kda_pre program on one chip with a real KDA layer's attention weights:
  * program outputs (q, k, v, g, beta, gate_out, the in-place conv-history shift) vs `kda_pre.reference_step` (float64
    on the device's own inputs: qkv / lr read back, state.C before the call, the active mask);
  * the whole attention decode step (OptimizedKDA.decode: projections -> kda_pre -> fused_recurrent_kda -> gated norm ->
    o_proj) with the program vs the op chain, from identical state copies: output PCC, state.C bitwise, state.S PCC;
at B = 1, 2, 3, 4 (ROWS mode, incl. inactive users) and 8, 32 (TILES mode). Dump-then-compare; rows ->
$GLM53_STAGE_EVIDENCE/{pcc.jsonl,tests.jsonl}. GLM53_MOCK=1: logic dry run on the mock cluster."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tt import kda_pre as KP
from models.autoports.zai_org_glm_5_3_flash.tt.config import LayerSpec
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedKDA
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
    FdTileCacheSource,
    OptimizedDecoderConfig,
    OptimizedKDA,
    _upgrade,
)

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
DUMP = os.environ.get("GLM53_DUMP_DIR", "/opt/glm53/work-s7a/dumps")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
PCC_GATE = 0.9999
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
ROWS = []
TR = None  # Traced(mesh), set in main


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name, **kw)
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def host(t):
    return ttnn.to_torch(t).float()


def build_kda(mesh, source, cfg, fused: bool) -> OptimizedKDA:
    """The layer's KDA attention exactly as FusedDecoderLayer + OptimizedDecoderLayer build it."""
    dev = lambda n: source.device(LAYER, n, mesh)
    hst = lambda n: source.host(LAYER, n)
    cat = lambda names: ttnn.concat([dev(n) for n in names], dim=-1, memory_config=DRAM)
    ocfg = OptimizedDecoderConfig(fused_kda_pre=fused)
    kda = FusedKDA(
        mesh,
        w_qkv=cat(["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]),
        w_lowrank=cat(["self_attn.f_a_proj.weight", "self_attn.g_a_proj.weight", "self_attn.b_proj.weight"]),
        w_fb=dev("self_attn.f_b_proj.weight"), w_gb=dev("self_attn.g_b_proj.weight"), w_o=dev("self_attn.o_proj.weight"),
        taps=cat(["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"]),
        o_norm=dev("self_attn.o_norm.weight"), dt_bias=hst("self_attn.dt_bias"), a_log=hst("self_attn.A_log"),
        num_heads=cfg.linear_num_heads, head_dim=cfg.linear_head_dim, hidden=cfg.hidden_size,
        lower_bound=cfg.linear_lower_bound, eps=cfg.rms_norm_eps,
    )  # fmt: skip
    kda.set_decode_weights()
    kda.fcfg = ocfg
    return _upgrade(kda, OptimizedKDA, ocfg)


def masks(mesh, active: torch.Tensor, lower_bound: float):
    m = active.float().reshape(1, 1, -1, 1)
    mk = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
    return mk(m, ttnn.float32), mk(m, ttnn.bfloat16), mk(m * lower_bound, ttnn.float32)


def run_case(mesh, cfg, kda_f, kda_c, b, g, dumps):
    torch.manual_seed(b)
    active = torch.tensor([(i % 3) != 2 for i in range(b)]) if b > 1 else torch.tensor([True])
    act = masks(mesh, active, kda_c.lower_bound)
    # attention input: the layer's real normalised-input scale (golden streams, collapsed + RMS-normalised on host)
    x = g[0, 400 : 400 + 13 * b : 13].float().mean(1)  # [b, 4096]
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    xd = ttnn.from_torch(x.reshape(1, 1, b, 4096).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=mesh, memory_config=L1)  # fmt: skip
    st_f, st_c = kda_f.allocate_state(b), kda_c.allocate_state(b)
    hist0 = (torch.randn(1, 3, b, 24576) * 0.5).to(torch.bfloat16)
    s0 = torch.randn(b, 64, 128, 128) * 0.01
    mk = lambda v, dt: ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM)
    c0_d, s0_d = mk(hist0, ttnn.bfloat16), mk(s0, ttnn.float32)  # the initial state, restored inside every traced call
    for st in (st_f, st_c):
        ttnn.copy(c0_d, st.C)
        ttnn.copy(s0_d, st.S)
    c_before = host(st_f.C)
    st_p = kda_f.allocate_state(b)

    # 1. the program alone vs the oracle on the device's own projections (traced; the history copy resets st_p.C)
    def step1():
        ttnn.copy(st_f.C, st_p.C)
        xm = ttnn.where(act[1], xd, 0.0)
        qkv = kda_f._lin("ds_qkv", xm, kda_f.w_qkv, kda_f.ckc_proj, ttnn.bfloat16)
        lr = kda_f._lin("ds_lowrank", xm, kda_f.w_lowrank, kda_f.ckc_fp32, ttnn.bfloat16)
        return (qkv, lr) + tuple(kda_f.kda_pre(qkv, lr, st_p.C, act, rows=b))

    qkv, lr, q, k, v, gg, beta, go, hist_done = TR.run(step1)
    outs = dict(q=host(q), k=host(k), v=host(v), g=host(gg), beta=host(beta), gate_out=host(go), hist=host(st_p.C),
                qkv=host(qkv), lr=host(lr), c_before=c_before)  # fmt: skip
    TR.replay()
    det1 = bool(torch.equal(host(q), outs["q"]) and torch.equal(host(go), outs["gate_out"])
                and torch.equal(host(st_p.C), outs["hist"]))  # fmt: skip
    TR.release()
    dumps[f"prog_B{b}"] = outs
    torch.save(dumps, os.path.join(DUMP, f"{RUN}_kda_pre.pt"))
    ref = KP.reference_step(outs["qkv"].reshape(b, -1).to(torch.bfloat16), outs["lr"].reshape(b, -1)[:, :320].to(torch.bfloat16),
                            c_before.reshape(3, b, -1).to(torch.bfloat16), kda_f.kda_pre.host_taps,
                            host(kda_f.w_fb).to(torch.bfloat16), host(kda_f.w_gb).to(torch.bfloat16),
                            host(kda_f.dt_bias).reshape(-1), host(kda_f.decay_rate).reshape(-1), kda_f.lower_bound,
                            active)  # fmt: skip
    r = dict(b=b, rows_mode=bool(hist_done), active=active.tolist(), execution=TR.mode, deterministic=det1)
    for name in ("q", "k", "v", "g", "beta", "gate_out"):
        got = outs[name].reshape(b, -1)
        r[f"{name}_pcc"] = pcc(got, ref[name])
        r[f"{name}_max_abs"] = float((got.double() - ref[name]).abs().max())
    if hist_done:
        r["hist_bitwise"] = bool(torch.equal(outs["hist"].reshape(3, b, -1).to(torch.bfloat16), ref["hist_new"]))
    gates = [r[f"{n}_pcc"] for n in ("q", "k", "v", "g", "beta", "gate_out")]
    r["status"] = (
        "record" if MOCK else ("pass" if min(gates) >= PCC_GATE and r.get("hist_bitwise", True) and det1 else "fail")
    )
    row("kda_pre_program_vs_oracle", **r)

    # 2. whole attention decode: fused vs chain from identical state
    def step2():
        for st in (st_f, st_c):  # identical initial state for both paths, every replay
            ttnn.copy(c0_d, st.C)
            ttnn.copy(s0_d, st.S)
        return kda_f.decode(xd, st_f, act), kda_c.decode(xd, st_c, act)

    yf_d, yc_d = TR.run(step2)
    y_f, y_c = host(yf_d), host(yc_d)
    cf, cc = host(st_f.C), host(st_c.C)
    sf, sc = host(st_f.S), host(st_c.S)
    dumps[f"layer_B{b}"] = dict(y_f=y_f, y_c=y_c)
    torch.save(dumps, os.path.join(DUMP, f"{RUN}_kda_pre.pt"))
    TR.release()
    r = dict(b=b, execution=TR.mode, attn_out_pcc=pcc(y_f.reshape(b, -1), y_c.reshape(b, -1)), state_C_bitwise=bool(torch.equal(cf, cc)),
             state_S_pcc=pcc(sf, sc), state_S_max_abs=float((sf - sc).abs().max()))  # fmt: skip
    r["status"] = (
        "record"
        if MOCK
        else (
            "pass"
            if r["attn_out_pcc"] >= PCC_GATE and r["state_C_bitwise"] and r["state_S_pcc"] >= PCC_GATE
            else "fail"
        )
    )
    row("kda_pre_attention_vs_chain", **r)


def main():
    os.makedirs(DUMP, exist_ok=True)
    cfg = hf_config()
    if LayerSpec.from_config(cfg, LAYER).attention != "kda":
        raise SystemExit(f"layer {LAYER} is not a KDA layer")
    global TR
    mesh = (
        ttnn.open_device(device_id=0)
        if MOCK
        else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=300_000_000)
    )
    TR = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        kda_f, kda_c = build_kda(mesh, source, cfg, True), build_kda(mesh, source, cfg, False)
        g = golden(LAYER, 2048)["streams_in"].float()
        dumps = {}
        for b in (1, 2, 3, 4, 8, 32):
            try:
                run_case(mesh, cfg, kda_f, kda_c, b, g, dumps)
            except Exception:
                traceback.print_exc()
                row("kda_pre_program_vs_oracle", b=b, status="fail", error=traceback.format_exc()[-400:])
    finally:
        with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "b") if k in r}) + "\n")
        ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_KDA_PRE_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
