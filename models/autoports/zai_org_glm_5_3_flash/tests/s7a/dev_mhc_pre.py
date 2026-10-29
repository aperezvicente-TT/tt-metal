# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a W1 step (a): the fused mhc_pre program on one chip with real layer weights, against
  * the CPU oracle (`mhc_pre.reference_step`, float64) incl. the debug taps (M, S, z, plane, coefficients, 1/rms(h)),
  * today's op chain (OptimizedHyperConnection.pre / OptimizedStreamMajorHyperConnection.pre + the layer's sharded
    RMSNorm + typecast), on the same device and the same residual rows (golden streams of the layer),
at T = 1 / 2 / 4 (fused path: flat DRAM input with the stacked residual copy, and stacked L1 input) and T = 8 / 32
(stream-major: flat DRAM input with the residual copy, and the stream-major L1 input), for the attention and the FFN
sublayer (two programs with different weights back to back: the program-cache hit re-binds every address).
Also: the plane-layout probe (MHC_PLANE_PROBE), determinism (two calls bitwise equal), trace capture + replay ==
eager (bitwise; run with TT_METAL_TRACE_ALLOC_TRACKING=1). Every device output is dumped to $GLM53_DUMP_DIR before any
comparison (dump-then-compare); rows -> $GLM53_STAGE_EVIDENCE/{pcc.jsonl,tests.jsonl}. GLM53_MOCK=1: logic dry run on
the mock cluster (numbers are garbage, no asserts)."""

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
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import (
    FusedHyperConnection,
    StreamMajorHyperConnection,
    to_stacked,
)
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (
    FdTileCacheSource,
    OptimizedDecoderConfig,
    OptimizedHyperConnection,
    OptimizedStreamMajorHyperConnection,
    _upgrade,
)

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
DUMP = os.environ.get("GLM53_DUMP_DIR", "/opt/glm53/work-s7a/dumps")
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
PCC_GATE = 0.9999
TILE = 32
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
ROWS = []
TR = None  # Traced(mesh), set in main


def row(name: str, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name, **kw)
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)
    return r


def host(t) -> torch.Tensor:
    return ttnn.to_torch(t).float()


class Chain:
    """Today's decode op chain of one sublayer: the optimized fused-path / stream-major hyperconnection modules and the
    layer's width-sharded RMSNorm (FusedDecoderLayer.__init__ / OptimizedDecoderLayer._norm, verbatim settings)."""

    def __init__(self, mesh, source, cfg, which: str, norm_name: str):
        ocfg = OptimizedDecoderConfig()
        dev = lambda n: source.device(LAYER, n, mesh)
        hst = lambda n: source.host(LAYER, n)
        kw = dict(hc_mult=cfg.hc_mult, hidden_size=cfg.hidden_size, sinkhorn_iters=cfg.hc_sinkhorn_iters,
                  hc_eps=cfg.hc_eps, norm_eps=cfg.rms_norm_eps)  # fmt: skip
        self.hc = FusedHyperConnection(mesh, dev(f"hc_{which}_fn"), hst(f"hc_{which}_base"), hst(f"hc_{which}_scale"),
                                       ksplit=ocfg.mhc_ksplit, fcfg=ocfg, **kw)  # fmt: skip
        self.sm = StreamMajorHyperConnection(mesh, dev(f"hc_{which}_fn"), hst(f"hc_{which}_base"),
                                             hst(f"hc_{which}_scale"), **kw)  # fmt: skip
        self.sm.fcfg = ocfg
        self.sm.__class__ = OptimizedStreamMajorHyperConnection
        _upgrade(self.hc, OptimizedHyperConnection, ocfg, mesh)
        self.hc.consts = self.sm.consts
        d = cfg.hidden_size
        cores = ttnn.CoreGrid(x=8, y=4)
        self.shard = ttnn.create_sharded_memory_config(
            (TILE, d // cores.num_cores), cores, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )  # fmt: skip
        bw = d // cores.num_cores // TILE
        self.prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[cores.x, cores.y], subblock_w=max(s for s in (4, 3, 2, 1) if bw % s == 0),
            block_h=1, block_w=bw, inplace=False,
        )  # fmt: skip
        w = dev(norm_name)
        self.w_rm = ttnn.to_layout(ttnn.reshape(w, (1, 1, d // TILE, TILE)), ttnn.ROW_MAJOR_LAYOUT)
        self.eps = cfg.rms_norm_eps
        self.ckc = self.hc.ckc

    def norm(self, h):
        out = ttnn.rms_norm(ttnn.to_memory_config(h, self.shard), weight=self.w_rm, epsilon=self.eps,
                            program_config=self.prog, memory_config=self.shard, compute_kernel_config=self.ckc)  # fmt: skip
        return ttnn.typecast(ttnn.sharded_to_interleaved(out, L1), ttnn.bfloat16)

    def fused_path(self, x_l1, t):
        h, post, comb = self.hc.pre(x_l1, to_stacked(x_l1, 4))
        return self.norm(ttnn.reshape(h, (1, 1, t, 4096))), post, comb

    def stream_major(self, x_sm):
        h, post, comb = self.sm.pre(x_sm)
        return self.norm(h), post, comb


def make_program(mesh, source, cfg, which: str, norm_name: str, fn_dtype=ttnn.bfloat16) -> M.MhcPre:
    n = cfg.hc_mult
    scale = source.host(LAYER, f"hc_{which}_scale").float().reshape(-1)
    per_mix = torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)])
    return M.MhcPre(
        mesh, source.host(LAYER, f"hc_{which}_fn").reshape(n * cfg.hidden_size, (2 + n) * n), per_mix,
        source.host(LAYER, f"hc_{which}_base").float().reshape(-1), source.host(LAYER, norm_name).float().reshape(-1),
        hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters,
        hidden=cfg.hidden_size, n=n, fn_dtype=fn_dtype,
    )  # fmt: skip


def oracle(prog: M.MhcPre, x: torch.Tensor, cfg):
    h = prog.host
    return M.reference_step(x, h["fn_t"], h["scale"], h["bias"], h["w"], hc_eps=cfg.hc_eps, x_eps=cfg.rms_norm_eps,
                            h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters)  # fmt: skip


def stats(name, dev_out: torch.Tensor, ref: torch.Tensor) -> dict:
    d, r = dev_out.double().reshape(-1), ref.double().reshape(-1)
    err = (d - r).abs()
    return {
        f"{name}_pcc": pcc(d, r),
        f"{name}_max_abs": float(err.max()),
        f"{name}_max_rel": float((err / (r.abs() + 1e-6)).max()),
    }


def post_comb_host(post, comb, t, pc_mode):
    """Device post / comb in either layout -> ([t, 4], [t, 4, 4])."""
    p, c = host(post), host(comb)
    if pc_mode == M.PC_STACKED:
        return p.reshape(t, 4), c.reshape(t, 4, 4)
    return p.reshape(4, t).t(), c.reshape(t, 4, 4)


def check_case(dumps, mesh, cfg, prog, chain, sub, x_rows, t, variant):
    """One (sublayer, T, input layout) case: fused program vs oracle vs chain."""
    x_host = x_rows.reshape(t, 16384).float()
    x_dram = ttnn.from_torch(x_host.reshape(1, 1, t, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                             memory_config=DRAM)  # fmt: skip
    x_l1 = ttnn.to_memory_config(x_dram, L1)
    stacked_path = t <= 4
    pc = M.PC_STACKED if stacked_path else M.PC_STREAM_MAJOR
    if variant == "flat_dram":
        src, xm = x_dram, M.X_FLAT
        xo = M.XOUT_STACKED if stacked_path else M.XOUT_FLAT
    elif variant == "stacked_l1":
        src, xm, xo = to_stacked(x_l1, 4), M.X_STACKED, M.XOUT_NONE
    else:  # stream-major L1 (the layer's concat output) as input
        # the stream-major residual [1,4,T,4096] (stream s = columns [4096 s, 4096 (s+1))) — slices + concat, as the
        # layer builds it; ttnn.reshape would be a row-major reinterpretation (torch.reshape semantics), not this split
        src = ttnn.concat([ttnn.slice(x_l1, (0, 0, 0, i * 4096), (1, 1, t, (i + 1) * 4096)) for i in range(4)], dim=1,
                          memory_config=L1)  # fmt: skip
        xm, xo = M.X_FLAT, M.XOUT_NONE

    def step():
        """the fused program and today's chain on the same residual rows (one traced call)"""
        out = prog(src, t=t, x_mode=xm, xout_mode=xo, pc_mode=pc, debug=True)
        if stacked_path:
            chain_out = chain.fused_path(x_l1, t)
        else:
            x_sm = ttnn.concat([ttnn.slice(x_l1, (0, 0, 0, i * 4096), (1, 1, t, (i + 1) * 4096)) for i in range(4)],
                               dim=1, memory_config=L1)  # fmt: skip
            chain_out = chain.stream_major(x_sm)
        return out, chain_out

    out, (hn_cd, post_cd, comb_cd) = TR.run(step)  # traced (eager only as the compile step)
    hn = host(out.hn).reshape(-1, 4096)[:t]
    post, comb = post_comb_host(out.post, out.comb, t, pc)
    taps = host(prog.dbg)
    if stacked_path:
        post_c, comb_c = host(post_cd).reshape(t, 4), host(comb_cd).reshape(t, 4, 4)
    else:
        post_c, comb_c = host(post_cd).reshape(4, t).t(), host(comb_cd).reshape(t, 4, 4)
    hn_c = host(hn_cd).reshape(-1, 4096)[:t]
    x_out = host(out.x_out) if out.x_out is not None else None
    TR.replay()  # determinism: a second replay of the same trace
    hn2 = host(out.hn).reshape(-1, 4096)[:t]
    TR.release()
    key = f"{sub}_T{t}_{variant}"
    dumps[key] = dict(x=x_host, hn=hn, hn2=hn2, post=post, comb=comb, taps=taps, hn_chain=hn_c, post_chain=post_c,
                      comb_chain=comb_c, x_out=x_out)  # fmt: skip
    torch.save(dumps, os.path.join(DUMP, f"{RUN}_mhc_pre.pt"))  # dump before comparing
    ref = oracle(prog, x_host, cfg)
    tp = M.decode_taps(taps, t)
    r = dict(sublayer=sub, t=t, variant=variant, pc_mode=pc, x_mode=xm, xout_mode=xo, execution=TR.mode)
    r.update(stats("hn_vs_oracle", hn, ref["hn"]))
    r.update(stats("hn_vs_chain", hn, hn_c))
    r.update(stats("chain_hn_vs_oracle", hn_c, ref["hn"]))
    r.update(stats("post_vs_chain", post, post_c))
    r.update(stats("comb_vs_chain", comb, comb_c))
    r.update(stats("post_vs_oracle", post, ref["post"]))
    r.update(stats("comb_vs_oracle", comb, ref["comb"]))
    r.update(stats("chain_post_vs_oracle", post_c, ref["post"]))
    r.update(stats("chain_comb_vs_oracle", comb_c, ref["comb"]))
    r.update(stats("tap_z_vs_oracle", tp["z_tok"], ref["z"]))
    r.update(stats("tap_pre_vs_oracle", tp["pre"], ref["pre"]))
    r.update(stats("tap_inv_h_vs_oracle", tp["inv_h"], ref["inv_h"].reshape(-1)))
    # K core (0,0) owns column tiles 0, 1 of the 4 streams: its M partial (x tf32 @ fn^T slice) and S partial (exact)
    cols = torch.cat([torch.arange(s * 4096, s * 4096 + 64) for s in range(4)])
    xs = x_host[:, cols].double()
    m0 = M.tf32(x_host[:, cols]).double() @ prog.host["fn_t"][cols].double()
    s0 = sum((x_host[:, cols[i * 32 : i * 32 + 32]].double() ** 2) for i in range(8))
    r.update(stats("tap_k0_M_vs_host", tp["k0_M"][:t, :24], m0))
    r["tap_k0_S_bitwise_fp32"] = bool(
        torch.equal(
            tp["k0_S"][:t].float(),
            torch.stack(
                [sum(x_host[tt, cols[i * 32 : i * 32 + 32]].float() ** 2 for i in range(8)) for tt in range(t)]
            ),
        )
    )
    r.update(stats("tap_k0_S_vs_host", tp["k0_S"][:t], s0))
    r["deterministic"] = bool(torch.equal(hn, hn2))
    if x_out is not None:
        want = to_stacked_host(x_host, t) if stacked_path else x_host.reshape(t, 4, 4096).permute(1, 0, 2)
        r["x_out_bitwise"] = bool(torch.equal(x_out.reshape(want.shape), want))
    # gate: the program vs the CPU oracle >= 0.9999 for hn / post / comb, and vs today's chain >= 0.9999 unless the chain
    # itself is further than that from the oracle (then: no further from the chain than the chain is from the oracle,
    # within 1e-4) — W1a showed the chain's hn / post at 0.99973 vs the oracle where the program is at 0.999999
    vs_oracle = [r["hn_vs_oracle_pcc"], r["post_vs_oracle_pcc"], r["comb_vs_oracle_pcc"]]
    vs_chain_ok = all(r[f"{q}_vs_chain_pcc"] >= min(PCC_GATE, r[f"chain_{q}_vs_oracle_pcc"] - 1e-4)
                      for q in ("hn", "post", "comb"))  # fmt: skip
    r["gate_rule"] = "oracle>=0.9999 & chain>=min(0.9999, chain_vs_oracle-1e-4)"
    ok = min(vs_oracle) >= PCC_GATE and vs_chain_ok and r["deterministic"] and r.get("x_out_bitwise", True)
    r["status"] = "record" if MOCK else ("pass" if ok else "fail")
    row("mhc_pre_numerics", **r)
    for tns in (x_dram, x_l1):
        ttnn.deallocate(tns)


def to_stacked_host(x_host, t):
    return x_host.reshape(t, 4, 4096)


def probe(mesh, prog):
    """MHC_PLANE_PROBE: coef plane p must equal 1000 * plane p + p on every element of plane_off(p, .)."""
    t = 32
    x = ttnn.from_torch(torch.randn(1, 1, t, 16384), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=DRAM)  # fmt: skip
    TR.run(lambda: prog(x, t=t, x_mode=M.X_FLAT, pc_mode=M.PC_STREAM_MAJOR, debug=True, probe=True))
    taps = host(prog.dbg).reshape(M.N_TAPS, 32, 32)
    TR.release()
    plane, coef = M.face_major(taps[3]).double(), M.face_major(taps[4]).double()
    want = torch.zeros(1024, dtype=torch.float64)
    for p in range(32):
        for tt in range(32):
            o = M.plane_off(p, tt)
            want[o] = plane[o] * 1000.0 + p
    bad = int(((coef - want).abs() > 1e-3 * (want.abs() + 1)).sum())
    # which plane index does each wrong element carry? (diagnoses a wrong lane / parity assumption)
    carried = ((coef - plane * 1000.0).round()).reshape(-1)
    torch.save(dict(plane=plane, coef=coef), os.path.join(DUMP, f"{RUN}_probe.pt"))
    row("mhc_pre_plane_probe", bad_elements=bad, status="record" if MOCK else ("pass" if bad == 0 else "fail"),
        carried_plane_hist=torch.bincount(carried.clamp(0, 40).long(), minlength=41).tolist())  # fmt: skip
    ttnn.deallocate(x)


def saturation_check(mesh, source, cfg, x_rows):
    """A pre bias of -1000 drives sigmoid to 0 (pre = eps): no NaN / Inf anywhere in the outputs."""
    n = cfg.hc_mult
    scale = source.host(LAYER, "hc_attn_scale").float().reshape(-1)
    per_mix = torch.cat([scale[0].repeat(n), scale[1].repeat(n), scale[2].repeat(n * n)])
    bias = source.host(LAYER, "hc_attn_base").float().reshape(-1).clone()
    bias[0] = -1000.0
    bias[5] = -1000.0
    prog = M.MhcPre(mesh, source.host(LAYER, "hc_attn_fn").reshape(n * cfg.hidden_size, (2 + n) * n), per_mix, bias,
                    source.host(LAYER, "input_layernorm.weight").float().reshape(-1), hc_eps=cfg.hc_eps,
                    x_eps=cfg.rms_norm_eps, h_eps=cfg.rms_norm_eps, iters=cfg.hc_sinkhorn_iters)  # fmt: skip
    t = 4
    x = ttnn.from_torch(x_rows.reshape(1, 1, t, 16384).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=mesh, memory_config=DRAM)  # fmt: skip
    out = TR.run(lambda: prog(x, t=t, x_mode=M.X_FLAT, pc_mode=M.PC_STACKED, debug=True))
    hn, post, comb = host(out.hn).reshape(-1, 4096)[:t], *post_comb_host(out.post, out.comb, t, M.PC_STACKED)
    tp = M.decode_taps(host(prog.dbg), t)
    TR.release()
    finite = all(bool(torch.isfinite(v).all()) for v in (hn, post, comb, tp["pre"]))
    ref = oracle(prog, x_rows.reshape(t, 16384).float(), cfg)
    r = dict(finite=finite, pre0=tp["pre"][:, 0].tolist(), post1=post[:, 1].tolist())
    r.update(stats("hn_vs_oracle", hn, ref["hn"]))
    r["status"] = "record" if MOCK else ("pass" if finite and r["hn_vs_oracle_pcc"] >= PCC_GATE else "fail")
    row("mhc_pre_sigmoid_saturation", **r)


def repeat_check(mesh, prog, g, cfg, n_calls=48):
    """Back-to-back calls with changing T (1, 4, 1, 32, ...) under the same program-cache entry per mode, every result
    bitwise equal to the first call of the same (T, input): stale runtime args or protocol races would show here."""
    cases = {}
    for t, xm, pc in ((1, M.X_FLAT, M.PC_STACKED), (4, M.X_FLAT, M.PC_STACKED), (32, M.X_FLAT, M.PC_STREAM_MAJOR)):
        rows = g[0, 500 : 500 + 11 * t : 11]
        cases[t] = (ttnn.from_torch(rows.reshape(1, 1, t, 16384).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                    device=mesh, memory_config=DRAM), xm, pc)  # fmt: skip
    order = [1, 4, 1, 32, 4, 1, 32, 32, 1, 4]
    # traced: the alternating sequence (program-cache hits with changing T / addresses) is ONE trace, replayed
    # n_calls / len(order) times; every call's outputs bitwise equal to the first call of the same T
    outs = TR.run(lambda: [prog(cases[t][0], t=t, x_mode=cases[t][1], pc_mode=cases[t][2]) for t in order])
    first, mismatches, calls = {}, 0, 0
    for rep in range(max(1, n_calls // len(order))):
        if rep:
            TR.replay()
        for t, out in zip(order, outs):
            res = (host(out.hn), host(out.post), host(out.comb))
            calls += 1
            if t not in first:
                first[t] = res
            elif not all(torch.equal(a, b) for a, b in zip(res, first[t])):
                mismatches += 1
    TR.release()
    row("mhc_pre_repeat_bitwise", calls=calls, mismatches=mismatches, execution=TR.mode,
        status="record" if MOCK else ("pass" if mismatches == 0 else "fail"))  # fmt: skip


def trace_check(mesh, prog, x_rows, cfg):
    """Captured mhc_pre (T = 1, flat DRAM input, stacked residual copy) replays == eager, bitwise."""
    t = 1
    x = ttnn.from_torch(x_rows.reshape(1, 1, t, 16384).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=mesh, memory_config=DRAM)  # fmt: skip
    kw = dict(t=t, x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED, pc_mode=M.PC_STACKED)
    eager = prog(x, **kw)
    e_hn, e_post, e_comb = host(eager.hn), host(eager.post), host(eager.comb)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        traced = prog(x, **kw)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    same = []
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        same.append(bool(torch.equal(host(traced.hn), e_hn) and torch.equal(host(traced.post), e_post)
                         and torch.equal(host(traced.comb), e_comb)))  # fmt: skip
    ttnn.release_trace(mesh, tid)
    row("mhc_pre_trace_replay", replays=len(same), bitwise_equal=all(same),
        status="record" if MOCK else ("pass" if all(same) else "fail"))  # fmt: skip


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py, a unit submesh); None = open chip 0 here."""
    os.makedirs(DUMP, exist_ok=True)
    cfg = hf_config()
    global TR
    own = mesh is None
    if own:
        mesh = ttnn.open_device(device_id=0) if MOCK else ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), trace_region_size=200_000_000)  # fmt: skip
    TR = Traced(mesh)
    rc = 0
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        progs = {"attn": make_program(mesh, source, cfg, "attn", "input_layernorm.weight"),
                 "ffn": make_program(mesh, source, cfg, "ffn", "post_attention_layernorm.weight")}  # fmt: skip
        chains = {"attn": Chain(mesh, source, cfg, "attn", "input_layernorm.weight"),
                  "ffn": Chain(mesh, source, cfg, "ffn", "post_attention_layernorm.weight")}  # fmt: skip
        g = golden(LAYER, 2048)["streams_in"].float()  # [1, 2048, 4, 4096]
        try:
            probe(mesh, progs["attn"])
        except Exception:
            traceback.print_exc()
            row("mhc_pre_plane_probe", status="fail", error=traceback.format_exc()[-400:])
        dumps = {}
        # fn^T in fp32 (today's copy) as a control for the bf16-fn matmul (SrcA bf16 x SrcB tf32)
        try:
            progs["attn_fn32"] = make_program(
                mesh, source, cfg, "attn", "input_layernorm.weight", fn_dtype=ttnn.float32
            )
            check_case(dumps, mesh, cfg, progs["attn_fn32"], chains["attn"], "attn_fn32", g[0, 300:301], 1, "flat_dram")
        except Exception:
            traceback.print_exc()
            row("mhc_pre_numerics", sublayer="attn_fn32", t=1, variant="flat_dram", status="fail",
                error=traceback.format_exc()[-400:])  # fmt: skip
        for t, variants in (
            (1, ("flat_dram", "stacked_l1")),
            (2, ("flat_dram", "stacked_l1")),
            (4, ("flat_dram", "stacked_l1")),
            (8, ("flat_dram", "sm_l1")),
            (32, ("flat_dram", "sm_l1")),
        ):
            x_rows = g[0, 300 : 300 + 37 * t : 37]  # t distinct positions, realistic magnitudes
            for sub in ("attn", "ffn"):
                for v in variants:
                    try:
                        check_case(dumps, mesh, cfg, progs[sub], chains[sub], sub, x_rows, t, v)
                    except Exception:
                        traceback.print_exc()
                        row("mhc_pre_numerics", sublayer=sub, t=t, variant=v, status="fail",
                            error=traceback.format_exc()[-400:])  # fmt: skip
        for name, fn in (
            ("mhc_pre_sigmoid_saturation", lambda: saturation_check(mesh, source, cfg, g[0, 700:704])),
            ("mhc_pre_repeat_bitwise", lambda: repeat_check(mesh, progs["attn"], g, cfg)),
        ):
            try:
                fn()
            except Exception:
                traceback.print_exc()
                row(name, status="fail", error=traceback.format_exc()[-400:])
        try:
            if not MOCK:  # the mock cluster cannot capture traces
                trace_check(mesh, progs["attn"], g[0, 1000:1001], cfg)
        except Exception:
            traceback.print_exc()
            row("mhc_pre_trace_replay", status="fail", error=traceback.format_exc()[-400:])
    finally:
        with open(os.path.join(OUT, "pcc.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(OUT, "tests.jsonl"), "a") as f:
            for r in ROWS:
                f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status") if k in r}
                                   | {k: r[k] for k in ("sublayer", "t", "variant") if k in r}) + "\n")  # fmt: skip
        TR.release()
        if own and MOCK:
            ttnn.close_device(mesh)
        elif own:
            ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MHC_PRE_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else rc


if __name__ == "__main__":
    sys.exit(main())
