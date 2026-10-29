# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cycle model of ttnn.experimental.indexer_score_dsa (the classic factory) per program config, host only.

It replays the factory's banded schedule (indexer_score_work_split.hpp: groups -> grid rows, k-bands -> columns, the
per-core (phase x band) rectangle) and the compute kernel's per-unit instruction sequence (compute_indexer_score.cpp:
the streaming fallback, the full-strip path, the exact serial path) with the reader's deliveries (reader_indexer_score.cpp:
q per (k tile, head group) block when heads stream, q once per group when resident; k once per band down the column), and
reports the critical path per core and the op time per chip.  Costs are parameters; ONE is fitted: the per-block q
delivery of the default config (a DRAM read on the row's sender core + the row multicast handshake, serialized per head
block) from the measured 1M window ladder (README 'Window ladder' 2026-10-22 18:13Z: 0.118 ms per 1K tokens of window at
TP2 = 2,015 cycles per (q tile, k tile, head) per core with 104 of 130 cores lit; indexer_scaling.md section 1b).
usage: python ix_cycle_model.py [--json out.jsonl]"""

from __future__ import annotations

import argparse
import json

HI, DT, TILE = 32, 4, 32
GRID_X, GRID_Y = 13, 10  # p150 / p300a compute grid
GHZ = 1.35

# ---- per-event costs (cycles) -------------------------------------------------------------------------------
COST = dict(
    mm_tile=64,  # one 32x32x32 bf16 x bf16 tile matmul at HiFi4 (tech_reports/GEMM_FLOPS: 16/32/64 at LoFi/HiFi2/HiFi4)
    pack_tile=48,  # one bf16 tile pack (overlaps the next pass's math under half-sync; counted at half)
    pack_l1acc=96,  # one bf16 tile pack with the packer's L1 read-modify-write accumulate
    mul_tile=32,  # one bcast-col multiply (SFPU-free FPU eltwise)
    regs_sync=110,  # tile_regs acquire / commit / wait / release round trip (TRISC handshakes)
    mode_switch=160,  # reconfig_data_format + matmul_block_init / bcast-mul init + pack config
    cb_op=20,  # reserve / push / wait / pop
    untilize_tile=70,  # pack_untilize per tile
    untilize_bracket=200,  # per pack_untilize bracket
    k_band_deliver=1900,  # the column sender's KC x Dt tile DRAM read (latency-bound) + the column multicast
    q_block_deliver=None,  # FITTED below: the row sender's q block DRAM read + the row multicast, per (k tile, head group)
    q_group_deliver=2400,  # resident q / w: one row read + multicast per group (hidden behind the band run)
    dram_lat_tile=40,  # incremental per-tile cost of a multi-tile read beyond the first (bandwidth term)
)
MEASURED_DEFAULT_CYCLES_PER_UNIT = 2015  # 1M ladder slope, TP2 (see the module docstring)


def largest_divisor_leq(n, cap):
    best = 1
    for d in range(2, min(cap, n) + 1):
        if n % d == 0:
            best = d
    return best


def schedule(Sqt, Tt, QC, KC):
    group_count = Sqt // QC
    band_count = -(-Tt // KC)
    group_rows = largest_divisor_leq(group_count, GRID_Y)
    cols = min(band_count, GRID_X)
    fill = GRID_Y // group_rows
    deep = band_count // cols
    num_blocks = max(1, min(fill, deep))
    rows_used = group_rows * num_blocks
    phases = group_count // group_rows
    bands_in_widest_block = -(-band_count // num_blocks)
    max_bands = -(-bands_in_widest_block // cols)
    return dict(group_count=group_count, band_count=band_count, group_rows=group_rows, cols=cols, num_blocks=num_blocks,
                rows_used=rows_used, cores=rows_used * cols, phases=phases, max_bands=max_bands)  # fmt: skip


def resolve(QC, KC, HB, serial, pass_heads=1):
    """the factory's derived knobs: heads per DEST pass, qk batching, the compute path"""
    stream_heads = HB < HI
    subblock_basis = HB
    hp = pass_heads if serial else min(8, largest_divisor_leq(subblock_basis, 8)) if subblock_basis > 1 else 1
    cap = subblock_basis if QC == 1 else TILE
    qk_batch_heads = min(subblock_basis, cap)
    single_chunk = qk_batch_heads == subblock_basis and not stream_heads
    qk_col_batch = KC if (KC >= 2 and single_chunk) else 1
    if serial:
        qk_batch_heads, qk_col_batch = 1, 1
    path = "serial" if serial else ("fullstrip" if qk_col_batch > 1 else "streaming")
    return dict(stream_heads=stream_heads, hp=hp, qk_batch_heads=qk_batch_heads, qk_col_batch=qk_col_batch, path=path)


def unit_compute_cycles(QC, KC, HB, r, c):
    """compute-thread cycles of one QC x KC unit (all heads), by path -- no delivery stalls"""
    hp, qbh, qcb, path = r["hp"], r["qk_batch_heads"], r["qk_col_batch"], r["path"]
    groups = HI // HB
    cyc = 0.0
    if path == "serial":
        mm_phase = (HB // hp) * (c["regs_sync"] + hp * DT * c["mm_tile"] + hp * c["pack_tile"] + 2 * c["cb_op"])
        mul_phase = HB * (c["regs_sync"] + c["mul_tile"] + c["pack_l1acc"] + 2 * c["cb_op"])
        cyc += QC * KC * groups * (2 * c["mode_switch"] + mm_phase + mul_phase)
    elif path == "streaming":
        chunks = HB // qbh
        per_chunk = (
            c["mode_switch"]
            + (qbh // hp) * (c["regs_sync"] + hp * DT * c["mm_tile"] + hp * c["pack_tile"])
            + c["mode_switch"]
            + (c["regs_sync"] + qbh * c["mul_tile"] + c["pack_l1acc"])
            + 4 * c["cb_op"]
        )
        cyc += QC * KC * groups * chunks * per_chunk
    else:  # full strip: matmul phase (head-major packs) + the blocked custom MAC phase per batch of qcb columns
        batches = -(-KC // qcb)
        ct_dim = min(KC, hp)
        mm_phase = c["mode_switch"] + qcb * (HI // hp) * (c["regs_sync"] + hp * DT * c["mm_tile"] + hp * c["pack_tile"])
        mul_phase = c["mode_switch"] + (-(-qcb // ct_dim)) * (
            c["regs_sync"] + HI * ct_dim * c["mul_tile"] + ct_dim * c["pack_tile"]
        )
        cyc += QC * batches * (mm_phase + mul_phase)
    cyc += c["untilize_bracket"] + QC * KC * c["untilize_tile"]  # PHASE 3
    return cyc


def op_model(Sq, T, QC=32, KC=32, HB=1, serial=False, pass_heads=1, c=None, kv_frac=1.0):
    """critical-path cycles per core and the op time per chip. kv_frac: the fraction of the Tt bands scored (lever 2)"""
    c = c or COST
    Sqt, Tt = Sq // TILE, T // TILE
    QCt, KCt = QC // TILE, KC // TILE
    HBr = HI if HB == 0 else HB
    s = schedule(Sqt, Tt, QCt, KCt)
    r = resolve(QCt, KCt, HBr, serial, pass_heads)
    bands_per_core = s["max_bands"]  # the widest cell bounds the op
    real_bands = max(1, round(bands_per_core * kv_frac))
    per_unit = unit_compute_cycles(QCt, KCt, HBr, r, c)
    groups = HI // HBr
    cycles = 0.0
    for _ in range(s["phases"]):
        if not r["stream_heads"]:
            cycles += c["q_group_deliver"]  # resident q / w, once per group (the band run hides the rest)
        for _ in range(real_bands):
            k_deliver = c["k_band_deliver"] + (KCt * DT - 1) * c["dram_lat_tile"]
            if r["stream_heads"]:
                # per (tile_idx, head group) one q block: the sender's DRAM read + the row rendezvous, serialized
                # (the receivers compute the previous block meanwhile: critical path = max(delivery, compute))
                blocks = QCt * KCt * groups
                per_block_compute = per_unit / blocks
                per_block = max(c["q_block_deliver"], per_block_compute)
                band = blocks * per_block
                cycles += max(band, k_deliver)
            else:
                cycles += max(per_unit, k_deliver)  # k of the next band streams behind the compute (2-deep cb_k)
        if r["stream_heads"]:
            cycles += (bands_per_core - real_bands) * QCt * KCt * groups * (c["cb_op"] * 2)  # phantom drains
    units = Sqt * Tt * HI  # (q tile, k tile, head) per chip
    per_unit_eff = cycles * s["cores"] / units / kv_frac
    return dict(Sq=Sq, T=T, cfg=f"{QC},{KC},{HB},{int(serial)},{pass_heads}", path=r["path"], cores=s["cores"], phases=s["phases"],
                bands_per_core=bands_per_core, cycles_per_core=cycles, us=cycles / GHZ / 1e3,
                cycles_per_unit_per_core=per_unit_eff, compute_only_per_unit=per_unit / (QCt * KCt * HI))  # fmt: skip


def fit_default():
    """fit q_block_deliver so the default config reproduces the measured 1M ladder slope at TP2 (Sq 1024, T 262144)"""
    lo, hi = 100.0, 6000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        c = dict(COST, q_block_deliver=mid)
        v = op_model(1024, 262144, c=c)["cycles_per_unit_per_core"]
        if v < MEASURED_DEFAULT_CYCLES_PER_UNIT:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chunked_1m(cfg, c, tp_rows, chunks=512, chunk=2048, layers=11, bound=False):
    """the 1M prompt's indexer seconds: sum over chunks of the op at window T = (c + 1) x chunk pools / 4 (bound: the
    exact causal end -> kv_frac = that over the captured window; else the whole 1M window every chunk)"""
    T_full = chunks * chunk // 4
    total_us = 0.0
    for k in range(chunks):
        if (k + 1) * chunk <= 2051:  # dense chunks: no indexer
            continue
        frac = (k + 1) / chunks if bound else 1.0
        total_us += op_model(tp_rows, T_full, *cfg, c=c, kv_frac=frac)["us"]
    return total_us * layers / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    q_fit = fit_default()
    c = dict(COST, q_block_deliver=q_fit)
    print(
        f"fit: q_block_deliver = {q_fit:.0f} cycles per (k tile, head) q block (default config, 1M slope "
        f"{MEASURED_DEFAULT_CYCLES_PER_UNIT} cycles/unit)"
    )
    rows = []
    cfgs = [
        ("default q32/k32/hg1", (32, 32, 1, False)),
        ("rejected k128/hg0 (DEST-MAC, not exact)", (32, 128, 0, False)),
        ("rejected q64/k128/hg0 (not exact)", (64, 128, 0, False)),
        ("EXACT serial q32/k32/hg0", (32, 32, 0, True)),
        ("EXACT serial q64/k32/hg0", (64, 32, 0, True)),
        ("EXACT serial q32/k128/hg0", (32, 128, 0, True)),
        ("EXACT serial q64/k128/hg0", (64, 128, 0, True)),
        ("EXACT serial q32/k32/hg8 (streams 4 blocks)", (32, 32, 8, True)),
        ("EXACT serial q32/k32/hg0 pass8 (stretch)", (32, 32, 0, True, 8)),
        ("EXACT serial q64/k128/hg0 pass8 (stretch)", (64, 128, 0, True, 8)),
    ]
    for name, cfg in cfgs:
        for Sq, T, lab in (
            (1024, 8192, "32K TP2"),
            (256, 8192, "32K TP8"),
            (1024, 262144, "1M TP2"),
            (256, 262144, "1M TP8"),
        ):
            m = op_model(Sq, T, *cfg, c=c)
            m.update(name=name, geometry=lab)
            rows.append(m)
    base = {r["geometry"]: r for r in rows if r["name"].startswith("default")}
    print(f"{'config':44s} {'geom':8s} {'path':10s} {'cores':>5s} {'us':>10s} {'cyc/unit':>9s} {'x vs def':>8s}")
    for r in rows:
        print(
            f"{r['name']:44s} {r['geometry']:8s} {r['path']:10s} {r['cores']:5d} {r['us']:10.1f} "
            f"{r['cycles_per_unit_per_core']:9.0f} {base[r['geometry']]['us'] / r['us']:8.2f}"
        )
    print("\n1M prompt (512 chunks of 2,048, 11 sparse_dsa layers), indexer seconds at TP8 rows 256 per chip:")
    out = []
    for name, cfg in cfgs:
        if "rejected" in name:
            continue
        full = chunked_1m(cfg, c, 256)
        bnd = chunked_1m(cfg, c, 256, bound=True)
        out.append(dict(name=name, cfg=cfg, one_window_s=full, causal_bound_s=bnd))
        print(f"  {name:44s} one 1M window every chunk {full:7.1f} s   per-chunk causal bound {bnd:7.1f} s")
    if a.json:
        with open(a.json, "a") as f:
            f.write(
                json.dumps(dict(test="ix_cycle_model", fit_q_block_deliver=q_fit, cost=c, rows=rows, one_m=out)) + "\n"
            )


if __name__ == "__main__":
    main()
