# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host CB + cycle model of the SP-row sparse attention phase (Stage 16, M-B2 section 2b; program 2026-10-25
08:21:18Z: "host CB model, pair unit vs the B-q path on the same selections, then the TP8 estimate"), host only.

Per chip, one chunk of a sparse_dsa layer: the chip's R query rows (SP rows: TP8 256, the pair 1,024), all 64 heads,
q_lat [R, 64, 512] (after the q a2a), each query's 512 selected pools + tail (ixp's selection), the latent cache token-
sharded over the N chips (pages of 32 tokens, owner = page % N for the model).
  gatherer (one MM-only core per 32-query block, BRISC + NCRISC): the block's union of selected POOLS (merged from the
            32 queries' pool lists into a window bitmap), each pool's 4 latent rows (4 KiB bf16) read from the owner --
            local DRAM or a remote chip over the fabric -- in pool order into a staging ring, multicast to the block's
            compute cores; plus the per-query selection mask tiles over the union order (0 / -inf, 32 queries x 32
            union tokens per tile = 8 pools), built from the per-query pool bitmaps
  compute  (HG heads x 32 queries per core = HG row tiles of 32): flash-MLA over the union in KU-token chunks:
            S = Q K^T (16 K-tiles of the 512 latent) + mask, online max / sum, P V (V = K = the latent rows),
            o [HG x 32 rows, 512] -> DRAM (or the v_b phase)
CB rule as tests/s16/ix_phase_model.py (one producer / one consumer, pushes tile the capacity, L1 budget). Cycle costs as
there plus: exp tile ~1,000 (the online softmax: one exp per S tile), a fabric read ~0.4 us latency + 25 GB/s per link
(2 links per chip), local DRAM 512 GB/s per chip. Union sizes from the real 32K golden (layer 3, union_stats.py): tokens
per 32-query block mean 5,313 (min 2,104, max 8,104), pools 1,328; assumed to grow toward 8,192 at 1M (parameter).
usage: python spa_phase_model.py"""

from __future__ import annotations

import argparse
import json

GHZ = 1.35
MM = dict(LoFi=16, HiFi2=32, HiFi4=64)


def model(rows: int, union_tokens: float, n_chips: int, fid: str = "HiFi2", cores: int = 100, hg: int = 8,
          ku: int = 128) -> dict:  # fmt: skip
    blocks = rows // 32
    items = blocks * (64 // hg)  # (block, head group) work items
    waves = -(-items // max(1, cores - blocks))  # the gatherers take one core per block
    ut = -(-int(union_tokens) // 32)  # union tiles of 32 tokens
    # per item: QK = HG row tiles x ut column tiles x 16 K-steps; PV = HG x 16 out tiles x ut K-steps; softmax per S tile
    mm_tiles = hg * ut * 16 + hg * 16 * ut
    sm_tiles = hg * ut
    cyc = mm_tiles * MM[fid] + sm_tiles * (1000 + 3 * 120) + (ut // (ku // 32)) * 4 * 300  # + chunk-level rescales / inits
    t_comp = waves * cyc / GHZ / 1e3  # us
    # gather per block: union pools x 4 KiB; (N-1)/N of them remote
    pools = union_tokens / 4
    bytes_blk = pools * 4096
    remote = bytes_blk * (n_chips - 1) / n_chips
    local = bytes_blk - remote
    t_gather_blk = max(local / 512e3, remote / 50e3) + 0.4 * 4  # us: the slower of DRAM / fabric (2 links x 25 GB/s)
    t_gather = blocks * t_gather_blk / min(blocks, cores) * blocks / max(1, blocks)  # gatherers run in parallel
    t_mask = blocks * (32 * 512 * 4 * 3 + ut * 32 * 20) / GHZ / 1e3 / blocks  # per block on its gatherer (parallel)
    span = max(t_comp, t_gather, t_mask)
    fabric_mib = blocks * remote / 2**20
    return dict(rows=rows, blocks=blocks, items=items, waves=waves, union_tokens=int(union_tokens), fid=fid,
                compute_us=round(t_comp, 1), gather_us=round(t_gather, 1), mask_us=round(t_mask, 1), span_us=round(span, 1),
                fabric_mib=round(fabric_mib, 1), local_mib=round(blocks * local / 2**20, 1),
                bound=max((("compute", t_comp), ("gather", t_gather), ("mask", t_mask)), key=lambda kv: kv[1])[0])  # fmt: skip


def cb_check(hg: int = 8, ku: int = 128) -> dict:
    """member CBs: Q (HG x 16 tiles bf16), K/V ring (2 x ku tokens x 512 bf16), mask (HG? no: 32 x ku per query block:
    ku / 32 tiles bf16, x 2), S / P (HG x ku / 32 tiles fp32 / bf16), O accumulators (HG x 16 fp32), stats"""
    t16, t32 = 2048, 4096
    cbs = dict(Q=hg * 16 * t16, KV=2 * (ku // 32) * 16 * t16, MASK=2 * (ku // 32) * t16, S=hg * (ku // 32) * t32,
               P=hg * (ku // 32) * t16, O=hg * 16 * t32, STAT=4 * hg * t32)
    return dict(cbs=cbs, total=sum(cbs.values()), ok=sum(cbs.values()) <= 900 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = []
    chk = cb_check()
    print("member CBs (HG 8, KU 128):", chk)
    for tp, rows, n in (("TP8", 256, 8), ("pair", 1024, 2)):
        for label, ut in (("32K measured mean", 5313), ("32K max", 8104), ("1M assumed", 8192)):
            for fid in ("HiFi2", "HiFi4"):
                m = model(rows, ut, n, fid)
                m.update(tp=tp, case=label)
                out.append(m)
                print("%-4s %-18s %-5s: compute %7.1f us  gather %6.1f us  mask %5.1f us -> span %7.1f us (%s); fabric %6.1f MiB, "
                      "local %5.1f MiB per layer per chip" % (tp, label, fid, m["compute_us"], m["gather_us"], m["mask_us"],
                                                               m["span_us"], m["bound"], m["fabric_mib"], m["local_mib"]))
    if a.json:
        with open(a.json, "w") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
