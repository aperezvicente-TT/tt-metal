# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host CB model of the indexer-score + running top-512 phase (Stage 16, M-B2 section 2 S2 / S3; program
2026-10-25 07:15:21Z: "host CB model first (one producer / one consumer per CB)"), host only.

The phase, per chip, one chunk of a sparse_dsa layer (SP rows: the chip's R query rows, all 32 indexer heads):
  streamer (MM-only core, BRISC)  reads the slot's pooled-key pages in pool order (32 pools x 128 = 4 bf16 tiles = one
                                  key tile KT) and multicasts KC-tile blocks into every member's CB_K ring (PfStream's
                                  PARK / VALID protocol: the streamer is CB_K's only producer, a remote writer);
  member compute (TRISC)          per block and per resident query q: S_q = relu(q_idx_q @ K^T) (4 bf16 matmuls per key
                                  tile at HiFi2, relu applied by the packer) -> CB_SR; then the head weights
                                  wsc_q @ S_q accumulated over the core's NQ queries into ONE fp32 score tile per key tile
                                  (row r = query r's 32 pool scores) -> CB_SC (the decode I4 form, kernels/mla/att_c.cpp
                                  score(): no per-head mode switch, no SFPU multiply);
  member BRISC + NCRISC (filter)  CB_SC's only consumer (the two RISCs split the NQ rows): per query a running candidate
                                  buffer of order-preserving uint32 keys + pool ids, filtered by the current threshold;
                                  on overflow an exact radix select (4-bit digits, att_dsa_dm1.cpp I5) keeps the top 512
                                  and raises the threshold; at the end the exact top-512 + ties -> the selection rows.
CB rule checked here: every CB has exactly ONE producer (role, RISC) and ONE consumer (role, RISC); every push / wait size
divides the capacity (memory: CB push sizes must tile capacity); the member's CB bytes fit the L1 budget. The event
simulation runs the streamer -> compute -> filter pipeline of one member over the whole window with the costs below and
reports the span, the bound resource and the stall shares; the per-window table is the phase time per chip.
Costs (cycles at 1.35 GHz): tile matmul 16 / 32 / 64 at LoFi / HiFi2 / HiFi4 (tech_reports GEMM_FLOPS); fp32 tile pack
~60 (packer ReLU free); a mode switch (reconfig + matmul init) ~160; a filter compare ~4 RISC-V cycles per score;
a radix select over the buffer ~20 cycles per candidate per round x 8 rounds. usage: python ix_phase_model.py"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

GHZ = 1.35
KT_BYTES = 4 * 2048  # one key tile: 32 pools x 128 bf16
COST = dict(mm=dict(LoFi=16, HiFi2=32, HiFi4=64), pack=60, mode=160, cmp=4, radix=20, dram_gbs=512.0, mcast_gbs=40.0)


@dataclass
class CB:
    name: str
    role: str
    pages: int
    page_bytes: int
    producer: tuple
    consumer: tuple
    push: int
    wait: int


@dataclass
class Plan:
    rows: int  # query rows per chip (TP8 256, the TP2 pair 1,024)
    cores: int = 64  # member (query) cores
    kc: int = 4  # key tiles per block (one mode switch per block)
    depth: int = 2  # CB_K blocks in flight
    fid: str = "HiFi2"
    cand: int = 2048  # candidates per query before a radix select
    s_dtype: str = "fp32"  # CB_SR format (the relu'd per-head scores)
    skip: bool = False  # the filter first compares a block row's max (computed with the scores) against the threshold
    cbs: list = field(default_factory=list)

    @property
    def nq(self) -> int:  # resident queries per member core (one row tile of 32 head-rows each)
        return -(-self.rows // self.cores)

    def build(self):
        nq, kc = self.nq, self.kc
        s = ("member", "streamer")
        self.cbs = [
            CB("Q", "member", 4 * nq, 2048, ("member", "brisc"), ("member", "trisc"), 4 * nq, 4 * nq),
            CB("WS", "member", nq, 4096, ("member", "brisc"), ("member", "trisc"), nq, nq),
            CB("K", "member", self.depth * 4 * kc, 2048, ("streamer", "brisc"), ("member", "trisc"), 4 * kc, 4 * kc),
            CB("SR", "member", 2 * kc, 4096, ("member", "trisc.pack"), ("member", "trisc.unpack"), kc, kc),
            CB("SC", "member", 2 * kc, 4096, ("member", "trisc.pack"), ("member", "dm.filter"), kc, kc),
        ]
        _ = s
        return self

    def check(self) -> dict:
        prod, cons, bad = {}, {}, []
        for c in self.cbs:
            prod.setdefault((c.role, c.name), set()).add(c.producer)
            cons.setdefault((c.role, c.name), set()).add(c.consumer)
            if c.pages % c.push or c.pages % c.wait:
                bad.append(f"{c.name}: push {c.push} / wait {c.wait} do not tile {c.pages} pages")
        bad += [f"{k}: {len(v)} producers" for k, v in prod.items() if len(v) != 1]
        bad += [f"{k}: {len(v)} consumers" for k, v in cons.items() if len(v) != 1]
        cb_bytes = sum(c.pages * c.page_bytes for c in self.cbs)
        cand_bytes = self.nq * self.cand * 8  # uint32 key + uint32 pool id per candidate (DM scratch, not a CB)
        l1 = cb_bytes + cand_bytes
        if l1 > 900 * 1024:
            bad.append(f"L1 {l1} B over the 900 KB phase budget")
        return dict(ok=not bad, problems=bad, cb_bytes=cb_bytes, cand_bytes=cand_bytes, l1_bytes=l1)

    def simulate(self, pools: int) -> dict:
        """one member over the window: per block the streamer's delivery, the compute, the filter (event times in cycles)"""
        nq, kc = self.nq, self.kc
        nkt = -(-pools // 32)
        nblk = -(-nkt // kc)
        mm = COST["mm"][self.fid]
        # streamer: a block = kc key tiles, read once per chip from DRAM and multicast; its delivery rate is the
        # slower of the DRAM share and the multicast write into each member (one streamer pair serves every member)
        blk_bytes = kc * KT_BYTES
        t_deliver = blk_bytes / min(COST["dram_gbs"], COST["mcast_gbs"]) * GHZ  # bytes / (GB/s) * GHz = cycles
        # compute per block and resident query: kc S tiles (4 K-steps each) packed with the packer's ReLU into CB_SR, then
        # the kc weight matmuls wsc_q @ S_q packed onto CB_SC with the packer's L1 accumulation (query 0 overwrites):
        # full-sync DST cannot keep score accumulators across the S pack. S fp32 (weights at HiFi4, tf32 operands) or
        # S bf16 (the op's own precision class: it accumulates in bf16 DEST; weights at HiFi2); 2 mode switches per query
        sp = COST["pack"] if self.s_dtype == "fp32" else COST["pack"] // 2
        mw = COST["mm"]["HiFi4"] if self.s_dtype == "fp32" else COST["mm"]["HiFi2"]
        t_comp = nq * (4 * kc * mm + kc * sp + kc * mw + kc * COST["pack"] + 2 * COST["mode"])
        # filter per block: nq rows x 32 kc scores compared against the threshold, split over BRISC + NCRISC; with skip, a
        # row's block max first (kc tiles' row max on the SFPU: + nq kc / 32 tiles of work) and only rows whose max passes
        # are scanned (P(a row block holds a candidate) ~ min(1, 32 kc x 512 / pools seen); its mean over the window)
        import math as _m

        if self.skip:
            nkt_b = 32 * kc
            frac = sum(min(1.0, nkt_b * 512 / max(1, (b + 1) * nkt_b)) for b in range(nblk)) / nblk
            t_filt = nq * (COST["cmp"] + frac * 32 * kc * COST["cmp"]) / 2
            t_comp += nq * kc * 100 // 32 + COST["pack"]
        else:
            t_filt = nq * 32 * kc * COST["cmp"] / 2
        # radix selects: a query's buffer fills with ~(cand - 512) accepted scores between selects; acceptance falls as
        # 512 / seen (a running top-k over a random order accepts ~512 ln(n / 512) in total)
        import math

        acc = 512 * (1 + math.log(max(pools, 512) / 512))
        selects = max(0.0, (acc - 512) / (self.cand - 512))
        t_radix_total = nq * (selects + 1) * self.cand * 8 * COST["radix"] / 2
        # pipeline: CB_K depth decouples the streamer, CB_SC (2 kc) decouples the filter
        t = dict(stream=nblk * t_deliver, compute=nblk * t_comp, filter=nblk * t_filt + t_radix_total)
        bound = max(t, key=t.get)
        span = max(t.values()) + t_deliver + t_comp + t_filt  # fill + drain of one block
        return dict(pools=pools, nkt=nkt, blocks=nblk, nq=nq, span_us=round(span / GHZ / 1000, 1), bound=bound,
                    stream_us=round(t["stream"] / GHZ / 1000, 1), compute_us=round(t["compute"] / GHZ / 1000, 1),
                    filter_us=round(t["filter"] / GHZ / 1000, 1), radix_selects=round(selects, 2),
                    dram_mib=round(nkt * KT_BYTES / 2**20, 1))  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = []
    for tp, r in (("TP2 pair", 1024), ("TP8", 256)):
        for fid, sd, sk in (("HiFi4", "fp32", False), ("HiFi2", "fp32", False), ("HiFi2", "bf16", True)):
            p = Plan(rows=r, fid=fid, s_dtype=sd, skip=sk).build()
            fid = f"{fid}/S {sd}/skip {sk}"
            chk = p.check()
            print(f"== {tp}: rows {r}, nq {p.nq} per member, kc {p.kc}, {fid}; CB check ok={chk['ok']} {chk['problems']} "
                  f"CB {chk['cb_bytes']} B + candidates {chk['cand_bytes']} B = {chk['l1_bytes']} B")
            for w in (65536, 131072, 524288, 1048576):
                s = p.simulate(w // 4)
                s.update(tp=tp, fid=fid, window=w, cb_ok=chk["ok"])
                rows.append(s)
                print("  window %8d: pools %7d  span %8.1f us  bound %-7s (stream %7.1f / compute %7.1f / filter %7.1f us; "
                      "radix selects/query %.2f; keys %.1f MiB)" % (w, s["pools"], s["span_us"], s["bound"], s["stream_us"],
                                                               s["compute_us"], s["filter_us"], s["radix_selects"], s["dram_mib"]))
    if a.json:
        with open(a.json, "w") as fh:
            for r_ in rows:
                fh.write(json.dumps(r_) + "\n")


if __name__ == "__main__":
    main()
