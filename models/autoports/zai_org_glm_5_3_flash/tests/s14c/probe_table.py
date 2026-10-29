# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tool (stdlib only): an m3_tp.py probe job's .out (RESULT["probes"]) -> the per-phase timeline of the TP layer
program on one chip, as markdown rows. The marks are the program's own wall clock (us after its earliest mark):
readers (activation mover: per phase (inputs ready, partial written); weight streamer: phase done), owner / root
marks (decode_probe medians), the fabric lanes (lane BRISC: launch start, inputs ready, round 1 sent, round 2 sent,
result done, launch end, 2nd all-reduce inputs ready / result done).
usage: probe_table.py <job.out> [chip]"""
import json
import sys

PHASES = ["qkv", "low-rank", "o_proj", "gate_up", "down"]


def table(p: dict) -> list:
    mk, lane = p["mk"], p["lane_brisc"][0]
    a, w = mk["reader0_a"], mk["reader0_w"]
    rows = [("mhc_pre (attention): hn ready", 0.0, a[1])]
    for i, name in enumerate(PHASES):
        rows.append((f"{name}: inputs ready -> partial written (weights streamed by {w[1 + i]})", a[1 + 2 * i],
                     a[2 + 2 * i]))  # fmt: skip
    rows.insert(3, ("kda_pre + kda_rec (low-rank written -> o_proj inputs ready)", a[4], a[5]))
    rows.insert(5, ("all-reduce 1 (o_proj partial): lane inputs ready -> result", lane[1], lane[4]))
    rows.insert(6, ("mhc_post + FFN mhc_pre (all-reduce 1 result -> gate_up inputs ready)", lane[4], a[7]))
    rows.insert(8, ("swiglu (gate_up written -> down inputs ready)", a[8], a[9]))
    rows.append(("all-reduce 2 (down partial): lane inputs ready -> result", lane[6], lane[7]))
    rows.append(("mhc_post (FFN) -> layer output written", lane[7], mk["owner_ncrisc_leaders"][-1]))
    return rows


def main():
    res = json.load(open(sys.argv[1]))["result"]
    chip = sys.argv[2] if len(sys.argv) > 2 else "0"
    p = res["probes"][chip]
    print("| phase | start us | end us | us |\n|---|---:|---:|---:|")
    for name, t0, t1 in table(p):
        print(f"| {name} | {t0:.1f} | {t1:.1f} | {t1 - t0:.1f} |")
    print(f"\nspan: layer output written at {p['mk']['owner_ncrisc_leaders'][-1]:.1f} us after the first mark")


if __name__ == "__main__":
    main()



def phase_rows(p: dict, rec_cores: int) -> list:
    """Stage 14c: the dense layer's phase table as [phase, start us, end us, us] rows (one chip). Reader 0's activation
    mover marks a[1..10] = (inputs ready, partial written) of qkv, low-rank, o_proj, gate_up, down; the fabric lanes
    (absent on a 1-chip run: no all-reduce) give the all-reduce intervals; "kda_pre + kda_rec" is split by the owner
    NCRISC marks — kda_pre written = the LAST rec core's 2nd mark (every kda_rec head waits for its kda_pre producers,
    incl. core 0's beta tile for all heads; a median would mix in the owners without kda_pre work on a TP8 shard and
    move kda_pre time into kda_rec), kda_rec written = the LAST
    rec core's 5th-from-last mark (every owner's NCRISC list ends with mhc_post written, FFN mhc_pre done, m written,
    layer output written)."""
    a = p["mk"]["reader0_a"]
    lane = p["lane_brisc"][0] if p.get("lane_brisc") else None
    span = p["mk"]["owner_ncrisc_leaders"][-1]
    r = [("mhc_pre (attention)", 0.0, a[1]), ("qkv", a[1], a[2]), ("low-rank", a[3], a[4])]
    own = [m for m in p.get("owner_ncrisc_all", [])[:rec_cores] if len(m) >= 7]
    if own:
        kp, kr = max(m[1] for m in own), max(m[-5] for m in own)
        r += [("kda_pre", a[4], kp), ("kda_rec", kp, kr), ("kda_rec -> o_proj hand-off", kr, a[5])]
    else:
        r.append(("kda_pre + kda_rec", a[4], a[5]))
    r.append(("o_proj", a[5], a[6]))
    if lane:
        r += [("all-reduce 1", lane[1], lane[4]), ("mhc_post + FFN mhc_pre", lane[4], a[7])]
    else:
        r.append(("mhc_post + FFN mhc_pre", a[6], a[7]))
    r += [("gate_up", a[7], a[8]), ("swiglu", a[8], a[9]), ("down", a[9], a[10])]
    if lane:
        r += [("all-reduce 2", lane[6], lane[7]), ("mhc_post (FFN)", lane[7], span)]
    else:
        r.append(("mhc_post (FFN)", a[10], span))
    r.append(("span (layer output written)", 0.0, span))
    return [[n, round(t0, 1), round(t1, 1), round(t1 - t0, 1)] for n, t0, t1 in r]


def sparse_rows(p: dict) -> list:
    """Stage 14c: MkSparseLayer (fold_y2) phase table from reader 0's activation-mover marks (reader_moe_a.cpp: start,
    then (inputs ready, partial written) of qkv, low-rank, o_proj, router, shared gate_up, shared down, then the routed
    experts phase (xa.cpp: start, routing list ready, last expert done; the 16-mark probe drops its output mark)), the
    owners' layer-output mark as the span."""
    a = p["mk"]["reader0_a"]
    span = p["mk"]["owner_ncrisc_leaders"][-1]
    r = [("mhc_pre (attention)", 0.0, a[1]), ("qkv", a[1], a[2]), ("low-rank", a[3], a[4]),
         ("kda_pre + kda_rec", a[4], a[5]), ("o_proj", a[5], a[6]), ("mhc_post + FFN mhc_pre", a[6], a[7]),
         ("router", a[7], a[8]), ("router tail -> shared gate_up inputs", a[8], a[9]), ("shared gate_up", a[9], a[10]),
         ("shared swiglu", a[10], a[11]), ("shared down", a[11], a[12])]
    if len(a) > 15:
        r += [("routed experts: routing list", a[13], a[14]), ("routed experts: stream + compute", a[14], a[15]),
              ("FFN mhc_post (+ expert output hand-off)", a[15], span)]
    r.append(("span (layer output written)", 0.0, span))
    return [[n, round(t0, 1), round(t1, 1), round(t1 - t0, 1)] for n, t0, t1 in r]
