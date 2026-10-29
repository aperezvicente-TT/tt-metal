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
