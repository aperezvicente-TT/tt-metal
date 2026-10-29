# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tool (stdlib only): the .out files of a queue of m3_tp.py jobs -> one markdown row per job (layer, B, spreads,
PCC vs Stage 3 / golden, replicas, traced single-launch median, steady state per layer from the chain).
usage: m3_summary.py <queue dir>"""
import glob
import json
import os
import sys


def main():
    print("| job | ok | PCC vs Stage 3 | PCC vs golden | replicas | single launch us | steady us/layer | error |")
    print("|---|---|---:|---:|---|---:|---:|---|")
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.out")), key=os.path.getmtime):
        d = json.load(open(f))
        r = d.get("result") or {}
        ch = r.get("chain") or {}
        err = (d.get("error") or "").strip().splitlines()[-1][:80] if d.get("error") else ""
        fmt = lambda v, n=6: f"{v:.{n}f}" if isinstance(v, float) else str(v)  # noqa: E731
        print(f"| {d['name']} | {d['ok']} | {fmt(r.get('pcc_ref'))} | {fmt(r.get('pcc_gold'))} | {r.get('same')} | "
              f"{fmt(r.get('us'), 1)} | {fmt(ch.get('steady_per_layer_us'), 1)} | {err} |")  # fmt: skip


if __name__ == "__main__":
    main()
