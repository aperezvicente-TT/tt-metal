# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c host tool (stdlib only): the README tables from the evidence rows.
  tables  <perf.jsonl> <tag> [B ...]   -> the phase table of the probe rows with that tag (one column per B, latest row)
  lines   <perf.jsonl> <tag>            -> one line per row (config -> traced steady us / layer)
  numerics <pcc.jsonl> <tag>            -> one line per numerics row (PCC vs Stage 3 / golden, phase compare)
"""
import json
import sys


def rows(path, tag):
    out = []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("tag") == tag:
                out.append(r)
    return out


def knobs(r):
    k = dict(r.get("knobs") or {})
    for x in ("GLM53_MK_READERS", "GLM53_MK_KB_QKV", "GLM53_MK_RING_KB_QKV"):
        k.pop(x, None)
    short = {"GLM53_REC_UGROUPS": "ug", "GLM53_REC_MODE": "rec", "GLM53_MHC_MODE": "mhc", "GLM53_KDA_MODE": "kda",
             "GLM53_MK_DEFER_INV": "defer", "GLM53_REC_EARLY_SIGNAL": "early", "GLM53_MK_W_DELAY_US": "wdelay",
             "GLM53_MHC_POST_ROWWRITE": "rowwrite", "GLM53_MHC_PRE_EARLY": "preearly", "GLM53_MK_FOLD_Y2": "fold"}
    return " ".join(f"{short.get(a, a)}={b}" for a, b in sorted(k.items())) or "-"


def tables(path, tag, bs):
    sel = {}
    for r in rows(path, tag):
        if isinstance(r.get("phase_table"), list):
            sel[r["B"]] = r
    bs = [int(b) for b in bs] if bs else sorted(sel)
    names = [n for n, *_ in sel[bs[0]]["phase_table"]]
    print("| phase (us) | " + " | ".join(f"B={b}" for b in bs) + " |")
    print("|---|" + "---:|" * len(bs))
    for i, n in enumerate(names):
        vals = []
        for b in bs:
            t = sel[b]["phase_table"]
            vals.append(f"{t[i][3]:.1f}" if i < len(t) and t[i][0] == n else "-")
        print(f"| {n} | " + " | ".join(vals) + " |")
    print("| **traced steady us / layer** | " + " | ".join(
        f"**{(sel[b].get('chain') or {}).get('steady_per_layer_us', sel[b]['us'])}**" for b in bs) + " |")


def lines(path, tag):
    for r in rows(path, tag):
        c = r.get("chain") or {}
        print(f"B={r['B']:>2} rec_spread={r.get('rec_spread')} [{knobs(r)}]: steady {c.get('steady_per_layer_us')} us / "
              f"layer (chain-{c.get('n')} mean {c.get('mean_per_layer_us')}, single launch "
              f"{(r.get('single_launch') or {}).get('median_us')})")


def numerics(path, tag):
    for r in rows(path, tag):
        pc = r.get("phase_cmp") or {}
        worst = min(((k, v["pcc_min"]) for k, v in pc.items()), key=lambda kv: kv[1], default=None)
        bit = sorted(k for k, v in pc.items() if v.get("bitwise"))
        print(f"{r['test']} [{knobs(r)}] status={r['status']} vs Stage 3 {r.get('pcc_vs_stage3')} vs golden "
              f"{r.get('pcc_vs_golden')} state {r.get('state_pcc_vs_stage3')} hist bitwise {r.get('conv_history_bitwise')}"
              + (f" | phase cmp vs {r.get('phase_cmp_vs')}: worst {worst}, bitwise {bit}" if pc else "")
              + (f" | soak {r['soak']}" if r.get("soak") else ""))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tables":
        tables(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "lines":
        lines(sys.argv[2], sys.argv[3])
    elif cmd == "numerics":
        numerics(sys.argv[2], sys.argv[3])
