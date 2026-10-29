# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host mock (CPU, torch only; no device, no ttnn): the sharded DSA step (tt/dsa_shard.py) vs the replicated one on
random data. Per (L, B, C, ties) case: the selected pool set bitwise vs the replicated exact top-512 (no ties: the two
tie orders cannot differ) and bitwise vs the exact top-512 under the sharded tie rule (with deliberate ties), the
threshold key bitwise, the exact-merge invariant (every globally selected pool is in its owner's local top-512), and the
attention output of the split-K partials merged across chips vs the replicated attention: PCC / max |d| with the peer
partials crossing as fp32 (merge only), bf16 (1 term), bf16 hi + lo (2 terms, the v0 payload) and 3 terms (exact).
Also the per-chip byte budget at 1M (TP2 / TP8). Writes one JSON row per case to --out (default
$GLM53_OUT_DIR/dsa_shard_ref.jsonl) and a summary line. Bars: sel bitwise, T bitwise, invariant, PCC(2 terms) >= 0.9999.
usage: python dsa_shard_ref.py [--quick] [--out path]"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
import torch  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS  # noqa: E402

BAR = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="L = 4096 / 65536 only, B = 1 / 4")
    ap.add_argument("--out", default=os.path.join(os.environ.get("GLM53_OUT_DIR", "/tmp"), "dsa_shard_ref.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Ls = (4096, 65536) if a.quick else (4096, 65536, 1048576)
    Bs = (1, 4) if a.quick else (1, 32)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fails = 0
    with open(a.out, "a") as f:
        for L in Ls:
            for B in Bs:
                for C in (2, 8):
                    for ties in (0, 64):
                        t0 = time.time()
                        r = DS.reference_case(L, B, C, ties=ties, seed=a.seed)
                        ok = (
                            r["T_bitwise"]
                            and r["local_union_holds"]
                            and r["sel_bitwise_sharded_rule"]
                            and (r["sel_bitwise"] or ties > 0)
                            and r["pcc_min"][2] >= BAR
                        )
                        r.update(
                            status="pass" if ok else "fail",
                            seconds=round(time.time() - t0, 1),
                            bar=BAR,
                            time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            test="dsa_shard_ref",
                        )
                        r["pcc_min"] = {str(k): round(v, 8) for k, v in r["pcc_min"].items()}
                        r["maxabs"] = {str(k): float(f"{v:.3e}") for k, v in r["maxabs"].items()}
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        fails += not ok
                        print(f"REF L={L} B={B} C={C} ties={ties}: {r['status']} sel_bitwise={r['sel_bitwise']} "
                              f"sel_rule={r['sel_bitwise_sharded_rule']} T={r['T_bitwise']} inv={r['local_union_holds']} "
                              f"pcc(fp32)={r['pcc_merge_fp32']:.8f} pcc(1/2/3 terms)={r['pcc_min']['1']}/"
                              f"{r['pcc_min']['2']}/{r['pcc_min']['3']} maxabs(2)={r['maxabs']['2']} "
                              f"{r['seconds']}s", flush=True)  # fmt: skip
        for C in (2, 8):
            for users in (1, 32):
                b = DS.budget(1048576, C, users=users)
                row = dict(test="dsa_shard_budget", L=b["L"], C=C, users=users,
                           replicated_gib={k: DS.gib(v) for k, v in b["replicated"].items()},
                           sharded_gib={k: DS.gib(v) for k, v in b["sharded"].items()},
                           time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))  # fmt: skip
                f.write(json.dumps(row) + "\n")
                print(f"BUDGET 1M C={C} users={users}: replicated {row['replicated_gib']} sharded {row['sharded_gib']}")
    print(f"DSA_SHARD_REF_DONE fails={fails} out={a.out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
