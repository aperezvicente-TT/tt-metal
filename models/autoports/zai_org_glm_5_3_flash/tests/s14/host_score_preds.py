# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host only (no device): score a jobs/tf_lag_preds.py predictions file against a teacher-forcing reference that was
built after the device run (the Stage 6 TokenAccuracy rule: row u's argmax at forced position i hits top-1 / top-5 /
top-100 if it is in the reference's topk_tokens[i, :k]). Checks that the reference's prompt + generated tokens are the
tokens the device consumed (sha256 prefix), then appends the scored row (pass: top-5 >= 0.98, top-100 = 1.0, traced,
every row complete) to <evidence>/pcc.jsonl.

usage: no-tt-devices python host_score_preds.py --preds P.json --ref R.refpt [--evidence DIR]"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--evidence")
    a = ap.parse_args()
    d = json.load(open(a.preds))
    r, preds = d["row"], d["preds"]
    e = torch.load(a.ref, weights_only=True)["entries"][0]
    seq = e["prompt_tokens"][0].tolist() + e["generated_tokens"][0].tolist()
    ng = int(e["generated_tokens"].shape[-1])
    topk = e["topk_tokens"]
    sha = hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]
    assert sha == r["tokens_sha256_16"] and ng == r["generated"], (sha, r["tokens_sha256_16"], ng, r["generated"])
    m = [0, 0, 0]
    for p in preds:
        for i, t in enumerate(p[:ng]):
            m[0] += int(t == int(topk[i, 0]))
            m[1] += int(t in topk[i, :5].tolist())
            m[2] += int(t in topk[i, :100].tolist())
    tot = sum(min(len(p), ng) for p in preds)
    r = dict(r, top1=round(m[0] / tot, 4), top5=round(m[1] / tot, 4), top100=round(m[2] / tot, 4), matches=m + [tot],
             reference=a.ref, scored_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             note="device run saved its argmax per forced position before the reference existed; scored on the host")  # fmt: skip
    ok = r["decode_traced_no_eager"] and r["rows_complete"] and r["top5"] >= 0.98 and r["top100"] >= 1.0
    r["status"] = "pass" if ok else "fail"
    print("SCORED ROW", json.dumps(r, default=str), flush=True)
    if a.evidence:
        with open(os.path.join(a.evidence, "pcc.jsonl"), "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
