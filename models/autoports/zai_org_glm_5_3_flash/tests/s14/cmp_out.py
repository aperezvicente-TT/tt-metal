# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tool: device-vs-device comparison of two m3_tp.py `save=` outputs (work-s14/m3_out/<name>.pt: the layer output
y, the new KDA state S1 and conv history C1 of the same inputs under two knob sets) -> one JSON line per pair:
bitwise flags, PCC, max-abs of each tensor. usage: cmp_out.py <base> <other> [<other> ...]"""
import json
import os
import sys

import torch

W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s14")


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def load(n):
    return torch.load(os.path.join(W, "m3_out", f"{n}.pt"), weights_only=False)


base = load(sys.argv[1])
for other in sys.argv[2:]:
    o = load(other)
    row = dict(base=sys.argv[1], other=other)
    for k in ("y", "S1", "C1"):
        a, b = base[k].float(), o[k].float()
        row[k] = dict(bitwise=bool(torch.equal(a, b)), pcc=pcc(a, b), max_abs=float((a - b).abs().max()),
                      n_diff=int((a != b).sum()))  # fmt: skip
    print(json.dumps(row))
