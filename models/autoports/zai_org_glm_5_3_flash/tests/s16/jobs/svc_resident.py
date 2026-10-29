# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (mock / pair rehearsal of 14b's shared mesh service): build the resident serving backend exactly as 14b's
m5common.serving does (CTX[('m5_serving', B, ids, ndev, native, rev)]), so s16_service_prefill.py can attach to it.

ARGS: B (32), layers ([0, 1, 2, 3, 4]), native (True), max_seq_len (2048)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "s14b"))
import m5common as M5  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

args, ctx, mesh = ARGS, CTX, MESH  # noqa: F821
key = M5.serving(ctx, mesh, SM, G, B=int(args.get("B", 32)), ids=[int(v) for v in args.get("layers", [0, 1, 2, 3, 4])],
                 cfg=hf_config(), native=bool(args.get("native", True)),
                 max_seq_len=int(args.get("max_seq_len", 2048)), log=print)  # fmt: skip
RESULT = {"key": str(key)}
