# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, MkMla native t): per-role in-program cycle probes (kernels/mk/probe.hpp marks,
tt/mk_mla.py decode_probe) of TPMla(probe=True) on the open mesh at ARGS["ts"] (default [32]) -- where the t = 32 MLA
half spends its time (latent core cache write vs readers vs owners' attention vs mhc_post). Layer GLM53_S7B_LAYER (3).
ARGS: ts, vtp, env (set before the modules load), tag. One traced replay per t; rows -> GLM53_STAGE_EVIDENCE pcc.jsonl."""

import importlib
import os

import ttnn

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import dev_mk_tp_mesh as TM
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

    for m in (MK, TPM, MC, TM):
        importlib.reload(m)
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced

    mesh = MESH  # noqa: F821
    ts = [int(v) for v in ARGS.get("ts", [32])]  # noqa: F821
    rows = MC.Rows(TM.LAYER)
    tr = Traced(mesh)
    L = MC.layer_of(mesh, CTX, TM.LAYER)  # noqa: F821
    vtp = ARGS.get("vtp")  # noqa: F821  (8 on a pair: the TP8 per-chip shard, timing only)
    tp = TPM.TPMla(mesh, L.a_args, L.attn, kb=TM.KB_SHARD, rings=TM.RINGS_SHARD, ts=tuple(sorted(set(ts))), probe=True,
                   vtp=vtp)  # fmt: skip
    for t in ts:
        c = L.case(t, tag="probe")
        fn = lambda c=c, t=t: tp(c["x"], c["cache"].latent, c["pt"], c["cur"], t=t, keep=("s1",))["s1"]
        tr.warm(fn)
        tr.run(fn)
        us = MC.timed(tr, 20)
        for d, raw in enumerate(ttnn.get_device_tensors(tp.mk.probe)):
            p = MK.decode_probe(ttnn.to_torch(raw), len(tp.mk.last_readers.assign))
            rows("tp_mla_probe", t=t, chip=d, status="info", timing_us={"tp_step": us}, lanes=tp.mk.att_lanes(t), vtp=vtp,
                 ot=tp.mk.ot(t), tag=ARGS.get("tag"), **p)  # noqa: F821
        tr.release()
    rows.flush()
    RESULT = dict(ok=True)
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
