# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b sweep-in-job (program JOB PIPELINE RULE 2026-10-19T23:52:53Z): ONE job loops over MkMla configurations
on the process's resident state (tests/s7b/mla_common.py resident(): the layer, its decoder weights, every config's
MkMla object with its re-laid-out weights stay in CTX across jobs) and writes one perf row per (config, t): the traced
median of the one-program half (single replay; GLM53_SWEEP_STEADY=1: also the half twice in one trace = per program with
the next launch's dispatch overlapped, as in a decode loop), its in-program span (cycle probes, GLM53_SWEEP_PROBE=1) and
the config's own cost
(build + compile/warm + capture + replays, seconds) so the pipeline's per-config median is on record.

GLM53_SWEEP = configs separated by ';', each 'key=value' pairs separated by ' ' (colon-separated lists):
  rings=210:350:128:350   kb=8:4:2:4:16   look=0|1   qfold=0|1 (default 1)   qn=6 (q-norm cores, qfold=0)
  place=MESH_ROW_PLUS_X|ROW_PLUS_X|NOC_HOPS (the DS readers' placement)
GLM53_MK_TS (default 1), GLM53_MK_REPLAYS (default 30)."""

from __future__ import annotations

import os
import sys
import time
import traceback

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b import reload_deps

reload_deps.refresh()  # helper modules fresh per job (serve.py reloads only the job module)
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import Rows, aiclk_mhz, layer_of, resident, timed
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK

LAYER = int(os.environ.get("GLM53_S7B_LAYER", "3"))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1").split(",")]
REPLAYS = int(os.environ.get("GLM53_MK_REPLAYS", "30"))
PROBE = os.environ.get("GLM53_SWEEP_PROBE", "1") == "1"
STEADY = os.environ.get("GLM53_SWEEP_STEADY", "1") == "1"


def configs() -> list:
    out = []
    for spec in os.environ.get("GLM53_SWEEP", "look=1").split(";"):
        c = {}
        for kv in spec.split():
            k, v = kv.split("=")
            c[k] = tuple(int(x) for x in v.split(":")) if ":" in v else (int(v) if v.lstrip("-").isdigit() else v)
        out.append(c)
    return out


def build(mesh, L, c: dict, probe: bool):
    kw = {}
    if "rings" in c:
        kw["rings"] = c["rings"]
    if "kb" in c:
        kw["kb"] = c["kb"]
    if "look" in c:
        kw["lookahead"] = bool(c["look"])
    if "ukv" in c:  # W_uk / W_uv bfloat8_b through the shared bfp8 ring
        kw["ukv_bf8"] = bool(c["ukv"])
    if "qn" in c:
        os.environ["GLM53_MLA_QN"] = str(c["qn"])
    os.environ["GLM53_MLA_QFOLD"] = str(c.get("qfold", 1))
    if "place" in c:
        os.environ["GLM53_MLA_PLACEMENT"] = c["place"]
    else:
        os.environ.pop("GLM53_MLA_PLACEMENT", None)
    return MK.MkMla(mesh, L.a_args, L.attn, probe=probe, **kw)


def main(mesh=None, ctx=None):
    own = mesh is None
    if own:
        wl1 = int(os.environ.get("GLM53_WORKER_L1", "1264512"))
        mesh = (ttnn.open_device(device_id=0, worker_l1_size=wl1) if os.environ.get("GLM53_MOCK") == "1" else
                ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000, worker_l1_size=wl1))  # fmt: skip
    rows = Rows(LAYER)
    tr = Traced(mesh)
    qn0, qf0 = os.environ.get("GLM53_MLA_QN"), os.environ.get("GLM53_MLA_QFOLD")
    try:
        t_job = time.perf_counter()
        L = layer_of(mesh, ctx, LAYER)
        setup_s = round(time.perf_counter() - t_job, 2)
        costs = []
        for c in configs():
            for t in TS:
                t0 = time.perf_counter()
                key = tuple(sorted(c.items()))
                try:
                    mk = resident(ctx, ("MkMla", LAYER, key), lambda: build(mesh, L, c, False))
                    cm = L.case(t, tag="sweep")
                    fn = lambda mk=mk, cm=cm, t=t: mk(cm["x"], cm["cache"].latent, cm["pt"], cm["cur"], t=t)
                    tr.warm(fn)
                    tr.run(fn)
                    us = timed(tr, REPLAYS)
                    tr.release()
                    timing = {"mk_one": us}
                    if STEADY:  # the half twice in one trace: per program, dispatch overlapped as in a decode loop
                        f2 = lambda fn=fn: (fn(), fn())[1]
                        tr.run(f2)
                        us2 = timed(tr, REPLAYS)
                        tr.release()
                        timing["mk_one_x2"] = us2
                        timing["mk_one_steady"] = round(us2 / 2, 1) if us2 else None
                    r = dict(t=t, execution=tr.mode, variant="mk_mla_one_sweep", config={k: list(v) if isinstance(v, tuple)
                             else v for k, v in c.items()}, timing_us=timing, rev=MK.MLA_REV)  # fmt: skip
                    if PROBE:
                        mkp = resident(ctx, ("MkMlaProbe", LAYER, key), lambda: build(mesh, L, c, True))
                        fp = lambda mkp=mkp, cm=cm, t=t: mkp(cm["x"], cm["cache"].latent, cm["pt"], cm["cur"], t=t)
                        tr.warm(fp)
                        tr.run(fp)
                        tr.release()
                        p = MK.decode_probe(ttnn.to_torch(mkp.probe), len(mkp.last_readers.assign), chunks=mkp.chunks,
                                            groups=mkp.groups)  # fmt: skip
                        ends = [v[-1] for k_, v in p.items() if k_.endswith("_max") and v]
                        span = max(ends) if ends else None
                        r.update(in_program_span_us=span, probe=p, aiclk_mhz=aiclk_mhz())
                    r["config_cost_s"] = round(time.perf_counter() - t0, 2)
                    costs.append(r["config_cost_s"])
                    rows("mk_mla_sweep", status="info", **r)
                except Exception as e:  # noqa: BLE001 -- recorded, the config dropped
                    traceback.print_exc()
                    msg = str(e)
                    l1 = "clash with L1" in msg or "beyond max L1" in msg
                    rows("mk_mla_sweep", t=t, status="info" if l1 else "fail", l1_reject=l1, error=msg[:600],
                         config={k: list(v) if isinstance(v, tuple) else v for k, v in c.items()},
                         config_cost_s=round(time.perf_counter() - t0, 2))  # fmt: skip
                    tr.release()
        if costs:
            s_ = sorted(costs)
            rows("pipeline_cost", status="info", setup_s=setup_s, per_config_median_s=s_[len(s_) // 2],
                 per_config_s=costs, configs=len(costs))  # fmt: skip
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rows("mk_mla_sweep", status="fail", error=traceback.format_exc()[-800:])
    finally:
        for k_, v_ in (("GLM53_MLA_QN", qn0), ("GLM53_MLA_QFOLD", qf0)):
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
        rows.flush()
        if own:
            ttnn.close_device(mesh) if os.environ.get("GLM53_MOCK") == "1" else ttnn.close_mesh_device(mesh)
    bad = rows.failed()
    print(f"DEV_MK_SWEEP_DONE rows={len(rows.rows)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
