# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 HOST MOCK; Stage 14b, operator rulings 2026-10-23 15:28Z / 15:34Z): THE 45-LAYER SERVED PMK
COMPOSITE -- every decoder layer of the model (31 sparse_kda + 3 dense_kda + 11 sparse_dsa) under GLM53_PF_MODE=pmk
through pmk_stack_serve_mock.py's flow (tt/pmk_stack.ServedLayer per layer on the bucket's persistent residual: every
piece program JIT-built and launched through TPPrefill.forward, the strict L1 check, the launch census, one capture key
per PMK bucket, the op forward after detach), plus the 45-layer summary this job adds: programs / op fillers per
replay by kind, the transient hand-off sets (one per bucket and kind group, owned by the group's first layer), the
DRAM per bank after the backend and after the chain, the trace-region inputs.

MOCK WEIGHTS ALIASED PER KIND (alias, default on under GLM53_MOCK=1): layer L reads the tile-cache entries and the
permuted experts of its kind's FIRST layer (TileCacheSource._entry + the permuted-expert cache dirs symlinked to the
first layer's). The composite's programs, CB maps, census, launch count and DRAM sizes do not depend on the weight
values; the host reads 3 layers' files instead of 45 and writes ~5 GB of permuted experts instead of ~114 GB (the
2026-10-20 root-disk fill). Every layer still allocates its own device tensors (the DRAM numbers are the real ones).
Numerics are out of scope here (the TP8 row: pmk_stack_serve_unit.py on the device).

ARGS: pmk_stack_serve_mock.py's (layers default = every layer), alias (True), record (True)."""
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
args = dict(ARGS)  # noqa: F821
mesh, ctx = MESH, CTX  # noqa: F821
t_job = time.time()

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

cfg = hf_config()
NL = int(cfg.num_hidden_layers)
args.setdefault("layers", list(range(NL)))
args.setdefault("tag", f"mock-pmk-stack45-L{args['layers'][0]}-{args['layers'][-1]}-2x4")
REC = bool(args.pop("record", True))
ALIAS = bool(args.pop("alias", True)) and os.environ.get("GLM53_MOCK") == "1"
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(HERE))
import pmk45_alias as A45  # noqa: E402

alias_info = A45.install(cfg) if ALIAS else dict(on=False)
SHARE = bool(args.pop("share", False))  # the pair: the layers of a kind share the device weights (pmk45_alias.install_device_share)
alias_info.update(A45.install_device_share(cfg, ctx) if SHARE else dict(device_share=False))

ARGS = dict(args, record=False)  # the serve mock's own row is not recorded: this job's row is
_src = os.path.join(HERE, "pmk_stack_serve_mock.py")
exec(compile(open(_src).read(), _src, "exec"), globals())  # noqa: S102 -- the same flow, same process
base = RESULT  # noqa: F821

import ttnn  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

summ45 = {}
try:
    st_ = stack  # noqa: F821 -- the serve mock's PmkStack
    c_ = int(base["pmk_bucket"][0]) if base.get("pmk_bucket") else 2048
    per_kind = {}
    for (cc, L_), lay in sorted(st_.progs.items()):
        if cc != c_:
            continue
        k = lay.kind
        calls = max(1, int(lay.counters.get("calls", 0)))
        d = per_kind.setdefault(k, dict(layers=[], programs_per_layer=set(), fillers_per_layer=set(), pmk_slots=set(), op_slots=set()))
        d["layers"].append(L_)
        d["programs_per_layer"].add(round(lay.counters["launches"] / calls, 2))
        d["fillers_per_layer"].add(round(lay.counters["fillers"] / calls, 2))
        d["pmk_slots"].update(s for s, v in lay.modes.items() if v == "pmk")
        d["op_slots"].update(s for s, v in lay.modes.items() if v != "pmk")
    for k, d in per_kind.items():
        for f in ("programs_per_layer", "fillers_per_layer", "pmk_slots", "op_slots"):
            d[f] = sorted(d[f])
        d["n"] = len(d.pop("layers"))
    groups = {}
    for (cc, grp), owner in st_.shared.items():
        if grp == "B":  # program B's scratch owner: every layer whose B shares its scratch buffers
            key_ = ("scratch", cc)
            users = [L_ for (c2, L_), lay in st_.progs.items() if c2 == cc and lay.b is not None and lay.b.bufs.get(key_) is owner.b.bufs.get(key_)]
        else:
            users = [L_ for (c2, L_), lay in st_.progs.items() if c2 == cc and (lay is owner or lay.reuse is owner)]
        groups[f"{cc}:{grp}"] = dict(owner=owner.L, users=len(users),
                                     a_set=(None if owner.a is None else dict(hns=len(owner.a.hns), posts=len(owner.a.posts), combs=len(owner.a.combs), ys1_shared=True)),
                                     b_scratch=(None if owner.b is None else sorted(owner.b.scratch(cc))))  # fmt: skip
    a_programs = sum(d["n"] * (d["programs_per_layer"][-1] if d["programs_per_layer"] else 0) for d in per_kind.values())
    # the trace-light levers (program 17:52:10Z): the knobs this build ran with and the per-program DRAM arg tables
    # (program A: engine.arg_tables, 1024-B rows per core; program B: arg_table_report) per kind and in total, per chip
    from models.autoports.zai_org_glm_5_3_flash.tt.pmk_layer import ARG_PAGE  # noqa: E402

    tl = dict(rect_merge=os.environ.get("GLM53_PMK_RECT_MERGE", "0"), args=os.environ.get("GLM53_PMK_ARGS", "trace"),
              c_trim=os.environ.get("PMK_PFS_C_TRIM", "1"), per_kind={})  # fmt: skip
    for (cc, L_), lay in sorted(st_.progs.items()):
        if cc != c_:
            continue
        d = tl["per_kind"].setdefault(lay.kind, dict(n=0, a_tables=0, a_table_bytes=0, b_table_bytes=0))
        d["n"] += 1
        eng = getattr(getattr(lay, "a", None), "engine", None)
        tabs = (getattr(eng, "arg_tables", None) or {}) if eng is not None else {}
        d["a_tables"] += len(tabs)
        d["a_table_bytes"] += sum(int(v["nrows"]) * ARG_PAGE for v in tabs.values())
        b_ = getattr(lay, "b", None)
        rep = getattr(b_, "arg_table_report", None) or getattr(getattr(b_, "prog", None), "arg_table_report", None)
        d["b_table_bytes"] += int((rep or {}).get("bytes", 0))
    tl["total_table_bytes_per_chip"] = sum(d["a_table_bytes"] + d["b_table_bytes"] for d in tl["per_kind"].values())
    dram_ab = {L_: v.get("dram_mb_ab") for L_, v in info["layers"].items()}  # noqa: F821
    summ45 = dict(per_kind=per_kind, trace_light=tl, dram_mb_ab_by_layer=dram_ab, kind_groups=groups, launches_per_replay=base.get("programs_per_replay"),
                  counted_device_programs_per_replay=a_programs, alias=alias_info,
                  dram_after_backend=(base.get("dram") or {}).get("after_backend"), dram_after_chain=(base.get("dram") or {}).get("after_chain_build"),
                  dram_after_compile=(base.get("dram") or {}).get("after_compile"),
                  trace_region_inputs=dict(launches=a_programs, fillers=sum(d["n"] * (d["fillers_per_layer"][-1] if d["fillers_per_layer"] else 0) for d in per_kind.values()),
                                           note="trace bytes are a device measurement (the mock captures nothing): the TP8 row reads them"))  # fmt: skip
except Exception:  # noqa: BLE001
    import traceback

    summ45 = dict(error=traceback.format_exc()[-3000:])
row45 = dict(base)
row45.update(test=f"PMK 45-layer served composite mock (2x4, weights aliased per kind: {ALIAS}): {len(args['layers'])} layers "
                  f"under GLM53_PF_MODE=pmk -- every piece program JIT-built + launched, strict L1 check, launch census, "
                  f"one capture key, the op forward after detach",  # fmt: skip
             milestone="14b-PMK-45", summ45=summ45, tag=args["tag"], wall_job_s=round(time.time() - t_job, 1))
if summ45.get("error") and row45.get("status") == "pass":
    row45["status"] = "fail"
H.log({k: v for k, v in row45.items() if k in ("status", "summ45", "launch_census_max", "fits", "programs_per_replay", "wall_job_s")})  # noqa: F821
if REC:
    C.record("tests", row45)  # noqa: F821
RESULT = row45
