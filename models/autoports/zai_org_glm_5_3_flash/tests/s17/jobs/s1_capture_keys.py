# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK — a FRESH mock process: the build must happen inside this job): Stage 17 capture-key check.
A device trace capture may only launch programs the warm-up already compiled ("not yet in program cache" otherwise —
e.g. a builder that reads layer-kind knobs from the environment at call time). The mock captures eagerly, so it
cannot see that by itself: this job hooks ttnn.generic_op BEFORE the build
(StackBackend.prepare_decode compiles the spec step + bootstrap), keeps every program key of the build, then runs the
capture order (step, bootstrap) from several entry environments (as left by the build, then after each layer kind's
knobs) and fails on any launch whose key the build did not produce. The PLAIN decode captures too: a mock's
capture_decode records a tuple without running the step, so the job replays the plain step of every decode width
(what StackBackend._capture runs) after the spec build ("plain:w<width>" rows).
ARGS: layers, b, k, narrow, dbg_ids (0), profile, env ({knob: value}), max_seq_len (2048), tag."""
import importlib
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402


def pd_key(pd):
    """the program cache's structural key (runtime args / addresses excluded) of a (mesh) program descriptor"""
    if isinstance(pd, ttnn.ProgramDescriptor):
        return ("p", int(ttnn.compute_program_descriptor_hash(pd)))
    parts = []
    for item in pd:
        r, p = item if isinstance(item, tuple) else (item, pd[item])
        parts.append((str(r), int(ttnn.compute_program_descriptor_hash(p))))
    return ("m", tuple(sorted(parts)))


def site():
    import inspect

    out = []
    for fr in inspect.stack()[2:14]:
        b = os.path.basename(fr.filename)
        if b.endswith(".py") and b not in ("decorators.py", "s1_capture_keys.py"):
            out.append(f"{b}:{fr.function}:{fr.lineno}")
    return " < ".join(out[:3])


orig, log = ttnn.generic_op, []


def hooked(io, pd):
    log.append((pd_key(pd), site()))
    return orig(io, pd)


cfg = hf_config()
_L = str(args.get("layers", "0,1,2,3,4"))
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L.split(",")]
B, K = int(args.get("b", 4)), int(args.get("k", 3))
fresh = not any(isinstance(v, dict) and "backend" in v for v in ctx.values())
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}  # knobs for this backend's build (e.g. the served set)
ttnn.generic_op = hooked
try:
    key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow=str(args.get("narrow", "none")), profile=str(args.get("profile", "auto")),
                             extra_env=dict(XENV, **({"GLM53_SPEC_DBG_IDS": "1"} if int(args.get("dbg_ids", 0)) else {})) or None,
                             max_seq_len=int(args.get("max_seq_len", 2048)))  # fmt: skip
    build = {k for k, _ in log}
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import set_knobs

    sp, bad, n = be.spec, [], 0
    kv = be._kv
    for w in sorted(kv.extra.get("decode_traces", {})) or [None]:  # the plain capture order after the spec build
        log.clear()
        be._step(kv, None, w)
        n += len(log)
        bad += [dict(entry="as-built", run=f"plain:w{w}", site=s) for k, s in log if k not in build]
    for entry in ("as-built", False, True, "dsa"):
        if entry != "as-built":
            set_knobs(entry, be.tp, 1)
        for name, fn in (("step", sp._step), ("boot", sp._boot)):
            log.clear()
            fn()
            n += len(log)
            bad += [dict(entry=str(entry), run=name, site=s) for k, s in log if k not in build]
finally:
    ttnn.generic_op = orig
ok = fresh and not bad
S.record("tests", dict(test="s1-mock-capture-keys", milestone="S1", status="pass" if ok else "fail", K=K, B=B, layers=_L,
                       tp=be.tp, fresh_build=fresh, build_keys=len(build), launches_checked=n, new_keys=len(bad),
                       first=bad[:6], env=XENV, tag=args.get("tag")))  # fmt: skip
print(f"S17-KEYS fresh {fresh} build keys {len(build)} checked {n} new {len(bad)} {bad[:6]}", flush=True)
RESULT = dict(ok=ok, new=len(bad))
