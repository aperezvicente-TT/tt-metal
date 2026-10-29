# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the mesh): Stage 14b — the B=32 generator nondeterminism (#67) at the generator level, on the backend
m5c_batchinv.py built (its ctx key: ARGS rev / native must match). generate_batch(32 greedy users, short prompts, NEW
tokens) is run N times (gen.reset between). "S" runs snapshot the device inputs (chip 0's token rows + positions,
blocking reads) before EVERY replay: S1 vs S2 gives the first replay whose INPUTS differ (host logic) or none (then
the outputs differ from equal inputs: device); "N" runs are untouched (timing as served).
ARGS: b (32), layers ("all"), native (false), users (32), new (8), runs ("SSN"), patch, shared_weights, rev (q1)."""
import gc
import importlib
import os
import sys

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
NU, NEW = int(args.get("users", 32)), int(args.get("new", 8))
NATIVE = bool(args.get("native", False))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]


def log(*a):
    print(f"M5C-GDET {C.now()}", *a, flush=True)


key = ("m5c_binv", B, tuple(ids), int(mesh.get_num_devices()), NATIVE, args.get("rev", "q1"))
if args.get("serving"):  # the gate batch's shared backend (m5common.serving)
    import m5common as M5
    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM

    key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE,
                     rev=args.get("rev"), log=log)  # fmt: skip
if key not in ctx:  # its own backend (m5c_batchinv.py's ctx key: shared when that job ran with the same args)
    import importlib

    import m5common as M5
    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM

    for mod in (M5, G, SM):
        importlib.reload(mod)
    M5.prune_ctx(ctx, [key])
    M5.keep_decoders(ctx, [])
    _be = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=cfg, native_t=NATIVE,
                                 weight_cache=ctx.setdefault("s14b_weights", {}) if args.get("shared_weights")
                                 else None)  # fmt: skip
    _gen = G.build_generator(None, mesh, backend=_be, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=512,
                             stop_token_ids=())  # fmt: skip
    _gen.warmup()
    ctx[key] = dict(backend=_be, gen=_gen)
backend, gen = ctx[key]["backend"], ctx[key]["gen"]
io = backend._kv.extra["io"]
tk = load_tokenizer()
P = tk.encode("Solve: a + b = 100.")
prompts = [P + [1000 + u] for u in range(NU)]



def one(snap: bool):
    gen.reset()
    ttnn.synchronize_device(mesh)
    log_in = []
    orig = backend.replay
    if snap:
        def rp(t):
            ti, pi = backend.read(io.tokens)[:B].clone(), backend.read(io.cur_pos)[:B].clone()
            orig(t)
            log_in.append((ti, pi, backend.read(io.tokens)[:B].clone()))  # inputs, then the step's output rows

        backend.replay = rp
    try:
        out = gen.generate_batch(prompts, NEW, sampling=[GREEDY] * NU)
    finally:
        backend.replay = orig
    ttnn.synchronize_device(mesh)
    return out, log_in


PATCH = args.get("patch")  # bisect: "noseed" (the generator's per-request RNG page writes skipped: greedy rows never
# draw), "sync" (a synchronize after every replay: cross-chip timing)
_saved = {}
if PATCH == "noseed":
    _saved["_write_seeds"] = backend._write_seeds
    backend._write_seeds = lambda seeds: None
elif PATCH == "sync":
    _saved["replay"] = backend.replay
    _r0 = backend.replay

    def _rs(t):
        _r0(t)
        ttnn.synchronize_device(mesh)

    backend.replay = _rs
try:
    runs = [(m, one(m == "S")) for m in args.get("runs", "SSN")]
finally:
    for k_, v_ in _saved.items():
        setattr(backend, k_, v_)
    for k_ in _saved:
        if k_ in backend.__dict__:
            del backend.__dict__[k_]
res_patch = PATCH
base_out = runs[0][1][0]
res = dict(users=NU, new=NEW, prompt_len=len(prompts[0]), modes="".join(m for m, _ in runs), patch=PATCH)
res["outputs_equal_to_run0"] = [r[0] == base_out for _, r in runs]
res["users_differing_vs_run0"] = [sum(int(a != b) for a, b in zip(r[0], base_out)) for _, r in runs]
res["first_div_vs_run0"] = [[next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None) for a, b in zip(r[0], base_out)]
                            for _, r in runs]  # fmt: skip
s_runs = [r for m, r in runs if m == "S"]
if len(s_runs) >= 2:
    a, b = s_runs[0][1], s_runs[1][1]
    fd = None
    fo = None
    for i, ((ta, pa, oa), (tb, pb, ob)) in enumerate(zip(a, b)):
        act = (pa >= 0) | (pb >= 0)  # rows that run this replay (inactive rows' token / output rows are not read)
        dp = (pa != pb).nonzero().reshape(-1).tolist()
        dt = ((ta != tb) & act).nonzero().reshape(-1).tolist()
        do = ((oa != ob) & act).nonzero().reshape(-1).tolist()
        if fd is None and (dp or dt):
            fd = dict(replay=i, of=len(a), pos_rows=dp[:8], token_rows=dt[:8], a_pos=[int(pa[j]) for j in dp[:4]],
                      b_pos=[int(pb[j]) for j in dp[:4]], a_tok=[int(ta[j]) for j in dt[:4]],
                      b_tok=[int(tb[j]) for j in dt[:4]], active=act.nonzero().reshape(-1).tolist()[:8])  # fmt: skip
        if fo is None and do:
            fo = dict(replay=i, out_rows=do[:8], inputs_equal=not (dp or dt), a_out=[int(oa[j]) for j in do[:4]],
                      b_out=[int(ob[j]) for j in do[:4]], pos=[int(pa[j]) for j in do[:4]],
                      active=act.nonzero().reshape(-1).tolist()[:8])  # fmt: skip
        if fd is not None and fo is not None:
            break
    res["S_active_inputs_first_diff"] = fd
    res["S_active_outputs_first_diff"] = fo
    torch.save(dict(a=a, b=b), os.path.join(os.environ.get("GLM53_WORK", "/opt/glm53/work-s14b"),
                                            f"gendet_{args.get('tag', 'q')}.pt"))  # fmt: skip
    res["S_replays"] = [len(a), len(b)]
log("RESULT", res)
ok = all(res["outputs_equal_to_run0"])
r = dict(test=f"m5c generator determinism generate_batch {NU} greedy users x {NEW} tokens ({len(ids)} layers, B={B}"
              f"{' native-t' if NATIVE else ''}), runs {res['modes']} (S = per-replay input snapshots)",
         status="pass" if ok else "record" if args.get("accepted") else "fail",
             outcome="pass" if ok else "fail", accepted=args.get("accepted"), milestone="M5c", **res)  # fmt: skip
if args.get("record"):
    C.record("tests", r)
RESULT = dict(ok=ok, **{k: v for k, v in res.items() if k != "first_div_vs_run0"})
