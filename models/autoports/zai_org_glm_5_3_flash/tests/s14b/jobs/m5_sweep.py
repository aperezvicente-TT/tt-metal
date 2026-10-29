# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd single, chip 3): Stage 14b sweep-in-job (program JOB PIPELINE RULE 2026-10-19 23:52Z) — ONE
job loops over stack configurations on RESIDENT state, one pcc row + one perf row per configuration:

  resident in CTX["m5sweep"] (kept across jobs): the megakernel layer objects per (layer, kind, knobs), the stack
  programs per (sequence, t, knobs, rev), the per-layer KDA states per (n, t), the traced step per configuration;
  a configuration whose layers / stacks / traces exist costs only its replays.

Per configuration: the layer sequence (`dense` ids then `sparse` ids, x `repeat`) as ONE stack program per kind run on
one residual vs the same layers as single-layer launches (one trace each, device vs device: residual + every layer's
state / conv history BITWISE; `chain: false` skips the reference), then the stack's traced step timing.
ARGS: configs = [dict(name, dense ([0, 1, 2]), sparse ([]), repeat (1), t (1), env ({}), sparse_env ({}), reps (20),
chain (True))], rev, tag, record."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.mk_stack as MKS  # noqa: E402

importlib.reload(M5)
from models.autoports.zai_org_glm_5_3_flash.tests.common import golden, hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_kda_pre import masks  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
REV = args.get("rev")
tag = args.get("tag", "m5-sweep")
cfg_hf = hf_config()
code = os.environ.get("GLM53_CODE_SHA", "")
R = ctx.get("m5sweep")
if R is None or R.get("code") != code:  # a new code version: nothing resident is valid
    M5.prune_ctx(ctx, [])
    R = ctx["m5sweep"] = dict(code=code, layers={}, stacks={}, states={}, traces={}, x={}, res={}, pool={})
M5.prune_ctx(ctx, ["m5sweep"])  # other jobs' 14b objects (L1 pools) go; this job's resident state stays


def log(*a):
    print(f"M5-SWEEP {C.now()}", *a, flush=True)


def knob_key(env):
    return tuple(sorted((k, str(v)) for k, v in env.items()))


def layer_entry(L, sparse, env_d, env_s):
    key = (L, sparse, knob_key(env_s if sparse else env_d))
    if key not in R["layers"]:
        dsa = [L] if sparse == "dsa" else []
        R["layers"][key] = M5.build_layers(mesh, cfg_hf, ctx, [] if sparse else [L], [L] if sparse is True else [],
                                           env_d, env_s, dsa=dsa)[0]  # fmt: skip
    return R["layers"][key]


def latent_case(i, e, T):
    """a sparse_dsa layer's paged latent cache (random rows, snapshot kept), page table and positions at T rows"""
    k = (i, e["L"], T)
    if k not in R.setdefault("lat", {}):
        bps = 32  # 2048 positions: the dense-exact regime (no DSA indexer)
        cache = e["attn"].allocate_cache(T * bps)
        g = torch.Generator().manual_seed(17 + e["L"])
        snap = ttnn.from_torch(torch.randn(tuple(cache.shape), generator=g).to(torch.bfloat16), dtype=cache.dtype,
                               layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        pt = ttnn.from_torch(torch.arange(T * bps, dtype=torch.int32).reshape(T, bps), dtype=ttnn.int32,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        cur = ttnn.from_torch(torch.tensor([1000 + 37 * r for r in range(T)], dtype=torch.int32), dtype=ttnn.int32,
                              device=mesh)  # fmt: skip
        R["lat"][k] = dict(cache=cache, snap=snap, pt=pt, cur=cur)
    return R["lat"][k]


def mk_t(v, sh, dt=ttnn.float32):
    return ttnn.from_torch(v.reshape(sh), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def run_config(c):
    t_start = time.time()
    T = int(c.get("t", 1))
    dense, sparse = [int(v) for v in c.get("dense", [0, 1, 2])], [int(v) for v in c.get("sparse", [])]
    rep, reps, do_chain = int(c.get("repeat", 1)), int(c.get("reps", 20)), bool(c.get("chain", True))
    env_d = {**M5.DENSE_KNOBS, **{k: str(v) for k, v in (c.get("env") or {}).items()}}
    env_s = {**M5.SPARSE_KNOBS, **{k: str(v) for k, v in (c.get("sparse_env") or {}).items()}}
    dsa = [int(v) for v in c.get("dsa", [])]  # sparse_dsa layers (7b's MkMla + 7a's MkMoE): t <= 4
    assert not dsa or T <= 4, "sparse_dsa: t <= 4"
    base = sorted([layer_entry(L, False, env_d, env_s) for L in dense] + [layer_entry(L, True, env_d, env_s)
                   for L in sparse] + [layer_entry(L, "dsa", env_d, env_s) for L in dsa], key=lambda e: e["L"])  # fmt: skip
    for kind in (False, True):  # one set of persistent hand-off buffers per kind (shared by chain and stack)
        same = [e["mk"] for e in base if e["sparse"] == kind]
        for mk_ in same[1:]:
            mk_.persist = same[0].persist
    seq = base * rep
    n = len(seq)
    env_of = lambda e: env_s if e["sparse"] else env_d  # noqa: E731
    skey = (tuple(dense), tuple(sparse), tuple(dsa), rep, T, knob_key(env_d), knob_key(env_s), REV)
    t_build = time.time()
    if skey not in R["stacks"]:
        # ONE intermediates pool and ONE residual per t for every resident stack (stacks never run concurrently): the
        # resident set costs L1 once, not once per configuration
        stacks, res, pool = [], R.setdefault("res", {}).get(T), R.setdefault("pool", {}).get(T)
        for run in M5.kind_runs(seq):
            if seq[run[0]]["sparse"] == "dsa":
                stacks.append((run, None))
                continue
            M5.set_env(env_of(seq[run[0]]))
            stk = MKS.MkStack(mesh, [seq[i]["mk"] for i in run], t=T, rev=REV, res=res, pool=pool)
            res, pool = stk.residual(), stk.pool
            stacks.append((run, stk))
        R["res"][T], R["pool"][T] = res, pool
        R["stacks"][skey] = stacks
    stacks = R["stacks"][skey]
    if (n, T) not in R["states"]:
        R["states"][(n, T)] = [seq[0]["kda"].allocate_state(T) for _ in range(n)]
    sts = R["states"][(n, T)]
    if (seq[0]["L"], T) not in R["x"]:
        g = golden(seq[0]["L"], 2048)["streams_in"].float()
        x_rows = g[0, 300 : 300 + 37 * T : 37].reshape(T, 16384)
        gen = torch.Generator().manual_seed(4242)
        R["x"][(seq[0]["L"], T)] = dict(
            flat=mk_t(x_rows, (1, 1, T, 16384)), stk=mk_t(x_rows, (T, 1, 4, 4096) if T <= 4 else (1, 1, T, 16384)),
            hist0=mk_t((torch.randn(1, 3, T, 24576, generator=gen) * 0.5).to(torch.bfloat16), (1, 3, T, 24576),
                       ttnn.bfloat16),
            s0=mk_t(torch.randn(T, 64, 128, 128, generator=gen) * 0.1, (T, 64, 128, 128)),
            act=masks(mesh, torch.tensor([(i % 3) != 2 for i in range(T)]) if T > 1 else torch.tensor([True]),
                      seq[0]["kda"].lower_bound))  # fmt: skip
    X = R["x"][(seq[0]["L"], T)]
    lats = {i: latent_case(i, e, T) for i, e in enumerate(seq) if e["sparse"] == "dsa"}
    if c.get("build_only"):  # COMPILE-AHEAD: the host-side layer objects + stack descriptors, no launch
        for run, stk in stacks:
            if stk is not None and stk.program is None:
                M5.set_env(env_of(seq[run[0]]))
                stk.build([sts[i].C for i in run], X["act"], [sts[i].S for i in run])
        wall = dict(build_s=round(time.time() - t_build, 2), config_s=round(time.time() - t_start, 2))
        log("BUILT", c.get("name"), wall)
        return True, 0.0, wall

    def restore():
        for st in sts:
            ttnn.copy(X["hist0"], st.C)
            ttnn.copy(X["s0"], st.S)
        for lt in lats.values():
            ttnn.copy(lt["snap"], lt["cache"])

    def chain(do_restore):
        if do_restore:
            restore()
        y = X["flat"]
        for i, (e, st) in enumerate(zip(seq, sts)):
            M5.set_env(env_of(e))
            if e["sparse"] == "dsa":  # the attention half (flat input) then the FFN half, as 7b / 7a launch them
                lt = lats[i]
                s1 = e["mla"](y, lt["cache"], lt["pt"], lt["cur"], t=T)["s1"]
                y = e["moe"](s1, t=T)["out"]
                continue
            y = e["mk"](y, st.C, X["act"], t=T, state=st.S)["out"]
        return y

    def stack_step(do_restore):
        if do_restore:
            restore()
            ttnn.copy(X["stk"], R["res"][T])
        res = R["res"][T]
        for run, stk in stacks:
            M5.set_env(env_of(seq[run[0]]))
            if stk is None:  # sparse_dsa in stack mode: MkMla reads / keeps the residual, MkMoE writes it in place
                lt, e = lats[run[0]], seq[run[0]]
                s1 = e["mla"](res, lt["cache"], lt["pt"], lt["cur"], t=T, keep=("s1",), stack=True)["s1"]
                e["moe"](s1, t=T, out=res)
                del s1
                continue
            stk([sts[i].C for i in run], X["act"], [sts[i].S for i in run])
        return res

    t_num = time.time()
    num = {}
    if do_chain:
        trc = C.Traced(ttnn, mesh, lambda: chain(True))
        trc.replay()
        want = ttnn.to_torch(trc.out).float().reshape(-1, 16384)[:T]
        want_s = [ttnn.to_torch(st.S).float() for st in sts]
        want_c = [ttnn.to_torch(st.C) for st in sts]
        want_l = {i: ttnn.to_torch(lt["cache"]) for i, lt in lats.items()}
        trc.release()
        trs = C.Traced(ttnn, mesh, lambda: stack_step(True))
        trs.replay()
        got = ttnn.to_torch(R["res"][T]).float().reshape(T, 16384)
        l_bit = sum(bool(torch.equal(ttnn.to_torch(lt["cache"]), want_l[i])) for i, lt in lats.items())
        s_bit = sum(bool(torch.equal(ttnn.to_torch(st.S).float(), w)) for st, w in zip(sts, want_s))
        c_bit = sum(bool(torch.equal(ttnn.to_torch(st.C), w)) for st, w in zip(sts, want_c))
        trs.release()
        num = dict(out_bitwise=bool(torch.equal(got, want)), state_bitwise_layers=s_bit, conv_bitwise_layers=c_bit,
                   latent_bitwise_layers=l_bit,
                   max_abs=float((got - want).abs().max()), pcc=C.pcc(got, want))  # fmt: skip
    t_time = time.time()
    # fresh_x: the layer input is rewritten before every timed step (one 4-row x copy, ~2 us at t <= 4) so the MoE
    # layers keep routing realistic rows — without it a long replay series feeds the stack its own output and the
    # routed experts collapse onto a few (fewer weight bytes: optimistic timing)
    fresh = bool(c.get("fresh_x", False))
    tkey = ("stack", skey, fresh)

    def timed():
        if fresh:
            ttnn.copy(X["stk"], R["res"][T])
        return stack_step(False)

    if tkey not in R["traces"]:
        R["traces"][tkey] = C.Traced(ttnn, mesh, timed, warm=not do_chain)
    tr = R["traces"][tkey]
    st_t, st_b = C.stats(tr.replays(reps)), tr.batch(reps)
    t_end = time.time()
    names = [f"L{e['L']}{'a' if e['sparse'] == 'dsa' else 's' if e['sparse'] else 'd'}" for e in base]
    ok = (not do_chain) or (num["out_bitwise"] and num["state_bitwise_layers"] == n and num["conv_bitwise_layers"] == n
                            and num["latent_bitwise_layers"] == len(lats))  # fmt: skip
    wall = dict(build_s=round(t_num - t_build, 2), numerics_s=round(t_time - t_num, 2),
                timing_s=round(t_end - t_time, 2), config_s=round(t_end - t_start, 2))  # fmt: skip
    r = dict(test=f"m5 sweep [{c.get('name', '')}]: {names} x {rep} = {n} layers as {len(stacks)} stack program(s) vs "
                  f"{n} single-layer launches T={T}", status=("pass" if ok else "fail") if do_chain else "info",
             milestone="M5a", n=n, T=T, sequence=names, repeat=rep, programs_per_step=len(stacks), **num,
             knobs=dict(dense=env_d, sparse=env_s), tag=tag, config=c.get("name"), wall=wall)  # fmt: skip
    prow = dict(test=f"m5 sweep [{c.get('name', '')}] {n} layers T={T} step", status="info", milestone="M5a", fresh_x=fresh,
                traced=True, n=n, T=T, us=st_t["median_us"], per_layer_us=round(st_t["median_us"] / n, 1), stack=st_t,
                stack_batch_us=round(st_b, 1) if st_b else None, launches_per_step=len(stacks),
                knobs=dict(dense=env_d, sparse=env_s), tag=tag, config=c.get("name"), wall=wall)  # fmt: skip
    log("ROW", r)
    log("PERF", prow)
    if args.get("record"):
        C.record("pcc", r)
        C.record("perf", prow) if not os.environ.get("TT_METAL_WATCHER") else None
    return ok, st_t["median_us"] / n, wall


def drop_resident():
    """free the resident stacks / traces / pools (L1): a configuration that does not fit next to them gets a retry"""
    for k in list(R["traces"]):
        R["traces"].pop(k).release()
    R["stacks"].clear()
    R["res"].clear()
    R["pool"].clear()
    R["layers"].clear()  # their persistent hand-off buffers are L1 too (rebuilt from the cached decoders in ~1 s)
    R["x"].clear()
    gc.collect()


out = []
for c in args.get("configs", [dict(name="L0-L2 T1")]):
    for attempt in (0, 1):
        try:
            ok, per_layer, wall = run_config(c)
            out.append(dict(name=c.get("name"), ok=ok, per_layer_us=round(per_layer, 1), retried=attempt, **wall))
            break
        except Exception as e:  # a config that does not fit (L1 clash) must not cost the others
            log("CONFIG FAILED", c.get("name"), f"attempt {attempt}", repr(e)[:300])
            log("L1 live tensors", M5.l1_state(mesh), M5.l1_tensors())
            fits = "clash" in repr(e) or "Out of Memory" in repr(e) or "beyond max L1" in repr(e)
            drop_resident()
            log("L1 after drop", M5.l1_state(mesh), M5.l1_tensors())
            if attempt or not fits:
                out.append(dict(name=c.get("name"), ok=False, error=repr(e)[:200]))
                break
log("SUMMARY", out)
RESULT = dict(ok=all(o["ok"] for o in out), configs=out)
