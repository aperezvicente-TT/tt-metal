# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 job helpers (serve.py jobs: MESH / ARGS / CTX globals): the served construction + the speculative step
(tt/spec_mtp.py, GLM53_SPEC_K), evidence rows into $GLM53_OUT_DIR (s14common.record)."""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
S14B = os.path.join(os.path.dirname(HERE), "s14b")
if S14B not in sys.path:
    sys.path.insert(0, S14B)

import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

# Stage 17's rows always land in Stage 17's evidence (a job may run in another stage's server: 14b's q_mesh service sets
# GLM53_OUT_DIR to 14b's); mocks keep their own sink
E17 = "/opt/glm53/kit/doc/ttnn/17-mtp-spec-decode"  # (host-8chip and host-p300)
if not C.MOCK:
    C.OUT = os.environ.get("GLM53_E17_OUT", E17)

# pair units on another host: row ids carry GLM53_ROW_PREFIX (e.g. "d0:"), every row
# names its host
ROW_PREFIX = os.environ.get("GLM53_ROW_PREFIX", "")


def record(kind: str, row: dict):
    import socket

    row = dict(row, host=socket.gethostname())
    if ROW_PREFIX and not str(row.get("test", "")).startswith(ROW_PREFIX):
        row["test"] = ROW_PREFIX + str(row.get("test", ""))
    return C.record(kind, row)


# the 14b served construction (work-s14b/served_env.sh, launch_resident.sh) + Stage 17's knobs
SERVED_ENV = dict(GLM53_STACK_KNOBS="s7a283", GLM53_TP_XSL="1", GLM53_TP_XSL_SLICED="1", GLM53_STACK_WIDTHS="none",
                  GLM53_STACK_PREFILL_TRACES="0", GLM53_STACK_SHARE_FABRIC="1",
                  GLM53_STACK_KNOB_SET="s:GLM53_MHC_PRE_SPLIT=1", GLM53_STACK_NATIVE_WIDTHS="8,16")  # fmt: skip
# a TP2 pair (14b's pair profile, work-s14b/q_pair20w w01_env: the s14 knobs, no sliced XSL experts — the served s7a283
# profile's 8 kda_rec user groups at t > 4 need <= 8 heads per chip)
PAIR_ENV = dict(SERVED_ENV, GLM53_STACK_KNOBS="s14", GLM53_TP_XSL="0", GLM53_TP_XSL_SLICED="0", GLM53_STACK_KNOB_SET="")
# 14b's int24 served set (work-s14b q_mesh c01_env, TP8; q_pair24w wp00_env_served2 adds s:GLM53_REC_UGROUPS=2 at TP2)
SERVED24_ENV = dict(SERVED_ENV, GLM53_MLA_ATT_NKMAX="8",
                    GLM53_STACK_KNOB_SET="s:GLM53_MHC_PRE_SPLIT=1;s:GLM53_KDA_TILES_MIN=9;s:GLM53_MK_DS_FIDELITY=LoFi;"
                                         "s:GLM53_MK_LR_FIDELITY=HiFi2;s:GLM53_MK_QKV_ALIAS=1;s:GLM53_MHC_MODE=dz,pv2,cp,fr")  # fmt: skip
PROFILES = dict(served=SERVED_ENV, pair=PAIR_ENV, served24=SERVED24_ENV)


def modules():
    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp as SP
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM

    for mod in (M5, SP, SM, G):  # (a long-lived server: the job's code, not the first job's)
        importlib.reload(mod)
    return G, SM, SP


def serving(
    ctx,
    mesh,
    *,
    B: int,
    ids,
    spec_k: int,
    narrow: str = "none",
    native: bool = True,
    log=print,
    extra_env: dict | None = None,
    profile: str = "auto",
    max_seq_len: int = 2048,
    rev: str | None = None,
):
    """the served construction with the speculative step (K = spec_k) built in allocate_decode_io and compiled in
    prepare_decode (before any capture) -> (ctx key, backend, generator); max_seq_len > 2,048 with GLM53_STACK_DSA=1
    (extra_env): the long-context (sparse) DSA rows. profile "asis": the process env exactly as the caller's env job
    set it (e.g. 14b's int27 service env with GLM53_SPEC_K) and M5.serving's key with the caller's `rev` -> the SAME
    backend as the integrator's gate jobs (no second full-model build, no knob change)"""
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config

    G, SM, SP = modules()
    if profile == "asis":
        if os.environ.get("GLM53_SPEC_K") != str(spec_k):
            raise ValueError(
                f"profile asis: the env job must set GLM53_SPEC_K={spec_k} (is {os.environ.get('GLM53_SPEC_K')})"
            )
        key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=hf_config(), native=native, max_seq_len=max_seq_len,
                         rev=rev, log=log)  # fmt: skip
        return key, ctx[key]["backend"], ctx[key]["gen"]
    # profile: "auto" (the mesh: the served s7a283 profile; a pair: 14b's pair profile), "served" (s7a283 on any mesh:
    # the knobs the mesh jobs will run, at B <= 4 no user groups), "pair", "served24" (14b's int24 served set)
    tp2 = int(mesh.get_num_devices()) < 8
    base = PROFILES.get(profile, PAIR_ENV if tp2 else SERVED_ENV)
    if profile == "served24" and tp2:
        base = dict(base, GLM53_STACK_KNOB_SET=base["GLM53_STACK_KNOB_SET"] + ";s:GLM53_REC_UGROUPS=2")
    for k_ in set().union(*PROFILES.values()) - set(base):  # another profile's knobs (a long-lived server)
        os.environ.pop(k_, None)
    env = dict(base, GLM53_SPEC_K=str(spec_k), GLM53_STACK_NARROW_WIDTHS=narrow, **(extra_env or {}))
    for k_ in os.environ.pop("GLM53_S17_XENV", "").split(","):  # an earlier job's extra knobs (whatever way it ended)
        if k_ and k_ not in env:
            os.environ.pop(k_, None)
    os.environ.update(env)  # (stays set for the whole job: a capture after the build relaunches its programs)
    os.environ["GLM53_S17_XENV"] = ",".join(sorted(extra_env or {}))
    dbg = "-dbgids" if env.get("GLM53_SPEC_DBG_IDS") == "1" else ""
    xk = sorted((k, v) for k, v in (extra_env or {}).items() if k != "GLM53_SPEC_DBG_IDS")
    if xk:  # (a backend built under other knobs is another cache key)
        import hashlib

        dbg += "-x" + hashlib.sha1(repr(xk).encode()).hexdigest()[:8]
    if not dbg:
        os.environ.pop("GLM53_SPEC_DBG_IDS", None)
    msl = "" if max_seq_len == 2048 else f"-msl{max_seq_len}"
    key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=hf_config(), native=native, max_seq_len=max_seq_len,
                     rev=f"s17k{spec_k}-{narrow}{dbg}-{profile}{msl}", log=log)  # fmt: skip
    return key, ctx[key]["backend"], ctx[key]["gen"]


def prompt_ids(pid: str) -> list:
    """P5's AIME24 prompt ids, or "g4096:N" = the first N tokens of goldens/prompts/4096.pt (the DSA sparse rows past
    2,051 tokens; a mock takes 8: its prefill computes nothing)"""
    import json

    import torch

    ws = "/opt/glm53/kit"  # (host-8chip and host-p300)
    if pid.startswith("g4096:"):
        n = 8 if C.MOCK else int(pid.split(":")[1])
        return [int(v) for v in torch.load(os.path.join(ws, "goldens/prompts/4096.pt"), map_location="cpu",
                                           weights_only=False).reshape(-1)[:n]]  # fmt: skip
    prompts = json.load(open(os.path.join(ws, "doc/prework/p5_mtp_acceptance/prompts.json")))["prompts"]
    return [int(v) for v in next(q for q in prompts if q["id"] == pid)["ids"]]


def mtp_prefill(be, sp, io, prompt: list, plain, pt_row, B: int) -> int:
    """teacher-force `prompt` into slot 0 with the plain decode trace AND keep the MTP layer's own latent cache complete:
    after the plain step at position p (the head's h row 0 = h_p) the bootstrap replay, with token row 0 = x_{p+1} (the
    prompt's next token), writes the MTP latent of position p = (h_p, E[x_{p+1}]), the reference's MTP row p. Without
    it the drafts attend over MTP cache positions nothing wrote (stale DRAM: drafts differ between processes; outputs
    are unaffected -- verify decides every token). After the last prompt token the caller's enter() is that position's
    bootstrap (token row 0 = the first generated token). -> the first generated token"""
    import torch

    import ttnn

    be.reset_all(be._kv)
    sp.set_page_table(pt_row)
    for p, x in enumerate(prompt):
        tk = torch.zeros(B, dtype=torch.int64)
        tk[0] = x
        pos = torch.full((B,), -1, dtype=torch.int64)
        pos[0] = p
        be.write(io.tokens, tk)
        be.write(io.cur_pos, pos)
        be.replay(plain)
        if p + 1 < len(prompt):
            tk[0] = prompt[p + 1]
            be.write(io.tokens, tk)
            sp._replay("boot")
    ttnn.synchronize_device(be.mesh)
    return int(be.read(io.tokens)[0])


# --------------------------------------------------------------------- the mock's trace-allocation guard
def live_device_tensors() -> dict:
    """every allocated device ttnn.Tensor the Python process holds, keyed by (buffer address, shape, dtype) -- the
    mock never captures, so TT_METAL_TRACE_ALLOC_TRACKING cannot fire there; a function the capture WOULD record is
    guarded by the diff of this set across its call (new_live_tensors): any tensor it created that is still alive
    after it returns is what the device guard refuses before a checked replay ('device buffer(s) still alive before
    trace replay', Stage 17 u2 / u3 2026-10-22: DFlash2Kind.last held the block's op outputs)"""
    import gc
    import types

    import ttnn

    gc.collect()
    out = {}

    def take(o):
        if isinstance(o, ttnn.Tensor):
            try:
                if o.storage_type() == ttnn.StorageType.DEVICE and o.is_allocated():
                    out[(int(o.buffer_address()), tuple(int(v) for v in o.shape), str(o.dtype))] = o
            except Exception:
                pass

    # ttnn.Tensor (nanobind) instances are NOT GC-tracked: gc.get_objects() never lists them -- and CPython also
    # UNTRACKS a dict / tuple whose contents are all non-GC objects (a dict of tensors is itself invisible). So: start
    # from every tracked container and walk its values recursively (bounded) through the untracked child containers
    # (instance __dict__ -> a dict of tensors -> the tensors; lists / tuples / sets / cells alike).
    seen = set()

    def values(o):
        if isinstance(o, dict):
            return list(o.values())
        if isinstance(o, (list, tuple, set, frozenset)):
            return list(o)
        if isinstance(o, types.CellType):
            try:
                return [o.cell_contents]
            except ValueError:
                return []
        return None

    def walk(o, depth):
        vals = values(o)
        if vals is None or id(o) in seen:
            return
        seen.add(id(o))
        for v in vals:
            if isinstance(v, ttnn.Tensor):
                take(v)
            elif depth > 0 and values(v) is not None:
                walk(v, depth - 1)

    for o in gc.get_objects():
        walk(o, 3)
    return out


def new_live_tensors(before: dict) -> list:
    """the tensors alive now that were not alive at `before` (live_device_tensors()): [(key, tensor)]"""
    now = live_device_tensors()
    return [(k, t) for k, t in now.items() if k not in before]


def guard_capture_fn(fn, log=print, name: str = "") -> list:
    """run fn() (a function a capture would record) under the mock's guard: returns the list of tensor keys it left
    alive (empty = clean); logs them"""
    before = live_device_tensors()
    fn()
    leaked = new_live_tensors(before)
    for k, _ in leaked:
        log(
            f"TRACE-ALLOC GUARD {name}: tensor allocated inside the captured function is still alive: addr={k[0]:#x} shape={k[1]} {k[2]}"
        )
    return [k for k, _ in leaked]
