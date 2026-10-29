# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, a d0 pair, watched): u0 of rung 3 -- the per-phase device TAPS of the mk programs (GLM53_DF2_TAPS=1,
tt/dflash2_mk.py) against the host emulation (tests/s17/test_dflash2_anchors_host.host_stages: the same drafter, the
same taps dump as the context, the same golden anchor) STAGE BY STAGE, so one run names the first wrong phase.
Flow (as u1): the served construction with the drafter kind (mk, taps on); the ctx program captured once and replayed
per 8-row chunk of the taps dump (host writes of the tap slots / positions / cos-sin tiles between replays); ONE boot
replay of the largest-L golden anchor; then every tap tensor, the per-head caches, hid and the drafts are read back
(chip 0) and compared with the emulation's stages (rows 0..K of the block; the ctx: the last chunk's rows and every
written cache position). Row: {stage: pcc} in program order + first_wrong (the first stage below `bar`).
ARGS: layers ("0,1,2,3,4"), profile ("served24"), b (32), taps, golden, max_seq_len (4096), bar (0.999), env (the
dflash2 knobs; GLM53_DF2_LAYER=mk and GLM53_DF2_TAPS=1 are forced), tag."""
import importlib
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s17common as S  # noqa: E402

S = importlib.reload(S)
C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.s17 import test_dflash2_anchors_host as AH  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import read_rows  # noqa: E402

AH = importlib.reload(AH)
cfg = hf_config()
ids = [int(v) for v in str(args.get("layers", "0,1,2,3,4")).split(",")]
B, K, TILE = int(args.get("b", 32)), 7, 32
BAR = float(args.get("bar", 0.999))
XENV = {k: str(v) for k, v in (args.get("env") or {}).items()}
XENV.setdefault("GLM53_SPEC_DRAFTER", "dflash2")
XENV.setdefault("GLM53_SPEC_VERIFY_W", "plain")
XENV["GLM53_DF2_LAYER"] = "mk"
XENV["GLM53_DF2_TAPS"] = "1"
key, be, gen = S.serving(ctx, mesh, B=B, ids=ids, spec_k=K, narrow="none", profile=str(args.get("profile", "served24")),
                         extra_env=XENV, max_seq_len=int(args.get("max_seq_len", 4096)))  # fmt: skip
sp, kv = be.spec, be._kv
io = kv.extra["io"]
df2 = sp.df2
assert (
    df2 is not None and df2.mk is not None and df2.mk.taps_on
), "the taps unit needs GLM53_DF2_LAYER=mk GLM53_DF2_TAPS=1"


def log(*a):
    print(f"S17-DF2T {C.now()}", *a, flush=True)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() != b.numel() or a.numel() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def chip0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


tp = int(mesh.get_num_devices())
AH.DRAFTER = XENV.get("GLM53_DF2_DIR", AH.DRAFTER)
AH.TAPS, AH.GOLDEN = str(args.get("taps", AH.TAPS)), str(args.get("golden", AH.GOLDEN))
T = torch.load(AH.TAPS, weights_only=False)
G = torch.load(AH.GOLDEN, weights_only=False)
taps = T["taps"].permute(1, 0, 2).contiguous()  # [n, 9, H] (stream means)
n = int(taps.shape[0])
anchor = sorted(G["anchors"], key=lambda a_: -int(a_["L"]))[0]
row = dict(test="dflash2-mk-taps", milestone="DF2", K=K, B=B, layers=str(args.get("layers")), tp=tp, L=int(anchor["L"]),
           bar=BAR, env=XENV, tag=args.get("tag"))  # fmt: skip
if C.MOCK:
    sp.capture()
    S.record("tests", dict(row, status="info", mock=True))
    RESULT = dict(ok=True, mock=True)
else:
    sp.capture()
    if "df2_ctx_trace" not in ctx[key]:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            df2.ctx_program()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ctx[key]["df2_ctx_trace"] = tid
    tid = ctx[key]["df2_ctx_trace"]
    from models.autoports.zai_org_glm_5_3_flash.tt.dflash2_ref import rope_cos_sin

    for r0 in range(0, n, 8):  # the context, 8 rows per replay (the all-reduce's rows)
        part = taps[r0 : r0 + 8]
        m = int(part.shape[0])
        for s_ in range(df2.d.n_taps):
            slot = torch.zeros(TILE, df2.tap_w)
            slot[:m] = part[:, s_].float()
            df2.ops.write(df2.taps[s_], slot.reshape(1, 1, TILE, -1))
        pos = torch.arange(r0, r0 + TILE, dtype=torch.int32)
        df2.ops.write(df2.st.cpos, pos)
        df2.ops.write(df2.st.cposu, pos.reshape(1, TILE))
        cos_t, sin_t = rope_cos_sin(pos.clamp(0, df2.cap - 1), df2.d.theta, df2.d.hd)
        df2.ops.write(df2.rope_tiles[2], cos_t.to(torch.bfloat16).reshape(1, 1, TILE, -1))
        df2.ops.write(df2.rope_tiles[3], sin_t.to(torch.bfloat16).reshape(1, 1, TILE, -1))
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    last_rows = n - (n - 1) // 8 * 8  # valid rows of the last ctx chunk
    dev_ctx = {k_: chip0(t) for k_, t in df2.mk.taps.items() if k_.startswith("ctx.")}
    L, bonus = int(anchor["L"]), int(anchor["bonus"])
    tk = torch.zeros(B, dtype=torch.int64)
    posb = torch.full((B,), -1, dtype=torch.int64)
    tk[0], posb[0] = bonus, L + 1
    be.write(io.tokens, tk)
    be.write(io.cur_pos, posb)
    sp._replay("boot")
    ttnn.synchronize_device(mesh)
    dev = dict(dev_ctx, **{k_: chip0(t) for k_, t in df2.mk.taps.items() if not k_.startswith("ctx.")})
    dev_caches = [([chip0(t) for t in Ly["Kc"]], [chip0(t) for t in Ly["Vc"]]) for Ly in df2.L]
    tin = read_rows(mesh, sp.tin).reshape(TILE, -1)
    drafts = [int(v) for v in tin[1 : K + 1, 0]]
    log("device: drafts", drafts, "golden", [int(v) for v in anchor["drafts"]])
    H = AH.host_stages(tp=tp, anchor_index=0, log=log)  # the emulation on the same inputs (the host, ~1 min)
    table, order = {}, []

    def put(name, a, b):
        v = pcc(a, b)
        table[name] = round(v, 6)
        order.append(name)
        return v

    nL = len(df2.L)
    # the ctx stages: the last chunk's valid rows; the caches: every position written (0 .. n-1)
    for name in ["ctx.fc", "ctx.xn"] + [f"ctx.L{i}.{nm}" for i in range(nL) for nm in ("qkv", "k_rows", "v_rows")]:
        if name in dev and name in H["stages"]:
            put(name, dev[name].reshape(TILE, -1)[:last_rows], H["stages"][name].float().reshape(TILE, -1)[:last_rows])
    for i in range(nL):
        for nm, j in (("K", 0), ("V", 1)):
            nkvh = df2.mk.nkvh
            dk = torch.cat(
                [t.float().reshape(nkvh, -1, df2.d.hd)[:, :n].reshape(-1, df2.d.hd) for t in dev_caches[i][j]], 0
            )
            hk = torch.cat([t.float().reshape(-1, df2.d.hd)[:n] for t in H["caches"][i][j]], 0)
            put(f"cache.L{i}.{nm}", dk, hk)
    # the block stages: rows 0 .. K (the bonus row and the K mask rows)
    kb = K + 1
    for i in range(nL):
        for nm in (
            "x_ln",
            "cf",
            "conv_a",
            "qkv",
            "q",
            "k_rows",
            "v_rows",
            "att",
            "o",
            "res",
            "x_post",
            "cf2",
            "conv_m",
            "gu",
            "m",
            "y",
            "res2",
        ):
            name = f"L{i}.{nm}"
            if name in dev and name in H["stages"]:
                a, b = dev[name], H["stages"][name].float()
                if nm == "att":  # [1, nqh, 32, hd] (or the fold's [1, nkvh, G 32, hd]: the same bytes): rows = dim 2
                    a = a.reshape(1, -1, TILE, a.shape[-1])
                    put(name, a[0, :, :kb], b[0, :, :kb])
                else:
                    put(name, a.reshape(TILE, -1)[:kb], b.reshape(TILE, -1)[:kb])
    put("hid", dev["hid"].reshape(TILE, -1)[:kb], H["hid"].float().reshape(TILE, -1)[:kb])
    hid_golden = pcc(dev["hid"].reshape(TILE, -1)[1:kb], anchor["hid"].float())
    first = next((nm for nm in order if not (table[nm] >= BAR)), None)
    for nm in order:
        log(f"  {nm:<18} {table[nm]:.6f}{'  <-- first below the bar' if nm == first else ''}")
    ok = first is None and drafts == H["golden"]
    row.update(status="pass" if ok else "fail", stages=table, first_wrong=first, drafts=drafts, golden=H["golden"],
               host_drafts=H["drafts"], draft_eq=drafts == H["golden"], hid_vs_golden=round(hid_golden, 6),
               ctx_rows=n, last_chunk_rows=last_rows)  # fmt: skip
    S.record("tests", row)
    log("first wrong stage:", first, "drafts eq", drafts == H["golden"], "hid vs golden", round(hid_golden, 5))
    RESULT = dict(ok=ok, first_wrong=first, stages=table)
