# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host diagnosis of the in-trace DFlash2 kind on the prototype's golden anchors (no device, no ttnn): the kind's op
chain on the TORCH backend (tt/dflash2_kind.py TorchOps, TP2 shards + collectives modelled, the real Apache-2.0
drafter weights) against the reference (tt/dflash2_ref.py) STAGE BY STAGE, and both against the GPU golden
(golden_aime24_00_k7.pt: bf16 weights; the drafts, hidden, candidates, unary, scores per anchor). Reports the first
stage whose PCC drops below the bar, per anchor and overall. The context is seeded from the taps dump (the mean over
the 4 mHC streams, tap mode "mean"), the boot per anchor as the device unit does it (record c = L, n = 0: c' = L + 1,
the block attends positions <= L + 1 + K = the anchor position bound, NOT the ctx length).

Stages (the probe names of DFlash2Kind._p / DFlash2Ref._p): ctx.nrm (fc + hidden_norm), ctx.L{i}.k / .v (the context
K / V rows in the paged cache: the rotary + the cache write), emb, per layer x_ln, cf (kernel_projection), conv_a, q, k
(post norm + RoPE), v, att (the block attention with cur_pos = c' + K), o (all-reduced o_proj), conv_o, res, x_post,
cf2, conv_m, y (swiglu + down, all-reduced), h, then hid, hp, scores; the head vs the golden: hid rows 1..K, the
top-16 candidate sets, unary, scores, the drafts (the device selector walk).

DF2_RACE=1 (or --race): the same run with TorchOps.race = True, which emulates paged_update_cache's read-modify-write
race of 32 users on shared pages (one row per destination tile survives) -- the Stage 17 u1 silicon signature
(hidden PCC 0.3, drafts 0 / 16); printed as corroboration, not asserted.

    python -m pytest tests/s17/test_dflash2_anchors_host.py -q -s        (~3 min, ~25 GB host RAM)
    python tests/s17/test_dflash2_anchors_host.py [--race] [--anchors N]
Env: GLM53_DF2_DIR (the drafter), DF2_TAPS, DF2_GOLDEN (defaults: work-17-d0/dflash2)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_ref as DR  # noqa: E402

D2 = "/opt/glm53/work-17-d0/dflash2"
DRAFTER = os.environ.get("GLM53_DF2_DIR", f"{D2}/drafter")
TAPS = os.environ.get("DF2_TAPS", f"{D2}/taps/taps_aime24_00.pt")
GOLDEN = os.environ.get("DF2_GOLDEN", f"{D2}/golden_aime24_00_k7.pt")
K = 7
BAR = 0.999


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() != b.numel():
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _rows(t, k):  # the block rows 0..K of a kind tensor ([1, 1, 32, W] / [1, 32, nh, hd] / [32, W]) as [K+1, ...]
    t = t.float()
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 32:  # heads [1, 32, nh, hd]
        return t[0, : k + 1]
    return t.reshape(32, *t.shape[-1:] if t.dim() <= 4 else t.shape[-2:])[: k + 1] if t.dim() <= 4 else t[: k + 1]


def _stage_pcc(name, kind_t, ref_t, S):
    """the kind's chip-0 tensor vs the reference's (the chip's head slice for head tensors)"""
    kt = kind_t.float()
    rt = ref_t.float()
    if name.endswith((".q", ".k", ".v", ".att")) and rt.dim() == 3:  # ref [rows, heads, hd] (att: [rows, nh*hd])
        nh = kt.shape[2]
        return pcc(kt[0, : rt.shape[0], :nh], rt[:, :nh])
    if name.endswith(".att"):
        nh = kt.shape[2]
        rt = rt.reshape(rt.shape[0], -1, kt.shape[3])[:, :nh]
        return pcc(kt[0, : rt.shape[0], :nh], rt)
    if name == "scores":
        return pcc(
            kt.reshape(32, 32, 32)[1 : K + 1, :16, :16] if kt.numel() == 32 * 32 * 32 else kt[1 : K + 1, :16, :16], rt
        )
    rows = rt.shape[0]
    return pcc(kt.reshape(-1, kt.shape[-1])[:rows], rt.reshape(rows, -1))


def run(anchors: int | None = None, race: bool = False, tp: int = 2, log=print, layer: str = "ops") -> dict:
    """layer "mk" (rung 3, tt/dflash2_mk.py): the kind's layer / ctx programs emulated on the host (bf16 partials before
    the fabric all-reduce, the per-head caches); the stage probes are the emulation's (a subset of the reference's)"""
    cfg = DR.hf_config(os.path.join(DRAFTER, "config.json"))
    d = DR.Dims(cfg)
    w = DR.load_weights(DRAFTER)
    T = torch.load(TAPS, weights_only=False, map_location="cpu")
    G = torch.load(GOLDEN, weights_only=False, map_location="cpu")
    taps_mean = T["taps"].permute(1, 0, 2).contiguous()  # [n, 9, H] fp32 (the stream means)
    n = int(taps_mean.shape[0])
    anc = sorted(G["anchors"], key=lambda a: -int(a["L"]))
    if anchors:
        anc = anc[:anchors]
    log(
        f"drafter {DRAFTER}: L={d.L} H={d.H} nh={d.nh} nkv={d.nkv} V={d.V}; taps {n} rows; golden {G.get('weights')} "
        f"{len(anc)} anchors (descending L), race={race}, tp={tp}, layer={layer}"
    )
    ops = DK.TorchOps(tp, layer=layer)
    ops.race = race
    tin = ops.zeros((32, 64), torch.int64, "rm")
    rec = ops.zeros((1, 64), torch.int64, "rm")
    kind = DK.DFlash2Kind(
        ops,
        cfg,
        w,
        K=K,
        cap=n + 64,
        tin=tin,
        rec=rec,
        tap_mode="mean",
        conv="fused",
        layer=layer,
        conv_res=(layer == "mk") or None,
        log=lambda *a: None,
    )
    kind.probe = {}
    if kind.mk is not None:
        kind.mk.probe = {}
    ref = DR.DFlash2Ref(cfg, w)
    ref.probe = {}
    # the context: positions 0 .. n-1 (both from the bf16 taps)
    ref.ctx(taps_mean.to(torch.bfloat16), 0)
    kind.seed_ctx(taps_mean, 0)
    table = {}  # stage -> list of pcc over anchors (ctx stages once)

    def put(name, v):
        table.setdefault(name, []).append(v)

    if "ctx.nrm" in kind.probe:
        nrm_k = torch.cat([ops.host(t).reshape(32, -1) for t in kind.probe["ctx.nrm"]], 0)[:n]
        put("ctx.nrm", pcc(nrm_k, ref.probe["ctx.nrm"]))
    for i in range(d.L):
        for nm, cache, refc in (("k", kind.L[i]["Kc"], ref.Kc[i]), ("v", kind.L[i]["Vc"], ref.Vc[i])):
            if isinstance(cache, list):  # mk: per kv head [blocks, 1, 32, hd]
                ct = torch.cat([ops.host(t).float() for t in cache], 1)
            else:
                ct = ops.host(cache["t"]).float()
            c = ct.permute(0, 2, 1, 3).reshape(-1, kind.S.nkv_l, d.hd)[:n]
            put(f"ctx.L{i}.{nm}", pcc(c, refc[:n, : kind.S.nkv_l]))
    rows = []
    rec_host = torch.zeros(1, 64, dtype=torch.int64)
    tin_host = torch.zeros(32, 64, dtype=torch.int64)
    for a in anc:
        L, bonus = int(a["L"]), int(a["bonus"])
        kind.probe.clear()
        ref.probe.clear()
        rec_host[0, 0], rec_host[0, 1] = 0, L  # BOOT: n = 0, c = L -> c' = L + 1 (the bonus position), cur = L + 1 + K
        tin_host[0, 0] = bonus
        ops.write(rec, rec_host)
        ops.write(tin, tin_host)
        if kind.mk is not None:
            kind.mk.probe.clear()
        kind.boot()
        if (
            kind.mk is not None
        ):  # the emulation's stages under the reference's names (att: [1, G, 32, hd] -> heads layout)
            for nm, t in kind.mk.probe.items():
                if nm.endswith(".att"):
                    t = t.permute(0, 2, 1, 3)
                kind.probe.setdefault(nm, [t])
        out = ref.block(bonus, L + 1, L + 1, K)
        sel = ref.select(out["hid"], out["logits"], bonus, K)
        row = dict(L=L, bonus=bonus)
        first = None
        for name, rt in ref.probe.items():
            if name not in kind.probe:
                continue
            kt = kind.probe[name]
            v = _stage_pcc(name, ops.host(kt) if isinstance(kt, list) else kt, rt, kind.S)
            put(name, v)
            if first is None and not (v >= BAR):
                first = (name, round(v, 5))
        hid_k = ops.host(kind.last["hid"]).reshape(32, -1)[1 : K + 1].float()
        cand_k = ops.host(kind.last["cand"])[1 : K + 1].long()
        unary_k = ops.host(kind.last["unary"]).reshape(32, -1)[1 : K + 1].float()
        drafts_k = [int(ops.host(tin)[s, 0]) for s in range(1, K + 1)]
        gold = [int(v) for v in a["drafts"]]
        put("head.hid_vs_golden", pcc(hid_k, a["hid"].float()))
        put("head.hid_ref_vs_golden", pcc(out["hid"][1:].float(), a["hid"].float()))
        put("head.unary_vs_golden", pcc(unary_k, a["unary"].float()))
        overlap = [len(set(cand_k[s].tolist()) & set(a["cand"][s].tolist())) for s in range(K)]
        row.update(first_drop=first, drafts=drafts_k, golden=gold, ref_drafts=sel["drafts"], draft_eq=drafts_k == gold,
                   ref_eq=sel["drafts"] == gold, hid_pcc=round(table["head.hid_vs_golden"][-1], 5),
                   hid_ref_pcc=round(table["head.hid_ref_vs_golden"][-1], 5), top16_overlap=overlap)  # fmt: skip
        rows.append(row)
        log(
            f"anchor L={L:4d} first drop {first}  hid vs golden {row['hid_pcc']:.5f} (ref {row['hid_ref_pcc']:.5f})  "
            f"drafts {drafts_k} golden {gold} eq={row['draft_eq']} ref_eq={row['ref_eq']} top16 {overlap}"
        )
    # the table: stage order as the reference emits it
    order = (
        ["ctx.nrm"]
        + [f"ctx.L{i}.{nm}" for i in range(d.L) for nm in ("k", "v")]
        + [k for k in table if not k.startswith("ctx.")]
    )
    order = [k for k in order if k in table]  # (mk: no ctx.nrm probe)
    log(f"{'stage':<22} {'min pcc':>9} {'mean pcc':>9}  (kind torch backend vs reference; head.* vs the golden)")
    first_overall = None
    for name in order:
        vs = table[name]
        mn = min(vs)
        log(f"{name:<22} {mn:9.5f} {sum(vs) / len(vs):9.5f}")
        if first_overall is None and not (mn >= BAR) and not name.startswith("head."):
            first_overall = name
    eq = sum(r["draft_eq"] for r in rows)
    ref_eq = sum(r["ref_eq"] for r in rows)
    log(f"draft_eq kind {eq} / {len(rows)}, reference {ref_eq} / {len(rows)}; first stage below {BAR}: {first_overall}")
    return dict(rows=rows, table={k: (min(v), sum(v) / len(v)) for k, v in table.items()}, draft_eq=eq, ref_eq=ref_eq,
                first_drop=first_overall, race=race)  # fmt: skip


def host_stages(tp: int = 2, anchor_index: int = 0, log=print) -> dict:
    """rung 3 taps unit (tests/s17/jobs/dflash2_mk_taps.py): the mk emulation (tt/dflash2_mk.Df2MkTorch) on the real
    drafter, the taps dump as the context (8 rows per ctx call: the last call's stages are the ctx probes) and the
    boot of ONE golden anchor (the anchors in descending L; index 0 = the largest) -> dict(stages={name: chip-0
    tensor}, caches=[(K per head, V per head) per layer], hid, drafts, golden, L, bonus): the per-stage reference of
    the device taps (the same names as Df2Mk.taps)"""
    cfg = DR.hf_config(os.path.join(DRAFTER, "config.json"))
    d = DR.Dims(cfg)
    w = DR.load_weights(DRAFTER)
    T = torch.load(TAPS, weights_only=False, map_location="cpu")
    G = torch.load(GOLDEN, weights_only=False, map_location="cpu")
    taps_mean = T["taps"].permute(1, 0, 2).contiguous()
    n = int(taps_mean.shape[0])
    a = sorted(G["anchors"], key=lambda a_: -int(a_["L"]))[anchor_index]
    ops = DK.TorchOps(tp, layer="mk")
    tin = ops.zeros((32, 64), torch.int64, "rm")
    rec = ops.zeros((1, 64), torch.int64, "rm")
    kind = DK.DFlash2Kind(ops, cfg, w, K=K, cap=n + 64, tin=tin, rec=rec, tap_mode="mean", conv="fused", layer="mk",
                          conv_res=True, log=lambda *a_: None)  # fmt: skip
    kind.probe = {}
    kind.mk.probe = {}
    kind.seed_ctx(taps_mean, 0)
    ctx_stages = {k_: v for k_, v in kind.mk.probe.items() if k_.startswith("ctx.")}
    L, bonus = int(a["L"]), int(a["bonus"])
    rec_host = torch.zeros(1, 64, dtype=torch.int64)
    rec_host[0, 0], rec_host[0, 1] = 0, L
    tin_host = torch.zeros(32, 64, dtype=torch.int64)
    tin_host[0, 0] = bonus
    ops.write(rec, rec_host)
    ops.write(tin, tin_host)
    kind.mk.probe.clear()
    kind.boot()
    stages = dict(ctx_stages, **kind.mk.probe)
    caches = [([ops.host(t) for t in kind.L[i]["Kc"]], [ops.host(t) for t in kind.L[i]["Vc"]]) for i in range(d.L)]
    drafts = [int(ops.host(tin)[s_, 0]) for s_ in range(1, K + 1)]
    log(
        f"host stages: anchor L={L} bonus={bonus} drafts {drafts} golden {[int(v) for v in a['drafts']]}; stages {len(stages)}"
    )
    return dict(stages=stages, caches=caches, hid=ops.host(kind.last["hid"]), drafts=drafts, golden=[int(v) for v in a["drafts"]],
                golden_hid=a["hid"], L=L, bonus=bonus, n_ctx=n)  # fmt: skip


def test_kind_on_anchors():
    r = run()
    assert r["ref_eq"] >= len(r["rows"]) - 1, r["rows"]  # the reference reproduces the GPU golden
    assert r["first_drop"] is None, r["first_drop"]  # every stage of the kind's chain tracks the reference
    assert r["draft_eq"] >= len(r["rows"]) - 2 and r["table"]["head.hid_vs_golden"][0] >= BAR, (
        r["draft_eq"],
        r["table"],
    )


if __name__ == "__main__":
    race = "--race" in sys.argv or os.environ.get("DF2_RACE") == "1"
    na = int(sys.argv[sys.argv.index("--anchors") + 1]) if "--anchors" in sys.argv else None
    tp = int(sys.argv[sys.argv.index("--tp") + 1]) if "--tp" in sys.argv else 2
    layer = sys.argv[sys.argv.index("--layer") + 1] if "--layer" in sys.argv else "ops"
    r = run(anchors=na, race=race, tp=tp, layer=layer)
    print("RESULT", {k: v for k, v in r.items() if k != "rows"})
