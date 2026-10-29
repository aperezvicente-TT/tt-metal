# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test (no device, no ttnn): the in-trace DFlash2 kind's op sequence (tt/dflash2_kind.py on the torch backend,
TP shards + collectives modelled) against the drafter reference (tt/dflash2_ref.py) on random weights of a small
config: context K / V by position, the block's hidden / logits / candidates, the drafts (the selector walk) and the
token / record pages the step writes; two consecutive steps (the second's context comes from the first's taps).
Bars: hidden PCC >= 0.999, logits PCC >= 0.999, drafts equal, context K PCC >= 0.999, launch census recorded.
Rung 2 (the fused conv program, kernels/spec/df2_conv_*.cpp): test_fused_conv runs the program's tile arithmetic with
the kernels' index math (dflash2_kind.conv_fused_host) on random data against the reference conv and the stock-op
chain (bitwise on the host: the same fp32 formula), on the real drafter's geometry (H 4096, groups of 16) and the
small config's; the kind test runs both conv modes (hid / drafts bars unchanged) and prints the per-op census.
    python -m pytest tests/s17/test_dflash2_kind_host.py -q   (any python with torch; run with -I from the tree root)"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_ref as DR  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _run(
    tp: int,
    tap_mode: str,
    K: int = 7,
    seed: int = 0,
    conv: str = "fused",
    layers: int = 2,
    mm: str = "stream",
    conv_res: bool = False,
    layer: str = "ops",
):
    torch.manual_seed(seed)
    cfg = DR.small_config(tp=tp, n_taps=3, layers=layers)
    d = DR.Dims(cfg)
    w = DR.random_weights(cfg, seed=seed, scale=0.05)
    ops = DK.TorchOps(tp, mm=mm, layer=layer)
    tin = ops.zeros((32, 64), torch.int64, "rm")
    rec = ops.zeros((1, 64), torch.int64, "rm")
    kind = DK.DFlash2Kind(
        ops,
        cfg,
        w,
        K=K,
        cap=256,
        tin=tin,
        rec=rec,
        tap_mode=tap_mode,
        conv=conv,
        conv_res=conv_res,
        layer=layer,
        log=lambda *a: None,
    )
    ref = DR.DFlash2Ref(cfg, w)
    # a committed prompt of n0 positions: random 4-stream residual taps (the drafter's input is the stream mean)
    n0 = 20
    streams = torch.randn(n0, d.n_taps, 4, d.H) * 0.5
    taps_mean = streams.mean(2).to(torch.bfloat16)
    ref.ctx(taps_mean, 0)
    kind.seed_ctx(taps_mean, 0)
    # the state after the plain step that produced the bonus b at position n0 (BOOT: n = 0, c = n0 - 1)
    bonus = 11
    rec_host = torch.zeros(1, 64, dtype=torch.int64)
    rec_host[0, 0], rec_host[0, 1] = 0, n0 - 1
    ops.write(rec, rec_host)
    tin_host = torch.zeros(32, 64, dtype=torch.int64)
    tin_host[0, 0] = bonus
    ops.write(tin, tin_host)
    ops.launches = 0
    kind.boot()  # the bootstrap: the block at positions n0 .. n0 + K attending to context < n0
    boot_launches = ops.launches
    out = ref.block(bonus, n0, n0, K)
    sel = ref.select(out["hid"], out["logits"], bonus, K)
    hid_k = ops.host(kind.last["hid"]).reshape(32, -1)[: K + 1].float()
    cand_k = ops.host(kind.last["cand"])[1 : K + 1].long()
    unary_k = ops.host(kind.last["unary"]).reshape(32, -1)[1 : K + 1].float()
    drafts = [int(ops.host(tin)[s, 0]) for s in range(1, K + 1)]
    rec_drafts = [int(ops.host(rec).reshape(-1)[16 + s]) for s in range(K)]
    # the selector walk on the kind's own candidates (the reference's walk code): the device-side scores + the kernel
    walk = ref.walk(hid_k[1:], cand_k, unary_k, bonus)
    overlap = [len(set(cand_k[s].tolist()) & set(sel["cand"][s].tolist())) for s in range(K)]
    r = dict(hid_pcc=pcc(hid_k, out["hid"].float()), drafts=drafts, ref_drafts=sel["drafts"], rec_drafts=rec_drafts,
             walk_drafts=walk["drafts"], plain_eq=sum(int(a) == b for a, b in zip(cand_k[:, 0], sel["plain"])), top16_overlap=overlap,
             boot_launches=boot_launches)  # fmt: skip
    # step 1: the verify accepted n = 3 of the drafts (rows 0..3 committed at positions c .. c + 3, c = n0): their taps
    # arrive on the slots (all wv = 8 rows; rows past n are garbage the later writes overwrite) -> ctx -> block at
    # c' = c + n + 1
    n, c = 3, n0
    wv = 8
    step_streams = torch.randn(wv, d.n_taps, 4, d.H) * 0.5
    for s in range(d.n_taps):
        slot = torch.zeros(32, kind.tap_w)
        v = step_streams[:, s].reshape(wv, 4 * d.H) if tap_mode == "fold" else step_streams[:, s].mean(1)
        slot[:wv] = v.to(torch.bfloat16).float()
        ops.write(kind.taps[s], slot.reshape(1, 1, 32, -1))
    rec_host[0, 0], rec_host[0, 1] = n, c
    bonus2 = 23
    tin_host[0, 0] = bonus2
    ops.write(rec, rec_host)
    ops.write(tin, tin_host)
    ops.launches = 0
    ops.census.clear()
    kind.prep_program()
    kind.ctx_program()
    ctx_launches = ops.launches
    ctx_census = dict(ops.census)
    ops.launches = 0
    ops.census.clear()
    kind.block_program()
    step_launches = ops.launches
    block_census = dict(ops.census)
    ref.ctx(step_streams[: n + 1].mean(2).to(torch.bfloat16), c)  # the committed rows only
    c1 = c + n + 1
    out2 = ref.block(bonus2, c1, c1, K)
    sel2 = ref.select(out2["hid"], out2["logits"], bonus2, K)
    hid_k2 = ops.host(kind.last["hid"]).reshape(32, -1)[: K + 1].float()
    if layer == "mk":  # per kv head caches [blocks, 1, 32, hd] (tt/dflash2_mk.py)
        k_cache = torch.cat([ops.host(t).float() for t in kind.L[0]["Kc"]], 1)
    else:
        k_cache = ops.host(kind.L[0]["Kc"]["t"]).float()  # [blocks, nkv_l, 32, hd] of chip 0 -> positions 0 .. c + n
    k_pos = k_cache.permute(0, 2, 1, 3).reshape(-1, kind.S.nkv_l, d.hd)[: c + n + 1]
    cand_k2 = ops.host(kind.last["cand"])[1 : K + 1].long()
    walk2 = ref.walk(hid_k2[1:], cand_k2, ops.host(kind.last["unary"]).reshape(32, -1)[1 : K + 1].float(), bonus2)
    r.update(hid_pcc2=pcc(hid_k2, out2["hid"].float()), drafts2=[int(ops.host(tin)[s, 0]) for s in range(1, K + 1)],
             ref_drafts2=sel2["drafts"], walk_drafts2=walk2["drafts"], ctx_k_pcc=pcc(k_pos, ref.Kc[0][: c + n + 1, : kind.S.nkv_l]),
             top16_overlap2=[len(set(cand_k2[s].tolist()) & set(sel2["cand"][s].tolist())) for s in range(K)],
             ctx_launches=ctx_launches, step_launches=step_launches, tap_mode=tap_mode, tp=tp, conv=conv, mm=mm,
             block_census=block_census, ctx_census=ctx_census, hid2=hid_k2.clone(), hid1=hid_k.clone(),
             k_cache2=k_cache.clone(), cand2=cand_k2.clone(), layer=layer)  # fmt: skip
    return r


@pytest.mark.parametrize(
    "tp,tap_mode,conv",
    [(8, "mean", "ops"), (8, "fold", "ops"), (2, "fold", "ops"), (8, "fold", "fused"), (2, "fold", "fused")],
)
def test_kind_matches_reference(tp, tap_mode, conv):
    r = _run(tp, tap_mode, conv=conv)
    print(r)
    assert r["hid_pcc"] >= 0.999 and r["hid_pcc2"] >= 0.999, r
    assert r["ctx_k_pcc"] >= 0.999, r
    assert min(r["top16_overlap"]) >= 14 and min(r["top16_overlap2"]) >= 14, r  # (random flat logits: near-tie swaps)
    assert r["plain_eq"] >= 6, r  # the plain argmax per mask row (7 rows; one near-tie flip tolerated)
    assert r["drafts"] == r["walk_drafts"] == r["rec_drafts"], r  # the device selector == the reference walk
    assert r["drafts2"] == r["walk_drafts2"], r


def _conv_inputs(H: int, group: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    ng = H // group
    x = (torch.randn(32, H, generator=g)).to(torch.bfloat16)
    cf = (0.1 * torch.randn(32, 4 * ng, generator=g)).to(torch.bfloat16)
    base = (torch.tensor([[1.0, 0.0]]).reshape(1, 2, 1).repeat(2, 1, H) + 0.05 * torch.randn(2, 2, H, generator=g)).to(
        torch.bfloat16
    )
    return x, cf, base


@pytest.mark.parametrize("H,group", [(4096, 16), (256, 16), (512, 32), (256, 8)])
def test_fused_conv(H, group):
    """the rung-2 program's arithmetic (conv_fused_host = the kernels' tile ops and index math) == the reference conv
    == the stock-op chain, both sides, on random data: bitwise on the host (the same fp32 formula; the device program's
    fp32 SFPU products / sums may differ by rounding: PCC >= 0.9999 is the device bar)"""
    cfg = DR.small_config(tp=8, n_taps=3, layers=1)
    cfg["hidden_size"], cfg["dflash_config"]["conv_group_size"] = H, group
    cfg["num_key_value_heads"], cfg["num_attention_heads"], cfg["head_dim"] = 8, 8, 32
    d = DR.Dims(cfg)
    K = DK.conv_consts(group)
    x, cf, base = _conv_inputs(H, group)
    base4 = DK.DFlash2Shards._base4(base)
    ref = DR.DFlash2Ref(cfg, {})
    ops = DK.TorchOps(1)
    tin, rec = ops.zeros((32, 64), torch.int64, "rm"), ops.zeros((1, 64), torch.int64, "rm")
    w = DR.random_weights(cfg, seed=0)
    kind = DK.DFlash2Kind(ops, cfg, w, K=7, cap=64, tin=tin, rec=rec, conv="ops", log=lambda *a: None)
    for side in (0, 1):
        want = ref.conv(x, cf, side, base)
        got = DK.conv_fused_host(x, cf, base4, side, K)
        chain = ops.host(
            kind._conv(
                ops.rep(x.reshape(1, 1, 32, H)),
                ops.rep(cf.reshape(1, 1, 32, -1)),
                side,
                [ops.rep(base.float().reshape(4, H)[j].reshape(1, 1, 1, -1), torch.float32) for j in range(4)],
            )
        )
        chain = chain.reshape(32, H)
        err = (got.float() - want.float()).abs().max().item()
        assert pcc(got.float(), want.float()) >= 0.9999 and err <= 2 * 2**-8 * want.float().abs().max().item(), (
            side,
            err,
        )
        assert torch.equal(got, want), (side, "fused vs reference not bitwise", err)
        assert torch.equal(chain, want), (side, "stock chain vs reference not bitwise")
    # the x_prev shift: row 0 has no predecessor (coef_1 x_prev term is 0 on row 0)
    x0 = x.clone()
    x0[1:] = 0
    got0 = DK.conv_fused_host(x0, cf, base4, 0, K)
    coef0 = base[0, 0].float() + cf[:, : d.ng].float().repeat_interleave(group, dim=-1)
    assert torch.equal(got0[0], (coef0[0] * x0[0].float()).to(torch.bfloat16)) and torch.equal(
        got0[1],
        ((base[0, 1].float() + cf[1, d.ng : 2 * d.ng].float().repeat_interleave(group, dim=-1)) * x0[0].float()).to(
            torch.bfloat16
        ),
    )


def test_launch_census_8_layers():
    """the rung-3 input: the block program's launches per op at the real drafter's layer count (8), both conv modes;
    the two modes are the same arithmetic on the host (hidden bitwise, drafts equal) and the fused mode saves 9
    launches per conv call (4 calls x 8 layers = 288). (The reference PCC at 8 random-weight layers is ~0.99 in BOTH
    modes: the kind's bf16 rounding points differ from the reference's and compound with depth; the 2-layer bars above
    and the device unit carry the numerics.)"""
    rs = {conv: _run(8, "fold", conv=conv, layers=8) for conv in ("ops", "fused")}
    for conv, r in rs.items():
        print(
            conv,
            "block launches",
            r["step_launches"],
            "ctx+prep",
            r["ctx_launches"],
            sorted(r["block_census"].items(), key=lambda kv: -kv[1]),
        )
        assert r["drafts"] == r["walk_drafts"] == r["rec_drafts"], r
    a, b = rs["ops"], rs["fused"]
    assert a["step_launches"] - b["step_launches"] == 9 * 4 * 8, (a["step_launches"], b["step_launches"])
    assert a["ctx_launches"] == b["ctx_launches"] and a["hid_pcc2"] == b["hid_pcc2"] and a["drafts2"] == b["drafts2"], (
        a,
        b,
    )
    assert b["block_census"]["conv_fused"] == 32 and "linear_bias" not in b["block_census"], b["block_census"]


def test_mm_modes_same_ops():
    """rung 3 M1 (tt/dflash2_mm.py, GLM53_DF2_MM=ds): the matmul launch units -- on the host backend the two modes are
    the same arithmetic (hidden / context cache BITWISE, drafts equal: the kind's op order is unchanged); the launch
    census moves by the same-input groups only: ctx 8 qkv -> 1 (-7), head lm_head + hproj -> 1 (-1); the block's 48
    per-layer matmuls stay 48 launches (one unit each: the per-layer program is M2). (On device the TP8 lm_head's
    padded DS output adds one slice launch; the DS matmul's K-block accumulation order equals mm_stream's.)"""
    a, b = (_run(8, "fold", conv="fused", layers=8, mm=m) for m in ("stream", "ds"))
    for r in (a, b):
        assert r["drafts"] == r["walk_drafts"] == r["rec_drafts"], r
    assert torch.equal(a["hid2"], b["hid2"]) and torch.equal(
        a["k_cache2"], b["k_cache2"]
    ), "mm modes differ on the host"
    assert a["drafts2"] == b["drafts2"] and a["drafts"] == b["drafts"]
    assert a["ctx_launches"] - b["ctx_launches"] == 7, (a["ctx_launches"], b["ctx_launches"])
    assert a["step_launches"] - b["step_launches"] == 1, (a["step_launches"], b["step_launches"])
    assert a["block_census"]["stream_mm"] == 50 and a["ctx_census"]["stream_mm"] == 9, (
        a["block_census"],
        a["ctx_census"],
    )
    assert b["block_census"]["ds_mm"] == 48 and b["block_census"]["ds_group"] == 1, b["block_census"]
    assert b["ctx_census"]["ds_mm"] == 1 and b["ctx_census"]["ds_group"] == 1, b["ctx_census"]
    print("stream", a["ctx_launches"], a["step_launches"], "ds", b["ctx_launches"], b["step_launches"])


def test_conv_res_fold():
    """rung 3 M2 prep (kernels/spec DF2_CONV_RES, GLM53_DF2_CONV_RES=1): the two "finish" convs of a layer take the
    fp32 all-reduced partial and add the residual inside the program -- on the host emulation BITWISE the chain
    (typecast -> conv -> add: the same roundings in the same places), drafts equal, and -32 launches per step at 8
    layers (16 typecast + 16 add), the census moving to 16 conv_fused + 16 conv_fused_res"""
    a, b = (_run(8, "fold", conv="fused", layers=8, conv_res=r) for r in (False, True))
    assert torch.equal(a["hid2"], b["hid2"]) and a["drafts2"] == b["drafts2"] and a["drafts"] == b["drafts"]
    assert a["step_launches"] - b["step_launches"] == 32, (a["step_launches"], b["step_launches"])
    assert b["block_census"]["conv_fused"] == 16 and b["block_census"]["conv_fused_res"] == 16, b["block_census"]
    assert "typecast" not in b["block_census"] and "add" not in b["block_census"], b["block_census"]
    assert a["block_census"]["typecast"] == 16 and a["block_census"]["add"] == 16, a["block_census"]


@pytest.mark.parametrize("tp", [8, 2])
def test_mk_mode(tp):
    """rung 3 M2 (tt/dflash2_mk.py, GLM53_DF2_LAYER=mk): the layers / ctx as megakernel programs, emulated on the host
    in the device order (bf16 partials before the all-reduce, the fabric's per-round rounding, rows >= 8 zero, the
    per-head caches, the SDPA layout trick) vs the rung-2 chain (fused conv + the residual fold) on the same random
    weights: hidden PCC >= 0.9999 (both steps), the context K cache PCC >= 0.9999, the drafts equal and == the
    reference walk, the reference bars of the kind test, and the launch census: block = a0 1 + L x (sdpa 1 + ba 1) +
    the embedding / head chain, ctx = concat 1 + 1 program"""
    # the numerics at the host reference's depth (2 layers: the kind test's bars), the census at the real depth (8)
    a2 = _run(tp, "fold", conv="fused", layers=2, mm="ds", conv_res=True, layer="ops")
    b2 = _run(tp, "fold", conv="fused", layers=2, mm="ds", conv_res=True, layer="mk")
    a = _run(tp, "fold", conv="fused", layers=8, mm="ds", conv_res=True, layer="ops")
    b = _run(tp, "fold", conv="fused", layers=8, mm="ds", conv_res=True, layer="mk")
    print("mk block census", sorted(b["block_census"].items(), key=lambda kv: -kv[1]), "ctx", b["ctx_census"])
    print(
        "mk vs ops hid PCC: 2 layers",
        pcc(a2["hid1"], b2["hid1"]),
        pcc(a2["hid2"], b2["hid2"]),
        "8 layers",
        pcc(a["hid1"], b["hid1"]),
        pcc(a["hid2"], b["hid2"]),
        "ctx K",
        pcc(a2["k_cache2"], b2["k_cache2"]),
    )
    for r in (a, b, a2, b2):  # the device-side selector == the reference walk on the kind's own candidates
        assert r["drafts"] == r["walk_drafts"] == r["rec_drafts"], r
        assert r["drafts2"] == r["walk_drafts2"], r
    # mk vs the chain on RANDOM weights: 0.9996 at 2 layers = the bf16 partials before the all-reduce (|y| >> |res|
    # with random weights); the numerics gate is test_dflash2_anchors_host.py --layer mk on the real drafter (16 / 16,
    # hid vs the golden 0.99993+ at TP2 and TP8)
    assert pcc(a2["hid1"], b2["hid1"]) >= 0.999 and pcc(a2["hid2"], b2["hid2"]) >= 0.999
    assert pcc(a2["k_cache2"], b2["k_cache2"]) >= 0.9999
    assert (
        pcc(a["hid1"], b["hid1"]) >= 0.99 and pcc(a["hid2"], b["hid2"]) >= 0.99
    )  # (8 random layers compound: ~0.99 in every mode)
    assert (
        b2["ctx_k_pcc"] >= 0.9999 and b2["hid_pcc"] >= 0.998 and b2["hid_pcc2"] >= 0.998
    ), b2  # (the chain: 0.9990; the bf16 partials on random weights)
    assert min(b2["top16_overlap"]) >= 14 and min(b2["top16_overlap2"]) >= 14 and b2["plain_eq"] >= 6, b2
    bc, cc = b["block_census"], b["ctx_census"]
    assert bc["mk_a0"] == 1 and bc["mk_ba"] == 7 and bc["mk_b_last"] == 1 and bc["sdpa"] == 8, bc
    assert cc["mk_ctx"] == 1 and cc["concat"] == 1 and cc.get("ds_mm", 0) == 0, cc
    assert b["ctx_launches"] == 3 and b["step_launches"] < a["step_launches"] // 3, (
        b["ctx_launches"],
        b["step_launches"],
        a["step_launches"],
    )


@pytest.mark.parametrize("tp", [8, 2])
def test_mk_cb_model(tp):
    """the owner kernels' CB traffic (tt/dflash2_mk.cb_ops: the kernels' push / pop chunks per RISC per phase) over the
    four program kinds at the real drafter's geometry: balanced per phase (one producer, one consumer, equal pages)
    and no chunk straddles a CB's end with the per-phase rewinds; without the rewinds the model reproduces the
    2026-10-23 01:02Z watcher assert (CB_OUT: the SWI / HEADS chunks after RMS / CONV left the pointers mid-CB)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_mk as DM

    nqh, nkvh, ht, it = 32 // tp, 8 // tp, 4, 12288 // tp // 32
    nts = max(1, -(-it // 64))
    geo = dict(nqh=nqh, nkvh=nkvh, ht=ht, nts=nts, it=it)
    for kind in ("a0", "ba", "b_last", "ctx"):
        worst = DM.cb_check(DM.program_phase_types(kind), **geo)
        assert worst["OUT"] <= DM.cb_pages(ht, nts)["OUT"], (kind, worst)
    straddled = False
    try:
        DM.cb_check(DM.program_phase_types("ba"), rewind=False, **geo)
    except AssertionError as e:
        straddled = "straddles" in str(e)
    assert straddled, "the model must reproduce the no-rewind straddle (the 2026-10-23 watcher assert)"
    # the second silicon defect (01:52Z taps: the block's k heads read stale CB_OUT pages): a CB with two consumer RISCs
    # over the program -- the model with the k heads popping OUT (as the kernels did before CB_KOUT) must flag it
    two = False
    orig = DM.cb_ops

    def two_consumers(phases, k, **kw):
        ops = orig(phases, k, **kw)
        for r in ("ncrisc", "trisc"):
            ops[r] = [[("OUT", op, n) if cb == "KOUT" else (cb, op, n) for cb, op, n in ph] for ph in ops[r]]
        return ops

    DM.cb_ops = two_consumers
    try:
        DM.cb_check(DM.program_phase_types("ba"), **geo)
    except AssertionError as e:
        two = "pop RISCs" in str(e)
    finally:
        DM.cb_ops = orig
    assert two, "the model must flag a CB with two consumer RISCs"
    # the cache-row model of that defect: the block's K + 1 rows garbage among the n_ctx compared positions -> the
    # 0.96-0.98 cache PCC of the taps row (every ctx row right); exact rows -> 1.0
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(260, 128, generator=g)
    dev = ref.clone()
    dev[252:260] = torch.randn(8, 128, generator=g)
    assert 0.95 <= pcc(dev, ref) <= 0.99 and pcc(ref, ref) >= 0.999999, pcc(dev, ref)


def test_topk_rows_model():
    """kernels/spec/df2_topk.cpp on the host (dflash2_kind.topk_rows_host: the two-stage exact top-k of the draft rows
    with order-preserving int32 keys, ties to the lower column) == torch.topk's sorted values on rows 1 .. K for
    random fp32 logits with negatives, exact ties and -0 / +0; the ids agree where the values are unique; the
    padding is -FLT_MAX; and the kind's "kernel" top-k mode gives the same drafts as "ttnn" on the host"""
    g = torch.Generator().manual_seed(3)
    V, K, k = 2420 * 32, 7, 16
    x = torch.randn(32, V, generator=g) * 3
    x[2, 100] = x[2, 5000]  # an exact tie
    x[3, :1000] = -0.0
    x[4, 77] = float("inf")
    vals, ids = DK.topk_rows_host(x, K, k)
    for r in range(1, K + 1):
        tv, ti = torch.topk(x[r], k, dim=-1, largest=True, sorted=True)
        assert torch.equal(vals[r, :k], tv), (r, vals[r, :k], tv)
        assert torch.equal(torch.gather(x[r], 0, ids[r, :k]), tv), r
        uniq = tv.unique().numel() == k
        if uniq:
            assert torch.equal(ids[r, :k], ti.long()), r
    assert (
        bool((vals[:, k:] <= -3.0e38).all())
        and bool((vals[0] <= -3.0e38).all())
        and bool((vals[K + 1 :] <= -3.0e38).all())
    )
    os.environ["GLM53_DF2_TOPK"] = "kernel"
    try:
        b = _run(8, "fold", conv="fused", layers=2, mm="ds", conv_res=True, layer="mk")
    finally:
        os.environ.pop("GLM53_DF2_TOPK", None)
    a = _run(8, "fold", conv="fused", layers=2, mm="ds", conv_res=True, layer="mk")
    assert a["drafts"] == b["drafts"] and a["drafts2"] == b["drafts2"], (a["drafts"], b["drafts"])
    assert b["block_census"]["topk_rows"] == 2 and "topk" in b["block_census"], b["block_census"]


def test_ds_geometry():
    """tt/dflash2_mm.py host arithmetic: readers / padding / per-core L1 of every drafter matmul at TP8 and TP2 (the
    real config's shapes): no padding but the lm_head (sliced), the TP2 lm_head (2,420 column tiles) over the L1
    budget -> stays an mm_stream program; everything else fits"""
    from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_mm as DM

    H, I, V, nh, nkv, hd, ng = 4096, 12288, 154880, 32, 8, 128, 256
    for tp in (8, 2):
        shapes = dict(fc=(H // tp, torch.bfloat16), kp=(4 * ng, torch.bfloat16), qkv=((nh + 2 * nkv) // tp * hd, torch.bfloat16),
                      o=(H, torch.float32), ug=(2 * I // tp, torch.bfloat16), down=(H, torch.float32),
                      lm_head=(V // tp, torch.float32), hproj=(256, torch.float32))  # fmt: skip
        plans = {k: DM.ds_plan(n, out) for k, (n, out) in shapes.items()}
        print(tp, plans)
        for k, p in plans.items():
            assert p["n_pad"] == shapes[k][0] or k == "lm_head", (tp, k, p)
            assert p["fits"] or (tp == 2 and k == "lm_head"), (tp, k, p)
        assert plans["qkv"]["readers"] == 3 and plans["hproj"]["readers"] == 1 and plans["o"]["readers"] == 2, plans


if __name__ == "__main__":
    for tp, mode, conv in [(8, "mean", "ops"), (8, "fold", "fused"), (2, "fold", "fused")]:
        print(_run(tp, mode, conv=conv))


def test_sdpa_fold_view():
    """rung 3: the folded SDPA Q form [1, nkvh, G 32, hd] is the per-q-head form's bytes (the owner kernels' page
    h HT + c lands in tile (h // G, h % G, c) of the fold), the o matmul reads the SDPA output by address in both
    forms, and the cores per kv head follow the grid / the op's 64-core tree cap"""
    from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_mk as MK

    nqh, G, hd = 16, 4, 128
    q = torch.randn(1, nqh, 32, hd)
    f = MK.fold_view(q, G)
    assert f.shape == (1, nqh // G, G * 32, hd)
    # tile (g, j, c) of the fold == tile (g G + j, c) of the per-head form, at the same flat page index
    tiles = lambda t: t.reshape(-1, 32, hd).reshape(-1, 32, hd // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)  # noqa
    assert torch.equal(tiles(q), tiles(f))
    for h in range(nqh):
        assert torch.equal(f[0, h // G, (h % G) * 32:(h % G + 1) * 32], q[0, h])
    assert f.reshape(1, nqh, 32, hd).data_ptr() == q.data_ptr()  # a view, no copy
    assert MK.sdpa_cores(4, 130) == 32 and MK.sdpa_cores(1, 130) == 64 and MK.sdpa_cores(1, 120) == 64
    assert MK.sdpa_cores(4, 130, "8") == 8 and MK.sdpa_cores(16, 130) == 8
    assert MK.FOLD_MODES == ("spec", "off")
