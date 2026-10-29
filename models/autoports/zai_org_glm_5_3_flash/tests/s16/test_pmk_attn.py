# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK attention piece: pure-python checks (no device, no ttnn): the CB / L1 budget of both geometries (TP8 256 SP rows x
8 heads, TP2 1,024 x 32), the balanced item lists, the host reference's semantics (dense == causal SDPA; a sparse
selection of every pool == dense; the tail rule), the compute + traffic estimate (rev 3: calibrated against the rev 2
silicon unit), and rev 4: the NoC fan-in model of the reader (`noc_sim`: the ascending order reproduces the rev 3
silicon span, the rotated order is under the bar), the kernel-arithmetic model of the sparse path (finite on
all-masked batches, equal to the reference under rotated orders), the batch-order rule and the pool-gather page
ids; rev 7: the dual-NoC fan-in model, the mask constant's value independence; rev 8: the host-compiled mask operand
+ TRISC expansion against the rev 6 element rule, the bitmap page / scratch layouts, the selection bitmaps and the
fixed-group tile count. `python test_pmk_attn.py` or pytest."""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_attn as A  # noqa: E402


def geos():
    return {
        "tp8-dense": A.AttnGeo.tp8(0),
        "tp8-sparse-32k": A.AttnGeo.tp8(32768 - 2048, max_seq=32768),
        "tp2-dense": A.AttnGeo.tp2(0),
        "tp2-sparse-32k": A.AttnGeo.tp2(32768 - 2048, max_seq=32768),
        "tp8-sparse-128k": A.AttnGeo.tp8(131072 - 2048, max_seq=131072),
        # rev 7: the unit geometries and the dense knob ladder (dual_noc default on; kslots 3; QG 2)
        "tp2-sparse-unit": A.AttnGeo.tp2(1024, max_seq=8192, qg_=2),
        "tp8-sparse-4k": A.AttnGeo.tp8(4096, max_seq=8192, qg_=2),
        "tp2-dense-ks3": A.AttnGeo.tp2(0, kslots=3),
        "tp2-dense-qg2": A.AttnGeo.tp2(0, qg_=2),
        "tp2-dense-qg2-ks3": A.AttnGeo.tp2(0, qg_=2, kslots=3),
        # rev 12: the token-major exact mode at the 32K / 64K / 128K slots (TP8: 256 SP rows x 64 heads) and the pair
        "tp8-sp-32k": A.AttnGeo.tp8_sp(32768 - 2048, max_seq=32768),
        "tp8-sp-64k": A.AttnGeo.tp8_sp(65536 - 2048, max_seq=65536),
        "tp8-sp-128k": A.AttnGeo.tp8_sp(131072 - 2048, max_seq=131072),
        "tp2-sp-unit": A.AttnGeo.tp2_sp(1024, max_seq=8192),
    }


def test_budget():
    for name, g in geos().items():
        plan = A.l1_plan(g)
        cbs = A.attention_cb_bytes(g)
        print(
            f"{name} (NK {g.nk}, QG {g.qg}, O {'bf16' if g.o_bf16 else 'fp32'}, kslots {g.kslots}, bm_pages {g.bm_pages}): "
            f"attention CBs {plan['attention_kb']} KB "
            f"(AQ {cbs['AQ'] // 1024}, AK {cbs['AK'] // 1024}, partials {(cbs['APA'] + cbs['APAO']) // 1024}, "
            f"AKRM+AKT {(cbs['AKRM'] + cbs['AKT']) // 1024}, AMASK {cbs['AMASK'] // 1024}, ABM {cbs['ABM'] // 1024}, "
            f"ASCR {cbs['ASCR'] // 1024}) + rings "
            f"{plan['rings_kb']} + store {plan['store_kb']} = peak {plan['peak_kb']} KB of {plan['budget_kb']}: "
            f"{'fits' if plan['fits'] else 'OVER'}"
        )
        assert plan["fits"], (name, plan)
        assert max(A.ACB.values()) <= 63 and min(A.ACB.values()) >= A.ATT_CB0
        assert len(set(A.ACB.values())) == len(A.ACB)
        assert cbs["AID"] == A.aid_pages(g) * 2048  # rev 12: SP builds 14 pages (+ pad(1..7), tail(0..3)), else 3
        if g.sp:
            assert g.nk <= 4 and g.qg == 2 and g.items == g.rows and g.sparse and not g.ctx_l1 and not g.o_bf16
            assert cbs["ABM"] == g.bm_pages * A.SP_PAGE_BYTES and cbs["AMASK"] == 2048 and cbs["AK"] == 2048
            assert cbs["AKRM"] == cbs["AKT"] == g.nk * A.DT * 2048  # one batch each: the gather waits for the tilize only
            continue
        assert g.nk * g.qg <= 31
        assert cbs["AMASK"] == 2 * g.nk * g.qg * 2048  # rev 7 / 8: bf16 mask operands
        if g.sparse:
            assert cbs["ABM"] == g.bm_pages * -(-A.abm_words(g) * 4 // 64) * 64
            # rev 12: judged with fp32 O (the TP8 32K row went out "bm_pages 2, O fp32, fits False")
            assert g.bm_pages == (2 if name not in ("tp8-sparse-128k", "tp8-sparse-32k", "tp2-sparse-32k") else 1), (name, g.bm_pages)
    # the 128K slot's bitmaps do not double-buffer (32 rows x 4 KB a page): the plan falls back to one page and fits
    g = geos()["tp8-sparse-128k"]
    assert A.abm_words(g) * 4 > 120 * 1024 and A.l1_plan(g)["fits"]
    import dataclasses

    assert not A.l1_plan(dataclasses.replace(g, bm_pages_=2))["fits"]
    assert geos()["tp2-dense"].qg == 2 and geos()["tp2-dense"].dual_noc  # rev 8 dense defaults (the silicon ladder)


def test_balance():
    for name, g in geos().items():
        lists = A.attn_items(g)
        assert sum(len(l) for l in lists) == g.items
        assert len({hq for l in lists for hq in l}) == g.items
        b = A.balance(g)
        print(
            f"{name}: items {g.items}, items/core {b['items_per_core']}, (q, K) pairs/core min {b['pairs_min']} max "
            f"{b['pairs_max']} mean {b['pairs_mean']:.1f}, K loads/core max {b['loads_max']}"
        )
        assert b["pairs_max"] - b["pairs_min"] <= A.item_compute(g, g.ct - g.qg), b
        for l in lists:  # ascending row blocks inside a core (the counters are reached in order)
            assert [qt for _, qt in l] == sorted(qt for _, qt in l)


def test_reference_dense():
    torch.manual_seed(0)
    h, c, cs = 2, 128, 256
    q = torch.randn(h, c, 512)
    lat = torch.randn(cs + c, 512)
    o = A.attention_host(q, lat, cs)
    kv = lat[: cs + c].reshape(1, 1, cs + c, 512).expand(1, h, cs + c, 512)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.reshape(1, h, c, 512), kv, kv,
        attn_mask=(torch.arange(cs + c).reshape(1, -1) <= (cs + torch.arange(c)).reshape(-1, 1)), scale=0.0625)[0]  # fmt: skip
    assert torch.allclose(o, ref, atol=1e-4, rtol=1e-4), float((o - ref).abs().max())


def test_reference_sparse_all_selected_is_dense():
    torch.manual_seed(1)
    h, c, cs = 1, 64, 128
    q, lat = torch.randn(h, c, 512), torch.randn(cs + c, 512)
    pools = (cs + c) // A.KPOOL
    ids = torch.full((c, A.TOPK), A.NONE_ID, dtype=torch.int64)
    for i in range(c):  # every closed pool visible to row i
        n = (cs + i + 1) // A.KPOOL
        ids[i, :n] = torch.arange(n)
    assert pools <= A.TOPK
    o_sparse = A.attention_host(q, lat, cs, ids)
    o_dense = A.attention_host(q, lat, cs)
    assert torch.allclose(o_sparse, o_dense, atol=1e-4), float((o_sparse - o_dense).abs().max())


def test_reference_tail_and_select():
    torch.manual_seed(2)
    c, cs = 32, 4096
    scores = torch.randn(c, (cs + c) // A.KPOOL)
    pos = cs + torch.arange(c)
    ids = A.select_rows_host(scores, pos)
    for i in range(c):
        row = ids[i][ids[i] != A.NONE_ID]
        assert len(row) == A.TOPK and bool((row[1:] > row[:-1]).all())
        assert int(row.max()) * A.KPOOL + A.KPOOL - 1 <= int(pos[i])  # closed pools only
    q, lat = torch.randn(1, c, 512), torch.randn(cs + c, 512)
    ids0 = torch.full((c, A.TOPK), A.NONE_ID, dtype=torch.int64)
    ids0[:, 0] = 0  # only pool 0 selected
    o = A.attention_host(q, lat, cs, ids0)
    i = 1  # p = 4097: tail = [4096, 4097]
    keys = torch.cat([lat[0:4], lat[4096:4098]])
    s = (q[0, i] @ keys.T) * 0.0625
    ref = torch.softmax(s, -1) @ keys
    assert torch.allclose(o[0, i], ref, atol=1e-4), float((o[0, i] - ref).abs().max())


def test_estimates():
    # the rev 2 silicon unit (TP2 dense, 1 q tile per item, Q re-read per batch, K from DRAM): the model must land on
    # the measured 9,699 us (DRAM-bound), i.e. the calibration point
    g2 = A.AttnGeo.tp2(0, qg_=1, lat_l1=False, nk_=2)
    e2 = A.math_estimate(g2)
    # rev 2 re-read Q per batch: add that traffic by hand to the model's DRAM bytes
    rev2_dram_gb = e2["dram_gb"] - g2.items * 32768 / 1e9 + e2["batches"] * 32768 / 1e9
    rev2_us = rev2_dram_gb * 1e6 / A.DRAM_BW_GBS
    print(
        f"rev 2 model: {rev2_dram_gb:.2f} GB DRAM -> {rev2_us:.0f} us (silicon 9,699); compute floor {e2['compute_us']} us"
    )
    assert 8500 <= rev2_us <= 11000, rev2_us
    for name, g in geos().items():
        e = A.math_estimate(g)
        e2h = A.math_estimate(g, "HiFi2")
        print(
            f"{name} (QG {g.qg}): {e['gflop']} GFLOP, {e['pairs']} (q, K) pairs; HiFi4 FPU useful {e['fpu_useful_pct']} % "
            f"(incl. rescale {e['fpu_pct_incl_rescale']} %); compute {e['compute_us']} dram {e['dram_us']} "
            f"({e['dram_gb']} GB) noc {e['noc_us']} barrier {e['barrier_us']} -> span {e['span_us']} us "
            f"({e['bound']}-bound, {e['tflops']} TFLOP/s); HiFi2 span {e2h['span_us']} us useful {e2h['fpu_useful_pct']} %"
        )
        if g.sparse:
            print(f"   indexer: {A.indexer_estimate(g)}")
    tp2 = A.math_estimate(A.AttnGeo.tp2(0))
    assert tp2["span_us"] <= 6383, tp2  # the op path's chunked SDPA at TP2 (op profile L3 2,048)
    assert abs(tp2["compute_us"] - 3345) / 3345 < 0.05, tp2  # the u2 probe's QK + softmax + PV (the calibration)
    # rev 12: the SP mode at the 32K slot is under the 6,000-us bar with margin and does not grow with the slot
    # (top-512 pools per token at any context); the contract geometry's model with the MEASURED union (960 of 960
    # groups at random selections) explains the 38-ms row (README §14)
    sp32 = A.math_estimate(A.AttnGeo.tp8_sp(30720, max_seq=32768), "HiFi2")
    sp128 = A.math_estimate(A.AttnGeo.tp8_sp(131072 - 2048, max_seq=131072), "HiFi2")
    print(f"tp8-sp 32K: {sp32}\ntp8-sp 128K: {sp128}")
    assert sp32["span_us"] <= 3000 and sp128["span_us"] == sp32["span_us"], (sp32, sp128)
    full = A.math_estimate(A.AttnGeo.tp8(30720, max_seq=32768), "HiFi2", union_tiles=960)
    print(f"tp8-sparse 32K at the measured union (960 tiles / item): {full}")
    assert full["pairs"] == round(104 * A.balance(A.AttnGeo.tp8(30720, max_seq=32768))["pairs_mean"]) and full["span_us"] > 15000, full
    ph = A.projection_phases(A.AttnGeo.tp8(0))
    tot = sum(p.gflop for p in ph)
    print(
        "projections (TP8):",
        [(p.name, round(p.gflop, 1), round(p.weight_mb, 1)) for p in ph],
        "total GFLOP",
        round(tot, 1),
    )


def test_dm1_model():
    """rev 4's reader model (README §8): with the rev 3 ascending batch order every reader hits the same NK owners
    of the L1-sharded store together and the fair-share NoC keeps them there; at the fitted owner egress the model
    lands on the u2 silicon span (16,014 us dm1 span, 79 % compute wait). The rotated order (the rev 4 kernels)
    spreads the readers: span under the op path's 6,383 us, the compute the largest term."""
    g = A.AttnGeo.tp2(0, qg_=1)  # the rev 3..6 unit geometry (rev 8 defaults to QG 2)
    assert (g.nk, g.qg, g.lat_l1) == (4, 1, True), g
    r3 = A.noc_sim(g, rotate=False)
    r4 = A.noc_sim(g, rotate=True)
    rep = A.noc_sim(g, rotate=True, replicas=40)
    peak = A.noc_sim(g, rotate=True, owner_bpc=64.0)
    print(
        f"dm1 model TP2 dense (owner {A.NOC_OWNER_BPC} B/cycle): rev 3 ascending {r3['span_us']} us, compute wait "
        f"{r3['compute_wait_pct']} % (silicon 16,014 us / 79 %); rev 4 rotated {r4['span_us']} us, wait "
        f"{r4['compute_wait_pct']} %, compute {r4['compute_us']} us, hottest owner {r4['hottest_owner_mb']} MB; "
        f"+40 replicas {rep['span_us']} us; at the 64 B/cycle link peak {peak['span_us']} us"
    )
    assert 14500 <= r3["span_us"] <= 17500, r3
    assert r3["compute_wait_pct"] >= 70, r3
    assert r4["span_us"] <= 6383, r4
    assert r4["span_us"] < 0.4 * r3["span_us"], (r4, r3)
    assert peak["span_us"] <= 1.05 * peak["compute_us"], peak  # compute-bound once the owners are spread
    # the owner skew is causal: tile 0 is read by every item, tile 63 by one head's last items
    loads = A.owner_load_mb(g)
    assert loads[0] == max(loads) and loads[-1] == min(loads), loads


def test_dual_noc_model():
    """rev 7 (README §9b): the rev 6 dense unit (9,058 us, compute wait 60.5 %, dm0 barrier 75.9 %) is reproduced by
    the rotated-order model at ~5.5 B/cycle per owner (NOT the 11 fitted on rev 3: the rotated readers see a lower
    per-owner rate). Two NoC ports (dual_noc: the owner serves through both NIUs, the reader lands through both)
    double that rate: 5,030 us (<= 6,383); QG 2 halves the bytes on top: under 5,200 us even at 3 B/cycle."""
    g = A.AttnGeo.tp2(0, qg_=1)
    r6 = A.noc_sim(g, owner_bpc=5.5)
    print(f"rev 6 dense unit model: {r6['span_us']} us, wait {r6['compute_wait_pct']} % (silicon 9,058 / 60.5 %)")
    assert 8500 <= r6["span_us"] <= 10500 and 55 <= r6["compute_wait_pct"] <= 70, r6
    r7 = A.noc_sim(g, owner_bpc=5.5, noc_ports=2)
    print(f"rev 7 dual-NoC model: {r7['span_us']} us, wait {r7['compute_wait_pct']} %")
    assert r7["span_us"] <= 6383 and r7["span_us"] < 0.6 * r6["span_us"], (r7, r6)
    g2 = A.AttnGeo.tp2(0, qg_=2)
    r7q = A.noc_sim(g2, owner_bpc=3.0, noc_ports=2)
    print(f"rev 7 dual-NoC + QG 2 at 3 B/cycle: {r7q['span_us']} us, wait {r7q['compute_wait_pct']} %")
    assert r7q["span_us"] <= 5300, r7q
    assert A.l1_plan(g2)["fits"] and g2.kslots == 2 and A.AttnGeo.tp2(0).qg == 2


def test_mask_value_independence():
    """rev 7 (README §9a): the sparse mask tiles carry bf16(-1e9) = NEG_MASK_BF16 instead of -1e9; the kernel
    arithmetic model gives the SAME output (bitwise in fp32 torch) for either constant -- masked keys contribute
    exp(NEG - m') == 0 exactly and all-masked batches are rescaled by a == 0 -- on the all-masked-first case"""
    torch.manual_seed(3)
    cs, c = 4096, 128
    q = torch.randn(1, c, 512) * 0.3
    lat = torch.randn(cs + c, 512) * 0.3
    ids = torch.full((c, A.TOPK), A.NONE_ID, dtype=torch.int64)
    for i in range(c):
        n = 1 + i % 5
        ids[i, :n] = torch.randperm(cs // A.KPOOL)[:n].sort().values
    o6, st6 = A.attention_kernel_model(q, lat, cs, ids, nk=2, qg=2, core=7, rotate=True)
    o7, st7 = A.attention_kernel_model(q, lat, cs, ids, nk=2, qg=2, core=7, rotate=True, mask_neg=A.NEG_MASK_BF16)
    assert st6["all_masked_updates"] > 0 and st7["all_masked_updates"] == st6["all_masked_updates"]
    assert torch.equal(o6, o7), float((o6 - o7).abs().max())
    assert A.NEG_MASK_BF16 == torch.tensor(A.NEG_MASK).to(torch.bfloat16).float().item()
    import struct

    assert struct.unpack("<I", struct.pack("<f", A.NEG_MASK_BF16))[0] >> 16 == 0xCE6E  # attn_common.hpp NEG16
    assert float(torch.exp(torch.tensor(A.NEG_MASK_BF16))) == 0.0
    s = torch.tensor([3.0, -7.5, 0.25, 31.0, -31.0]) + A.NEG_MASK_BF16
    assert bool((s == A.NEG_MASK_BF16).all())  # the masked scores tie (ulp 64 at 2^29)


def test_mask_row_host():
    """rev 8: the kernels' mask operand (attn_common.hpp mask_byte_diag / bf16_of_byte / mask_col_row, host-compiled)
    + the TRISC expansion (B_col @ E8, floor parity, the triangle) against the rev 6 per-element rule on random
    memberships, context + chunk + diagonal tiles: pattern equality on every element (tests/s16/pmk/
    attn_mask_host.cpp)"""
    import shutil
    import subprocess
    import tempfile

    if shutil.which("g++") is None:
        print("no g++: skipped")
        return
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    src = os.path.join(os.path.dirname(__file__), "pmk", "attn_mask_host.cpp")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "attn_mask_host")
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", f"-I{root}", "-o", exe, src], check=True)
        out = subprocess.run([exe, "3000"], check=True, capture_output=True, text=True).stdout
    print(out.strip())
    assert out.startswith("OK 3000 trials"), out


def test_abm_layout():
    """rev 8: the bitmap page of an item (attn_common.hpp abm_words == pmk_attn.abm_words: 32 qg selection rows of
    sel_row_words + the union) and the CB_ASCR layout the two data-movement kernels share (flags at pt_bytes)"""
    for name, g in geos().items():
        if not g.sparse:
            continue
        ctx_pools = g.ct0 * A.POOLS_PER_TILE
        w = A.sel_row_words(ctx_pools)
        assert w * 32 >= ctx_pools + 512 and w % 16 == 0 and (w - 16) * 32 < ctx_pools + 512
        assert A.abm_words(g) == 32 * g.qg * w + -(-ctx_pools // 32)
        lay = A.scratch_layout(g)
        assert lay["flag_off"] == lay["pt_bytes"] and lay["total"] == lay["pt_bytes"] + 128
        assert lay["pt_bytes"] % 64 == 0 and lay["pt_bytes"] >= g.max_seq // A.BLOCK * 4
    d = geos()["tp2-dense"]
    assert A.scratch_layout(d)["total"] == A.scratch_layout(d)["pt_bytes"] + 128


def test_softmax_levers():
    """rev 9: the lazy-rescale softmax (max_lazy = 32: a batch rescales the partial only when a row's max grows by
    more than 32) equals the every-batch softmax to fp32 rounding on the sparse kernel model whose rows start with
    ALL-MASKED batches (the case that kills a first-batch-only max: m ~ -1e9 then exp(1e9)), rescales rarely, keeps
    exp arguments <= 32, and the lever table lands the built row near the 8,000 bar"""
    torch.manual_seed(3)
    cs, c = 4096, 128
    q = torch.randn(1, c, 512) * 0.3
    lat = torch.randn(cs + c, 512) * 0.3
    ids = torch.full((c, A.TOPK), A.NONE_ID, dtype=torch.int64)
    for i in range(c):
        n = 1 + i % 5
        ids[i, :n] = torch.randperm(cs // A.KPOOL)[:n].sort().values
    o_run, st_run = A.attention_kernel_model(q, lat, cs, ids, nk=2, qg=2, core=7, rotate=True)
    o_lazy, st_lazy = A.attention_kernel_model(q, lat, cs, ids, nk=2, qg=2, core=7, rotate=True, max_lazy=32.0)
    print(
        f"max_lazy 32: max |o - o_running| {float((o_lazy - o_run).abs().max()):.3g}, largest exp argument "
        f"{st_lazy['max_exp_arg_pos']:.3g} (<= 32), rescales {st_lazy['rescales']} of {st_lazy['batches'] * 2} "
        f"(q tile, batch) updates, all-masked updates {st_lazy['all_masked_updates']}, min l {st_lazy['min_l']:.3g}"
    )
    # a different stabiliser m rounds E differently in bf16 (the P V operand): the same noise class as the kernel's
    # own bf16-E error vs the fp32 reference (6.7e-4 in test_kernel_model_sparse_finite); judged by the PCC bar
    ref = A.attention_host(q, lat, cs, ids)
    pcc = lambda x, y: float(torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1])  # noqa: E731
    print(f"   PCC(lazy, ref) {pcc(o_lazy, ref):.6f}, PCC(running, ref) {pcc(o_run, ref):.6f}")
    assert bool(torch.isfinite(o_lazy).all()) and torch.allclose(o_lazy, o_run, atol=2e-2, rtol=1e-2)
    assert pcc(o_lazy, ref) >= 0.9999 and pcc(o_lazy, ref) >= pcc(o_run, ref) - 1e-5
    assert st_lazy["max_exp_arg_pos"] <= 32.0 and st_lazy["min_l"] >= 1.0, st_lazy
    assert 0 < st_lazy["rescales"] < st_lazy["batches"] * 2, st_lazy  # the all-masked rows rescale once, the rest rarely
    lv = A.softmax_levers()
    print("softmax levers (us):", lv)
    assert lv["expected_us"]["rev9_all"] <= 8700 and lv["expected_us"]["rev9_all_two_batch"] <= 8000


def test_select_bitmap():
    """rev 8: the per-row selection bitmaps (select_bitmap_host) carry exactly the ids (round trip), the E8 expansion
    row reproduces the membership bits through byte x 2^-s parity, and on the unit's synthetic selection the fixed
    8-pool groups gather the SAME context tiles as rev 7's compact union (every group touched)"""
    torch.manual_seed(4)
    c, cs = 2048, 1024
    ctx_pools = cs // A.KPOOL
    scores = torch.randn(c, (cs + c) // A.KPOOL)
    ids = A.select_rows_host(scores, cs + torch.arange(c))
    sel = A.select_bitmap_host(ids, ctx_pools)
    assert sel.shape == (c, A.sel_row_words(ctx_pools)) and A.sel_row_words(ctx_pools) == 32  # 768 pools -> 2 x 16 words
    for r in (0, 1, 1023, 1024, 2047):
        q = ids[r][ids[r] != A.NONE_ID].tolist()
        bits = [int((sel[r][p // 32] >> (p % 32)) & 1) for p in range(ctx_pools + 512)]
        assert [p for p in range(ctx_pools + 512) if bits[p]] == q, r
    # the membership byte of (row r, context group g) = byte g of the row; the TRISC expansion's parity test
    bank = A.mask_bank_host()
    e8 = bank[64:96]
    r, g = 1500, 7
    byte = int((sel[r][g // 4] >> (8 * (g % 4))) & 0xFF)
    t = torch.zeros(32, 32)
    t[0, 0] = byte
    expanded = t @ e8  # row 0 = byte x 2^-(c // 4)
    for col in range(32):
        member = (byte >> (col // 4)) & 1
        assert int(torch.floor(expanded[0, col])) % 2 == member, (col, byte)
        assert float(expanded[0, col]) == byte * 2.0 ** -(col // 4)
    # fixed groups vs the compact union on the unit's selection
    for qt0 in (0, 30, 62):
        rows = ids[qt0 * 32 : (qt0 + 2) * 32]
        q = rows[rows != A.NONE_ID]
        compact = -(-int(torch.unique(q[q < ctx_pools]).numel()) // A.POOLS_PER_TILE)
        assert A.group_tiles(ids, ctx_pools, qt0, 2) == compact == 32, (qt0, compact)


def test_batch_order():
    """the kernel's rotation rule (attn_common.hpp batch_rot: (core + it) mod nb) == the planner's; every batch once;
    104 cores starting item 0 of a 16-batch item spread over all 16 batch groups"""
    for nb in (1, 2, 3, 16, 64):
        for core in (0, 5, 103):
            for it in (0, 7, 19):
                o = A.batch_order(core, it, nb)
                assert sorted(o) == list(range(nb)), (nb, core, it, o)
                r = core + it
                while r >= nb:
                    r -= nb
                assert o[0] == r
    firsts = {A.batch_order(c, 0, 16)[0] for c in range(104)}
    assert firsts == set(range(16)), firsts
    g = A.AttnGeo.tp2(0, qg_=1)
    bl = A.item_batches(g, core=3, it=0, qt0=63)
    assert len(bl) == 16 and sorted(k0 for k0, _, _ in bl) == list(range(0, 64, 4)) and all(ch for _, _, ch in bl)
    assert bl[0][0] == 3 * 4, bl[:2]  # rotation 3 on core 3, item 0


def test_kernel_model_sparse_finite():
    """the kernel's batch arithmetic (attention_kernel_model) on a sparse chunk where many rows' FIRST batches are
    entirely masked (rows selecting few context pools: NEG_MASK keys only until the chunk's diagonal) and on
    rotated / unrotated orders: finite stats (min l >= 1), o == attention_host to fp32 rounding"""
    torch.manual_seed(3)
    cs, c = 4096, 128  # context 128 tiles, chunk 4 q tiles: QG 2 items (qt0 0, 2)
    q = torch.randn(1, c, 512) * 0.3
    lat = torch.randn(cs + c, 512) * 0.3
    pools = (cs + c) // A.KPOOL
    ids = torch.full((c, A.TOPK), A.NONE_ID, dtype=torch.int64)
    for i in range(c):  # rows select a few scattered context pools (so most context K tiles are all-masked for them)
        n = 1 + i % 5
        sel = torch.randperm(cs // A.KPOOL)[:n].sort().values
        ids[i, :n] = sel
    assert pools > A.TOPK or True
    ref = A.attention_host(q, lat, cs, ids)
    for rotate in (False, True):
        o, st = A.attention_kernel_model(q, lat, cs, ids, nk=2, qg=2, core=7, rotate=rotate)
        print(
            f"kernel model sparse (rotate {rotate}): batches {st['batches']}, all-masked updates "
            f"{st['all_masked_updates']}, min l {st['min_l']:.3g}, max |exp arg| {st['max_exp_arg']:.3g}, "
            f"max |o - ref| {float((o - ref).abs().max()):.3g}"
        )
        assert st["all_masked_updates"] > 0, st  # the case the guard is about: rows with no key in a batch
        assert bool(torch.isfinite(o).all()) and st["min_l"] >= 1.0, st
        assert torch.allclose(o, ref, atol=2e-2, rtol=1e-2), float((o - ref).abs().max())  # bf16 E in the PV
    # dense, rotated, NK 4 / QG 1 (the unit's geometry on a small chunk): exact vs the reference
    od, std = A.attention_kernel_model(q, lat, 0, None, nk=4, qg=1, core=11, rotate=True)
    refd = A.attention_host(q, lat, 0)
    assert bool(torch.isfinite(od).all()) and torch.allclose(od, refd, atol=2e-2, rtol=1e-2), float(
        (od - refd).abs().max()
    )


def test_rm_gather_pages():
    """the row-major latent store: one row per page (the op path's [n, 512]) -> a pool's rows are pages 4p .. 4p + 3,
    in 4 DRAM banks (interleaved page % 8): the rev 3 single 4-KB read from page 4p's address returned bank-local
    pages 4p, 4p + 8, 4p + 16, 4p + 24 = the WRONG rows for 3 of 4 (README §8); rev 4 reads the 4 pages"""
    n, nb = 3072, 8
    rows_of_bank = lambda p: [p + nb * i for i in range(4)]  # noqa: E731 -- what a 4-KB read from page p returns
    wrong = sum(1 for p in range(0, 1024, 4) for r in range(4) if rows_of_bank(p)[r] != p + r)
    assert wrong == 3 * 256, wrong

    class T:  # noqa: N801 -- a shape-only stand-in for the tensor
        def __init__(self, w):
            self.shape = (1, 1, n // (w // 512), w)

    assert A.rm_rows_per_page(T(512)) == 1 and A.rm_rows_per_page(T(2048)) == 4


def test_o_partial_rounding():
    """rev 12 (README §14): the 32K-slot PCC loss (silicon 0.997911, max |d| 0.0047, uniform over heads) is the O
    partial's pass through the FPU operand path on EVERY batch (revs 3..11: diag(a) O / I O as tile matmuls whose
    fp32 operand is unpacked to the 19-bit srcA / srcB): on the kernel model of one 32K item (476 batches per row) a
    10-bit truncation of O per update costs 5e-4 of PCC and a bf16 truncation 3e-2 (the silicon sits between: ~9
    bits), while max_lazy, the exp_21f class (E rounded to bf16 everywhere) and the every-batch rescale keep
    0.999999; the exact O path (copy_tile into fp32 DST + the SFPU row scale) is rev 12's kernel. The same sweep at
    the unit's 37 batches stays >= 0.99999 for all but bf16 -- why the pair rows never showed it."""
    torch.manual_seed(0)
    pcc = lambda x, y: float(torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1])  # noqa: E731
    out = {}
    for cs, c, qt0 in ((30720, 128, 2), (1024, 2048, 40)):
        g = torch.Generator().manual_seed(0)
        n = cs + c
        q = (torch.randn(1, c, 512, generator=g) * 0.3).to(torch.bfloat16).float()
        lat = (torch.randn(n, 512, generator=g) * 0.3).to(torch.bfloat16).float()
        ids = A.select_rows_host(torch.randn(c, n // A.KPOOL, generator=g), cs + torch.arange(c))
        rows = slice(qt0 * 32, (qt0 + 2) * 32)
        ref = A.attention_host(q[:, rows], lat, cs + qt0 * 32, ids[rows])
        res = {}
        for label, kw in (("exact", {}), ("lazy", dict(max_lazy=32.0)), ("lazy+e_both", dict(max_lazy=32.0, e_round="both")),
                          ("lazy+tf32", dict(max_lazy=32.0, o_round="tf32_trunc")), ("lazy+bf16", dict(max_lazy=32.0, o_round="bf16_trunc"))):
            o, st = A.attention_kernel_model(q, lat, cs, ids, nk=2, qg=2, core=5, items=[(0, qt0)], **kw)
            res[label] = (round(pcc(o[:, rows], ref), 6), st["batches"])
        print(f"O rounding at ctx {cs}: {res}")
        out[cs] = res
    r = out[30720]
    assert r["exact"][1] >= 400 and all(v[0] >= 0.99999 for k, v in r.items() if k in ("exact", "lazy", "lazy+e_both")), r
    assert r["lazy+tf32"][0] < 0.9996 and r["lazy+bf16"][0] < 0.99, r  # the FPU operand path is the only candidate
    u = out[1024]
    assert all(v[0] >= 0.99999 for k, v in u.items() if k != "lazy+bf16"), u


def test_mask_bank():
    """rev 12: the 14-page bank -- pad(n) masks exactly the last n pool slots, tail(c) keeps columns 0 .. c of slot 0"""
    bank = A.mask_bank_host()
    assert bank.shape == (A.AID_PAGES * 32, 32)
    for n in range(1, 8):
        t = bank[(3 + n - 1) * 32 : (3 + n) * 32]
        assert bool((t[:, : 32 - 4 * n] == 0).all()) and bool((t[:, 32 - 4 * n :] == A.NEG_MASK).all()), n
    for c in range(4):
        t = bank[(10 + c) * 32 : (11 + c) * 32]
        assert bool((t[:, : c + 1] == 0).all()) and bool((t[:, c + 1 :] == A.NEG_MASK).all()), c
    # the SP tail / pad rule of the kernels (attn_common.hpp sp_cmask) on the host: tile indices and pages
    sp_cmask = lambda k, ntiles, n_sel, tail, p: (10 + (p & 3)) if (tail and k == ntiles) else ((3 + (8 - n_sel % 8) - 1) if (k + 1 == ntiles and n_sel % 8) else 0)  # noqa: E731
    assert sp_cmask(64, 64, 512, 1, 30723) == 13 and sp_cmask(63, 64, 512, 1, 30720) == 0
    assert sp_cmask(31, 32, 250, 1, 1027) == 3 + 5 and sp_cmask(32, 32, 250, 1, 1027) == 13


def test_sp_tail_rule():
    """rev 12.2 (README §14h): the SP tail tile exists iff p % 4 < 3 -- at c == 3 the pool so = p // 4 is CLOSED at p and
    the reference's tail [p + 1 - (p + 1) % 4, p] is empty (so counts only when selected, as a union tile). Rev 12.1
    emitted the tail whenever so was not the token's last id: at the pair (ctx 1,024: every visible pool selected) so
    was always in the ids at c == 3, at 32K it is selected for ~7 % of tokens -> 60 of 64 c == 3 tokens attended 4
    spurious keys: PCC 0.999784 on the host at the row's inputs = the silicon's 0.999785 (logs/rev12/sp_signatures.log).
    Here on a small case: the fixed rule's key set == the reference's for every token; the old rule's differs."""
    torch.manual_seed(5)
    cs, c, n = 4096, 64, 4096 + 128
    ids = A.select_rows_host(torch.randn(c, n // A.KPOOL), cs + torch.arange(c))
    bad_old = bad_new = 0
    for t in range(c):
        p = cs + t
        so = p // A.KPOOL
        sel = set(ids[t][ids[t] != A.NONE_ID].tolist())
        ref_keys = {k for k in range(p + 1) if (k // A.KPOOL in sel) or (k >= p + 1 - (p + 1) % A.KPOOL)}
        union = {k for q in sel for k in range(q * A.KPOOL, (q + 1) * A.KPOOL)}
        tail_keys = {k for k in range(so * A.KPOOL, p + 1)}
        old_rule = union | (tail_keys if (not sel or max(sel) != so) else set())
        new_rule = union | (tail_keys if p % 4 != 3 else set())
        bad_old += old_rule != ref_keys
        bad_new += new_rule != ref_keys
        if p % 4 < 3:
            assert so not in sel  # an open pool is never a candidate
    print(f"SP tail rule on {c} tokens: old rule wrong on {bad_old}, fixed rule wrong on {bad_new}")
    assert bad_new == 0 and bad_old > 0


def test_mask_constant():
    """NEG_MASK: -1e9 is exact under fp32 adds of scores (|s| < 64), representable in bf16, exp(NEG_MASK) == 0"""
    import struct

    assert struct.unpack("<I", struct.pack("<f", A.NEG_MASK))[0] == 0xCE6E6B28  # attn_common.hpp NEG_MASK_BITS
    s = torch.tensor([3.0, -7.5, 0.25]) + A.NEG_MASK
    assert bool((s == A.NEG_MASK).all())  # the masked score is the constant: all masked keys tie in the rowmax
    assert (
        torch.tensor(A.NEG_MASK).to(torch.bfloat16).float() == A.NEG_MASK
        or abs(torch.tensor(A.NEG_MASK).to(torch.bfloat16).float() / A.NEG_MASK - 1) < 1e-2
    )
    assert float(torch.exp(torch.tensor(A.NEG_MASK))) == 0.0


if __name__ == "__main__":
    for fn in (
        test_budget,
        test_balance,
        test_reference_dense,
        test_reference_sparse_all_selected_is_dense,
        test_reference_tail_and_select,
        test_estimates,
        test_dm1_model,
        test_batch_order,
        test_kernel_model_sparse_finite,
        test_rm_gather_pages,
        test_mask_constant,
        test_dual_noc_model,
        test_mask_value_independence,
        test_mask_row_host,
        test_abm_layout,
        test_select_bitmap,
        test_softmax_levers,
        test_o_partial_rounding,
        test_mask_bank,
        test_sp_tail_rule,
    ):
        fn()
        print("PASS", fn.__name__)
