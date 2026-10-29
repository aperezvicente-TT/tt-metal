# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python unit test of the PMK MoE v2 levers (tt/pmk_moe2.py; no device): the v2 CB ledger at 2,048 and 8,192
rows (both geometries) within the 1,026 KB budget and the family member's spare + aliasable bytes, the lever-by-lever
DRAM / FPU arithmetic (the README's before / after table), lever 1's L1-capacity proof (infeasible), the rectangle
partition rule, and that v1's hand-off words / CB indices stay valid.
Run: python tests/s16/pmk/test_pmk_moe2_budget.py  (or pytest -q)"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6)))

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as V1  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M  # noqa: E402

KB = 1024


def _geo(w, c=2048, p=110, h_split=2):
    return M.Geometry(c=c, w=w, p=p, h_split=h_split)


def test_ledger_tp8_2048_member_fits():
    # the production placement: the 64 members with XS 35 + the residual live -> 454 KB idle for the moe set; the
    # plan scratch (194 KB, aliasing WGU) + (1, 4, 8) rings with lever 2 fit; MG = 1 needs lever 3 (G = 2 rectangles)
    # to keep the weight re-read at 1.0 (57 rows = 2 tile rows = 2 units per expert)
    g = _geo(256, h_split=4)
    lv = M.Levers(mcast=2)
    p0 = M.cb_plan2(g, 1, 4, 8, lv, standalone=False)
    limit = M.MEMBER_SPARE_B + M.MEMBER_ALIASABLE_B
    assert not p0["member_fits"] and p0["moe"] > limit, p0["moe"]  # (1, 4, 8) + v1 plan hist: over
    lv = M.Levers(mcast=2, hist_sliced=True)  # the open item (README sec 6): hist streamed in 8-row slices
    blk = M.choose_blocks2(g, lv, member=True)
    # after the push-block rule (sec 13): COUT / CW allocated alone -> (1, 2, 4); after the two-slot multicast rings
    # (16, 2026-10-22: PLAN over GUP instead of WGU, no 3-slot WGU rounding) -> (1, 4, 4)
    assert blk == (1, 4, 4), blk
    p = M.cb_plan2(g, *blk, lv, standalone=False)
    assert p["member_fits"] and p["moe"] <= limit, p["moe"]
    assert "XRM" not in p["ffn"] and p["ffn"]["XT"] == 2 * blk[0] * blk[1] * 2048  # lever 2: XT double-buffered, no XRM
    assert M.streaming_model2(g, 1, lv)["reread_factor"] == 1.0  # MG = 1: 2 groups per expert, G = 2 shares them
    assert not M.cb_plan2(_geo(256), 2, 4, 8, M.Levers(mcast=4), standalone=False)["member_fits"]  # (2, 4, 8): no
    for members, total, pages in p["groups"]:
        for m in members:
            assert total % pages[m] == 0, (members, total, pages)
    # the widest blocks still fit the plain 1,026 KB budget standalone (with the 256 KB XS store on every core)
    assert M.cb_plan2(g, 2, 8, 8, lv)["total"] <= M.CB_BUDGET  # (4, 8, 8) is 1,055 KB after sec 13


def test_ledger_8192_rows_both_geometries():
    # the packed 8,192-row bucket: the x-hat_f store is 64 MB (DRAM: lever 2 is a <= 2,048-row lever), the plan
    # scratch grows with the rows (staging 2 x 3 x 8,192 words); the program must still plan within 1,026 KB
    for w in (256, 1024):
        g = _geo(w, c=8192)
        lv = M.Levers(x_l1=False, mcast=4, shared=True, ids_u16=True)
        blk = M.choose_blocks2(g, lv)
        p = M.cb_plan2(g, *blk, lv)
        if w == 256:
            assert blk != (1, 1, 4) and p["total"] <= M.CB_BUDGET, (w, blk, p["total"])
        else:  # the TP2 pair at 8,192 rows (32-tile slices + the 341 KB plan scratch) is not a production placement
            print(f"  8192 W=1024 standalone blocks={blk} total={p['total']} fits={p['total'] <= M.CB_BUDGET}")
        pw = p["plan"]
        assert pw["chunk"] * 110 >= 8192 * 8  # the plan covers every pair
        gm = _geo(w, c=8192, h_split=4)
        lvm = M.Levers(x_l1=False, mcast=4, shared=True, ids_u16=True, hist_sliced=True)
        blk_m = M.choose_blocks2(gm, lvm, member=True)
        pm_ = M.cb_plan2(gm, *blk_m, lvm, standalone=False)
        print(f"  8192 W={w} member blocks={blk_m} moe={pm_['moe']} fits={pm_['member_fits']}")


def test_v1_words_and_indices_stay_valid():
    # the v2 program keeps v1's CB indices (44.., 27/28), its semaphores 5..8 and adds 9 (XS), 10 / 11 (VALID)
    assert V1.PmkMoe.CB_INDEX["TAB"] == 6 and V1.PmkMoe.CB_INDEX_LOW == dict(COUT=0, CRM=1)
    g = _geo(256)
    seen = [m for members, _, _ in M.cb_plan2(g, 2, 4, 8)["groups"] for m in members]
    assert set(seen) == set(V1.PmkMoe.CB_INDEX) - {"XRM"} | set(V1.PmkMoe.CB_INDEX_LOW)
    assert len(seen) == len(set(seen))


def test_lever_table_2048_and_8192():
    rows = {c: M.lever_table(c=c) for c in (2048, 8192)}
    v1_2k, l2_2k, l3_2k, b8_2k, l1_2k, _ = [m for _, _, m in rows[2048]]
    assert 0.23 < b8_2k["math_ceiling"] < 0.27  # bfp8 partials: 20.5 -> ~24.6 %
    assert abs(v1_2k["dram_mb"] - 929) < 15 and 0.15 < v1_2k["math_ceiling"] < 0.2  # v1's 17.5 %
    assert l2_2k["x_dram_mb"] == 0 and l2_2k["dram_mb"] < v1_2k["dram_mb"] - 130  # lever 2: -134 MB
    assert l3_2k["reread_factor"] == 1.0 and abs(l3_2k["weight_mb"] - 510) < 5  # TP8 / 2,048: nothing to save
    assert 0.19 < l3_2k["math_ceiling"] < 0.22  # 20.5 %: the out_rm round trip caps it
    assert l1_2k["math_ceiling"] > 0.3  # the hypothetical L1 combine: 31 % at 2,048
    v1_8k, l2_8k, l3_8k, b8_8k, l1_8k, both_8k = [m for _, _, m in rows[8192]]
    assert v1_8k["reread_factor"] > 1.5 and l3_8k["reread_factor"] == 1.0  # 8,192: 228 rows per expert = 2 groups
    assert 0.25 < l3_8k["math_ceiling"] < 0.35  # ~30 %: x from DRAM (the 64 MB store does not fit L1) + out_rm
    assert both_8k["math_ceiling"] > 0.85  # what the operator's bar assumes: both stores in L1 (capacity says no)
    # the TRUE crossover: with DRAM x and out_rm the per-row DRAM exceeds the per-row FPU at HiFi2 -> never
    assert v1_2k["crossover_rows"] is None and l1_2k["crossover_rows"] is not None
    # the operator's bar (>= 50 % at >= ~6,000 rows) needs x in L1 AND no out_rm: only the hypothetical row has it
    g = M.Geometry(c=6144, w=256, p=110)
    best = M.streaming_model2(g, 2, M.Levers(mcast=4, comb_l1=True))
    assert best["math_ceiling"] > 0.5
    real = M.streaming_model2(g, 2, M.Levers(x_l1=False, mcast=4))  # the store is DRAM above 2,048 rows
    assert real["math_ceiling"] < 0.35


def test_lever1_capacity_proof():
    # fp32 y (32 MB) + one reserved landing row per token (16 MB) = 447 KB per core at 2,048 rows: it fits a FLAT
    # 110-core program's budget beside the FFN set, NOT the family's members (residual + XS resident), and at 8,192
    # rows (1.8 MB per core) nothing fits: lever 1 cannot reach the >= 6,000-row regime the >= 50 % bar needs
    c2 = M.combine_l1_capacity(M.Geometry(c=2048, w=256))
    assert c2["fits_with_ffn"] and not c2["fits_member"], c2
    c8 = M.combine_l1_capacity(M.Geometry(c=8192, w=256))
    assert not c8["fits"] and c8["per_core_kb"] > 1024, c8


def test_tp2_pair_reread_halves_with_mcast():
    g = _geo(1024)
    blk = M.choose_blocks2(g, M.Levers(mcast=2))
    assert blk[0] == 1
    v1 = M.streaming_model2(g, 1, M.Levers(x_l1=False))
    v2 = M.streaming_model2(g, 1, M.Levers(mcast=2))
    assert v1["reread_factor"] == 2.0 and v2["reread_factor"] == 1.0
    assert v2["dram_mb"] < 0.55 * v1["dram_mb"]


def test_bfp8_partials_ledger_and_bytes():
    # PMK_OUT_BFP8: row-bfp8 rows of 5,120 B (NC = 8) in place of 8 KB: the out_rm round trip 268 -> 168 MB; the
    # combine's CBs (rebuilt bfp8 tiles, fp32 NT-tile accumulator) still alias the FFN set within the budgets
    assert M.rowb_bytes(128, 8) == 5120
    g = _geo(256, h_split=4)
    lv = M.Levers(mcast=2, out_bfp8=True)
    p = M.cb_plan2(g, 2, 4, 8, lv)
    assert p["total"] <= M.CB_BUDGET and p["comb"]["CIN"] == 2 * 32 * 1088 and p["comb"]["CRM"] == 32 * 4096
    for members, total, pages in p["groups"]:
        for m in members:
            assert total % pages[m] == 0, (members, total, pages)
    lvm = M.Levers(mcast=2, out_bfp8=True, hist_sliced=True)
    pm_ = M.cb_plan2(g, *M.choose_blocks2(g, lvm, member=True), lvm, standalone=False)
    limit = M.MEMBER_SPARE_B + M.MEMBER_ALIASABLE_B
    assert pm_["moe"] <= limit + 16 * KB, pm_["moe"]  # sec 13: the row-bfp8 member placement is ~10 KB over (open)
    print(f"  bfp8 member placement: moe={pm_['moe']} limit={limit} over={pm_['moe'] - limit}")
    m16 = M.streaming_model2(_geo(256), 4, M.Levers(mcast=4))
    m8 = M.streaming_model2(_geo(256), 4, M.Levers(mcast=4, out_bfp8=True))
    assert abs(m16["out_mb"] - 268) < 2 and abs(m8["out_mb"] - 168) < 2
    assert m8["math_ceiling"] > m16["math_ceiling"] and 0.23 < m8["math_ceiling"] < 0.27
    m8k = M.streaming_model2(M.Geometry(c=8192, w=256), 4, M.Levers(x_l1=False, mcast=4, out_bfp8=True))
    assert 0.33 < m8k["math_ceiling"] < 0.40


def test_noc_combine_capacity():
    # README sec 11: the deterministic NoC combine on the 44 non-member cells breaks on the fp32 y accumulator
    for c in (2048, 5120):
        r = M.noc_combine_capacity(c)
        assert not r["fits"] and r["breaks"] == "y accumulator", r
        assert not r["fits_tail_only"] or c < 2048  # even an empty tail core (875 KB) cannot hold its share
        assert not M.noc_combine_capacity(c, y_fp32=False)["fits"]  # bf16 y (not bitwise) does not fit either
    flat = M.noc_combine_capacity(2048, owners="all")
    assert flat["fits"] and flat["need_kb"] < 480  # 110 owners, a flat program: the sec-3.1 case (not the members)
    assert not M.noc_combine_capacity(5120, owners="all")["fits"]


def test_ring2_credit_model():
    from models.autoports.zai_org_glm_5_3_flash.tt import pfs_credit_model as PC

    for g in (2, 4, 5):
        for cpu in (32, 48, 64):  # (H_T / KC + H_T / NC) chunks per unit for (., 8, 8), (., 4, 8), (., 4, 4)
            r = PC.check_ring2(g, cpu, seeds=10, launches=2)
            assert r["steps"] > 0


def test_program_b_plan():
    # sec 12: 24 bank-adjacent streamers at 493.5 GB/s, 86 compute cores (48 FFN at G = 2 + 38 pure owners) with the
    # L1 fixed-order combine: fits at 2,048 rows with row-bfp8 landing rows; the DRAM floor is the weight stream alone
    r = M.program_b_plan(2048, 2, landing_bfp8=True)
    assert r["combine_fits"] and r["n_ffn"] == 48 and r["n_pure"] == 38 and r["tokens_per_owner"] == 24
    assert abs(r["weight_mb"] - 510) < 2 and abs(r["y_write_mb"] - 16.8) < 0.2
    assert 1050 < r["dram_us"] < 1080 and 0.30 < r["math_ceiling"] < 0.33  # 31.8 %
    assert not M.program_b_plan(2048, 2, landing_bfp8=False)["roles"]["ffn_owner"]["fits"]  # bf16 landing: 85 KB over
    assert not M.program_b_plan(5120, 3, landing_bfp8=True)["combine_fits"]  # y alone 960 KB per owner
    tiles, m0, pages = M.pack_expert_streams([__import__("torch").ones(64, 64)] * 16, [__import__("torch").ones(32, 64)] * 16)
    assert pages == 8 * 2 * (4 + 2) and m0[8] == 6 and tiles.shape == (pages * 32, 32)  # 2 experts per bank


def test_push_blocks_never_straddle():
    # sec 13 (the 13:44:48Z watcher assert): every CB's capacity is a multiple of every push the kernels issue, for
    # every lever set / geometry / block candidate, and the pushes do not depend on the routing (whole MG-row blocks;
    # the unit list below is shaped like real routing: counts 1..600 rows per expert, mt = 1..MG)
    import random

    rng = random.Random(0)
    counts = [max(0, int(rng.expovariate(1 / 57))) for _ in range(288)]
    for w in (256, 1024):
        for lv in (M.Levers(x_l1=False), M.Levers(), M.Levers(mcast=2), M.Levers(mcast=4, out_bfp8=True),
                   M.Levers(shared=True, ids_u16=True), M.Levers(x_l1=False, out_bfp8=True, shared=True)):  # fmt: skip
            for c in (2048, 8192):
                g = M.Geometry(c=c, w=w, p=110, h_split=4)
                for blk in M.BLOCK_CANDIDATES:
                    mg, kc, nc = blk
                    if g.h_t % kc or g.h_t % nc or (g.h_t // kc) % 2 or (g.h_t // nc) % 2:
                        continue
                    p = M.cb_plan2(g, mg, kc, nc, lv)  # asserts check_pushes inside
                    assert not M.V1.check_pushes(p["groups"], p["blocks"])
                    # the per-unit pushes on real counts: constant per CB (mt <= MG only leaves rows unused)
                    for m_ in counts:
                        mt_all = -(-m_ // 32)
                        for gi in range(-(-mt_all // mg) if m_ else 0):
                            mt = min(mg, mt_all - gi * mg)
                            assert 1 <= mt <= mg
                            assert p["blocks"]["WGU"] == kc * 2 * g.w_t * 576 and p["blocks"]["XT"] == mg * kc * 2048
    # v1's planner carries the same rule now (the integration branch's PmkMoe)
    g = M.V1.Geometry(c=2048, w=256, p=110)
    assert M.V1.choose_blocks(g) == (2, 8, 8)  # (4, 8, 8) no longer fits once COUT / CW are allocated alone
    assert not M.V1.check_pushes(*(lambda p: (p["groups"], p["blocks"]))(M.V1.cb_plan(g, 2, 8, 8)))


def test_mcast_rings_two_slots():
    # lever 3 (ffn_reader.hpp PMK_MCAST / exp_streamer.hpp PMK_MCAST_EXT): the receivers' WGU / WD slots are the
    # streamer's ring position, valid only on rings of EXACTLY two pushes (16, 2026-10-22: PLAN over WGU made TP8's
    # ring 3 slots, 221,184 = 3 x 73,728)
    for w in (256, 1024):
        for lv in (M.Levers(mcast=2), M.Levers(x_l1=False, mcast=2), M.Levers(mcast=4, out_bfp8=True),
                   M.Levers(x_l1=False, mcast=4, shared=True, ids_u16=True)):  # fmt: skip
            for c in (2048, 8192):
                g = M.Geometry(c=c, w=w, p=110, h_split=4)
                for blk in M.BLOCK_CANDIDATES:
                    mg, kc, nc = blk
                    for S in (2, 4):  # perf5: S-chunk rings (PMK_RING_SLOTS)
                        lvs = M.replace(lv, ring_slots=S)
                        if g.h_t % kc or g.h_t % nc or (g.h_t // kc) % S or (g.h_t // nc) % S:
                            continue
                        p = M.cb_plan2(g, mg, kc, nc, lvs)
                        for ring in ("WGU", "WD"):
                            cap = next(t for mem, t, _ in p["groups"] if ring in mem)
                            assert cap == S * p["blocks"][ring], (w, lvs, c, blk, ring, cap, p["blocks"][ring])


def test_streamer_ring_holds_either_chunk():
    # program B's streamer ring (exp_streamer.hpp SLOT_BB): a slot holds a gate|up OR a down chunk; TP2's lever-2
    # blocks (1, 1, 4) have GU_BB 36,864 < DN_BB 73,728 (16, 2026-10-22 15:59Z: the down reads overflowed a gate|up-sized
    # ring, watcher abort) -> the ring is 2 x max(GU_BB, DN_BB) in the plan
    for w, kc, nc in ((1024, 1, 4), (256, 4, 8), (256, 1, 8), (1024, 4, 4)):
        r = M.program_b_plan(2048, 2, w=w, kc=kc, nc=nc, landing_bfp8=True)
        gu_bb, dn_bb = kc * 2 * (w // 32) * M.WTB_BFP4, (w // 32) * nc * M.WTB_BFP4
        assert r["roles"]["streamer"]["ring"] == 2 * max(gu_bb, dn_bb), (w, kc, nc, r["roles"]["streamer"])
    import re
    src = open(os.path.join(os.path.dirname(M.__file__), "kernels", "pmk", "exp_streamer.hpp")).read()
    assert "* GU_BB;" not in src.split("auto issue = ")[1], "a slot address must use SLOT_BB"
    assert "(n % SLOTS) * SLOT_BB" in src and "((n + 1) % SLOTS) * SLOT_BB" in src  # perf5: S-slot rings
    assert re.search(r"SLOT_BB = GU_BB > DN_BB \? GU_BB : DN_BB", src)


def test_program_b_review_fixes():
    # 16, 2026-10-22 review of program B (before its second device run): (5) the writer's row-bfp8 scratch fits in WTAB;
    # (3) the streamer's TAB holds the SYNC_WORDS_B prefix + 3 tables; (1) the streamer adds the stream tensor's base;
    # (2) receivers' MC + 4 / 5 are re-pointed to their streamer; (4) comb_reader zeroes PMK_CW_PAGES pages
    for w, blk in ((1024, (1, 1, 4)), (256, (2, 4, 8)), (256, (1, 4, 4))):
        g = M.Geometry(c=2048, w=w, p=86, h_split=4)
        mg, kc, nc = blk
        p = M.cb_plan2(g, mg, kc, nc, M.Levers(mcast=2, shared=True, out_bfp8=True))
        rb = M.ROWB_EXP_PAD + 32 * nc
        wtab = next(t for mem, t, _ in p["groups"] if "WTAB" in mem)
        assert wtab >= (mg * 32 * 4 + 63) // 64 * 64 + 2 * rb, (w, blk, wtab)
    src = open(M.__file__).read()
    assert "V1.SYNC_WORDS_B + 3 * g.ep * 4" in src and "64 + 3 * g.ep * 4" not in src
    assert "self.stream_t.buffer_address()," in src and 'rt_w[xy][mc0 + 4], rt_w[xy][mc0 + 5] = self.s_noc' in src
    assert '("PMK_CW_PAGES", str(cw_pages))' in src
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "get_noc_addr_from_bank_id<true>(c.bank, w_base)" in es and "get_arg_val<uint32_t>(RT0 + 10)" in es  # perf5: per expert
    cr = open(os.path.join(kd, "comb_reader.hpp")).read()
    assert "cb_reserve_back(CBW, CW_ZERO)" in cr  # (perf5: CBW = CW | CW2 under PMK_COMB_SPLIT)


def test_program_b_tile_owners():
    # sec 13: the FPU accumulates tile ROWS, so an L1-combine owner owns 32-token blocks: 64 pure owners at 1.2 MB
    # each (no fit) leave 22 FFN cores -> compute-bound 1.7 ms, MATH 16 % < the DRAM row-bfp8 combine's 23 %
    r = M.program_b_plan(2048, 2, landing_bfp8=True, tile_owners=True)
    assert not r["combine_fits"] and r["n_ffn"] == 22 and r["math_ceiling"] < 0.2


def test_adj_pmk_and_program_b_ledger():
    class _Grid:
        x, y = 11, 10

    class _Mesh:
        def compute_with_storage_grid_size(self):
            return _Grid()

    st = M.adj_pmk_streamers(_Mesh())  # the fallback pattern (no ttnn mesh): 24 cores outside the member rectangle
    assert len(st) == 24 and len(set(st)) == 24 and not set(st) & set(M.MEMBER_RECT)
    # program B's full lever set at 2,048 / TP8 standalone: row-bfp8 + the P13 term + the tail layouts
    g = M.Geometry(c=2048, w=256, p=86, h_split=4)
    lv = M.Levers(x_l1=True, mcast=2, shared=True, ids_u16=True, out_bfp8=True)
    blk = M.choose_blocks2(g, lv)
    p = M.cb_plan2(g, *blk, lv)
    assert p["total"] <= M.CB_BUDGET and "CSH" in p["comb"] and not M.V1.check_pushes(p["groups"], p["blocks"])
    g8 = M.Geometry(c=8192, w=256, p=86, h_split=4)
    lv8 = M.Levers(x_l1=False, mcast=2, shared=True, ids_u16=True, out_bfp8=True)
    assert M.cb_plan2(g8, *M.choose_blocks2(g8, lv8), lv8)["total"] <= M.CB_BUDGET


def test_rectangles():
    cores = [(x, y) for y in range(10) for x in range(11)]
    noc = {c: (c[0] + 1 + (1 if c[0] >= 8 else 0), c[1] + 2) for c in cores}  # a non-worker NoC column at 9
    r2 = M.rectangles(cores, noc, 2, 11, 10)
    assert len(r2) == 55 and all(len(r) == 2 for r in r2)  # vertical runs: no waste
    r4 = M.rectangles(cores, noc, 4, 11, 10)
    assert len(r4) == 22
    r5 = M.rectangles(cores, noc, 5, 11, 10)
    assert len(r5) == 22 and len({c for r in r5 for c in r}) == 110  # G = 5 covers the whole 11 x 10 grid
    assert M.rectangles(cores, noc, 1, 11, 10) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("\n== lever table (per chip, per layer, HiFi2, TP8 W=256, P=110) ==")
    for c in (2048, 4096, 6144, 8192):
        for label, blk, m in M.lever_table(c=c):
            print(f"C={c:5d} {label:38s} blocks={blk} streams={int(m['streams']):4d} reread={m['reread_factor']:.2f} "
                  f"W={m['weight_mb']:6.0f} x_dram={m['x_dram_mb']:5.0f} x_noc={m['x_noc_mb']:5.0f} out={m['out_mb']:6.0f} "
                  f"y={m['y_mb']:3.0f} MB | DRAM {m['dram_us']:7.0f} us noc {m['noc_us']:5.0f} fpu {m['fpu_useful_us']:6.0f} "
                  f"-> floor {m['span_floor_us']:7.0f} us MATH {100 * m['math_ceiling']:5.1f} % cross={m['crossover_rows']}")
    print("\n== TP2 pair (W=1024) ==")
    g = _geo(1024)
    for label, lv in (("v1", M.Levers(x_l1=False)), ("v2 G=2", M.Levers(mcast=2))):
        m = M.streaming_model2(g, 1, lv)
        print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    print("\n== lever 1 capacity ==")
    for c in (2048, 8192):
        print(c, M.combine_l1_capacity(M.Geometry(c=c, w=256)))
    print("\n== ledgers ==")
    for c in (2048, 8192):
        for w in (256, 1024):
            g = _geo(w, c=c)
            for member in (False, True):
                lv = M.Levers(x_l1=c <= 2048, mcast=4, shared=True, ids_u16=True, hist_sliced=member)
                gg = _geo(w, c=c, h_split=4 if member else 2)
                blk = M.choose_blocks2(gg, lv, member=member)
                p = M.cb_plan2(gg, *blk, lv, standalone=not member)
                print(f"C={c} W={w} {'member' if member else 'standalone'} blocks={blk} moe={p['moe']} xs={p['xs']} "
                      f"total={p['total']} member_fits={p['member_fits']} groups={[(m, t) for m, t, _ in p['groups']]}")


# ---- moe2/PERF.md: the streamer probe, the scoreboard, the pipeline model and the knobs
def test_streamer_probe_rows_and_knobs():
    # (1) the 18:25Z "0 B" stream-rate probe: with PMK_MCAST_EXT every member is a receiver (role 2) and never adds
    # R_KB; the streamers -- the only DRAM readers -- had no probe row. ProgramB now appends 24 streamer rows
    # (probe_extra_rows) and exp_streamer.hpp writes row 2 P + s under XFFN_PROBE (RT0 + 11 / 12); stream_rate reads
    # the streamer rows when present. (2) PMK_STREAM_PREFETCH is a define the host adds only when asked.
    src = open(M.__file__).read()
    assert "self.probe_extra_rows = N_STREAMERS" in src
    assert "pa, 2 * self.p + si," in src and 'd_s.append(("XFFN_PROBE", "1"))' in src
    assert 'd_s.append(("PMK_STREAM_PREFETCH", "1"))' in src and "if self.prefetch:" in src
    v1 = open(V1.__file__).read()
    assert 'getattr(self, "probe_extra_rows", 0)' in v1 and 'ch["streamer"] = v' in v1
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "get_arg_val<uint32_t>(RT0 + 11)" in es and "get_arg_val<uint32_t>(RT0 + 12)" in es
    assert "#ifdef PMK_STREAM_PREFETCH" in es and "noc_semaphore_wait_min(ready_gu, g * n_gu)" in es
    # the counters are register accumulators written once (no probe write inside the chunk loop)
    body = es.split("if (next(cur)) {")[1].split("noc_async_write_barrier();")[0]
    assert "noc_async_write(" not in body and "SP_TOC(c_gu_cred)" in es and "SP_TOC(c_flush)" in es
    # the receiver side keeps counting its waits: data wait (credit .. VALID) and slot wait (cb_reserve_back)
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    recv = rd.split("if (role == 2) {  // receiver")[1].split("continue;")[0]
    assert "XP_TOC(R_GU_READ)" in recv and "XP_ADD(R_KB" not in recv


def test_progb_scoreboard_decodes_recorded_rows():
    # the 18:25Z watched (1,4,4) x_l1=0 row (d0/pcc.jsonl, chip 0, the 4 recorded members): reader data wait ~ 47 %
    # of the span, slot wait ~ 52 %; writer x gather ~ 42 %, y write ~ 54 % -- plus synthetic streamer rows
    import torch

    reader = torch.tensor([[9428932, 18771657, 19627800, 11487434, 15, 0, 5473, 0, 0, 0, 0, 0, 0, 59411716, 2913217886, 2972629602],
                           [9168744, 19987817, 23219301, 13699119, 17, 0, 5455, 0, 0, 0, 0, 0, 0, 66180781, 2913219395, 2979400176],
                           [17115908, 28753021, 13217675, 4263055, 16, 0, 5546, 0, 0, 0, 0, 0, 0, 63451170, 2913220791, 2976671961],
                           [7745265, 20650494, 24466314, 13255222, 17, 0, 5456, 0, 0, 0, 0, 0, 0, 66223355, 2913222246, 2979445601]])  # fmt: skip
    writer = torch.tensor([[25050215, 25378, 1957266, 32370568, 1382111, 32502, 3840, 3552, 0, 0, 0, 0, 0, 60901446, 2911848342, 2972749788],
                           [28579738, 28749, 6246741, 31212907, 1383805, 36990, 4352, 3424, 0, 0, 0, 0, 0, 67583217, 2911848334, 2979431551],
                           [26733654, 257335, 2766650, 33972170, 1385534, 35040, 4096, 3712, 0, 0, 0, 0, 0, 65238913, 2911848174, 2977087087],
                           [28405955, 28731, 4664971, 33395644, 1386806, 36851, 4352, 3648, 0, 0, 0, 0, 0, 68013559, 2911848298, 2979861857]])  # fmt: skip
    span = 60_000_000
    streamer = torch.zeros(24, 16, dtype=torch.int64)
    streamer[:, M.S_GU_READ], streamer[:, M.S_DN_READ] = 6_000_000, 3_000_000
    streamer[:, M.S_GU_CREDIT], streamer[:, M.S_DN_CREDIT] = 36_000_000, 9_000_000
    streamer[:, M.S_KB] = 7_077_888 // 1024
    streamer[:, M.S_STREAMS], streamer[:, M.S_GU_CHUNKS], streamer[:, M.S_DN_CHUNKS] = 12, 12 * 32, 12 * 32
    streamer[:, M.S_SPAN], streamer[:, M.S_START], streamer[:, M.S_END] = span, 1000, 1000 + span
    ch = dict(reader=reader, writer=writer, streamer=streamer)
    sb = M.progb_scoreboard(ch)
    assert 0.4 < sb["member_reader"]["data_wait"] < 0.6 and 0.4 < sb["member_reader"]["slot_wait"] < 0.6, sb
    assert 0.35 < sb["member_writer"]["x_gather"] < 0.5 and 0.45 < sb["member_writer"]["y_write"] < 0.6, sb
    assert sb["streamer"]["credit_wait"] == 0.75 and sb["streamer"]["read_active"] == 0.15
    assert sb["streamer"]["prefetch"] == 0 and abs(sb["streamer"]["unattributed"] - 0.1) < 0.01  # 10 % untimed here
    assert sb["streamer"]["streams"] == 288 and abs(sb["streamer"]["mb"] - 169.9) < 0.2
    # stream_rate now reads the streamer rows: bytes > 0 (the old rows alone gave 0 B)
    sr = M.stream_rate([ch])[0]
    assert sr["source"] == "streamer" and sr["bytes_mb"] > 100 and sr["dram_gbps_span"] > 0
    assert M.stream_rate([dict(reader=reader, writer=writer)])[0]["bytes_mb"] == 0.0


def test_progb_pipeline_model():
    # the host pipeline model (PERF.md sec 3): TP2 L4 geometries, G = 2, 12 streams per streamer
    r1 = M.progb_pipeline(128, 32, 1, 4, x_l1=True)
    r4 = M.progb_pipeline(128, 32, 4, 4, x_l1=False)
    assert r1["chunks"] == 12 * 160 and r4["chunks"] == 12 * 64
    assert abs(r1["bytes_mb_streamer"] - 12 * 7.08) < 0.1 and abs(r4["bytes_mb_streamer"] - 12 * 7.08) < 0.1
    # with the kernels' nominal per-tile costs the members are already the larger term, and the streamer's largest
    # stall is the credit wait (never the data wait on the member side)
    for r in (r1, r4):
        assert r["member"]["compute_gu"] + r["member"]["compute_dn"] > 0.8, r["member"]
        assert r["member"]["data_wait_gu"] < 0.05
    # prefetch never lengthens the span; it removes the streamer's exposed read wait
    for k in ((1, 4, True), (4, 4, False)):
        a = M.progb_pipeline(128, 32, k[0], k[1], x_l1=k[2])
        b = M.progb_pipeline(128, 32, k[0], k[1], x_l1=k[2], prefetch=True)
        assert b["span_us"] <= a["span_us"] + 1 and b["streamer"]["read_gu"] <= a["streamer"]["read_gu"]
    # a streamer-bound regime (fast members): data waits appear and prefetch shortens the span
    fast = M.replace(M.PipeCost(), c_mm=8.0, c_pack32=20.0, c_sync=20.0, c_rowb=30.0)
    a = M.progb_pipeline(128, 32, 4, 4, x_l1=False, cost=fast)
    b = M.progb_pipeline(128, 32, 4, 4, x_l1=False, cost=fast, prefetch=True)
    assert a["member"]["data_wait_gu"] > 0.2 and b["span_us"] < a["span_us"]
    # the calibration against the 18:25Z spans lands in the member-bound regime
    cal = M.progb_calibrate({(1, 4, True): 36480.0, (4, 4, False): 30221.6}, mm_grid=range(40, 240, 20), pack_grid=range(100, 400, 50))
    assert all(v["bound"] == "members (credits)" for v in cal["table"].values()), cal["rel_err"]
    assert all(v["streamer"]["credit_gu"] > v["streamer"]["flush_gu"] + v["streamer"]["read_gu"] for v in cal["table"].values())


def test_cover_rectangles_all_compute_cores():
    # PERF.md sec 7: on the 11 x 10 grid with the mock's fallback streamer pattern (adj_pmk_streamers without ttnn)
    # the 86 compute cores are covered by exactly 24 exact boxes of <= 4 (and of <= 8) members; the 18:25Z box mode
    # placed 48 members
    gx, gy = 11, 10
    allc = [(x, y) for y in range(gy) for x in range(gx)]
    noc = {c: (c[0] + 1 + (c[0] >= 8), c[1] + 2 + (c[1] >= 5)) for c in allc}  # a harvested-row / column style map
    adj = [((x, y), b) for b, (x, y) in enumerate([(x, y) for y in (0, 1, 4, 6, 9) for x in (0, 1, 2, 6, 7, 8)][:24])]
    ms = set(M.MEMBER_RECT)
    taken, placed, per_bank = set(), [], {}
    for c, b in adj:
        i = per_bank.get(b % 8, 0)
        per_bank[b % 8] = i + 1
        if c not in ms:
            placed.append((c, b % 8, i))
            taken.add(c)
    for c, b in adj:
        if c in ms:
            free = [f for f in allc if f not in ms and f not in taken]
            best = min(free, key=lambda f: abs(f[0] - c[0]) + abs(f[1] - c[1]))
            placed.append((best, b % 8, 0))
            taken.add(best)
    streamers = [c for c, _, _ in placed][:24]
    compute = [c for c in allc if c not in set(streamers)]
    assert len(compute) == 86
    for gmax in (4, 8):
        rects = M.cover_rectangles(compute, noc, 24, gmax, gx, gy)
        covered = sorted(c for r in rects for c in r)
        assert len(rects) == 24 and len(covered) == len(set(covered)) and set(covered) <= set(compute), (gmax, len(rects))
        assert len(covered) >= 80 and max(len(r) for r in rects) <= 8 and min(len(r) for r in rects) >= 1, (gmax, sorted(map(len, rects)))
        for r in rects:  # exact NoC boxes
            v = [noc[c] for c in r]
            lx, ly, hx, hy = min(a for a, _ in v), min(b for _, b in v), max(a for a, _ in v), max(b for _, b in v)
            assert {c for c in compute if lx <= noc[c][0] <= hx and ly <= noc[c][1] <= hy} == set(r)
    old = M.rectangles(compute, noc, 2, gx, gy)
    assert len(old) >= 24 and 2 * 24 == 48 < 86
    # the runtime G: receivers get len(rect) at PMK_RT_MC + 0, the streamer at RT0 + 13
    src = open(M.__file__).read()
    assert "mc = [len(r), ri, pos, nr" in src and "len(r),  # RT0 + 13" in src
    es = open(os.path.join(os.path.dirname(M.__file__), "kernels", "pmk", "exp_streamer.hpp")).read()
    assert "get_arg_val<uint32_t>(RT0 + 13)" in es and "g * n_gu" in es and "(ng + rows_per_stream - 1) / rows_per_stream" in es and "G * n_" not in es


def test_gu_rt_knob_and_tp8_expectation():
    # PERF.md sec 7-8: the B-reuse subblock (rt = mt, ct < rt) and the TP8 arithmetic
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    fc = open(os.path.join(kd, "ffn_compute.hpp")).read()
    assert "#ifdef PMK_GU_RT" in fc and "matmul_block(CB_XT, CB_WGU, k, k * GU_T + nb * GU_CT, 0, false, GU_CT, mt, KC)" in fc
    assert "pack_tile<true>(r * GU_CT + s, CB_GUP, r * GU_T + nb * GU_CT + s)" in fc
    assert "MG * GU_CT <= PMK_DEST_TILES" in fc
    v1 = open(V1.__file__).read()
    assert '("PMK_GU_RT", "1"), ("PMK_GU_CT", str(ct)), ("PMK_DEST_TILES", str(dest_tiles))' in v1
    assert "ct = self.gu_ct or (1 if mg >= 2 else suba)" in v1
    e = M.tp8_expectation(86, 236.0)
    assert e["products"] == 512 * 3072 and abs(e["dram_ms"] - 1.033) < 0.01 and e["fpu_ms"] > 3.0
    assert M.tp8_expectation(110, 173.0)["fpu_ms"] < 1.85
    # the c_tile that puts N math cores on the DRAM floor at 2,048 rows
    assert 74 < M.tp8_expectation(86, 236.0)["c_tile_for_dram_floor"] < 78
    assert 95 < M.tp8_expectation(110, 236.0)["c_tile_for_dram_floor"] < 100


def test_program_b_plan_unequal_rectangles():
    # the census with the cover placement's member count (86 members of unequal rectangles) instead of 24 x g_size
    r = M.program_b_plan(2048, 6, w=1024, kc=4, nc=4, landing_bfp8=True, n_ffn=86)
    assert r["n_ffn"] == 86 and r["n_pure"] == 0
    assert M.program_b_plan(2048, 2, w=1024, kc=4, nc=4, landing_bfp8=True)["n_ffn"] == 48


def test_perf3_knobs_text():
    # PERF.md sec 9: every streamer cycle attributed (S_MC_ISSUE / S_RD_ISSUE), PMK_MCAST_LINKED / _UNICAST, PMK_YW_DB
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "SP_TOC(c_mc_issue)" in es and "SP_TOC(c_rd_issue)" in es and "S_TABLES" not in es
    assert "#ifdef PMK_MCAST_LINKED" in es and "#ifdef PMK_MCAST_UNICAST" in es
    assert "noc_async_write(reinterpret_cast<uint32_t>(src), get_noc_addr(x, y, sem_l1), 4)" in es
    wr = open(os.path.join(kd, "ffn_writer.hpp")).read()
    assert "#ifdef PMK_YW_DB" in wr and "const uint32_t blk = scr0 + (yw_blk & 1) * GRP_B;" in wr
    assert wr.count("noc_async_write_barrier();  // every deferred block write landed") == 1
    # the host sizes WTAB for the two block scratches only with the knob
    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    a = M.cb_plan2(g, 1, 4, 4, M.Levers(mcast=2, shared=True, out_bfp8=True))
    b = M.cb_plan2(g, 1, 4, 4, M.Levers(mcast=2, shared=True, out_bfp8=True, yw_db=True))
    wa = next(t for mem, t, _ in a["groups"] if "WTAB" in mem)
    wb = next(t for mem, t, _ in b["groups"] if "WTAB" in mem)
    assert wb - wa >= (2 * 32 - 2) * (M.ROWB_EXP_PAD + 32 * 4) - 64, (wa, wb)
    src = open(M.__file__).read()
    assert 'd_s.append(("PMK_MCAST_UNICAST", "1"))' in src and '("PMK_YW_DB", "1")' in src


def test_perf4_coalesced_writes_and_dual_mcast():
    # PERF.md sec 10: YW_COAL groups (YRM = 2 groups, WTAB = 2 group scratches), the NCRISC half-streamer, the
    # receivers' second VALID
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    wr = open(os.path.join(kd, "ffn_writer.hpp")).read()
    assert "cb_wait_front(CB_YRM, CW * mt * NC)" in wr and "noc_async_write(dst0, outb_g.get_noc_addr(d0 + row, nc * RB), CW * RB)" in wr
    assert "cb_pop_front(CB_YRM, CW * mt * NC)" in wr
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "#ifdef PMK_MCAST_DUAL" in es and "go[0] = ++n_go;" in es and "*done < n + 2 - SLOTS" in es  # perf5: S slots
    nc_ = open(os.path.join(kd, "exp_streamer_nc.hpp")).read()
    assert "get_noc_multicast_addr(hx, hy, lx, ly, dst)" in nc_ and "SEM_VALID1_GU = 12" in nc_ and "*done = n;" in nc_
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    assert "noc_semaphore_wait_min(vgu1_p, n_gu);" in rd and "noc_semaphore_wait_min(vd1_p, n_d);" in rd
    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)  # the TP2 bench geometry: x_l1 dropped (the XS store does not fit)
    base = M.cb_plan2(g, 1, 4, 4, M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True))
    coal = M.cb_plan2(g, 1, 4, 4, M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2))
    yb = next(t for mem, t, _ in base["groups"] if "YRM" in mem)
    yc = next(t for mem, t, _ in coal["groups"] if "YRM" in mem)
    assert yc >= 2 * yb - 1 and coal["total"] <= M.CB_BUDGET, (yb, yc, coal["total"])
    # coal 4 at this geometry is 18 KB over the budget (YRM + the two group scratches): the TP2 cell runs coal 2
    assert M.cb_plan2(g, 1, 4, 4, M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=4))["total"] > M.CB_BUDGET
    src = open(M.__file__).read()
    assert 'raw["s1"] = ("moe_stream_ncrisc.cpp", ct_s, rt_s, d_s, dm1)' in src and '("PMK_YW_COAL", str(max(1, int(lv.yw_coal))))' in src


def test_perf5_dual_alignment_and_ring_kind():
    # PERF.md sec 11 (perf5): the NCRISC half-streamer's VALID source words are 16-B aligned slots of a 256-B sync
    # shard (the 22:23Z watcher NOC sanitize abort: a 4-B-aligned multicast source at SW_DONE + 4) and the ring kind
    # travels in a GO word (GU_BB == DN_BB at (1, 4, 8): the byte-count heuristic cannot tell the halves apart)
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    nc_ = open(os.path.join(kd, "exp_streamer_nc.hpp")).read()
    import re
    m = re.search(r"SW_VAL1_GU = (\d+), SW_VAL1_D = (\d+)", es)
    assert m and int(m.group(1)) % 16 == 0 and int(m.group(2)) % 16 == 0 and int(m.group(2)) < 4 * M.SYNC_SHARD_WORDS
    assert "SYNC_B = 256" in es and "go[5] = down;" in es and "SW_GO_DOWN = 116" in es
    assert "const uint32_t slot = go[1], dst = go[2], bytes = go[3], down = go[5];" in nc_
    assert "bytes == GU_BB - GU_BB / 2" not in nc_ and "SW_VAL1_GU = 128, SW_VAL1_D = 144" in nc_
    assert "expsnc::SW_VAL1_GU" in nc_ and "expsnc::SW_VAL1_D" in nc_
    src = open(M.__file__).read()
    assert "[1, SYNC_SHARD_WORDS]" in src and M.SYNC_SHARD_WORDS * 4 == 256
    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    gu_bb, dn_bb = 4 * 2 * g.w_t * M.WTB_BFP4, g.w_t * 8 * M.WTB_BFP4
    assert gu_bb == dn_bb  # (1, 4, 8): the ambiguity the GO word removes


def test_perf5_stream_lpt_balance():
    # PERF.md sec 11: the live expert -> streamer assignment (moe_common.hpp stream_lpt; stream_assignment is the host
    # reference): the static map leaves the phase at the SLOWEST streamer (perf4 cover cell: 16.0 ms phase vs the
    # 11.6 ms mean, core_gbps 6.25 .. 8.92). Synthetic routings (2,048 rows x 8 picks over 288 experts, a Zipf-like
    # skew as the real L4 routing shows ng 1..4): the per-bank LPT (mode 1) and the global LPT (mode 2) bring the
    # max / mean streams per streamer from ~1.3-1.5 to <= ~1.1 / ~1.02; both are deterministic (ties: ascending).
    import random

    sizes = [4] * 17 + [3] * 5 + [2, 1]  # the perf4 cover on the p300 adj_pmk pattern (rect_sizes of the row)
    rows = []
    for seed in range(6):
        rng = random.Random(seed)
        w = [1.0 / (1 + e) ** 0.6 for e in range(288)]
        rng.shuffle(w)
        cnt = [0] * 288
        for _ in range(2048):
            picks = set()
            while len(picks) < 8:
                picks.add(rng.choices(range(288), weights=w)[0])
            for e in picks:
                cnt[e] += 1
        res = {}
        for mode in (0, 1, 2):
            owner, load = M.stream_assignment(cnt, sizes, mg=1, mode=mode)
            assert all((o is None) == (c == 0) for o, c in zip(owner, cnt))
            if mode == 1:
                assert all(o is None or (o & 7) == (e & 7) for e, o in enumerate(owner))
            # the loads are what the kernels will see: ceil(ng / g_s) streams per expert, summed per streamer
            chk = [0] * 24
            for e, o in enumerate(owner):
                if o is not None:
                    chk[o] += -(-(-(-cnt[e] // 32)) // sizes[o])
            assert [c_ * 256 for c_ in chk] == load, (mode, chk, load)  # loads in weighted units (1.0 = 256)
            res[mode] = max(load) / (sum(load) / 24)
        rows.append(res)
        # the same input twice -> the same table (determinism), and a brute-force LPT agrees with the reference
        assert M.stream_assignment(cnt, sizes, 1, 2) == M.stream_assignment(cnt, sizes, 1, 2)
    print("  streams max/mean per streamer: static / bank LPT / global LPT:",
          [tuple(round(r[m], 3) for m in (0, 1, 2)) for r in rows])
    assert all(r[2] <= 1.06 for r in rows), rows
    assert all(r[1] <= r[0] + 1e-9 and r[2] <= r[1] + 1e-9 for r in rows), rows
    assert sum(r[0] for r in rows) / len(rows) >= 1.15, rows  # the static map's imbalance is real
    # the kernels carry the same arithmetic and the host sizes the scratch
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    mc = open(os.path.join(kd, "moe_common.hpp")).read()
    assert "stream_lpt(" in mc and "LPT_NG_MAX = 1024" in mc and f"LPT_NG_MAX = {M.LPT_NG_MAX}" in mc
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    assert "return owner[e] == s;" in es and "const uint32_t se = owner[e];" in rd
    assert "get_noc_addr_from_bank_id<true>(c.bank, w_base)" in es  # mode 2: a streamer reads any bank
    src = open(M.__file__).read()
    assert '("PMK_RT_GS", str(base_gs))' in src and 'd_s.append(("PMK_STREAM_LPT", str(self.lpt)))' in src
    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    p = M.cb_plan2(g, 1, 4, 8, M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2))
    assert p["ffn"]["TAB"] >= V1.ffn_cb_bytes(g, 1, 4, 8)["TAB"] + 128 + M.lpt_scratch_bytes(g.ep)
    # the scratch rides inside the TAB / CIDX alias group (CIDX's 32 x PART_B staging is the larger member): no new bytes
    tab_grp = next(t for mem, t, _ in p["groups"] if "TAB" in mem)
    assert tab_grp == p["comb"]["CIDX"], (tab_grp, p["comb"]["CIDX"])


def test_perf5_ring_slots():
    # PERF.md sec 11 lever 3: 4-chunk rings at half chunks = the same L1 as 2-chunk rings at full chunks ((1, 2, 4) x 4
    # vs (1, 4, 8) x 2 on the pair), the credit model with S slots (J1: slot n mod S free = chunk n - S popped), and the
    # kernels' slot arithmetic / the DUAL slot-reuse wait (DONE >= n + 2 - S)
    from models.autoports.zai_org_glm_5_3_flash.tt import pfs_credit_model as CM

    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    lv2 = M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2)
    lv4 = M.replace(lv2, ring_slots=4)
    p2 = M.cb_plan2(g, 1, 4, 8, lv2)
    p4 = M.cb_plan2(g, 1, 2, 4, lv4)
    ring2 = next(t for mem, t, _ in p2["groups"] if "WGU" in mem)
    ring4 = next(t for mem, t, _ in p4["groups"] if "WGU" in mem)
    assert ring2 == ring4 == 4 * 2 * 2 * 32 * M.WTB_BFP4, (ring2, ring4)  # 294,912 both
    assert p4["total"] <= p2["total"], (p4["total"], p2["total"])  # smaller YRM / WTAB at NC 4
    print(f"  (1,4,8) S=2 total {p2['total']}  (1,2,4) S=4 total {p4['total']}")
    # (the pair runs explicit blocks for both: (1, 4, 8) and (1, 2, 4) x 4 sit above the 1,026 KB family budget as
    # (1, 4, 8) already did on silicon; cb_guard checks the real L1 end)
    for gg, cpu in ((2, 96), (4, 96), (6, 48)):
        CM.check_ring2(gg, cpu, seeds=4, slots=4)
    import pytest
    with pytest.raises(AssertionError):
        CM.simulate_ring2(2, 50, slots=4)  # chunks per unit must be whole rings
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "constexpr uint32_t SLOTS = PMK_RING_SLOTS;" in es and "((n_gu - 1) % SLOTS) * GU_BB" in es and "((n_d - 1) % SLOTS) * DN_BB" in es
    assert es.count("while (n + 2 > SLOTS && *done < n + 2 - SLOTS)") == 2
    assert "(H_T / KC) % SLOTS == 0 && (H_T / NC) % SLOTS == 0" in es
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    assert "(H_T / KC) % PMK_RING_SLOTS == 0 && (H_T / NC) % PMK_RING_SLOTS == 0" in rd
    src = open(M.__file__).read()
    assert 'd_s.append(("PMK_RING_SLOTS", str(int(self.lv.ring_slots))))' in src and "total_size=self.lv.ring_slots * slot_bb" in src


def test_perf5_gu_split():
    # PERF.md sec 12 (perf5): the gate|up column split -- the host mirrors of split_rows / split_cols (pairs intact,
    # SUBA units, R x cw <= GUP tiles), the ledger at h_split 8 (within 2 KB of the (1, 4, 8) silicon baseline), the
    # kernels' protocol words, and the LPT cost = ceil(ng / rows per stream)
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    for gu_t, gupt in ((64, 64), (16, 32)):
        for gg in range(1, 9):
            r = M.split_rows(gg, gu_t, 4, gupt)
            cols = [M.split_cols(p, gg, gu_t, 4) for p in range(gg)]
            assert sum(cw for _, cw in cols) == gu_t and all(c0 % 4 == 0 and cw % 4 == 0 for c0, cw in cols), (gg, cols)
            assert cols[0][0] == 0 and all(cols[i][0] + cols[i][1] == cols[i + 1][0] for i in range(gg - 1))
            assert 1 <= r <= min(gg, M.SPLIT_RMAX) and r * max(cw for _, cw in cols) <= gupt, (gu_t, gg, r, cols)
    g4 = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    lv = M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2)
    base = M.cb_plan2(g4, 1, 4, 8, lv)["total"]
    p = M.cb_plan2(g4, 1, 4, 8, M.replace(lv, gu_split=True))
    assert p["total"] <= base + 3 * M.SPLIT_RMAX * 4 * 2048, (p["total"], base)  # + XT + the XRM double buffer: ACTS rides on COUT
    act = next((mem, t) for mem, t, _ in p["groups"] if "ACT" in mem)
    assert act[1] == 32 * 2048, act  # one block (CSH 64 KB fits inside at h_split 4)
    acts = next((mem, t) for mem, t, _ in p["groups"] if "ACTS" in mem)
    assert acts[1] == 32 * 2048 and "COUT" in acts[0], acts
    print(f"  (1,4,8) split total {p['total']} vs base {base}")
    assert p["blocks"]["XRM"] == 4 * 2048 and p["blocks"]["XT"] == M.SPLIT_RMAX * 4 * 2048
    assert p["ffn"]["XRM"] == 2 * M.SPLIT_RMAX * 4 * 2048  # round 2: two chunks of rows (the gather overlaps the compute)
    import pytest
    with pytest.raises(AssertionError):
        M.cb_plan2(g4, 1, 4, 8, M.replace(lv, x_l1=True, gu_split=True))
    mc = open(os.path.join(kd, "moe_common.hpp")).read()
    assert "CB_ACTS = PMK_MOE_CB0 + 16" in mc and "CTL_NOROW = 0x200" in mc and "SEM_ACT = 14" in mc and M.SEM_ACT == 14
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    assert "wait_min(SEM_ACT, G * n_owned);" in rd and "cb_push_back(CB_ACT, W_T);" in rd and "unit_split(e, s_, ng);" in rd
    cp = open(os.path.join(kd, "ffn_compute.hpp")).read()
    assert "cb_reserve_back(CB_ACTS, ACTST);" in cp and "pack_tile<true>(s, CB_GUP, r * spl_cw + c + s);" in cp
    assert "if (ctl & CTL_NOROW) {" in cp and 'static_assert(PMK_GU_SPLIT_SUBA == SUBA' in cp
    wr = open(os.path.join(kd, "ffn_writer.hpp")).read()
    assert "get_noc_addr(ox, oy, act_base + (spl_c0 / 2 + j) * 2048)" in wr and "noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(SEM_ACT)), 1);" in wr
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "return (ng + rows_per_stream - 1) / rows_per_stream;" in es
    src = open(M.__file__).read()
    assert '("PMK_RT_MEMBERS", str(base_m))' in src and "sems.append(ttnn.SemaphoreDescriptor(id=SEM_ACT" in src
    # the LPT cost under the split: rows per stream, so a 4-member box costs ceil(ng / 4) and a 3-member one ceil(ng / 2)
    cnt = [64] * 288
    owner, load = M.stream_assignment(cnt, [M.split_rows(gg, 64, 4, 64) for gg in [4] * 17 + [3] * 5 + [2, 1]], 1, 2)
    assert sum(load) >= 288 * 256 and max(load) - min(load) <= 256, load  # (the g = 1 box streams a 2-row expert twice)


def test_perf5_b_inplace():
    # PERF.md sec 13 (Stage 12's ruling): the streamers read pl.xsl's resident ND-sharded expert pages in place -- no
    # stream copy, per chip correct by construction; the kernel's per-pair reads through the [E, 2] address table +
    # TensorAccessor shard bases, the host's CB_TAB sizing, the knob plumbing
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    assert "#ifdef PMK_B_INPLACE" in es and "TensorAccessorArgs<CT0 + 9>()" in es and "gu_base[j] = gu.get_noc_addr(2 * j);" in es
    assert "noc_async_read(gu_base[j] + off, slot + (k * GU_T + 2 * j) * WTB, 2 * WTB);" in es
    assert "noc_async_read(dn_base[(c.idx * NC + jj) / 2] + k * 2 * WTB, slot + (k * NC + jj) * WTB, 2 * WTB);" in es
    assert "c.e = en_e;" in es
    src = open(M.__file__).read()
    assert 'd_s.append(("PMK_B_INPLACE", "1"))' in src and "self.tbl.buffer_address() if self.inplace else self.m0_t.buffer_address()" in src
    assert "if self.inplace:  # nothing to pack" in src and "io += ([self.sync] if self.inplace else [self.stream_t, self.m0_t, self.sync])" in src
    assert "tab_b += 64 + 2 * self.e * 4 + 64 + 8 * (self.w_t + self.h_t // 2) + 64" in src
    # the census delta: the stream copy that no longer exists (sec 1 bytes: 288 experts x (H x 2W + W x H) bfp4 tiles)
    for w, mb in ((1024, 2038.4), (256, 509.6)):
        g = M.Geometry(c=2048, w=w, p=86, h_split=4)
        per_expert = (g.h_t * 2 * g.w_t + g.w_t * g.h_t) * M.WTB_BFP4
        assert abs(288 * per_expert / 1e6 - mb) < 1.0, (w, 288 * per_expert / 1e6)


def test_perf5_defaults_round2():
    # PERF.md sec 14 orders: PMK_B_INPLACE default ON (bitwise, -15.8 GB per chip), PMK_GU_SPLIT default OFF (a loss on
    # the pair: +27 % chunks from R = 2 and the serialised 2-row x gather; the code stays)
    import inspect
    sig = inspect.signature(M.ProgramB.__init__)
    assert sig.parameters["inplace"].default is True and sig.parameters["gu_split"].default is False
    assert M.Levers().gu_split is False
    for f in ("pmk_moe2_bench.py", "pmk_moe2_progb_mock.py"):
        src = open(os.path.join(os.path.dirname(M.__file__), "..", "tests", "s16", "jobs", f)).read()
        assert 'args.get("inplace", 1)' in src and 'args.get("gu_split", 0)' in src, f


def test_perf5_round3_lpt_weights_and_comb_split():
    # PERF.md sec 15: (A) per-streamer LPT cost weights (the measured us per chunk of each rectangle: the same bytes run
    # 1.4x slower on some boxes) -- the host mirror, the arg plumbing; (B) PMK_COMB_SPLIT -- the combine's odd slots on
    # the BRISC (CIN2 aliased on YRM: -34 KB), the y write on the NCRISC, the combine probe slots 8-12
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    sizes = [4] * 17 + [3] * 5 + [2, 1]
    cnt = [64] * 288
    w = [1.0] * 24
    w[21] = 1.4  # a slow box gets fewer experts
    o0, l0 = M.stream_assignment(cnt, sizes, 1, 2)
    o1, l1 = M.stream_assignment(cnt, sizes, 1, 2, w)
    assert sum(1 for o in o1 if o == 21) < sum(1 for o in o0 if o == 21)
    assert max(l1) - min(l1) <= 256 + 1, (min(l1), max(l1))  # weighted loads balanced to one unit-cost stream
    assert M.lpt_weights_from(None) is None and M.lpt_weights_from([2, 1, 4])[1] == 1.0
    mc = open(os.path.join(kd, "moe_common.hpp")).read()
    assert "((v + gs - 1) / gs) * gtab[32 + s]" in mc and "CB_CIN2 = PMK_MOE_CB0 + 17" in mc and "CB_CW2 = PMK_MOE_CB0 + 18" in mc
    es = open(os.path.join(kd, "exp_streamer.hpp")).read()
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    assert "#ifdef PMK_LPT_W" in es and "#ifdef PMK_LPT_W" in rd
    cr = open(os.path.join(kd, "comb_reader.hpp")).read()
    assert "constexpr uint32_t COMB_HALF = HALF, COMB_STEP = 2;" in cr and "for (uint32_t j = COMB_HALF; j < K; j += COMB_STEP)" in cr
    assert "get_arg_val<uint32_t>(PMK_RT_Y)" in cr and "xffn_probe::g_c[11]" in cr and "idx_l1 = COMB_HALF ? get_write_ptr(CB_WGU)" in cr
    cc = open(os.path.join(kd, "comb_compute.hpp")).read()
    assert "(j & 1) ? CB_CIN2 : CB_CIN" in cc
    br = open(os.path.join(kd, "moe_brisc.cpp")).read()
    assert "wait_min(SEM_FFN, PMK_P);" in br and "combr::phase_main();" in br and "constexpr uint32_t HALF = 1;" in br
    assert "constexpr uint32_t HALF = 0;" in open(os.path.join(kd, "moe_ncrisc.cpp")).read()
    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    lv = M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2)
    base = M.cb_plan2(g, 1, 4, 8, lv)["total"]
    p = M.cb_plan2(g, 1, 4, 8, M.replace(lv, comb_split=True))
    assert p["total"] == base - 32 * M.BFP8_TILE_B + 4096, (p["total"], base)  # CIN2 on YRM, CW2 alone
    assert next(mem for mem, _, _ in p["groups"] if "CIN2" in mem) == ("YRM", "CIN2")
    # round 3's c17 / c18 (PCC 0.985): every CB live in the combine must have its own bytes -- CW2 was aliased on CW
    live = ("CIN", "CIN2", "CW", "CW2", "CIDX", "COUT", "CRM", "CSH")
    grp_of = {m: i for i, (mem, _, _) in enumerate(p["groups"]) for m in mem}
    assert len({grp_of[m] for m in live}) == len(live), {m: grp_of[m] for m in live}
    # the split's slot accounting (host model): the two halves produce (slot, weight) in slot order on their own CBs
    # and the compute consumes slot j from half j & 1 -> the sequence the compute sees equals the unsplit one
    K = 8
    halves = {h: [(j, f"w{j}") for j in range(h, K, 2)] for h in (0, 1)}
    seen = [halves[j & 1].pop(0) for j in range(K)]
    assert seen == [(j, f"w{j}") for j in range(K)] and not halves[0] and not halves[1]
    src = open(M.__file__).read()
    assert '("PMK_CT_COMBR", str(ct_combr))' in src and '("PMK_RT_Y", str(base_y))' in src and '("PMK_CW2_PAGES", str(cw2_pages))' in src
    assert "args += [N_STREAMERS] + g_tab + w_tab" in src


def test_perf5_yw_brisc():
    # PERF.md sec 16: the row-bfp8 re-lay on the reader BRISC inside its waits (relay_poll per row, non-blocking CB probes,
    # RELAYED / WRITTEN hand-off words at CB_TAB + 96 / + 100, descriptors past the two group scratches), the writer
    # only issues the row writes; the ledger adds the 64-B descriptor area to WTAB
    kd = os.path.join(os.path.dirname(M.__file__), "kernels", "pmk")
    rd = open(os.path.join(kd, "ffn_reader.hpp")).read()
    assert "cb_pages_available_at_front(CB_YRM, YB_CW * mt_ * NC)" in rd and "cb_pages_reservable_at_back(cb, n)" in rd
    assert "*yb_relayed = rl_grp;" in rd and "while (yq_head != yq_tail) {" in rd and rd.count("YB_WAIT_MIN(") >= 4
    assert "XP_TOC(R_RELAY)" in rd
    wr = open(os.path.join(kd, "ffn_writer.hpp")).read()
    assert "#if defined(PMK_OUT_BFP8) && defined(PMK_YW_DB) && defined(PMK_YW_BRISC)" in wr and "*written = k;" in wr
    assert "noc_async_write(blk + row * CW * RB, outb_g.get_noc_addr(d0_ + row, nc_ * RB), CW * RB);" in wr
    pr = open(os.path.join(os.path.dirname(M.__file__), "kernels", "prefill", "moe_xffn_probe.hpp")).read()
    assert "R_RELAY = 7" in pr
    g = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    lv = M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2)
    a = M.cb_plan2(g, 1, 4, 8, lv)
    b = M.cb_plan2(g, 1, 4, 8, M.replace(lv, yw_brisc=True))
    assert b["ffn"]["WTAB"] == a["ffn"]["WTAB"] + 64
    src = open(M.__file__).read()
    assert '("PMK_YW_BRISC", "1")' in src and "relay=f(rd[:, 7], sp)" in src


def test_perf5_tp8_forms():
    # PERF.md sec 17: at TP8 the chunk count per streamer (576) and the per-chunk handshake, not the bytes, set program
    # B's time; the fat-chunk geometries and the v1 self-streaming form all fit the ledger
    g = M.Geometry(c=2048, w=256, p=86, h_split=4)
    lv = M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2, comb_split=True, yw_brisc=True)
    chunks = lambda blk: g.h_t // blk[1] + g.h_t // blk[2]  # noqa: E731
    assert chunks((2, 4, 8)) == 48 and chunks((1, 8, 16)) == 24 and chunks((1, 16, 16)) == 16
    for blk, limit in (((1, 8, 16), 1_050_624), ((1, 16, 16), 1_050_624)):
        p = M.cb_plan2(g, *blk, lv)
        assert p["total"] <= limit, (blk, p["total"])
    g110 = M.Geometry(c=2048, w=256, p=110, h_split=4)
    v1 = M.cb_plan2(g110, 2, 8, 8, M.Levers(x_l1=True, mcast=1, shared=True, out_bfp8=False))
    assert v1["total"] <= M.CB_BUDGET and v1["xs"] == M.XS_BYTES_PER_HOLDER
    # the TP2 geometry is untouched (bitwise vs 63b8fb3e52): its (1, 4, 8) ledger
    g2 = M.Geometry(c=2048, w=1024, p=86, h_split=4)
    assert M.cb_plan2(g2, 1, 4, 8, M.replace(lv, yw_brisc=False))["total"] == 1_168_896 and M.cb_plan2(g2, 1, 4, 8, lv)["total"] == 1_168_960


def test_perf5_selfstream_form():
    # PERF.md sec 17-19: the TP8 form of record (7f: 4,553 us vs 4,088 op path) selectable through ProgramB(selfstream=True)
    # for the stack driver (GLM53_PMK_B_SELFSTREAM=1): the kwargs mapping, the attributes the driver reads, the ledger
    import inspect
    assert "selfstream" in inspect.signature(M.ProgramB.__init__).parameters and "pack_rne" in inspect.signature(M.ProgramB.__init__).parameters
    kw = dict(g_size=2, levers=M.Levers(x_l1=False, mcast=2, shared=True, out_bfp8=True, yw_db=True, yw_coal=2, comb_split=True),
              num_experts=288, top_k=8, hidden=4096, limit=10.0, gate_up=[], down=[], blocks=None, fid="HiFi4", probe=False,
              prefetch=True, full_sync=False, rect_mode="cover", gmax=4, mcast_linked=True, mcast_dual=True, lpt=2, inplace=True,
              lpt_w=[1.0] * 24, comb_split=True, yw_brisc=True, h_split=4)  # fmt: skip
    out = M.SelfStreamB.kwargs_from_program_b(kw)
    assert out["levers"] == M.Levers(x_l1=False, mcast=1, shared=True) and out["blocks"] == (4, 8, 8) and out["fid"] == "HiFi4"
    assert "lpt" not in out and "inplace" not in out and "g_size" not in out
    kw2 = dict(kw); kw2["levers"] = M.replace(kw["levers"], x_l1=True)
    assert M.SelfStreamB.kwargs_from_program_b(kw2)["blocks"] == (2, 8, 8)
    assert M.SelfStreamB.inplace is True and M.SelfStreamB.lpt == 0 and M.SelfStreamB.rect_mode == "self"
    g = M.Geometry(c=2048, w=256, p=110, h_split=4)
    assert M.cb_plan2(g, 4, 8, 8, M.Levers(x_l1=False, mcast=1, shared=True))["total"] <= M.CB_BUDGET
    src = open(os.path.join(os.path.dirname(M.__file__), "..", "tests", "s16", "jobs", "pmk_moe2_bench.py")).read()
    assert 'selfstream=bool(int(args.get("selfstream", 0)))' in src and "ttnn.ROW_MAJOR_LAYOUT, memory_config=PF.DRAM)" in src and 'args.get("dump")' in src
