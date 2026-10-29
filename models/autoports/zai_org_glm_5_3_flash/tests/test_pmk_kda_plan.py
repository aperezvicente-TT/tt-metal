# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python unit test of the PMK KDA phase plan (tt/pmk_kda.py): shapes, CB tables, L1 budgets, hand-off page
offsets (kernel <-> host agreement), both geometries (TP8: 2,048 rows x 8 heads; TP2: 2,048 rows x 32 heads), the
utilization estimate. No ttnn, no device: `python -m pytest tests/test_pmk_kda_plan.py` or run as a script."""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_kda as P  # noqa: E402

TP8 = P.KdaGeometry(rows=2048, heads=8)
TP2 = P.KdaGeometry(rows=2048, heads=32)
HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "tt", "kernels", "pmk")


def test_geometries():
    assert TP8.n_chunks == 64 and TP8.cph == 8 and TP8.spread == 4 and TP8.vt == 1 and TP8.scan_cores == 32
    assert TP2.n_chunks == 64 and TP2.cph == 2 and TP2.spread == 1 and TP2.vt == 4 and TP2.scan_cores == 32
    small = P.KdaGeometry(rows=256, heads=8)  # the 256-row instantiation (8 chunks per head)
    assert small.n_chunks == 8
    for bad in (dict(rows=100, heads=8), dict(rows=2048, heads=3), dict(rows=2048, heads=64, scan_budget=32)):
        try:
            P.KdaGeometry(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_cb_tables_and_budgets():
    for geo in (TP8, TP2):
        pl = P.plan(geo)
        assert pl.check() == [], pl.check()
        assert [c.index for c in pl.item] == list(range(32, 55))  # §4b prepare range 32..49 extended to 54
        assert [c.index for c in pl.scan] == list(range(50, 64))  # §4b scan range 50..63
        assert pl.item_l1 <= P.L1_CB_BUDGET and pl.scan_l1 <= P.L1_CB_BUDGET
    tp8 = P.plan(TP8)
    assert tp8.scan_l1 <= 384 * 1024, tp8.scan_l1  # the contract's §4 scan-core plan
    # on an mHC member (256 KB residual resident) the item CBs still fit
    assert tp8.check(resident_bytes=256 * 1024) == []
    # the L1 form of the H3 store does not fit next to the item CBs at TP8: the DRAM form is the one built
    assert not tp8.l1_store_fits()
    assert not P.plan(TP2).l1_store_fits()


def test_page_offsets_match_kernel_header():
    """the TERMS / IN ring page layout of kda_common.hpp == pmk_kda.py's (the host sizes the CBs and scratch)"""
    src = open(os.path.join(KDIR, "kda_common.hpp")).read()
    assert "constexpr uint32_t out_pages(uint32_t kt, uint32_t vtf) { return vtf + 4 * kt + 2; }" in src
    assert "constexpr uint32_t in_pages(uint32_t kt, uint32_t vt) { return 2 * vt + 4 * kt + 2; }" in src
    assert P.out_pages(4, 4) == 22 and P.in_pages(4, 1) == 20 and P.in_pages(4, 4) == 26
    for name, idx in (("CB_U", 0), ("CB_OUT", 22), ("CB_S", 0), ("CB_ON", 13)):
        base = "PK_ITEM_CB0" if idx <= 22 and name != "CB_S" and name != "CB_ON" else "PK_SCAN_CB0"
        assert re.search(rf"{name}\s*=\s*{base}\s*\+\s*{idx};", src), name  # clang-format may wrap the line
    sems = dict(re.findall(r"constexpr uint32_t (SEM_\w+) = (\d+);", src))
    assert {k: int(v) for k, v in sems.items()} == dict(SEM_QKV=P.SEM_QKV, SEM_KDA=P.SEM_KDA, SEM_ON=P.SEM_ON)
    assert "#define PK_SEM_RMS 0" in src and "#define PK_SEM_ONAGG 1" in src and P.SEM_RMS == 0 and P.SEM_ONAGG == 1


def test_runtime_arg_layout_matches_kernels():
    """the kernels' documented runtime-arg counts == the host's (checked textually on the kernel sources)"""
    w = open(os.path.join(KDIR, "kda_scan_w.cpp")).read()
    assert "A_PEERS = ph_rt0 + 13" in w and "A_CONS = A_PEERS + 2 * (SPREAD - 1)" in w
    r = open(os.path.join(KDIR, "kda_item_r.cpp")).read()
    assert (
        "ph_rt0 + 13" in r and "ph_rt0 + 10)" in r and "ph_rt0 + 7)" in r
    )  # 13 runtime args: .. taps 0..3 at 7..10, dt 11, rate 12
    iw = open(os.path.join(KDIR, "kda_item_w.cpp")).read()
    assert (
        "ph_rt0 + 7 + 2 * s" in iw and "ph_rt0 + 4), ny = get_arg_val<uint32_t>(ph_rt0 + 5)" in iw
    )  # baton core at 4, 5; scan cores from 6


def test_unpack_modes_only_on_copy_cbs():
    """UnpackToDestFp32 only on CBs the compute never feeds to the FPU (the r3-probe zero-matmul lesson)"""
    item = {c.name for c in P.item_cbs(TP8)}
    scan = {c.name for c in P.scan_cbs(TP8)}
    assert set(P.ITEM_COPY_ONLY_CBS) <= item and set(P.SCAN_COPY_ONLY_CBS) <= scan
    fpu_item = {
        "CONST",
        "G",
        "NQ",
        "NK",
        "W0",
        "W1",
        "W2",
        "W3",
        "TW0",
        "TW1",
        "TW2",
        "SD",
        "CID",
        "ANCH",
        "AKK",
        "OUT",
    }
    fpu_scan = {"S", "IN", "W", "SR", "VN", "OI", "SCR", "SU", "ST", "FS", "PSUM"}
    assert not (set(P.ITEM_COPY_ONLY_CBS) & fpu_item) and not (set(P.SCAN_COPY_ONLY_CBS) & fpu_scan)
    src = open(os.path.join(KDIR, "kda_scan_c.cpp")).read()
    for cb in ("CB_S", "CB_IN", "CB_SR", "CB_VN", "CB_OI", "CB_SCR", "CB_SU", "CB_ST"):
        assert cb in src


def test_cb_block_rule():
    """a reserved block never straddles a CB's wrap: every ring holds a whole number of the blocks pushed to it and
    all blocks on a ring share one size (the r4 lesson: beta as a 1-tile block on the 4-tile W1 ring)"""
    for geo in (TP8, TP2):
        pl = P.plan(geo)
        kt, vt, ct = geo.kt, geo.vt, 3 * geo.kt
        item_blocks = dict(U=2 * ct, F=kt, BETA=1, MASK=1, CONST=12, TAPS=4 * ct, DTR=2 * kt, CONV=ct, G=kt, NQ=kt, NK=kt, W0=kt,
                           W1=kt, W2=kt, W3=kt, TW0=1, TW1=1, TW2=1, SD=kt, CID=kt, ANCH=kt, AKK=1, OUT=P.out_pages(kt, kt))  # fmt: skip
        scan_blocks = dict(S=kt * vt, IN=P.in_pages(kt, vt), W=vt + 1, SR=kt * vt, VN=vt, OI=vt, SCR=vt, SU=kt * vt, ST=kt * vt,
                           FS=kt * vt, O=vt, PSUM=1, PIN=max(1, geo.spread - 1), ON=vt)  # fmt: skip
        for cbs, blocks in ((pl.item, item_blocks), (pl.scan, scan_blocks)):
            for c in cbs:
                assert c.pages % blocks[c.name] == 0, (geo.heads, c.name, c.pages, blocks[c.name])
    src = open(os.path.join(KDIR, "kda_item_c.cpp")).read()
    assert "cb_reserve_back(CB_TW1, 1)" in src and "cb_pop_front(CB_W1, 1)" not in src
    src = open(os.path.join(KDIR, "kda_scan_c.cpp")).read()
    assert "cb_reserve_back(CB_OI, Vt)" in src and "cb_reserve_back(CB_OI, 1)" not in src


def test_scratch_and_estimate():
    pl = P.plan(TP8)
    s = pl.scratch_bytes()
    assert s["u"] == 2048 * 3072 * 2 and s["terms"] == 8 * 64 * 22 * 4096 and s["state_s"] == 8 * 128 * 128 * 4
    e = pl.estimate()
    assert 0.5 <= e["compute_core_busy"] <= 1.0, e  # fp32 chunk terms: the value-split scan re-reads them (DRAM-bound)
    assert e["scan_dram_mb"] > 100
    g16 = P.KdaGeometry(rows=2048, heads=8, terms_dtype="bf16")  # the chunk-term knob: bf16 terms + gate_out
    p16 = P.plan(g16)
    assert [c.fmt for c in p16.item if c.name == "OUT"] == ["bf16"] and [c.fmt for c in p16.scan if c.name == "IN"] == [
        "bf16"
    ]
    assert p16.scratch_bytes()["terms"] == pl.scratch_bytes()["terms"] // 2
    e16 = p16.estimate()
    assert e16["scan_dram_mb"] * 2 == e["scan_dram_mb"] and 0.6 <= e16["compute_core_busy"] <= 1.0, e16
    e2 = P.plan(TP2).estimate()
    assert e2["items_per_core"] == 32 and e2["scan_compute_us"] > e["scan_compute_us"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    for geo in (TP8, TP2):
        pl = P.plan(geo)
        print(f"{geo.heads} heads: item CBs {pl.item_l1} B, scan CBs {pl.scan_l1} B, scratch {pl.scratch_bytes()}")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in pl.estimate().items()})
