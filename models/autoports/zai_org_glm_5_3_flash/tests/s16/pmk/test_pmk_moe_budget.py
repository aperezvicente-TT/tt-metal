# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python unit test of the PMK MoE phase set's CB / L1 budget and streaming model (tt/pmk_moe.py; no device, no
ttnn op): both geometries of prefill_megakernel.md sec 1b (TP8: 256-wide slices, TP2: 1,024-wide) at the 2,048 bucket
fit the ~1,026 KB CB budget with the aliases, every alias group's total is a multiple of its members' page sizes, the
alias members never share a CB index, and the streaming arithmetic reproduces the README numbers.
Run: python -m pytest tests/s16/pmk/test_pmk_moe_budget.py -q  (or python tests/s16/pmk/test_pmk_moe_budget.py)"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6)))

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as M  # noqa: E402


def _geo(w, p=110, c=2048):
    return M.Geometry(c=c, w=w, p=p)


def test_blocks_and_budget_tp8():
    g = _geo(256)
    assert M.choose_blocks(g) == (4, 8, 8)  # xffn's TP8 choice
    plan = M.cb_plan(g, 4, 8, 8)
    assert plan["total"] <= M.CB_BUDGET, plan["total"]
    assert plan["ffn"]["WGU"] == 2 * 8 * 16 * 576 and plan["ffn"]["GUP"] == 4 * 16 * 4096
    for members, total, pages in plan["groups"]:
        for m in members:
            assert total % pages[m] == 0, (members, total, pages)


def test_blocks_and_budget_tp2():
    g = _geo(1024)
    assert M.choose_blocks(g) == (1, 4, 8)  # W = 32 tiles: MG = 1 (the fp32 partials: 64 tiles x 4 KB)
    assert M.choose_blocks(g, budget=900 * 1024) == (1, 4, 4)  # xffn's own 900 KB rule (same numerics: NC is layout)
    plan = M.cb_plan(g, 1, 4, 8)
    assert plan["total"] <= M.CB_BUDGET, plan["total"]


def test_aliases_cover_all_cbs_once():
    g = _geo(256)
    plan = M.cb_plan(g, 4, 8, 8)
    seen = [m for members, _, _ in plan["groups"] for m in members]
    assert len(seen) == len(set(seen)) == len(M.PmkMoe.CB_INDEX) + len(M.PmkMoe.CB_INDEX_LOW)
    idx = [44 + M.PmkMoe.CB_INDEX[m] if m in M.PmkMoe.CB_INDEX else 27 + M.PmkMoe.CB_INDEX_LOW[m] for m in seen]
    assert len(set(idx)) == len(idx) and max(idx) < 64 and all(27 + v < 32 for v in M.PmkMoe.CB_INDEX_LOW.values())


def test_mock_grid_budget():
    # the 2x4 mock's grid may differ from the p300a's 11 x 10; every P from 64 to 140 must plan
    for p in (64, 110, 120, 130, 140):
        for w in (256, 1024):
            g = _geo(w, p=p)
            mg, kc, nc = M.choose_blocks(g)
            assert M.cb_plan(g, mg, kc, nc)["total"] <= M.CB_BUDGET, (p, w)


def test_streaming_model_tp8():
    g = _geo(256)
    m = M.streaming_model(g, 4)
    assert abs(m["per_expert_mb"] - 1.769) < 0.01  # 1.18 MB gate|up + 0.59 MB down per slice
    assert abs(m["weight_mb"] - 288 * 1.769) < 1.0  # uniform 57 rows per expert: one M group each
    assert abs(m["gflop"] - 103.1) < 0.5
    assert 1.0 < m["dram_floor_us"] / 1000 < 2.5
    assert 0.15 < m["math_ceiling"] < 0.4  # DRAM-bound at 2,048 rows (README)
    assert m["crossover_rows"] > 4000


def test_streaming_model_tp2_reread():
    g = _geo(1024)
    m = M.streaming_model(g, 1)  # MG = 1: the weights re-read per tile row
    assert m["reread_factor"] > 1.5  # 57 rows -> 2 tile rows per expert
    assert m["weight_mb"] > 3000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    g8, g2 = _geo(256), _geo(1024)
    for name, g, blk in (("TP8", g8, (4, 8, 8)), ("TP2", g2, (1, 4, 4))):
        plan = M.cb_plan(g, *blk)
        print(
            name,
            "blocks",
            blk,
            "CB total",
            plan["total"],
            "B",
            {k: v for k, v in plan["ffn"].items()},
            "comb",
            plan["comb"],
            "plan words",
            plan["plan"],
        )
        for fid in ("HiFi2", "LoFi"):
            print(
                name,
                fid,
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in M.streaming_model(g, blk[0], fid).items()},
            )
