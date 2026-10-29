# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test (no device, plain pytest on CPU) of FabricStreamA2A2D's page-run lists (tt/fabric_a2a.py, doc
pmk/merge/a2a_fs_design.md): the ENCODED runtime args of every lane of every chip (decoded as a2a_read.cpp /
a2a_send.cpp read them) are replayed on host tensors by fabric_a2a.simulate -- an adversarial scheduler that runs the
B lanes first and every lane as far as its relay-counter / credit waits allow -- and the result must equal the op's
a2a (fabric_a2a.ref_op_a2a: all_to_all_async_generic per axis in MlaPrefill._a2a's order, torch chunk / cat) bitwise,
for q (heads -> rows) and o (rows -> heads), on the 2x4 (two stages) and the pair (one axis), at C = 2,048. The
simulator also fails on a page written twice or never, a relay run read before it landed, a send before its
destination's credit, a deadlock, or a landed counter that is not exactly g x its per-launch count."""
import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import fabric_a2a as A

ROWS = 2048
# (name, R, C, form, axis, heads per chip): the served TP8 (2x4, 8 heads), the pair (TP2, 32 heads), the pair at the TP8
# payload (8 heads), and the 2x4's single-axis forms (every row of 4 / column of 2 its own line)
CASES = [("tp8", 2, 4, "2d", None, 8), ("pair", 1, 2, "line", None, 32), ("pair_h8", 1, 2, "line", None, 8),
         ("rows4", 2, 4, "line", 1, 8), ("cols2", 2, 4, "line", 0, 8)]  # fmt: skip


def _xs(plan, seed):
    g = torch.Generator().manual_seed(seed)
    return {rc: torch.randn(*plan.x_shape()[1:], generator=g).to(torch.bfloat16) for rc in plan.chips}


def _plan(R, C, form, axis, hr, q, rows=ROWS, **kw):
    return A.A2APlan(R, C, hr=hr, rows=rows, heads_to_rows=q, form=form, axis=axis, **kw)


def _ref(plan, xs):
    return A.ref_op_a2a(xs, plan.R, plan.C, plan.q, axes=None if plan.form == "2d" else [plan.axis_b])


@pytest.mark.parametrize("q", [True, False], ids=["q", "o"])
@pytest.mark.parametrize("name,R,C,form,axis,hr", CASES, ids=[c[0] for c in CASES])
def test_lists_equal_the_op(name, R, C, form, axis, hr, q):
    plan = _plan(R, C, form, axis, hr, q)
    xs = _xs(plan, 11 + hr)
    ys = A.simulate(plan, xs)
    ref = _ref(plan, xs)
    for rc in plan.chips:
        assert tuple(ys[rc].shape) == plan.y_shape()[1:]
        assert torch.equal(ys[rc], ref[rc]), (name, "q" if q else "o", rc)


@pytest.mark.parametrize("name,R,C,form,axis,hr", CASES[:2], ids=[c[0] for c in CASES[:2]])
def test_launches_and_shared_counters(name, R, C, form, axis, hr):
    """the monotonic counters across launches: q, o, q at 2,048 rows then q, o at 256 rows on ONE counter state and ONE
    relay store (as TPPrefill shares them across keys), each launch with new data; a too-weak relay / credit / landed
    wait would read the previous launch's relay rows or miscount"""
    st = {}
    seq = [(True, ROWS), (False, ROWS), (True, ROWS), (True, 256), (False, 256)]
    for i, (q, rows) in enumerate(seq):
        plan = _plan(R, C, form, axis, hr, q, rows=rows)
        xs = _xs(plan, 100 + i)
        ys = A.simulate(plan, xs, state=st)
        ref = _ref(plan, xs)
        assert all(torch.equal(ys[rc], ref[rc]) for rc in plan.chips), (name, i, q, rows)
    g = {v[A.GEN_B] for v in st["cnt"].values()} | {v[A.GEN_NC] for v in st["cnt"].values()}
    assert g == {len(seq)}, g


@pytest.mark.parametrize("name,R,C,form,axis,hr", CASES[:2], ids=[c[0] for c in CASES[:2]])
def test_round_trip(name, R, C, form, axis, hr):
    pq, po = _plan(R, C, form, axis, hr, True), _plan(R, C, form, axis, hr, False)
    xs = _xs(pq, 5)
    back = A.simulate(po, A.simulate(pq, xs))
    assert all(torch.equal(back[rc], xs[rc]) for rc in pq.chips)


def test_tp8_head_order_is_the_ops():
    """the closed form of the op's 2x4 q a2a (axis 0, then axis 1): chip d = (rd, cd) holds row block rd C + cd and the
    heads of source s = (rs, cs) at slot cs R + rs (column-major, NOT the rank rs C + cs)"""
    R, C, hr = 2, 4, 8
    plan = _plan(R, C, "2d", None, hr, True)
    xs = _xs(plan, 3)
    ref = _ref(plan, xs)
    rl = ROWS // (R * C)
    for rd, cd in plan.chips:
        k = rd * C + cd
        exp = torch.cat([xs[(rs, cs)][:, k * rl:(k + 1) * rl] for cs in range(C) for rs in range(R)], dim=0)
        assert torch.equal(ref[(rd, cd)], exp)
    assert plan.slot((1, 0)) == 1 and plan.slot((0, 1)) == 2


@pytest.mark.parametrize("rows", [256, 512, 2048])
def test_tp8_resources(rows):
    """per chip and lane: <= 12 packet headers per BRISC, one worker per (link, axis, direction) (A2APlan._audit), the
    runtime args within the 341 limit with room for 2 fabric connections, every run a whole number of 16-page batches,
    and the per-launch counts independent of the rows and of the direction (one sync tensor for every key)"""
    noc = {("B", 0): (1, 2), ("B", 1): (1, 3), ("A", 0): (1, 4), ("A", 1): (1, 5)}
    sig = None
    for q in (True, False):
        plan = _plan(2, 4, "2d", None, 8, q, rows=rows)
        assert plan.rpb % 16 == 0
        assert plan.relay_pages == 3 * 8 * plan.rpb
        cnts = []
        for rc in plan.chips:
            enc = plan.encode(rc, noc)
            for k_, (rd, sd) in enc.items():
                assert len(rd) <= A.MAX_RT_ARGS and len(sd) + 32 <= A.MAX_RT_ARGS, (rc, k_, len(rd), len(sd))
                lp = plan.lanes[rc][k_]
                assert A.NHDR * (int(lp["up"] > 0) + int(lp["dn"] > 0)) <= A.HDR_BUDGET
                cnts.append((rc, k_, lp["relay_per"], len(lp["credits"]), len(lp["runs"])))
            cnts.append((rc, tuple(sorted(plan.waiter[rc]["wait"].items())), plan.waiter[rc]["lanes"]))
        sig = sig or cnts
        assert cnts == sig, "per-launch counts differ between keys / directions: one shared sync would miscount"


def test_tp8_traffic():
    """the bytes per chip of the design note: 14 MiB to 7 peers + 2 MiB local; stage A 8 MiB to the partner (2 MiB of
    it the partner's own), stage B 4 MiB per row-mate (12 MiB), the middle links the busiest"""
    plan = _plan(2, 4, "2d", None, 8, True)
    mib = plan.rpb * A.TB / 2**20
    for rc in plan.chips:
        a = sum(1 for k_, lp in plan.lanes[rc].items() if k_[0] == "A" for r in lp["runs"] if r["route"] != A.ROUTE_LOCAL)
        b = sum(1 for k_, lp in plan.lanes[rc].items() if k_[0] == "B" for r in lp["runs"] if r["route"] != A.ROUTE_LOCAL)
        loc = sum(1 for lp in plan.lanes[rc].values() for r in lp["runs"] if r["route"] == A.ROUTE_LOCAL)
        assert (a * mib, b * mib, loc * mib) == (8.0, 12.0, 2.0), (rc, a * mib, b * mib, loc * mib)


@pytest.mark.parametrize("bug", ["early_relay", "relay_per", "no_credit", "dup_page"])
def test_simulator_catches_broken_lists(bug):
    """the checks bite: a relay run that waits one count short (reads before its row landed), a wrong relay count per
    launch (launch 2 reads launch 1's relay rows), a missing credit (deadlock), two runs to one page"""
    plan = _plan(2, 4, "2d", None, 8, True)
    lp = plan.lanes[(0, 1)][("B", 0)]
    relay_runs = [r for r in lp["runs"] if r["src"] == "relay"]
    if bug == "early_relay":
        relay_runs[2]["need"] -= 1
    elif bug == "relay_per":
        for rc in plan.chips:
            for k_, l_ in plan.lanes[rc].items():
                if k_[0] == "B":
                    l_["relay_per"] -= 1
    elif bug == "no_credit":
        plan.lanes[(0, 2)][("B", 0)]["credits"].pop(0)
    else:
        own = [r for r in lp["runs"] if r["src"] == "x"]
        own[1]["dst_page"], own[1]["dst_rc"] = own[0]["dst_page"], own[0]["dst_rc"]
        own[1]["route"], own[1]["hops"], own[1]["cpos"] = own[0]["route"], own[0]["hops"], own[0]["cpos"]
    st = {}
    with pytest.raises(AssertionError):
        for i in range(2):
            xs = _xs(plan, 200 + i)
            ys = A.simulate(plan, xs, state=st)
            ref = _ref(plan, xs)
            assert all(torch.equal(ys[rc], ref[rc]) for rc in plan.chips), "a wrong result"
