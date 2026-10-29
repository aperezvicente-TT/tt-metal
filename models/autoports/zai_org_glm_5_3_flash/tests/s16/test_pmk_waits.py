# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python unit test of the PMK family's hand-off counters (no ttnn, no device): walk the full P0..P16 sequence of
the sparse_kda plan in BOTH geometries (TP8 production, TP2 pair units) and assert that, per program semaphore id, the
wait_min values the composite emits are STRICTLY INCREASING in program order (cumulative counters, reset once per
layer: a later hand-off on a shared id must wait for base + its increments -- the Stage 14 finding: P16's wait on 14 once
passed on P8's increments), that every wait equals the increments accumulated on that id up to it, and that the
generated composite kernels carry exactly those values. Run: python -m pytest tests/s16/test_pmk_waits.py -q"""

import re

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK

AVOID = tuple(([(x, y) for x in (0, 1, 7, 8, 9) for y in (1, 4, 6, 9)] + [(2, 1), (2, 4), (2, 6), (2, 9)])[:24])


def _plan(tp, kind="sparse_kda"):
    return PMK.PmkSparseKdaPlan(PMK.PmkGeom(tp=tp, kind=kind), PMK.Placement(11, 10, avoid=AVOID)).plan()


@pytest.mark.parametrize("tp,kind", [(8, "sparse_kda"), (2, "sparse_kda"), (8, "dense_kda"), (2, "dense_kda")])
def test_waits_strictly_increasing_per_semaphore(tp, kind):
    fam = _plan(tp, kind)
    sched = fam.wait_schedule()
    assert sched, "no semaphore waits in the plan"
    for sid, uses in sched.items():
        # one hand-off INSTANCE may be consumed by several phases (H8 by P12 / P13 / P14, H4 by P7 / P9): the same event,
        # the same value; DISTINCT instances on one id must wait strictly increasing values in program order
        inst, values = [], []
        for hid, pid, v in uses:
            if inst and inst[-1] == hid:
                assert (
                    v == values[-1]
                ), f"sem {sid}: {hid} consumed again at {pid} with another value {v} != {values[-1]}"
                continue
            assert hid not in inst, f"sem {sid}: {hid} re-consumed at {pid} after another instance: {uses}"
            inst.append(hid)
            values.append(v)
        assert all(b > a for a, b in zip(values, values[1:])), f"sem {sid}: waits not strictly increasing: {uses}"
        # the wait equals the increments accumulated on the id by the instances waited so far (nothing waits for more
        # than the producers add in the layer, nothing waits for less than an earlier use already consumed)
        acc = 0
        for hid, v in zip(inst, values):
            acc += fam.handoffs[hid].expected
            assert v == acc, f"sem {sid}: {hid} waits {v}, increments so far {acc}"


@pytest.mark.parametrize("tp,kind", [(8, "sparse_kda"), (2, "sparse_kda"), (8, "dense_kda"), (2, "dense_kda")])
def test_shared_ids_have_bases(tp, kind):
    fam = _plan(tp, kind)
    h6, h10 = fam.handoffs["H6"], fam.handoffs["H10"]
    assert (
        h6.counter == h10.counter == PMK.SEM_FS_REDUCED
        and h10.wait_base == h6.expected
        and h10.wait_value > h6.wait_value
    )
    h2, h8 = fam.handoffs["H2"], fam.handoffs["H8"]
    assert h8.wait_base == h2.expected and h8.wait_value == 2 * h2.expected
    h0, h0f = fam.handoffs["H0"], fam.handoffs["H0f"]
    assert h0.wait_value == 1 and h0f.wait_value == 2  # P2 waits 1, P11 waits 2 on the members' XS-FREE 12


@pytest.mark.parametrize("tp,kind", [(8, "sparse_kda"), (2, "sparse_kda"), (8, "dense_kda"), (2, "dense_kda")])
def test_composite_emits_wait_values(tp, kind):
    fam = _plan(tp, kind)
    for p in fam.phases:  # register pseudo-bodies so the composite emits the waits
        p.body.status = "body"
    emitted = {}
    for role in ("member", "root", "mm_only"):
        for risc in ("dm0", "dm1"):
            src = fam.composite_source(role, risc)
            for sid, val in re.findall(r"pmk::wait_min\((\d+), (\d+)\)", src):
                emitted.setdefault((role, risc, int(sid)), []).append(int(val))
    assert emitted
    for (role, risc, sid), vals in emitted.items():
        distinct = [v for n, v in enumerate(vals) if n == 0 or v != vals[n - 1]]
        assert all(
            b > a for a, b in zip(distinct, distinct[1:])
        ), f"{role}/{risc} sem {sid}: emitted waits {vals} not increasing"
    sched = fam.wait_schedule()
    for (role, risc, sid), vals in emitted.items():
        allowed = {v for _, _, v in sched[sid]}
        assert set(vals) <= allowed, (role, risc, sid, vals, allowed)
    # the KDA scan phases (sub-role of the MM-only cores) are hosted by the MM-only composite: P6 waits H3b (6), P9 H4 (7)
    assert ("mm_only", "dm0", PMK.SEM_KDA) in emitted and ("mm_only", "dm0", PMK.SEM_ON) in emitted


@pytest.mark.parametrize("tp,kind", [(8, "sparse_kda"), (2, "sparse_kda"), (8, "dense_kda"), (2, "dense_kda")])
def test_census_and_hazard(tp, kind):
    fam = _plan(tp, kind)
    fam.check_l1()
    fam.credits = {}
    raised = False
    try:  # the cross-core alias hazard must be refused without the XS-FREE credit
        fam._liveness_check()
    except ValueError:
        raised = True
    assert raised, "liveness check passed without the XS-FREE credit (cross-core hazard)"


@pytest.mark.parametrize("tp,kind", [(8, "sparse_kda"), (2, "sparse_kda"), (8, "dense_kda"), (2, "dense_kda")])
def test_kda_cbs_never_aliased(tp, kind):
    """the kda piece's CBs (items on the members, scan on the MM-only cores, at the engine's free indices) are SOLO: each
    its own allocation, never in an ALIASES group / VIEW (13:3xZ silicon: scan W and SR shared one allocation through
    the mhc X / X32 view pair -> the gated-norm `on` collapsed)"""
    fam = _plan(tp, kind)
    for role, key in (("member", "item"), ("mm_only", "scan")):
        idx = set(fam.kda_map[key])
        for slot_idx, _, _, _ in fam._slots(role):
            if set(slot_idx) & idx:
                assert len(slot_idx) == 1, (role, slot_idx)


@pytest.mark.parametrize("tp", [8, 2])
def test_program_scope_split(tp):
    """the v1 split (ruling 13:5xZ): program A = P0..P11, B = P12..P16. A scoped composite runs only its program's
    phases, and a hand-off whose producer sits in the OTHER program is never waited on (its data is in DRAM)"""
    fam = _plan(tp, "sparse_kda")
    for p in fam.phases:
        p.body.status = "body"
    A = {f"P{i}" for i in range(0, 12)}
    B = {f"P{i}" for i in range(12, 17)}
    fam.programs = [A, B]
    for sc in (A, B):
        fam.scope = sc
        try:
            for role in ("member", "root", "mm_only"):
                for risc in ("dm0", "dm1", "c"):
                    src = fam.composite_source(role, risc)
                    ran = set(re.findall(r"ph(\d+)::phase_main\(\);", src))
                    assert {f"P{i}" for i in ran} <= sc, (role, risc, sorted(ran))
                    for hid in re.findall(r"// (H\w+) [^\n]*no registered producer yet", src):
                        first = [q for q in fam.phases if hid in q.signals][0]  # the counter's raiser
                        assert first.pid not in sc, (hid, first.pid)
        finally:
            fam.scope = None
    # H8 (P11 -> P12..P14) crosses the boundary: no wait for it in B
    fam.scope = B
    try:
        assert not fam._has_live_producer("H8")
    finally:
        fam.scope = None


@pytest.mark.parametrize("tp", [8, 2])
def test_dsa_moe_half_wait_bases(tp, capsys):
    """G v2a's program (the DSA MoE half, P10..P14) has P11's AG but no P2's: a program semaphore starts at 0 every
    launch, so H8 must wait FS_ARRIVE >= 8, not the layer-wide 16 (the gv2w hang, 2026-10-25 21:30Z). wait_base_scope
    scopes the bases; the reach check refuses (or, unscoped, reports) a wait the program's producers cannot reach"""
    fam = _plan(tp, "sparse_kda")
    for p in fam.phases:
        p.body.status = "body"
    fam.scope = {"P10", "P11", "P12", "P13", "P14"}
    try:
        h8 = fam.handoffs["H8"]
        assert fam._wait_value(h8) == h8.wait_value == 2 * h8.expected  # unscoped: the layer-wide value
        fam._wait_reach_check({})
        assert "H8" in capsys.readouterr().out  # reported: unreachable in this program
        fam.wait_base_scope = True
        assert fam._wait_value(h8) == h8.expected
        fam._wait_reach_check({})  # raises on any unreachable wait
        for role in ("member", "mm_only"):
            for risc in ("dm0", "dm1"):
                src = fam.composite_source(role, risc)
                vals = {int(v) for v in re.findall(rf"pmk::wait_min\({PMK.SEM_FS_ARRIVE}, (\d+)\)", src)}
                assert vals <= {h8.expected}, (role, risc, vals)
    finally:
        fam.scope = None
        fam.wait_base_scope = False
    assert fam._wait_value(fam.handoffs["H8"]) == 2 * fam.handoffs["H8"].expected  # the fused layer program: unchanged


@pytest.mark.parametrize("tp", [8, 2])
def test_h0f_credit_after_resident_x_store(tp):
    """the TP8 x1 race (w7tp8 / w7tp8c / w7tp8d, 2026-10-25): CB_X shares the window allocation with XS 35, so P10's
    XS-FREE credit H0f must come from the member RISC that stores the resident residual (dm0, 'xstore') when P11's slice
    writers run in the same program; otherwise (no store, or no P11 here) it stays on dm1 as P1's H0"""
    fam = _plan(tp, "sparse_kda")
    p10 = next(p for p in fam.phases if p.pid == "P10")
    p10.body.status = "body"
    assert fam._h0f_risc() == "dm1"  # no xstore part
    p10.body.parts[("member", "dm0")] = [dict(name="xload"), dict(name="mhc"), dict(name="xstore")]
    assert fam._h0f_risc() == "dm0"  # whole layer in scope: P11's writers follow
    fam.scope = {"P10"}
    try:
        assert fam._h0f_risc() == "dm1"  # P11 not in this program: no writer waits on the credit here
        fam.scope = {"P10", "P11", "P12"}
        assert fam._h0f_risc() == "dm0"
    finally:
        fam.scope = None


@pytest.mark.parametrize("tp", [8, 2])
def test_own_xstore_guard(tp):
    """the window-5 X / XS race (16's pmk/w7/ovl_model.md section 2; program ORDER 2026-10-27 10:05:53Z on 14 / 14b's
    ruling): with P10's resident-X store on member dm0 and P11 in the same program, dm0 raises SEM_XSTORED once after
    P10's parts (the store's write barrier, with H0f) and P11's member dm1 waits it before the stub; without the store
    (the streamed residual) or with P11 in another program, neither line exists"""
    fam = _plan(tp, "sparse_kda")
    for p in fam.phases:
        p.body.status = "body"
    sig = f"mk::signal_self({PMK.SEM_XSTORED});"
    wait = f"pmk::wait_min({PMK.SEM_XSTORED}, 1);"

    def srcs():
        return fam.composite_source("member", "dm0"), fam.composite_source("member", "dm1")

    d0, d1 = srcs()
    assert sig not in d0 and wait not in d1  # no xstore part: H0f stays on dm1, no guard
    p10 = next(p for p in fam.phases if p.pid == "P10")
    p10.body.parts[("member", "dm0")] = [dict(name="xload"), dict(name="mhc"), dict(name="xstore")]
    d0, d1 = srcs()
    assert d0.count(sig) == 1 and d1.count(wait) == 1, (d0.count(sig), d1.count(wait))
    last10 = max(m.end() for m in re.finditer(r"::phase_main\(\);  // P10 ", d0))
    assert d0.index(sig) > last10  # after every P10 part on dm0 (the xstore among them)
    assert d0.rfind("noc_async_write_barrier();", 0, d0.index(sig)) > last10  # the store's writes landed first
    first11 = min(m.start() for m in re.finditer(r"::phase_main\(\);  // P11 ", d1))
    assert d1.index(wait) < first11  # before the stub (the XS-FREE credit and the XS write)
    assert max(m.end() for m in re.finditer(r"::phase_main\(\);  // P10 ", d1)) < d1.index(wait)
    assert "member" in fam.sems[PMK.SEM_XSTORED]
    fam.scope = {"P10"}
    try:
        d0, d1 = srcs()
        assert sig not in d0 and wait not in d1  # P11 in another program: no guard (the launch boundary orders it)
    finally:
        fam.scope = None
