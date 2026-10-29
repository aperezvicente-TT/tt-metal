# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit test (no device: fake submeshes and a fake loader) of MultichipDecoder's per-chip parallel weight
loading (`parallel_load=True`): load jobs grouped per chip (a chip holding two stages — the wrap plan's head + tail
on chip 0 — is ONE job list), split units first on every chip (Stage 04a review C-7), loads of different chips
concurrent, each chip's loads on one thread in job order, the same decoder per (stage, layer) as the serial load,
and a failing chip re-raised after every chip finished.

run: python -m pytest -q tests/test_multichip_parallel_load_cpu.py   (under .agents/scripts/no-tt-devices)
"""
import threading
import time

from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import (
    MultichipDecoder,
    StagePlan,
    load_jobs,
    parse_units,
    unit_halves,
)

# wrap-plan shape in small: chip (1,3) runs the head and the tail, splits inside L1 and L5
PLAN = [
    ((1, 3), ["L0", "L1.attn"]),
    ((1, 2), ["L1.ffn", "L2", "L3"]),
    ((1, 1), ["L4", "L5.attn"]),
    ((1, 0), ["L5.ffn", "L6"]),
    ((1, 3), ["L7", "L8"]),
]
DELAY = 0.05  # per layer load (time.sleep releases the GIL like the patched load path)


class FakeDecoder:
    def __init__(self, layer, half, sub):
        self.layer, self.half, self.sub = layer, half, sub
        self.is_kda = layer % 4 != 3


class FakeLoader:
    def __init__(self, fail_layer=None):
        self.calls, self.lock, self.fail_layer = [], threading.Lock(), fail_layer

    def __call__(self, layer, sub, half=None):
        t0 = time.perf_counter()
        time.sleep(DELAY)
        if layer == self.fail_layer:
            raise ValueError(f"fake load failure L{layer}")
        with self.lock:
            self.calls.append(
                dict(layer=layer, half=half, sub=sub, thread=threading.get_ident(), t0=t0, t1=time.perf_counter())
            )
        return FakeDecoder(layer, half, sub)


def stages():
    return [StagePlan(coord=c, units=parse_units(us), chip=i) for i, (c, us) in enumerate(PLAN)]


def build(parallel, loader):
    st = stages()
    subs = {c: f"sub{c}" for c, _ in PLAN}
    t0 = time.perf_counter()
    m = MultichipDecoder(
        None, st, loader=loader, handoff="host", submeshes=subs, drop_unused=False, parallel_load=parallel
    )
    return m, time.perf_counter() - t0


def test_load_jobs_grouping_and_order():
    st = stages()
    jobs = load_jobs(st, [unit_halves(p) for p in st])
    assert sorted(jobs) == sorted({c for c, _ in PLAN})
    head_tail = jobs[(1, 3)]  # head (stage 0) and tail (stage 4) on one chip: one job list, split unit first
    assert head_tail == [(0, 1, "attn"), (0, 0, None), (4, 7, None), (4, 8, None)]
    assert jobs[(1, 2)] == [(1, 1, "ffn"), (1, 2, None), (1, 3, None)]
    assert jobs[(1, 0)] == [(3, 5, "ffn"), (3, 6, None)]


def test_load_jobs_keep_the_serial_per_chip_order():
    """A split unit in the tail stage of the head chip loads after the head stage's layers (serial order), not first."""
    plan = [((0, 0), ["L0", "L1.attn"]), ((0, 1), ["L1.ffn", "L2", "L3.attn"]), ((0, 0), ["L3.ffn", "L4"])]
    st = [StagePlan(coord=c, units=parse_units(us), chip=i) for i, (c, us) in enumerate(plan)]
    jobs = load_jobs(st, [unit_halves(p) for p in st])
    assert jobs[(0, 0)] == [(0, 1, "attn"), (0, 0, None), (2, 3, "ffn"), (2, 4, None)]
    assert jobs[(0, 1)] == [(1, 1, "ffn"), (1, 3, "attn"), (1, 2, None)]


def test_parallel_matches_serial_and_overlaps():
    ls, lp = FakeLoader(), FakeLoader()
    ms, t_serial = build(False, ls)
    mp, t_par = build(True, lp)
    # same decoder (layer, half, submesh) per (stage, layer)
    for a, b in zip(ms.stages, mp.stages):
        assert a.plan.coord == b.plan.coord and sorted(a.decoders) == sorted(b.decoders)
        for l in a.decoders:
            da, db = a.decoders[l], b.decoders[l]
            assert (da.layer, da.half, da.sub) == (db.layer, db.half, db.sub)
    # every chip's loads on one thread, in its job order; chips on different threads, overlapping in time
    st = stages()
    jobs = load_jobs(st, [unit_halves(p) for p in st])
    by_sub = {}
    for c in lp.calls:
        by_sub.setdefault(c["sub"], []).append(c)
    threads = set()
    for coord, js in jobs.items():
        calls = sorted(by_sub[f"sub{coord}"], key=lambda c: c["t0"])
        assert [(c["layer"], c["half"]) for c in calls] == [(l, h) for _, l, h in js]
        assert len({c["thread"] for c in calls}) == 1
        threads |= {c["thread"] for c in calls}
    assert len(threads) == len(jobs)
    n_loads = sum(len(j) for j in jobs.values())
    assert t_serial >= n_loads * DELAY * 0.9
    assert t_par < t_serial / 1.8, (t_par, t_serial)  # longest chip list = 4 loads of 9


def test_parallel_failure_reraised_after_all_chips(expect_error):
    lp = FakeLoader(fail_layer=4)  # chip (1,1)
    with expect_error(RuntimeError, r"parallel load failed on chip\(s\) \[\(1, 1\)\]") as ei:
        build(True, lp)
    assert isinstance(ei.value.__cause__, ValueError)
    loaded = {c["layer"] for c in lp.calls}
    assert loaded >= {0, 1, 2, 3, 6, 7, 8}  # the other chips finished their lists
