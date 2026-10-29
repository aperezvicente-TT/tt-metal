# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python unit tests of the prefix-cache bookkeeping (tt/prefix_cache.py): keys, LRU, page pins, resume positions,
trailing (generation) snapshots, the bytes model and the TTFT ladder. No torch, no vLLM, no device.

    python -m pytest tests/vllm/test_prefix_store.py -q --confcutdir tests/vllm   (any python >= 3.10)
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("prefix_cache", os.path.join(_HERE, "..", "..", "tt", "prefix_cache.py"))
PC = importlib.util.module_from_spec(_SPEC)
sys.modules["prefix_cache"] = PC  # dataclasses resolve the module's namespace through sys.modules
_SPEC.loader.exec_module(PC)

B = PC.BOUNDARY  # 128


def ids(seed: int, n: int) -> list[int]:
    x, out = seed * 7919 + 17, []
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2**31
        out.append(x % 150000)
    return out


def store(entries=4, host=0):
    return PC.PrefixStore(max_batch=8, device_entries=entries, host_bytes=host, sig="sig-A", entry_bytes=1000)


# ------------------------------------------------------------------------------------------------------- keys
def test_keys_are_chained_boundary_aligned_and_salted(expect_error):
    p = ids(1, 4 * B + 5)
    st = store()
    hs = st.hashes(p)
    assert len(hs) == 4  # complete 128-token blocks only
    assert st.key(p, 2 * B) == hs[1]
    assert st.key(p[: 2 * B] + ids(2, 3 * B), 2 * B) == hs[1]  # the key depends on the prefix only
    assert st.key(p[: 2 * B], 2 * B) == hs[1]
    q = list(p)
    q[B + 3] += 1
    assert st.key(q, 2 * B) != hs[1] and st.key(q, B) == hs[0]  # a changed token changes every later key
    other = PC.PrefixStore(max_batch=8, device_entries=4, sig="sig-B")
    assert other.key(p, B) != hs[0]  # another image / code / knob set: no cross hits
    with expect_error(ValueError, "positive multiple of 128"):
        st.key(p, B + 1)
    assert st.boundaries(4 * B + 5) == [B, 2 * B, 3 * B, 4 * B]
    assert st.boundaries(4 * B) == [B, 2 * B, 3 * B]  # a prompt of exactly k blocks keeps its last token to prefill
    assert st.last_boundary(4 * B) == 3 * B and st.last_boundary(4 * B + 1) == 4 * B and st.last_boundary(5) == 0


# ---------------------------------------------------------------------------------------- register / attach
def test_register_attach_resume_positions(expect_error):
    st = store(entries=4)
    p = ids(3, 6 * B + 40)
    pages = list(range(100, 100 + 12))
    e0 = st.reserve_device()
    st.register(st.key(p, 2 * B), n_tokens=2 * B, pages=pages, kind="P", device_entry=e0)
    e1 = st.reserve_device()
    st.register(st.key(p, 4 * B), n_tokens=4 * B, pages=pages, kind="P", device_entry=e1)
    # longest cached boundary of the same prompt, capped by vLLM's page hit
    assert st.longest_cached(p)[0] == 4 * B
    assert st.longest_cached(p, 4 * B)[0] == 4 * B
    assert st.longest_cached(p, 4 * B - 64)[0] == 2 * B  # a shorter page hit: the lower boundary
    assert st.longest_cached(p, 3 * B)[0] == 2 * B
    assert st.longest_cached(p, B)[0] == 0
    # a partial match: a different prompt sharing the first 3 blocks resumes at 2 * B
    q = p[: 3 * B] + ids(9, 2 * B)
    pos, ent = st.longest_cached(q)
    assert pos == 2 * B and ent.device_entry == e0
    # a prompt of exactly 4 blocks cannot resume at 4 * B (its last token must be prefilled): 2 * B
    assert st.longest_cached(p[: 4 * B])[0] == 2 * B
    # attach: readers and pins
    pos, ent = st.attach(ent.key, slot=3)
    assert (
        pos == 2 * B and ent.readers == 1 and st.pinned(100) and st.pinned(107) and not st.pinned(108)
    )  # 4 * B = 8 pages
    assert st.stats["tokens_saved"] == 2 * B
    st.release_slot(3)
    assert ent.readers == 0
    with expect_error(KeyError, "nope"):
        st.attach("nope", slot=1)


def test_lru_eviction_respects_readers_and_frees_pins():
    st = store(entries=2)
    p = ids(4, 8 * B)
    pages = list(range(8))
    a = st.reserve_device()
    st.register(st.key(p, B), n_tokens=B, pages=pages, kind="P", device_entry=a)
    b = st.reserve_device()
    st.register(st.key(p, 2 * B), n_tokens=2 * B, pages=pages, kind="P", device_entry=b)
    assert st.reserve_device() == a  # LRU: the first entry goes
    assert not st.has(st.key(p, B)) and st.pinned(0) and st.pinned(3) and not st.pinned(4)
    st.register(st.key(p, 3 * B), n_tokens=3 * B, pages=pages, kind="P", device_entry=a)
    # a reader protects its entry: the other one is evicted instead
    st.attach(st.key(p, 2 * B), slot=0)
    c = st.reserve_device()
    assert c == a and st.has(st.key(p, 2 * B)) and not st.has(st.key(p, 3 * B))
    st.unreserve(c)
    st.attach(st.key(p, 2 * B), slot=1)  # two readers
    st.register(st.key(p, 4 * B), n_tokens=4 * B, pages=pages, kind="P", device_entry=st.reserve_device())
    st.attach(st.key(p, 4 * B), slot=2)
    assert st.reserve_device() is None  # every device entry has readers: nothing to evict
    st.release_slot(0)
    st.release_slot(1)
    assert st.reserve_device() is not None
    assert st.stats["evicted_device"] == 3


def test_host_tier_spill_promote_and_budget(expect_error):
    st = store(entries=1, host=2500)
    p = ids(5, 4 * B)
    e = st.reserve_device()
    st.register(st.key(p, B), n_tokens=B, pages=[0, 1], kind="P", device_entry=e)
    st.spill(st.key(p, B), host_blob={"states": "blob"}, bytes=1000)
    assert st.entries[st.key(p, B)].tier == "host" and st.host_used == 1000 and st.free_device == [e]
    assert st.longest_cached(p)[0] == B  # a host entry still answers lookups
    st.register(st.key(p, 2 * B), n_tokens=2 * B, pages=[0, 1, 2, 3], kind="P", host_blob="b2", bytes=1000)
    st.register(st.key(p, 3 * B), n_tokens=3 * B, pages=list(range(6)), kind="P", host_blob="b3", bytes=1000)
    assert st.host_used == 2000 and not st.has(st.key(p, B))  # budget 2500: the LRU host entry went
    st.promote(st.key(p, 3 * B), st.reserve_device())
    assert st.entries[st.key(p, 3 * B)].tier == "device" and st.host_used == 1000
    with expect_error(ValueError, "no host tier"):
        PC.PrefixStore(max_batch=8, device_entries=0, host_bytes=0).register(
            "k", n_tokens=B, pages=[0, 1], kind="P", host_blob="x"
        )  # no host tier


def test_trailing_generation_snapshots_commit_on_release(expect_error):
    st = store(entries=3)
    prompt = ids(6, 500)
    # the slot crossed 512 and 640 while decoding: one entry, re-pointed each time
    e = st.trailing_entry(slot=2, position=512)
    assert e is not None and st.trailing_entry(2, 640) == e and st.trailing_position(2) == 640
    with expect_error(ValueError, "multiple of 128"):
        st.trailing_entry(2, 650)
    full = prompt + ids(7, 160)  # 660 tokens at release
    ent = st.commit_trailing(2, full, pages=list(range(11)))
    assert ent is not None and ent.kind == "G" and ent.n_tokens == 640 and ent.device_entry == e
    # the next turn: prompt + answer + a new question resumes at 640
    turn2 = full + ids(8, 50)
    pos, got = st.longest_cached(turn2)
    assert pos == 640 and got is ent
    # a preempted request (no ids) drops its trailing snapshot
    e2 = st.trailing_entry(slot=4, position=128)
    st.release_slot(4)
    assert e2 in st.free_device and 4 not in st.trailing
    # a slot move carries the trailing snapshot
    e3 = st.trailing_entry(slot=5, position=256)
    st.move_slot(5, 1)
    assert st.trailing[1] == (e3, 256) and 5 not in st.trailing


def test_invalidate_on_signature_change():
    st = store(entries=2)
    p = ids(1, 2 * B)
    st.register(st.key(p, B), n_tokens=B, pages=[0, 1], kind="P", device_entry=st.reserve_device())
    assert st.invalidate("sig-B") == 1 and not st.entries and sorted(st.free_device) == [0, 1] and st.sig == "sig-B"
    assert st.longest_cached(p)[0] == 0


# ---------------------------------------------------------------------------------------- bytes and the ladder
def test_bytes_model_matches_the_stack_shapes():
    bm = PC.SnapshotBytes()
    assert bm.s_bytes == 4 * 2**20 and bm.c_bytes == 147456
    assert bm.kda_total == 34 * (4 * 2**20 + 147456) == 147619840
    assert bm.kda_per_chip == 18452480
    assert bm.dsa_per_chip(2048, sparse=False) == 0
    assert bm.dsa_per_chip(32768, sparse=True) == 11 * (256 * 8192 + 16384)
    assert bm.pages_per_chip(2048) == 32 * 11 * 65536


def test_ttft_ladder_rows(expect_error):
    rows = PC.ttft_ladder()
    by = {r["context"]: r for r in rows}
    assert by[2048]["cold_ms"] == pytest.approx(441 + 22)
    assert by[2048]["cached_prefix"] == 1920 and by[2048]["cached_prefix_suffix_ms"] == pytest.approx(202 + 22 + 1.5)
    assert by[32768]["cold_ms"] == pytest.approx(16 * 441 + 16 * 22)
    assert by[128]["cached_prefix"] == 0 and by[128]["cached_prefix_suffix_ms"] == by[128]["cold_ms"]
    assert by[32768]["resume_suffix"] == 96 and by[32768]["full_turn_resume_ms"] == pytest.approx(202 + 22 + 1.5)
    c = PC.PrefillCosts()
    assert c.ttft_ms(2048 + 200, cached=2048) == pytest.approx(259 + 22 + 1.5)  # a 200-token suffix: the 512 bucket
    with expect_error(ValueError, "last token"):
        c.ttft_ms(128, cached=128)


# ----------------------------------------------------------------------- spill-on-evict (Stage 12 host KV tier)
def _fill(st, n, seed=50):
    keys = []
    for k in range(n):
        p = ids(seed + k, B + 3)
        key = st.key(p, B)
        e = st.reserve_device()
        assert e is not None
        st.register(key, n_tokens=B, pages=[100 + 2 * k, 101 + 2 * k], kind="P", device_entry=e)
        keys.append(key)
    return keys


def test_reserve_device_spills_lru_to_the_host_tier():
    st = store(entries=2, host=2500)
    exported = []
    st.spill_fn = lambda e: (exported.append(e.key) or (f"blob-{e.key[:6]}", 1000))
    k0, k1 = _fill(st, 2)
    e = st.reserve_device()  # full: the LRU device entry (k0) moves to the host tier
    assert e is not None and exported == [k0]
    assert st.entries[k0].tier == "host" and st.entries[k0].host_blob == f"blob-{k0[:6]}"
    assert st.host_used == 1000 and st.stats["spilled"] == 1 and st.stats["evicted_device"] == 0
    assert st.entries[k1].tier == "device"
    pos, ent = st.longest_cached(ids(50, B + 3), B)  # still a hit, now from the host tier
    assert pos == B and ent.tier == "host"


def test_spill_refused_or_failed_drops_the_entry():
    st = store(entries=1, host=2500)
    st.spill_fn = lambda e: None  # e.g. the sparse regime (no DSA host export yet)
    (k0,) = _fill(st, 1)
    assert st.reserve_device() is not None
    assert k0 not in st.entries and st.stats["evicted_device"] == 1 and st.stats["spill_refused"] == 1
    st2 = store(entries=1, host=500)
    st2.spill_fn = lambda e: ("blob", 1000)  # larger than the whole host tier
    (k1,) = _fill(st2, 1, seed=70)
    assert st2.reserve_device() is not None
    assert k1 not in st2.entries and st2.stats["spill_failed"] == 1 and st2.host_used == 0
    st3 = store(entries=1, host=2500)

    def boom(e):
        raise RuntimeError("export failed")

    st3.spill_fn = boom
    (k2,) = _fill(st3, 1, seed=90)
    assert st3.reserve_device() is not None and k2 not in st3.entries and st3.stats["spill_failed"] == 1


def test_host_tier_lru_makes_room_for_new_spills():
    st = store(entries=1, host=1500)  # room for ONE host entry of 1000 bytes
    st.spill_fn = lambda e: ("blob", 1000)
    (k0,) = _fill(st, 1)
    (k1,) = _fill(st, 1, seed=60)  # reserving k1's entry spills k0
    assert st.entries[k0].tier == "host"
    _fill(st, 1, seed=61)  # spills k1; the host LRU (k0) is evicted to make room
    assert k0 not in st.entries and st.entries[k1].tier == "host"
    assert st.stats["spilled"] == 2 and st.stats["evicted_host"] == 1 and st.host_used == 1000


def test_no_spill_without_a_host_tier_or_hook():
    st = store(entries=1, host=0)
    st.spill_fn = lambda e: ("blob", 10)
    (k0,) = _fill(st, 1)
    assert st.reserve_device() is not None and k0 not in st.entries and st.stats["spilled"] == 0
    st2 = store(entries=1, host=5000)  # host tier but no hook: drop as before
    (k1,) = _fill(st2, 1, seed=80)
    assert st2.reserve_device() is not None and k1 not in st2.entries
