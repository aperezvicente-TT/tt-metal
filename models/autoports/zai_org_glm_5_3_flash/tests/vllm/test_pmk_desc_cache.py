# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: GLM53_PMK_DESC_CACHE (Stage 12, program 2026-10-26T23:19:10Z (A)). A PMK layer keeps each program's
assembled descriptors per key (the io tensors' addresses / shapes / dtypes / layouts, the scopes, fs_sync_addr, ci's
identity, the GLM53_ / PMK_ env). A hit returns the cached descriptors with this forward's io; an address, shape or env
change rebuilds. "verify" rebuilds on every hit and compares the per-rank ProgramDescriptor digests. Off by default.
Fake programs (no device); _pd_digest is checked on real ttnn descriptor objects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as S


class _T:
    """a device tensor stand-in: buffer_address / shape / dtype / layout"""

    def __init__(self, addr, shape=(1, 32, 64)):
        self.addr, self.shape, self.dtype, self.layout = addr, shape, "BFLOAT16", "TILE"

    def buffer_address(self):
        return self.addr


def _prog(addrs):
    body = SimpleNamespace(status="body", io=[_T(a) for a in addrs])
    engine = SimpleNamespace(phases=[SimpleNamespace(body=body)], programs=[("P1", "P2")])
    return SimpleNamespace(engine=engine, pl=SimpleNamespace(scratch=None), layer=SimpleNamespace(fs_sync_addr=64))


@pytest.fixture
def builds(monkeypatch):
    """_asm_descs replaced by a counting fake: descs = one object per call; digests = the io addresses (+ a bias)"""
    calls = {"n": 0, "bias": 0}

    def fake(a, x, ci, io=None, digests=None):
        calls["n"] += 1
        io = io if io is not None else S._asm_io(a, x)
        if digests is not None:
            digests.append(tuple(str(t.buffer_address() + calls["bias"]) for t in io))
        descs = [object()]
        a.layer.descs, a.layer.launch_census, a.layer.desc = descs, {"P1+P2": {}}, descs[0]
        return io, descs

    monkeypatch.setattr(S, "_asm_descs", fake)
    for k in list(S._DESC_STATS):
        monkeypatch.setitem(S._DESC_STATS, k, 0)
    return calls


def test_off_by_default_every_forward_builds(builds, monkeypatch):
    monkeypatch.delenv("GLM53_PMK_DESC_CACHE", raising=False)
    owner, a, x, ci = SimpleNamespace(L=3), _prog([1024, 2048]), _T(4096), object()
    for _ in range(3):
        S._asm_descs_cached(owner, "a", a, x, ci)
    assert builds["n"] == 3 and "_desc_cache" not in owner.__dict__


def test_a_hit_reuses_the_descriptors_with_this_forwards_io(builds, monkeypatch):
    monkeypatch.setenv("GLM53_PMK_DESC_CACHE", "1")
    owner, a, x, ci = SimpleNamespace(L=3), _prog([1024, 2048]), _T(4096), object()
    io1, d1 = S._asm_descs_cached(owner, "a", a, x, ci)
    io2, d2 = S._asm_descs_cached(owner, "a", a, x, ci)
    assert builds["n"] == 1 and d2 is d1 and a.layer.desc is d1[0]
    assert io2 is not io1 and [t.addr for t in io2] == [t.addr for t in io1]  # the fresh io list (same addresses)
    assert S._DESC_STATS["builds"] == 1 and S._DESC_STATS["hits"] == 1


def test_an_address_shape_ci_or_env_change_rebuilds(builds, monkeypatch):
    monkeypatch.setenv("GLM53_PMK_DESC_CACHE", "1")
    owner, a, x, ci = SimpleNamespace(L=3), _prog([1024, 2048]), _T(4096), object()
    S._asm_descs_cached(owner, "a", a, x, ci)
    a.engine.phases[0].body.io[1] = _T(8192)  # a re-allocated body tensor at another address
    S._asm_descs_cached(owner, "a", a, x, ci)
    a.engine.phases[0].body.io[1] = _T(8192, shape=(1, 64, 64))
    S._asm_descs_cached(owner, "a", a, x, ci)
    S._asm_descs_cached(owner, "a", a, x, object())  # another chunk-input set
    monkeypatch.setenv("PMK_P12_RECTS", "4")  # a construction knob
    S._asm_descs_cached(owner, "a", a, x, ci)
    assert builds["n"] == 5
    S._asm_descs_cached(owner, "g", a, x, ci)  # program G of the same layer: its own entry
    assert builds["n"] == 6


def test_verify_rebuilds_on_every_hit_and_counts_a_mismatch(builds, monkeypatch, capsys):
    monkeypatch.setenv("GLM53_PMK_DESC_CACHE", "verify")
    owner, a, x, ci = SimpleNamespace(L=7), _prog([1024]), _T(4096), object()
    _io, d1 = S._asm_descs_cached(owner, "a", a, x, ci)
    _io, d2 = S._asm_descs_cached(owner, "a", a, x, ci)
    assert builds["n"] == 2 and d2 is d1 and S._DESC_STATS["verified"] == 1 and S._DESC_STATS["mismatches"] == 0
    builds["bias"] = 1  # the rebuild now produces other descriptors
    S._asm_descs_cached(owner, "a", a, x, ci)
    assert S._DESC_STATS["mismatches"] == 1 and "DESC_CACHE MISMATCH L7 a" in capsys.readouterr().out


def test_a_bad_mode_is_refused(monkeypatch):
    monkeypatch.setenv("GLM53_PMK_DESC_CACHE", "yes")
    with pytest.raises(ValueError, match="GLM53_PMK_DESC_CACHE"):
        S._desc_cache_mode()


def test_the_digest_sees_runtime_args_cbs_and_semaphores():
    ttnn = pytest.importorskip("ttnn")

    def pd(rt=7, size=4096, sem=0, writer=False, define=None):
        k = ttnn.KernelDescriptor()
        k.kernel_source = "models/x/reader.cpp"
        k.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
        k.compile_time_args = [1, 2]
        args = ttnn.RuntimeArgs()  # (mk_dense._rt_args: per core x, y)
        args[0][0] = [rt, 3]
        args[1][0] = [5, 6]
        k.runtime_args = args
        k.config = ttnn.WriterConfigDescriptor() if writer else ttnn.ReaderConfigDescriptor()
        if define:
            k.defines = [define]
        cb = ttnn.CBDescriptor()
        cb.total_size = size
        cb.core_ranges = k.core_ranges
        s_ = ttnn.SemaphoreDescriptor()
        s_.core_ranges, s_.initial_value = k.core_ranges, sem
        p = ttnn.ProgramDescriptor()
        p.kernels, p.cbs, p.semaphores = [k], [cb], [s_]
        return p

    base = S._pd_digest(pd())
    assert S._pd_digest(pd()) == base  # deterministic: no object addresses in the digest
    assert S._pd_digest(pd(rt=8)) != base and S._pd_digest(pd(size=8192)) != base and S._pd_digest(pd(sem=1)) != base
    assert S._pd_digest(pd(writer=True)) != base and S._pd_digest(pd(define=("PMK_X", "1"))) != base


def test_fabric_connect_core_test_matches_the_set_expansion():
    """lever (C): fabric_connect's per-core semaphore ids via CoreRangeSet.contains equal the old _sem_cores expansion,
    on real ttnn semaphore descriptors with multi-core rectangles"""
    ttnn = pytest.importorskip("ttnn")
    from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS

    def rect(x0, y0, x1, y1):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])

    sems = [ttnn.SemaphoreDescriptor(id=i % 16, core_ranges=rect(i % 7, i % 5, i % 7 + (i % 3), i % 5 + (i % 2)),
                                     initial_value=0) for i in range(300)]  # fmt: skip
    for x in range(10):
        for y in range(7):
            old = {int(s_.id) for s_ in sems if (x, y) in FS._sem_cores(s_)}
            new = {int(s_.id) for s_ in sems if s_.core_ranges.contains(ttnn.CoreCoord(x, y))}
            assert new == old, (x, y)


def test_the_miss_log_names_what_differs(capsys):
    """GLM53_PMK_DESC_CACHE_MISSLOG: a miss against the same assembly's cached keys names the differing parts"""
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS_

    owner = type("O", (), {"L": 3})()
    io = ((100, (1, 2), "bf16", "TILE"), (200, (4,), "bf16", "TILE"))
    k0 = ("a", (7, 11, io, ("s",), 5, (("GLM53_X", "1"),)))
    cache = {k0: None}
    io2 = ((100, (1, 2), "bf16", "TILE"), (300, (4,), "bf16", "TILE"))
    PS_._desc_miss_log(owner, "a", ("a", (7, 11, io2, ("s",), 5, (("GLM53_X", "2"),))), cache)
    PS_._desc_miss_log(owner, "a", ("a", (8, 11, io, ("s",), 5, ())), cache)
    out = capsys.readouterr().out
    assert "DESC_CACHE MISS L3 a: vs nearest of 1: io[1]:addr env:GLM53_X=1->2" in out
    assert "DESC_CACHE MISS L3 a: first of its assembly" in out


def test_the_key_ignores_the_leftover_decode_knobs(monkeypatch):
    """the width-8/16 decode captures' leftover knob env does not change the key (their descriptors were digest-equal);
    any other GLM53_ key still does; GLM53_PMK_DESC_KEY_IGNORE="" keeps every key"""
    from types import SimpleNamespace as NS

    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS_

    a = NS(engine=NS(programs=[]), layer=NS(fs_sync_addr=None))
    monkeypatch.delenv("GLM53_PMK_DESC_KEY_IGNORE", raising=False)
    monkeypatch.setenv("GLM53_MK_S1_DRAM", "1")
    monkeypatch.delenv("GLM53_REC_UGROUPS", raising=False)
    k0 = PS_._desc_key(a, [], None)
    monkeypatch.delenv("GLM53_MK_S1_DRAM")
    monkeypatch.setenv("GLM53_REC_UGROUPS", "8")
    monkeypatch.setenv("GLM53_REC_V2_UG", "1")
    assert PS_._desc_key(a, [], None) == k0
    monkeypatch.setenv("GLM53_PMK_SOMETHING", "2")
    assert PS_._desc_key(a, [], None) != k0
    monkeypatch.delenv("GLM53_PMK_SOMETHING")
    monkeypatch.setenv("GLM53_PMK_DESC_KEY_IGNORE", "")
    assert PS_._desc_key(a, [], None) != k0


def test_no_prefill_builder_reads_an_ignored_key():
    """14's SHOULD on dc81ce38c51: served mode never rebuilds a hit, so a pmk_* / prefill builder that reads a key the
    cache key ignores would get stale descriptors silently -- this fails at the commit that adds such a reader (put
    the key back into the cache key, i.e. out of _DESC_KEY_IGNORE_DEFAULT, in that commit)"""
    import pathlib

    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS_

    tt = pathlib.Path(PS_.__file__).parent
    files = sorted(set(tt.glob("pmk_*.py")) | set(tt.glob("prefill*.py")))
    assert any(f.name == "pmk_stack.py" for f in files) and any(f.name == "prefill_tp.py" for f in files)
    keys = [k for k in PS_._DESC_KEY_IGNORE_DEFAULT.split(",") if k]
    readers = []
    for f in files:
        text = f.read_text()
        if f.name == "pmk_stack.py":  # its own ignore list names them
            text = text.replace(f'_DESC_KEY_IGNORE_DEFAULT = "{PS_._DESC_KEY_IGNORE_DEFAULT}"', "")
        for k in keys:
            for line in text.splitlines():
                code = line.split("#", 1)[0]
                if k in code:
                    readers.append(f"{f.name}: {line.strip()[:100]}")
    assert readers == [], readers
