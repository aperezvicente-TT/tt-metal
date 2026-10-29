# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17: spec_mtp.verify_allocator restores the module's `empty` (ttnn.empty) and the pools' allocators -- after a
stack build under the patch (AllocPool.__exit__ sets mod.empty = pool.orig: the 10:41Z leak), after an exception, when
nested, and between calls (reset). Host only (a fake module and an AllocPool-shaped fake; no device, no allocation).
usage: no-tt-devices python -I -m pytest -q tests/s17/test_verify_allocator.py   (or run the file directly)"""
import types

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt.spec_mtp import verify_allocator


class FakePool:
    """AllocPool's install / uninstall: __enter__ sets mod.empty = self, __exit__ sets mod.empty = self.orig"""

    def __init__(self, mod, child=None):
        self.mod, self.orig, self._child = mod, mod.empty, child

    def __call__(self, *a, **kw):
        return self.orig(*a, **kw)

    def __enter__(self):
        self.mod.empty = self
        return self

    def __exit__(self, *exc):
        self.mod.empty = self.orig
        return False


def real_empty(*a, **kw):
    return ("real", a, kw)


def make():
    mod = types.SimpleNamespace(empty=real_empty)
    child = FakePool(mod)
    pool = FakePool(mod, child)
    return mod, pool, child


def test_stack_build_does_not_leak():
    mod, pool, child = make()
    with verify_allocator(pool, 4, True, mod=mod) as va:
        assert pool.orig is not real_empty and child.orig is not real_empty  # patched
        with pool:  # a stack build: on exit the patched allocator becomes mod.empty (the leak)
            pass
        assert mod.empty is pool.orig and mod.empty is not real_empty
        va.reset()
        assert mod.empty is real_empty  # the MkMla calls between stack builds see the real allocator
        with pool:
            pass
    assert mod.empty is real_empty and pool.orig is real_empty and child.orig is real_empty


def test_exception_restores():
    mod, pool, child = make()
    with pytest.raises(RuntimeError):
        with verify_allocator(pool, 2, True, mod=mod):
            with pool:
                raise RuntimeError("build failed")
    assert mod.empty is real_empty and pool.orig is real_empty and child.orig is real_empty


def test_reentrant():
    mod, pool, child = make()
    with verify_allocator(pool, 4, True, mod=mod):
        outer_orig = pool.orig
        with verify_allocator(pool, 4, True, mod=mod):
            with pool:
                pass
        assert pool.orig is outer_orig and mod.empty is real_empty  # the inner restored what it saw
        with pool:
            pass
    assert mod.empty is real_empty and pool.orig is real_empty and child.orig is real_empty


def test_disabled_touches_nothing_but_restores():
    mod, pool, child = make()
    with verify_allocator(pool, 4, False, mod=mod):
        assert pool.orig is real_empty and child.orig is real_empty
        mod.empty = "stray"
    assert mod.empty is real_empty


def test_patched_allocator_routes_shapes():
    """the patch sends the verify's stacked post / comb shapes (fp32, L1 interleaved) to DRAM and nothing else"""
    import ttnn

    seen = []
    mod = types.SimpleNamespace(empty=lambda shape, *a, **kw: seen.append((tuple(shape), kw.get("memory_config"))))
    pool = FakePool(mod)
    with verify_allocator(pool, 3, True, mod=mod):
        pool.orig((3, 1, 4, 1), dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        pool.orig((3, 1, 4, 1), dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        pool.orig((2, 1, 4, 1), dtype=ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert seen[0][1].buffer_type == ttnn.BufferType.DRAM
    assert seen[1][1].buffer_type == ttnn.BufferType.L1 and seen[2][1].buffer_type == ttnn.BufferType.L1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
