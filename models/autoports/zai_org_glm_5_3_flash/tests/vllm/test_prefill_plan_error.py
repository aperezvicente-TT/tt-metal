"""Stage 12 (program 2026-10-26T17:09:24Z, a k3 MUST): Glm53Generator.prefill_plan_error -- the HOST-ONLY planning of
a prefill (plan_prefill -> the backend's slice rule -> TPPrefill.plan per chunk), no device work -- returns the plan
error a step would raise ('chunk [s, s+c) past the cache'), so the plugin refuses the request instead of latching a
device fault."""

from types import SimpleNamespace

from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

MAX_SEQ = 66048


class _Tpp:
    """TPPrefill.plan's cache-end rule in miniature: a chunk's 2,048 bucket must fit the cache row"""

    def __init__(self):
        self.calls = []

    def plan(self, s, n):
        self.calls.append((s, n))
        if s + 2048 > MAX_SEQ:
            raise ValueError(f"chunk [{s}, {s + 2048}) past the cache ({MAX_SEQ} tokens)")
        return 2048, 65536


def _gen(slice_rule=True):
    tpp = _Tpp()
    be = SimpleNamespace(prefill_chunk_tokens=2048, prefill_align=128, continuation_align=128, supported_context=1 << 20,
                         tpp=tpp, prefill_slice_plan=(lambda s, n, f: ([n], "")) if slice_rule else None)  # fmt: skip
    return SimpleNamespace(backend=be), tpp


def test_a_plannable_prefill_returns_none_and_plans_every_chunk():
    g, tpp = _gen()
    assert Glm53Generator.prefill_plan_error(g, 0, 5000) is None
    assert tpp.calls == [(0, 2048), (2048, 2048), (4096, 904)]


def test_a_continuation_past_the_cache_returns_the_error():
    g, _tpp = _gen()
    err = Glm53Generator.prefill_plan_error(g, 64256, 1000)  # its chunk at 64,256 replays the 2,048 bucket past 66,048
    assert err is not None and "past the cache" in err


def test_past_the_supported_context_is_an_error_too():
    g, _tpp = _gen()
    g.backend.supported_context = 4096
    assert "supported context" in Glm53Generator.prefill_plan_error(g, 0, 5000)


def test_no_slice_rule_plans_the_raw_chunks():
    g, tpp = _gen(slice_rule=False)
    assert Glm53Generator.prefill_plan_error(g, 128, 2100) is None
    assert tpp.calls == [(128, 2048), (2176, 52)]
