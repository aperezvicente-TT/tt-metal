"""Per-slot DSA pooled keys and the checkpoint export (program 2026-10-24T20:17:03Z; Stage 12 selfhit / dump /
fixtest, doc/ttnn/12-fast-start/selfhit). With GLM53_DSA_PAGED_POOLS=0 a checkpoint must carry the slot's pooled keys:
without the DSA export staging export_kda_state used to return a KDA-only blob, and a hit into another slot read zero
pooled keys at L3 (prefill logits 0.0625 off). Host only: StackBackend methods over fakes (no KDA layers, so the export
is the DSA part alone)."""

from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM


class FakeTensor:
    def __init__(self, addr):
        self.addr = addr

    def buffer_address(self):
        return self.addr


def _backend():
    b = SM.StackBackend.__new__(SM.StackBackend)
    b.chunk, b.tp, b.mesh, b.counters = 4, 1, None, {"slot_ops": 0}
    b.seq = [{"L": 3}]
    kv = SimpleNamespace(states=[[None]], extra={})
    return b, kv


def test_a_kda_only_checkpoint_of_a_per_slot_pooled_cache_is_refused(monkeypatch):
    b, kv = _backend()
    pools_t, ring_t = FakeTensor(0x1000), FakeTensor(0x2000)
    monkeypatch.setattr(b, "_dsa_tensors", lambda kv_: [("L3.pools", pools_t, 4, True), ("L3.ring", ring_t, 4, False)])
    monkeypatch.setattr(b, "export_dsa_state", lambda kv_, slot, n, pools=True: {"rows": (slot, n, pools)})
    with pytest.raises(RuntimeError, match="ensure_state_export"):
        b.export_kda_state(kv, 1, n_tokens=1024)
    assert b.export_kda_state(kv, 1, n_tokens=1024, pools=False) == []  # the pooled keys ride the LMCache pages
    monkeypatch.setattr(b, "_dsa_tensors", lambda kv_: [("L3.ring", ring_t, 4, False)])  # paged pools: ring only
    assert b.export_kda_state(kv, 1, n_tokens=1024) == []
    monkeypatch.setattr(b, "_dsa_tensors", lambda kv_: [("L3.pools", pools_t, 4, True), ("L3.ring", ring_t, 4, False)])
    kv.extra["dsa_stage"] = FakeTensor(0x3000)
    assert b.export_kda_state(kv, 1, n_tokens=1024) == [{"dsa": {"rows": (1, 1024, True)}}]


@pytest.mark.parametrize("paged,env,want", [(False, None, True), (True, None, False), (True, "256", True), (False, "0", True)])
def test_the_dsa_export_staging_comes_with_a_per_slot_pooled_cache(monkeypatch, paged, env, want):
    """allocate_kv_cache (via _dsa_stage_with_cache): per-slot pooled keys (PAGED_POOLS=0) always get the staging with
    the cache, before any capture; paged pools only when GLM53_DSA_EXPORT_STAGE_TILES is set (else on demand)"""
    b, kv = _backend()
    calls = []
    monkeypatch.setattr(SM, "dsa_paged_pools", lambda: paged)
    monkeypatch.setattr(b, "_ensure_dsa_stage", lambda kv_: calls.append(kv_))
    if env is None:
        monkeypatch.delenv("GLM53_DSA_EXPORT_STAGE_TILES", raising=False)
    else:
        monkeypatch.setenv("GLM53_DSA_EXPORT_STAGE_TILES", env)
    b._dsa_stage_with_cache(kv)
    assert (calls == [kv]) == want
