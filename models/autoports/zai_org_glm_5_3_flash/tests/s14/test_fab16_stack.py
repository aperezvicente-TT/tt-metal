# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fab16 in the served stack (14b, program 2026-10-27; 14's d9547643bf2 + dsa/shard_w16/twolink_16row_audit.md):
the sparse_dsa layer's MLA half gets its own 16-row phase under GLM53_FAB16 (default ON since k4d6) while TPMoE keeps the
stack's ts, and the first launch of a 16-row phase follows ONE full-mesh synchronize."""

from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa as TPD
from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

SERVED_TS = (1, 2, 3, 4, 32)  # stack_model: {1, 2, 3, 4, chunk} with the weight cache, chunk = 32


@pytest.mark.parametrize("ts", [SERVED_TS, (32,), (1, 2, 3, 4)])
def test_off_by_env(monkeypatch, ts):
    monkeypatch.setenv("GLM53_FAB16", "0")
    assert TPD.fab16_mla_ts(ts) == tuple(sorted(ts))


def test_on_by_default_since_k4d6(monkeypatch):
    monkeypatch.delenv("GLM53_FAB16", raising=False)
    assert TPD.fab16_mla_ts(SERVED_TS) == (1, 2, 3, 4, 16, 32)


def test_on_adds_16_to_the_served_mla_ts(monkeypatch):
    monkeypatch.setenv("GLM53_FAB16", "1")
    assert TPD.fab16_mla_ts(SERVED_TS) == (1, 2, 3, 4, 16, 32)
    assert TPD.fab16_mla_ts((32,)) == (16, 32)


def test_on_without_native_widths_is_unchanged(monkeypatch):
    monkeypatch.setenv("GLM53_FAB16", "1")
    assert TPD.fab16_mla_ts((1, 2, 3, 4)) == (1, 2, 3, 4)  # no t > 16 phase: no native widths to serve
    assert TPD.fab16_mla_ts((1, 16, 32)) == (1, 16, 32)  # idempotent


def _wire(monkeypatch, fab16="1", **kw):
    """TPDsaLayer's constructor with the builders stubbed -> (the ts each builder got, the layer)"""
    monkeypatch.setenv("GLM53_FAB16", fab16)
    seen = {}

    def fake_build(self, cfg, source, ocfg, ts, share, dsa, **kw_):
        seen["mla"] = tuple(ts)
        return None, SimpleNamespace(fabrics={})

    class FakeMoE:
        def __init__(self, mesh, args, moe, ts, fabrics=None):
            seen["moe"] = tuple(ts)

    monkeypatch.setattr(TPD.TPDsaLayer, "_build_mla", fake_build)
    monkeypatch.setattr(TPD, "TPMoE", FakeMoE)
    monkeypatch.setattr(TPD, "mhc_host_args", lambda *a, **k: {})
    lay = TPD.TPDsaLayer(object(), None, 3, None, None, moe=None, ts=SERVED_TS, **kw)
    return seen, lay


def test_deferred_placeholder_keeps_ts_attach_gets_mla_ts(monkeypatch):
    """GLM53_STACK_DSA_DEFER (the served image path): the DENSE placeholder is part of the DRAM image's recorded
    construction, so it keeps ts -- fab16's extra 16-row FabricPhase there shifted the image's uploads (program
    2026-10-27T18:52:00Z, ImageMismatch at upload #184) -- and attach_dsa (after the restore) rebuilds the DSA half with
    mla_ts. TPMoE gets ts either way."""
    seen, lay = _wire(monkeypatch, dsa=True, defer=True, shard=True)
    assert seen == {"mla": SERVED_TS, "moe": SERVED_TS}
    assert lay._attach[3] == (1, 2, 3, 4, 16, 32)


@pytest.mark.parametrize("dsa,shard", [(True, True), (True, False), (False, False)])
def test_direct_build_gets_mla_ts(monkeypatch, dsa, shard):
    """no deferral: the MLA built at construction IS the served one -> mla_ts (the 16-row phase); nothing to attach"""
    seen, lay = _wire(monkeypatch, dsa=dsa, defer=False, shard=shard)
    assert seen == {"mla": (1, 2, 3, 4, 16, 32), "moe": SERVED_TS}
    assert lay._attach is None


def test_deferred_placeholder_equals_fab16_off(monkeypatch):
    """the placeholder's construction under FAB16 = its construction with FAB16 off (the image's k4d5 sequence)"""
    on, _ = _wire(monkeypatch, dsa=True, defer=True, shard=True)
    off, _ = _wire(monkeypatch, fab16="0", dsa=True, defer=True, shard=True)
    assert on == off == {"mla": SERVED_TS, "moe": SERVED_TS}


def test_first_launch_sync_once_per_16_row_phase(monkeypatch):
    calls = []
    monkeypatch.setattr(TPM.ttnn, "synchronize_device", lambda mesh: calls.append(mesh), raising=False)
    mesh = object()
    f16, f32, f4 = SimpleNamespace(brows=16), SimpleNamespace(brows=32), SimpleNamespace(brows=4)
    for f in (f32, f4, f16, f16, f32, f16):
        TPM.first_launch_sync(mesh, f)
    assert calls == [mesh]  # only the 16-row phase's FIRST launch, once
    f16b = SimpleNamespace(brows=16)  # another 16-row phase object (another mesh / stack): its own first launch
    TPM.first_launch_sync(mesh, f16b)
    assert len(calls) == 2
