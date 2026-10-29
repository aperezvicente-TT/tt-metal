"""TPPrefill start-time attribution (GLM53_PF_COST, Stage 12): per (phase, key) device-synchronized seconds and the
program-cache entries added, one summary line per phase; GLM53_PF_COST=0 records nothing."""

from types import SimpleNamespace

import pytest

ttnn = pytest.importorskip("ttnn")
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT  # noqa: E402


class FakeMesh:
    def __init__(self):
        self.programs = 10

    def num_program_cache_entries(self):
        return self.programs


def test_cost_rows_and_report(monkeypatch, capsys):
    monkeypatch.setattr(PT.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.delenv("GLM53_PF_COST", raising=False)
    tp = SimpleNamespace(mesh=FakeMesh(), cost=[])
    with PT.TPPrefill._cost(tp, "compile", (2048, 0)):
        tp.mesh.programs += 7  # the key's eager forward compiled 7 programs
    with PT.TPPrefill._cost(tp, "compile", ("packed", 2048, 3)):
        pass
    with PT.TPPrefill._cost(tp, "capture", (2048, 0)):
        pass
    assert [(r["phase"], r["key"], r["programs"]) for r in tp.cost] == [
        ("compile", "(2048, 0)", 7), ("compile", "('packed', 2048, 3)", 0), ("capture", "(2048, 0)", 0)]
    assert all(r["s"] >= 0 for r in tp.cost)
    PT.TPPrefill._cost_report(tp, "compile")
    out = capsys.readouterr().out
    assert out.startswith("TPPrefill compile: 2 keys") and "programs +7" in out and "(2048, 0)" in out


def test_cost_off_records_nothing(monkeypatch):
    monkeypatch.setenv("GLM53_PF_COST", "0")
    tp = SimpleNamespace(mesh=FakeMesh(), cost=[])
    with PT.TPPrefill._cost(tp, "warm", (512, 0)):
        pass
    assert tp.cost == []


def test_a_failing_key_records_nothing_and_raises(monkeypatch, capsys):
    """the failing key re-raises its own error and prints which keys of the phase fitted before it (n29h7s2 1616 /
    1623: the trace region overflowed inside the capture loop and the log could not say after how many keys)"""
    monkeypatch.setattr(PT.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.delenv("GLM53_PF_COST", raising=False)
    tp = SimpleNamespace(mesh=FakeMesh(), cost=[])
    with PT.TPPrefill._cost(tp, "capture", (512, 0)):
        pass
    with pytest.raises(RuntimeError, match="capture failed"):
        with PT.TPPrefill._cost(tp, "capture", (256, 0)):
            raise RuntimeError("capture failed")
    assert [r["key"] for r in tp.cost] == ["(512, 0)"]
    out = capsys.readouterr().out
    assert out.startswith("TPPrefill capture RAISED at (256, 0) after: 1 keys") and "(512, 0)" in out
