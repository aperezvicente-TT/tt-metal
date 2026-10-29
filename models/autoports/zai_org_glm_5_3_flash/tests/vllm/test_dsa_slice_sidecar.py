# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the DSA slice sidecar (GLM53_DSA_SLICE_CACHE, tt/tp_dsa.py) never serves an image mock's zero read-backs.
2026-10-23: a mock wrote all-zero q_b / o_proj / W_uk / W_uv sidecars and every later device run loaded them (strict
selftest 0.62 / 0.61). A mock never writes the sidecar; a device-written sidecar records its writer; loading one that
holds an all-zero tensor or names another writer aborts (program 2026-10-23T01:09:36Z)."""

from __future__ import annotations

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa as D
from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

KEYS = TPM.MeshHeadShard.FULL_KEYS


def full(fill):
    return {k: torch.full((4, 8), fill, dtype=torch.float32) for k in KEYS}


def test_mock_never_writes_the_sidecar(monkeypatch):
    monkeypatch.setenv("GLM53_MOCK", "1")
    assert not D.TPDsaLayer.sidecar_writable()
    monkeypatch.delenv("GLM53_MOCK")
    assert D.TPDsaLayer.sidecar_writable()


def test_device_sidecar_round_trips_with_its_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_RUN", "resident_test")
    p = str(tmp_path / "dsa_slices-x-L11-tp8.pt")
    D.TPDsaLayer._save_slices(p, full(0.5))
    d = D.TPDsaLayer._load_slices(p)
    assert d is not None and all(torch.equal(d[k], full(0.5)[k]) for k in KEYS)
    assert d["_meta"]["writer"] == "device" and d["_meta"]["run"] == "resident_test"


def test_pre_provenance_sidecar_loads(tmp_path):
    p = str(tmp_path / "dsa_slices-x-L11-tp8.pt")
    torch.save(full(0.5), p)  # written before sidecars carried _meta
    assert D.TPDsaLayer._load_slices(p) is not None


def test_all_zero_sidecar_aborts(tmp_path):
    p = str(tmp_path / "dsa_slices-x-L11-tp8.pt")
    torch.save(full(0.0), p)
    with pytest.raises(RuntimeError, match="all-zero"):
        D.TPDsaLayer._load_slices(p)
    one = full(0.5)
    one[KEYS[-1]] = torch.zeros(4, 8)  # one zero tensor is enough
    torch.save(one, p)
    with pytest.raises(RuntimeError, match="all-zero"):
        D.TPDsaLayer._load_slices(p)


def test_non_device_writer_aborts(tmp_path):
    p = str(tmp_path / "dsa_slices-x-L11-tp8.pt")
    torch.save({**full(0.5), "_meta": {"writer": "mock"}}, p)
    with pytest.raises(RuntimeError, match="not device-written"):
        D.TPDsaLayer._load_slices(p)


# ---- plausibility (program 2026-10-27T10:31:44Z: the Galaxy's sidecars, written from read-backs under a broken device
# state, held a non-finite / ~3e38 o_proj that every later run uploaded): refused + moved aside + rebuilt on load; never
# written from implausible read-backs
def test_non_finite_sidecar_is_refused_and_moved_aside(tmp_path, capsys):
    p = tmp_path / "dsa_slices-x-L03-tp8.pt"
    bad = full(0.5)
    bad["wo"][1, 3] = float("nan")
    torch.save(bad, str(p))
    assert D.TPDsaLayer._load_slices(str(p)) is None  # -> attach_dsa rebuilds from the device read-backs
    assert not p.exists() and len(list(tmp_path.glob("dsa_slices-x-L03-tp8.pt.implausible_*"))) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_huge_finite_sidecar_is_refused(tmp_path):
    p = str(tmp_path / "dsa_slices-x-L03-tp8.pt")
    bad = full(0.5)
    bad["qb"][0, 0] = 3e38  # a bfp8 block exponent saturated by a neighbouring inf
    torch.save(bad, p)
    assert D.TPDsaLayer._load_slices(p) is None


def test_implausible_read_backs_are_never_written(tmp_path):
    p = tmp_path / "dsa_slices-x-L03-tp8.pt"
    bad = full(0.5)
    bad["wuv"][2, 2] = float("inf")
    with pytest.raises(RuntimeError, match="device copy .* is corrupt"):
        D.TPDsaLayer._save_slices(str(p), bad)
    assert not p.exists() and not list(tmp_path.iterdir())


def test_plausible_values_pass_both_ways(tmp_path):
    p = str(tmp_path / "dsa_slices-x-L03-tp8.pt")
    ok = full(0.25)
    ok["wo"][0, 0] = -0.17
    D.TPDsaLayer._save_slices(p, ok)
    assert D.TPDsaLayer._load_slices(p) is not None


def test_every_load_logs_its_provenance(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GLM53_RUN", "resident_x")
    p = str(tmp_path / "dsa_slices-x-L03-tp8.pt")
    D.TPDsaLayer._save_slices(p, full(0.25))
    assert D.TPDsaLayer._load_slices(p) is not None
    out = capsys.readouterr().out
    assert "writer device run resident_x" in out and f"sha256 {D.TPDsaLayer._file_sha(p)}" in out


def test_a_verified_stamp_skips_the_rescan_until_the_file_changes(tmp_path, capsys, monkeypatch):
    # 14's NIT (2026-10-27T10:34:08Z): the first load hashes + scans and stamps; the next trusts the stamp (same size +
    # mtime); a rewritten file (new mtime) is verified again -- and an implausible rewrite is still refused
    p = tmp_path / "dsa_slices-x-L03-tp8.pt"
    D.TPDsaLayer._save_slices(str(p), full(0.25))
    assert D.TPDsaLayer._load_slices(str(p)) is not None
    assert (tmp_path / "dsa_slices-x-L03-tp8.pt.verified.json").exists()
    calls = []
    monkeypatch.setattr(D.TPDsaLayer, "_slices_bad", classmethod(lambda cls, d: calls.append(1) or {}))
    capsys.readouterr()
    assert D.TPDsaLayer._load_slices(str(p)) is not None and not calls  # trusted: no scan
    assert "(verified " in capsys.readouterr().out
    monkeypatch.undo()
    import os, time
    bad = full(0.5)
    bad["wo"][0, 0] = float("nan")
    torch.save(bad, str(p))
    os.utime(str(p), ns=(time.time_ns(), time.time_ns() + 10**9))  # a rewrite: new mtime
    assert D.TPDsaLayer._load_slices(str(p)) is None  # re-verified -> refused
