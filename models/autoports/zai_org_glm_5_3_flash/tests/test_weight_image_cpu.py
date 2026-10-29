# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only unit tests of tt/weight_image.py (no device: run under .agents/scripts/no-tt-devices)."""

import os
import subprocess
import types

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.host_evidence import expect_error
from models.autoports.zai_org_glm_5_3_flash.tt import weight_image as wi


class FakeTensor:
    def __init__(self, addr, shape=(32, 32), dtype="bf16", layout="TILE", mem="DRAM", allocated=True):
        self.addr, self.shape, self.dtype, self.layout, self.mem, self.alloc = (
            addr,
            list(shape),
            dtype,
            layout,
            mem,
            allocated,
        )

    def storage_type(self):
        return "DEVICE"

    def is_allocated(self):
        return self.alloc

    def buffer_address(self):
        return self.addr

    def memory_config(self):
        return self.mem


@pytest.fixture
def fake_ttnn(monkeypatch):
    ns = types.SimpleNamespace(Tensor=FakeTensor, StorageType=types.SimpleNamespace(DEVICE="DEVICE"))
    monkeypatch.setattr(wi, "ttnn", ns)
    return ns


def test_merge_segments_overlap_adjacent_and_gaps():
    rows = [
        {"address": 0x1000, "extent": 0x100},
        {"address": 0x1100, "extent": 0x40},  # adjacent -> merged
        {"address": 0x1000, "extent": 0x80},  # alias inside -> merged
        {"address": 0x2000, "extent": 0x40},  # gap -> new segment
        {"address": 0x2020, "extent": 0x40},  # overlap -> extends
    ]
    assert wi.merge_segments(rows) == [[0x1000, 0x1140], [0x2000, 0x2060]]
    assert wi.merge_segments([]) == []


class Leaf:  # module "tests..." does not start with "models." -> not walked unless reached via a models.* object
    pass


def test_device_tensors_walk_order_and_dedup(fake_ttnn):
    from models.autoports.zai_org_glm_5_3_flash.tt import weight_image as m

    root = types.SimpleNamespace()
    a, b, c = FakeTensor(0x100), FakeTensor(0x200), FakeTensor(0x300, allocated=False)
    root.w = a
    root.lst = [b, a, c]  # a seen twice -> once; c deallocated -> skipped
    root.d = {"k": types.SimpleNamespace(x=b)}
    root.n = 3
    out = m.device_tensors(root, "be")
    assert [p for p, _ in out] == ["be.w", "be.lst[0]"]
    assert [t.addr for _, t in out] == [0x100, 0x200]


def test_weightio_replay_detects_address_and_spec_mismatch(fake_ttnn, monkeypatch):
    io = wi.WeightIO("replay", {7: [{"key": "a", "address": 0x100, "spec": "[32, 32]|bf16|TILE|DRAM"}]})
    monkeypatch.setattr(io, "_chip_id", lambda mesh: 7)
    io._note(None, "a", FakeTensor(0x100))  # matches
    with expect_error(wi.ImageMismatch, "no recorded counterpart"):
        io._note(None, "b", FakeTensor(0x200))  # no recorded counterpart
    io2 = wi.WeightIO("replay", {7: [{"key": "a", "address": 0x100, "spec": "[32, 32]|bf16|TILE|DRAM"}]})
    monkeypatch.setattr(io2, "_chip_id", lambda mesh: 7)
    with expect_error(wi.ImageMismatch, "!= recorded"):
        io2._note(None, "a", FakeTensor(0x140))  # address moved
    io3 = wi.WeightIO("replay", {7: [{"key": "a", "address": 0x100, "spec": "[32, 32]|bf16|TILE|DRAM"}]})
    monkeypatch.setattr(io3, "_chip_id", lambda mesh: 7)
    with expect_error(wi.ImageMismatch, "!= recorded"):
        io3._note(None, "a", FakeTensor(0x100, dtype="bfp8"))  # spec changed
    io4 = wi.WeightIO("replay", {7: [{"key": "a", "address": 0x100, "spec": "x"}] * 2})
    monkeypatch.setattr(io4, "_chip_id", lambda mesh: 7)
    with expect_error(wi.ImageMismatch, "replay made"):
        io4.check_complete()  # fewer loads than recorded


def test_weightio_record_logs_in_order(fake_ttnn, monkeypatch):
    io = wi.WeightIO("record")
    monkeypatch.setattr(io, "_chip_id", lambda mesh: 3)
    for i in range(3):
        io._note(None, f"k{i}", FakeTensor(0x100 * (i + 1)))
    assert [r["key"] for r in io.loads[3]] == ["k0", "k1", "k2"]
    assert [r["address"] for r in io.loads[3]] == [0x100, 0x200, 0x300]


def test_code_identity_tracks_layout_trees_not_head(tmp_path):
    root = tmp_path / "repo"
    model = root / "models/autoports/zai_org_glm_5_3_flash/tt"
    model.mkdir(parents=True)
    (model / "a.py").write_text("x = 1\n")
    tests = root / "models/autoports/zai_org_glm_5_3_flash/tests"
    tests.mkdir(parents=True)
    (tests / "common.py").write_text("c = 1\n")
    (tests / "common_optimized.py").write_text("o = 1\n")
    for d in ("ttnn", "tt_metal"):
        (root / d).mkdir()
        (root / d / "f.cpp").write_text("int f();\n")
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    run = lambda *c: subprocess.run(
        ["git", "-C", str(root), *c], check=True, capture_output=True, env=env
    )  # noqa: E731
    run("init", "-q")
    run("add", ".")
    run("commit", "-qm", "init")
    clean = wi.code_identity(str(root))
    assert clean["dirty"] == "" and clean["id"].startswith("t") and len(clean["trees"]) == len(wi.CODE_PATHS)
    (root / "README").write_text("doc\n")
    run("add", "README")
    run("commit", "-qm", "docs")
    assert wi.code_identity(str(root))["id"] == clean["id"]  # a docs commit keeps the image set valid
    (model / "a.py").write_text("x = 2\n")
    dirty = wi.code_identity(str(root))
    assert dirty["dirty"] and dirty["id"].startswith(clean["id"] + "-d")
    (model / "new.py").write_text("y = 1\n")
    dirty2 = wi.code_identity(str(root))
    assert dirty2["dirty"] != dirty["dirty"]  # an untracked tt/ module changes the key
    run("add", ".")
    run("commit", "-qm", "model change")
    committed = wi.code_identity(str(root))
    assert committed["dirty"] == "" and committed["id"] != clean["id"]  # a tt/ commit changes the key
    (root / "ttnn" / "f.cpp").write_text("int g();\n")
    run("commit", "-qam", "ttnn change")
    assert wi.code_identity(str(root))["id"] != committed["id"]


def test_weights_identity(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text('{"a": 1}')
    w = wi.weights_identity(str(p))
    assert len(w["sha256"]) == 64 and w["id"] == w["sha256"][:12]


def test_find_image_identity_checks(tmp_path, monkeypatch):
    import json

    code = {"id": "abc-d1", "head": "abc"}
    weights = {"id": "w1", "sha256": "w1" * 32}
    monkeypatch.setenv("GLM53_IMAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("GLM53_LAYOUT_VERSION", "op-chain-v0")
    assert wi.find_image(code, weights) is None
    d = tmp_path / "abc-d1-w1-op-chain-v0"  # explicit layout via GLM53_LAYOUT_VERSION
    d.mkdir()
    man = {"format": wi.FORMAT, "layout_version": "op-chain-v0", "code": code, "weights": weights, "chips": {}}
    (d / "manifest.json").write_text(json.dumps(man))
    got = wi.find_image(code, weights)
    assert got is not None and got[0] == str(d)
    (d / "manifest.json").write_text(json.dumps(dict(man, layout_version="mk-stream-v1")))
    with expect_error(wi.ImageMismatch, "refusing to replay"):
        wi.find_image(code, weights)  # an op-chain build never replays a megakernel image (and vice versa)
    (d / "manifest.json").write_text(json.dumps(dict(man, format="other")))
    with expect_error(wi.ImageMismatch, "format other"):
        wi.find_image(code, weights)
    monkeypatch.setenv("GLM53_LAYOUT_VERSION", "mk-stream-v1")
    assert wi.find_image(code, weights) is None  # different key -> different directory
