# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host unit tests of the DSA host export (Stage 12 host KV tier, tt/stack_model.py StackBackend.export_dsa_state /
import_dsa_state; doc/ttnn/12-fast-start/page_dma/host_kv_tier.md 5b): the copy plan through the staging buffer, the
per-chip / replicated blob, the checkpoint records of export_kda_state / import_kda_state, and the adapter's spill hook
and sparse-regime connector guard. The device is modelled: tensors are per-chip tile arrays, SlotOps copies tiles,
the staging read / write are the ttnn host transfers. No device, no vLLM.

    no-tt-devices venv-lmcache/bin/python -m pytest tests/vllm/test_dsa_export.py -q -p no:cacheprovider \\
        --confcutdir tests/vllm
"""

from __future__ import annotations

import types

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as GV
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM

REG: dict = {}


class FakeT:
    """a per-chip TILE tensor: mem [chips, tiles, 32, 32] bf16 at a unique buffer address"""

    _next = [0x10000]

    def __init__(self, shape, chips, fp32=False):
        self.padded_shape = tuple(shape)
        self.dtype = SM.ttnn.float32 if fp32 else SM.ttnn.bfloat16
        self.mem = torch.zeros(chips, SM._pages(self), 32, 32, dtype=torch.float32 if fp32 else torch.bfloat16)
        self.addr = FakeT._next[0]
        FakeT._next[0] += 0x100000
        REG[self.addr] = self

    def buffer_address(self):
        return self.addr


class FakeSlotOps:
    def __init__(self):
        self.launches = 0

    def run(self, ops):
        self.launches += 1
        for kind, sa, s0, da, d0, n, pb, rows in ops:
            src, dst = REG[sa], REG[da]
            assert pb == (4096 if dst.mem.dtype == torch.float32 else 2048) and src.mem.dtype == dst.mem.dtype
            assert 0 <= s0 and s0 + n <= src.mem.shape[1] and 0 <= d0 and d0 + n <= dst.mem.shape[1], (s0, d0, n)
            if kind == 1:
                assert rows == 0
                dst.mem[:, d0 : d0 + n] = src.mem[:, s0 : s0 + n].clone()
            else:  # kind 2: row `dst row` of every dst tile <- row `src row` of the matching src tile
                assert kind == 2
                sr, dr = rows >> 16, rows & 0xFFFF
                dst.mem[:, d0 : d0 + n, dr, :] = src.mem[:, s0 : s0 + n, sr, :].clone()


@pytest.fixture
def fake_ttnn(monkeypatch):
    t = SM.ttnn

    def to_torch(x, mesh_composer=None):
        m = x.mem
        return m.reshape(-1, 1, 32, 32).clone() if mesh_composer is not None else m[0].reshape(-1, 1, 32, 32).clone()

    def from_torch(x, dtype=None, layout=None, mesh_mapper=None, **kw):
        return types.SimpleNamespace(x=x, mapper=mesh_mapper)

    def copy_host_to_device_tensor(h, dev):
        n = dev.mem.shape[1]
        x = h.x.reshape(-1, n, 32, 32)
        if isinstance(h.mapper, tuple) and h.mapper[0] == "shard":
            assert x.shape[0] == dev.mem.shape[0]
            dev.mem[:] = x
        else:
            assert x.shape[0] == 1
            dev.mem[:] = x[0]

    monkeypatch.setattr(t, "to_torch", to_torch, raising=False)
    monkeypatch.setattr(t, "from_torch", from_torch, raising=False)
    monkeypatch.setattr(t, "copy_host_to_device_tensor", copy_host_to_device_tensor, raising=False)
    monkeypatch.setattr(t, "synchronize_device", lambda mesh: None, raising=False)
    monkeypatch.setattr(t, "ConcatMeshToTensor", lambda mesh, dim: ("concat", dim), raising=False)
    monkeypatch.setattr(t, "ShardTensorToMesh", lambda mesh, dim: ("shard", dim), raising=False)
    monkeypatch.setattr(t, "ReplicateTensorToMesh", lambda mesh: ("repl",), raising=False)
    return t


def backend(chips=2, rows=6, pools_per_slot=256, cap=10, layers=(3, 9)):
    """a StackBackend with only sparse_dsa layers (no KDA state): store [rows * pools / 32, 1, 32, 128] (4 tiles per
    128-token unit), ring [rows, 1, 32, 256] (8 tiles per slot), staging [cap, 1, 32, 32]"""
    b = SM.StackBackend.__new__(SM.StackBackend)
    b.tp, b.mesh, b.chunk, b.max_batch = chips, "mesh", 4, 4
    b.seq = [dict(L=L, sparse="dsa") for L in layers]
    b.slot_ops, b.counters = FakeSlotOps(), dict(slot_ops=0)
    dsa = {i: dict(pools=FakeT((rows * pools_per_slot // 32, 1, 32, 128), chips), ring=FakeT((rows, 1, 32, 256), chips),
                   slots=rows, tp=pools_per_slot) for i in range(len(layers))}  # fmt: skip
    kv = types.SimpleNamespace(extra=dict(dsa=dsa, dsa_stage=FakeT((cap, 1, 32, 32), chips)),
                               states=[[None] * len(layers) for _ in range(-(-rows // 4))])  # fmt: skip
    b._kv = kv
    return b, kv


def fill(kv, distinct_chips=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    for st in kv.extra["dsa"].values():
        for t in (st["pools"], st["ring"]):
            base = torch.randn(t.mem.shape[1:], generator=g).to(torch.bfloat16)
            for c in range(t.mem.shape[0]):
                t.mem[c] = base + (c if distinct_chips else 0)


def slot_tiles(t, rows, slot):
    per = t.mem.shape[1] // rows
    return t.mem[:, slot * per : (slot + 1) * per]


def test_rounds_cover_the_units_and_the_ring(fake_ttnn):
    b, kv = backend(cap=10)
    rounds = b._dsa_rounds(kv, 2, units=3)
    moved = {}
    for rnd in rounds:
        assert sum(n for _k, _t0, n, _s0 in rnd) <= 10
        used = sorted((s0, s0 + n) for _k, _t0, n, s0 in rnd)
        assert all(a[1] <= c[0] for a, c in zip(used, used[1:])) and used[-1][1] <= 10  # no staging overlap
        for k, t0, n, _s0 in rnd:
            moved.setdefault(k, []).append((t0, n))
    # per layer: 3 units x 4 tiles of the store (slot 2 of 32 tiles each) + the 8 ring tiles (slot 2 of 8 each)
    assert sorted(moved) == ["L3.pools", "L3.ring", "L9.pools", "L9.ring"]
    for k, runs in moved.items():
        per, want = (32, 12) if k.endswith("pools") else (8, 8)
        assert runs[0][0] == 2 * per and sum(n for _t0, n in runs) == want
        assert all(runs[j][0] + runs[j][1] == runs[j + 1][0] for j in range(len(runs) - 1))  # contiguous
    assert len(rounds) == 4  # 40 tiles through a 10-tile staging buffer
    assert len(b._dsa_rounds(kv, 2, units=None)) == 8  # every unit of the slot: 2 x (32 + 8) = 80 tiles


def test_export_import_roundtrip_per_chip(fake_ttnn):
    b, kv = backend()
    fill(kv, distinct_chips=True)
    blob = b.export_dsa_state(kv, 2, n_tokens=300)  # ceil(300 / 128) = 3 units
    assert blob["kind"] == SM.StackBackend.DSA_BLOB and blob["units"] == 3 and blob["chips"] == 2
    assert not blob["replicated"] and blob["tensors"]["L3.pools"].shape == (2, 12, 32, 32)
    assert blob["tensors"]["L9.ring"].shape == (2, 8, 32, 32)
    before = {k: (st["pools"].mem.clone(), st["ring"].mem.clone()) for k, st in kv.extra["dsa"].items()}
    b.import_dsa_state(kv, 5, blob)
    for i, st in kv.extra["dsa"].items():
        p, r = st["pools"], st["ring"]
        assert torch.equal(slot_tiles(p, 6, 5)[:, :12], slot_tiles(p, 6, 2)[:, :12])  # every chip its own copy
        assert torch.equal(slot_tiles(p, 6, 5)[:, 12:], slot_tiles(FakeView(before[i][0]), 6, 5)[:, 12:])  # units past 3
        assert torch.equal(slot_tiles(r, 6, 5), slot_tiles(r, 6, 2))
        for s in (0, 1, 2, 3, 4):  # other slots untouched
            assert torch.equal(slot_tiles(p, 6, s), slot_tiles(FakeView(before[i][0]), 6, s))
    assert b.counters["dsa_exports"] == 1 and b.counters["dsa_imports"] == 1
    assert b.counters["dsa_export_replicated"] == 0


class FakeView:
    def __init__(self, mem):
        self.mem = mem


def test_replicated_copies_are_kept_once_and_written_to_every_chip(fake_ttnn):
    b, kv = backend(chips=4)
    fill(kv, distinct_chips=False, seed=3)
    blob = b.export_dsa_state(kv, 1, n_tokens=128)
    assert blob["replicated"] and blob["tensors"]["L3.pools"].shape == (1, 4, 32, 32)
    b.import_dsa_state(kv, 4, blob)
    for st in kv.extra["dsa"].values():
        p, r = st["pools"], st["ring"]
        assert torch.equal(slot_tiles(p, 6, 4)[:, :4], slot_tiles(p, 6, 1)[:, :4])
        assert torch.equal(slot_tiles(r, 6, 4), slot_tiles(r, 6, 1))
    assert b.counters["dsa_export_replicated"] == 1


def test_checkpoint_records_carry_the_dsa_rows(fake_ttnn):
    b, kv = backend()
    fill(kv, seed=5)
    states = b.export_kda_state(kv, 0, n_tokens=256)
    assert len(states) == 1 and states[0]["dsa"]["units"] == 2  # no KDA layers in this backend
    assert GV._states_dsa(states) and not GV._states_dsa([dict(L=0, S=torch.zeros(1), C=torch.zeros(1))])
    assert GV._blob_bytes(states) == 2 * 2 * (8 + 8) * 2048  # 2 chips x 2 layers x (2 units x 4 + 8 ring) tiles
    b.import_kda_state(kv, 3, states)
    for k in ("kda_export_s", "dsa_export_s", "kda_import_s", "dsa_import_s", "dsa_import_last_s"):  # T1 timing
        assert b.counters[k] >= 0.0, k
    assert b.counters["dsa_export_bytes"] == 2 * 2 * (8 + 8) * 2048
    for st in kv.extra["dsa"].values():
        assert torch.equal(slot_tiles(st["pools"], 6, 3)[:, :8], slot_tiles(st["pools"], 6, 0)[:, :8])
        assert torch.equal(slot_tiles(st["ring"], 6, 3), slot_tiles(st["ring"], 6, 0))
    kv.extra["dsa_stage"] = None  # GLM53_DSA_EXPORT_STAGE_TILES=0: no DSA record (the adapter then refuses)
    assert not GV._states_dsa(b.export_kda_state(kv, 0, n_tokens=256))


def test_import_refuses_a_foreign_or_short_blob(fake_ttnn):
    b, kv = backend()
    fill(kv)
    blob = b.export_dsa_state(kv, 0, n_tokens=128)
    other, kv2 = backend(layers=(3, 10))
    with pytest.raises(ValueError, match="DSA blob tensors"):
        other.import_dsa_state(kv2, 0, blob)
    short = dict(blob, units=3)  # claims more units than it holds
    with pytest.raises(ValueError, match="tiles"):
        b.import_dsa_state(kv, 1, short)
    with pytest.raises(TypeError):
        b.import_dsa_state(kv, 1, {"kind": "other"})
    with pytest.raises(ValueError, match="rows"):
        b.export_dsa_state(kv, 6, n_tokens=128)  # slot past the store's rows


def test_dsa_ops_unchanged_by_the_census(fake_ttnn):
    b, kv = backend()
    ops = b._dsa_ops(kv, 4, 1)
    st = kv.extra["dsa"][0]
    assert ops[0] == (1, st["pools"].addr, 32, st["pools"].addr, 128, 32, 2048, 0)
    assert ops[1] == (1, st["ring"].addr, 8, st["ring"].addr, 32, 8, 2048, 0)
    assert len(ops) == 4 and b.has_dsa_state(kv)


# ----------------------------------------------------------------------------------- the adapter (generator_vllm)
class _Gen:
    def __init__(self, blob):
        self.blob, self.calls = blob, []

    def export_snapshot(self, kv, entry, n_tokens=None):
        self.calls.append((entry, n_tokens))
        return self.blob


def _adapter(blob, sparse):
    a = types.SimpleNamespace(generator=_Gen(blob), kv_cache="kv", counters={}, prefix=types.SimpleNamespace(entry_bytes=77))
    a._dsa_state = lambda: sparse
    return a


def test_spill_hook_exports_the_dsa_rows_in_the_sparse_regime():
    dsa_blob = [dict(L=0, S=torch.zeros(4, dtype=torch.float32)), dict(dsa=dict(tensors={"x": torch.zeros(8, dtype=torch.bfloat16)}))]
    ent = types.SimpleNamespace(device_entry=3, n_tokens=4096)
    a = _adapter(dsa_blob, sparse=True)
    blob, nbytes = GV.Glm5NextForCausalLM._prefix_spill_export(a, ent)
    assert blob is dsa_blob and nbytes == 16 + 16 and a.generator.calls == [(3, 4096)]
    assert a.counters["prefix_spills"] == 1 and a.counters["prefix_spill_bytes"] == 32 and a.counters["prefix_spill_s"] >= 0
    a = _adapter([dict(L=0, S=torch.zeros(4))], sparse=True)  # no DSA record: refused (dropped as before)
    assert GV.Glm5NextForCausalLM._prefix_spill_export(a, ent) is None and a.counters["prefix_spill_no_dsa"] == 1
    a = _adapter([dict(L=0, S=torch.zeros(4))], sparse=False)  # dense regime: unchanged (the entry-bytes model)
    assert GV.Glm5NextForCausalLM._prefix_spill_export(a, ent)[1] == 77
    assert GV.Glm5NextForCausalLM._prefix_spill_export(a, types.SimpleNamespace(device_entry=None, n_tokens=128)) is None


def test_staging_on_demand_and_never_after_a_capture(fake_ttnn, monkeypatch):
    b, kv = backend()
    kv.extra["dsa_stage"] = None
    made = []
    monkeypatch.setattr(SM.ttnn, "zeros", lambda shape, **kw: made.append(shape) or FakeT(shape, 2), raising=False)
    b.counters["captures"] = 1  # a trace exists: a new buffer could sit on its transient memory
    with pytest.raises(RuntimeError, match="before the first trace capture"):
        b.ensure_dsa_export(kv)
    b.counters["captures"] = 0
    assert b.ensure_dsa_export(kv) and made == [(256, 1, 32, 32)]
    assert b.ensure_dsa_export(kv) and len(made) == 1  # idempotent
    b2, kv2 = backend()
    kv2.extra["dsa"] = {}  # no DSA state (dense regime): nothing to stage
    assert not b2.ensure_dsa_export(kv2)


# ------------------------------------------------------------------------------- the KDA pack (one-gather checkpoints)
def kda_backend(chips=2, n_kda=34, chunk=4, monkeypatch=None, stage_layers="8"):
    """a StackBackend with n_kda KDA layers (S [chunk, 2, 32, 32] fp32 = 2 pages per slot, C [1, 1, chunk, 96] bf16 =
    3 tiles (padded to 32 rows), users = tile rows) + one sparse_dsa layer; the KDA pack staging allocated by ensure_state_export"""
    b = SM.StackBackend.__new__(SM.StackBackend)
    b.tp, b.mesh, b.chunk, b.max_batch = chips, "mesh", chunk, chunk
    b.seq = [dict(L=L, sparse=None) for L in range(n_kda)] + [dict(L=99, sparse="dsa")]
    b.slot_ops, b.counters = FakeSlotOps(), dict(slot_ops=0)
    states = [[(FakeT((chunk, 2, 32, 32), chips, fp32=True), FakeT((1, 1, 32, 96), chips)) for _ in range(n_kda)] + [None]
              for _ in range(2)]  # fmt: skip
    kv = types.SimpleNamespace(extra=dict(), states=states)
    b._kv = kv
    monkeypatch.setenv("GLM53_KDA_EXPORT_STAGE_LAYERS", stage_layers)
    monkeypatch.setattr(SM.ttnn, "zeros", lambda shape, dtype=None, **kw: FakeT(shape, chips, fp32=dtype == SM.ttnn.float32),
                        raising=False)  # fmt: skip
    assert b.ensure_state_export(kv)
    g = torch.Generator().manual_seed(4)
    for c in range(2):
        for S, C in states[c][:n_kda]:
            S.mem.copy_(torch.randn(S.mem.shape, generator=g))
            C.mem.copy_(torch.randn(C.mem.shape, generator=g).to(torch.bfloat16))
    return b, kv


def test_kda_pack_roundtrip_through_the_staging(fake_ttnn, monkeypatch):
    b, kv = kda_backend(monkeypatch=monkeypatch, stage_layers="8")
    assert tuple(kv.extra["kda_stage_s"].padded_shape) == (16, 1, 32, 32)  # 8 layers x 2 pages
    assert tuple(kv.extra["kda_stage_c"].padded_shape) == (3, 1, 32, 32)  # one layer's conv tiles
    before = [[(S.mem.clone(), C.mem.clone()) for S, C in kv.states[c][:34]] for c in range(2)]
    states = b.export_kda_state(kv, 2, n_tokens=256)  # slot 2 = chunk 0, row 2
    assert len(states) == 1 and "kda" in states[0] and states[0]["kda"]["S"].shape == (2, 34, 2, 32, 32)
    assert states[0]["kda"]["C"].shape == (2, 34, 3, 32)
    launches = b.slot_ops.launches
    b.import_kda_state(kv, 5, states)  # slot 5 = chunk 1, row 1
    assert b.slot_ops.launches - launches == 5 + 2  # ceil(34 / 8) S rounds + ceil(34 / 32) conv rounds
    for li in range(34):
        S0, C0 = kv.states[0][li]
        S1, C1 = kv.states[1][li]
        assert torch.equal(S1.mem[:, 2:4], S0.mem[:, 4:6])  # slot 5's 2 pages == slot 2's
        assert torch.equal(S1.mem[:, :2], before[1][li][0][:, :2]) and torch.equal(S1.mem[:, 4:], before[1][li][0][:, 4:])
        assert torch.equal(C1.mem[:, :, 1, :], C0.mem[:, :, 2, :])  # conv row 1 of chunk 1 == row 2 of chunk 0
        rows = [r for r in range(32) if r != 1]
        assert torch.equal(C1.mem[:, :, rows, :], before[1][li][1][:, :, rows, :])
        assert torch.equal(S0.mem, before[0][li][0]) and torch.equal(C0.mem, before[0][li][1])  # the source untouched
    assert b.counters["kda_pack_exports"] == 1 and b.counters["kda_pack_imports"] == 1


def test_kda_pack_refuses_a_foreign_blob_and_the_gather_path_stays(fake_ttnn, monkeypatch):
    b, kv = kda_backend(monkeypatch=monkeypatch, n_kda=3)
    blob = b.export_kda_pack(kv, 0)
    b2, kv2 = kda_backend(monkeypatch=monkeypatch, n_kda=4)
    with pytest.raises(ValueError, match="does not match"):
        b2.import_kda_pack(kv2, 0, blob)
    b3, kv3 = kda_backend(monkeypatch=monkeypatch, n_kda=3)
    b3.counters["captures"] = 1
    kv3.extra.pop("kda_stage_s")
    with pytest.raises(RuntimeError, match="before the first trace capture"):
        b3.ensure_state_export(kv3)


def test_mock_stub_skips_the_host_transfers(fake_ttnn, monkeypatch):
    b, kv = kda_backend(monkeypatch=monkeypatch, n_kda=3)
    monkeypatch.setattr(SM, "MOCK", True)
    called = []
    monkeypatch.setattr(SM.ttnn, "to_torch", lambda *a, **k: called.append(1), raising=False)
    blob = b.export_kda_pack(kv, 1)
    assert not called and blob["S"].abs().sum() == 0 and blob["S"].shape == (2, 3, 2, 32, 32)
    b.import_kda_pack(kv, 2, blob)  # writes skipped, SlotOps still planned and run
    assert b.counters["kda_pack_imports"] == 1


# ------------------------------------------------- Stage 12: pooled keys in the LMCache pages (GLM53_KV_POOLED_PAGES)
def test_pools_less_checkpoint_carries_the_ring_only(fake_ttnn):
    """pools=False: the checkpoint's DSA record holds the ring and NO pooled-key rows (units 0, 'pools' False) -- O(n)
    per checkpoint; the import writes the ring and leaves the slot's pooled keys (the hit pages bring them)"""
    b, kv = backend()
    fill(kv, distinct_chips=False, seed=7)  # a replicated pool's copies (a per-chip blob needs the shard's C)
    states = b.export_kda_state(kv, 2, n_tokens=300, pools=False)
    blob = states[0]["dsa"]
    assert blob["units"] == 0 and blob["pools"] is False and blob["n_tokens"] == 300
    assert blob["tensors"]["L3.pools"].shape[1] == 0 and blob["tensors"]["L9.ring"].shape[1] == 8
    assert GV._states_dsa(states) and GV._states_pools_less(states)
    assert not GV._states_pools_less(b.export_kda_state(kv, 2, n_tokens=300))
    before = {i: st["pools"].mem.clone() for i, st in kv.extra["dsa"].items()}
    b.import_kda_state(kv, 5, states)
    for i, st in kv.extra["dsa"].items():
        assert torch.equal(st["pools"].mem, before[i]), "pooled keys untouched by a ring-only import"
        assert torch.equal(slot_tiles(st["ring"], 6, 5), slot_tiles(st["ring"], 6, 2))
    assert b.counters["dsa_export_bytes"] == 2 * 8 * 2048 + 2 * (12 + 8) * 2048  # ring only, then the full one (kept once)


def test_pooled_store_targets_replicated_and_sharded(fake_ttnn):
    """the page-DMA's view of the stores: the 11 (here 2) stores in layer order; a slot's unit u at (None, slot pps + u)
    replicated, at (u % C, slot pps + u // C) on the sharded pool"""
    b, kv = backend(rows=6, pools_per_slot=256)  # pps = 256 / 32 = 8 unit pages per slot
    kv.extra["latent"] = {3: "lat3", 9: "lat9"}
    stores = b.pooled_store_tensors(kv)
    assert [t.addr for t in stores] == [kv.extra["dsa"][i]["pools"].addr for i in (0, 1)]
    assert b.pooled_unit_pages(kv, 2, [0, 1, 7]) == [(None, 16), (None, 17), (None, 23)]
    kv.extra["dsa_shard"] = (4, 99, 2)  # C = 4: each chip holds a slot's local units (8 local pages per slot)
    assert b.pooled_unit_pages(kv, 2, [0, 1, 4, 5, 31]) == [(0, 16), (1, 16), (0, 17), (1, 17), (3, 23)]
    with pytest.raises(ValueError, match="local unit"):
        b.pooled_unit_pages(kv, 2, [32])
    with pytest.raises(ValueError, match="rows"):
        b.pooled_unit_pages(kv, 6, [0])
    kv.extra["latent"] = {3: "lat3"}  # stores and latent caches disagree: no pooled-key DMA
    assert b.pooled_store_tensors(kv) == []


class _PooledDma:
    """the LatentHitDma pooled interface over a host dict {(chip, page): [11 x 8,192 B]}"""

    def __init__(self):
        self.mem, self.reads, self.loads, self.waits = {}, [], [], 0

    def read_pooled(self, targets):
        import numpy as np

        self.reads.append(list(targets))
        z = np.zeros(8192, np.uint8)
        return [np.stack([self.mem.get(t, [z] * 11)[l] for t in targets]) for l in range(11)]

    def load_pooled(self, data, targets):
        self.loads.append(list(targets))
        for k, t in enumerate(targets):
            self.mem[t] = [d[k].copy() for d in data]

    def load_pooled_blocks(self, rows, targets):  # the reference layout (LatentHitDma builds it in the ring)
        import numpy as np

        from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

        pages = [np.concatenate([PP.block_page_of_half(PP.pooled_halves(u[l]).reshape(-1, PP.POOLED_PAGE_BYTES)[:n])
                                 for u, n in rows]) for l in range(11)]  # fmt: skip
        self.load_pooled(pages, targets)

    def wait(self):
        self.waits += 1


def test_adapter_reads_page_halves_and_loads_hit_units():
    """kv_read_pooled_pages: (vLLM slot, column) -> the unit page's half (column % 2) of the generator slot's unit
    (column // 2), units read once; _load_prefix_pooled: a hit's units 0..U-1 -> the generator slot's targets, waited"""
    import numpy as np

    from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

    rng = np.random.default_rng(9)
    backend_ = types.SimpleNamespace(pooled_unit_pages=lambda kv, gs, units: [(u % 2, gs * 100 + u // 2) for u in units])
    a = types.SimpleNamespace(gen_slot={0: 3, 1: 5}, generator=types.SimpleNamespace(backend=backend_), kv_cache="kv",
                              counters={}, _hit_dma=_PooledDma())  # fmt: skip
    units = {(1, 300): [rng.integers(0, 256, 8192, dtype=np.uint8) for _ in range(11)],
             (0, 300): [rng.integers(0, 256, 8192, dtype=np.uint8) for _ in range(11)],
             (1, 501): [rng.integers(0, 256, 8192, dtype=np.uint8) for _ in range(11)]}  # fmt: skip
    a._hit_dma.mem.update(units)
    # vLLM slot 0 = generator slot 3: column 1 -> unit 0 (chip 0, page 300) half 1; columns 2, 3 -> unit 1 (chip 1,
    # page 300); vLLM slot 1 = generator slot 5: column 2 -> unit 1 (chip 1, page 500 + 0)... page 501 = unit 3 (column 6)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 6)]
    got = GV.Glm5NextForCausalLM.kv_read_pooled_pages(a, pairs)
    assert a._hit_dma.reads == [[(0, 300), (1, 300), (1, 501)]], "each unit read once"
    for l in range(11):
        want = [PP.pooled_halves(units[(0, 300)][l][None])[0, 1], PP.pooled_halves(units[(1, 300)][l][None])[0, 0],
                PP.pooled_halves(units[(1, 300)][l][None])[0, 1], PP.pooled_halves(units[(1, 501)][l][None])[0, 0]]
        assert np.array_equal(got[l], np.stack(want)), l
    assert a.counters["pooled_page_reads"] == 4
    # the load: a hit of 3 units into generator slot 5, two of 1 unit into slot 3
    data = [rng.integers(0, 256, (3, 8192), dtype=np.uint8) for _ in range(11)]
    one = [rng.integers(0, 256, (1, 8192), dtype=np.uint8) for _ in range(11)]
    GV.Glm5NextForCausalLM._load_prefix_pooled(a, {1: (5, dict(units=3, data=data), [11, 12, 13, 14, 15, 16]),
                                                   0: (3, dict(units=1, data=one), [21, 22])})  # fmt: skip
    assert a._hit_dma.loads == [[(0, 300), (0, 500), (1, 500), (0, 501)]] and a._hit_dma.waits == 1
    assert all(np.array_equal(a._hit_dma.mem[(1, 500)][l], data[l][1]) for l in range(11))
    assert all(np.array_equal(a._hit_dma.mem[(0, 300)][l], one[l][0]) for l in range(11))
    assert a.counters["pooled_unit_loads"] == 4


# --------------------------------------- GLM53_DSA_PAGED_POOLS (paged_pooled_keys.md rev 3: one page per latent block)
def test_paged_pool_state_is_refused_until_the_kernels_address_it(fake_ttnn, monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MKM

    b, kv = backend()
    b.seq = [dict(L=L, sparse="dsa", mla=types.SimpleNamespace(mk=types.SimpleNamespace(dsa=MKM.DsaIndexer)))
             for L in (3, 9)]  # fmt: skip
    b.dsa_shard = False
    with pytest.raises(ValueError, match="GLM53_DSA_SHARD"):
        b._paged_pool_state(0, 100, 6, 4096)
    b.dsa_shard = True
    monkeypatch.delattr(MKM, "DSA_PAGED_POOLS", raising=False)
    with pytest.raises(NotImplementedError, match="kernels"):
        b._paged_pool_state(0, 100, 6, 4096)
    monkeypatch.setattr(MKM, "DSA_PAGED_POOLS", True, raising=False)
    monkeypatch.setattr(SM.ttnn, "zeros", lambda shape, **kw: FakeT(shape, 2), raising=False)
    st = b._paged_pool_state(0, 100, 6, 4096)
    assert st["paged"] and st["pools"].padded_shape == (100, 1, 32, 128) and st["ring"].padded_shape == (6, 1, 32, 256)
    assert st["slots"] == 6 and st["tp"] == MKM.DsaIndexer.pool_rows(4096)


def test_paged_stores_leave_the_slot_copies_and_checkpoints(fake_ttnn):
    """a paged store is per BLOCK: _dsa_tensors (slot copies, checkpoints, snapshots) keeps only the rings; the page-DMA
    sees the stores (pooled_store_tensors); a block's page = (g % C, g // C)"""
    b, kv = backend()
    for st in kv.extra["dsa"].values():
        st["paged"] = True
    kv.extra["latent"] = {3: "lat3", 9: "lat9"}
    kv.extra["dsa_shard"] = (4, 99, 2)
    assert [k for k, _t, _r, _p in b._dsa_tensors(kv)] == ["L3.ring", "L9.ring"]
    assert b.pooled_paged(kv) and [t.addr for t in b.pooled_store_tensors(kv)] == [
        kv.extra["dsa"][i]["pools"].addr for i in (0, 1)]  # fmt: skip
    assert b.pooled_block_pages(kv, [0, 5, 13, 30]) == [(0, 0), (1, 1), (1, 3), (2, 7)]
    states = b.export_kda_state(kv, 2, n_tokens=300)
    assert sorted(states[0]["dsa"]["tensors"]) == ["L3.ring", "L9.ring"]


def test_paged_prefill_pool_idx_follows_the_latent_blocks():
    from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH

    sm = DSH.ShardMap(4, 64)
    blocks = torch.tensor([8, 13, 6, 3])  # physical: owners 0, 1, 2, 3 at local blocks 2, 3, 1, 0
    idx = sm.prefill_pool_idx_paged(blocks, n_valid=200)  # 200 valid rows -> blocks 0..3 (64 k < 200)
    assert idx.shape == (4, 16)
    assert idx[0, 0:4].tolist() == [8, 9, 10, 11] and idx[1, 4:8].tolist() == [12, 13, 14, 15]
    assert idx[2, 8:12].tolist() == [4, 5, 6, 7] and idx[3, 12:16].tolist() == [0, 1, 2, 3]
    assert (idx[0, 4:] == DSH.SKIP).all() and (idx[1, :4] == DSH.SKIP).all()
    short = sm.prefill_pool_idx_paged(blocks, n_valid=64)  # only block 0 holds valid rows
    assert (short[:, 4:] == DSH.SKIP).all() and short[0, :4].tolist() == [8, 9, 10, 11]
    packed = sm.packed_pool_idx_paged(256, [(0, 64, torch.tensor([8])), (128, 100, torch.tensor([13, 6]))])
    assert packed.shape == (4, 16)
    assert packed[0, 0:4].tolist() == [8, 9, 10, 11] and (packed[:, 4:8] == DSH.SKIP).all()
    assert packed[1, 8:12].tolist() == [12, 13, 14, 15] and packed[2, 12:16].tolist() == [4, 5, 6, 7]


def test_adapter_paged_reads_and_loads_go_to_the_blocks_pages():
    """GLM53_DSA_PAGED_POOLS: kv_read_pooled_pages reads each LOGICAL block's physical page (rows 0-15); a hit's units
    are split into halves, each written PADDED to its admitted block's page"""
    import numpy as np

    from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

    rng = np.random.default_rng(10)
    phys = {40: 9, 41: 14, 42: 3}
    obm = types.SimpleNamespace(physical=lambda b: phys.get(int(b)))
    backend_ = types.SimpleNamespace(pooled_paged=lambda kv: True,
                                     pooled_block_pages=lambda kv, gs: [(g % 4, g // 4) for g in gs])  # fmt: skip
    a = types.SimpleNamespace(gen_slot={0: 3}, generator=types.SimpleNamespace(backend=backend_,
                              dsa_shard_map=lambda kv: obm), kv_cache="kv", counters={}, _hit_dma=_PooledDma())  # fmt: skip
    for g in phys.values():
        a._hit_dma.mem[(g % 4, g // 4)] = [rng.integers(0, 256, 8192, dtype=np.uint8) for _ in range(11)]
    got = GV.Glm5NextForCausalLM.kv_read_pooled_pages(a, [(0, 0), (0, 1)], pages=[41, 40])
    assert a._hit_dma.reads == [[(2, 3), (1, 2)]]
    for l in range(11):
        assert np.array_equal(got[l][0], PP.half_of_block_page(a._hit_dma.mem[(2, 3)][l][None])[0])
    with pytest.raises(ValueError, match="logical blocks"):
        GV.Glm5NextForCausalLM.kv_read_pooled_pages(a, [(0, 0)])
    with pytest.raises(ValueError, match="without device content"):
        GV.Glm5NextForCausalLM.kv_read_pooled_pages(a, [(0, 0)], pages=[77])
    data = [rng.integers(0, 256, (2, 8192), dtype=np.uint8) for _ in range(11)]  # 2 units = 4 halves; 3 blocks hit
    GV.Glm5NextForCausalLM._load_prefix_pooled(a, {0: (3, dict(units=2, pages=3, data=data), [40, 41, 42, 0])})
    assert a._hit_dma.loads[-1] == [(1, 2), (2, 3), (3, 0)]
    for l in range(11):
        halves = PP.pooled_halves(data[l]).reshape(4, -1)
        assert np.array_equal(a._hit_dma.mem[(2, 3)][l], PP.block_page_of_half(halves[1:2])[0])
        assert not a._hit_dma.mem[(3, 0)][l].reshape(4, 2, 1024)[:, 1].any(), "rows 16-31 stay zero"


def _prefill_alias(b, kv, rows=6):
    """the TP prefill bound to the decode's DSA state (prefill_tp MlaPrefill: idx.store = st["pools"], idx.ring =
    st["ring"], the paged store included)"""
    layers = [types.SimpleNamespace(mla=types.SimpleNamespace(idx=types.SimpleNamespace(store=st["pools"], ring=st["ring"])))
              for _i, st in sorted(kv.extra["dsa"].items())]  # fmt: skip
    b.tpp = types.SimpleNamespace(kv=kv, layers=layers, slot_rows=rows, has_dsa=lambda: True)


def test_paged_checkpoint_is_pools_less_and_skips_the_prefill_alias(fake_ttnn):
    """n29h7s2 (2026-10-23): under GLM53_DSA_PAGED_POOLS the TP prefill aliases the paged store. _dsa_tensors must not
    take that alias as a per-slot store (its 'slot units' are OTHER blocks' pages: the export read them and a hit's
    import wrote them back over those blocks), and the ring-only record says pools=False, so a connector hit restarts
    at 0 unless it brings its blocks' pooled pages"""
    b, kv = backend()
    for st in kv.extra["dsa"].values():
        st["paged"] = True
    kv.extra["dsa_shard"] = (2, 99, 2)
    fill(kv, distinct_chips=False, seed=3)
    _prefill_alias(b, kv)
    assert [k for k, _t, _r, _p in b._dsa_tensors(kv)] == ["L3.ring", "L9.ring"]
    states = b.export_kda_state(kv, 2, n_tokens=4000)
    blob = states[0]["dsa"]
    assert blob["pools"] is False and blob["units"] == 0 and sorted(blob["tensors"]) == ["L3.ring", "L9.ring"]
    assert GV._states_pools_less(states)
    before = {i: st["pools"].mem.clone() for i, st in kv.extra["dsa"].items()}
    b.import_kda_state(kv, 5, states)
    for i, st in kv.extra["dsa"].items():
        assert torch.equal(st["pools"].mem, before[i]), "a checkpoint import never writes the paged store"
        assert torch.equal(slot_tiles(st["ring"], 6, 5), slot_tiles(st["ring"], 6, 2))


def test_dense_store_keeps_its_prefill_alias_once():
    """the dense store: the prefill's alias of the decode's store is the same buffer, kept once (as before the fix)"""
    b, kv = backend()
    _prefill_alias(b, kv)
    assert [k for k, _t, _r, _p in b._dsa_tensors(kv)] == ["L3.pools", "L3.ring", "L9.pools", "L9.ring"]


def test_hit_restart_reasons():
    """the connector-hit guard (prefill_forward): which checkpoints resume and which restart at 0"""
    full = [dict(dsa=dict(pools=True, tensors={}))]
    ring = [dict(dsa=dict(pools=False, tensors={}))]
    pooled = dict(units=40)  # 5,120 tokens of pooled keys
    R = GV._hit_restart
    assert R(dict(states=[dict(kda={})]), 4096, False, "dma") == "no_dsa"
    assert R(dict(states=full), 4096, False, None) is None  # dense store, the record carries the pooled rows
    assert R(dict(states=ring), 4096, False, None) == "no_pooled"
    assert R(dict(states=ring, pooled=pooled), 4096, False, "dma") is None
    assert R(dict(states=ring, pooled=pooled), 8192, False, "dma") == "no_pooled"  # pooled keys short of the hit
    # GLM53_DSA_PAGED_POOLS: the cache decides
    assert R(dict(states=full), 4096, True, "dma") == "no_pooled"  # a pre-fix record (pools=True) on a paged store
    assert R(dict(states=full, pooled=pooled), 4096, True, "dma") == "no_pooled"
    assert R(dict(states=ring), 4096, True, "dma") == "no_pooled"  # GLM53_KV_T2=0: nothing loads the blocks' pages
    assert R(dict(states=ring, pooled=pooled), 4096, True, None) == "no_pooled"
    assert R(dict(states=ring, pooled=pooled), 4096, True, "dma") is None  # T2=1 + GLM53_KV_POOLED_PAGES=1


def test_paged_snapshot_is_pools_less_like_the_export(fake_ttnn, monkeypatch):
    """the async prompt-end checkpoint (snapshot_state + collect_snapshot, GLM53_KV_T2=1) under GLM53_DSA_PAGED_POOLS:
    the same ring-only, pools=False record as export_kda_state, bitwise; the prefill alias is not snapshotted"""
    b, kv = kda_backend(monkeypatch=monkeypatch)
    kv.extra["dsa"] = {34: dict(pools=FakeT((64, 1, 32, 128), 2), ring=FakeT((8, 1, 32, 256), 2), slots=8, tp=4096,
                                paged=True)}  # fmt: skip
    kv.extra["dsa_stage"], kv.extra["dsa_shard"] = FakeT((10, 1, 32, 32), 2), (2, 99, 2)
    g = torch.Generator().manual_seed(9)
    for t in (kv.extra["dsa"][34]["pools"], kv.extra["dsa"][34]["ring"]):
        t.mem.copy_(torch.randn(t.mem.shape, generator=g).to(torch.bfloat16))
    _prefill_alias(b, kv, rows=8)
    snaps = b.state_snapshot_tensors(kv)
    assert len(snaps) == 3
    plan = b.snapshot_state(kv, 6, n_tokens=4000)
    assert plan["pools"] is False and plan["units"] == 0 and [e[0] for e in plan["dsa"]] == ["L99.ring"]
    got = SM.StackBackend.collect_snapshot(plan, lambda i: snaps[i].mem)
    ref = b.export_kda_state(kv, 6, n_tokens=4000)
    dg, dr = (next(d["dsa"] for d in x if "dsa" in d) for x in (got, ref))
    assert dg["pools"] is False and dr["pools"] is False and sorted(dg["tensors"]) == sorted(dr["tensors"]) == ["L99.ring"]
    assert torch.equal(dg["tensors"]["L99.ring"].view(torch.int16), dr["tensors"]["L99.ring"].view(torch.int16))
    assert GV._states_pools_less(got) and GV._states_pools_less(ref)


def test_checkpoints_resumable_only_with_pooled_pages_on_a_paged_store():
    """kv_checkpoints_resumable (n29h7p2 1843: every hit retrieved, then refused): a paged store needs the pooled pages
    over the page-DMA; a dense store (or no DSA state) always resumes"""
    def adapter(paged, pooled, dma, dsa=True):
        a = types.SimpleNamespace(kv_cache="kv", _pooled_pages=pooled, _hit_dma=dma,
                                  generator=types.SimpleNamespace(backend=types.SimpleNamespace(pooled_paged=lambda kv: paged)))
        a._dsa_state = lambda: dsa
        return a

    R = GV.Glm5NextForCausalLM.kv_checkpoints_resumable
    assert R(adapter(False, False, None)) is True  # the dense store: the checkpoint carries the pooled rows
    assert R(adapter(True, False, None)) is False  # GLM53_KV_T2=0: nothing loads the hit blocks' pooled pages
    assert R(adapter(True, True, None)) is False
    assert R(adapter(True, True, "dma")) is True  # T2=1 + GLM53_KV_POOLED_PAGES=1
    assert R(adapter(True, False, None, dsa=False)) is True  # no DSA state: nothing to refuse
