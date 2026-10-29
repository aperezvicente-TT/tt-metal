"""Stage 12 lever 1 (b): a hit resuming at the checkpoint the device snapshot buffers still hold is restored ON DEVICE
(StackBackend.import_snapshot_state: the inverse of snapshot_state's SlotOps copies). Host only: the SlotOps launches
run against a host model of DRAM (kernels/head/slot_op.cpp kinds 0 / 1 / 2, byte for byte), so export slot A ->
import into slot B is checked bitwise against slot A's rows, every other row untouched; a later snapshot makes the
handle stale -> the host fallback."""

import numpy as np
import pytest
import ttnn

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM

CHUNK = 4


class FakeTensor:
    def __init__(self, mem, padded_shape, dtype):
        self.mem, self.padded_shape, self.shape, self.dtype = mem, list(padded_shape), list(padded_shape), dtype
        self.addr = mem.alloc(SM._pages(self), SM._pages_bytes(self))

    def buffer_address(self):
        return self.addr

    def pages(self) -> np.ndarray:
        return self.mem.bufs[self.addr]


class FakeDram:
    """address -> [pages, page bytes] uint8, with slot_op.cpp's semantics"""

    def __init__(self, seed=0):
        self.bufs, self.next, self.rng = {}, 0x10000, np.random.default_rng(seed)

    def alloc(self, pages, pb):
        a, self.next = self.next, self.next + 0x10000 + pages * pb
        self.bufs[a] = self.rng.integers(0, 256, size=(pages, pb), dtype=np.uint8)
        return a

    @staticmethod
    def _tile_off(r, c):
        return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)

    def run(self, ops):
        for kind, src, src0, dst, dst0, pages, pb, rows in ops:
            d = self.bufs[dst]
            assert d.shape[1] == pb and dst0 + pages <= d.shape[0], (kind, dst0, pages, d.shape)
            if kind == 0:
                d[dst0 : dst0 + pages] = 0
            elif kind == 1:
                s = self.bufs[src]
                assert s.shape[1] == pb and src0 + pages <= s.shape[0]
                d[dst0 : dst0 + pages] = s[src0 : src0 + pages]
            else:
                sr, dr, eb = rows >> 16, rows & 0xFFFF, pb // 1024
                for p in range(pages):
                    x = d[dst0 + p].view(np.uint32 if eb == 4 else np.uint16)
                    y = self.bufs[src][src0 + p].view(np.uint32 if eb == 4 else np.uint16) if src else None
                    for c in range(32):
                        x[self._tile_off(dr, c)] = y[self._tile_off(sr, c)] if src else 0


class FakeSlotOps:
    def __init__(self, mem):
        self.mem, self.calls = mem, []

    def run(self, ops):
        self.calls.append(list(ops))
        self.mem.run(ops)


def _model(mem, nl=3, dsa=True):
    """a StackBackend with nl KDA layers (S fp32 [CHUNK, 2, 64, 64] = 8 pages per slot; conv bf16 [1, 3, 32, 64]) and
    one DSA layer whose pooled store (paged by unit) and ring are per slot; the snapshot buffers as
    state_snapshot_tensors allocates them"""
    m = object.__new__(SM.StackBackend)
    m.chunk, m.counters, m.slot_ops = CHUNK, {"slot_ops": 0}, FakeSlotOps(mem)
    m.seq = [dict(L=i, sparse="kda") for i in range(nl)] + [dict(L=nl, sparse="dsa")]
    states = [(FakeTensor(mem, [CHUNK, 2, 64, 64], ttnn.float32), FakeTensor(mem, [1, 3, 32, 64], ttnn.bfloat16))
              for _ in range(nl)] + [None]  # fmt: skip
    kv = SM.StackCache.__new__(SM.StackCache)
    kv.states, kv.extra = [states], {}
    n_s, n_c = SM._pages(states[0][0]) // CHUNK, SM._pages(states[0][1])
    kv.extra["snap_s"] = FakeTensor(mem, [nl * n_s, 1, 32, 32], ttnn.float32)
    kv.extra["snap_c"] = FakeTensor(mem, [-(-nl // 32) * n_c, 1, 32, 32], ttnn.bfloat16)
    pools = FakeTensor(mem, [CHUNK * 8, 1, 32, 32], ttnn.bfloat16) if dsa else None
    ring = FakeTensor(mem, [CHUNK * 2, 1, 32, 32], ttnn.bfloat16) if dsa else None
    if dsa:
        kv.extra["snap_d"] = FakeTensor(mem, [64, 1, 32, 32], ttnn.bfloat16)
        kv.extra["dsa_stage"] = kv.extra["snap_d"]
    m._dsa_tensors = lambda kv_: [("L3.pools", pools, CHUNK, True), ("L3.ring", ring, CHUNK, False)] if dsa else []
    m.pooled_paged = lambda kv_: False
    return m, kv, dict(pools=pools, ring=ring)


def _rows(m, kv, slot, extra):
    """slot's bytes: S pages per layer, its conv row of every tile per layer, its DSA rows"""
    c, rd = divmod(slot, CHUNK)
    out = []
    for _i, _L, (S, C) in m._kda_layers(kv, c):
        n_s = SM._pages(S) // CHUNK
        out.append(S.pages()[rd * n_s : (rd + 1) * n_s].copy())
        eb = SM._pages_bytes(C) // 1024
        t = C.pages().view(np.uint32 if eb == 4 else np.uint16).reshape(SM._pages(C), 1024)
        out.append(np.stack([t[:, FakeDram._tile_off(rd, col)] for col in range(32)], axis=1).copy())
    for key in ("pools", "ring"):
        t = extra.get(key)
        if t is not None:
            per = SM._pages(t) // CHUNK
            out.append(t.pages()[slot * per : (slot + 1) * per].copy())
    return out


def _all(mem):
    return {a: b.copy() for a, b in mem.bufs.items()}


@pytest.mark.parametrize("pools", [False, True])
def test_hit_restores_the_exporters_rows_bitwise_from_the_device_snapshot(pools):
    mem = FakeDram()
    m, kv, extra = _model(mem)
    src, dst = 1, 3
    plan = m.snapshot_state(kv, src, n_tokens=256, pools=pools)
    want = _rows(m, kv, src, extra)
    h = SM.KDASnapshotHandle(m, plan, m._snap_seq)
    before = _all(mem)
    m.import_kda_state(kv, dst, h.states(lambda: pytest.fail("a valid snapshot must not take the host fallback")))
    got = _rows(m, kv, dst, extra)
    if not pools:  # pools=False (GLM53_KV_POOLED_PAGES): the pooled keys ride the pages; only the ring moves
        want, got = want[:-2] + want[-1:], got[:-2] + got[-1:]
    for a, b in zip(want, got):
        assert np.array_equal(a, b)
    # nothing but slot dst's rows changed: the other slots' S pages / conv rows / DSA rows and the snapshot itself
    for s in (0, 1, 2):
        for a, b in zip(_rows(m, kv, s, extra), _rows_from(before, m, kv, s, extra)):
            assert np.array_equal(a, b)
    for a in (kv.extra["snap_s"].addr, kv.extra["snap_c"].addr, kv.extra["snap_d"].addr):
        assert np.array_equal(mem.bufs[a], before[a])
    assert m.counters["kda_snapshot_imports"] == 1 and len(m.slot_ops.calls) == 2  # snapshot + import: one launch each


def _rows_from(snapshot, m, kv, slot, extra):
    saved = {a: b for a, b in m.slot_ops.mem.bufs.items()}
    m.slot_ops.mem.bufs = snapshot
    try:
        return _rows(m, kv, slot, extra)
    finally:
        m.slot_ops.mem.bufs = saved


def test_a_later_snapshot_makes_the_handle_stale_and_the_import_takes_the_host_fallback():
    mem = FakeDram(1)
    m, kv, _extra = _model(mem)
    plan = m.snapshot_state(kv, 0, n_tokens=128, pools=False)
    h = SM.KDASnapshotHandle(m, plan, m._snap_seq)
    m.snapshot_state(kv, 2, n_tokens=128, pools=False)  # another export rewrote the snapshot buffers
    assert not h.valid()

    class Fallback(Exception):
        pass

    def fallback():
        raise Fallback

    with pytest.raises(Fallback):
        m.import_kda_state(kv, 3, h.states(fallback))
    assert m.counters["kda_snapshot_import_stale"] == 1 and "kda_snapshot_imports" not in m.counters


def test_the_marker_reads_like_the_host_record_for_the_adapters_dsa_checks():
    """generator_vllm._hit_restart reads the states' DSA record: the marker carries collect_snapshot's pools flag"""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as GV

    mem = FakeDram(2)
    m, kv, _extra = _model(mem)
    for pools in (False, True):
        plan = m.snapshot_state(kv, 1, n_tokens=256, pools=pools)
        st = SM.KDASnapshotHandle(m, plan, m._snap_seq).states(lambda: None)
        assert GV._states_dsa(st) and GV._states_pools_less(st) == (not pools)
    m2, kv2, _ = _model(FakeDram(3), dsa=False)
    plan = m2.snapshot_state(kv2, 1, n_tokens=256, pools=False)
    assert not GV._states_dsa(SM.KDASnapshotHandle(m2, plan, m2._snap_seq).states(lambda: None))
