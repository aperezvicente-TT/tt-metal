"""One-replay store, M2 (doc/ttnn/09-vllm/perf/one_replay_store/design.md, Stage 12): the KDA checkpoint at the cut
exported from the in-replay snapshot staging instead of the slot's live rows. Host only: the real StackBackend export
code over fake tensors, with SlotOps kinds 1 (page copy) and 2 (row of every tile) executed on a host model of the
device memory and _stage_read reading it. The claims: the snapshot export returns the live export's layout bitwise
(sync pack and the async snapshot -> collect_snapshot path), it reads only what the staging contract names, the
(slot, cut) tag gates it and is consumed, the geometry is validated, and the generator / adapter route `at=`."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import ttnn

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as GV
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM

CHUNK, N_S, N_C = 4, 6, 3  # slots per state chunk, S pages per slot, conv tiles per layer


class FakeTensor:
    _next = [0x1000000]

    def __init__(self, pages, dtype, mem):
        self.addr = FakeTensor._next[0]
        FakeTensor._next[0] += 0x100000
        self.padded_shape = self.shape = [pages, 1, 32, 32]
        self.dtype = dtype
        np_dt = np.float32 if dtype == ttnn.float32 else np.uint16
        mem[self.addr] = np.zeros((pages, 32, 32), dtype=np_dt)

    def buffer_address(self):
        return self.addr


class HostSlotOps:
    """SlotOps kinds 0 / 1 / 2 (kernels/head/slot_op.cpp) on the host memory model"""

    def __init__(self, mem):
        self.mem, self.calls = mem, []

    def run(self, ops):
        self.calls.append(list(ops))
        for kind, src, s0, dst, d0, pages, _pb, rows in ops:
            d = self.mem[dst]
            if kind == 0:
                d[d0 : d0 + pages] = 0
            elif kind == 1:
                d[d0 : d0 + pages] = self.mem[src][s0 : s0 + pages]
            else:
                sr, dr = rows >> 16, rows & 0xFFFF
                for p in range(pages):
                    d[d0 + p][dr] = 0 if src == 0 else self.mem[src][s0 + p][sr]


def _torch(a):
    return torch.from_numpy(a.copy()) if a.dtype == np.float32 else torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)


def _backend(n_kda=3, stage_layers=2, snap=True):
    """a StackBackend over fake tensors: n_kda KDA layers (L = 0, 2, 4, ...) and one DSA-type layer (no recurrent state),
    2 state chunks of CHUNK slots; the export staging (pack) of stage_layers layers; the async snapshot buffers"""
    mem = {}
    b = SM.StackBackend.__new__(SM.StackBackend)
    b.chunk, b.tp, b.mesh = CHUNK, 1, None
    b.counters = {"slot_ops": 0}
    b.seq = [{"L": 2 * i} for i in range(n_kda)] + [{"L": 2 * n_kda}]
    b.slot_ops = HostSlotOps(mem)
    states = [[(FakeTensor(CHUNK * N_S, ttnn.float32, mem), FakeTensor(N_C, ttnn.bfloat16, mem)) for _ in range(n_kda)]
              + [None] for _c in range(2)]  # fmt: skip
    kv = SimpleNamespace(states=states, extra={})
    kv.extra["kda_stage_s"] = FakeTensor(stage_layers * N_S, ttnn.float32, mem)
    kv.extra["kda_stage_c"] = FakeTensor(N_C, ttnn.bfloat16, mem)
    if snap:
        kv.extra["snap_s"] = FakeTensor(n_kda * N_S, ttnn.float32, mem)
        kv.extra["snap_c"] = FakeTensor(-(-n_kda // 32) * N_C, ttnn.bfloat16, mem)
    b._stage_read = lambda stage: _torch(mem[stage.buffer_address()])[None]
    b._kv = kv
    return b, kv, mem


def _random_state(rng, n_kda):
    """per KDA layer: S [N_S, 32, 32] fp32, conv row [N_C, 32] bf16 bits"""
    return [(rng.standard_normal((N_S, 32, 32)).astype(np.float32),
             rng.integers(0, 1 << 16, size=(N_C, 32), dtype=np.uint16)) for _ in range(n_kda)]  # fmt: skip


def _put_live(kv, mem, slot, state):
    c, rd = divmod(slot, CHUNK)
    for (S, C), (s, crow) in zip([st for st in kv.states[c] if st is not None], state):
        mem[S.addr][rd * N_S : (rd + 1) * N_S] = s
        mem[C.addr][:, rd] = crow


def _put_stage(kv, mem, state, slot, cut, s0=2, crow0=5):
    """the staging contract: per layer its own S_t (state at pages [s0, s0 + N_S)) and one shared conv tensor with
    layer j at row crow0 + j"""
    C_t = FakeTensor(N_C, ttnn.bfloat16, mem)
    stage = []
    for j, (s, crow) in enumerate(state):
        S_t = FakeTensor(s0 + N_S + 1, ttnn.float32, mem)
        mem[S_t.addr][s0 : s0 + N_S] = s
        mem[C_t.addr][:, crow0 + j] = crow
        stage.append((S_t, s0, C_t, crow0 + j))
    kv.extra["kda_snap_stage"] = stage
    kv.extra["kda_snap_at"] = (slot, cut)
    return stage


def _pack_equal(a, b):
    return (a["kind"] == b["kind"] and a["layers"] == b["layers"] and a["n_s"] == b["n_s"]
            and torch.equal(a["S"], b["S"]) and torch.equal(a["C"].view(torch.int16), b["C"].view(torch.int16)))  # fmt: skip


def test_live_pack_reads_the_slots_rows():
    """the refactored live path: pack S = the slot's N_S pages, C = the slot's row of every conv tile"""
    rng = np.random.default_rng(1)
    b, kv, mem = _backend()
    x = _random_state(rng, 3)
    _put_live(kv, mem, 5, x)
    pack = b.export_kda_pack(kv, 5)
    assert pack["layers"] == [0, 2, 4] and pack["n_s"] == N_S
    for j, (s, crow) in enumerate(x):
        assert torch.equal(pack["S"][0, j], torch.from_numpy(s))
        assert torch.equal(pack["C"][0, j].view(torch.int16), torch.from_numpy(crow.view(np.int16)))


@pytest.mark.parametrize("stage_layers", [1, 2, 3])  # the pack's rounds of GLM53_KDA_EXPORT_STAGE_LAYERS
def test_snapshot_pack_is_the_live_pack_of_the_same_state(stage_layers):
    """slot 5 holds its END state Y (live rows); the staging holds the state at the cut X. The snapshot export equals
    the live export of a slot holding X, bitwise, and not the slot's own live state."""
    rng = np.random.default_rng(2)
    b, kv, mem = _backend(stage_layers=stage_layers)
    x, y = _random_state(rng, 3), _random_state(rng, 3)
    _put_live(kv, mem, 2, x)  # the reference: X in another slot's live rows
    _put_live(kv, mem, 5, y)
    ref = b.export_kda_pack(kv, 2)
    own = b.export_kda_pack(kv, 5)
    _put_stage(kv, mem, x, slot=5, cut=1920)
    snap = b.export_kda_pack(kv, 5, n_tokens=1920, source="snapshot")
    assert _pack_equal(snap, ref) and not _pack_equal(snap, own)
    assert kv.extra["kda_snap_at"] is None and b.counters["kda_snapshot_exports"] == 1  # consumed
    # every snapshot copy read the staging tensors named by the contract, never the slot's rows
    stage_addrs = {S_t.addr for S_t, *_ in kv.extra["kda_snap_stage"]} | {kv.extra["kda_snap_stage"][0][2].addr}
    live_addrs = {t.addr for st in kv.states[1] if st is not None for t in st}
    snap_calls = b.slot_ops.calls[-(-(-3 // stage_layers) + 1) :]  # this export's S rounds + its one conv round
    srcs = {op[1] for call in snap_calls for op in call}
    assert srcs <= stage_addrs and not srcs & live_addrs


def test_async_snapshot_collects_the_live_layout():
    """the async path: snapshot_state(source="snapshot") copies the staging into the snapshot buffers;
    collect_snapshot over those buffers == export_kda_pack of the same state (the export_kda_state layout)"""
    rng = np.random.default_rng(3)
    b, kv, mem = _backend()
    x, y = _random_state(rng, 3), _random_state(rng, 3)
    _put_live(kv, mem, 2, x)
    ref = b.export_kda_pack(kv, 2)
    _put_live(kv, mem, 6, y)
    _put_stage(kv, mem, x, slot=6, cut=4096)
    plan = b.snapshot_state(kv, 6, 4096, pools=False, source="snapshot")
    assert plan is not None and kv.extra["kda_snap_at"] is None
    bufs = [kv.extra["snap_s"], kv.extra["snap_c"]]
    got = SM.StackBackend.collect_snapshot(plan, lambda i: _torch(mem[bufs[i].addr])[None])
    assert len(got) == 1 and _pack_equal(got[0]["kda"], ref)


def test_live_snapshot_state_is_unchanged():
    """source="live" (today's async export) still snapshots the slot's own rows"""
    rng = np.random.default_rng(4)
    b, kv, mem = _backend()
    y = _random_state(rng, 3)
    _put_live(kv, mem, 6, y)
    ref = b.export_kda_pack(kv, 6)
    plan = b.snapshot_state(kv, 6, 4096, pools=False)
    bufs = [kv.extra["snap_s"], kv.extra["snap_c"]]
    got = SM.StackBackend.collect_snapshot(plan, lambda i: _torch(mem[bufs[i].addr])[None])
    assert _pack_equal(got[0]["kda"], ref)


def test_the_tag_gates_the_snapshot_export():
    rng = np.random.default_rng(5)
    b, kv, mem = _backend()
    x = _random_state(rng, 3)
    with pytest.raises(ValueError, match="no staging"):
        b.export_kda_pack(kv, 5, n_tokens=1920, source="snapshot")
    _put_stage(kv, mem, x, slot=5, cut=1920)
    for slot, cut in ((4, 1920), (5, 1792), (5, None)):
        assert not b.kda_snapshot_holds(kv, slot, cut or 0)
        with pytest.raises(ValueError, match="does not hold"):
            b.export_kda_pack(kv, slot, n_tokens=cut, source="snapshot")
        with pytest.raises(ValueError, match="does not hold"):
            b.snapshot_state(kv, slot, cut, pools=False, source="snapshot")
    assert kv.extra["kda_snap_at"] == (5, 1920) and b.kda_snapshot_holds(kv, 5, 1920)  # a refusal consumes nothing
    b.export_kda_pack(kv, 5, n_tokens=1920, source="snapshot")
    assert not b.kda_snapshot_holds(kv, 5, 1920)
    with pytest.raises(ValueError, match="does not hold"):  # one export per snapshot
        b.export_kda_pack(kv, 5, n_tokens=1920, source="snapshot")
    with pytest.raises(ValueError, match="not in"):
        b.export_kda_pack(kv, 5, n_tokens=1920, source="stale")


@pytest.mark.parametrize("bad", ["s_dtype", "c_dtype", "c_pages", "s_range", "c_row", "layers"])
def test_the_staging_geometry_is_checked(bad):
    rng = np.random.default_rng(6)
    b, kv, mem = _backend()
    stage = _put_stage(kv, mem, _random_state(rng, 3), slot=5, cut=1920)
    S_t, s0, C_t, crow = stage[1]
    if bad == "s_dtype":
        stage[1] = (FakeTensor(s0 + N_S, ttnn.bfloat16, mem), s0, C_t, crow)
    elif bad == "c_dtype":
        stage[1] = (S_t, s0, FakeTensor(N_C, ttnn.float32, mem), crow)
    elif bad == "c_pages":
        stage[1] = (S_t, s0, FakeTensor(N_C + 1, ttnn.bfloat16, mem), crow)
    elif bad == "s_range":
        stage[1] = (S_t, s0 + 2, C_t, crow)
    elif bad == "c_row":
        stage[1] = (S_t, s0, C_t, 32)
    else:
        stage.pop()
    with pytest.raises(ValueError, match="staging"):
        b.export_kda_pack(kv, 5, n_tokens=1920, source="snapshot")
    assert kv.extra["kda_snap_at"] == (5, 1920)


def test_export_kda_state_routes_the_source():
    rng = np.random.default_rng(7)
    b, kv, mem = _backend()
    x = _random_state(rng, 3)
    _put_live(kv, mem, 2, x)
    ref = b.export_kda_pack(kv, 2)
    _put_stage(kv, mem, x, slot=5, cut=1920)
    out = b.export_kda_state(kv, 5, n_tokens=1920, pools=False, source="snapshot")
    assert len(out) == 1 and _pack_equal(out[0]["kda"], ref)
    kv.extra["kda_stage_s"] = None  # no KDA pack: the whole-chunk gather path has no snapshot variant
    _put_stage(kv, mem, x, slot=5, cut=1920)
    with pytest.raises(RuntimeError, match="KDA pack"):
        b.export_kda_state(kv, 5, n_tokens=1920, pools=False, source="snapshot")


def test_generator_passes_the_source_only_to_a_backend_that_takes_it():
    seen = []

    def new(kv, slot, n_tokens=None, pools=True, source="live"):
        seen.append((slot, n_tokens, pools, source))
        return "states"

    def old(kv, slot, n_tokens=None, pools=True):
        return "old"

    assert G._export_state(new, None, 3, 1920, pools=False, source="snapshot") == "states"
    assert seen[-1] == (3, 1920, False, "snapshot")
    assert G._export_state(new, None, 3, 2048, pools=False) == "states" and seen[-1] == (3, 2048, False, "live")
    with pytest.raises(ValueError, match="cannot export"):
        G._export_state(old, None, 3, 1920, source="snapshot")
    with pytest.raises(ValueError, match="cannot export"):
        G._export_state(new, None, 3, None, source="snapshot")


def _adapter(next_pos, holds=True):
    a = GV.Glm5NextForCausalLM.__new__(GV.Glm5NextForCausalLM)
    calls = []
    backend = SimpleNamespace(kda_snapshot_holds=lambda kv, g, n: holds and (g, n) == (7, 1920))
    gen = SimpleNamespace(
        backend=backend,
        export_slot_state=lambda kv, g, n_tokens, pools, source: calls.append(("sync", g, n_tokens, source)) or "S",
        export_slot_state_async=lambda kv, g, n_tokens, dma, pools, source: calls.append(("async", g, n_tokens, source))
        or (lambda: "S"),
    )  # fmt: skip
    a.generator, a.gen_slot, a._pooled_pages = gen, {3: 7}, True
    a.slot_table = SimpleNamespace(slots={7: SimpleNamespace(next_pos=next_pos)})
    a.counters = {"kda_exports": 0}
    a.kv_cache = kv = object()
    a._hit_dma = SimpleNamespace(state=[1])
    return a, kv, calls


def test_adapter_at_routes_live_and_snapshot():
    a, kv, calls = _adapter(next_pos=2000)  # an unaligned storing prompt after its one replay
    assert a.export_kda_state(kv, 3, at=1920) == {"next_pos": 1920, "states": "S"}
    assert calls[-1] == ("sync", 7, 1920, "snapshot") and a.counters["kda_exports_snapshot"] == 1
    assert a.export_kda_state_async(kv, 3, at=1920)() == {"next_pos": 1920, "states": "S"}
    assert calls[-1] == ("async", 7, 1920, "snapshot")
    assert a.kda_snapshot_ready(kv, 3, 1920) and not a.kda_snapshot_ready(kv, 3, 1792)
    for at in (1900, 2048, 0, -64):  # off the page / past the slot / not a cut
        with pytest.raises(ValueError, match="page boundary"):
            a.export_kda_state(kv, 3, at=at)
    with pytest.raises(ValueError, match="page boundary"):  # the live state of an unaligned slot is no checkpoint
        a.export_kda_state(kv, 3)
    b, kv2, calls2 = _adapter(next_pos=2048)
    assert b.export_kda_state(kv2, 3) == {"next_pos": 2048, "states": "S"} and calls2[-1] == ("sync", 7, 2048, "live")
    assert b.export_kda_state(kv2, 3, at=2048)["next_pos"] == 2048 and calls2[-1][3] == "live"


def test_a_kda_only_checkpoint_of_a_per_slot_dsa_cache_is_refused(monkeypatch):
    """2026-10-24 selfhit (doc/ttnn/12-fast-start/selfhit): without the DSA export staging the checkpoint carried no
    per-slot pooled keys, and a hit into another slot read zeros at L3. Now: refused when a per-slot (unit-paged) pooled
    store exists and the export would include pools; allowed with pools=False (they ride the LMCache pages) and when
    only the ring (record) is per slot; exported when the staging exists"""
    rng = np.random.default_rng(8)
    b, kv, mem = _backend()
    _put_live(kv, mem, 5, _random_state(rng, 3))
    pools_t, ring_t = FakeTensor(8, ttnn.bfloat16, mem), FakeTensor(8, ttnn.bfloat16, mem)
    monkeypatch.setattr(b, "_dsa_tensors", lambda kv_: [("L6.pools", pools_t, 4, True), ("L6.ring", ring_t, 4, False)])
    monkeypatch.setattr(b, "export_dsa_state", lambda kv_, slot, n, pools=True: {"rows": (slot, n, pools)})
    with pytest.raises(RuntimeError, match="ensure_state_export"):
        b.export_kda_state(kv, 5, n_tokens=1920)
    assert len(b.export_kda_state(kv, 5, n_tokens=1920, pools=False)) == 1  # KDA only: the pools ride the pages
    monkeypatch.setattr(b, "_dsa_tensors", lambda kv_: [("L6.ring", ring_t, 4, False)])  # paged pools: ring only
    assert len(b.export_kda_state(kv, 5, n_tokens=1920)) == 1
    monkeypatch.setattr(b, "_dsa_tensors", lambda kv_: [("L6.pools", pools_t, 4, True), ("L6.ring", ring_t, 4, False)])
    kv.extra["dsa_stage"] = FakeTensor(16, ttnn.bfloat16, mem)
    out = b.export_kda_state(kv, 5, n_tokens=1920)
    assert out[-1] == {"dsa": {"rows": (5, 1920, True)}}


@pytest.mark.parametrize("paged,env,want", [(False, None, True), (True, None, False), (True, "256", True), (False, "0", True)])
def test_the_dsa_export_staging_comes_with_a_per_slot_pooled_cache(monkeypatch, paged, env, want):
    """allocate_kv_cache (via _dsa_stage_with_cache): per-slot pooled keys (PAGED_POOLS=0) always get the staging with
    the cache, before any capture; paged pools only when GLM53_DSA_EXPORT_STAGE_TILES is set (else on demand)"""
    b, kv, _mem = _backend()
    calls = []
    monkeypatch.setattr(SM, "dsa_paged_pools", lambda: paged)
    monkeypatch.setattr(b, "_ensure_dsa_stage", lambda kv_: calls.append(kv_))
    if env is None:
        monkeypatch.delenv("GLM53_DSA_EXPORT_STAGE_TILES", raising=False)
    else:
        monkeypatch.setenv("GLM53_DSA_EXPORT_STAGE_TILES", env)
    b._dsa_stage_with_cache(kv)
    assert (calls == [kv]) == want


# ---------------------------------------------------------------- one-replay store A, M1 (the cut snapshot's plumbing)
def test_the_snapshot_scalar_marks_only_the_chunk_that_covers_the_cut():
    from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import kda_snap_rows

    assert kda_snap_rows(1920, 0, 2000, 2048) == (True, 1920)  # one replay of a 2,000-token storing prompt
    assert kda_snap_rows(1920, 0, 1920, 2048) == (True, 1920)  # a cut at the chunk's last row
    assert kda_snap_rows(None, 0, 2000, 2048) == (False, 32)  # no cut requested: the no-op one-chunk snapshot
    assert kda_snap_rows(1920, 2048, 512, 2048) == (False, 32)  # the cut lies in an earlier chunk
    assert kda_snap_rows(8064, 8192 - 2048, 2048, 2048) == (True, 1920)  # a later chunk of a long prompt
    with pytest.raises(ValueError, match="rows"):
        kda_snap_rows(1900, 0, 2000, 2048)  # off the 32-row chunk grid
    with pytest.raises(ValueError, match="rows"):
        kda_snap_rows(4096, 0, 4096, 2048)  # past the bucket


def test_the_generator_passes_the_cut_to_the_covering_chunk_only(monkeypatch):
    """_prefill_user: snap_at reaches backend.prefill_chunk for the plan chunk that covers the cut, nowhere else"""
    seen = []

    class B:
        prefill_chunk_tokens, prefill_align, continuation_align, block_size = 2048, 64, 64, 64
        max_seq_len = supported_context = 1 << 20
        kda_checkpoint_in_replay = True
        vocab_size = 8

        def prefill_chunk(self, chunk, **kw):
            seen.append((kw["chunk_start"], kw["n_valid"], kw.get("snap_at")))
            return torch.zeros(1, 8)

    gen = G.Glm53Generator.__new__(G.Glm53Generator)
    gen.backend, gen.host_sampling, gen.counters = B(), True, {"prefill_runs": 0, "prefill_run_chunks_max": 0,
                                                              "prefill_chunks": 0, "prefill_chunks_headless": 0,
                                                              "prefill_logits_readback_rows": 0}  # fmt: skip
    monkeypatch.setattr(gen, "_slice_safe", lambda plans: plans, raising=False)
    st = SimpleNamespace(seq_len=0)
    b = SimpleNamespace(slots={0: st}, max_seq_len=1 << 20, coverage={0: 1 << 20}, kv_cache=None, io=None)
    try:
        gen._prefill_user(b, 0, torch.arange(4000), start_pos=0, snap_at=1920)
    except (AttributeError, KeyError, TypeError) as ex:  # the fakes stop after the chunk calls (bookkeeping after them)
        print("fake stop:", repr(ex)[:200])
    assert (0, 2048, 1920) in seen and all(s_ is None for c0, _n, s_ in seen if c0 != 0), seen


@pytest.mark.parametrize("prefill,snap,want", [("tp", "1", True), ("tp", "0", False), ("decode", "1", False), (None, "1", False)])
def test_in_replay_answers_from_the_config_before_warmup(monkeypatch, prefill, snap, want):
    """kda_checkpoint_in_replay before prepare_prefill (no tpp): the vLLM connector's from_runner guard runs after
    initialize_kv_cache and BEFORE warmup phase 1 builds tpp (the cf3h-g mock's death, program 2026-10-24T22:41:57Z),
    so the property answers from GLM53_STACK_PREFILL / GLM53_PF_KDA_SNAPSHOT; once tpp exists, tpp decides"""
    for k, v in (("GLM53_STACK_PREFILL", prefill), ("GLM53_PF_KDA_SNAPSHOT", snap)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    b = SM.StackBackend.__new__(SM.StackBackend)
    assert b.kda_checkpoint_in_replay is want
    b.tpp = SimpleNamespace(kda_snapshot=not want)  # built: the prefill's own flag wins over the env
    assert b.kda_checkpoint_in_replay is (not want)
