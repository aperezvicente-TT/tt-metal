# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-python unit tests of the PMK stack driver (tt/pmk_stack.py; no device, no ttnn import on the tested paths):
the 45-layer kind table against the HF config's layer_kind, the prefill -> decode state-page layouts (bitwise
shape / offset equality of the three writers / readers: state_io.cpp STATE_OUT, cache_state.cpp, _state_ops), the
inter-program DRAM stream bytes and the TP8 time model's arithmetic. Run: python -m pytest tests/s16/test_pmk_stack.py -q"""

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_DSA, SPARSE_KDA, Glm5NextTextConfig


def _cfg():
    return Glm5NextTextConfig()  # the checkpoint's defaults: 45 layers, i % 4 == 3 -> indexed attention, 3 dense MLPs


def test_layer_schedule_counts_and_order():
    cfg = _cfg()
    sched = PS.layer_schedule(cfg)
    assert len(sched) == 45 and [L for L, _ in sched] == list(range(45))
    assert PS.kind_counts(cfg) == {SPARSE_KDA: 31, SPARSE_DSA: 11, DENSE_KDA: 3}
    kinds = dict(sched)
    assert all(kinds[L] == DENSE_KDA for L in (0, 1, 2))
    assert all(kinds[L] == SPARSE_DSA for L in range(3, 45, 4))
    assert all(kinds[L] == SPARSE_KDA for L in range(45) if L > 2 and L % 4 != 3)
    # the same map the served stack uses (tt/stack_model.layer_kind)
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind

    assert all(layer_kind(cfg, L) == k for L, k in sched)


def test_kind_builders_cover_every_kind():
    assert set(PS.KIND_BUILDERS) == {SPARSE_KDA, SPARSE_DSA, DENSE_KDA}


@pytest.mark.parametrize("tp", [8, 2])
@pytest.mark.parametrize("chunk", [4, 32])
def test_state_layouts_bitwise_equal(tp, chunk):
    """the slot's S pages and conv rows: the PMK's P9 (cache_state.cpp) == the op path's StateIO (state_io.cpp
    STATE_OUT) == what the decode stack reads (StackBackend._state_ops), for every slot of a chunk"""
    lay = PS.state_layout(tp, chunk, 0)
    n_s, n_c = lay["n_s"], lay["n_c"]
    assert n_s == 16 * (64 // tp) and n_c == 3 * (3 * (64 // tp) * 128 // 32)
    for slot in range(chunk * 2):  # two state chunks
        a = PS.state_io_layout(n_s, n_c, chunk, slot)
        b = PS.cache_state_layout(n_s, n_c, chunk, slot)
        d = PS.decode_state_ops(n_s, n_c, chunk, slot)
        s = PS.state_layout(tp, chunk, slot)
        assert a["cidx"] == b["cidx"] == d["cidx"] == s["cidx"] == slot // chunk
        assert a["rd"] == b["rd"] == d["rd"] == s["rd"] == slot % chunk
        # S: the same contiguous page run, bitwise the same byte ranges
        assert a["s"] == b["s"] == d["s"] == [off for off, _ in s["s_bytes"]]
        assert len(a["s"]) == n_s and a["s"][0] == (slot % chunk) * n_s * 4096
        # C: every conv tile, the same 64 bytes (2 x 32-B face rows) of row rd
        assert [p for p, _ in a["c"]] == [p for p, _ in b["c"]] == d["c_tiles"] == list(range(n_c))
        for (pa, ba), (pb, bb), (pc, lo, hi) in zip(a["c"], b["c"], s["c_rows"]):
            assert pa == pb == pc
            # the bytes each writer covers (state_io: 32 x uint16, cache_state: 2 x 8 x uint32): the same 64 bytes
            cov_a = {x + i for x in ba for i in range(2)}
            cov_b = {x + i for x in bb for i in range(4)}
            cov_s = set(range(lo, lo + 32)) | set(range(hi, hi + 32))
            assert cov_a == cov_b == cov_s and len(cov_s) == 64, (slot, pa)
        assert d["c_row"] == slot % chunk


def test_state_layout_bytes_tp8_tp2():
    s8, s2 = PS.state_layout(8, 4, 1), PS.state_layout(2, 4, 1)
    assert s8["s_total"] == 8 * 16 * 4096 == 524288  # cache README: 524,288 S bytes per sparse_kda layer at TP8
    assert s8["n_c"] == 288 and s2["n_c"] == 1152  # 288 / 1,152 conv tiles (cache README)
    assert s2["s_total"] == 2 * 2**20


def test_time_model_tp8():
    tm = PS.time_model(counts={SPARSE_KDA: 31, SPARSE_DSA: 11, DENSE_KDA: 3})
    per = tm["per_layer_us"]
    assert per[SPARSE_KDA] == sum(v[0] for v in PS.TIME_MODEL_TP8.values())
    assert tm["ttft_us"] == 31 * per[SPARSE_KDA] + 11 * per[SPARSE_DSA] + 3 * per[DENSE_KDA] + PS.EMBED_HEAD_US
    assert tm["measured_at_tp8"] == []  # nothing is measured at TP8 yet: the projection is a model
    assert 0.45 < tm["moe_share"] < 0.6 and "P14" in tm["dominant"]
    assert tm["contract_ttft_ms"] < tm["ttft_ms"] < tm["p4_bar_ms"]  # above the contract's 102 ms, below the 300 ms bar


def test_streams_and_geometry_without_device():
    """streams() / geometry() need only PmkGeom: a TP8 and a TP2 shim"""

    class Pf:
        def __init__(self, tp):
            self.cfg, self.ndev, self.buckets, self.layers = _cfg(), tp, (2048,), []
            self.pmk_layers = {}

        def mode(self, c):
            return "sp"

    for tp, sp_rows, res_mb in ((8, 256, 16.0), (2, 1024, 64.0)):
        st = PS.PmkStack.__new__(PS.PmkStack)
        st.pf = Pf(tp)
        g = st.geometry(2048)
        assert (g["sp_rows"], g["residual_mb"], g["heads_r"]) == (sp_rows, res_mb, 64 // tp)
        s = st.streams(2048)
        assert s["residual_bytes"] == sp_rows * 16384 * 4 == s["chain_total_bytes"]
        assert s["kda_scratch_bytes"] == 16 * (64 // tp) * 4096 + 3 * (3 * (64 // tp) * 128 // 32) * 2048


# ------------------------------------------------------------------------------------------------ long prompts
def test_bucket_geometry_large_buckets():
    g8 = PS.bucket_geometry(8192, 8)
    assert (g8["sp_rows"], g8["heads_r"], g8["sp_blocks"], g8["c_blocks"]) == (1024, 8, 32, 256)
    assert g8["residual_bytes"] == 64 * 2**20 and g8["xs_store_bytes"] == 64 * 2**20
    # the TP2 geometry at 2,048 has the same per-chip rows and chunk-store bytes (8 heads x 8,192 == 32 heads x 2,048)
    g2 = PS.bucket_geometry(2048, 2)
    assert g2["sp_rows"] == g8["sp_rows"] and g2["chunk_store_bytes"] == g8["chunk_store_bytes"]
    assert PS.bucket_geometry(16384, 8)["sp_rows"] == 2048


def test_cost_model_weight_term_amortises():
    """bytes per token of the expert stream fall 4x / 8x with the bucket; the indexer term is O(position)"""
    c2, c8, c16 = (PS.chunk_cost_us(r, 2048) for r in (2048, 8192, 16384))
    assert c2["weight_us"] == c8["weight_us"] == c16["weight_us"]  # paid once per replay
    assert c8["us_per_token"] < c2["us_per_token"] and c16["us_per_token"] < c8["us_per_token"]
    assert abs(c2["weight_us"] / c2["rows"] / (c8["weight_us"] / c8["rows"]) - 4.0) < 1e-9
    assert c2["indexer_us"] == 0.0  # dense regime at position 2,048
    far = PS.chunk_cost_us(8192, 1_048_576)
    near = PS.chunk_cost_us(8192, 131_072)
    assert far["indexer_us"] > 7 * near["indexer_us"]  # linear in the (mid-chunk) position
    lc = PS.layer_cost_us(PS.SPARSE_KDA, 2048, 2048)
    assert lc["expert_w_us"] == PS.EXPERT_BYTES_TP8 / (PS.EXPERT_GBPS_TARGET * 1e3)
    assert PS.layer_cost_us(PS.SPARSE_KDA, 2048, 2048, expert_gbps=PS.EXPERT_GBPS_FLOOR)["p14_us"] > lc["p14_us"]


def test_prefill_ttft_host_term_and_loop():
    t1 = PS.prefill_ttft(1_048_576, 2048, 1)
    t16 = PS.prefill_ttft(1_048_576, 2048, 16)
    assert t1["chunks"] == 512 and t1["traces"] == 512 and t16["traces"] == 32
    assert t1["device_us"] == t16["device_us"] and t1["host_us"] == 512 * PS.HOST_US_PER_TRACE
    assert t16["host_us"] == 32 * PS.HOST_US_PER_TRACE
    t8 = PS.prefill_ttft(1_048_576, 8192, 16)
    assert t8["chunks"] == 128 and t8["traces"] == 8 and t8["device_us"] < t1["device_us"]
    rows = PS.ttft_table()
    assert len(rows) == 27 and all(r["ttft_us"] == r["device_us"] + r["host_us"] for r in rows)
    # 1M at 8,192: the O(L^2) indexer term exceeds the (amortised) weight term; 32K at 2,048: the weight stream is
    # a third of the device time; the per-row compute term (the phase targets) is the largest everywhere
    big = PS.prefill_ttft(1_048_576, 8192, 16)
    assert big["indexer_share"] > big["weight_share"] and big["indexer_share"] > 0.15
    assert PS.prefill_ttft(32768, 2048, 1)["weight_share"] > 0.3


def test_chunk_loop_row_words_and_plan():
    class St:
        pmk_buckets = (2048,)

    cl = PS.ChunkLoop.__new__(PS.ChunkLoop)
    cl.c, cl.n = 2048, 4
    w = cl.row_words(2, slot=5, chunk_start=4096, n_valid=2048, t=8192)
    assert w[:6] == [5, 0, 4096, 2048, 8192, 2] and len(w) == PS.CHUNK_ARG_WORDS
    assert cl.row_words(0, slot=5, chunk_start=0, n_valid=100, t=0)[1] == 1  # fresh
    plan = cl.prompt_plan(1_048_576)
    assert plan == dict(
        L=1_048_576,
        bucket=2048,
        N=4,
        chunks=512,
        loop_traces=128,
        single_chunks=0,
        traces=128,
        head_on="one extra head-only replay of the last row",
        host_round_trips_removed=384,
    )
    assert cl.prompt_plan(10_000)["single_chunks"] == 1 and cl.prompt_plan(10_000)["traces"] == 2


# ------------------------------------------------------------------------------------- the served stack (GLM53_PF_MODE=pmk)
def test_slot_modes_from_env():
    assert PS.slot_modes_from_env({}) == {s: "pmk" for s in PS.SLOTS}
    assert PS.slot_modes_from_env({"GLM53_PMK_SLOTS": "none"}) == {s: "op" for s in PS.SLOTS}
    m = PS.slot_modes_from_env({"GLM53_PMK_SLOTS": "mhc_pre, attn"})
    assert m == dict(mhc_pre="pmk", attn="pmk", mhc_mid="op", ffn="op", mhc_post="op")
    with pytest.raises(AssertionError):
        PS.slot_modes_from_env({"GLM53_PMK_SLOTS": "router"})


@pytest.mark.parametrize("tp", [8, 2])
def test_served_layer_plan_per_kind(tp):
    """which slots each kind runs on the PMK under 'auto', and why the rest fall to the op graph: sparse_kda = A1 /
    A2 + A3 + A4 / A5 / B (5 + 1 programs), the mhc post pending (op on A5's pages); dense_kda = program A only (5
    programs, the dense MLP pending); sparse_dsa = A1 + A5 (the mhc-only program A, 2 programs) around the op attention,
    every slot pending under GLM53_PMK_DSA_MHC=0"""
    auto = PS.slot_modes_from_env({})
    kda = PS.served_layer_plan(SPARSE_KDA, tp, auto)
    assert kda["modes"] == dict(mhc_pre="pmk", attn="pmk", mhc_mid="pmk", ffn="pmk", mhc_post="op")
    assert kda["programs"] == 6 and set(kda["pending"]) == {"mhc_post"}
    assert "attn" not in kda["notes"]  # A4 carries the RS at both TPs (pair RSR, TP8 RS8): no op reduce-scatter filler
    dense = PS.served_layer_plan(DENSE_KDA, tp, auto)
    assert dense["modes"] == dict(mhc_pre="pmk", attn="pmk", mhc_mid="pmk", ffn="op", mhc_post="op")
    assert dense["programs"] == 5 and set(dense["pending"]) == {"ffn", "mhc_post"}
    dsa = PS.served_layer_plan(SPARSE_DSA, tp, auto, dsa_mhc=True)
    assert dsa["modes"] == dict(mhc_pre="pmk", attn="op", mhc_mid="pmk", ffn="op", mhc_post="op")
    assert dsa["programs"] == 2 and set(dsa["pending"]) == {"attn", "ffn", "mhc_post"}
    dsa0 = PS.served_layer_plan(SPARSE_DSA, tp, auto, dsa_mhc=False)
    assert dsa0["modes"] == {s: "op" for s in PS.SLOTS} and dsa0["programs"] == 0 and set(dsa0["pending"]) == set(PS.SLOTS)
    # without the XSL slices program B has no inputs: the ffn slot falls to the op graph
    no_xsl = PS.served_layer_plan(SPARSE_KDA, tp, auto, xsl=False)
    assert no_xsl["modes"]["ffn"] == "op" and "ffn" in no_xsl["pending"]
    # GLM53_PMK_SLOTS=none: the chain object with every slot on the op graph (the unit's reference)
    none = PS.served_layer_plan(SPARSE_KDA, tp, PS.slot_modes_from_env({"GLM53_PMK_SLOTS": "none"}))
    assert none["programs"] == 0 and none["pending"] == {}


def test_served_stack_45_layer_counts():
    """the 45-layer stack under 'auto': programs per replay and the DRAM the chain adds per chip (the second copies
    of the projection / expert weights dominate: the image must carry the stream layouts before n = 45)"""
    cfg = _cfg()
    auto = PS.slot_modes_from_env({})
    progs = sum(PS.served_layer_plan(k, 8, auto, dsa_mhc=True)["programs"] for _, k in PS.layer_schedule(cfg))
    assert progs == 31 * 6 + 3 * 5 + 11 * 2 == 223
    progs0 = sum(PS.served_layer_plan(k, 8, auto, dsa_mhc=False)["programs"] for _, k in PS.layer_schedule(cfg))
    assert progs0 == 31 * 6 + 3 * 5 + 11 * 0 == 201
    d = PS.served_stack_dram_bytes(8, 2048)
    assert d["per_layer_bytes"][SPARSE_DSA] == 0
    assert d["program_b_stream_bytes"] == 288 * (4096 * 2 * 256 + 256 * 4096) // 2  # 453 MB of bfp4 per chip
    assert d["stack_45_gib"] > 12  # 31 x the expert stream copy: does not fit beside the 25.9 GB served image
    assert d["residual_bytes"] == 256 * 16384 * 4


def test_served_stack_env_keys():
    env = PS.served_stack_env(8)
    assert env["GLM53_PF_MODE"][0] == "pmk" and env["GLM53_PMK_SLOTS"][0] == "auto"
    required = [k for k, v in env.items() if v[1].startswith("REQUIRED")]
    assert set(required) == {"GLM53_PF_MHC_PRE", "GLM53_TP_XSL", "GLM53_TP_XSL_SLICED", "GLM53_PF_MHC_POST"}


def test_served_stack_router_in_key():
    env = PS.served_stack_env(8)
    assert env["GLM53_PMK_ROUTER_IN"][0] == "fp32"


def test_pack_streams_per_chip_distinct_and_own():
    """program B's stream tensor must carry chip k's OWN slices (Stage 12: the replicated chip-0 copy was the 00:04Z L4
    defect): the per-chip packing equals pack_expert_streams of that chip's slices and differs across chips"""
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2

    torch.manual_seed(0)
    E, H, W = 16, 64, 32  # a small expert set: gate|up [H, 2W] per expert, down [W, H]
    chips = []
    for k in range(2):
        g = [torch.randn(H, 2 * W).to(torch.bfloat16) for _ in range(E)]
        d = [torch.randn(W, H).to(torch.bfloat16) for _ in range(E)]
        chips.append((g, d))
    stacked, m0, pages = PS.pack_streams_per_chip([c[0] for c in chips], [c[1] for c in chips])
    assert stacked.shape[0] == 2 and not torch.equal(stacked[0], stacked[1])
    for k in range(2):
        t_, m0_, p_ = M2.pack_expert_streams(*chips[k])
        assert torch.equal(stacked[k], t_.reshape(-1, 32)) and list(m0_) == m0 and p_ == pages


def test_served_stack_signature_keys():
    a = PS.served_stack_signature({"GLM53_PF_MODE": "pmk"}, tp=8)
    b = PS.served_stack_signature({"GLM53_PF_MODE": "pmk", "GLM53_PMK_SLOTS": "mhc_pre"}, tp=8)
    c = PS.served_stack_signature({"GLM53_PF_MODE": "op"}, tp=8)
    d = PS.served_stack_signature({"GLM53_PF_MODE": "pmk", "GLM53_PMK_LAYERS": "5"}, tp=8)
    assert len({a["signature"], b["signature"], c["signature"], d["signature"]}) == 4
    assert a == PS.served_stack_signature({"GLM53_PF_MODE": "pmk"}, tp=8)  # deterministic
    assert a["body"]["modes"] == {s: "pmk" for s in PS.SLOTS} and c["body"]["modes"] == {s: "op" for s in PS.SLOTS}


def test_b_config_of_record_and_overrides():
    """program B is built with perf5's c18 levers of record; GLM53_PMK_B_<KEY> and a job's b_<key> override them"""
    rec = PS.b_config({}, env={}, tp=8)
    assert (rec["g"], rec["x_l1"], rec["blocks"], rec["prefetch"], rec["full_sync"], rec["yw_db"], rec["yw_coal"]) == (2, 0, (1, 4, 8), 1, 0, 1, 2)
    assert (rec["mcast_linked"], rec["mcast_dual"], rec["lpt"], rec["ring_slots"], rec["inplace"], rec["comb_split"], rec["gu_split"]) == (1, 1, 2, 2, 1, 1, 0)
    assert rec["rect"] == "cover" and rec["gmax"] == 4 and rec["fid"] == "HiFi2" and rec["lpt_w"] is None  # TP8: unweighted
    e = PS.b_config({}, env={"GLM53_PMK_B_LPT": "1", "GLM53_PMK_B_BLOCKS": "2,8,8", "GLM53_PMK_B_LPT_W": "1,2,1.5"}, tp=8)
    assert e["lpt"] == 1 and e["blocks"] == (2, 8, 8) and e["lpt_w"] == [1.0, 2.0, 1.5]
    j = PS.b_config({"b_mcast_dual": 0, "b_fid": "HiFi4"}, env={}, tp=8)
    assert j["mcast_dual"] == 0 and j["fid"] == "HiFi4"


def test_b_config_pack_rne_knob():
    assert PS.b_config({}, env={}, tp=8)["pack_rne"] == 0
    assert PS.b_config({}, env={"GLM53_PMK_B_PACK_RNE": "1"}, tp=8)["pack_rne"] == 1


def test_b_config_selfstream_knob():
    assert PS.b_config({}, env={}, tp=8)["selfstream"] == 0
    e = PS.b_config({}, env={"GLM53_PMK_B_SELFSTREAM": "1", "GLM53_PMK_B_X_L1": "0", "GLM53_PMK_B_BLOCKS": "4,8,8"}, tp=8)
    assert (e["selfstream"], e["x_l1"], e["blocks"]) == (1, 0, (4, 8, 8))
    assert PS.b_config({"b_selfstream": 1}, env={}, tp=8)["selfstream"] == 1


def _steps_launch_count(scopes):
    """ServedLayer.steps with stub program-A / chain objects -> the program-A launches the slot 1-3 steps make"""
    from types import SimpleNamespace

    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS
    from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA

    sl = object.__new__(PS.ServedLayer)
    launches = []
    a = SimpleNamespace(scopes=[tuple(s) for s in scopes], hns=[0, 1], posts=[0, 1], combs=[0, 1], ys=[0], rsr=None,
                        rs8=object(), mm_o=None)  # fmt: skip
    a.index_of = lambda pid: next(i for i, s_ in enumerate(a.scopes) if pid in s_)
    sl.pf, sl.pl, sl.res, sl.sp, sl.a, sl.t = None, SimpleNamespace(_normed={}), None, True, a, {}
    sl.modes = {s: "pmk" for s in ("mhc_pre", "attn", "mhc_mid")} | {"ffn": "op", "mhc_post": "op"}
    sl.kind, sl.router_in, sl.taps, sl.spans_mode, sl.swap_in = DENSE_KDA, "fp32", {}, False, {}
    sl.counters = {"fillers": 0}
    sl._launch = launches.append
    sl._set = lambda *a_, **k_: None
    sl._get = lambda *a_, **k_: None
    for slot, _name, fn in sl.steps(None):
        if slot in ("mhc_pre", "attn", "mhc_mid"):
            fn()
    return launches


def test_fused_program_a_runs_once_per_layer():
    """realgate5 (2026-10-24 22:22Z): the A15 program (P1 .. P10 in one scope) was launched by slots 1, 2 AND 3 (3x in
    place on the residual); every scope layout must launch each of its programs exactly once per layer"""
    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM

    for scopes in (ASM.SCOPES, ASM.SCOPES_A34_MERGE, ASM.SCOPES_A2_FUSE, ASM.SCOPES_A15_FUSE, ASM.SCOPES_A3_MERGE):
        got = _steps_launch_count(scopes)
        assert sorted(got) == list(range(len(scopes))), (scopes, got)
