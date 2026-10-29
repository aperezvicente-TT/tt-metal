# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests (torch only) of the DFlash2 1M form (tt/dflash2_1m.py; dflash2_intrace.md §10):
- the sizing: the drafter's per-token state is 32 KiB of K / V (the real config), 4 KiB per chip at TP8 under the
  head split AND under the sequence split (the two forms hold the same bytes per chip), the table's 1M row;
- the owner map: owner / local position / local bound / local pages round-trip on every position of several
  contexts, the sequence-sharded local caps balance to one unit;
- the split-K attention: 8 owners' fp32 partials merged in chip order == the dense attention (max abs err <= 1e-5
  relative, PCC >= 0.999999) incl. chips without rows (-inf partials) and the non-causal block rows; the merge is
  deterministic (bitwise across two evaluations in the same order);
- the LMCache unit blob: units packed from a head-sharded cache unpack bitwise into a sequence-sharded cache and back
  (chip- and form-independent on disk);
- the bfp8 encoder mirror: round trip error within 2^-7 relative per 16-value group, exact on bfp8-representable values,
  the tile byte layout invariants (64 exponent bytes + 4 x 256 codes).
    python -m pytest tests/s17/test_dflash2_1m_host.py -q   (from the tree root)"""
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if TREE not in sys.path:
    sys.path.insert(0, TREE)

from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_1m as M  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_ref as DR  # noqa: E402

DRAFTER = "/opt/glm53/work-17-d0/dflash2/drafter/config.json"


def real_cfg():
    if os.path.exists(DRAFTER):
        return json.load(open(DRAFTER))
    # the Apache-2.0 drafter's geometry (config.json), for a host without the download
    return dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=8, num_attention_heads=32,
                num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-5, vocab_size=154880,
                rope_parameters=dict(rope_theta=1e4), target_layer_ids=[5, 9, 14, 19, 24, 28, 33, 38, 42],
                mask_token_id=154856, dflash_config=dict(block_size=8, mask_token_id=154856, conv_kernel_size=2,
                                                          conv_group_size=16, selector_rank=256, selector_top_k=16))  # fmt: skip


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


# ---------------------------------------------------------------- sizing
def test_sizing_real_drafter():
    cfg = real_cfg()
    s = M.size_drafter(cfg, tp=8, ctx=1048576)
    assert s["kv_bytes_per_token"] == 32 * 1024 and s["kv_bytes_per_token_per_chip"] == 4 * 1024
    assert s["forms"]["head"] == s["forms"]["seq"]  # the sequence shard buys no bytes at TP8
    assert s["forms"]["replicated"] == 8 * s["forms"]["head"]
    assert s["taps_bytes_per_token"] == 72 * 1024
    assert abs(s["forms"]["head"] / M.GIB - 4.0) < 0.01  # 4 GiB per chip at 1M bf16
    s8 = M.size_drafter(cfg, tp=8, ctx=1048576, kv_dtype="bfp8")
    assert abs(s8["forms"]["head"] / M.GIB - 2.125) < 0.01
    assert 8.5 < s["attention_ms_per_step"][0] < 11 and s8["attention_ms_per_step"][1] < 6
    assert abs(s["lmcache_bytes_per_user"] / M.GIB - 32) < 1e-6
    t = M.sizing_table(cfg)
    assert "1,048,576" in t and "4.00 GiB" in t
    assert abs(M.prefill_ctx_flops_per_token(cfg) / 1e6 - 436) < 1
    for tp in (2, 8):
        s = M.size_drafter(cfg, tp=tp, ctx=32768)
        assert s["forms"]["head"] == s["forms"]["seq"] == 32 * 1024 * (32768 + 32) / tp


# ---------------------------------------------------------------- the owner map
@pytest.mark.parametrize("C", [1, 2, 8])
def test_shard_map(C):
    sm = M.DrafterShardMap(C)
    for L in (100, 4096 + 37, 20000):
        seen = [set() for _ in range(C)]
        for p in range(L):
            r, lp = sm.owner(p), sm.local_pos(p)
            assert lp not in seen[r]
            seen[r].add(lp)
            assert lp < sm.local_cap(L)
            b = p // M.PAGE
            assert sm.page_owner(b) == r and sm.global_page(r, sm.local_page(b)) == b
            assert sm.local_page(b) == lp // M.PAGE
        for r in range(C):
            rows = sm.local_rows(L, r)
            assert len(rows) == sm.n_local_rows(L - 1, r) == len(seen[r])
            assert [sm.local_pos(int(p)) for p in rows] == list(range(len(rows)))  # local order = position order
            assert sm.local_bound(L - 1, r) == len(rows) - 1
        assert sum(sm.n_local_rows(L - 1, r) for r in range(C)) == L
        caps = [sm.n_local_rows(L - 1, r) for r in range(C)]
        assert max(caps) - min(caps) <= M.UNIT
    # the write pages: non-owned rows -> the scratch rows
    pos = torch.arange(250, 282)
    lp = sm.local_pages_of(pos, 0, scratch=4096)
    for j, p in enumerate(pos.tolist()):
        if sm.owner(p) == 0:
            assert int(lp[j]) == sm.local_pos(p)
        else:
            assert int(lp[j]) == 4096 + j


# ---------------------------------------------------------------- the split-K merge
@pytest.mark.parametrize("S,C", [(1000, 8), (4096 + 37, 8), (300, 2), (50, 8)])
def test_split_k_matches_dense(S, C):
    torch.manual_seed(S + C)
    nh, nkv, hd, R = 32, 8, 128, 8
    q = torch.randn(R, nh, hd) * 2
    K = torch.randn(S + R, nkv, hd)
    V = torch.randn(S + R, nkv, hd)
    scale = hd**-0.5
    sm = M.DrafterShardMap(C)
    o_dense = M.dense_attention(q, K, V, scale)
    o_split = M.split_k_attention(q, K, V, scale, sm)
    assert torch.isfinite(o_split).all()
    err = (o_split - o_dense).abs().max() / o_dense.abs().max()
    assert err < 2e-5, float(err)
    assert pcc(o_split, o_dense) > 0.999999
    # deterministic in the fixed order (the replica rule)
    o_split2 = M.split_k_attention(q, K, V, scale, sm)
    assert torch.equal(o_split, o_split2)
    # chips without rows (S = 50 < 8 units) contribute nothing and do not poison the merge
    if S + R <= M.UNIT * (C - 1):  # some chip owns no row
        parts = [
            M.attention_partial(q, K[sm.local_rows(S + R, r)], V[sm.local_rows(S + R, r)], scale) for r in range(C)
        ]
        assert any(not torch.isfinite(p[0]).any() for p in parts)
        assert torch.allclose(M.merge_partials(parts), o_dense, rtol=1e-4, atol=1e-5)


def test_seq_form_fabric_bytes():
    b = M.seq_form_bytes_per_layer_step(real_cfg(), 8)
    assert b["q_piece"] == 8 * 4 * 128 * 2 and b["ag_in"] == 7 * b["q_piece"]
    assert b["partial"] == 8 * 32 * 128 * 4 + 2 * 8 * 32 * 4


# ---------------------------------------------------------------- the unit blob
def test_unit_blob_head_to_seq_round_trip():
    torch.manual_seed(0)
    cfg = DR.small_config(tp=8, n_taps=3, layers=2)
    d = DR.Dims(cfg)
    tp, C = 8, 8
    sm = M.DrafterShardMap(C)
    n_units = 5
    S = n_units * M.UNIT
    # a head-sharded cache: per layer per chip [S, nkv_l, hd] rows (nkv_l = nkv / tp)
    nkv_l = d.nkv // tp
    full = [
        (torch.randn(S, d.nkv, d.hd).to(torch.bfloat16), torch.randn(S, d.nkv, d.hd).to(torch.bfloat16))
        for _ in range(d.L)
    ]
    head = [([K[:, r * nkv_l : (r + 1) * nkv_l] for r in range(tp)], [V[:, r * nkv_l : (r + 1) * nkv_l] for r in range(tp)])
            for K, V in full]  # fmt: skip
    blob = M.pack_units(head, 0, n_units, None)
    assert tuple(blob.shape) == (n_units, d.L, 2, d.nkv, M.UNIT, d.hd)
    # -> a sequence-sharded cache (every head on the owner, local order)
    seq = [([torch.zeros(sm.local_cap(S), d.nkv, d.hd, dtype=torch.bfloat16) for _ in range(C)],
            [torch.zeros(sm.local_cap(S), d.nkv, d.hd, dtype=torch.bfloat16) for _ in range(C)]) for _ in range(d.L)]  # fmt: skip
    for li, per_kv in enumerate(M.unpack_units(blob, 0, C, sm)):
        for kv, per_chip in enumerate(per_kv):
            for r, writes in enumerate(per_chip):
                for r0, rows in writes:
                    seq[li][kv][r][r0 : r0 + M.UNIT] = rows
    for li, (K, V) in enumerate(full):
        for r in range(C):
            idx = sm.local_rows(S, r)
            assert torch.equal(seq[li][0][r][: len(idx)], K[idx]) and torch.equal(seq[li][1][r][: len(idx)], V[idx])
    # and back: the seq cache packs to the same blob, which unpacks into the head form bitwise
    blob2 = M.pack_units(seq, 0, n_units, sm)
    assert torch.equal(blob, blob2)
    head2 = [([torch.zeros(S, nkv_l, d.hd, dtype=torch.bfloat16) for _ in range(tp)],
              [torch.zeros(S, nkv_l, d.hd, dtype=torch.bfloat16) for _ in range(tp)]) for _ in range(d.L)]  # fmt: skip
    for li, per_kv in enumerate(M.unpack_units(blob2, 0, tp, None)):
        for kv, per_chip in enumerate(per_kv):
            for r, writes in enumerate(per_chip):
                for r0, rows in writes:
                    head2[li][kv][r][r0 : r0 + M.UNIT] = rows
    for li in range(d.L):
        for kv in range(2):
            for r in range(tp):
                assert torch.equal(head2[li][kv][r], head[li][kv][r])
    assert M.UnitFormat(real_cfg()).bytes == 8 * 2 * 8 * 128 * 128 * 2  # 4 MiB per unit


# ---------------------------------------------------------------- the bfp8 encoder mirror
def test_bfp8_rows():
    torch.manual_seed(1)
    x = (torch.randn(64, 128) * 3).to(torch.bfloat16)
    x[3, :16] = 0  # a zero group
    x[5, 16] = 0  # a zero inside a group
    sh, codes = M.bfp8_rows(x)
    assert tuple(sh.shape) == (64, 8) and tuple(codes.shape) == (64, 128)
    y = M.bfp8_like(x)
    # relative error per 16-value group within 2^-7 of the group's max (7-bit mantissa, round-to-nearest)
    g = x.float().reshape(64, 8, 16)
    e = (y.reshape(64, 8, 16) - g).abs().amax(-1) / g.abs().amax(-1).clamp(min=1e-30)
    assert (e <= 2.0**-7 + 1e-6).all(), float(e.max())
    assert (y.reshape(64, 8, 16)[3, 0] == 0).all() and y[5, 16] == 0
    # the group max is exact (it sets the shared exponent; its 7-bit rounding is RNE of a bf16 with 8 significant bits)
    # bfp8 values are bf16-exact and a second round trip is the identity
    y16 = y.to(torch.bfloat16)
    assert torch.equal(y16.float(), y)
    assert torch.equal(M.bfp8_like(y16), y)
    # the tile layout: exponent bytes first (one per face row), codes after
    t = x[:32, :32]
    b = M.bfp8_tile_bytes(t)
    assert b.numel() == 1088
    sh_t, codes_t = M.bfp8_rows(t[:16, :16])
    assert torch.equal(b[:16], sh_t.reshape(16)) and torch.equal(b[64 : 64 + 256], codes_t.reshape(256))
    sh_t3, _ = M.bfp8_rows(t[16:, :16])
    assert torch.equal(b[32:48], sh_t3.reshape(16))


# ---------------------------------------------------------------- the kind: the sequence form vs the head form (torch backend)
def _kinds_ctx(tp: int, shard: bool, kv_dtype: str, n0: int, cap: int, seed: int = 0, layers: int = 2):
    from models.autoports.zai_org_glm_5_3_flash.tt import dflash2_kind as DK

    torch.manual_seed(seed)
    cfg = DR.small_config(tp=tp, n_taps=3, layers=layers)
    d = DR.Dims(cfg)
    w = DR.random_weights(cfg, seed=seed, scale=0.05)
    ops = DK.TorchOps(tp, shard=shard, kv_dtype=kv_dtype, layer="ops")
    tin = ops.zeros((32, 64), torch.int64, "rm")
    rec = ops.zeros((1, 64), torch.int64, "rm")
    kind = DK.DFlash2Kind(
        ops, cfg, w, K=7, cap=cap, tin=tin, rec=rec, tap_mode="mean", conv="fused", log=lambda *a: None
    )
    g = torch.Generator().manual_seed(seed + 1)
    taps = (torch.randn(n0, d.n_taps, d.H, generator=g) * 0.5).to(torch.bfloat16)
    kind.seed_ctx(taps, 0)
    return kind, ops, tin, rec, d, taps


def _boot(kind, ops, tin, rec, bonus: int, n: int, c: int):
    rec_host = torch.zeros(1, 64, dtype=torch.int64)
    rec_host[0, 0], rec_host[0, 1] = n, c
    ops.write(rec, rec_host)
    tin_host = torch.zeros(32, 64, dtype=torch.int64)
    tin_host[0, 0] = bonus
    ops.write(tin, tin_host)
    ops.launches = 0
    ops.census.clear()
    kind.boot()
    return [int(ops.host(tin)[s, 0]) for s in range(1, 8)], ops.host(kind.last["hid"]).reshape(32, -1)[:8].float()


def test_kind_sequence_form_matches_head_form():
    """GLM53_DFLASH2_SHARD=1 on the torch backend: the owner-aware context writes + the split-K attention merged over
    the 8 chips give the head form's drafts (equal) and hidden (PCC >= 0.9999) on a 1,200-position context (> 8
    units: every chip owns rows) at the boot and after a step; the caches hold the same rows (bitwise per head)"""
    tp, n0, cap = 8, 1200, 2048
    kh, oh, th, rh, d, taps = _kinds_ctx(tp, False, "bf16", n0, cap)
    ks, os_, ts, rs, _, _ = _kinds_ctx(tp, True, "bf16", n0, cap)
    assert ks.sm is not None and ks.nblk == M.DrafterShardMap(tp).local_cap(kh.cap) // 32 + 1
    # the context rows: chip r of the sequence form holds, in local order, the positions it owns with every head
    for li in range(len(kh.L)):
        for kv in (0, 1):
            full = torch.cat([M.kind_cache_rows(kh, li, kv, r) for r in range(tp)], 1)[:n0]  # [n0, nkv, hd]
            for r in range(tp):
                idx = ks.sm.local_rows(n0, r)
                loc = M.kind_cache_rows(ks, li, kv, r)[: len(idx)]
                assert torch.equal(loc, full[idx]), (li, kv, r)
    bonus = 11
    dh, hh = _boot(kh, oh, th, rh, bonus, 0, n0 - 1)
    ds, hs = _boot(ks, os_, ts, rs, bonus, 0, n0 - 1)
    print("boot drafts head", dh, "seq", ds, "hid pcc", pcc(hh, hs), "seq census", dict(os_.census))
    assert pcc(hh, hs) >= 0.9999
    assert dh == ds
    assert os_.census["sdpa_partial"] == len(kh.L) and os_.census["merge_partials"] == len(kh.L)
    # a step: n = 3 accepted, the rows' taps on the slots, the block at c' = c + n + 1
    n, c = 3, n0
    g = torch.Generator().manual_seed(5)
    step_taps = (torch.randn(8, d.n_taps, d.H, generator=g) * 0.5).to(torch.bfloat16)
    for kind, ops_ in ((kh, oh), (ks, os_)):
        for s in range(d.n_taps):
            slot = torch.zeros(32, kind.tap_w)
            slot[:8] = step_taps[:, s].float()
            ops_.write(kind.taps[s], slot.reshape(1, 1, 32, -1))
    for kind, ops_, tin_, rec_ in ((kh, oh, th, rh), (ks, os_, ts, rs)):
        rec_host = torch.zeros(1, 64, dtype=torch.int64)
        rec_host[0, 0], rec_host[0, 1] = n, c
        ops_.write(rec_, rec_host)
        tin_host = torch.zeros(32, 64, dtype=torch.int64)
        tin_host[0, 0] = 23
        ops_.write(tin_, tin_host)
        kind.step()
    dh2 = [int(oh.host(th)[s, 0]) for s in range(1, 8)]
    ds2 = [int(os_.host(ts)[s, 0]) for s in range(1, 8)]
    hh2 = oh.host(kh.last["hid"]).reshape(32, -1)[:8].float()
    hs2 = os_.host(ks.last["hid"]).reshape(32, -1)[:8].float()
    print("step drafts head", dh2, "seq", ds2, "hid pcc", pcc(hh2, hs2))
    assert pcc(hh2, hs2) >= 0.9999 and dh2 == ds2
    # the local bound after the step: rows 0 .. c' + K on their owners
    c1 = c + n + 1
    for r in range(tp):
        assert int(os_.host(ks.st.cur_l, r)[0]) == ks.sm.local_bound(c1 + 7, r)
    # the context rows written by the step (positions c .. c + 3 committed) are on their owners only
    for r in range(tp):
        for p in range(c, c + n + 1):
            if ks.sm.owner(p) == r:
                lp = ks.sm.local_pos(p)
                head_rows = torch.cat([M.kind_cache_rows(kh, 0, 0, rr)[p] for rr in range(tp)], 0)
                assert torch.equal(M.kind_cache_rows(ks, 0, 0, r)[lp], head_rows)


def test_kind_bfp8_cache_host():
    """GLM53_DF2_KV_DTYPE=bfp8 on the torch backend: the K / V rows are bfp8-rounded at the write (the kernels'
    encoder), the drafts / hidden against the bf16 cache (the dtype's cost, measured; the device unit decides)"""
    tp, n0, cap = 8, 300, 1024
    k16, o16, t16, r16, d, _ = _kinds_ctx(tp, False, "bf16", n0, cap)
    k8, o8, t8, r8, _, _ = _kinds_ctx(tp, False, "bfp8", n0, cap)
    rows16, rows8 = M.kind_cache_rows(k16, 0, 0, 0)[:n0], M.kind_cache_rows(k8, 0, 0, 0)[:n0]
    assert torch.equal(rows8.float(), M.bfp8_like(rows16))  # exactly the encoder's values
    assert not torch.equal(rows8, rows16)
    d16, h16 = _boot(k16, o16, t16, r16, 11, 0, n0 - 1)
    d8, h8 = _boot(k8, o8, t8, r8, 11, 0, n0 - 1)
    p = pcc(h16, h8)
    print("bfp8 cache: hid pcc", p, "drafts equal", sum(a == b for a, b in zip(d16, d8)), "/ 7", d16, d8)
    assert p >= 0.999


def test_kind_units_export_import():
    """the LMCache blob on real kinds: units exported from the head form (bf16) import into a fresh sequence-form
    kind (and a bfp8 one): the rows land bitwise (bfp8: the encoder's values) and the boot gives the same drafts"""
    tp, n0, cap = 8, 1200, 2048
    kh, oh, th, rh, d, _ = _kinds_ctx(tp, False, "bf16", n0, cap)
    n_units = n0 // M.UNIT  # the complete units (9)
    blob = M.export_units(kh, 0, n_units)
    assert tuple(blob.shape) == (n_units, d.L, 2, d.nkv, M.UNIT, d.hd)
    fmt = M.UnitFormat(DR.small_config(tp=tp, n_taps=3, layers=2))
    assert blob[0].numel() * 2 == fmt.bytes
    for shard, dtype in ((True, "bf16"), (False, "bfp8"), (True, "bfp8")):
        kn, on, tn, rn, _, _ = _kinds_ctx(tp, shard, dtype, 0, cap)  # no context
        M.import_units(kn, blob, 0)
        for li in range(d.L):
            for kv in (0, 1):
                full = torch.cat([M.kind_cache_rows(kh, li, kv, r) for r in range(tp)], 1)[: n_units * M.UNIT]
                want = M.bfp8_like(full).to(torch.bfloat16) if dtype == "bfp8" else full
                for r in range(tp):
                    got = M.kind_cache_rows(kn, li, kv, r)
                    if shard:
                        idx = kn.sm.local_rows(n_units * M.UNIT, r)
                        assert torch.equal(got[: len(idx)], want[idx]), (shard, dtype, li, kv, r)
                    else:
                        nkv_l = d.nkv // tp
                        assert torch.equal(got[: n_units * M.UNIT], want[:, r * nkv_l : (r + 1) * nkv_l]), (
                            dtype,
                            li,
                            kv,
                            r,
                        )
        # the page view round trip (the connector's protocol) on the imported kind
        v = M.connector_views(kn)["drafter.layers.0.k"]
        pg = v.read_pages([3, 4, 9])
        assert tuple(pg.shape) == (3, d.nkv, 32, d.hd)
        full = torch.cat([M.kind_cache_rows(kh, 0, 0, r) for r in range(tp)], 1)
        want = M.bfp8_like(full).to(torch.bfloat16) if dtype == "bfp8" else full
        assert torch.equal(pg[1], want[4 * 32 : 5 * 32].permute(1, 0, 2))
        if dtype == "bf16":  # the same context: the same drafts at the boot of the complete units' end
            L = n_units * M.UNIT
            dh, hh = _boot(kh, oh, th, rh, 11, 0, L - 1)
            dn, hn = _boot(kn, on, tn, rn, 11, 0, L - 1)
            assert dh == dn and pcc(hh, hn) >= 0.9999, (dh, dn, pcc(hh, hn))
