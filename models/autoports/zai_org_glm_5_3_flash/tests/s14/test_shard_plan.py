# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks (no device, no C++ build, no ttnn) of the sequence-sharded DSA exchange at TP2 and TP8
(tt/shard_plan.py, tt/dsa_shard.py; doc/ttnn/14-production-tp8-megakernel/dsa/shard_tp8_design.md):
  the header mirrors (dsa_shard.hpp / fs_common.hpp / fs_send.cpp vs the Python constants);
  ranks / pair and line peers / y_ag blocks / head tiles on 1x2 and 2x4;
  the four streams' blocks and destinations, simulated chunk by chunk (every chip ends with all 8 y_ag blocks, every
  head owner with exactly one slice from each row peer, no two writers on one destination block);
  the link holders per (chip, link, direction) and the AR release lists (audit_links), the packet-header budget;
  ShardMap at C = 8 against the kernels' formulas (dsa_shard.hpp owner / local_pos / n_local_rows / n_closed_local);
  the TP8 byte formats: the compact q piece + q-tile assembly, the 8-row slice runs, the x_rs2 layout;
  the merge's rank-order walk over the y_ag blocks = tt/dsa_shard.py sharded_select's (chip, core, local) tie order;
  a torch emulation of the 2-round AG + q assembly + 2-round RS with the fixed combine order (fp32) reproducing the
  one-chip attention over the union of the chips' rows (pair and 2x4), with junk rows in the receive tiles.
Run: no-tt-devices python -m pytest --noconftest -q tests/s14/test_shard_plan.py (PYTHONPATH = the tt-metal tree)."""

from __future__ import annotations

import math
import os
import re

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS
from models.autoports.zai_org_glm_5_3_flash.tt import shard_plan as SP

TT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tt")
MESHES = [(1, 2), (2, 1), (2, 4)]


def _src(rel):
    with open(os.path.join(TT, rel)) as f:
        return f.read()


# ------------------------------------------------------------------ header mirrors
def test_header_mirrors():
    h = SP.HDR
    assert (h["UNIT"], h["POOL_PAGE"], h["TB"], h["PPC"]) == (DS.UNIT, DS.POOL_PAGE, 2048, 16)
    assert (h["REC_BYTES"], h["REC_HDR"], h["RS_CHUNKS"], h["T_MAX"], h["TL_MAX"]) == (16, 16, 3, 32, 8)
    assert (h["AG8_PPC"], h["AG8_QPAGES"], h["RS2_PPC"], h["RS_PPC"]) == (10, 4, 10, 12)
    assert SP.merge_msgs() == 11 and SP.merge_scr_bytes() == 5 * 2048 + 2 * 512 * 4 + 256
    mk_ = _src("mk_mla.py")
    assert "SHARD_RS_PPC = 12" in mk_ and "SHARD_MERGE_MSGS = 11" in mk_ and "SHARD_PROBE_LANE0 = 236" in mk_
    assert re.search(r"^SHARD_TL_MAX = (\d+)$", mk_, re.M).group(1) == str(h["TL_MAX"])
    # rev 60: the shard mailbox is lane-local (dsa_shard.hpp TL = mladsa::T_MAX: 4 at t <= 4, TL_MAX at t > 4)
    assert "static_assert(TL == 4 || TL == TL_MAX" in _src("kernels/mla/dsa_shard.hpp")
    assert "return DSA_T_MAX if t <= DSA_T_MAX else SHARD_TL_MAX" in mk_
    # review 2 item 4: the fs_send probe flush words are named + asserted in fs_common.hpp
    # stack45 x 1M (16:25Z ruling): 56 = the free words 56..63 below OWNRDY (84 overlapped the PMK R1RDY 81..92)
    assert SP.FS_HDR["PROBE_FLUSH"] == 56 and "fs::PROBE_FLUSH" in _src("kernels/fabric_stream/fs_send.cpp")
    assert SP.FS_HDR["PROBE_FLUSH"] * 16 + 127 <= SP.FS_HDR["OWNRDY"] * 16 and SP.FS_HDR["R1RDY"] == 81
    assert SP.FS_HDR["PROBE_FLUSH"] * 16 + 127 <= SP.FS_HDR["SYNC_WORDS"] * 16
    # review 2 item 5: the fabric stream kernel text carries the hash of the headers it includes by path
    assert '_DEPS = ("mk/probe.hpp",)' in _src("fabric_stream.py")
    assert "SHARD_MERGE_SCR = 5 * 2048 + 2 * 512 * 4 + 64 * 4" in mk_
    # the probe rows: every shard row inside PROBE_ROWS (mk_mla.py) and the stack-layer block (probe.hpp MK_PROBE_LROWS)
    lrows = SP.header_constants_define("mk/probe.hpp", "MK_PROBE_LROWS")
    prows = int(re.search(r"^PROBE_ROWS = (\d+)", mk_, re.M).group(1))
    assert max(SP.PROBE_LANE_ROW.values()) < prows <= lrows and min(SP.PROBE_LANE_ROW.values()) == 236
    assert len(set(SP.PROBE_LANE_ROW.values())) == 4 and SP.PROBE_IX1 < 236
    assert SP.CAND_PAGES == 5 and SP.AG8_QPAGES + SP.CAND_PAGES <= SP.AG8_PPC
    # mk_mla.py's mirrors (that module imports ttnn: compared on its source text)
    mk = _src("mk_mla.py")
    m = re.search(r"SHARD_AG8_PPC, SHARD_AG8_QPAGES, SHARD_RS2_PPC = (\d+), (\d+), (\d+)", mk)
    assert m and tuple(int(v) for v in m.groups()) == (h["AG8_PPC"], h["AG8_QPAGES"], h["RS2_PPC"])
    m = re.search(r"SHARD_T_MAX, SHARD_CAND_MAX, SHARD_REC_BYTES, SHARD_REC_HDR, SHARD_RS_CHUNKS = "
                  r"(\d+), (\d+), (\d+), (\d+), (\d+)", mk)  # fmt: skip
    assert m and tuple(int(v) for v in m.groups()) == (h["T_MAX"], SP.CAND_MAX, h["REC_BYTES"], h["REC_HDR"],
                                                       h["RS_CHUNKS"])  # fmt: skip
    # fabric_stream.py's SYNC_WORDS / NHDR vs the kernels
    fs = SP.FS_HDR
    assert fs["SYNC_WORDS"] == 96 and fs["WORD"] == 16 and fs["ARR"] == 0 and fs["DONE"] == 16
    assert "SYNC_WORDS = 96" in _src("fabric_stream.py") and "NHDR = 4" in _src("fabric_stream.py")
    assert f"constexpr uint32_t NHDR = {SP.FS_NHDR};" in _src("kernels/fabric_stream/fs_send.cpp")
    assert "constexpr uint32_t NHDR = 2;" in _src("kernels/fabric_ar/lane_send.cpp")
    # the header's LINE is a define (1 or 4)
    assert "static_assert(LINE == 1 || LINE == 4" in _src("kernels/mla/dsa_shard.hpp")


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("R,C", MESHES)
def test_ranks_and_blocks(R, C):
    g = SP.Geometry(R, C)
    n = g.n
    assert sorted(g.rank(r, c) for r in range(R) for c in range(C)) == list(range(n))
    assert sorted(g.ag_blk(k) for k in range(n)) == list(range(n))  # a bijection onto y_ag's blocks
    for k in range(n):
        r, c = g.coord(k)
        assert g.rank(r, c) == k
        assert g.pair_peer(g.pair_peer(k)) == k and g.pair_peer(k) != k
        assert g.line_pos(g.pair_peer(k)) == g.line_pos(k)
        assert len(g.line_peers(k)) == g.L - 1
        # pass 0 attends the other pair position's tile, pass 1 its own; the two tiles = the 64 heads
        assert {g.pass_tile(k, 0), g.pass_tile(k, 1)} == {0, 1}
        assert k in g.tile_ranks(g.pass_tile(k, 1))
    if (R, C) == (2, 4):
        assert [g.rank(r, c) for r in range(2) for c in range(4)] == list(range(8))  # 4 row + col
        assert g.tile_ranks(0) == [0, 1, 2, 3] and g.tile_ranks(1) == [4, 5, 6, 7]
        for c in range(4):  # the round-2 bundle of column c = y_ag blocks 2c, 2c + 1 = ranks c (row 0), 4 + c (row 1)
            assert (g.ag_blk(c), g.ag_blk(4 + c)) == (2 * c, 2 * c + 1)
        heads = [list(g.heads(k, 8)) for k in range(8)]
        assert sum(heads, []) == list(range(64))
        for t_ in (0, 1):  # tile t_ rows 8 j .. = rank 4 t_ + j's 8 heads
            assert sum((heads[k] for k in g.tile_ranks(t_)), []) == list(range(32 * t_, 32 * t_ + 32))
    with pytest.raises(ValueError):
        SP.Geometry(1, 4)


def _simulate(g: SP.Geometry, t: int):
    """chunk-level simulation of the 4 streams: who writes which y block on which chip"""
    n = g.n
    y_ag = {k: {} for k in range(n)}  # chip rank -> {y_ag block: source rank}
    sp = g.streams(t)
    chips = [(r, c) for r in range(g.R) for c in range(g.C)]
    # AG round 1 (the pair axis): local copy + the multicast to the pair peer
    for r, c in chips:
        b = g.block("ag1", t, r, c)
        k = g.rank(r, c)
        assert b["dst"] % sp["ag1"]["chunks"] == 0
        blk = b["dst"] // sp["ag1"]["chunks"]
        assert blk == g.ag_blk(k)
        for dest in [k] + [g.pair_peer(k)]:
            assert blk not in y_ag[dest], ("two writers", dest, blk)
            y_ag[dest][blk] = k
        assert b["up"] + b["dn"] == 1
    if g.L > 1:
        snap = {k: dict(v) for k, v in y_ag.items()}
        for r, c in chips:
            b = g.block("ag2", t, r, c)
            k = g.rank(r, c)
            ch = sp["ag2"]["chunks"]
            assert b["src"] == b["dst"] and b["dst"] % ch == 0 and not b["local"]
            bundle = [2 * (b["dst"] // ch), 2 * (b["dst"] // ch) + 1]  # in y_ag blocks
            assert all(bb in snap[k] for bb in bundle), "the bundle landed before round 2 reads it"
            p = g.line_pos(k)
            reach = [p + d for d in range(1, b["up"] + 1)] + [p - d for d in range(1, b["dn"] + 1)]
            assert sorted(reach) == sorted(q for q in range(g.L) if q != p)
            for q in reach:
                dest = g.pair_pos(k) * g.L + q
                for bb in bundle:
                    assert bb not in y_ag[dest], ("two writers", dest, bb)
                    y_ag[dest][bb] = snap[k][bb]
    for k in range(n):
        assert sorted(y_ag[k]) == list(range(n)), (k, y_ag[k])
        assert all(y_ag[k][g.ag_blk(src)] == src for src in range(n))
    # RS round 1: x_rs1 block (1 - p) -> the pair peer's y_rs1
    for r, c in chips:
        b = g.block("rs1", t, r, c)
        k = g.rank(r, c)
        assert b["src"] == (1 - g.pair_pos(k)) * sp["rs1"]["chunks"] and b["dst"] == 0
    # RS round 2: x_rs2 block q -> peer q's y_rs2 block p
    if g.L > 1:
        y_rs2 = {k: {} for k in range(n)}
        for r, c in chips:
            b = g.block("rs2", t, r, c)
            k = g.rank(r, c)
            p = g.line_pos(k)
            assert b["dst"] == p * t
            assert sorted(q for q, *_ in b["dsts"]) == sorted(g.line_pos(x) for x in g.line_peers(k))
            for q, src, d, hops in b["dsts"]:
                assert src == q * t and hops == abs(q - p) and d == (0 if q > p else 1)
                assert hops <= (b["up"] if d == 0 else b["dn"])
                dest = g.pair_pos(k) * g.L + q
                assert p not in y_rs2[dest]
                y_rs2[dest][p] = k
        for k in range(n):
            assert sorted(y_rs2[k]) == sorted(g.line_pos(x) for x in g.line_peers(k))
    return y_ag


@pytest.mark.parametrize("R,C", MESHES)
@pytest.mark.parametrize("t", [1, 2, 4, 8, 16, 32])
def test_stream_blocks(R, C, t):
    _simulate(SP.Geometry(R, C), t)


# ------------------------------------------------------------------ links / headers
@pytest.mark.parametrize("R,C", MESHES)
def test_link_holders(R, C):
    g = SP.Geometry(R, C)
    rel = g.release_lists(4)
    assert rel == ({0: ["ag1"], 1: ["rs1"]} if g.L == 1 else {0: ["ag1", "ag2"], 1: ["rs1", "rs2"]})
    for r in range(R):
        for c in range(C):
            hs = g.holders(r, c, 4)
            out = SP.audit_links(hs)
            keys = {(h[1], h[2]) for h in hs}
            for key in keys:
                on = [h for h in hs if (h[1], h[2]) == key]
                assert sum(h[3] == "late" for h in on) == 1 and sum(h[3] == "gated" for h in on) <= 1, on
            # every direction the chip has carries both links' AR lanes; the streams sit on the design's links
            names = {h[0]: h[1] for h in hs if h[3] == "gated"}
            assert names["ag1"] == 0 and names["rs1"] == 1
            if g.L > 1:
                assert names["ag2"] == 0 and names["rs2"] == 1
                p = c
                assert len([h for h in hs if h[0] == "ar_lane0"]) == 1 + (p > 0) + (p < 3)
            assert out
    # negative: two gated streams on one (link, direction) / a gated stream released by the other link's lane
    hs = SP.Geometry(2, 4).holders(0, 1, 4)
    bad = [h if h[0] != "rs2" else ("rs2", 0, h[2], "gated", "ar_lane0") for h in hs]
    with pytest.raises(ValueError):
        SP.audit_links(bad)
    bad = [h if h[0] != "ag1" else ("ag1", 0, h[2], "gated", "ar_lane1") for h in hs]
    with pytest.raises(ValueError):
        SP.audit_links(bad)
    with pytest.raises(ValueError):
        SP.audit_links(hs + [("extra", 0, "a0+", "hold", None)])
    # the pair's old 3-tuple form still audits
    SP.audit_links([("ar_lane0", 0, "late"), ("ar_lane1", 1, "late"), ("ag", 0, "gated"), ("rs", 1, "gated")])


@pytest.mark.parametrize("R,C", MESHES)
def test_header_budget(R, C):
    g = SP.Geometry(R, C)
    worst = 0
    for r in range(R):
        for c in range(C):
            b = g.header_budget(r, c, 4)
            SP.audit_headers(b)
            worst = max(worst, max(b.values()))
            if (R, C) == (2, 4) and c in (1, 2):  # the middle chips of the row line: up + down + the axis-0 direction
                assert b[("ar_lane0", "brisc")] == 9 and b[("ag2", "brisc")] == 8 and b[("rs2", "brisc")] == 8
                assert b[("ag1", "brisc")] == 4 and b[("rs1", "brisc")] == 4
    assert worst <= SP.PKT_HDRS_PER_RISC
    with pytest.raises(ValueError):
        SP.audit_headers({("x", "brisc"): 3 * (4 + 1)})  # the 2026-10-22 overflow


def test_audit_targets():
    valid = {(1, 2), (3, 4), (5, 6)}
    ok = SP.audit_targets([("k", (0, 0), 5, 0, [(1, 2)])], 5, 0, valid, [("ag1.cons", (3, 4))])
    assert ok["rank"] == 5
    for entries, extra in (([("k", (0, 0), 4, 0, [(1, 2)])], []), ([("k", (0, 0), 5, 1, [(1, 2)])], []),
                           ([("k", (0, 0), 5, 0, [(0, 0)])], []), ([], [("rs2.lane", (9, 9))])):  # fmt: skip
        with pytest.raises(ValueError):
            SP.audit_targets(entries, 5, 0, valid, extra)


# ------------------------------------------------------------------ ShardMap at C = 8 vs the kernels' formulas
def _k_owner(p, C):
    return (p // 128) % C


def _k_local_pos(p, C):
    return (p // 128 // C) * 128 + p % 128


def _k_n_local_rows(p, r, C):
    n = p + 1
    fu, rem = n // 128, n % 128
    full = (fu - r + C - 1) // C if fu > r else 0
    return full * 128 + (rem if fu % C == r else 0)


def _k_n_closed_local(p, r, C):
    n = (p + 1) // 4
    fu, rem = n // 32, n % 32
    full = (fu - r + C - 1) // C if fu > r else 0
    return full * 32 + (rem if fu % C == r else 0)


@pytest.mark.parametrize("C", [2, 8])
def test_shardmap_general_c(C):
    sm = DS.ShardMap(C)
    gen = torch.Generator().manual_seed(3)
    ps = [0, 1, 127, 128, 129, 1023, 1024, 2050, 2051, 4097, 32767, 131071, 1048575] + torch.randint(
        0, 1 << 20, (40,), generator=gen
    ).tolist()
    for p in ps:
        assert sm.owner(p) == _k_owner(p, C) and sm.local_pos(p) == _k_local_pos(p, C)
        rows = [sm.n_local_rows(p, r) for r in range(C)]
        assert rows == [_k_n_local_rows(p, r, C) for r in range(C)] and sum(rows) == p + 1
        pools = [sm.n_closed_local(p, r) for r in range(C)]
        assert pools == [_k_n_closed_local(p, r, C) for r in range(C)] and sum(pools) == (p + 1) // 4
        assert sm.local_pos(p) < sm.local_cap(p + 1)
        g = p // 4
        assert sm.global_pool(sm.pool_owner(g), sm.local_pool(g)) == g
    # positions <-> local rows: a bijection per chip, in order
    L = 5000
    rows = torch.arange(L, dtype=torch.float32)[:, None].repeat(1, 3)
    seen = []
    for r in range(C):
        loc = sm.shard_rows(rows, r)
        n = sm.n_local_rows(L - 1, r)
        assert torch.equal(loc[n:], torch.zeros_like(loc[n:]))
        ids = loc[:n, 0].long().tolist()
        assert ids == sorted(ids) and all(sm.owner(i) == r and sm.local_pos(i) == j for j, i in enumerate(ids))
        seen += ids
    assert sorted(seen) == list(range(L))
    # page tables: b % C == owner, local block b // C
    bps = 2 * C * 3  # global column j = block of positions [64 j, 64 j + 64): unit j // 2, owner (j // 2) % C
    pt = torch.stack([torch.tensor([((j // 2) % C) + C * (100 + j) for j in range(bps)])])
    for r in range(C):
        loc = sm.page_table_local(pt, r)
        for jl in range(loc.shape[1]):
            jg = ((jl // 2) * C + r) * 2 + jl % 2
            assert int(loc[0, jl]) == (100 + jg if jg < bps else 0), (r, jl)
    bad = pt.clone()
    bad[0, 0] += 1
    with pytest.raises(ValueError):
        sm.page_table_local(bad, 0)


# ------------------------------------------------------------------ TP8 byte formats
def test_offsets():
    for L in (1, 4):
        for T in (1, 2, 3, 4):
            g = SP.Geometry(2, 4) if L == 4 else SP.Geometry(1, 2)
            per_block = (SP.AG8_PPC * T) if L > 1 else (2 * T * SP.PPC)
            used = {}
            for blk in range(g.n):
                for t in range(T):
                    for d in range(16):
                        pg, off = SP.ag_q_page(L, T, blk, t, d)
                        span = 512 if L > 1 else 2048
                        assert blk * per_block <= pg < (blk + 1) * per_block and off + span <= 2048
                        for b in range(off, off + span, 256):
                            assert (pg, b) not in used
                            used[(pg, b)] = ("q", blk, t, d)
                    for byte in range(0, SP.REC_HDR + SP.CAND_MAX * SP.REC_BYTES, 256):
                        pg, off = SP.ag_cand_page(L, T, blk, t, byte)
                        assert blk * per_block <= pg < (blk + 1) * per_block
                        assert (pg, off) not in used, (pg, off, used.get((pg, off)))
                        used[(pg, off)] = ("rec", blk, t)
    # slices: the 4 x 2 runs of a tile partition it, for fp32 (4096) and bf16 (2048) tiles
    for tb in (4096, 2048):
        cover = []
        for j in range(4):
            a = SP.slice_off(j, tb)
            cover += [(a, a + tb // 8), (a + tb // 4, a + tb // 4 + tb // 8)]
            assert a % 256 == 0
        cover.sort()
        assert cover[0][0] == 0 and cover[-1][1] == tb and all(x[1] == y[0] for x, y in zip(cover, cover[1:]))
    for otb in (4096, 2048):  # the x_rs2 slice chunk: no run straddles a 2-KiB page
        lay = SP.rs2_layout(otb)
        assert len(lay) == 18
        for kind, i, b in lay:
            tb = otb if kind == "o" else 4096
            for start in (b, b + tb // 8):
                assert start // 2048 == (start + tb // 8 - 1) // 2048 and start % 64 == 0


def test_q_assembly_bytes():
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(64, 512, generator=gen)  # heads x latent
    g = SP.Geometry(2, 4)
    # chip k's q_lat tile (t, d): rows 0..7 = its 8 heads, rows 8..31 padding (junk)
    y_ag = {}
    for k in range(8):
        pieces = []
        for d in range(16):
            tile = torch.randn(32, 32, generator=gen) * 1e6  # junk padding rows
            tile[:8] = q[8 * k : 8 * k + 8, 32 * d : 32 * d + 32]
            pieces.append(SP.q_piece(SP.tile_to_faces(tile)))
        y_ag[g.ag_blk(k)] = pieces
    for tile_id in (0, 1):
        for d in range(16):
            pcs = [y_ag[g.ag_blk(k)][d] for k in g.tile_ranks(tile_id)]
            got = SP.faces_to_tile(SP.assemble_q_tile(pcs))
            assert torch.equal(got, q[32 * tile_id : 32 * tile_id + 32, 32 * d : 32 * d + 32])


def test_slice_runs_roundtrip():
    gen = torch.Generator().manual_seed(2)
    for j in range(4):
        src = torch.randn(32, 32, generator=gen)
        junk = torch.randn(32, 32, generator=gen)
        f = SP.land_slice(SP.tile_to_faces(junk).clone(), j, SP.slice_runs(SP.tile_to_faces(src), j))
        out = SP.faces_to_tile(f)
        rows = slice(8 * j, 8 * j + 8)
        assert torch.equal(out[rows], src[rows])
        keep = torch.ones(32, dtype=torch.bool)
        keep[rows] = False
        assert torch.equal(out[keep], junk[keep])


# ------------------------------------------------------------------ the merge's walk over y_ag
def _merge_by_blocks(g, keys_by_rank, k=512):
    """ix_dm0.cpp: for q = 0 .. C - 1 read y_ag block ag_blk(q) (records (key, core) in local order); exact top-k of
    the union, ties in (chip, core, local) order -> T_g and the per-rank tie quotas"""
    y_ag = {g.ag_blk(q): keys_by_rank[q] for q in range(g.n)}
    mkey, mwho = [], []
    for q in range(g.n):
        for kk, core in y_ag[g.ag_blk(q)]:
            mkey.append(kk)
            mwho.append((q << 8) | core)
    if len(mkey) <= k:
        return 0, {}
    srt = sorted(mkey, reverse=True)
    T = srt[k - 1]
    k_rem = k - sum(1 for v in mkey if v > T)
    quota = {}
    for v, w in zip(mkey, mwho):
        if v == T and k_rem > 0:
            k_rem -= 1
            quota[w] = quota.get(w, 0) + 1
    return T, quota


@pytest.mark.parametrize("R,C", [(1, 2), (2, 4)])
@pytest.mark.parametrize("ties", [0, 64])
def test_merge_rank_order(R, C, ties):
    g = SP.Geometry(R, C)
    sm = DS.ShardMap(g.n)
    gen = torch.Generator().manual_seed(11 + ties)
    n_closed = 9000
    scores = torch.randn(n_closed, generator=gen)
    if ties:
        scores[torch.randint(0, n_closed, (ties,), generator=gen)] = scores[torch.randint(0, n_closed, (ties,),
                                                                                         generator=gen)]  # fmt: skip
    sel, T_g, local, _ = DS.sharded_select(scores, sm, 64)
    keys = DS.f32_key(scores)
    by_rank = []
    for r in range(g.n):  # each chip's records: its local top-512 in (core, local) order
        ids = local[r]
        lq = (ids // 32 // g.n) * 32 + ids % 32
        order = DS.core_of_pool(lq, 64) * (1 << 24) + DS.local_index_of_pool(lq, 64)
        srt = torch.sort(order).indices
        by_rank.append([(int(keys[ids[i]]), int(DS.core_of_pool(lq[i], 64))) for i in srt])
    T, quota = _merge_by_blocks(g, by_rank)
    assert T == T_g
    # the selected set from (T_g, the quotas) in the records' order = sharded_select's
    got = []
    for r in range(g.n):
        q_left = {w: v for w, v in quota.items() if (w >> 8) == r}
        ids = local[r]
        lq = (ids // 32 // g.n) * 32 + ids % 32
        order = DS.core_of_pool(lq, 64) * (1 << 24) + DS.local_index_of_pool(lq, 64)
        for i in torch.sort(order).indices.tolist():
            kk, core = int(keys[ids[i]]), int(DS.core_of_pool(lq[i], 64))
            w = (r << 8) | core
            if kk > T or (kk == T and q_left.get(w, 0) > 0):
                if kk == T:
                    q_left[w] -= 1
                got.append(int(ids[i]))
    assert sorted(got) == sel.tolist()


# ------------------------------------------------------------------ the exchange emulation (fp32)
def _parts_factory(n_chips, D, seed):
    """per chip a random set of selected K rows (its own pools); parts_fn = the chip's partial over them"""
    gen = torch.Generator().manual_seed(seed)
    rows = [torch.randn(int(torch.randint(40, 90, (1,), generator=gen)), D, generator=gen) for _ in range(n_chips)]
    scale = 1.0 / math.sqrt(576.0)

    def parts_fn(rank, q_tile):
        S = (q_tile @ rows[rank].T) * scale
        m = S.max(dim=1).values
        E = torch.exp(S - m[:, None])
        return m, E.sum(dim=1), E @ rows[rank]

    return rows, parts_fn, scale


@pytest.mark.parametrize("R,C", [(1, 2), (2, 4)])
@pytest.mark.parametrize("junk", [None, "random"])
def test_emulate_exchange(R, C, junk):
    g = SP.Geometry(R, C)
    D = 64
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(64, D, generator=gen) * 4
    rows, parts_fn, scale = _parts_factory(g.n, D, 5)
    jk = None
    if junk == "random":  # the receive tiles' stale rows: any finite values must not leak into the owner's rows
        jk = (torch.randn(32, generator=gen) * 50, torch.rand(32, generator=gen) * 1e3, torch.randn(32, D, generator=gen))
    out = SP.emulate_exchange(g, q, parts_fn, junk=jk)
    K = torch.cat(rows)
    ref = torch.softmax((q.double() @ K.double().T) * scale, dim=-1) @ K.double()
    hpc = 64 // g.n
    for k in range(g.n):
        o = out[k].double()
        assert o.shape == (hpc, D)
        err = float((o - ref[hpc * k : hpc * (k + 1)]).abs().max())
        assert err < 2e-5, (k, err)
    # deterministic (the fixed combine order): bitwise equal on a rerun
    out2 = SP.emulate_exchange(g, q, parts_fn, junk=jk)
    assert all(torch.equal(out[k], out2[k]) for k in range(g.n))


# ------------------------------------------------------------------ rev 59: the merge helpers, q_asm, RS1, the probe
def _lists(gen, n_chips, per=512, ties=0, span=1 << 20):
    """per chip a candidate list [(key, core)] in (core, local) order, keys with optional deliberate ties"""
    pool = torch.randint(1 << 30, (1 << 30) + span, (n_chips * per,), generator=gen)
    if ties:
        pool[torch.randint(0, pool.numel(), (ties,), generator=gen)] = pool[torch.randint(0, pool.numel(), (ties,),
                                                                                         generator=gen)]  # fmt: skip
    out = []
    for q in range(n_chips):
        cores = torch.sort(torch.randint(0, 64, (per,), generator=gen)).values
        out.append([(int(pool[q * per + i]), int(cores[i])) for i in range(per)])
    return out


@pytest.mark.parametrize("n_chips", [2, 8])
@pytest.mark.parametrize("ties,span", [(0, 1 << 20), (200, 1 << 20), (0, 300), (0, 40)])
def test_helper_merge_equals_serial(n_chips, ties, span):
    gen = torch.Generator().manual_seed(17 + ties + span + n_chips)
    lists = _lists(gen, n_chips, ties=ties, span=span)
    for rank in range(n_chips):
        a, b = SP.helper_merge(lists, rank), SP.serial_merge(lists, rank)
        assert a["T_g"] == b["T_g"] and a["core_quota"] == b["core_quota"], (rank, a["T_g"], b["T_g"])
        assert sum(a["helper_quota"]) == a["k_rem"] and a["messages"] == SP.merge_msgs() - 1  # (11 slots: 10 messages)
    # the replicas: every chip computes the same T_g and helper quotas (its own core quotas differ)
    res = [SP.helper_merge(lists, r) for r in range(n_chips)]
    assert len({(x["T_g"], tuple(x["helper_quota"])) for x in res}) == 1


@pytest.mark.parametrize("n_chips", [2, 8])
def test_helper_merge_few_candidates(n_chips):
    gen = torch.Generator().manual_seed(5)
    lists = _lists(gen, n_chips, per=512 // n_chips - 3)  # m < k: done after round 0, everything selected
    a = SP.helper_merge(lists, 0)
    assert a["done"] and a["T_g"] == 0 and a["messages"] == 2 and SP.serial_merge(lists, 0)["T_g"] == 0


def test_qasm_assembly_offsets():
    """ix_dm1.cpp: tile g of token t = pieces of ranks 4 g + j (y_ag blocks ag_blk) -> q_asm page (g T + t) 16 + d"""
    g = SP.Geometry(2, 4)
    gen = torch.Generator().manual_seed(9)
    T = 3
    q = torch.randn(T, 64, 512, generator=gen)
    y_ag = {}  # (page, offset) -> 256 elements (one 512-B piece of tile d)
    for t in range(T):
        for k in range(8):
            for d in range(16):
                tile = torch.zeros(32, 32)
                tile[:8] = q[t, 8 * k : 8 * k + 8, 32 * d : 32 * d + 32]
                y_ag[SP.ag_q_page(4, T, g.ag_blk(k), t, d)] = SP.q_piece(SP.tile_to_faces(tile))
    pages = set()
    for t in range(T):
        for tile_id in (0, 1):
            for d in range(16):
                pcs = [y_ag[SP.ag_q_page(4, T, g.ag_blk(tile_id * 4 + j), t, d)] for j in range(4)]
                page = (tile_id * T + t) * 16 + d  # dsa_shard.hpp qasm_page
                assert page not in pages and page < 2 * T * 16
                pages.add(page)
                got = SP.faces_to_tile(SP.assemble_q_tile(pcs))
                assert torch.equal(got, q[t, 32 * tile_id : 32 * tile_id + 32, 32 * d : 32 * d + 32])


@pytest.mark.parametrize("otb", [4096, 2048])
def test_rs1_message_fits_36_pages(otb):
    """att_dsa_shard_dm1.cpp: the RS round-1 message = DT O tiles (otb / 2048 pages each) + ST stats tiles (2 pages)"""
    o_pages, s_pages = otb // 2048, 2
    used = SP.DT * o_pages + SP.ST * s_pages
    assert used <= SP.RS_CHUNKS * SP.RS_PPC == 36 and (otb == 2048 or used == 36)
    g = SP.Geometry(2, 4)
    assert g.streams(4)["rs1"]["width"] == 384 and g.streams(4)["rs1"]["chunks"] == 12


def test_decode_probe():
    """a synthetic probe tensor (cycle marks at 1350 MHz) -> the spans"""
    p = torch.zeros(240, 16, dtype=torch.int64)
    base = 4_000_000_000  # (near the 32-bit wrap: the decoder takes differences mod 2^32)
    cyc = lambda us_: (base + int(us_ * 1350)) & 0xFFFFFFFF  # noqa: E731
    pre = [cyc(-50 + i) for i in range(5)]  # mhc_pre's marks first on the root NCRISC
    att = [0, 40, 120, 160, 175, 230, 245, 270, 271]
    p[132, : 5 + 9] = torch.tensor(pre + [cyc(v) for v in att])
    p[0, :8] = torch.tensor([cyc(-60 + i) for i in range(4)] + [cyc(v) for v in (0, 30, 150, 260)])
    p[220, :6] = torch.tensor([cyc(v) for v in (-70, -60, -40, 38, 80, 118)])
    p[221, :5] = torch.tensor([cyc(v) for v in (-70, -50, -45, 80, 95)])
    for row, (o, l_) in zip((236, 237, 238, 239), ((41, 60), (61, 79), (125, 172), (232, 244))):
        p[row, :5] = torch.tensor([cyc(v) for v in (-30, o, o + 5, o + 6, l_)])
    out = SP.decode_probe(p, line=4)
    sp_ = out["span"]
    assert sp_["attention"] == 271 and sp_["local_select"] == 40 and sp_["merge"] == 38 and sp_["ag_total"] == 42
    assert sp_["q_asm"] == 15 and sp_["rs1_open_to_landed"] == 47 and sp_["pass1"] == 55
    assert out["lanes"]["ag2"]["landed"] == 79 and out["root_br"]["pass0_pushed"] == 150


# ------------------------------------------------------------------ rev 60 (W): token lanes at t > 4
@pytest.mark.parametrize("R,C", MESHES)
@pytest.mark.parametrize("t", [8, 16, 32])
@pytest.mark.parametrize("L", [4, 8])
def test_lane_plan(R, C, t, L):
    """L = 4 / 8 lanes (the lane cores TPMlaShard finds after its stream lanes: 4 on the d0 / host-8chip grids; 8 with more
    free cores) at TP2 / TP8: <= TL_MAX tokens per lane, the 2 LINE helpers per lane, the lane-local slots distinct per
    lane, the AG producer count = per token (its lane's chunk cores + program) + 1"""
    g = SP.Geometry(R, C)
    assert SP.lanes_reasons(t, L, 64, g.L) == []
    slots = SP.mailbox_slots(t, L)
    assert sorted(slots) == list(range(min(L, t)))
    for lane, js in slots.items():
        assert js == list(range(len(js))) and len(js) <= SP.TL_MAX
        assert SP.lane_tokens(t, lane, L) == len(js)
    assert SP.ag_producers(t, 64, L) == t * (64 // L + 1) + 1
    assert SP.ag_producers(4, 64, 1) == 4 * 65 + 1  # (the t <= 4 path: rev 59's count)
    # the refusals: too few lanes for t (TL_MAX), too many for the TP8 helpers, t past T_MAX
    assert SP.lanes_reasons(32, 2, 64, g.L) and SP.lanes_reasons(64, 8, 64, g.L)
    assert bool(SP.lanes_reasons(t, 16, 64, g.L)) == (g.L > 1)  # 16 lanes: 4 chunk cores < 8 TP8 helpers
    # the stream asserts at t (fabric_stream.py): one lane per stream -> chunks % 1, ARR words lanes x n <= 16
    for name, sp_ in g.streams(t).items():
        n = g.axis_len(sp_["axis"])
        assert 1 * n <= 16 and sp_["chunks"] % 1 == 0
        assert sp_["width"] // 32 % 2 == 0  # page pairs (fs_send NB % 2)
    assert all(v <= SP.PKT_HDRS_PER_RISC for r in range(R) for c in range(C) for v in g.header_budget(r, c, t).values())


def _token_keys(gen, n_closed, ties):
    sc = torch.randn(n_closed, generator=gen)
    if ties:
        idx = torch.randint(0, n_closed, (ties,), generator=gen)
        sc[idx] = sc[torch.randint(0, n_closed, (ties,), generator=gen)]
    return sc


@pytest.mark.parametrize("C", [2, 8])
@pytest.mark.parametrize("t,L", [(8, 4), (16, 4), (32, 4), (32, 8)])
@pytest.mark.parametrize("ties", [0, 300])
def test_lane_merge_T_g(C, t, L, ties):
    """the t > 4 lanes emulation: per token on its lane's 64 / L chunk cores, every chip's local top-512 records in
    (lane-local core, local) order and the helpers' merge give the SAME T_g as the t <= 4 path (64 chunk cores: one
    lane) and as tt/dsa_shard.py sharded_select; the replicas agree; the per-core quotas account for the k_rem ties"""
    sm = DS.ShardMap(C)
    gen = torch.Generator().manual_seed(1000 + 7 * C + t + ties)
    for tt in range(t):
        n_closed = int(torch.randint(2052 // 4 + 64, 40000, (1,), generator=gen))
        sc = _token_keys(gen, n_closed, ties)
        keys = DS.f32_key(sc)
        lane, j = SP.lane_of(tt, L)
        assert j < SP.TL_MAX
        w = SP.emulate_lane_merge(keys, C, 64, L)
        ref = SP.emulate_lane_merge(keys, C, 64, 1)  # the t <= 4 path (rev 59: 64 chunk cores)
        _, T_sel, _, _ = DS.sharded_select(sc, sm, 64 // L)
        _, T_sel64, _, _ = DS.sharded_select(sc, sm, 64)
        assert w["T_g"] == ref["T_g"] == T_sel == T_sel64, (tt, w["T_g"], ref["T_g"], T_sel, T_sel64)
        # every record's core is lane-local; quotas only on cores of the chip's own list, summing to the chip's share
        for r in range(C):
            assert all(0 <= core < 64 // L for _, core in w["lists"][r])
            n_eq = sum(1 for kk, _ in w["lists"][r] if kk == w["T_g"])
            assert sum(w["core_quota"][r].values()) <= n_eq
        if w["T_g"]:  # selected = keys > T_g + the quotas: exactly k over the chips
            above = sum(1 for r in range(C) for kk, _ in w["lists"][r] if kk > w["T_g"])
            assert above + sum(sum(q.values()) for q in w["core_quota"]) == 512


def test_lane_merge_tie_order_replicated():
    """the tie order is the same on every chip: the per-chip lists are built from the same (lane-local core, local)
    rule, so each chip's quota walk sees identical lists; a deliberate all-ties token splits k_rem in chip order"""
    C, L = 8, 8
    keys = torch.full((9000,), 1 << 31, dtype=torch.int64)  # every key equal: T_g = that key, ties everywhere
    w = SP.emulate_lane_merge(keys, C, 64, L)
    assert w["T_g"] == 1 << 31
    tot = [sum(q.values()) for q in w["core_quota"]]
    assert sum(tot) == 512 and tot == sorted(tot, reverse=True)  # chip order: the first chips take the quota
