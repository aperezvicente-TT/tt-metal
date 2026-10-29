# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK mHC phases, pure-python checks (no ttnn, no device): tt/pmk_mhc.Plan for both geometries (TP8 256 SP rows
resident, TP2 1,024 SP rows streamed) -- the contract's CB index ranges (prefill_megakernel.md §4b: pre 0..20, post
21..28), uniform-address CBs, L1 bytes under the 1,026 KB owner budget, the placement inside the p300 grid, the
program-wide cumulative counter protocol over 3 layers (no resets), and the oracle: the pre + post composition on random streams equals the
modeling formula (tt/hyperconnection.py docstring) and the kernels' packed K / L coefficient layout reproduces the
post mix. Run: python3 models/autoports/zai_org_glm_5_3_flash/tests/s16/pmk_mhc_test.py"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5)))
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_mhc as M  # noqa: E402

BUDGET = 1_026 * 1024  # contract §4: CB budget per owner core
ALLOCATABLE = 1_461_120
MM_ONLY_NEEDS = 588 * 1024  # contract §4 member P3 peak minus the residual: x-hat store share 160 + MM CBs
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def test_plan(sp):
    p = M.Plan(sp_rows=sp)
    idx = {name: i for name, i, *_ in p.cbs()}
    pre = [n for n in idx if n not in ("Y", "PCOEF", "PSTAGE", "DBG")]
    check(f"sp{sp}: pre CBs in 0..20", all(idx[n] in M.CB_PRE_RANGE for n in pre))
    check(f"sp{sp}: post CBs in 21..28 (+ DBG 24 with the probe)", all(idx[n] in M.CB_POST_RANGE for n in ("Y", "PCOEF", "PSTAGE"))
          and M.CB["DBG"] in M.CB_POST_RANGE)  # fmt: skip
    check(f"sp{sp}: aliases X32 / H16 inside the pre range", M.CB["X32"] == 3 and M.CB["H16"] == 12)
    uni = {n for n, _, _, _, r in p.cbs() if r == "all"}
    check(f"sp{sp}: remote-written CBs uniform (GATHER, MSG, PCOEF)", uni == {"GATHER", "MSG", "PCOEF"})
    l1 = p.l1_bytes()
    check(f"sp{sp}: member CB bytes {l1['member']:,} <= budget {BUDGET:,}", l1["member"] <= BUDGET)
    check(f"sp{sp}: root CB bytes {l1['root']:,} <= budget", l1["root"] <= BUDGET)
    # the MM phases on a member: residual + lane-landed y + the MM CBs, with the dead pre CBs aliased
    live = l1["member"] - l1["member_dead_during_mm"]
    check(f"sp{sp}: member live-during-MM {live:,} + MM needs {MM_ONLY_NEEDS:,} <= allocatable {ALLOCATABLE:,}",
          live + MM_ONLY_NEEDS <= ALLOCATABLE, f"dead / aliasable {l1['member_dead_during_mm']:,}")  # fmt: skip
    sems = {M.SEM_G1, M.SEM_G2, M.SEM_M1, M.SEM_M2, p.sem_xh, p.sem_y}
    check(f"sp{sp}: semaphore ids distinct, < 16, none = 12 (the family's XS_FREE)", len(sems) == 6 and max(sems) < 16
          and 12 not in sems, sorted(sems))  # fmt: skip
    check(f"sp{sp}: contract ids (pre 0..3, FS_ARRIVE 15, FS_REDUCED 14)", p.sem_xh == 15 and p.sem_y == 14 and
          {M.SEM_G1, M.SEM_G2, M.SEM_M1, M.SEM_M2} == {0, 1, 2, 3})  # fmt: skip
    pl = p.placement()
    cores = list(pl["members"].values()) + list(pl["roots"].values())
    check(f"sp{sp}: 72 distinct cores inside the {p.grid} grid", len(set(cores)) == 72 and
          all(0 <= x < p.grid[0] and 0 <= y < p.grid[1] for x, y in cores))  # fmt: skip
    check(f"sp{sp}: member columns share a slab (member (b, j) at x = j: multicast-ready)",
          all(xy[0] == j for (b, j), xy in pl["members"].items()))  # fmt: skip
    cp = p.counter_protocol(layers=3)
    check(
        f"sp{sp}: cumulative counter protocol over 3 layers, no resets (member waits {cp['waits_member']}, root "
        f"{cp['waits_root']}, holder FS_ARRIVE +{cp['holder_per_layer']} per layer)",
        cp["holder_per_layer"] == 8 * p.nsub * p.nrb,
    )
    est = p.estimate_us()
    check(f"sp{sp}: estimate per sublayer {est['per_sublayer']['total']:.0f} us, FPU share "
          f"{est['per_sublayer']['fpu_busy_share']:.2f}", est["per_sublayer"]["total"] > 0)  # fmt: skip
    check(
        f"sp{sp}: nrb {p.nrb}, resident {p.resident}", (p.nrb, p.resident) == ((1, True) if sp == 256 else (4, False))
    )


# ============================================================================== the member -> root data path model
class CbModel:
    """One core's circular buffers as the kernels see them: reserve / push (writer) and wait / pop (reader) in pages
    with the fifo wrap, the packer's per-reserve tile pointer (pack_tile bumps it by one page; push resets it; an
    out-of-order pack to tile index i writes page i without a bump) and the L1 address of every page, so a pack or a
    remote write that leaves its CB is caught (the rev 4 defect: three bumping packs into the one-page COEF)."""

    def __init__(self, cbs: list, base: int = 0x10000):
        self.cb, addr = {}, base
        for name, idx, page, pages, _roles in cbs:
            self.cb[name] = dict(base=addr, page=page, pages=pages, size=page * pages, wr=0, rd=0, tile=0, fill=0)
            addr += page * pages
        self.end = addr

    def addr(self, name, page_index=0):  # L1 byte address of the write pointer (+ page_index pages)
        c = self.cb[name]
        return c["base"] + c["wr"] + page_index * c["page"]

    def reserve(self, name, n):
        c = self.cb[name]
        assert n <= c["pages"], (name, n)
        assert c["fill"] + n <= c["pages"], f"{name}: reserve {n} with {c['fill']} unconsumed (would block)"
        return c["base"] + c["wr"]

    def push(self, name, n):
        c = self.cb[name]
        c["wr"] = (c["wr"] + n * c["page"]) % c["size"]
        c["tile"] = 0
        c["fill"] += n

    def wait(self, name, n):
        c = self.cb[name]
        assert c["fill"] >= n, f"{name}: wait {n} with {c['fill']} pushed (would block)"
        return c["base"] + c["rd"]

    def pop(self, name, n):
        c = self.cb[name]
        assert c["fill"] >= n
        c["rd"] = (c["rd"] + n * c["page"]) % c["size"]
        c["fill"] -= n

    def pack(self, name, out_of_order: bool = False, tile_index: int = 0):
        """pack_tile: returns the byte address the packer writes; raises if it leaves the CB (into the next CB)"""
        c = self.cb[name]
        off = c["wr"] + (tile_index * c["page"] if out_of_order else c["tile"])
        if not out_of_order:
            c["tile"] += c["page"]
        if off + c["page"] > c["size"]:
            hit = [n for n, d in self.cb.items() if d["base"] <= c["base"] + off < d["base"] + d["size"]]
            raise AssertionError(f"pack into {name} leaves the CB: page offset {off // c['page']} -> {hit}")
        return c["base"] + off


def root_round_rev4(m: CbModel):
    """rev 4's root_coeffs pack sequence (three plain packs, one push)"""
    m.reserve("COEF", 1)
    for _ in range(3):
        m.pack("COEF")
    m.push("COEF", 1)


def root_round_rev5(m: CbModel, verified_attempts=(3, 2)):
    """rev 5's root_coeffs: out-of-order packs to page 0 only (verified packs: attempts + 1 each), one push"""
    m.reserve("COEF", 1)
    for a in verified_attempts:
        for _ in range(a + 1):
            assert m.pack("COEF", out_of_order=True, tile_index=0) == m.addr("COEF")
    m.push("COEF", 1)


def test_data_path(sp: int, layers: int = 2):
    """the member -> root data path of every (layer, sublayer, block) round in CB-pointer terms, both roles: the
    member's partial sends land at the root's GATHER base + slot pages (the root reserves / pushes / pops exactly the
    ring per round, so every round's gather starts at the base); the root's PCOEF block k is at base + 8 k pages on
    the root and on the member (uniform address: both reserve NRB x 8 once per sublayer, the member pops per block);
    RED / PLANE / COEF / INVR are one-page rings (always the base); X / Y / OUT / H / PART / S2 wrap per block; the
    root's coefficient pack stays inside COEF (rev 5) and the rev 4 sequence is caught (its packs 2 / 3 = SB)."""
    p = M.Plan(sp_rows=sp, y_lane=False)
    cbs = p.cbs()
    root = CbModel([c for c in cbs if c[4] in ("root", "all")])
    member = CbModel([c for c in cbs if c[4] in ("member", "all")])
    for n in ("GATHER", "MSG", "PCOEF"):  # the uniform-address CBs: same offsets on both roles (first three)
        assert root.cb[n]["base"] == member.cb[n]["base"], n
    # the rev 4 defect, reproduced: the third pack lands in SB (the root's next CB)
    try:
        root_round_rev4(CbModel([c for c in cbs if c[4] in ("root", "all")]))
        assert False, "rev 4's three packs must leave the one-page COEF"
    except AssertionError as e:
        assert "SB" in str(e), e
    rc = CbModel([c for c in cbs if c[4] in ("root", "all")])
    assert rc.addr("COEF") + M.FP32_T == rc.addr("SB") and rc.addr("COEF") + 2 * M.FP32_T == rc.addr("SB", 1)
    gather_base, pcoef_base = root.addr("GATHER"), root.addr("PCOEF")
    root.reserve("PLANE", 1)  # init(): zeroed, never pushed
    root.reserve("SCR", 2)
    root.reserve("PCOEF", p.nrb * 2 * M.N_STREAMS)
    member.reserve("PRE", M.N_STREAMS)  # member init(): zeroed
    member.reserve("INV", 1)
    for layer in range(layers):
        for sub in range(p.nsub):
            member.reserve("PCOEF", p.nrb * 2 * M.N_STREAMS)  # the member's per-sublayer reserve (pushed per block)
            assert member.addr("PCOEF") == pcoef_base
            root.reserve("SB", 3)  # rev 6: the stream part / prelude pushes THIS sublayer's scale / bias / identity
            root.push("SB", 3)
            root.wait("SB", 3)
            for k in range(p.nrb):
                # member: x block (streamed) -> phase 1 -> PART; the sends use the root's GATHER base + slot pages
                if not p.resident or (layer == 0 and sub == 0 and k == 0):
                    assert member.reserve("X", M.X_TILES) == member.addr("X") == member.cb["X"]["base"]
                    member.push("X", M.X_TILES)
                member.reserve("PART", 2)
                member.push("PART", 2)
                member.wait("PART", 2)
                for slot in range(M.P):
                    assert gather_base + slot * 2 * M.FP32_T == member.addr("GATHER") + slot * 2 * M.FP32_T
                member.pop("PART", 2)
                # root round 1: the whole 2P ring per round -> the gather is at the base every round
                assert root.reserve("GATHER", 2 * M.P) == gather_base
                root.push("GATHER", 2 * M.P)
                assert root.wait("GATHER", 2 * M.P) == gather_base
                root.pop("GATHER", 2 * M.P)
                assert root.reserve("RED", 1) == root.cb["RED"]["base"]
                root.push("RED", 1)
                root.wait("RED", 1)
                assert root.reserve("PLANE", 1) == root.cb["PLANE"]["base"]
                root.push("PLANE", 1)
                root.pop("RED", 1)
                root.wait("PLANE", 1)
                root_round_rev5(root)
                root.pop("PLANE", 1)
                assert root.wait("COEF", 1) == root.cb["COEF"]["base"]
                assert root.addr("PCOEF") == pcoef_base  # the root never pushes PCOEF: block k at base + 8 k pages
                k_tiles = pcoef_base + k * 2 * M.N_STREAMS * M.FP32_T
                root.pop("COEF", 1)
                # member: pre message -> PRE / collapse -> H / S2 -> the S2 send (GATHER base + slot pages)
                assert member.reserve("PRE", M.N_STREAMS) == member.cb["PRE"]["base"]
                member.push("PRE", M.N_STREAMS)
                member.wait("PRE", M.N_STREAMS)
                member.pop("PRE", M.N_STREAMS)
                if not p.resident:
                    member.pop("X", M.X_TILES)
                member.reserve("H", M.CPC)
                member.push("H", M.CPC)
                member.reserve("S2", 1)
                member.push("S2", 1)
                member.wait("S2", 1)
                for slot in range(M.P):
                    assert gather_base + slot * M.FP32_T == member.addr("GATHER") + slot * M.FP32_T
                member.pop("S2", 1)
                # root round 2
                assert root.reserve("GATHER", 2 * M.P) == gather_base
                root.push("GATHER", 2 * M.P)
                root.wait("GATHER", 2 * M.P)
                root.pop("GATHER", 2 * M.P)
                assert root.reserve("INVR", 1) == root.cb["INVR"]["base"]
                root.push("INVR", 1)
                root.wait("INVR", 1)
                root.pop("INVR", 1)
                # member: INV -> finish -> OUT; the block's PCOEF pages pushed (the post pops them in order)
                assert member.reserve("INV", 1) == member.cb["INV"]["base"]
                member.push("INV", 1)
                member.wait("INV", 1)
                member.pop("INV", 1)
                member.push("PCOEF", 2 * M.N_STREAMS)
                member.wait("H", M.CPC)
                member.pop("H", M.CPC)
                member.reserve("OUT", M.CPC)
                member.push("OUT", M.CPC)
                member.wait("OUT", M.CPC)
                member.pop("OUT", M.CPC)
                assert member.cb["PCOEF"]["base"] + k * 2 * M.N_STREAMS * M.FP32_T == k_tiles
            root.pop("SB", 3)  # the compute pops after the sublayer's rounds (the next sublayer's reserve waits for it)
            # the post of this sublayer: per block the front of PCOEF is block k's pages, y / x stream per block
            for k in range(p.nrb):
                assert member.wait("PCOEF", 2 * M.N_STREAMS) == pcoef_base + k * 2 * M.N_STREAMS * M.FP32_T
                if not p.resident:
                    assert member.reserve("X", M.X_TILES) == member.cb["X"]["base"]
                    member.push("X", M.X_TILES)
                assert (
                    member.reserve("Y", M.CPC)
                    == member.cb["Y"]["base"] + (k % (p.nrb if p.y_lane else 1)) * M.CPC * p.y_page
                )
                member.push("Y", M.CPC)
                member.wait("Y", M.CPC)
                for _j in range(M.CPC):
                    member.reserve("PSTAGE", M.N_STREAMS)
                    member.push("PSTAGE", M.N_STREAMS)
                    member.wait("PSTAGE", M.N_STREAMS)
                    member.pop("PSTAGE", M.N_STREAMS)
                member.pop("Y", M.CPC)
                member.pop("PCOEF", 2 * M.N_STREAMS)
                if not p.resident:
                    member.pop("X", M.X_TILES)
    check(f"sp{sp}: member -> root data path model, {layers} layers x {p.nsub} sublayers x {p.nrb} blocks: gather at "
          f"the base every round, PCOEF block k at +8k pages, one-page rings at the base, rev 5 packs inside COEF, "
          f"rev 4's packs 2 / 3 = SB scale / bias, SB pushed / popped per sublayer", True)  # fmt: skip


def test_marker():
    bits = [M.marker_bits(r) for r in range(16)]
    import struct

    vals = [struct.unpack("<f", struct.pack("<I", b))[0] for b in bits]
    check("marker bits: distinct, finite, nonzero fp32 per round (512 + 4 (r + 1)), never the zeroed page",
          len(set(bits)) == 16 and all(v == 512 + 4 * (r + 1) for r, v in enumerate(vals)) and 0 not in bits)  # fmt: skip
    finals = [M.marker_final(r) for r in range(16)]
    check("final (cycle 3) markers distinct from every cycle-1 marker and from each other (sign bit)",
          len(set(bits + finals)) == 32 and all(f & 0x80000000 for f in finals))  # fmt: skip
    check("diag record: tap TAP_DIAG < N_TAPS, DIAG_WORDS words fit the 512 B mailbox (MSG_FLOATS 128)",
          M.TAP_DIAG < M.N_TAPS and M.DIAG_WORDS <= 128 and M.F_SB_PRELUDE == 256)  # fmt: skip
    check("marker / diag planes are free planes of the plane layout (24 scratch, 25 marker, 26 diag < 32)",
          M.N_MIX <= 24 < M.PLANE_MARKER < M.PLANE_DIAG < 32)  # fmt: skip


def test_oracle():
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import reference_step as post_ref
    from models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre import reference_step as pre_ref

    g = torch.Generator().manual_seed(1)
    T, n, D = 32, 4, 4096
    x = torch.randn(T, n * D, generator=g)
    fn_t = torch.randn(n * D, 24, generator=g) * 0.01
    scale = torch.cat([torch.full((4,), 0.5), torch.full((4,), 0.5), torch.full((16,), 0.5)])
    bias = torch.zeros(24)
    bias[8:24] = torch.eye(4).reshape(-1) * 2
    w = 1 + 0.1 * torch.randn(D, generator=g)
    r = pre_ref(x, fn_t, scale, bias, w, hc_eps=1e-6, x_eps=1e-5, h_eps=1e-5, iters=20, matmul_tf32=False)
    # the modeling formula (tt/hyperconnection.py): m = rsqrt(mean x^2 + eps) * (x @ fn^T); pre / post / comb
    m = torch.rsqrt((x.double() ** 2).mean(-1, keepdim=True) + 1e-5) * (x.double() @ fn_t.double())
    z = m * scale.double() + bias.double()
    pre = torch.sigmoid(z[:, :4]) + 1e-6
    check("oracle: pre coefficients = modeling formula", torch.allclose(r["pre"], pre, atol=1e-9))
    comb = r["comb"]  # [T, 4, 4] doubly-stochastic after 20 Sinkhorn iterations
    check("oracle: comb rows / cols sum to 1 (Sinkhorn)", torch.allclose(comb.sum(-1), torch.ones(T, 4, dtype=comb.dtype), atol=1e-3)
          and torch.allclose(comb.sum(-2), torch.ones(T, 4, dtype=comb.dtype), atol=1e-3))  # fmt: skip
    y = torch.randn(T, D, generator=g)
    out = post_ref(y, x.reshape(T, n, D), r["post"], comb)  # [T, 4, D]
    # the kernels' post: out_s = K_s col0 * y + K_s col1 * x_0 + col16 * x_1 + col17 * x_2 + L_s col0 * x_3, with
    # K_s = post_s | C[0][s] | C[1][s] | C[2][s], L_s = C[3][s] (comb[t, i, s] = C[i][s])
    xs = x.reshape(T, n, D).double()
    mine = torch.stack([r["post"][:, s, None] * y.double() + sum(comb[:, i, s, None] * xs[:, i] for i in range(4))
                        for s in range(4)], dim=1)  # fmt: skip
    check("oracle: packed K / L coefficient mix = mhc_post reference", torch.allclose(mine, out, atol=1e-9))
    # composition: the new residual's stream mixing is the modeling X'_j = post_j y + sum_i C[i, j] X_i
    check("oracle: post output shape [T, 4, D]", tuple(out.shape) == (T, 4, D))


if __name__ == "__main__":
    for sp in (256, 1024):
        test_plan(sp)
    for sp in (256, 512, 1024):  # NRB 1 / 2 / 4
        test_data_path(sp)
    test_marker()
    try:
        test_oracle()
    except ImportError as e:  # torch-less CPU: the plan checks alone
        print(f"SKIP oracle (torch missing: {e})")
    print("RESULT", "PASS" if not fails else f"FAIL {fails}")
    sys.exit(1 if fails else 0)
