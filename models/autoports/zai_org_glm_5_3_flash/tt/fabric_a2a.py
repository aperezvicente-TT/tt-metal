# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 DSA option (i): the SPA q / o all-to-all as an in-house FabricStream (doc
pmk/merge/a2a_fs_design.md). FabricStreamA2A2D replaces prefill_tp.MlaPrefill._a2a's two
ttnn.experimental.all_to_all_async_generic reshards (knob PMK_SPA_A2A_FS=1) with page-run lists over the fabric. It is a
COPY of tile pages: bitwise the op's output.

Layouts (per chip, bf16 TILE, DRAM interleaved, D = 512 -> DT = 16 column tiles):
  q (heads -> rows): x [1, Hr, CR, D] (this chip's Hr heads, all CR rows) -> y [1, N Hr, CR / N, D] (all heads, this chip's
      row block). The run of (source s, destination d, head h) = RPB = (CR / N / 32) DT CONSECUTIVE pages on both sides:
      x pages h PPH + block(d) RPB .. -> y pages (slot(s) Hr + h) RPB .. (PPH = (CR / 32) DT pages per head).
  o (rows -> heads): the inverse: x [1, N Hr, CR / N, D] -> y [1, Hr, CR, D]; run (s, d, h): x pages (slot(d) Hr + h) RPB
      .. -> y pages h PPH + block(s) RPB ..
  block(d) = the row block chip d holds = its SP rank r C + c (2x4) / its line position; slot(s) = the head group of
  source s in a row-block chip's heads = THE OP's ORDER: MlaPrefill._a2a runs axis 0 then axis 1, so the gathered heads
  are column-major over the sources: slot(r, c) = c R + r on the 2x4 (NOT the rank r C + c), the position on a line.
  (The design note's "head 8 s' + h" with s' the rank is the rank-major order; the op's is c R + r. Any order is
  correct for the attention, but bitwise vs the op needs the op's.)

Forms:
  "2d"   (the 2x4, FABRIC_1D lines per axis; doc section 2): stage A on axis 0 -- every chip sends its column partner P
         the runs destined to P's ROW: P's own runs land in P's y, the others in P's RELAY store; stage B on axis 1 --
         every chip scatters to each row-mate its own runs and the runs P relayed for it (hop-count unicast, one
         connection per direction). Local runs (s == d) are copied by the A lanes.
  "line" (the pair 1x2, or one axis of a 2-D mesh: every line independent = one all_to_all_async_generic with that
         cluster_axis): stage B alone; local runs by the B lanes.
Lanes: one worker per (chip, link, direction): lane L of a stage = link link_base + L of its axis, both directions; run
h of every transfer goes to lane h % nl. The waiter = B lane 0. Sync words: kernels/fabric_stream/a2a_common.hpp (one
512-B L1 shard per lane core, monotonic counters, zeroed once): CREDITS at launch start ("this chip started": the op's
init barrier), landed counters per source lane (YLAND), the relay counter (RELAY) that gates every forwarded run, DONE,
the 1-hop end-of-launch row barrier (BAR_L / BAR_R: THE PER-HOP DRAIN of fs_common.hpp) on lines of > 2 chips.

A2APlan (pure Python, no device) builds the per-lane run lists; encode() turns them into the kernels' runtime args;
simulate() replays exactly those args on host tensors (tests/s16/test_a2a_fs.py) and ref_op_a2a() is the op's
semantics (torch chunk / cat per axis, in MlaPrefill._a2a's axis order)."""

from __future__ import annotations

import os
import re

import torch

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "fabric_stream")
TB = 2048  # bf16 tile page
NHDR = 4  # a2a_send.cpp NHDR: data headers per direction
HDR_BUDGET = 12  # BH packet-header pool per RISC (fabric-packet-header-pool-per-risc)
MAX_RT_ARGS = 341  # tt-metalium kernel_types.hpp max_runtime_args
CB_STAGE = 0


def _consts(name: str = "a2a_common.hpp") -> dict:
    with open(os.path.join(KDIR, name)) as f:
        return {m.group(1): int(m.group(2)) for m in re.finditer(r"constexpr uint32_t (\w+)\s*=\s*(\d+);", f.read())}


K = _consts()
WORD, WORDS = K["WORD"], K["WORDS"]
GEN_B, GEN_NC, DONE, BAR_L, BAR_R, RELAY = K["GEN_B"], K["GEN_NC"], K["DONE"], K["BAR_L"], K["BAR_R"], K["RELAY"]
CREDIT, NCREDIT, YLAND, NYLAND = K["CREDIT"], K["NCREDIT"], K["YLAND"], K["NYLAND"]
ROUTE_LOCAL = K["ROUTE_LOCAL"]
with open(os.path.join(KDIR, "a2a_send.cpp")) as _f:
    assert f"constexpr uint32_t NHDR = {NHDR};" in _f.read(), "a2a_send.cpp NHDR"


B_LANES = ((0, 0), (0, 1))  # the default lane cores (logical): B lane L / A lane L (any free cores: a standalone program)
A_LANES = ((0, 2), (0, 3))


def lane_cores(form: str, b_lanes=B_LANES, a_lanes=A_LANES) -> list:
    """the lane cores of a form in the sync tensor's order (B lanes, then the 2x4's A lanes)"""
    return [tuple(v) for v in b_lanes] + ([tuple(v) for v in a_lanes] if form == "2d" else [])


def knob() -> bool:
    """PMK_SPA_A2A_FS=1: prefill_tp.MlaPrefill._a2a through FabricStreamA2A2D (default off: the op)"""
    return os.environ.get("PMK_SPA_A2A_FS", "0") == "1"


def mesh_form(shape) -> str | None:
    """the form for a mesh shape: the 2x4 -> "2d", the pair (1x2 / 2x1) -> "line"; anything else None (the op)"""
    R, C = int(shape[0]), int(shape[1])
    if R == 2 and C == 4:
        return "2d"
    if (R, C) in ((1, 2), (2, 1)):
        return "line"
    return None


def row_barrier_on() -> bool:
    """the 1-hop end-of-launch row barrier (GLM53_FS_ROW_BARRIER=0 = the fabric_stream kill switch, A/B only)"""
    return os.environ.get("GLM53_FS_ROW_BARRIER", "1") == "1"


# ----------------------------------------------------------------------------------------------------------- the plan
class A2APlan:
    """The page-run lists of one a2a direction on an R x C mesh (pure Python).

    lanes[(r, c)][(role, L)] = dict(axis, up, dn, runs, credits, relay_per); waiter[(r, c)] = dict(wait={word: per launch},
    lanes, barrier). A run = dict(src "x" | "relay", src_page, route 0 up / 1 down / ROUTE_LOCAL, hops, dst_rc, dst "y" |
    "relay", dst_page, sig None | (rc, role, L, word), need (relay counter, 1-based, 0 = none), cpos (axis position of
    the credit to wait for, None = local)); credits = [(dir, hops, target rc, word)] (the target = the same-role lane L)."""

    def __init__(self, R: int, C: int, *, hr: int, rows: int, dim: int = 512, heads_to_rows: bool = True,
                 form: str = "2d", axis: int | None = None, nl: int = 2, order: str = "rr",
                 barrier: bool | None = None):  # fmt: skip
        self.R, self.C, self.hr, self.rows, self.dim, self.q = int(R), int(C), int(hr), int(rows), int(dim), bool(heads_to_rows)
        self.form, self.nl, self.order = form, int(nl), order
        assert form in ("2d", "line"), form
        assert order in ("rr", "dst"), order
        if form == "2d":
            assert self.R == 2 and self.C >= 2, ("the 2-D form: a column of 2 (one relay partner)", R, C)
            self.axis_b, self.n = 1, self.C  # stage B along the row
            self.ntot = self.R * self.C
        else:
            if axis is None:
                assert self.R == 1 or self.C == 1, "a 2-D mesh: pass axis= for the line form"
                axis = 1 if self.C > 1 else 0
            self.axis_b = int(axis)
            self.n = self.R if self.axis_b == 0 else self.C
            self.ntot = self.n
        assert self.n > 1, ("no chips along the axis", R, C, axis)
        assert self.n <= NCREDIT and self.n * self.nl + (self.nl if form == "2d" else 0) <= NYLAND, (self.n, self.nl)
        assert self.hr % self.nl == 0, ("heads per chip divide over the lanes", self.hr, self.nl)
        self.dt = self.dim // 32
        assert self.dim % 32 == 0 and self.rows % (32 * self.ntot) == 0, ("rows per chip: whole tiles", rows, self.ntot)
        self.rl = self.rows // self.ntot
        self.pph = (self.rows // 32) * self.dt  # pages per head of the head-sharded side
        self.rpb = (self.rl // 32) * self.dt  # pages per run (one head x one row block)
        self.barrier = (self.n > 2 and row_barrier_on()) if barrier is None else bool(barrier)
        self.chips = [(r, c) for r in range(self.R) for c in range(self.C)]
        self.roles = [("B", L) for L in range(self.nl)] + ([("A", L) for L in range(self.nl)] if form == "2d" else [])
        self.relay_pages = (self.n - 1) * self.hr * self.rpb if form == "2d" else 0
        self._build()

    # -- geometry
    def pos(self, rc) -> int:
        """position along the B axis (the row on the 2x4, the line otherwise)"""
        return rc[self.axis_b]

    def line_of(self, rc) -> list:
        return [(rc[0], q) if self.axis_b == 1 else (q, rc[1]) for q in range(self.n)]

    def block(self, rc) -> int:
        return rc[0] * self.C + rc[1] if self.form == "2d" else self.pos(rc)

    def slot(self, rc) -> int:
        return rc[1] * self.R + rc[0] if self.form == "2d" else self.pos(rc)

    def x_shape(self) -> tuple:
        return (1, self.hr, self.rows, self.dim) if self.q else (1, self.hr * self.ntot, self.rl, self.dim)

    def y_shape(self) -> tuple:
        return (1, self.hr * self.ntot, self.rl, self.dim) if self.q else (1, self.hr, self.rows, self.dim)

    def pages(self, shape) -> int:
        return shape[1] * (shape[2] // 32) * self.dt

    def run_pages(self, s, d, h) -> tuple:
        """(source page on s, destination page on d) of head run h of the transfer s -> d"""
        if self.q:
            return h * self.pph + self.block(d) * self.rpb, (self.slot(s) * self.hr + h) * self.rpb
        return (self.slot(d) * self.hr + h) * self.rpb, h * self.pph + self.block(s) * self.rpb

    def _dests(self, rc) -> list:
        """the line-mates of rc, nearest first (up before down at equal distance)"""
        p = self.pos(rc)
        line = self.line_of(rc)
        return [line[q] for q in sorted((q for q in range(self.n) if q != p), key=lambda q: (abs(q - p), q < p))]

    def _interleave(self, dests, hs) -> list:
        if self.order == "rr":
            return [(d, h) for h in hs for d in dests]
        return [(d, h) for d in dests for h in hs]

    def _route(self, src_rc, dst_rc, axis) -> tuple:
        a_, b_ = src_rc[axis], dst_rc[axis]
        assert src_rc[1 - axis] == dst_rc[1 - axis] and a_ != b_, (src_rc, dst_rc, axis)
        return (0 if b_ > a_ else 1), abs(b_ - a_)

    def _ysig(self, dst_rc, j) -> tuple:
        return (dst_rc, "B", 0, YLAND + j)

    def _build(self):
        nl, hr, n = self.nl, self.hr, self.n
        self.lanes = {rc: {} for rc in self.chips}
        two = self.form == "2d"
        partner = lambda rc: (1 - rc[0], rc[1])  # noqa: E731
        relay_order = {}  # (Y, L) -> [(d, h)]: the relay runs B lane L of Y forwards, in order
        for Y in self.chips:
            p = self.pos(Y)
            dests = self._dests(Y)
            up, dn = n - 1 - p, p
            for L in range(nl):
                hs = [h for h in range(hr) if h % nl == L]
                runs = []
                for d, h in self._interleave(dests, hs):  # own runs
                    sp, dp = self.run_pages(Y, d, h)
                    dr, hops = self._route(Y, d, self.axis_b)
                    runs.append(dict(src="x", src_page=sp, route=dr, hops=hops, dst_rc=d, dst="y", dst_page=dp,
                                     sig=self._ysig(d, p * nl + L), need=0, cpos=self.pos(d)))  # fmt: skip
                relay_per = 0
                if two:
                    P = partner(Y)
                    others = [q for q in range(n) if q != Y[1]]
                    ro = self._interleave(dests, hs)
                    relay_order[(Y, L)] = ro
                    for k, (d, h) in enumerate(ro):  # the partner's runs for my row-mates, from my relay store
                        j = others.index(d[1])
                        _, dp = self.run_pages(P, d, h)
                        dr, hops = self._route(Y, d, 1)
                        runs.append(dict(src="relay", src_page=(j * hr + h) * self.rpb, route=dr, hops=hops, dst_rc=d,
                                         dst="y", dst_page=dp, sig=self._ysig(d, p * nl + L), need=k + 1,
                                         cpos=self.pos(d)))  # fmt: skip
                    relay_per = len(ro)
                else:
                    for h in hs:  # the own block (line form: the B lanes copy it)
                        sp, dp = self.run_pages(Y, Y, h)
                        runs.append(dict(src="x", src_page=sp, route=ROUTE_LOCAL, hops=0, dst_rc=Y, dst="y", dst_page=dp,
                                         sig=None, need=0, cpos=None))  # fmt: skip
                credits = []
                for d in dests:
                    dr, hops = self._route(Y, d, self.axis_b)
                    credits.append((dr, hops, d, CREDIT + p))
                self.lanes[Y][("B", L)] = dict(axis=self.axis_b, up=up, dn=dn, runs=runs, credits=credits, relay_per=relay_per)
        if two:
            for X in self.chips:
                P = partner(X)
                r, c = X
                dr, hops = self._route(X, P, 0)
                others = [q for q in range(n) if q != c]
                for L in range(nl):
                    hs = [h for h in range(hr) if h % nl == L]
                    runs = []
                    for k, (d, h) in enumerate(relay_order[(P, L)]):  # P's forwarding order, so P's needs are k + 1
                        sp, _ = self.run_pages(X, d, h)
                        runs.append(dict(src="x", src_page=sp, route=dr, hops=1, dst_rc=P, dst="relay",
                                         dst_page=(others.index(d[1]) * hr + h) * self.rpb, sig=(P, "B", L, RELAY), need=0,
                                         cpos=P[0]))  # fmt: skip
                    for h in hs:  # P's own runs: straight into P's y
                        sp, dp = self.run_pages(X, P, h)
                        runs.append(dict(src="x", src_page=sp, route=dr, hops=1, dst_rc=P, dst="y", dst_page=dp,
                                         sig=self._ysig(P, n * nl + L), need=0, cpos=P[0]))  # fmt: skip
                    for h in hs:  # the own block
                        sp, dp = self.run_pages(X, X, h)
                        runs.append(dict(src="x", src_page=sp, route=ROUTE_LOCAL, hops=0, dst_rc=X, dst="y", dst_page=dp,
                                         sig=None, need=0, cpos=None))  # fmt: skip
                    self.lanes[X][("A", L)] = dict(axis=0, up=1 if r == 0 else 0, dn=1 if r == 1 else 0, runs=runs,
                                                   credits=[(dr, 1, P, CREDIT + r)], relay_per=0)  # fmt: skip
        # the waiter (B lane 0) of every chip: its YLAND words' per-launch counts
        self.waiter = {rc: dict(wait={}, lanes=len(self.roles), barrier=self.barrier) for rc in self.chips}
        for X in self.chips:
            for lp in self.lanes[X].values():
                for run in lp["runs"]:
                    if run["sig"] is not None and run["sig"][1:3] == ("B", 0) and run["sig"][3] >= YLAND:
                        w = self.waiter[run["sig"][0]]["wait"]
                        w[run["sig"][3]] = w.get(run["sig"][3], 0) + 1
        self._audit()

    def _audit(self):
        """host checks of the fabric resources: one worker per (chip, link, axis, direction); <= 12 packet headers per
        lane BRISC; every relay counter's per-launch count = the relay runs that signal it"""
        for rc in self.chips:
            seen = set()
            for (role, L), lp in self.lanes[rc].items():
                dirs = [d for d, rng in ((0, lp["up"]), (1, lp["dn"])) if rng]
                assert NHDR * len(dirs) <= HDR_BUDGET, ("packet headers per RISC", rc, role, L, dirs)
                for d in dirs:
                    key = (L, lp["axis"], d)
                    assert key not in seen, ("two workers on one EDM channel", rc, key)
                    seen.add(key)
        relay_in = {}
        for X in self.chips:
            for lp in self.lanes[X].values():
                for run in lp["runs"]:
                    if run["sig"] is not None and run["sig"][3] == RELAY:
                        k_ = (run["sig"][0], run["sig"][2])
                        relay_in[k_] = relay_in.get(k_, 0) + 1
        for rc in self.chips:
            for (role, L), lp in self.lanes[rc].items():
                if role == "B":
                    assert relay_in.get((rc, L), 0) == lp["relay_per"], ("relay count", rc, L, relay_in.get((rc, L)), lp["relay_per"])

    # -- the kernels' runtime args
    def encode(self, rc, noc: dict, addr: dict | None = None) -> dict:
        """runtime args of chip rc's lanes: {(role, L): (reader args, sender args without the fabric connections)},
        sender lists zero-padded to one CONN_AT per program (the kernels' compile-time arg: a cache hit never sees
        another arg count). noc: {(role, L): (x, y)} the lane cores' NoC coordinates (the same on every chip); addr:
        dict(sync, x, y, relay) buffer addresses (0 on the host)"""
        addr = addr or {}
        sync, xa, ya, ra = (int(addr.get(k_, 0)) for k_ in ("sync", "x", "y", "relay"))
        wx, wy = noc[("B", 0)]
        wt = self.waiter[rc]
        out = {}
        for (role, L), lp in self.lanes[rc].items():
            rd = [sync, xa, ra, lp["relay_per"], len(lp["runs"])]
            for run in lp["runs"]:
                rd += [int(run["src"] == "relay") | (run["need"] << 1), run["src_page"]]
            waiter = (role, L) == ("B", 0)
            sd = [sync, ya, ra, wx, wy, lp["up"], lp["dn"], int(waiter), int(waiter and wt["barrier"]),
                  wt["lanes"], len(lp["credits"])]  # fmt: skip
            for dr, hops, trc, word in lp["credits"]:
                tx, ty = noc[(role, L)]
                sd += [dr | (hops << 2), tx | (ty << 8), word]
            wl = sorted(wt["wait"].items()) if waiter else []
            sd += [len(wl)] + [v for kv in wl for v in kv]
            sd.append(len(lp["runs"]))
            for run in lp["runs"]:
                assert run["hops"] < 16 and (run["cpos"] is None or run["cpos"] < 15), run
                w0 = run["route"] | (run["hops"] << 2) | (int(run["dst"] == "relay") << 6)
                w0 |= (0 if run["cpos"] is None else run["cpos"] + 1) << 8
                w2 = 0
                if run["sig"] is not None:
                    _, srole, sl, word = run["sig"]
                    sx, sy = noc[(srole, sl)]
                    w0 |= 1 << 12
                    w2 = sx | (sy << 8) | (word << 16)
                sd += [w0, run["dst_page"], w2]
            out[(role, L)] = (rd, sd)
        conn_at = max(len(sd) for _, sd in out.values())
        for k_, (rd, sd) in out.items():
            out[k_] = (rd, sd + [0] * (conn_at - len(sd)))
            assert len(rd) <= MAX_RT_ARGS and conn_at + 2 * 16 <= MAX_RT_ARGS, ("runtime args", rc, k_, len(rd), conn_at)
        return out


# ----------------------------------------------------------------------------------------------------------- the host
def to_pages(t: torch.Tensor) -> torch.Tensor:
    """[H, rows, D] -> [H rows / 32 D / 32, 32, 32] in TILE page order (page = (h rows / 32 + row tile) D / 32 + col tile;
    the faces inside a page are irrelevant: pages move whole)"""
    h, rows, d = t.shape
    return t.reshape(h, rows // 32, 32, d // 32, 32).permute(0, 1, 3, 2, 4).reshape(-1, 32, 32).contiguous()


def from_pages(p: torch.Tensor, shape) -> torch.Tensor:
    h, rows, d = shape
    return p.reshape(h, rows // 32, d // 32, 32, 32).permute(0, 1, 3, 2, 4).reshape(h, rows, d).contiguous()


def ref_op_a2a(xs: dict, R: int, C: int, heads_to_rows: bool, axes=None) -> dict:
    """the op's semantics: MlaPrefill._a2a = all_to_all_async_generic per axis (axes of size > 1, 0 then 1 for q, the
    reverse for o; or the given axes) on {(r, c): [H, rows, D]}; one axis = every line along it: the destination at
    position pd gets cat over the line's sources (in position order) of chunk pd of their out_dim, along in_dim (q: in =
    heads, out = rows; o: the reverse)"""
    if axes is None:
        axes = [a for a, n in ((0, R), (1, C)) if n > 1]
    in_dim, out_dim = (0, 1) if heads_to_rows else (1, 0)
    for a in (axes if heads_to_rows else axes[::-1]):
        new = {}
        for rc in xs:
            line = [(rc[0], q) if a == 1 else (q, rc[1]) for q in range(C if a == 1 else R)]
            pd = rc[a]
            new[rc] = torch.cat([torch.chunk(xs[s], len(line), dim=out_dim)[pd] for s in line], dim=in_dim)
        xs = new
    return xs


def simulate(plan: A2APlan, xs: dict, launches: int = 1, state: dict | None = None, noc: dict | None = None) -> dict:
    """replay the ENCODED runtime args (decoded as a2a_read.cpp / a2a_send.cpp read them) on host tensors: xs = {rc: [H,
    rows, D]} (or a list of such dicts, one per launch) -> {rc: y} of the last launch. Lanes run as an adversarial
    scheduler (the B lanes first, every lane as far as its waits allow); a relay read before its relay counter, a send
    before its credit, a counter short of the waiter's per-launch count, a page written twice or never, or a deadlock
    raise. state: persistent counters / relay stores across calls (the monotonic-counter check)."""
    noc = noc or {("B", L): (1, 10 + L) for L in range(plan.nl)} | {("A", L): (2, 10 + L) for L in range(plan.nl)}
    core_of = {v: k for k, v in noc.items()}
    st = state if state is not None else {}
    cnt = st.setdefault("cnt", {})  # (rc, role, L) -> [WORDS]
    relay = st.setdefault("relay", {})
    for rc in plan.chips:
        for role_l in plan.roles:
            cnt.setdefault((rc,) + role_l, [0] * WORDS)
        if plan.relay_pages and (rc not in relay or relay[rc].shape[0] < plan.relay_pages):  # one store, the largest size
            old = relay.get(rc)
            new = torch.full((plan.relay_pages, 32, 32), float("nan"), dtype=torch.bfloat16)
            if old is not None:
                new[:old.shape[0]] = old
            relay[rc] = new
    seq = xs if isinstance(xs, list) else [xs] * launches
    enc = {rc: plan.encode(rc, noc) for rc in plan.chips}
    ys = None
    for xs_g in seq:
        xp = {rc: to_pages(xs_g[rc]) for rc in plan.chips}
        for rc in plan.chips:
            assert tuple(xs_g[rc].shape) == plan.x_shape()[1:], (rc, tuple(xs_g[rc].shape), plan.x_shape())
        ysh = plan.y_shape()[1:]
        yp = {rc: torch.full((plan.pages((1,) + ysh), 32, 32), float("nan"), dtype=torch.bfloat16) for rc in plan.chips}
        written = {rc: torch.zeros(yp[rc].shape[0], dtype=torch.int32) for rc in plan.chips}
        procs = []
        for rc in plan.chips:
            for role_l, (rd, sd) in enc[rc].items():
                c_ = cnt[(rc,) + role_l]
                procs.append(dict(rc=rc, role=role_l, rd=rd, sd=sd, axis=plan.lanes[rc][role_l]["axis"], k=0,
                                  g_b=c_[GEN_B] + 1, g_nc=c_[GEN_NC] + 1))  # fmt: skip
        g_ = {p_["g_b"] for p_ in procs} | {p_["g_nc"] for p_ in procs}
        assert len(g_) == 1, ("lanes out of step", g_)
        g = g_.pop()
        dec = {}
        for p_ in procs:  # decode (the kernels' order) + the launch's credits
            rd, sd = p_["rd"], p_["sd"]
            relay_per, nr = rd[3], rd[4]
            rruns = [(rd[5 + 2 * k] & 1, rd[5 + 2 * k] >> 1, rd[6 + 2 * k]) for k in range(nr)]
            a = 10
            n_credit = sd[a]
            a += 1
            for k in range(n_credit):
                w0, xy, word = sd[a + 3 * k:a + 3 * k + 3]
                dr, hops = w0 & 3, (w0 >> 2) & 0xF
                trc = _step(p_["rc"], p_["axis"], dr, hops)
                cnt[(trc,) + core_of[(xy & 0xFF, (xy >> 8) & 0xFF)]][word] += 1
            a += 3 * n_credit
            n_wait = sd[a]
            wl = [(sd[a + 1 + 2 * k], sd[a + 2 + 2 * k]) for k in range(n_wait)]
            a += 1 + 2 * n_wait
            ns = sd[a]
            a += 1
            sruns = [tuple(sd[a + 3 * k:a + 3 * k + 3]) for k in range(ns)]
            assert ns == nr, ("reader / sender run counts", p_["rc"], p_["role"], nr, ns)
            dec[id(p_)] = (relay_per, rruns, sruns, wl, sd[7], sd[9])
        def step(p_) -> bool:
            """one run of lane p_ if its waits allow (the reader's relay counter, the sender's credit)"""
            relay_per, rruns, sruns, _, _, _ = dec[id(p_)]
            if p_["k"] >= len(rruns):
                return False
            sel, need, spg = rruns[p_["k"]]
            w0, dpg, w2 = sruns[p_["k"]]
            me = cnt[(p_["rc"],) + p_["role"]]
            if need and me[RELAY] < (g - 1) * relay_per + need:
                return False
            cpos1 = (w0 >> 8) & 0xF
            if cpos1 and me[CREDIT + cpos1 - 1] < g:
                return False
            src = (relay[p_["rc"]] if sel else xp[p_["rc"]])[spg:spg + plan.rpb].clone()
            assert not torch.isnan(src.float()).any(), ("a run read unwritten pages", p_["rc"], p_["role"], p_["k"])
            route, hops, to_relay = w0 & 3, (w0 >> 2) & 0xF, (w0 >> 6) & 1
            drc = p_["rc"] if route == ROUTE_LOCAL else _step(p_["rc"], p_["axis"], route, hops)
            if to_relay:
                assert route != ROUTE_LOCAL
                relay[drc][dpg:dpg + plan.rpb] = src
            else:
                yp[drc][dpg:dpg + plan.rpb] = src
                written[drc][dpg:dpg + plan.rpb] += 1
            if (w0 >> 12) & 1:
                cnt[(drc,) + core_of[(w2 & 0xFF, (w2 >> 8) & 0xFF)]][w2 >> 16] += 1
            p_["k"] += 1
            return True

        # ADVERSARIAL: the consumers (B lanes) run as far as their waits allow after EVERY single producer (A lane) run,
        # so a relay wait one count short reads a row before it landed (NaN in launch 1, the previous launch's later)
        cons = [p_ for p_ in procs if p_["role"][0] == "B"]
        prod = [p_ for p_ in procs if p_["role"][0] != "B"]
        while True:
            moved = False
            for p_ in cons:
                while step(p_):
                    moved = True
            for p_ in prod:
                if step(p_):
                    moved = True
                    break
            if not moved:
                break
        stuck = [(p_["rc"], p_["role"], p_["k"]) for p_ in procs if p_["k"] < len(dec[id(p_)][1])]
        assert not stuck, ("deadlock: lanes blocked on a relay counter / credit", stuck)
        for p_ in procs:  # DONE
            wx, wy = p_["sd"][3], p_["sd"][4]
            cnt[(p_["rc"],) + core_of[(wx, wy)]][DONE] += 1
        for p_ in procs:  # the waiter's checks (exact: no counter overshoots its launch)
            _, _, _, wl, is_waiter, n_lanes = dec[id(p_)]
            if is_waiter:
                me = cnt[(p_["rc"],) + p_["role"]]
                for word, per in wl:
                    assert me[word] == g * per, ("landed count", p_["rc"], word, me[word], g * per)
                assert me[DONE] == g * n_lanes, ("DONE", p_["rc"], me[DONE], g * n_lanes)
                assert sum(me[YLAND:YLAND + NYLAND]) == g * sum(per for _, per in wl), ("an unwaited YLAND word", p_["rc"])
            me = cnt[(p_["rc"],) + p_["role"]]
            for q in range(NCREDIT):  # one credit per peer per launch (or none from a non-peer)
                assert me[CREDIT + q] in (0, g), ("credits", p_["rc"], p_["role"], q, me[CREDIT + q], g)
            me[GEN_B] = me[GEN_NC] = g
        for rc in plan.chips:
            assert bool((written[rc] == 1).all()), ("y pages written != once", rc, int((written[rc] == 0).sum()),
                                                    int((written[rc] > 1).sum()))  # fmt: skip
        ys = {rc: from_pages(yp[rc], ysh) for rc in plan.chips}
    return ys


def _step(rc, axis, dr, hops):
    d = hops if dr == 0 else -hops
    return (rc[0] + d, rc[1]) if axis == 0 else (rc[0], rc[1] + d)


# ---------------------------------------------------------------------------------------------------------- the device
_SYNC_SIG = {}  # id(sync tensor) -> the signature of the instances sharing it (same per-launch counts)


def _ktext(name: str) -> str:
    """kernel source with fs_common.hpp + a2a_common.hpp inlined (SOURCE_CODE kernels are hashed by their text)"""
    with open(os.path.join(KDIR, name)) as f:
        src = f.read()
    for inc in ("fs_common.hpp", "a2a_common.hpp"):
        with open(os.path.join(KDIR, inc)) as f:
            src = src.replace(f'#include "{inc}"', f.read().replace("#pragma once", ""))
    return src


class FabricStreamA2A2D:
    """The SPA q / o all-to-all over the fabric (module docstring). One instance per (heads per chip, chunk rows):
    __call__(x, heads_to_rows) -> a NEW y (the caller owns it; x is only read). Persistent (allocated here, never
    inside a trace): the sync words (one 512-B L1 shard per lane core; `sync=` shares another instance's -- every
    launch of every sharing instance has the same per-launch counts: same form, lanes and heads per chip) and, on the
    2x4, the relay store ((C - 1) Hr RPB pages; `relay=` shares a larger one)."""

    def __init__(self, mesh, heads_local: int, rows: int, *, dim: int = 512, form: str | None = None,
                 axis: int | None = None, b_lanes=B_LANES, a_lanes=A_LANES, nb: int = 16,
                 slots: int = 4, order: str = "rr", link_base: int = 0, sync=None, relay=None):  # fmt: skip
        import ttnn

        self.mesh = mesh
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        if form is None:
            form = mesh_form((self.R, self.C)) if axis is None else "line"
            assert form is not None, ("no a2a form for the mesh", self.R, self.C)
        self.form, self.axis = form, axis
        self.b_lanes, self.a_lanes = [tuple(v) for v in b_lanes], [tuple(v) for v in a_lanes]
        nl = len(self.b_lanes)
        assert form != "2d" or len(self.a_lanes) == nl, ("A lanes = B lanes", self.a_lanes, self.b_lanes)
        kw = dict(hr=heads_local, rows=rows, dim=dim, form=form, axis=axis, nl=nl, order=order)
        self.plans = {hq: A2APlan(self.R, self.C, heads_to_rows=hq, **kw) for hq in (True, False)}
        p0 = self.plans[True]
        self.nb, self.slots, self.link_base = int(nb), int(slots), int(link_base)
        assert p0.rpb % self.nb == 0 and self.nb % 2 == 0, ("pages per run: whole NB batches", p0.rpb, self.nb)
        self.lane_cores = {("B", L): self.b_lanes[L] for L in range(nl)}
        if form == "2d":
            self.lane_cores.update({("A", L): self.a_lanes[L] for L in range(nl)})
        assert len(set(self.lane_cores.values())) == len(self.lane_cores), ("lane cores", self.lane_cores)
        self.noc = {}
        for k_, xy in self.lane_cores.items():
            w = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
            self.noc[k_] = (int(w.x), int(w.y))
        assert all(v < 256 for xy in self.noc.values() for v in xy), self.noc
        self.signature = (form, axis, nl, int(heads_local), tuple(sorted(self.lane_cores.items())))
        self.own_sync = sync is None
        self.sync = self.make_sync(mesh, list(self.lane_cores.values())) if sync is None else sync
        sig0 = _SYNC_SIG.setdefault(id(self.sync), self.signature)
        assert sig0 == self.signature, ("a shared a2a sync of another form / lanes / heads", sig0, self.signature)
        self.own_relay = False
        self.relay = None
        if form == "2d":
            if relay is None:
                relay = self.make_relay(mesh, p0.relay_pages)
                self.own_relay = True
            assert self.relay_capacity(relay) >= p0.relay_pages, ("the shared relay store is too small", p0.relay_pages)
            self.relay = relay
        self._srcs = {n_: _ktext(n_) for n_ in ("a2a_send.cpp", "a2a_read.cpp")}
        self.calls = 0

    @staticmethod
    def make_sync(mesh, cores):
        import ttnn

        from models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream import _crs

        words = WORDS * WORD // 4
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs(cores), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        return ttnn.from_torch(torch.zeros(len(cores), words, dtype=torch.int32), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=mc,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip

    @staticmethod
    def make_relay(mesh, pages: int, dim: int = 512):
        import ttnn

        dt = dim // 32
        rows = -(-pages // dt) * 32
        return ttnn.zeros((1, 1, rows, dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    @staticmethod
    def relay_capacity(relay, dim: int = 512) -> int:
        return (int(relay.shape[-2]) // 32) * (int(relay.shape[-1]) // 32)

    def headers(self, r: int, c: int) -> dict:
        """fabric packet headers per lane BRISC of chip (r, c)"""
        lanes = self.plans[True].lanes[(r, c)]
        return {k_: NHDR * (int(lp["up"] > 0) + int(lp["dn"] > 0)) for k_, lp in lanes.items()}

    def program(self, x, y, r: int, c: int, heads_to_rows: bool):
        import ttnn

        from models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream import _crs, fabric_connect

        plan = self.plans[heads_to_rows]
        rel = self.relay if self.relay is not None else x
        addr = dict(sync=self.sync.buffer_address(), x=x.buffer_address(), y=y.buffer_address(),
                    relay=rel.buffer_address())  # fmt: skip
        enc = plan.encode((r, c), self.noc, addr)
        conn_at = len(next(iter(enc.values()))[1])
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ct_send = [self.nb, CB_STAGE, plan.rpb, conn_at] + acc(y) + acc(rel)
        ct_read = [self.nb, CB_STAGE, plan.rpb, max(len(rd) for rd, _ in enc.values())] + acc(x) + acc(rel)
        rt_send, rt_read = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k_, (rd, sd) in enc.items():
            lx, ly = self.lane_cores[k_]
            rt_send[lx][ly] = sd
            rt_read[lx][ly] = rd
        cores = _crs(list(self.lane_cores.values()))
        kd = lambda name, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=self._srcs[name], source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
            compile_time_args=ct, runtime_args=rt, config=cfg, defines=[])  # fmt: skip  # (q and o: one binary pair)
        kernels = [kd("a2a_send.cpp", ct_send, rt_send, ttnn.WriterConfigDescriptor()),
                   kd("a2a_read.cpp", ct_read, rt_read, ttnn.ReaderConfigDescriptor())]  # fmt: skip
        cbs = [ttnn.CBDescriptor(total_size=self.slots * self.nb * TB, core_ranges=cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_STAGE, data_format=ttnn.bfloat16, page_size=TB)])]  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        me = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        for k_, lp in plan.lanes[(r, c)].items():
            core = self.lane_cores[k_]
            step = (lambda d: (r, c + d)) if lp["axis"] == 1 else (lambda d: (r + d, c))  # noqa: E731
            for d, rng in ((1, lp["up"]), (-1, lp["dn"])):  # up first, then down (a2a_send.cpp's order)
                if rng:
                    peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                    fab = fabric_connect(me, peer, self.link_base + k_[1], prog, ttnn.CoreCoord(*core))
                    prog.kernels[0].runtime_args[core[0]][core[1]].extend(fab)
        return prog

    def __call__(self, x, heads_to_rows: bool, y=None):
        import ttnn

        plan = self.plans[heads_to_rows]
        assert tuple(int(v) for v in x.shape) == plan.x_shape(), ("a2a input shape", tuple(x.shape), plan.x_shape())
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
        mc = x.memory_config()
        assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
        if y is None:
            y = ttnn.empty(plan.y_shape(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        assert tuple(int(v) for v in y.shape) == plan.y_shape(), (tuple(y.shape), plan.y_shape())
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self.program(x, y, r, c, heads_to_rows)
        io = [x] + ([self.relay] if self.relay is not None else []) + [self.sync, y]
        ttnn.generic_op(io, mpd)
        self.calls += 1
        return y

    def owned(self) -> list:
        return [t_ for t_, own in ((self.sync, self.own_sync), (self.relay, self.own_relay)) if own and t_ is not None]

    def release(self):
        import ttnn

        for t_ in self.owned():
            _SYNC_SIG.pop(id(t_), None)
            ttnn.deallocate(t_)
