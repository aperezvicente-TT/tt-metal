# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 prefill-megakernel (PMK) test harness helpers, shared by jobs/pmk_layer_unit.py (ONE traced replay vs the
op-graph path + ONE 200-replay timing) and jobs/pmk_layer_mock.py (the 2x4 host-mock plan / L1 census).

THE BUILDER CONTRACT the harness drives (doc/ttnn/16-prefill-production/prefill_megakernel.md §0 / §4b / §4c; the
family agent's tt/pmk_layer.py; stubbed here while that module is absent):

    from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer
    prog = pmk_layer.PmkSparseKdaLayer(pf, li, c, sp, probe=True)   # or PmkSparseDsaLayer / PmkDenseKdaLayer, or
                                                                    # pmk_layer.build_layer(pf, li, c, sp, probe=...)
        pf  : the TPPrefill (its PrefillLayer pieces = the resident weights / scratch / kv), li = pf.layers index,
        c   : the bucket (rows of the replay; v1 = 2,048 at TP8, 1,024 on the TP2 pair), sp = sequence-parallel
    y = prog(x, ci)          # x [1, 1, C_local, 16384] fp32 DRAM (pf.streams_to_device layout) -> y same shape; ONE
                             # persistent program enqueue (captured by the harness; the first call is its compile launch)
    prog.counters["eager"]   # programs the builder launched OUTSIDE its persistent program (expected 0 at replay)
    prog.plan()              # dict: phases [{name, cores, l1_bytes, cbs}], l1_max_cb_end (B), ... (mock census)
    prog.probe_read()        # PROBE CB 31 rows of the last replay -> {"mhz", "cores", "span_cycles", "phases": {name:
                             #   {"span_cycles", "cores", "trisc1_busy_cycles" (sum over cores), "dram_bytes",
                             #    "dram_cycles", "fabric_bytes", "fabric_cycles"}}}; None without probes
    prog.taps()              # {hand-off name: host torch tensor} of the last replay, in the op-graph tap layout (§4b
                             #  H1.xhat / H6.y / x1 / H10.y_f, local rows [1, 1, C_local, cols]); {} = none
The side effects (KDA S / C scratch of pf.layers[li], the sparse_dsa latent pages of kv) are the SAME tensors the
op-graph path writes: the harness zeroes them before each path's replay and reads them after."""

from __future__ import annotations

import importlib
import os
import time

import torch

import ttnn

BF16, FP32 = ttnn.bfloat16, ttnn.float32
MOCK = os.environ.get("GLM53_MOCK") == "1"
# ideal-rate FLOP per prompt row over ALL chips, per layer kind (prefill_megakernel.md §3: sparse_kda 199 GFLOP per chip
# at 2,048 rows TP8 -> x 8 / 2,048; dense_kda = its KDA half + the 77 GFLOP MLP slice; sparse_dsa approximated by the
# sparse_kda figure (same FFN half, attention of comparable size) -- bounds only, refined per kind by its README)
FLOP_PER_ROW = {"sparse_kda": 199e9 * 8 / 2048, "dense_kda": (77.2e9 + 77e9) * 8 / 2048, "sparse_dsa": 199e9 * 8 / 2048}
PEAK_TFLOPS = 297.0  # HiFi2 grid peak per chip (§3)
AICLK_MHZ = 1350.0


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(*a):
    print(f"PMK {now()}", *a, flush=True)


# ------------------------------------------------------------------------------------------------ the builder / stub
class StubPmkLayer:
    """The contract with the op-graph path behind it (tt/pmk_layer.py absent): proves the harness end to end; rows it
    produces carry impl="stub" and never count as a PMK pass."""

    impl = "stub"

    def __init__(self, pf, li: int, c: int, sp: bool, probe: bool = False):
        self.pf, self.li, self.c, self.sp = pf, int(li), int(c), bool(sp)
        self.counters = dict(eager=0, calls=0)

    def __call__(self, x, ci, **_):
        self.counters["calls"] += 1
        return op_layer(self.pf, self.li, x, ci, self.sp)

    def plan(self) -> dict:
        return dict(stub=True, phases=[], l1_max_cb_end=None, note="op-graph path: no persistent program, no ledger")

    def probe_read(self):
        return None

    def taps(self) -> dict:
        return {}


def op_layer(pf, li: int, x, ci, sp: bool):
    """TPPrefill.layer with the PMK hook of this layer lifted for the call (the op-graph path, keep_input)"""
    L = pf.layers[li].L
    hooks = getattr(pf, "pmk_layers", None)
    saved = hooks.pop(L, None) if hooks is not None else None
    try:
        return pf.layer(li, x, ci, sp, keep_input=True)
    finally:
        if saved is not None:
            hooks[L] = saved


def hook(prog):
    """the TPPrefill.pmk_layers entry for a builder (integration ff3ec0e58b4: callable(x, *, tpp, pl, ci, sp) -> the
    layer output): a builder whose __call__ takes the hook's keywords gets them, else (x, ci)"""
    import inspect

    try:
        params = inspect.signature(prog.__call__).parameters
    except (TypeError, ValueError):
        params = {}
    if "tpp" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return lambda x, *, tpp, pl, ci, sp: prog(x, ci, tpp=tpp, pl=pl, sp=sp)
    return lambda x, *, tpp, pl, ci, sp: prog(x, ci)


def run_pmk(pf, li: int, x, ci, sp: bool, prog):
    """the PMK path THROUGH the integration hook: pf.pmk_layers[L] = hook(prog), then TPPrefill.layer (what the bucket
    trace will run); a TPPrefill without the hook (base tree) calls the builder directly"""
    hooks = getattr(pf, "pmk_layers", None)
    if hooks is None:
        return prog(x, ci)
    hooks[pf.layers[li].L] = hook(prog)
    return pf.layer(li, x, ci, sp, keep_input=True)


CLASS_BY_KIND = {"sparse_kda": "PmkSparseKdaLayer", "sparse_dsa": "PmkSparseDsaLayer", "dense_kda": "PmkDenseKdaLayer"}


def load_builder(kind: str):
    """(factory(pf, li, c, sp, probe=) , impl name): tt/pmk_layer.py's class for `kind` (CLASS_BY_KIND), else its
    build_layer(), else the stub. Import-guarded: a missing module -> stub; a module that imports but lacks the kind
    -> stub too (logged), so a half-built family never blocks the harness."""
    try:
        mod = importlib.import_module("models.autoports.zai_org_glm_5_3_flash.tt.pmk_layer")
        importlib.reload(mod)
    except ImportError as e:
        log(f"tt/pmk_layer.py not importable ({e!r}): STUB (op-graph path behind the contract)")
        return StubPmkLayer, "stub"
    cls = getattr(mod, CLASS_BY_KIND.get(kind, ""), None)
    if cls is not None:
        return cls, f"pmk:{cls.__name__}"
    fn = getattr(mod, "build_layer", None)
    if fn is not None:
        return (lambda pf, li, c, sp, probe=False: fn(pf, li, c, sp, probe=probe)), "pmk:build_layer"
    log(f"tt/pmk_layer.py has no {CLASS_BY_KIND.get(kind)} / build_layer: STUB")
    return StubPmkLayer, "stub"


# ------------------------------------------------------------------------------------------------ eager fallback guard
class EagerGuard:
    """Snapshot of every counter an eager launch would move: the TPPrefill's eager counter, the backend's decode
    counters, the builder's own eager counter, and the mesh program cache (a trace replay adds NO program-cache entry;
    any program launched eagerly between snapshot and check does). check() raises -> the job fails loudly."""

    def __init__(self, mesh, pf, backend, prog):
        self.mesh, self.pf, self.backend, self.prog = mesh, pf, backend, prog
        self.before = None

    def _state(self) -> dict:
        st = dict(pf_eager=int(self.pf.counters.get("eager", 0)),
                  backend_eager=int(self.backend.counters.get("decode_eager", 0)),
                  prog_eager=int(getattr(self.prog, "counters", {}).get("eager", 0)))  # fmt: skip
        try:
            st["program_cache"] = int(self.mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001 -- not every mesh binding exposes it
            st["program_cache"] = -1
        return st

    def snapshot(self):
        self.before = self._state()
        return self.before

    def check(self, label: str) -> dict:
        after = self._state()
        delta = {k: after[k] - self.before[k] for k in after if after[k] != self.before[k]}
        if delta and MOCK:  # the mock has no traces: its "replay" is the eager call by construction (logged, not fatal)
            log(f"eager guard (mock, informational) during {label}: {delta}")
            return after
        if delta:
            raise RuntimeError(
                f"EAGER FALLBACK during {label}: counters moved {delta} (before {self.before}, after {after})"
            )
        return after


# ------------------------------------------------------------------------------------------------ the op-graph reference
TAPS = ("H1.xhat", "H6.y", "x1", "H10.y_f")


def ref_layer(pf, li: int, x, ci, sp: bool, taps: dict | None):
    """The SAME op-graph layer as TPPrefill.layer (the fused-mHC path, GLM53_PF_MHC_POST=fused: _mhc_pre, the attention
    sublayer, _post, ffn, reduce -- identical ops in the identical order) with the §4b hand-off taps copied into
    persistent tensors (allocated on the compile launch, so they exist before any capture): H1.xhat = the normalised
    attention input (bf16, local rows), H6.y = the reduced attention output, x1 = the residual after the attention
    sublayer, H10.y_f = the reduced FFN output. taps None, or another mHC mode: pf.layer itself (no taps)."""
    pl = pf.layers[li]
    if taps is None or pf.mhc_mode != "comp" or pf.mhc_post != "fused":
        return op_layer(pf, li, x, ci, sp)

    def tap(name, t):
        if name not in taps:
            taps[name] = ttnn.zeros_like(t)
        ttnn.copy(t, taps[name])

    h, post, comb = pf._mhc_pre(pl, "attn", x)
    hn = pl.rms(h, "attn", BF16)  # the ops _attn_sublayer runs, opened up for the taps
    if hn is not h:
        ttnn.deallocate(h)
    tap("H1.xhat", hn)
    hb = pf.gather(hn, sp)
    if hb is not hn:
        ttnn.deallocate(hn)
    a = pf.reduce(pf.attention(pl, hb, ci, li, hn if hb is hn else None, None), sp)
    ttnn.deallocate(hb)
    tap("H6.y", a)
    x1 = pf._post(a, x, post, comb)
    for t_ in (a, post, comb):
        ttnn.deallocate(t_)
    tap("x1", x1)
    h2, post2, comb2 = pf._mhc_pre(pl, "ffn", x1)
    m = pf.reduce(pf.ffn(pl, h2, sp), sp)
    ttnn.deallocate(h2)
    tap("H10.y_f", m)
    out = pf._post(m, x1, post2, comb2)
    for t_ in (m, x1, post2, comb2):
        ttnn.deallocate(t_)
    return out


# ------------------------------------------------------------------------------------------------ host <-> mesh
def per_chip(t) -> torch.Tensor:
    """a mesh tensor -> [chips, ...] host fp32 (layout-agnostic: every chip's shard stacked)"""
    parts = ttnn.get_device_tensors(t) if t.device().get_num_devices() > 1 else [t]
    return torch.stack([ttnn.to_torch(p).float() for p in parts])


def rows_host(t, sp: bool) -> torch.Tensor:
    """[1, 1, rows(/N), cols] mesh tensor -> [rows, cols] host (sp: the chips' row shards concatenated)"""
    parts = ttnn.get_device_tensors(t) if t.device().get_num_devices() > 1 else [t]
    hs = [ttnn.to_torch(p).float() for p in parts]
    y = torch.cat(hs, dim=2) if sp else hs[0]
    return y.reshape(y.shape[-2], y.shape[-1])


def side_effects(pf, li: int, kv) -> dict:
    """the layer's state / cache side effects as host tensors: KDA kinds -> S, C of the layer's scratch; sparse_dsa ->
    the latent pages (whole paged cache, [chips, blocks, 1, 64, 512])"""
    from models.autoports.zai_org_glm_5_3_flash.tt.config import SPARSE_DSA

    pl = pf.layers[li]
    if pl.kind == SPARSE_DSA:
        return {"latent": per_chip(kv.extra["latent"][li])}
    return {"S": per_chip(pl.scratch.S), "C": per_chip(pl.scratch.C)}


def zero_state(pf, li: int, kv):
    """both paths start from the same state: zeroed KDA scratch (S, C) or zeroed latent pages"""
    from models.autoports.zai_org_glm_5_3_flash.tt.config import SPARSE_DSA

    pl = pf.layers[li]
    ts = [kv.extra["latent"][li]] if pl.kind == SPARSE_DSA else [pl.scratch.S, pl.scratch.C]
    for t_ in ts:
        ttnn.copy(ttnn.zeros_like(t_), t_)


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a (device under test) vs b (reference): bitwise, max |diff|, PCC, relative L2, norm ratio (PCC is scale-blind,
    §6.5: maxabs and norm ratios travel with it)"""
    from models.autoports.zai_org_glm_5_3_flash.tests.s14b.s14common import pcc

    a_, b_ = a.reshape(-1).double(), b.reshape(-1).double()
    bit = bool(torch.equal(a_, b_))
    rn = float(b_.norm().clamp_min(1e-30))
    return dict(bitwise=bit, max_abs=0.0 if bit else round(float((a_ - b_).abs().max()), 6),
                pcc=1.0 if bit else round(float(pcc(a, b)), 6), rel_l2=0.0 if bit else round(float((a_ - b_).norm() / rn), 6),
                norm_ratio=1.0 if bit else round(float(a_.norm() / rn), 6))  # fmt: skip


# ------------------------------------------------------------------------------------------------ MATH utilization
def decode_probe(probe: dict | None, *, ncores_chip: int = 110) -> dict | None:
    """prog.probe_read() -> MATH utilization and per-phase overlap (prefill_megakernel.md §3 / §9): math_util =
    sum over phases of TRISC1 busy cycles (summed over the cores of the phase) / (grid cores x the layer span);
    per phase: fpu (busy / (phase cores x phase span)), dram_busy (dram cycles / span), fabric_busy, dram GB/s."""
    if not probe:
        return None
    mhz = float(probe.get("mhz", AICLK_MHZ))
    cores = int(probe.get("cores", ncores_chip))
    span = int(probe.get("span_cycles", 0))
    out = dict(span_us=round(span / mhz, 1), cores=cores, phases={})
    busy_total = 0
    for name, ph in (probe.get("phases") or {}).items():
        ps, pc = int(ph.get("span_cycles", 0)), int(ph.get("cores", cores))
        busy = int(ph.get("trisc1_busy_cycles", 0))
        busy_total += busy
        dram_c, fab_c = int(ph.get("dram_cycles", 0)), int(ph.get("fabric_cycles", 0))
        row = dict(span_us=round(ps / mhz, 1), fpu=round(busy / (pc * ps), 4) if ps and pc else None,
                   dram_busy=round(dram_c / ps, 4) if ps else None, fabric_busy=round(fab_c / ps, 4) if ps else None)  # fmt: skip
        if dram_c:
            row["dram_gbps"] = round(int(ph.get("dram_bytes", 0)) / (dram_c / mhz) / 1e3, 1)
        if fab_c:
            row["fabric_gbps"] = round(int(ph.get("fabric_bytes", 0)) / (fab_c / mhz) / 1e3, 1)
        out["phases"][name] = row
    out["math_util"] = round(busy_total / (cores * span), 4) if span else None
    return out


def flop_bound(kind: str, rows: int, ndev: int, us: float) -> dict:
    """the arithmetic MATH-utilization bound of a replay time: per-chip FLOP (all-chip FLOP / chips) over us x the
    HiFi2 peak -- what the probes must not exceed, and a stand-in where no probe exists (row field math_util_bound)"""
    fpr = FLOP_PER_ROW.get(kind)
    if not fpr or not us:
        return {}
    flop_chip = fpr * rows / ndev
    return dict(
        gflop_per_chip=round(flop_chip / 1e9, 1),
        math_util_bound=round(flop_chip / (us * 1e-6) / (PEAK_TFLOPS * 1e12), 4),
    )
