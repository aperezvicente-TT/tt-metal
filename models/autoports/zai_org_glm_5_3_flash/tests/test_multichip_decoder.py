# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 4: the multichip decoder stack (tt/multichip_decoder.py) on the host-8chip mesh against the single-chip
optimized decoder (Stage 3) driven through its own public API with identical weights, inputs, page tables and positions.

One long-lived device process per invocation (DECISIONS 2026-10-17 ~21:50Z: <= 1 open/close per chip per 30 min).

Mesh / plan:
  --mgd 2x4 --plan partition   all 8 chips, the 45-layer whole-layer line plan of partition.json (Stage 4 acceptance)
  --mgd 2x2 --plan even        chips 4-7 (Stage 4's cards while Stage 3 owns cards 0, 1), a contiguous layer subset
                               (default L0-L7: every layer kind, a chip mixing dense_kda and sparse_dsa) split over the
                               line segment 5-6-7-4 (cable 5-6, on-board 6-7, cable 7-4)

Reference (the single-chip TTNN optimized baseline, per layer, chained on the host):
  prefill  per layer and model chunk: `OptimizedDecoder.run_prefill_trace(t, cache, user_slot, chunk_start, n_valid,
           chunk=<host chunk of the previous layer's valid rows, zero padded>)` + host readback = Stage 3's traced
           single-chip prefill (prefill_forward_traced) at the model-level chunk plan; runs BEFORE the pipeline on
           the same slot (the pipeline then rewrites the slot from chunk_start 0).
  decode   per layer: `decode_forward(x, cache=<reference cache>, page_table=<identity table>, cur_pos=...)` eagerly
           on the layer's chip, x = host copy of the previous layer's output; reference caches are separate (their
           own state chain), seeded with a copy of the pipeline state after prefill (MLA pages gathered through both
           page tables: the pipeline runs a random block permutation at B > 1, the reference the identity).
Checks: every layer output bitwise and PCC (gate >= 0.995 per layer kind, plan: drop <= 0.001), PCC vs the fp32 GPU
goldens (golden prompt rows; cumulative through the stack, info), non-aligned prompt lengths (tail 32 / 64 buckets,
multi-chunk), paged KV with a permuted page table, in-trace plus_one positions on every chip, inactive slots (state
bit-identical), replay determinism, no program compiled after the captures; rows -> $GLM53_E4/pcc_real_weights.jsonl,
tests.jsonl, footprint.jsonl. --mode perf: traced decode latency (pipeline vs per-chip compute-only traces = single-chip
baseline), micro-batch-pipelined throughput, traced prefill latency -> perf.jsonl.

  scripts/dev.sh python models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_decoder.py --mgd 2x2 --plan even
(TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 for the acceptance run.)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import (
    HC_WIDTH,
    MultichipDecoder,
    StagePlan,
    chip_coords,
    close_pipeline_mesh,
    dram_allocated,
    dram_total,
    even_plan,
    layer_bucket,
    open_pipeline_mesh,
    parse_units,
    plan_from_partition,
    prefill_chunk_plan,
    validate_plan,
)

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
E = os.environ.get("GLM53_E4", os.path.join(WS, "doc/ttnn/04-multichip-decoder"))
GOLD_LAYERS = os.path.join(WS, "goldens_gpu_fp32_layers")
MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mesh")
MGDS = {
    "2x4": ("glm8_2x4_mesh_graph_descriptor.textproto", (2, 4)),
    "2x2": ("glm4_2x2_mesh_graph_descriptor.textproto", (2, 2)),
    "1x2": (None, (1, 2)),  # one card (2 P300 chips = ClusterType P300, no descriptor needed)
    "1x1": (None, (1, 1)),  # mock cluster (only a 1x1 mesh opens there): every stage on chip 0, host hand-off
}
TRACKING = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
PCC_BAR = 0.995
DENSE = {0, 1, 2}
MLA = set(range(3, 45, 4))
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def kind_of(l):
    return "dense_kda" if l in DENSE else ("sparse_dsa" if l in MLA else "sparse_kda")


MAX_SEQ = {"kda": 2048 + 64, "mla": DSA_DENSE_EXACT_MAX_KV}  # --max-seq overrides both (Tracy runs: small buckets)


def max_seq(l):
    return MAX_SEQ["kda"] if l not in MLA else MAX_SEQ["mla"]


try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


PROFILING = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))


def flush(m):
    """Device profiler: read every chip's buffers (between replays; the profiler holds a bounded op count)."""
    if PROFILING:
        for st in m.stages:
            ttnn.ReadDeviceProfiler(st.mesh)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record(row, name="tests.jsonl"):
    row = {
        "time_utc": now(),
        "run_id": RUN_ID,
        **row,
        "trace_alloc_tracking": TRACKING,
        "watcher": os.environ.get("TT_METAL_WATCHER", ""),
        "visible": os.environ.get("TT_VISIBLE_DEVICES"),
        "lock": os.environ.get("TT_LOCK_GROUP"),
        "code": os.environ.get("GLM53_CODE_SHA"),
    }
    out_dir = os.environ.get("GLM53_OUT_DIR", E)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row)[:400], flush=True)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def prompt_rows(g0, n):
    """n rows of real layer-0 input activations: the 2048-row fp32 golden prompt, continued cyclically past 2048 (the
    golden holds 2048 rows; the multi-chunk + tail cases need n > 2048)."""
    if n <= g0.shape[0]:
        return g0[:n]
    reps = -(-n // g0.shape[0])
    return torch.cat([g0] * reps)[:n]


def golden(layer, prompt=2048):
    sys.path.insert(0, os.path.join(WS, "goldens"))
    import compare

    return compare.load_layer_io(layer, prompt, goldens_dir=GOLD_LAYERS)


class Prefetcher:
    """Reads the weight files of the layer being loaded and the next `ahead` layers into the page cache with a thread
    pool while the current layer uploads (DECISIONS 2026-10-18 ~23:00Z / doc/research/weight_loading_speed.md: the load
    is bound by serial NFS page faults of ttnn.load_tensor's mmap; the share serves 3.5-4 GB/s to 8-32 readers). Plain
    reads of the same files the loader opens; no change to the loaded bytes or the loader calls. The fused_experts
    permuted gate|up files replace the tile cache's gate|up files when GLM53_FUSED_EXPERTS_CACHE holds them."""

    def __init__(self, threads: int = 24, ahead: int = 2):
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor

        perm = os.environ.get("GLM53_FUSED_EXPERTS_CACHE", "")
        self.files = defaultdict(list)
        for k, e in json.load(open(MANIFEST))["tensors"].items():
            if not k.startswith("layers."):
                continue
            l = int(k.split(".")[1])
            if (
                perm
                and k.endswith(".gate_up_proj")
                and ".experts." in k
                and os.path.isdir(os.path.join(perm, f"L{l:02d}"))
            ):
                continue  # the loader reads the permuted copy instead
            self.files[l].append(os.path.join(CACHE_DIR, e["path"]))
        if perm and os.path.isdir(perm):
            for d in os.listdir(perm):
                if d.startswith("L") and d[1:].isdigit():
                    self.files[int(d[1:])] += [os.path.join(perm, d, f) for f in os.listdir(os.path.join(perm, d))]
        self.pool, self.ahead, self.done = ThreadPoolExecutor(threads), ahead, set()

    @staticmethod
    def _read(path):
        buf = bytearray(8 << 20)
        try:
            with open(path, "rb", buffering=0) as f:
                while f.readinto(buf):
                    pass
        except OSError:
            pass

    def touch(self, layer: int):
        for l in range(layer, layer + self.ahead + 1):
            if l in self.files and l not in self.done:
                self.done.add(l)
                for f in self.files[l]:
                    self.pool.submit(self._read, f)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class Loader:
    """One per-layer decoder per (layer, chip); records load time and the device DRAM each load takes. `half` is
    accepted but the layer loads whole (MultichipDecoder drops the unused half right after; split units load first on
    their chip, so the transient never exceeds the chip's final footprint — 4a review C-7)."""

    def __init__(self, cfg, impl, prefetch: bool = True):
        self.cfg, self.impl, self.load_s, self.dram = cfg, impl, {}, {}
        self.prefetch = Prefetcher() if prefetch else None

    def __call__(self, layer, mesh, half=None):
        if self.prefetch is not None:
            self.prefetch.touch(layer)
        t0 = time.perf_counter()
        b0 = dram_allocated(mesh)
        if self.impl == "optimized":
            from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import load_optimized_decoder

            dec = load_optimized_decoder(self.cfg, layer, mesh)
        else:
            from models.autoports.zai_org_glm_5_3_flash.tests.common_fused import load_fused_decoder

            dec = load_fused_decoder(self.cfg, layer, mesh)
        ttnn.synchronize_device(mesh)
        self.load_s[layer] = round(time.perf_counter() - t0, 2)
        self.dram[layer] = dram_allocated(mesh) - b0
        print(f"loaded L{layer} in {self.load_s[layer]} s ({self.dram[layer] / 2**30:.2f} GiB)", flush=True)
        return dec


# ================================================================================================== reference stack
class Reference:
    """Single-chip optimized decoder per layer through its public API (see module docstring). Whole layers use the
    pipeline's own instance (its chip); a layer the plan splits across two chips uses an extra WHOLE instance loaded on a
    chip with free DRAM (`extras`: layer -> decoder), with its own cache and its own Stage 3 prefill traces."""

    def __init__(self, model: MultichipDecoder, batch: int, extras: dict | None = None):
        self.m = model
        self.batch = batch
        self.extras = extras or {}
        self.caches, self.pt, self.cur_pos, self.traces = {}, {}, {}, {}
        self.dev_coord = {}
        mla_first = {}
        for l in model.layers:
            dec = self.dec(l)
            coord = self._coord(dec)
            self.dev_coord[l] = coord
            self.caches[l] = dec.allocate_cache(max_batch=batch, max_seq_len=max_seq(l))
            if not dec.is_kda:
                if coord not in mla_first:
                    mla_first[coord] = l
                    self.pt[coord] = dec.make_page_table(self.caches[l])  # identity
                else:
                    self.caches[l].page_table_host = self.caches[mla_first[coord]].page_table_host
            if coord not in self.cur_pos:
                self.cur_pos[coord] = ttnn.from_torch(
                    torch.full((batch,), -1, dtype=torch.int32), dtype=ttnn.int32, device=dec.device
                )
        self._mesh_of = {self._coord(self.dec(l)): self.dec(l).device for l in model.layers}

    def dec(self, l):
        if l in self.extras:
            return self.extras[l]
        st = self.m.stage_of(l)
        return st.decoders[l]

    def _coord(self, dec):
        for c, sub in self.m.submeshes.items():
            if sub is dec.device:
                return c
        raise KeyError("decoder device is not a pipeline submesh")

    def prepare_extras(self):
        """Before any capture: Stage 3 prefill-trace buffers of the extra whole instances (compiles their bodies)."""
        for l, dec in self.extras.items():
            self.traces[l] = dec.prepare_prefill_traces(self.caches[l], host_io=False)

    def capture_extras(self):
        for l, dec in self.extras.items():
            dec.capture_prefill_traces(self.caches[l], self.traces[l])

    # ---------------------------------------------------------------------------------------------- prefill
    def prefill(self, rows: torch.Tensor, slot: int, start_pos: int = 0, policy: str = "plan") -> dict:
        """Per layer, per model chunk: run_prefill_trace with a host chunk = previous layer's valid rows + zeros (whole
        layers: the PIPELINE's traces and cache, slot `slot`; extras: their own); returns {"L<l>": [1, n, 16384]}."""
        x = rows.reshape(-1, HC_WIDTH).float()
        n_all = x.shape[0]
        plan = prefill_chunk_plan(n_all, self.m.buckets(), policy)  # the pipeline's chunk plan for this prompt
        outs = {}
        for l in self.m.layers:
            dec = self.dec(l)
            if l in self.extras:
                cache, traces = self.caches[l], self.traces[l]
            else:
                g = self.m.stage_of(l).groups[0]
                cache, traces = g.caches[l], g.prefill_traces[(l, "full")]
            ys = []
            for off, n, phys in plan:
                t = traces[layer_bucket(dec, phys, n)]
                host = dec.prefill_host_chunk(x[off : off + n], t)
                y = dec.run_prefill_trace(t, cache, user_slot=slot, chunk_start=start_pos + off, n_valid=n, chunk=host)
                ys.append(ttnn.to_torch(y).float().reshape(-1, HC_WIDTH)[:n])
            x = torch.cat(ys)
            outs[f"L{l}"] = x.reshape(1, n_all, HC_WIDTH)
        return outs

    def compare_state_extras(self, slot: int) -> list:
        """Post-prefill state of every split layer: the pipeline's cache (chip of the attention half) vs the extra
        WHOLE instance's own cache, filled by its own prefill traces (an independent control of the pipeline's
        state, unlike the whole-layer references that share the pipeline's cache). KDA: S and C rows of `slot`; MLA:
        the slot's latent pages through each side's page table."""
        out = []
        for l, dec in self.extras.items():
            pc, rc = self.m.stage_of(l).groups[0].caches[l], self.caches[l]
            if dec.is_kda:
                pairs = [(ttnn.to_torch(pc.kda.S)[slot], ttnn.to_torch(rc.kda.S)[slot])]
                cp, cr = ttnn.to_torch(pc.kda.C), ttnn.to_torch(rc.kda.C)
                pairs.append((cp[:, :, slot], cr[:, :, slot]))
            else:
                bps = pc.blocks_per_seq
                ps = pc.page_table_host.reshape(-1, bps)[slot].long()
                rs = rc.page_table_host.reshape(-1, bps)[slot].long()
                pairs = [(ttnn.to_torch(pc.latent)[ps], ttnn.to_torch(rc.latent)[rs])]
            exact = all(bool(torch.equal(a, b)) for a, b in pairs)
            worst = min(pcc(a.float(), b.float()) for a, b in pairs)
            out.append(dict(layer=l, layer_kind=kind_of(l), bitexact=exact, pcc=round(worst, 8)))
        return out

    # ---------------------------------------------------------------------------------------------- decode
    def seed_from_pipeline(self):
        """Copy the pipeline's per-layer state (the chip running the layer's attention) into the reference caches
        (KDA S / C on device or through the host across chips; MLA pages gathered on the host from the pipeline's
        physical blocks to the reference's identity blocks)."""
        for l in self.m.layers:
            st = self.m.stage_of(l)
            pc, rc = st.groups[0].caches[l], self.caches[l]
            if self.dec(l).is_kda:
                for src, dst in ((pc.kda.S, rc.kda.S), (pc.kda.C, rc.kda.C)):
                    if self.dev_coord[l] == tuple(st.plan.coord):  # same chip: device copy
                        ttnn.copy(src, dst)
                    else:
                        h = ttnn.from_torch(ttnn.to_torch(src), dtype=src.dtype, layout=src.layout)
                        ttnn.copy_host_to_device_tensor(h, dst)
            else:
                lat = ttnn.to_torch(pc.latent)
                src = pc.page_table_host.reshape(-1).long()
                dst = rc.page_table_host.reshape(-1).long()
                out = torch.zeros_like(lat)
                out[dst] = lat[src]
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(out, dtype=pc.latent.dtype, layout=ttnn.TILE_LAYOUT), rc.latent
                )
        for mesh in self._mesh_of.values():
            ttnn.synchronize_device(mesh)

    def warm(self):
        """Compile the reference's eager decode programs (its own input tensors / page tables) before any capture, so
        nothing compiles while traces are live. State is overwritten later by seed_from_pipeline."""
        self.set_positions([0] * self.batch)
        self.decode(torch.zeros(self.batch, HC_WIDTH))
        self.set_positions([-1] * self.batch)
        for mesh in self._mesh_of.values():
            ttnn.synchronize_device(mesh)

    def free(self):
        for l, traces in self.traces.items():
            self.extras[l].release_prefill_traces(traces)
        self.traces = {}
        for c in self.caches.values():
            for t in (c.latent, c.chunk_table, c.user_table):
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
            if c.kda is not None:
                for t in vars(c.kda).values():
                    if isinstance(t, ttnn.Tensor) and t.is_allocated():
                        ttnn.deallocate(t)
        for t in list(self.pt.values()) + list(self.cur_pos.values()):
            ttnn.deallocate(t)
        self.caches, self.pt, self.cur_pos = {}, {}, {}

    def set_positions(self, positions):
        host = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32)
        for t in self.cur_pos.values():
            ttnn.copy_host_to_device_tensor(host, t)

    def decode(self, streams: torch.Tensor) -> dict:
        x = streams.reshape(1, 1, self.batch, HC_WIDTH).float()
        outs = {}
        for l in self.m.layers:
            dec, coord = self.dec(l), self.dev_coord[l]
            xd = ttnn.from_torch(
                x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dec.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            yd = dec.decode_forward(
                xd, cache=self.caches[l], page_table=self.pt.get(coord), cur_pos=self.cur_pos[coord]
            )
            x = ttnn.to_torch(yd).float().reshape(1, 1, self.batch, HC_WIDTH)
            ttnn.deallocate(xd)
            ttnn.deallocate(yd)
            outs[f"L{l}"] = x.reshape(self.batch, HC_WIDTH)
        return outs


# ===================================================================================================== comparisons
def negative_control(pipe, ref, failures):
    """4a review C-12: the comparator must reject wrong data. (a) one element of a bit-exact layer output perturbed by
    1e-3 must fail bit-exactness; (b) a layer compared with its predecessor's reference (a dropped / duplicated hop)
    must fail bit-exactness and fall below PCC 1."""
    names = sorted(ref, key=lambda k: int(k[1:]))
    mid = names[len(names) // 2]
    prev = names[names.index(mid) - 1]
    p = pipe[mid].reshape(-1, HC_WIDTH).clone()
    p.view(-1)[p.numel() // 3] += 1e-3
    pert_exact = bool(torch.equal(p, ref[mid].reshape(-1, HC_WIDTH)))
    shift_exact = bool(torch.equal(pipe[mid].reshape(-1, HC_WIDTH), ref[prev].reshape(-1, HC_WIDTH)))
    shift_pcc = pcc(pipe[mid].reshape(-1, HC_WIDTH), ref[prev].reshape(-1, HC_WIDTH))
    ok = (not pert_exact) and (not shift_exact) and shift_pcc < 1.0
    record(
        dict(
            test="multichip_stack/negative_control",
            layer=mid,
            predecessor=prev,
            perturbed_bitexact=pert_exact,
            shifted_bitexact=shift_exact,
            shifted_pcc=round(shift_pcc, 6),
            status="pass" if ok else "fail",
        )
    )
    if not ok:
        failures.append("negative_control")


def compare_layers(phase, pipe, ref, *, mode, rows=None, gold=None, extra=None, failures=None):
    """pipe / ref {"L<l>": tensors}; rows = active row indices (decode) or None (all)."""
    worst = {}
    for name in ref:
        l = int(name[1:])
        p, r = pipe[name].reshape(-1, HC_WIDTH), ref[name].reshape(-1, HC_WIDTH)
        if rows is not None:
            p, r = p[rows], r[rows]
        v = pcc(p, r)
        exact = bool(torch.equal(p, r))
        finite = bool(torch.isfinite(p).all())
        g = None
        if gold is not None and name in gold:
            gr = gold[name].reshape(-1, HC_WIDTH)
            g = round(pcc(p[: gr.shape[0]], gr), 6)  # rows with a golden (first 2048 positions)
        status = "pass" if v >= PCC_BAR and finite else "fail"
        row = dict(
            test=f"multichip_stack/{phase}/{name}",
            layer=l,
            layer_kind=kind_of(l),
            mode=mode,
            phase=phase,
            rows=int(p.shape[0]),
            pcc=round(v, 8),
            bitexact=exact,
            max_abs_diff=float((p - r).abs().max()),
            finite=finite,
            pcc_vs_golden=g,
            weights="real",
            reference="single-chip OptimizedDecoder public API",
            status=status,
            **(extra or {}),
        )
        record(row, "pcc_real_weights.jsonl")
        if status != "pass" and failures is not None:
            failures.append(row["test"])
        worst[kind_of(l)] = min(worst.get(kind_of(l), 1.0), v)
    return worst


class StateSnap:
    """Persistent copies of every stage's KDA state (S, C) for one group, allocated BEFORE any capture (no allocation
    while traces are live: TT_METAL_TRACE_ALLOC_TRACKING), filled / restored with device copies."""

    def __init__(self, m: MultichipDecoder, group=0):
        self.m, self.group, self.bufs = m, group, []
        for st in m.stages:
            g = st.groups[group]
            for l in st.state_layers:
                if st.decoders[l].is_kda:
                    k = g.caches[l].kda
                    self.bufs.append((k, ttnn.clone(k.S), ttnn.clone(k.C)))

    def take(self):
        for k, s_, c_ in self.bufs:
            ttnn.copy(k.S, s_)
            ttnn.copy(k.C, c_)

    def put(self):
        for k, s_, c_ in self.bufs:
            ttnn.copy(s_, k.S)
            ttnn.copy(c_, k.C)

    def free(self):
        for _, s_, c_ in self.bufs:
            ttnn.deallocate(s_)
            ttnn.deallocate(c_)
        self.bufs = []


def free_decoder(dec):
    """Release every device tensor of a decoder instance (the extra whole reference instances, before perf)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import _device_tensors

    for t in _device_tensors(dec.layer):
        if t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated():
            t.deallocate(True)


def probe_rows(m: MultichipDecoder, batch, group=0):
    out = {}
    for st in m.stages:
        for u, t in zip(st.plan.units, st.groups[group].probes):
            out[f"L{u.layer}" + (".attn" if u.part == "attn" else "")] = (
                ttnn.to_torch(t).float().reshape(-1, HC_WIDTH)[:batch]
            )
    return out


def program_entries(m: MultichipDecoder):
    return {str(st.plan.chip): st.mesh.num_program_cache_entries() for st in m.stages}


def trace_bytes(mesh):
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)
    except Exception:  # noqa: BLE001
        return None


def footprint(m: MultichipDecoder, tag, extra=None):
    rows = {}
    for st in m.stages:
        rows[str(st.plan.chip)] = dict(
            allocated=dram_allocated(st.mesh), total=dram_total(st.mesh), trace=trace_bytes(st.mesh)
        )
    record(dict(test=f"footprint/{tag}", per_chip=rows, status="pass", **(extra or {})), "footprint.jsonl")
    return rows


# ====================================================================================================== test phases
def build(a):
    cfg = hf_config()
    mgd, shape = MGDS[a.mgd]
    t0 = time.perf_counter()
    subs = None
    if a.mgd == "1x1":
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=a.trace_region)
        subs = {(0, 0): mesh}
    else:
        mesh = open_pipeline_mesh(
            shape, trace_region_size=a.trace_region, mgd=mgd and os.path.join(MESH_DIR, mgd), fabric=bool(a.fabric)
        )
    open_s = time.perf_counter() - t0
    coords = chip_coords(mesh)
    if a.plan == "partition":
        stages = plan_from_partition(a.partition, coords, key=a.partition_key)
    elif a.plan == "units":  # "chip:U,U;chip:U,..." e.g. "0:L0,L1.attn;0:L1.ffn,L2"
        # stages separated by ";" or "/" ("/" survives `python -m tracy`, which re-runs the command through a shell)
        segs = a.units.replace("/", ";").split(";")
        spec = [(int(seg.split(":")[0]), seg.split(":")[1].split(",")) for seg in segs]
        stages = [StagePlan(coord=coords[c], units=parse_units(us), chip=c) for c, us in spec]
        validate_plan(stages)
    elif a.plan == "wrap4":
        # the ring wrap-around in small on chips 5-6-7-4-(5): two segments on chip 5, a split in every layer kind
        # (L1 dense_kda, L3 / L7 sparse_dsa, L5 sparse_kda), cable hops 5-6 / 7-4, on-board 6-7 / closing 4-5
        spec = [
            (5, ["L0", "L1.attn"]),
            (6, ["L1.ffn", "L2", "L3.attn"]),
            (7, ["L3.ffn", "L4", "L5.attn"]),
            (4, ["L5.ffn", "L6", "L7.attn"]),
            (5, ["L7.ffn"]),
        ]
        stages = [StagePlan(coord=coords[c], units=parse_units(us), chip=c) for c, us in spec]
        validate_plan(stages)
    else:
        layers = list(range(a.first, a.last + 1))
        chips = [int(c) for c in a.chips.split(",")]
        stages = even_plan(layers, chips, coords)
    loader = Loader(cfg, a.decoder, prefetch=bool(a.prefetch))
    t0 = time.perf_counter()
    m = MultichipDecoder(mesh, stages, loader=loader, handoff=a.handoff, submeshes=subs)
    load_s = time.perf_counter() - t0
    man = json.load(open(MANIFEST))["bytes_by_layer"]
    for l, b in loader.dram.items():
        record(
            dict(
                test="footprint/weights",
                layer=l,
                layer_kind=kind_of(l),
                chip=m.stage_of(l).plan.chip,
                device_bytes=b,
                manifest_bytes=sum(man[str(l)].values()),
                load_s=loader.load_s[l],
                decoder=a.decoder,
                status="pass",
            ),
            "footprint.jsonl",
        )
    record(
        dict(
            test="multichip_stack/build",
            mgd=a.mgd,
            mesh_shape=list(shape),
            coord_map={str(k): list(v) for k, v in coords.items()},
            plan=[{"chip": s.plan.chip, "coord": list(s.plan.coord), "layers": s.plan.layers} for s in m.stages],
            open_s=round(open_s, 1),
            load_s=round(load_s, 1),
            prefetch=bool(a.prefetch),
            decoder=a.decoder,
            traced_prefill=m.traced_prefill,
            handoff=a.handoff,
            hops=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()} for h in m.hops],
            released_unused_half={f"{c}/L{l}": b for (c, l), b in m.released.items()},
            status="pass",
        )
    )
    extras = place_extras(m, loader, man) if a.ref_split and a.mode in ("correctness", "both", "mock") else {}
    if loader.prefetch is not None:
        loader.prefetch.close()
    return mesh, m, extras


def place_extras(m: MultichipDecoder, loader, man):
    """A WHOLE single-chip instance of every split layer (the single-chip reference of a layer the plan cuts in two),
    on the chip with the most DRAM left after the pipeline's own future allocations (B=32 state twice: pipeline +
    reference caches, 0.53 GB prefill buffers, 1 GiB margin)."""
    gb = 2**30
    state = lambda l: 143_147_008 + 4 * 2**20 if l not in MLA else 83_894_784
    future = {}
    for st in m.stages:
        c = tuple(st.plan.coord)
        future[c] = future.get(c, 0.53 * gb + gb) + sum(2 * state(l) for l in st.state_layers)
    extras = {}
    for l in m.split_layers:
        need = sum(man[str(l)].values()) * 1.02 + state(l) + 0.53 * gb
        best, room = None, -1
        for c, sub in m.submeshes.items():
            free = dram_total(sub) - dram_allocated(sub) - future.get(c, 0)
            if free > room:
                best, room = c, free
        if room < need:
            record(dict(test=f"multichip_stack/reference_extra/L{l}", status="skipped", need=need, best_room=room))
            continue
        extras[l] = loader(l, m.submeshes[best], None)
        future[best] = future.get(best, 0) + state(l) + 0.53 * gb
        record(
            dict(
                test=f"multichip_stack/reference_extra/L{l}",
                layer=l,
                layer_kind=kind_of(l),
                coord=list(best),
                room_before=room,
                status="pass",
            )
        )
    return extras


def setup_batch(m: MultichipDecoder, batch, *, probes, plus_one, groups=1, permute=False, seed=0):
    t0 = time.perf_counter()
    msl = {l: max_seq(l) for l in m.layers}
    before = {st.plan.chip: dram_allocated(st.mesh) for st in m.stages}
    m.allocate(max_batch=batch, max_seq_len=msl, microbatches=groups, probe_units=probes, plus_one=plus_one)
    state = {st.plan.chip: dram_allocated(st.mesh) - before[st.plan.chip] for st in m.stages}
    bps = m.blocks_per_seq()
    table = None
    if bps is not None:
        n = batch * bps
        table = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if permute else torch.arange(n)
        for gi in range(groups):
            m.set_page_table(table.reshape(batch, bps), group=gi)
    return dict(alloc_s=round(time.perf_counter() - t0, 1), state_bytes_per_chip=state), table


def fallback_audit(m: MultichipDecoder, batch):
    """Runtime host-fallback audit (Stage 1 tests/test_fallback_audit.py method: ttnn host-transfer entry points
    replaced by tripwires + a torch call profiler) of exactly what the traces capture: every stage's decode program
    sequence (recv, units, send / out_buf copy, plus_one) as one eager pipeline step, and every unit's traced prefill
    body at a full 128-row chunk and a 40-token tail (host chunk inputs written outside the window, as Stage 1 /
    Stage 3 do). Runs before the captures (state restored after). Positive control first."""
    from models.autoports.zai_org_glm_5_3_flash.tests.test_fallback_audit import Audit

    st0 = m.stages[0]
    with Audit() as ctl:
        ttnn.from_torch(torch.zeros(1, 32), layout=ttnn.TILE_LAYOUT, device=st0.mesh)
    control = sorted(ctl.torch_calls)[:2] + ctl.tripwires
    ok = bool(ctl.torch_calls) and ctl.tripwires == ["from_torch"]
    snaps = [st.snapshot(st.groups[0]) for st in m.stages]
    pos = m._snapshot_positions(0)
    rows = []
    with Audit() as a:
        out = m.decode_eager()
    m.synchronize()
    rows.append(("decode_step_all_stages", a, out))
    for st, sn in zip(m.stages, snaps):
        st.restore(st.groups[0], sn)
    m._restore_positions(0, pos)
    for phys, n in ((128, 128), (64, 40)):
        for st in m.stages:
            g = st.groups[0]
            for i, u in enumerate(st.plan.units):
                dec = st.decoders[u.layer]
                cache = g.caches.get(u.layer)
                p = layer_bucket(dec, phys, n)
                t = m._unit_traces(st, g, i, u)[p]
                if u.part != "ffn":  # host chunk boundary: outside the audited window
                    dec.begin_prefill_chunk(cache, user_slot=0, chunk_start=0, n_valid=n)
                    if dec.is_kda:
                        dec._scratch_state_in(cache, 0)
                with Audit() as a:
                    m._unit_body(dec, u, t, cache)
                if u.part != "ffn" and dec.is_kda:
                    ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, 0)
                rows.append((f"prefill_body_{u}_{n}of{p}", a, t.y))
    m.reset_slot(0)
    m.synchronize()
    bad = []
    for mode, a, outt in rows:
        row = dict(
            test=f"multichip_stack/fallback_audit_B{batch}/{mode}",
            mode=mode,
            batch=batch,
            torch_calls=sum(a.torch_calls.values()),
            torch_call_sites=sorted(a.torch_calls)[:10],
            host_transfer_calls=a.tripwires,
            output_on_device=ttnn.is_tensor_storage_on_device(outt),
            positive_control_detected=control,
        )
        row["status"] = (
            "pass"
            if ok and not row["torch_calls"] and not row["host_transfer_calls"] and row["output_on_device"]
            else "fail"
        )
        record(row, "fallback_audit.jsonl")
        if row["status"] != "pass":
            bad.append(row["test"])
    return bad


def capture(m: MultichipDecoder, audit_batch=None, failures=None):
    t0 = time.perf_counter()
    m.prepare_prefill()
    t1 = time.perf_counter()
    if audit_batch is not None:
        bad = fallback_audit(m, audit_batch)
        if failures is not None:
            failures += bad
    tb = {st.plan.chip: trace_bytes(st.mesh) for st in m.stages}
    m.warm_and_capture()
    t2 = time.perf_counter()
    ta = {st.plan.chip: trace_bytes(st.mesh) for st in m.stages}
    return dict(
        prepare_prefill_s=round(t1 - t0, 1),
        warm_and_capture_s=round(t2 - t1, 1),
        trace_bytes_per_chip={str(c): (None if ta[c] is None or tb[c] is None else ta[c] - tb[c]) for c in ta},
    )


def correctness(a, mesh, m: MultichipDecoder, extras):
    g0 = golden(m.layers[0])["streams_in"].reshape(-1, HC_WIDTH).float()  # real activations (2048-token prompt)
    gold_out = {}
    if a.gold:
        for l in m.layers:
            gold_out[f"L{l}"] = golden(l)["streams_out"].reshape(-1, HC_WIDTH).float()
    failures = []
    # "<n>" / "<n>p" = the decoder's default chunk policy "plan" (pure power-of-two plan + tail); "<n>a" = the opt-in
    # "auto" policy (one padded chunk when < 32 padded rows and cheaper). The v2 run (20261018T234752Z, 9238180f1cd)
    # predates this: there a bare "<n>" meant "auto" and its "_plan" rows the plan.
    pol_of = lambda x: "auto" if x.endswith("a") else "plan"
    lengths = [(int(x.rstrip("pa")), pol_of(x)) for x in a.prefill_lengths.split(",") if x]
    for batch in [int(b) for b in a.batches.split(",")]:
        info, table = setup_batch(m, batch, probes=True, plus_one=True, permute=batch > 1, seed=batch)
        ref = Reference(m, batch, extras)  # reference caches / extra prefill buffers: allocated before any capture
        ref.prepare_extras()
        ref.warm()
        det_snaps = None
        if batch == 1:  # pre-capture persistent snapshot buffers for the replay-determinism check
            det_snaps = [st.snapshot(st.groups[0]) for st in m.stages]
        info.update(capture(m, audit_batch=batch, failures=failures))
        ref.capture_extras()
        entries0 = program_entries(m)
        foot = footprint(m, f"after_capture_B{batch}", dict(batch=batch, **info))
        record(
            dict(test=f"multichip_stack/setup_B{batch}", batch=batch, **info, program_entries=entries0, status="pass")
        )
        # ---- prefill: reference (single-chip API, host chained) then the pipeline, same slot, real activations
        slots_used = {}
        plan_slots = []
        if batch == 1:
            plan_slots = [(0, n, pol) for n, pol in lengths] + [(0, a.decode_prompt, "plan")]
        else:
            # several users at non-aligned lengths, the rest start decoding at position 0, two stay inactive
            ls = [int(x) for x in a.b32_prompts.split(",")]
            plan_slots = [(s, n, "plan") for s, n in zip(range(min(batch, len(ls))), ls)]
        neg_done = False
        for slot, n, pol in plan_slots:
            t0 = time.perf_counter()
            rows_n = prompt_rows(g0, n)
            rl = ref.prefill(rows_n, slot, policy=pol)
            t_ref = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, pl = m.prefill_forward(rows_n, user_slot=slot, keep_unit_outputs=True, chunk_policy=pol)
            t_pipe = time.perf_counter() - t0
            ng = min(n, 2048)  # golden rows exist for the first 2048 positions only
            gold = {k: v[:ng] for k, v in gold_out.items()} if gold_out else None
            plan_n = prefill_chunk_plan(n, m.buckets(), pol)
            chunks = [c[2] for c in plan_n]
            if not neg_done and n >= 64:
                neg_done = True
                negative_control(pl, rl, failures)
            compare_layers(
                f"prefill_B{batch}_slot{slot}_len{n}" + ("_auto" if pol == "auto" else ""),
                pl,
                rl,
                mode="prefill",
                gold=gold,
                extra=dict(
                    batch=batch,
                    slot=slot,
                    seq_len=n,
                    chunks=chunks,
                    chunk_policy=pol,
                    padded_rows=sum(c[2] - c[1] for c in plan_n),
                    in_trace=True,
                    t_pipe_s=round(t_pipe, 2),
                    t_ref_s=round(t_ref, 2),
                ),
                failures=failures,
            )
            slots_used[slot] = n
        # ---- independent post-prefill state control: split layers' pipeline state vs the extras' own caches
        for slot in sorted(slots_used):
            for r in ref.compare_state_extras(slot):
                ok = r["bitexact"]
                record(
                    dict(
                        test=f"multichip_stack/post_prefill_state_B{batch}_slot{slot}/L{r['layer']}",
                        batch=batch,
                        slot=slot,
                        prompt=slots_used[slot],
                        **r,
                        status="pass" if ok else "fail",
                    )
                )
                if not ok:
                    failures.append(f"post_prefill_state_B{batch}_slot{slot}/L{r['layer']}")
        # ---- decode: separate reference state chain seeded from the pipeline's post-prefill state
        ref.seed_from_pipeline()
        if batch == 1:
            pos0 = [slots_used.get(0, 0)]
        else:
            pos0 = [slots_used.get(s, 0) for s in range(batch)]
            for s in (batch - 1, batch - 2):  # inactive slots
                pos0[s] = -1
        inactive = [s for s, p in enumerate(pos0) if p < 0]
        active = [s for s, p in enumerate(pos0) if p >= 0]
        kda_watch = []  # (stage, layer) of one KDA layer per chip: inactive-slot state must stay bit-identical
        if inactive:
            for st in m.stages:
                kl = [l for l in st.state_layers if st.decoders[l].is_kda]
                if kl:
                    S = ttnn.to_torch(st.groups[0].caches[kl[0]].kda.S)
                    kda_watch.append((st, kl[0], S[inactive].clone()))
        m.set_decode_inputs(pos0)
        ref.set_positions(pos0)
        positions = list(pos0)
        # input rows: prompt slots continue their golden activations; slots without a prompt get distinct rows (so the
        # B>1 batch routes to distinct experts, not 22 copies of one row)
        row_off = [0 if s in slots_used else (1000 + 37 * s) for s in range(batch)]
        for si in range(a.steps):
            rows_in = torch.stack(
                [g0[((positions[s] if positions[s] >= 0 else 0) + row_off[s]) % 2048] for s in range(batch)]
            )
            if si > 0:  # in-trace plus_one advanced the pipeline's copies; the reference gets the host values
                ref.set_positions(positions)
            else:
                m.set_decode_inputs(positions)
            m.write_streams(rows_in)
            m.decode_step()
            pr = probe_rows(m, batch)
            rr = ref.decode(rows_in)
            gold = None
            if si == 0 and gold_out and 0 <= positions[0] < 2048 and active and active[0] == 0:  # slot 0 (B=1 / B>1)
                gold = {k: v[positions[0] : positions[0] + 1] for k, v in gold_out.items()}
            compare_layers(
                f"decode_B{batch}_step{si}",
                pr,
                rr,
                mode="decode",
                rows=active,
                gold=gold,
                extra=dict(batch=batch, step=si, positions=positions[:8], in_trace=True, paged_permuted=batch > 1),
                failures=failures,
            )
            positions = [p + 1 if p >= 0 else -1 for p in positions]
        # in-trace plus_one: every chip's cur_pos copy advanced by `steps` on active slots, -1 kept
        got = m.read_positions()
        ok = all(c == positions for c in got)
        record(
            dict(
                test=f"multichip_stack/plus_one_B{batch}",
                expected=positions[:8],
                got=[c[:8] for c in got],
                chips=len(got),
                status="pass" if ok else "fail",
            )
        )
        if not ok:
            failures.append(f"plus_one_B{batch}")
        for st, l, before in kda_watch:
            after = ttnn.to_torch(st.groups[0].caches[l].kda.S)[inactive]
            same = bool(torch.equal(before, after))
            record(
                dict(
                    test=f"multichip_stack/inactive_slot_state_B{batch}/L{l}",
                    chip=st.plan.chip,
                    slots=inactive,
                    bitexact=same,
                    status="pass" if same else "fail",
                )
            )
            if not same:
                failures.append(f"inactive_slot_state_B{batch}/L{l}")
        # replay determinism (B=1): the same step twice from the same pre-capture state
        if det_snaps is not None:
            m.set_decode_inputs([5])
            m.write_streams(g0[5:6])
            outs = []
            for _ in range(2):
                for st, sn in zip(m.stages, det_snaps):
                    st.restore(st.groups[0], sn)
                m.set_decode_inputs([5])
                outs.append(ttnn.to_torch(m.decode_step()).float().clone())
            same = bool(torch.equal(outs[0], outs[1]))
            record(
                dict(
                    test=f"multichip_stack/decode_B{batch}_replay_determinism",
                    bitexact=same,
                    in_trace=True,
                    status="pass" if same else "fail",
                )
            )
            if not same:
                failures.append("replay_determinism")
            # prefill determinism: the same short prompt twice -> bitwise equal last-layer rows
            y1 = m.prefill_forward(g0[:65], user_slot=0)
            y2 = m.prefill_forward(g0[:65], user_slot=0)
            same = bool(torch.equal(y1, y2))
            record(
                dict(
                    test=f"multichip_stack/prefill_B{batch}_replay_determinism",
                    seq_len=65,
                    bitexact=same,
                    in_trace=True,
                    status="pass" if same else "fail",
                )
            )
            if not same:
                failures.append("prefill_determinism")
        # stress (B=1): a fresh 100-token prompt, then `stress_steps` consecutive traced steps with positions advanced
        # only by the in-trace plus_one (no host position write), every layer compared with the reference each step
        if batch == 1 and a.stress_steps:
            n0 = 100
            m.prefill_forward(g0[:n0], user_slot=0)
            ref.seed_from_pipeline()
            m.set_decode_inputs([n0])
            worst, exact_all = 1.0, True
            for si in range(a.stress_steps):
                p0 = n0 + si
                ref.set_positions([p0])
                m.write_streams(g0[p0 : p0 + 1])
                m.decode_step()
                pr, rr = probe_rows(m, 1), ref.decode(g0[p0 : p0 + 1])
                for k in rr:
                    worst = min(worst, pcc(pr[k], rr[k]))
                    exact_all &= bool(torch.equal(pr[k], rr[k]))
                if si == a.stress_steps - 1:
                    compare_layers(
                        f"decode_B1_stress_step{si}",
                        pr,
                        rr,
                        mode="decode",
                        extra=dict(batch=1, step=si, positions=[p0], in_trace=True, stress=True),
                        failures=failures,
                    )
            ok = worst >= PCC_BAR and exact_all
            record(
                dict(
                    test="multichip_stack/decode_B1_stress",
                    steps=a.stress_steps,
                    start_position=n0,
                    min_pcc=round(worst, 8),
                    all_bitexact=exact_all,
                    positions_from="in-trace plus_one",
                    status="pass" if ok else "fail",
                )
            )
            if not ok:
                failures.append("decode_B1_stress")
        entries1 = program_entries(m)
        same = entries1 == entries0
        record(
            dict(
                test=f"multichip_stack/no_compile_after_capture_B{batch}",
                after_capture=entries0,
                end=entries1,
                status="pass" if same else "fail",
            )
        )
        if not same:
            failures.append(f"compile_after_capture_B{batch}")
        m.free()
        ref.free()
        del ref, det_snaps
    record(
        dict(test="multichip_stack/correctness_summary", failures=failures, status="pass" if not failures else "fail")
    )
    return failures


def perf(a, mesh, m: MultichipDecoder):
    g0 = golden(m.layers[0])["streams_in"].reshape(-1, HC_WIDTH).float()
    reps = a.reps
    perf_batches = [int(b) for b in (a.perf_batches or a.batches).split(",")]
    for batch in perf_batches:
        info, _ = setup_batch(m, batch, probes=False, plus_one=True, permute=batch > 1)
        footprint(m, f"perf_before_capture_B{batch}", dict(batch=batch, note="production buffers, no snapshots"))
        snap = StateSnap(m)  # test-only, allocated before the captures
        info.update(capture(m))
        footprint(m, f"perf_after_capture_B{batch}", dict(batch=batch, snapshot_buffers_included=True, **info))
        n_chips = len({st.plan.chip for st in m.stages})
        row = dict(
            test="multichip_stack/perf",
            batch=batch,
            layers=len(m.layers),
            chips=n_chips,
            stages=len(m.stages),
            reps=reps,
            decode_inputs="distinct users: slot s at position decode_prompt - s, golden row of that position",
            **info,
        )
        # prefill: traced pipeline, host input -> last chip output (valid rows), several logical lengths; "<n>a" = the
        # opt-in "auto" chunk policy instead of the default pure power-of-two plan
        specs = [x for x in a.perf_prefill.split(",") if x] if batch == perf_batches[0] else []  # batch-independent
        for spec in specs:
            n, pol = int(spec.rstrip("pa")), ("auto" if spec.endswith("a") else "plan")
            tag = f"{n}" + ("_auto" if pol == "auto" else "")
            m.prefill_forward(prompt_rows(g0, n), user_slot=0, chunk_policy=pol)  # warm the replay path
            hc = m.prefill_host_chunks(prompt_rows(g0, n), chunk_policy=pol)  # host tilization outside the timing
            ts, te = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                m.prefill_forward(
                    prompt_rows(g0, n), user_slot=0, return_output=False, host_chunks=hc, chunk_policy=pol
                )
                m.synchronize()
                ts.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                m.prefill_forward(prompt_rows(g0, n), user_slot=0, return_output=True, chunk_policy=pol)
                te.append(time.perf_counter() - t0)
            row[f"prefill_{tag}_ms"] = round(statistics.median(ts) * 1e3, 2)
            row[f"prefill_{tag}_host_io_ms"] = round(statistics.median(te) * 1e3, 2)
            row[f"prefill_{tag}_chunks"] = [c[2] for c in prefill_chunk_plan(n, m.buckets(), pol)]
        # decode state: the 2047-token prompt in slot 0; slot s > 0 decodes at position decode_prompt - s with the
        # golden row of that position (distinct users -> distinct routed experts, as Stage 3's B>1 perf rows; the first
        # 8-chip run had fed slots 1.. the same row at position 0, which collapses the MoE expert union)
        m.prefill_forward(g0[: a.decode_prompt], user_slot=0, return_output=False)
        pos = [a.decode_prompt - s for s in range(batch)]
        m.set_decode_inputs(pos)
        m.write_streams(torch.stack([g0[p % 2048] for p in pos]))
        snap.take()
        m.decode_step()  # warm replay
        ts = []
        for _ in range(reps):  # consecutive autoregressive steps (plus_one advances the positions in-trace)
            t0 = time.perf_counter()
            m.decode_step()
            ts.append(time.perf_counter() - t0)
        row["decode_step_ms"] = round(statistics.median(ts) * 1e3, 3)
        row["decode_step_ms_all"] = [round(t * 1e3, 3) for t in ts]
        # overhead composition: the same steps waiting only on the tail stage's queue (every other stage finished
        # before the tail could: it consumes their output), the host time of the 9 enqueues alone, and the host cost of
        # synchronize() on an idle mesh (the blocking step pays it after the tail completes)
        tail_mesh = m.stages[-1].mesh
        ts_tail, ts_enq, ts_idle = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            m.decode_step(blocking=False)
            ts_enq.append(time.perf_counter() - t0)
            ttnn.synchronize_device(tail_mesh)
            ts_tail.append(time.perf_counter() - t0)
            m.synchronize()
            t0 = time.perf_counter()
            m.synchronize()
            ts_idle.append(time.perf_counter() - t0)
        row["decode_step_tail_sync_ms"] = round(statistics.median(ts_tail) * 1e3, 3)
        row["decode_enqueue_ms"] = round(statistics.median(ts_enq) * 1e3, 3)
        row["synchronize_idle_ms"] = round(statistics.median(ts_idle) * 1e3, 3)
        # single-chip baseline: each chip's layers alone as a compute-only trace (no recv / send), same state
        comp = []
        for st in m.stages:
            snap.put()
            m.set_decode_inputs(pos)
            g = st.groups[0]
            tid = ttnn.begin_trace_capture(st.mesh, cq_id=0)
            try:
                o = st.decode_units(g, g.streams_in)
            finally:
                ttnn.end_trace_capture(st.mesh, tid, cq_id=0)
            ttnn.deallocate(o)

            def run(mm=st.mesh, t=tid):
                ttnn.execute_trace(mm, t, cq_id=0, blocking=False)
                ttnn.synchronize_device(mm)

            run()
            tt = []
            for _ in range(reps):
                t0 = time.perf_counter()
                run()
                tt.append(time.perf_counter() - t0)
            ttnn.release_trace(st.mesh, tid)
            comp.append(round(statistics.median(tt) * 1e3, 3))
        row["chip_compute_only_ms"] = comp
        row["single_chip_sum_ms"] = round(sum(comp), 3)
        row["handoff_overhead_ms"] = round(row["decode_step_ms"] - sum(comp), 3)
        row["handoff_overhead_tail_sync_ms"] = round(row["decode_step_tail_sync_ms"] - sum(comp), 3)
        row["speedup_latency"] = round(sum(comp) / row["decode_step_ms"], 4)
        row["efficiency_latency"] = round(row["speedup_latency"] / n_chips, 4)  # per chip (stages share chip 0)
        row["max_chip_ms"] = max(comp)
        record(dict(**row, status="pass"), "perf.jsonl")
        snap.free()
        m.free()
    # micro-batch pipelining: G groups of B=1 in flight (each its own caches / traces): aggregate throughput
    for G in [int(x) for x in a.groups_perf.split(",") if x]:
        info, _ = setup_batch(m, 1, probes=False, plus_one=True, groups=G)
        snaps = [StateSnap(m, gi) for gi in range(G)]  # before the captures
        t0 = time.perf_counter()
        m.warm_and_capture(prefill=False)  # decode-only micro-batch groups (no prefill traces per group)
        info["warm_and_capture_s"] = round(time.perf_counter() - t0, 1)
        for gi in range(G):
            m.set_decode_inputs([100 + gi], group=gi)
            m.write_streams(g0[100 + gi : 101 + gi], group=gi)
        m.decode_steps(list(range(G)), 2)  # warm
        # pipelined dispatch == serial replays, bitwise (same start state, 3 steps per group)
        for sn in snaps:
            sn.take()

        def reset_groups():
            for gi in range(G):
                snaps[gi].put()
                m.set_decode_inputs([100 + gi], group=gi)

        reset_groups()
        m.decode_steps(list(range(G)), 3)
        piped = [ttnn.to_torch(m.output(group=gi)).float().clone() for gi in range(G)]
        reset_groups()
        for _ in range(3):
            for gi in range(G):
                m.decode_step(gi)
        serial = [ttnn.to_torch(m.output(group=gi)).float().clone() for gi in range(G)]
        same = all(torch.equal(x, y) for x, y in zip(piped, serial))
        record(
            dict(
                test=f"multichip_stack/pipelined_dispatch_G{G}",
                groups=G,
                steps=3,
                bitexact=same,
                status="pass" if same else "fail",
            )
        )
        t0 = time.perf_counter()
        m.decode_steps(list(range(G)), reps)  # pipelined dispatch: G micro-batches in flight
        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(reps // 4 or 1):  # same groups, one at a time (no overlap): the serial reference
            for gi in range(G):
                m.decode_step(gi)
        dt_serial = (time.perf_counter() - t0) / ((reps // 4 or 1) * G)
        record(
            dict(
                test="multichip_stack/perf_microbatch",
                groups=G,
                batch_per_group=1,
                reps=reps,
                wall_s=round(dt, 3),
                ms_per_group_step=round(dt / (reps * G) * 1e3, 3),
                tokens_per_s=round(reps * G / dt, 2),
                serial_ms_per_group_step=round(dt_serial * 1e3, 3),
                max_inflight=m.max_inflight(G),
                pipelining_speedup=round(dt_serial / (dt / (reps * G)), 3),
                trace_alloc_tracking_in_process=TRACKING,
                **info,
                status="pass",
            ),
            "perf.jsonl",
        )
        for sn in snaps:
            sn.free()
        m.free()


def mock_run(a, mesh, m: MultichipDecoder, extras):
    """Mock cluster (no capture: the mock segfaults in trace dispatch generation, Stage 3 work log): allocation, the
    eager prefill pass of every bucket, eager decode, eager unit-level prefill with per-unit outputs."""
    g0 = golden(m.layers[0])["streams_in"].reshape(-1, HC_WIDTH).float()
    for batch in [int(b) for b in a.batches.split(",")]:
        setup_batch(m, batch, probes=True, plus_one=True, permute=batch > 1)
        ref = Reference(m, batch, extras)
        ref.prepare_extras()
        ref.warm()
        m.prepare_prefill()
        bad = fallback_audit(m, batch)
        record(dict(test=f"mock/fallback_audit_B{batch}", bad=bad, status="pass"))
        m.set_decode_inputs([3] * batch)
        m.write_streams(g0[:batch])
        m.decode_eager()
        m.synchronize()
        for n in (20, 65):
            _, units = m._prefill_traced(
                g0[:n], user_slot=0, start_pos=0, group=0, keep_unit_outputs=True, eager=True, return_output=True
            )
            record(dict(test=f"mock/prefill_eager_{n}_B{batch}", units=sorted(units), status="pass"))
        # code paths of the hardware-only checks (values are garbage on the mock): the split-layer state comparison
        # and the comparator's negative control
        st_rows = ref.compare_state_extras(0)
        record(dict(test=f"mock/post_prefill_state_path_B{batch}", layers=[r["layer"] for r in st_rows], status="pass"))
        whole = {k: v for k, v in units.items() if "." not in k}
        negative_control(whole, {k: v.clone() for k, v in whole.items()}, [])
        record(dict(test=f"mock/decode_eager_B{batch}", status="pass"))
        m.free()
        ref.free()


def tracy_run(a, mesh, m: MultichipDecoder):
    """Profiled windows (run under `python -m tracy -r -p`; no correctness checks in this process), the device profiler
    read after every replay: PERF_DECODE_B<b> (traced decode steps) and, with --tracy-prefill 1, PERF_PREFILL_<n>_B<b>
    (traced pipeline prefill; needs the eager compile pass of every bucket, so keep --max-seq small and the stack short:
    the 45-layer profile of that pass would exceed what tracy-capture handles, Stage 3 work_log 22:57Z)."""
    g0 = golden(m.layers[0])["streams_in"].reshape(-1, HC_WIDTH).float()
    for batch in [int(b) for b in a.batches.split(",")]:
        setup_batch(m, batch, probes=False, plus_one=True, permute=batch > 1)
        if a.tracy_prefill:
            m.prepare_prefill()
            flush(m)
        m.warm_and_capture(prefill=bool(a.tracy_prefill))
        m.synchronize()
        flush(m)
        if a.tracy_prefill:
            for n in [int(x) for x in a.perf_prefill.split(",") if x]:
                hc = m.prefill_host_chunks(g0[:n])
                m.prefill_forward(g0[:n], user_slot=0, return_output=False, host_chunks=hc)
                m.synchronize()
                flush(m)
                signpost(f"PERF_PREFILL_{n}_B{batch}")
                m.prefill_forward(g0[:n], user_slot=0, return_output=False, host_chunks=hc)
                m.synchronize()
                flush(m)
                signpost(f"PERF_PREFILL_{n}_B{batch}_END")
        base = min(100, MAX_SEQ["mla"] - 8)
        pos = [max(base - s, 0) for s in range(batch)]  # distinct users (distinct rows -> distinct routed experts)
        m.set_decode_inputs(pos)
        m.write_streams(torch.stack([g0[p] for p in pos]))
        m.decode_step()  # warm replay
        flush(m)
        signpost(f"PERF_DECODE_B{batch}")
        for _ in range(a.tracy_steps):
            m.decode_step()
            flush(m)
        signpost(f"PERF_DECODE_B{batch}_END")
        record(
            dict(
                test="multichip_stack/tracy_windows",
                batch=batch,
                decode_steps=a.tracy_steps,
                prefill=a.perf_prefill if a.tracy_prefill else None,
                max_seq=MAX_SEQ,
                status="pass",
            ),
            "perf.jsonl",
        )
        m.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgd", default="2x4", choices=sorted(MGDS))
    ap.add_argument("--plan", default="partition", choices=["partition", "even", "wrap4", "units"])
    ap.add_argument("--units", default="", help='"chip:U,U;chip:U,..." ("/" also separates stages)')
    ap.add_argument("--ref-split", type=int, default=1, help="whole-layer reference instances for split layers")
    ap.add_argument("--partition", default=os.path.join(E, "partition.json"))
    ap.add_argument("--partition-key", default="pipeline8_ring_wrap")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=7)
    ap.add_argument("--chips", default="5,6,7,4")
    ap.add_argument("--decoder", default="optimized", choices=["optimized", "fused"])
    ap.add_argument("--handoff", default="eth_direct", choices=["eth_direct", "eth_fifo", "host"])
    ap.add_argument("--mode", default="correctness", choices=["correctness", "perf", "both", "tracy", "mock"])
    ap.add_argument("--max-seq", type=int, default=0, help="cap every layer's max_seq_len (Tracy runs)")
    ap.add_argument("--tracy-steps", type=int, default=2)
    ap.add_argument("--tracy-prefill", type=int, default=0)
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--prefill-lengths", default="1,20,33,63,64,65,100,2051")
    ap.add_argument("--decode-prompt", type=int, default=2047)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--stress-steps", type=int, default=32)
    ap.add_argument("--b32-prompts", default="2047,20,33,65,100,1,64,130", help="B>1: prompt length per slot 0..")
    ap.add_argument("--gold", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--perf-prefill", default="2048,128,20,1999,2047,2047a", help='"<n>a" = opt-in "auto" chunk policy')
    ap.add_argument("--perf-batches", default="", help="perf phase batches (default: --batches)")
    ap.add_argument("--prefetch", type=int, default=1, help="page-cache prefetch of the next layers' weight files")
    ap.add_argument("--groups-perf", default="")
    ap.add_argument("--trace-region", type=int, default=600_000_000)
    ap.add_argument("--fabric", type=int, default=1, help="0: no fabric (mock cluster; needs --handoff host)")
    a = ap.parse_args()
    if a.max_seq:
        MAX_SEQ["kda"] = MAX_SEQ["mla"] = a.max_seq
    mesh, m, extras = build(a)
    failures = []
    try:
        if a.mode in ("correctness", "both"):
            failures = correctness(a, mesh, m, extras)
        if a.mode in ("perf", "both"):
            for dec in extras.values():  # reference-only instances: not part of the measured footprint
                free_decoder(dec)
            extras = {}
            perf(a, mesh, m)
        if a.mode == "tracy":
            tracy_run(a, mesh, m)
        if a.mode == "mock":
            mock_run(a, mesh, m, extras)
    finally:
        close_pipeline_mesh(mesh, m, fabric=bool(a.fabric))
    print("FAILURES:", failures, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
