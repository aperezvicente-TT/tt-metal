# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 04a: 2-chip pipeline of real GLM-5.3-Flash layers (tt/multichip_decoder.py) against the single-chip fused
decoder (Stage 2) running the same layers, on one card (TT_VISIBLE_DEVICES = two Ethernet-adjacent chips; card2 = 4,5).

Layers 0-4 cover every layer kind (dense_kda 0-2, sparse_dsa 3, sparse_kda 4) and have fp32 goldens (goldens/layer_io).
Plans (stage A = mesh coord (0,0), stage B = (0,1)):
  whole  A: L0 L1            | B: L2 L3 L4                  (layer boundary)
  split  A: L0 L1 L2 L3.attn | B: L3.ffn L4                 (mid-layer boundary inside the MLA+MoE layer 3)
  wrap   A: L0 L1.attn       | B: L1.ffn L2 L3 L4.attn | A: L4.ffn
         (the ring wrap-around in small: two pipeline segments = two traces on chip A, a B -> A closing hop, mid-layer
          boundaries inside a dense_kda layer (L1) and a sparse_kda layer (L4))
Configurations: every plan x {eth_direct, host}, plus `whole` eth_direct with 2 micro-batch groups (each group its own
caches and traces; groups replayed interleaved and compared against two independent single-chip references).
Reference: the same FusedDecoder layers all on chip A, driven through their own public API (prefill_forward per layer
with the host loop; decode_forward chained on device, one trace).
Checks per unit output: bitwise equality pipeline vs reference (same programs, same inputs, exact fp32 hand-off), PCC
pipeline vs reference (gate >= 0.995, the goal's per-layer-kind bar), PCC vs the fp32 golden (reported).
Phases: prefill 20 tokens (KDA bucket 32 -> MLA bucket 64 re-bucketing), prefill 2047 tokens, traced decode B=1 at
position 2047 (golden row) + 2 more steps, then (traces live) prefill 20 tokens again + a traced decode step at position
20 (prefill after capture), traced decode B=32 from position 0 (3 steps); replay determinism. Latency (same state, same
inputs for every measurement): traced decode step with hand-off (pipeline chain) vs without (per-chip compute-only
traces; single-chip reference trace). Reserve check: DRAM ballast leaves only the activation budget of the partition's
2 GiB per-chip reserve free (minus the trace region) and the `wrap` plan runs prefill 2047 + B=32 traced decode in it.
Rows -> $GLM53_E4A/tests.jsonl, perf -> pipeline_perf.jsonl, footprint -> footprint_rows.jsonl.

  GLM53_LOCK=card2 scripts/dev.sh python models/autoports/zai_org_glm_5_3_flash/tests/test_multichip_pipeline.py
(TT_METAL_TRACE_ALLOC_TRACKING=1 for the acceptance run; a separate untracked run for the perf rows.)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedDecoder
from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import (
    HC_WIDTH,
    MultichipDecoder,
    StagePlan,
    close_pipeline_mesh,
    dram_allocated,
    open_pipeline_mesh,
    parse_units,
)

E = os.environ.get(
    "GLM53_E4A", "/opt/glm53/kit/doc/ttnn/04a-multichip-handoff-prework"
)
TRACKING = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
LAYERS = [0, 1, 2, 3, 4]
KIND = {0: "dense_kda", 1: "dense_kda", 2: "dense_kda", 3: "sparse_dsa", 4: "sparse_kda"}
PLANS = {
    "whole": [((0, 0), ["L0", "L1"]), ((0, 1), ["L2", "L3", "L4"])],
    "split": [((0, 0), ["L0", "L1", "L2", "L3.attn"]), ((0, 1), ["L3.ffn", "L4"])],
    "wrap": [((0, 0), ["L0", "L1.attn"]), ((0, 1), ["L1.ffn", "L2", "L3", "L4.attn"]), ((0, 0), ["L4.ffn"])],
}
# (name, plan, hand-off, micro-batch groups)
CONFIGS = [
    ("whole/eth_direct", "whole", "eth_direct", 1),
    ("whole/host", "whole", "host", 1),
    ("split/eth_direct", "split", "eth_direct", 1),
    ("split/host", "split", "host", 1),
    ("wrap/eth_direct", "wrap", "eth_direct", 1),
    ("wrap/host", "wrap", "host", 1),
    ("whole_mb2/eth_direct", "whole", "eth_direct", 2),
]
GIB = 2**30
PCC_BAR = 0.995


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def record(row, name="tests.jsonl"):
    row = {
        "time_utc": now(),
        **row,
        "trace_alloc_tracking": TRACKING,
        "watcher": os.environ.get("TT_METAL_WATCHER", ""),
        "visible": os.environ.get("TT_VISIBLE_DEVICES"),
        "lock": os.environ.get("TT_LOCK_GROUP"),
    }
    out_dir = os.environ.get("GLM53_TESTS_OUT_DIR", E) if name == "tests.jsonl" else E  # perf rows always to E
    with open(os.path.join(out_dir, name), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row), flush=True)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def max_seq(layer):
    return 2048 + 64 if KIND[layer].endswith("kda") else DSA_DENSE_EXACT_MAX_KV


class Loader:
    """One FusedDecoder per (layer, chip, half), shared by the reference and every pipeline plan; half-layer instances
    are private (the pipeline releases their unused half). Records the device DRAM each load takes."""

    def __init__(self, cfg):
        self.cfg, self.cache, self.load_s, self.dram = cfg, {}, {}, {}

    def __call__(self, layer, mesh, half=None):
        key = (layer, id(mesh), half)
        if key not in self.cache:
            t0 = time.perf_counter()
            b0 = dram_allocated(mesh)
            self.cache[key] = FusedDecoder.from_weight_cache(
                hf_config=self.cfg, layer_idx=layer, mesh_device=mesh, cache_dir=CACHE_DIR, manifest_path=MANIFEST
            )
            ttnn.synchronize_device(mesh)
            tag = f"L{layer}{'.' + half if half else ''}@{mesh.get_device_ids()[0]}"
            self.load_s[tag] = round(time.perf_counter() - t0, 1)
            self.dram[tag] = dram_allocated(mesh) - b0
        return self.cache[key]


class Reference:
    """Single-chip fused decoder over LAYERS on one submesh, through its public API."""

    def __init__(self, mesh, loader):
        self.mesh = mesh
        self.decs = {l: loader(l, mesh) for l in LAYERS}
        self.caches, self.pts = {}, {}
        self.trace = None

    def allocate(self, batch):
        self.release()
        self.batch = batch
        self.caches, self.pts = {}, {}
        self.state_bytes = {}
        for l, d in self.decs.items():
            b0 = dram_allocated(self.mesh)
            self.caches[l] = d.allocate_cache(max_batch=batch, max_seq_len=max_seq(l))
            self.pts[l] = None if d.is_kda else d.make_page_table(self.caches[l])
            ttnn.synchronize_device(self.mesh)
            self.state_bytes[l] = dram_allocated(self.mesh) - b0
        self.cur_pos = ttnn.from_torch(torch.full((batch,), -1, dtype=torch.int32), dtype=ttnn.int32, device=self.mesh)
        self.x_in = ttnn.from_torch(
            torch.zeros(1, 1, batch, HC_WIDTH),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def reset_slot(self, slot):
        for l, d in self.decs.items():
            d.reset_slot(self.caches[l], slot)

    def prefill(self, rows):
        """Stage 2 host loop: FusedDecoder.prefill_forward per layer (pad to the layer's bucket, read valid rows)."""
        x, outs = rows, {}
        for l in LAYERS:
            x = self.decs[l].prefill_forward(x, cache=self.caches[l], user_slot=0)
            outs[f"L{l}"] = x
        return outs

    def _decode(self, x):
        outs = []
        for l in LAYERS:
            x = self.decs[l].decode_forward(x, cache=self.caches[l], page_table=self.pts[l], cur_pos=self.cur_pos)
            outs.append(x)
        return outs

    def capture(self):
        snap = {l: d.snapshot_state(self.caches[l]) for l, d in self.decs.items()}
        for t in self._decode(self.x_in):
            ttnn.deallocate(t)
        for l, d in self.decs.items():
            d.restore_state(self.caches[l], snap[l])
        ttnn.synchronize_device(self.mesh)
        self.trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.outs = self._decode(self.x_in)
        ttnn.end_trace_capture(self.mesh, self.trace, cq_id=0)

    def set_inputs(self, positions, streams):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32), self.cur_pos
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                streams.reshape(1, 1, self.batch, HC_WIDTH).float().contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            ),
            self.x_in,
        )

    def step(self):
        ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.mesh, self.trace)
            self.trace = None


def unit_names(plan):
    return [str(u) for _, us in PLANS[plan] for u in parse_units(us)]


def layer_out_names(plan):
    """Unit outputs that are layer outputs (a split layer's output is its ffn half)."""
    return {n: n.split(".")[0] for n in unit_names(plan) if not n.endswith(".attn")}


def rows_of(t, n):
    return ttnn.to_torch(t).float().reshape(-1, HC_WIDTH)[:n]


def compare_rows(test, cfg_name, plan, phase, pipe_units, ref_layers, n, gold=None, extra=None):
    """pipe_units {unit name: torch [1|rows, ...]} vs ref_layers {"L<l>": torch}; one row per layer output."""
    ok = True
    for uname, lname in layer_out_names(plan).items():
        p = pipe_units[uname].reshape(-1, HC_WIDTH)[:n]
        r = ref_layers[lname].reshape(-1, HC_WIDTH)[:n]
        exact = bool(torch.equal(p, r))
        v = pcc(p, r)
        g = None
        if gold is not None and lname in gold:
            g = pcc(p, gold[lname].reshape(-1, HC_WIDTH)[:n])
        status = "pass" if (v >= PCC_BAR and torch.isfinite(p).all()) else "fail"
        ok &= status == "pass"
        record(
            dict(
                test=f"{test}/{cfg_name}/{phase}/{lname}",
                layer=int(lname[1:]),
                layer_kind=KIND[int(lname[1:])],
                plan=cfg_name.split("/")[0],
                handoff=cfg_name.split("/")[1],
                phase=phase,
                rows=n,
                pcc=round(v, 8),
                bitexact=exact,
                max_abs_diff=float((p - r).abs().max()),
                pcc_vs_golden=None if g is None else round(g, 6),
                status=status,
                **(extra or {}),
            )
        )
    return ok


def median_time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def probe_rows(p, batch, group=0):
    """{unit name: torch rows} from the persistent per-unit probe buffers (filled by the last replay of the group)."""
    out = {}
    for st in p.stages:
        for u, t in zip(st.plan.units, st.groups[group].probes):
            out[str(u)] = rows_of(t, batch)
    return out


def ref_eager_step(ref, batch):
    """Single-chip reference step through its public decode_forward (programs compiled before any capture)."""
    outs = ref._decode(ref.x_in)
    rl = {f"L{l}": rows_of(t, batch) for l, t in zip(LAYERS, outs)}
    for t in outs:
        ttnn.deallocate(t)
    return rl


def ref_snapshot(r):
    """Full reference state: KDA recurrent state + conv rows (decoder snapshot) AND the MLA latent pages, which a later
    phase can rewrite (a post-capture prefill + decode at position 20 rewrites the page row of position 20 with the
    decode-path value) and which a later configuration's decode attends to."""
    snap = {}
    for l, d in r.decs.items():
        snap[l] = ("kda", d.snapshot_state(r.caches[l])) if d.is_kda else ("mla", ttnn.clone(r.caches[l].latent))
    return snap


def ref_restore(r, snap):
    for l, (kind, sn) in snap.items():
        if kind == "kda":
            r.decs[l].restore_state(r.caches[l], sn)
        else:
            ttnn.copy(sn, r.caches[l].latent)


def snapshot_all(p, group=0):
    return [st.snapshot(st.groups[group]) for st in p.stages]


def restore_all(p, snaps, group=0):
    for st, sn in zip(p.stages, snaps):
        st.restore(st.groups[group], sn)


def reserve_check(p, subs, steps, g0, budget_bytes, trace_region):
    """Leave only `budget_bytes` of DRAM free on both chips (ballast tensors), then run prefill 2047 + traced decode at
    B = p.max_batch through the plan; succeeds => transient activations fit the partition's per-chip reserve. Host
    hand-off plan (no device-side waits that an OOM on one chip could leave hanging on the other)."""
    ballast = []
    free = {}
    for c, m in subs.items():
        v = ttnn.get_memory_view(m, ttnn.BufferType.DRAM)
        total = int(v.total_bytes_per_bank) * int(v.num_banks)
        free_now = total - dram_allocated(m)
        for rows, chunk in ((8192, 256 << 20), (512, 16 << 20)):  # coarse then fine ballast: free ends <= budget
            while free_now - chunk >= budget_bytes:
                ballast.append(
                    ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, 1, rows, 16384]), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, ttnn.DRAM_MEMORY_CONFIG
                    )
                )
                free_now = total - dram_allocated(m)
        free[str(c)] = free_now
    row = dict(
        test="two_chip_pipeline/reserve_check",
        plan=[str(u) for st in p.stages for u in st.plan.units],
        batch=p.max_batch,
        free_bytes_during=free,
        budget_bytes=budget_bytes,
        trace_region_bytes=trace_region,
        reserve_bytes=budget_bytes + trace_region,
    )
    try:
        p.reset_slot(0)
        t0 = time.perf_counter()
        p.prefill_forward(g0[:, :2047], user_slot=0)
        row["prefill2047_s"] = round(time.perf_counter() - t0, 3)
        for st in p.stages:
            ttnn.synchronize_device(st.mesh)
        p.set_decode_inputs(steps[0][0])
        p.write_streams(steps[0][1])
        p.warm_and_capture()
        for _ in range(3):
            p.decode_step()
        row["status"] = "pass"
    except Exception as e:  # noqa: BLE001
        row["status"] = "fail"
        row["error"] = repr(e)[:400]
    finally:
        p.release()
        for t in ballast:
            ttnn.deallocate(t)
        record(row, name="footprint_rows.jsonl")
    return row["status"] == "pass"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--trace-region", type=int, default=800_000_000)
    ap.add_argument("--reserve-gib", type=float, default=2.0)
    ap.add_argument("--no-reserve-check", action="store_true")
    a = ap.parse_args()
    configs = [c for c in CONFIGS if c[0] in a.configs.split(",")]
    cfg = hf_config()
    mesh = open_pipeline_mesh((1, 2), trace_region_size=a.trace_region)
    subs = {
        (0, 0): mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0)),
        (0, 1): mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1)),
    }
    loader = Loader(cfg)
    pipes = {}
    try:
        t0 = time.perf_counter()
        ref = Reference(subs[(0, 0)], loader)
        ref2 = Reference(subs[(0, 0)], loader)  # independent state for micro-batch group 1
        for i, (name, plan, ho, mb) in enumerate(configs):
            stages = [StagePlan(coord=c, units=parse_units(us)) for c, us in PLANS[plan]]
            pipes[name] = MultichipDecoder(
                mesh, stages, loader=loader, handoff=ho, submeshes=subs, link_core_row=(4 * i) % 8
            )
        print(f"loaded in {time.perf_counter() - t0:.0f} s: {loader.load_s}", flush=True)
        man = json.load(open(MANIFEST))["bytes_by_layer"]
        released = {}
        for p in pipes.values():  # the first pipeline that releases a half records it; later ones reuse the instance
            for k, v in p.released.items():
                released[f"{k[0]}/L{k[1]}"] = max(v, released.get(f"{k[0]}/L{k[1]}", 0))
        for tag, b in loader.dram.items():
            l = int(tag[1:].split("@")[0].split(".")[0])
            record(
                dict(
                    test="footprint/weights",
                    tag=tag,
                    layer=l,
                    layer_kind=KIND[l],
                    device_bytes=b,
                    manifest_bytes=sum(man[str(l)].values()),
                    status="pass",
                ),
                name="footprint_rows.jsonl",
            )
        record(
            dict(test="footprint/released_unused_half", released=released, status="pass"), name="footprint_rows.jsonl"
        )
        g0 = golden(0, 2048)["streams_in"].reshape(1, -1, HC_WIDTH).float()
        gold = {f"L{l}": golden(l, 2048)["streams_out"].reshape(1, -1, HC_WIDTH).float() for l in LAYERS}

        for batch in [int(b) for b in a.batches.split(",")]:
            # ---- allocation (no trace is live here)
            for r in (ref, ref2):
                r.allocate(batch)
            for l, b in ref.state_bytes.items():
                record(
                    dict(
                        test="footprint/state",
                        layer=l,
                        layer_kind=KIND[l],
                        batch=batch,
                        max_seq_len=max_seq(l),
                        device_bytes=b,
                        status="pass",
                    ),
                    name="footprint_rows.jsonl",
                )
            for name, plan, ho, mb in configs:
                pipes[name].allocate(
                    max_batch=batch, max_seq_len={l: max_seq(l) for l in LAYERS}, microbatches=mb, probe_units=True
                )
            if batch == 1:
                steps = [([2047], g0[:, 2047:2048]), ([2048], g0[:, 5:6]), ([2049], g0[:, 6:7])]
                steps_g1 = [([20 + s], g0[:, 20 + s : 21 + s]) for s in range(3)]
            else:
                steps = [([s] * batch, g0[:, s * batch : (s + 1) * batch]) for s in range(3)]
                steps_g1 = [([s] * batch, g0[:, 100 + s * batch : 100 + (s + 1) * batch]) for s in range(3)]
            # ---- prefill (eager, before any capture): reference host loop vs every pipeline
            if batch == 1:
                pipes["wrap/eth_direct"].prepare_prefill((20, 2047)) if "wrap/eth_direct" in pipes else None
                for n in (20, 2047):
                    ref.reset_slot(0)
                    rl = ref.prefill(g0[:, :n])
                    for name, p in pipes.items():
                        p.reset_slot(0)
                        t1 = time.perf_counter()
                        _, units = p.prefill_forward(g0[:, :n], user_slot=0, keep_unit_outputs=True)
                        dt = time.perf_counter() - t1
                        compare_rows(
                            "two_chip_pipeline",
                            name,
                            PLANS_OF[name],
                            f"prefill{n}",
                            units,
                            rl,
                            n,
                            gold={k: v[:, :n] for k, v in gold.items()},
                            extra={"prefill_wall_s": round(dt, 3)},
                        )
                for name, p in pipes.items():  # micro-batch group 1: its own 20-token prompt state
                    if p.groups > 1:
                        ref2.reset_slot(0)
                        rl2 = ref2.prefill(g0[:, :20])
                        p.reset_slot(0, group=1)
                        _, units = p.prefill_forward(g0[:, :20], user_slot=0, group=1, keep_unit_outputs=True)
                        compare_rows(
                            "two_chip_pipeline",
                            name,
                            PLANS_OF[name],
                            "prefill20_group1",
                            units,
                            rl2,
                            20,
                            gold={k: v[:, :20] for k, v in gold.items()},
                        )
            # ---- references: compile every decode program now (before any capture) + pre-capture state snapshots
            ref.set_inputs(*steps[0])
            ref_snap = ref_snapshot(ref)
            ref2.set_inputs(*steps_g1[0])
            ref2_snap = ref_snapshot(ref2)
            for t in ref._decode(ref.x_in):
                ttnn.deallocate(t)
            ttnn.synchronize_device(ref.mesh)
            perf = dict(batch=batch, layers=LAYERS)
            for name, plan, ho, mb in configs:
                p = pipes[name]
                for r, sn in ((ref, ref_snap), (ref2, ref2_snap)):
                    ref_restore(r, sn)
                det_snap = [snapshot_all(p, g) for g in range(mb)]  # pre-capture => trace-safe
                p.warm_and_capture()  # one trace per (stage, group); only this pipeline's traces are live
                for si in range(len(steps)):
                    for g, (st_list, r) in enumerate(((steps, ref), (steps_g1, ref2))[:mb]):
                        pos, x = st_list[si]
                        r.set_inputs(pos, x)
                        p.set_decode_inputs(pos, group=g)
                        p.write_streams(x, group=g)
                        p.decode_step(group=g)
                        units = probe_rows(p, batch, g)  # persistent probes: valid across the other group's replay
                        rl = ref_eager_step(r, batch)
                        gref = None
                        if batch == 1 and si == 0:
                            row = 2047 if g == 0 else 20
                            gref = {k: v[:, row : row + 1] for k, v in gold.items()}
                        compare_rows(
                            "two_chip_pipeline",
                            name,
                            plan,
                            f"decode_B{batch}_step{si}" + ("_group1" if g else ""),
                            units,
                            rl,
                            batch,
                            gold=gref,
                            extra={"in_trace": True},
                        )
                        if batch > 1 and si == 0 and g == 0:  # user 0 at position 0 = prompt token 0: golden row 0
                            for uname, lname in layer_out_names(plan).items():
                                record(
                                    dict(
                                        test=f"two_chip_pipeline/{name}/decode_B{batch}_step0_user0/{lname}",
                                        layer=int(lname[1:]),
                                        layer_kind=KIND[int(lname[1:])],
                                        status="pass",
                                        info_only=True,
                                        pcc_vs_golden=round(pcc(units[uname][:1], gold[lname][:, 0]), 6),
                                    )
                                )
                if batch == 1:  # prefill while this pipeline's decode traces are live, then a traced step on it
                    p.reset_slot(0)
                    _, units = p.prefill_forward(g0[:, :20], user_slot=0, keep_unit_outputs=True)
                    ref.reset_slot(0)
                    rl = ref.prefill(g0[:, :20])
                    compare_rows("two_chip_pipeline", name, plan, "prefill20_after_capture", units, rl, 20)
                    pos, x = [20], g0[:, 20:21]
                    ref.set_inputs(pos, x)
                    p.set_decode_inputs(pos)
                    p.write_streams(x)
                    p.decode_step()
                    compare_rows(
                        "two_chip_pipeline",
                        name,
                        plan,
                        "decode_after_prefill_after_capture",
                        probe_rows(p, batch),
                        ref_eager_step(ref, batch),
                        batch,
                        gold={k: v[:, 20:21] for k, v in gold.items()},
                        extra={"in_trace": True},
                    )
                # determinism: step 0 replayed twice from the same pre-capture state -> identical output
                p.set_decode_inputs(steps[0][0])
                p.write_streams(steps[0][1])
                outs = []
                for _ in range(2):
                    restore_all(p, det_snap[0])
                    outs.append(rows_of(p.decode_step(), batch))
                record(
                    dict(
                        test=f"two_chip_pipeline/{name}/decode_B{batch}_replay_determinism",
                        status="pass" if torch.equal(outs[0], outs[1]) else "fail",
                        bitexact=bool(torch.equal(outs[0], outs[1])),
                        in_trace=True,
                    )
                )
                # latency with the hand-off, from the pre-capture state and step-0 inputs (same as every timing below)
                restore_all(p, det_snap[0])
                pt, _ = median_time(p.decode_step, a.reps)
                perf[f"{name}/pipeline_step_ms"] = round(pt * 1e3, 3)
                p.release()
                # latency without the hand-off: each stage's units alone (no recv/send), same state and inputs
                comp = []
                for st in p.stages:
                    restore_all(p, det_snap[0])
                    g = st.groups[0]
                    tid = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                    try:
                        o = st.decode_units(g, g.streams_in)
                    finally:
                        ttnn.end_trace_capture(st.mesh, tid, cq_id=0)
                    ttnn.deallocate(o)

                    def run(m=st.mesh, t=tid):
                        ttnn.execute_trace(m, t, cq_id=0, blocking=False)
                        ttnn.synchronize_device(m)

                    run()
                    restore_all(p, det_snap[0])
                    ct, _ = median_time(run, a.reps)
                    ttnn.release_trace(st.mesh, tid)
                    comp.append(round(ct * 1e3, 3))
                restore_all(p, det_snap[0])
                perf[f"{name}/stage_compute_only_ms"] = comp
                perf[f"{name}/handoff_overhead_ms"] = round(perf[f"{name}/pipeline_step_ms"] - sum(comp), 3)
                del det_snap
            # single-chip reference trace: same pre-capture state, same step-0 inputs (no pipeline trace live)
            ref_restore(ref, ref_snap)
            ref.set_inputs(*steps[0])
            ref.capture()
            ref.step()
            ref_restore(ref, ref_snap)
            rt, _ = median_time(ref.step, a.reps)
            perf["reference_single_chip_traced_ms"] = round(rt * 1e3, 3)
            perf["timing_protocol"] = "every timed replay starts from the same pre-capture state with step-0 inputs"
            ref.release()
            del ref_snap, ref2_snap
            record(dict(test="two_chip_pipeline_perf", status="pass", **perf), name="pipeline_perf.jsonl")
        if not a.no_reserve_check and "wrap/host" in pipes and batch == 32:
            budget = int(a.reserve_gib * GIB) - a.trace_region
            reserve_check(pipes["wrap/host"], subs, steps, g0, budget, a.trace_region)
        record(
            dict(test="two_chip_pipeline/load", status="pass", load_s=loader.load_s, note="weight load times"),
            name="pipeline_perf.jsonl",
        )
    finally:
        for p in pipes.values():
            p.release()
        for s_ in subs.values():
            ttnn.synchronize_device(s_)
        close_pipeline_mesh(mesh)


PLANS_OF = {c[0]: c[1] for c in CONFIGS}

if __name__ == "__main__":
    main()
