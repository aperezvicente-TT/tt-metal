# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the host-8chip 2x4 mesh, also the 2x4 mock): the RING-WORD PROBE of the TP8 composite hang (program
2026-10-23 20:36:41Z item 1; 14's lane-send audit doc/ttnn/14-production-tp8-megakernel/lane_send_audit.md). The served
PMK chain's FIRST eager forward (the pass that timed out in a fresh process, 14 M5 20:46:08Z) with a probe point
BEFORE every program launch -- ttnn.generic_op (the PMK programs, state_io), ttnn.all_gather / ttnn.reduce_scatter (the
op CCLs) -- up to the first `ccl_points` op CCL launches. At each point: ttnn.synchronize_device, then a raw L1
snapshot of every worker core of every chip (read_core_l1, UMD reads: no dispatch):
  mailbox  [96, +13440)          launch-message ring + rd ptr, go messages + index, watcher / dprint / core info
  fabconn  [36976, +4112)        the tensix fabric connection info + the packet-header pool
  ring     [41088, +70656)       the kernel-config ring (rt args, semaphores incl. the reserved fabric ids, CB configs)
and the core's current kernel-config slot (capture_kernel_config: base, size). Diff vs the previous point: ring words
changed OUTSIDE the slot of the program that just ran are FOREIGN (a write into another program's words); mailbox /
fabconn offsets are histogrammed per chip (an offset that changed on only some cores is listed). If a sync times out
(the hang), a final snapshot is taken WITHOUT the sync (UMD reads still work): the stuck state's go messages and
launch pointers next to the last clean point. Eager only inside the one compile pass the hang lives in (no extra
program runs; the probe adds synchronizes and host reads). On the mock: the hooks and the point labels only (reads
skipped).
FORMS (program 20:48:07Z): "launch" = A1..A5 of L0 via ServedLayer._launch (16's (d) form) then pf.gather on a
zeros x (the (d) gather); "forward" = the served pf.forward (M5's hanging form); "both" = launch, quiesce, forward.
Also read at every point: every FabricStream* object's sync words on its lane cores (the lanes' CREDIT / ARR / GEN
words, region "sync<k>"), and at P0 the decoded fabric connection table of every core (hostdevcommon fabric_common.h
tensix_fabric_connections_l1_info_t: per eth channel edm_noc_x / y + worker_free_slots_stream_id = the (x, y, stream)
of every per-packet stream-register credit write, + the valid mask).
ARGS: layers ([0..4]), rows (2048), windows (GLM53_PF_SPARSE_T), form ("both"), ccl_points (3), max_points (24),
regions ("mailbox,fabconn,ring"), probe_layers (=[layers[-1]]), b (2), tag, record (True)."""
import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = int(args.get("rows", 2048))
WINDOWS = str(args.get("windows", "65536"))
CCL_POINTS = int(args.get("ccl_points", 3))
FORM = str(args.get("form", "both"))
assert FORM in ("launch", "forward", "both"), FORM
MAX_POINTS = int(args.get("max_points", 24))
REGIONS_ON = [r for r in str(args.get("regions", "mailbox,fabconn,ring")).split(",") if r]
READ_BUDGET_S = float(args.get("read_budget_s", 30))
RUN_GAP = int(args.get("run_gap_words", 64))  # changed words closer than this join one run
BULK_W = int(args.get("bulk_words", 64))  # a run this long = a dispatcher config write
LIVE_SPAN = int(args.get("live_span", 16384))  # bytes from the live kernel_config_base counted as the live slot  # one point's reads over this: later points skip the ring region
B_BUILD = int(args.get("b", 2))
tag = args.get("tag", f"pmk-ring-probe-L{LAYERS[0]}-{LAYERS[-1]}-tp{TPN}")
rec = bool(args.get("record", True))
# Blackhole worker L1 (tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h, evaluated at s14h7d 8f28c0fd0a8):
# MEM_MAILBOX_BASE 96 / SIZE 13440; MEM_TENSIX_FABRIC_CONNECTIONS_BASE 36976 (656) + the packet-header pool 37632
# (3456); MEM_MAP_END 41088 = the kernel-config ring base, 69 KiB (bh_hal_tensix.cpp default_l1_kernel_config_size)
REGIONS = dict(mailbox=(96, 13440), fabconn=(36976, 656 + 3456), ring=(41088, 69 * 1024))
import pmk45_alias as A45  # noqa: E402

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(v) for v in args.get("probe_layers", [LAYERS[-1]]))
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PF_SPARSE_T"] = WINDOWS
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
t0 = time.time()
info, row, points = {}, {}, []
EV = os.path.join(os.environ.get("GLM53_OUT_DIR", "."), "ring_probe")
os.makedirs(EV, exist_ok=True)

try:
    from ttnn._ttnn.multi_device.experimental import capture_kernel_config, read_core_l1
except Exception as e:  # noqa: BLE001
    read_core_l1 = capture_kernel_config = None
    info["reader_error"] = f"{type(e).__name__}: {e}"


class Snapper:
    """raw L1 snapshots of every worker core of every chip + the diff against the previous point"""

    def __init__(self):
        g = mesh.compute_with_storage_grid_size()
        self.cores = [(x, y) for y in range(int(g.y)) for x in range(int(g.x))]
        R, Cc = int(mesh.shape[0]), int(mesh.shape[1])
        self.coords = [(r, c) for r in range(R) for c in range(Cc)]
        self.prev, self.prev_cfg = None, None
        self.read = (not C.MOCK) and read_core_l1 is not None
        self.syncs = []  # (name, lane logical cores, L1 address, bytes): the FabricStream sync tensors

    def find_syncs(self):
        """every live FabricStream* object with a sync tensor on its lanes (C1X, RS8's stages, RSL, ...)"""
        seen = set()
        for o in gc.get_objects():
            try:
                if not type(o).__name__.startswith("FabricStream") or id(o) in seen:
                    continue
                t, lanes = getattr(o, "sync", None), getattr(o, "lanes", None)
                if t is None or not lanes:
                    continue
                seen.add(id(o))
                try:
                    addr = int(t.buffer_address())
                except Exception:  # noqa: BLE001
                    addr = int(ttnn.get_device_tensors(t)[0].buffer_address())
                nbytes = int(t.shape[-1]) * 4
                self.syncs.append((f"sync{len(self.syncs)}:{type(o).__name__}", [tuple(int(v) for v in l_) for l_ in lanes],
                                   addr, nbytes))  # fmt: skip
            except Exception:  # noqa: BLE001
                continue
        return [dict(name=n, lanes=l_, addr=a, bytes=b) for n, l_, a, b in self.syncs]

    def conn_tables(self) -> dict:
        """P0: the decoded fabric connection table per chip (distinct tables over the chip's cores)"""
        if not self.read or "fabconn" not in REGIONS_ON or not self.prev:
            return {}
        out = {}
        for (r, c) in self.coords:
            tabs = {}
            for (x, y) in self.cores:
                w = self.prev.get((r, c, x, y, "fabconn"))
                if w is None:
                    continue
                b = w[: 656 // 4].astype("<u4").tobytes()
                mask = int.from_bytes(b[384:388], "little")
                ents = []
                for e in range(16):
                    q = b[24 * e : 24 * e + 24]
                    ents.append([e, (mask >> e) & 1, q[18], q[19], q[20], q[21], int.from_bytes(q[22:24], "little"),
                                 int.from_bytes(q[12:16], "little")])  # fmt: skip
                key = json.dumps([v for v in ents if v[1] or any(v[2:7])])
                tabs.setdefault(key, []).append([x, y])
            out[f"{r},{c}"] = [dict(entries=json.loads(k), cols="[eth_ch, valid, dir, x, y, nbuf, stream, bufidx_sem]",
                                    n_cores=len(v), cores=v[:4]) for k, v in tabs.items()]  # fmt: skip
        return out

    def take(self):
        snap, cfg = {}, {}
        if not self.read:
            return snap, cfg
        for (r, c) in self.coords:
            mc = ttnn.MeshCoordinate(r, c)
            for name, lanes, addr, nbytes in self.syncs:
                for (x, y) in lanes:
                    snap[(r, c, x, y, name)] = np.asarray(read_core_l1(mesh, ttnn.CoreCoord(x, y), addr, nbytes, mc), dtype=np.uint32)
            for (x, y) in self.cores:
                cc = ttnn.CoreCoord(x, y)
                for name in REGIONS_ON:
                    base, size = REGIONS[name]
                    snap[(r, c, x, y, name)] = np.asarray(read_core_l1(mesh, cc, base, size, mc), dtype=np.uint32)
                try:
                    k = capture_kernel_config(mesh, cc, mc)
                    cfg[(r, c, x, y)] = (int(k["kernel_config_base"]), int(k["kernel_config_size"]))
                except Exception:  # noqa: BLE001
                    cfg[(r, c, x, y)] = (-1, 0)
        return snap, cfg

    def diff(self, snap, cfg) -> dict:
        out = dict(changed_words={}, foreign=[], partial_offsets={})
        if not snap or self.prev is None:
            return out
        per_off = {}  # (region, chip, offset) -> number of cores changed
        for key, cur in snap.items():
            r, c, x, y, name = key
            old = self.prev.get(key)
            if old is None:
                continue
            idx = np.nonzero(old != cur)[0]
            if not len(idx):
                continue
            out["changed_words"][name] = out["changed_words"].get(name, 0) + int(len(idx))
            base = REGIONS[name][0] if name in REGIONS else 0
            if name == "ring":
                # the dispatcher's config write of the program(s) that ran = long runs of changed words; a stray / late
                # +1 = a SHORT isolated run outside every long run and outside the live slot (the core's latest launch
                # message: mailbox launch[(rd_ptr - 1) % 8] word 0 = kernel_config_base; program 21:03:12Z offsets
                # rd_ptr 0x6C, launch slots 0x70 + 0x90 i)
                live = self.live_base(snap, r, c, x, y)
                runs, st_, pv_ = [], int(idx[0]), int(idx[0])
                for i in idx[1:].tolist():
                    if i - pv_ <= RUN_GAP:
                        pv_ = i
                    else:
                        runs.append((st_, pv_))
                        st_ = pv_ = i
                runs.append((st_, pv_))
                bulk = [(u, v) for u, v in runs if v - u + 1 >= BULK_W]
                for u, v in runs:
                    if v - u + 1 >= BULK_W:
                        continue
                    a0 = base + 4 * u
                    near_bulk = any(bu - RUN_GAP <= u and v <= bv + RUN_GAP for bu, bv in bulk)
                    in_live = live is not None and live <= a0 < live + LIVE_SPAN
                    if near_bulk or in_live:
                        continue
                    for i in range(u, v + 1):
                        if old[i] != cur[i] and len(out["foreign"]) < 300:
                            out["foreign"].append(dict(chip=[r, c], core=[x, y], addr=base + 4 * i, old=int(old[i]),
                                                       new=int(cur[i]), delta=int(cur[i]) - int(old[i]), live=live,
                                                       bulk=[[base + 4 * bu, 4 * (bv - bu + 1)] for bu, bv in bulk][:4]))  # fmt: skip
            else:
                for i in idx.tolist():
                    k_ = (name, r, c, base + 4 * i)
                    per_off[k_] = per_off.get(k_, 0) + 1
        ncores = len(self.cores)
        part = {}
        for (name, r, c, a), n in per_off.items():
            if n < ncores:
                part.setdefault(f"{name}@{r},{c}", []).append([a, n])
        out["partial_offsets"] = {k: sorted(v)[:64] for k, v in part.items()}
        out["n_foreign"] = len(out["foreign"])
        return out

    @staticmethod
    def live_base(snap, r, c, x, y):
        """the core's latest launch's kernel_config_base from the mailbox snapshot (None if not read)"""
        mb = snap.get((r, c, x, y, "mailbox"))
        if mb is None:
            return None
        rd = int(mb[(0x6C - 96) // 4]) & 0xFF
        slot = (rd - 1) % 8
        return int(mb[(0x70 + 0x90 * slot - 96) // 4])

    def point(self, label: str, sync: bool = True) -> dict:
        t_ = time.time()
        err = None
        if sync:
            ttnn.synchronize_device(mesh)
        try:
            snap, cfg = self.take()
        except Exception as e:  # noqa: BLE001
            snap, cfg, err = {}, {}, f"{type(e).__name__}: {e}"[:300]
        d = self.diff(snap, cfg)
        p = dict(i=len(points), label=label, sync=sync, read_s=round(time.time() - t_, 2), err=err, **d)
        if p["read_s"] > READ_BUDGET_S and "ring" in REGIONS_ON:  # the mesh cap: drop the 69 KiB ring from later points
            REGIONS_ON.remove("ring")
            p["dropped_ring_after"] = True
            if self.prev:
                self.prev = {k: v for k, v in self.prev.items() if k[4] != "ring"}
        if snap:
            self.prev, self.prev_cfg = snap, cfg
        points.append(p)
        H.log(f"{tag}: point {p['i']} {label}: read {p['read_s']} s, changed {d['changed_words']}, foreign "
              f"{d.get('n_foreign', 0)}, partial {sum(len(v) for v in d['partial_offsets'].values())}")  # fmt: skip
        return p

    def dump_mailboxes(self, name: str):
        if not self.prev:
            return
        mb = {f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in self.prev.items() if k[4] == "mailbox"}
        np.savez_compressed(os.path.join(EV, f"{tag}_{name}.npz"), **mb)


try:
    max_seq = max(int(w) for w in WINDOWS.split(",")) + N + 512
    ent = S.build(mesh, ctx, LAYERS, b=B_BUILD, buckets=(N,), max_seq=max_seq)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    info["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=B_BUILD)
    c = pf.bucket_for(N)
    g32 = S.golden_prompt(32768).reshape(-1).long()
    toks = g32[:N]
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
    opts.update({k: v for k, v in args.items() if k.startswith("b_")})
    os.environ["GLM53_PF_MODE"] = "pmk"
    stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    stack.attach()
    sn = Snapper()
    info["cores_per_chip"], info["chips"] = len(sn.cores), len(sn.coords)
    info["syncs"] = sn.find_syncs()
    pf.write_inputs(c, toks, slot=0, chunk_start=0, n_valid=N)

    def quiesce():
        if not C.MOCK and os.environ.get("GLM53_PF_QUIESCE", "1") == "1":
            mesh.quiesce_devices()

    quiesce()
    sn.point(f"P0 after the quiesce (form {FORM})")
    info["conn_tables"] = sn.conn_tables()

    state = dict(armed=True, n=0, ccl=0)

    def wrap(name, fn):
        def w(*a, **k):
            if state["armed"]:
                state["n"] += 1
                desc = name
                if name == "generic_op" and len(a) > 1:
                    try:
                        ks = [os.path.basename(str(kd.kernel_source)) for kd in a[1].kernels][:4]
                        desc = f"generic_op[{','.join(ks)}]"
                    except Exception:  # noqa: BLE001
                        pass
                sn.point(f"before launch {state['n']}: {desc}")
                if name != "generic_op":
                    state["ccl"] += 1
                if state["ccl"] > CCL_POINTS or len(points) >= MAX_POINTS:
                    state["armed"] = False
            return fn(*a, **k)

        return w

    orig = dict(generic_op=ttnn.generic_op, all_gather=ttnn.all_gather, reduce_scatter=ttnn.reduce_scatter)
    for k_, f_ in orig.items():
        setattr(ttnn, k_, wrap(k_, f_))
    hang = None
    sp = pf.mode(c) == "sp"
    try:
        if FORM in ("launch", "both"):  # 16's (d) form: A1..A5 of L0 via lay._launch, then pf.gather on zeros
            ci = pf.chunk_inputs(c)
            lay = stack.progs[(c, LAYERS[0])]
            assert lay.a is not None and lay.a.attn, (LAYERS[0], lay.kind)
            lay.prepare(stack.res[c], ci)
            idx = sorted({lay.a.index_of(p_) for p_ in ("P1", "P2", "P4", "P7", "P10")})
            xz = ttnn.from_torch(torch.zeros(1, 1, N // TPN, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
            for i_ in idx:
                lay._launch(i_)
            yz = pf.gather(xz, sp)
            ttnn.synchronize_device(mesh)
            sn.point("after the launch form (A1..A5 + the gather, synchronized)", sync=False)
            ttnn.deallocate(yz)
            ttnn.deallocate(xz)
        if FORM in ("forward", "both"):
            if FORM == "both":
                pf.write_inputs(c, toks, slot=0, chunk_start=0, n_valid=N)
                quiesce()
                state.update(armed=True, ccl=0)
                sn.point("P0b after the second quiesce (the forward form)")
            pf.forward(c)
            ttnn.synchronize_device(mesh)
            sn.point("after the forward (synchronized)", sync=False)
    except Exception as e:  # noqa: BLE001
        hang = f"{type(e).__name__}: {e}"[:600]
        traceback.print_exc()
        sn.point("AFTER THE TIMEOUT (no sync: the stuck state)", sync=False)
        sn.dump_mailboxes("stuck_mailboxes")
    finally:
        for k_, f_ in orig.items():
            setattr(ttnn, k_, f_)
    status = "mock" if C.MOCK else ("hang" if hang else "pass")
    n_for = sum(p.get("n_foreign", 0) for p in points)
    row = dict(test=f"PMK ring-word probe: the served chain's first eager forward L{LAYERS[0]}-L{LAYERS[-1]}, tp{TPN}",
               milestone="PMK-RING", status=status, tp=TPN, layers=LAYERS, bucket=c, hang=hang, n_points=len(points),
               foreign_total=n_for, regions=REGIONS_ON, form=FORM, info={k: v for k, v in info.items() if k != "page_table"},
               points=[{k: v for k, v in p.items() if k != "foreign"} | {"foreign": p.get("foreign", [])[:20]} for p in points],
               wall_s=round(time.time() - t0, 1), tag=tag)  # fmt: skip
    with open(os.path.join(EV, f"{tag}_points.json"), "w") as f:
        json.dump(points, f)
    H.log(f"{tag}: status {status}, {len(points)} points, foreign words {n_for}, hang {hang}")
    if rec and not C.MOCK:
        C.record("pcc", {k: v for k, v in row.items() if k != "points"} | {"points_file": os.path.join(EV, f"{tag}_points.json")})
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    row = dict(test="PMK ring-word probe", milestone="PMK-RING", status="fail", error=f"{type(e).__name__}: {e}"[:2000],
               tp=TPN, layers=LAYERS, tag=tag, n_points=len(points), info={k: v for k, v in info.items() if k != "page_table"})  # fmt: skip
    if rec and not C.MOCK:
        C.record("pcc", row)
RESULT = row
