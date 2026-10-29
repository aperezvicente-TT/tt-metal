# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, MOCK; host-side checks of every launch): Stage 14b — the wm6 / wm7 root-cause harness (2026-10-20).
ARGS mode "install": wraps ttnn.generic_op (and records the per-chip programs of every ttnn.MeshProgramDescriptor) for
the rest of the serve process, so the NEXT jobs (the device jobs, unchanged) are checked launch by launch:

  cache: generic_op's program cache keys a launch by the descriptors without the runtime args; two launches with one
      key and other per-core argument counts -> the later one runs with unwritten (stale) arguments;
  cb: every program's circular buffers placed as tt-metal places them (per core, in order, 64-B aligned from the
      allocator base); the fabric lanes' CBs (indices of tt/fabric_ar.py) must sit where FabricPhase.probe() found
      them (a lane-only program), else producers / consumers address other memory;
  args: a stack kernel's argument table (MK_LARGS_RT) must be a live DRAM buffer at the launch.

ARGS mode "report": the findings so far (counts + the first ones) -> RESULT; "uninstall" restores ttnn."""
import json
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

args, ctx = ARGS, CTX  # noqa: F821
mesh = MESH  # noqa: F821
mode = args.get("mode", "install")


def log(*a):
    print(f"PROBE {C.now()}", *a, flush=True)


def cores_of(crs):
    return {(x, y) for r in crs.ranges() for x in range(int(r.start.x), int(r.end.x) + 1)
            for y in range(int(r.start.y), int(r.end.y) + 1)}  # fmt: skip


def place_cbs(prog, base):
    """-> {(core, cb index): (addr, size)} as tt-metal allocates (max region end over the CB's cores, 64-B aligned)"""
    top, out = {}, {}
    for cb in prog.cbs:
        cs = cores_of(cb.core_ranges)
        addr = max([top.get(c, base) for c in cs] + [base])
        addr = -(-addr // 64) * 64
        for c in cs:
            top[c] = addr + int(cb.total_size)
            for f in cb.format_descriptors:
                out[(c, int(f.buffer_index))] = (addr, int(cb.total_size))
    return out, top


def rt_len(kd, x, y) -> int:
    try:
        return len(kd.runtime_args[x][y])
    except IndexError:  # a core of the kernel's range without runtime args
        return 0


def rt_counts(prog):
    return tuple(tuple(sorted((xy, rt_len(kd, *xy)) for xy in cores_of(kd.core_ranges))) for kd in prog.kernels)


os.environ.update({k: str(v) for k, v in (args.get("env") or {}).items()})
if mode == "install":
    if "probe_hooks" in ctx:
        log("already installed")
    else:
        st = dict(orig_op=ttnn.generic_op, orig_mpd=ttnn.MeshProgramDescriptor, keys={}, findings=[], launches=0,
                  lane_ref={}, max_end=0)  # fmt: skip
        base = int(ttnn._ttnn.reports.get_device_info(mesh).address_at_first_l1_cb_buffer)
        st["base"] = base
        st["dump_dir"] = args.get("dump_dir")
        if st["dump_dir"]:
            os.makedirs(st["dump_dir"], exist_ok=True)

        class RecMPD(st["orig_mpd"]):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.rec = []

            def __setitem__(self, k, v):
                self.rec.append(v)
                super().__setitem__(k, v)

        def find(kind, msg):
            if len(st["findings"]) < 400:
                st["findings"].append(dict(kind=kind, launch=st["launches"], msg=msg))
            log("FINDING", kind, msg)

        def check(io, desc):
            st["launches"] += 1
            progs = list(desc.rec) if isinstance(desc, RecMPD) else [desc]
            if not progs or not isinstance(progs[0], ttnn.ProgramDescriptor):
                return
            key = tuple(int(ttnn.compute_program_descriptor_hash(p)) for p in progs)
            cnt = tuple(rt_counts(p) for p in progs)
            old = st["keys"].setdefault(key, cnt)
            if old != cnt:
                srcs = [str(k.kernel_source).split("kernels/")[-1] for k in progs[0].kernels]
                bad = [srcs[i] for i, (a, b) in enumerate(zip(old[0], cnt[0])) if a != b]
                find("cache", f"key reused with other runtime-arg counts: kernels {bad}")
            live = None
            for p in progs[:1]:
                placed, top = place_cbs(p, base)
                st["max_end"] = max([st["max_end"]] + list(top.values()))
                lane_cores = set()  # the fabric lanes: the only SOURCE_CODE kernels (tt/fabric_ar.py)
                for kd in p.kernels:
                    if kd.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
                        lane_cores |= cores_of(kd.core_ranges)
                for cb in p.cbs:
                    cs = cores_of(cb.core_ranges)
                    if cs & lane_cores and not cs <= lane_cores:
                        find("cb", f"program CB {[int(f.buffer_index) for f in cb.format_descriptors]} also covers "
                                   f"lane cores {sorted(cs & lane_cores)}")  # fmt: skip
                for c in sorted(lane_cores):
                    a = placed.get((c, 19))
                    if a is None or a[0] != base:
                        find("cb", f"lane core {c}: CB OWN at {a}, probe() expects {base:#x}")
                for kd in p.kernels:
                    d = dict(kd.defines)
                    if "MK_LARGS_RT" not in d:
                        continue
                    if live is None:
                        live = {int(b.address) for b in ttnn._ttnn.reports.get_buffers(mesh)
                                if b.buffer_type == ttnn.BufferType.DRAM}  # fmt: skip
                    i = int(d["MK_LARGS_RT"])
                    tabs = {int(kd.runtime_args[x][y][i]) for x, y in cores_of(kd.core_ranges) if rt_len(kd, x, y) > i}
                    if not tabs <= live:
                        find("args", f"{str(kd.kernel_source).split('kernels/')[-1]}: argument table "
                                     f"{sorted(hex(v) for v in tabs - live)} is not a live DRAM buffer")  # fmt: skip

        def dump(desc):
            """the structure of every new stack program (MK_LAYERS) -> dump_dir/<launch>.json (first of each key)"""
            progs = list(desc.rec) if isinstance(desc, RecMPD) else [desc]
            if not st.get("dump_dir") or not progs or not isinstance(progs[0], ttnn.ProgramDescriptor):
                return
            p0 = progs[0]
            if not any("MK_LAYERS" in dict(kd.defines) for kd in p0.kernels):
                return
            key = int(ttnn.compute_program_descriptor_hash(p0))
            if key in st.setdefault("dumped", set()):
                return
            st["dumped"].add(key)
            ks = []
            for kd in p0.kernels:
                cs = sorted(cores_of(kd.core_ranges))
                ks.append(dict(src=str(kd.kernel_source).split("kernels/")[-1][:80], ct=[int(a) for a in kd.compile_time_args],
                               defines=sorted((str(a), str(b)) for a, b in kd.defines), cores=len(cs),
                               rt_lens=sorted({rt_len(kd, *xy) for xy in cs}),
                               rt0=[int(a) for a in (kd.runtime_args[cs[0][0]][cs[0][1]] if rt_len(kd, *cs[0]) else [])]))  # fmt: skip
            cbs = [dict(idx=[int(f.buffer_index) for f in cb.format_descriptors], size=int(cb.total_size),
                        cores=len(cores_of(cb.core_ranges)),
                        ranges=[[int(r.start.x), int(r.start.y), int(r.end.x), int(r.end.y)] for r in cb.core_ranges.ranges()])
                   for cb in p0.cbs]  # fmt: skip
            placed, top = place_cbs(p0, base)
            core00 = sorted((i, a, n) for (c, i), (a, n) in placed.items() if c == (0, 0))
            l1_low = min((int(b.address) for b in ttnn._ttnn.reports.get_buffers(mesh)
                          if b.buffer_type == ttnn.BufferType.L1), default=None)  # fmt: skip
            path = os.path.join(st["dump_dir"], f"{st['launches']:06d}_{key & 0xffffffff:08x}.json")
            json.dump(dict(launch=st["launches"], key=key, kernels=ks, cbs=cbs, n_io=len(io_last), core00=core00,
                           core00_end=top.get((0, 0)), max_end=max(top.values()), l1_lowest=l1_low,
                           cb30_cores=sorted(c for (c, i) in placed if i == 30)), open(path, "w"))  # fmt: skip

        io_last = []

        def op(io, desc):
            io_last[:] = list(io)
            try:
                dump(desc)
            except Exception as e:  # noqa: BLE001 (diagnostic harness)
                find("harness", "dump " + repr(e)[:300])
            try:
                check(io, desc)
            except Exception as e:  # noqa: BLE001 (diagnostic harness)
                find("harness", repr(e)[:300])
            return st["orig_op"](io, desc)

        ttnn.generic_op = op
        ttnn.MeshProgramDescriptor = RecMPD
        ctx["probe_hooks"] = st
        log(f"installed (CB base {base:#x})")
    RESULT = dict(ok=True)
elif mode == "report":
    st = ctx.get("probe_hooks", {})
    fs = st.get("findings", [])
    kinds = {}
    for f in fs:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    log(
        "REPORT launches",
        st.get("launches"),
        "keys",
        len(st.get("keys", {})),
        "findings",
        kinds,
        f"max CB end {st.get('max_end', 0):#x}",
    )
    for f in fs[:40]:
        log("  ", f)
    RESULT = dict(ok=not fs, kinds=kinds, launches=st.get("launches"), first=fs[:10])
else:
    st = ctx.pop("probe_hooks", None)
    if st:
        ttnn.generic_op, ttnn.MeshProgramDescriptor = st["orig_op"], st["orig_mpd"]
    RESULT = dict(ok=True)
