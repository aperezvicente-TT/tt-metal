# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 MOCK): the PMK pass's DRAM budget per bank at TP8 (program 2026-10-23 21:38:15Z: the 1M
PMK construction OOMs at the (2048, 0) compile key; target <= 300 MB per bank for the whole PMK pass). Reads the DRAM
memory view (lock-step: chip 0 = every chip) at:
  build   after S.build (the op-path construction at the served windows: pools, KV, trace region),
  attach  after PmkStack.attach (the PMK programs' persistent buffers: A scratch / hand-off stores, B scratch + tables),
  forward the first eager PMK forward at the (2048, 0) key under tt/mock_dram.watch (GLM53_MOCK_DRAM_PROBE=1: the
          peak allocated and the smallest largest-free block per bank, and the op it followed).
Run it at two layer counts: the slope is the per-layer PMK cost, the intercept the shared part.
ARGS: layers ([0..4]), rows (2048), windows (the 1M served list), b (2), tag."""
import gc
import os
import sys
import time
import traceback

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import mock_dram as MD  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = int(args.get("rows", 2048))
WINDOWS = str(args.get("windows", "65536,131072,262144,524288,1048576"))
B_BUILD = int(args.get("b", 2))
tag = args.get("tag", f"pmk-dram-budget-L{LAYERS[0]}-{LAYERS[-1]}")
import pmk45_alias as A45  # noqa: E402

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PF_SPARSE_T"] = WINDOWS
os.environ["PMK_L1_CHECK"] = "1"
os.environ["GLM53_MOCK_DRAM_PROBE"] = "1"
os.environ["GLM53_MOCK_DRAM_PROBE_MIN_ELEMS"] = str(args.get("probe_min_elems", 1 << 20))
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
MiB = 2**20
row, views = {}, {}


def view(name):
    gc.collect()
    lf, al, banks = MD._view(mesh)
    views[name] = dict(alloc_mib=round(al / MiB, 1), largest_free_mib=round(lf / MiB, 1), banks=banks)
    H.log(f"{tag}: DRAM {name}: {views[name]}")


try:
    view("open")
    max_seq = max(int(w) for w in WINDOWS.split(",")) + N + 512
    ent = S.build(mesh, ctx, LAYERS, b=B_BUILD, buckets=(N,), max_seq=max_seq)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    A45.publish_page_table(backend, kv, io, pf, rows=B_BUILD)
    c = pf.bucket_for(N)
    view("build")
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
    opts.update({k: v for k, v in args.items() if k.startswith("b_")})
    os.environ["GLM53_PF_MODE"] = "pmk"
    t0 = time.time()
    # every DRAM allocation during the PMK construction + attach, by the ServedLayer under construction and its call site (the slope)
    import collections  # noqa: PLC0415
    import traceback as _tb  # noqa: PLC0415

    alloc_log = collections.defaultdict(lambda: collections.Counter())
    cur = {"L": "pre"}
    _SL = PS.ServedLayer
    _orig_init = _SL.__init__

    def _init(self, stack_, li, *a_, **k_):
        cur["L"] = f"L{getattr(stack_.pf.layers[li], 'index', li)}/{li}"
        lf0, al0, _ = MD._view(mesh)
        try:
            return _orig_init(self, stack_, li, *a_, **k_)
        finally:
            lf1, al1, _ = MD._view(mesh)
            alloc_log[cur["L"]]["__delta_mib_per_bank"] += round((al1 - al0) / MiB, 2)
            cur["L"] = "post"

    _SL.__init__ = _init
    _fns = {}
    for nm in ("empty", "zeros", "from_torch", "allocate_tensor_on_device", "ones", "full"):
        f = getattr(ttnn, nm, None)
        if f is None:
            continue

        def mk(f, nm):
            def w(*a_, **k_):
                al0 = MD._view(mesh)[1]
                out = f(*a_, **k_)
                d = MD._view(mesh)[1] - al0
                if d > 0:
                    fr = [x for x in _tb.extract_stack(limit=8)[:-1] if "zai_org_glm_5_3_flash/tt/" in x.filename]
                    site = f"{os.path.basename(fr[-1].filename)}:{fr[-1].lineno}" if fr else nm
                    alloc_log[cur["L"]][f"{nm}@{site}"] += round(d / MiB, 3)
                return out

            return w

        _fns[nm] = f
        setattr(ttnn, nm, mk(f, nm))
    try:
        stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
        stack.attach()
    finally:
        _SL.__init__ = _orig_init
        for nm, f in _fns.items():
            setattr(ttnn, nm, f)
    view("attach")
    for k_, cnt in alloc_log.items():
        H.log(f"PMKALLOC {tag} {k_}: delta {cnt.get('__delta_mib_per_bank', 0)} MiB/bank; top " +
              str([(s_, round(v, 2)) for s_, v in cnt.most_common(8) if not s_.startswith('__')]))  # fmt: skip

    def dram_tensors(o, depth=0, seen=None, path=""):
        """(buffer address, bytes, path) of every DRAM ttnn.Tensor reachable from o (attributes / dicts / lists, depth 4)"""
        seen = set() if seen is None else seen
        out = []
        if id(o) in seen or depth > 4:
            return out
        seen.add(id(o))
        if isinstance(o, ttnn.Tensor):
            try:
                if o.storage_type() == ttnn.StorageType.DEVICE and o.memory_config().buffer_type == ttnn.BufferType.DRAM:
                    nb = int(o.volume()) * int(o.element_size()) if callable(getattr(o, "volume", None)) else 0
                    out.append((int(o.buffer_address()), nb, path))
            except Exception:  # noqa: BLE001
                pass
            return out
        if isinstance(o, dict):
            items = list(o.items())
        elif isinstance(o, (list, tuple)):
            items = list(enumerate(o))
        elif hasattr(o, "__dict__") and type(o).__module__.startswith("models."):
            # not the back-references (the stack / prefill / backend / the group owner hold every layer and the weights)
            items = [(k, v) for k, v in vars(o).items() if k not in ("stack", "pf", "backend", "reuse", "owner", "tpp", "layer", "pl", "kda", "mesh", "device")]
        else:
            return out
        for k, v in items:
            out += dram_tensors(v, depth + 1, seen, f"{path}.{k}")
        return out

    owners, per_layer = {}, {}
    for (c_, L_), sl in sorted(stack.progs.items(), key=lambda kv: kv[0][1]):
        ts = dram_tensors(sl, path=f"L{L_}")
        uniq = {}
        for addr, nb, pth in ts:
            uniq.setdefault(addr, (nb, pth))
        mine = {a_: v for a_, v in uniq.items() if a_ not in owners}
        for a_ in uniq:
            owners.setdefault(a_, L_)
        top = sorted(mine.values(), key=lambda v: -v[0])[:6]
        per_layer[f"L{L_}"] = dict(kind=getattr(sl, "kind", "?"), new_mib_per_chip=round(sum(v[0] for v in mine.values()) / MiB, 1),
                                   reached_mib_per_chip=round(sum(v[0] for v in uniq.values()) / MiB, 1),
                                   top=[[round(v[0] / MiB, 2), v[1][:90]] for v in top])  # fmt: skip
    for k_, v_ in per_layer.items():
        H.log(f"PMKOWN {tag} {k_} {v_['kind']}: new {v_['new_mib_per_chip']} MiB/chip (reached {v_['reached_mib_per_chip']}); top {v_['top'][:4]}")
    toks = S.golden_prompt(32768).reshape(-1).long()[:N]
    pf.write_inputs(c, toks, slot=0, chunk_start=0, n_valid=N)
    with MD.watch(mesh, f"{tag} pmk forward (2048, 0)") as st:
        pf.forward(c)
    view("after_forward")
    fw = dict(st or {})
    b, a = views["build"]["alloc_mib"], views["attach"]["alloc_mib"]
    peak = round(fw.get("max_alloc_per_bank", 0) / MiB, 1)
    row = dict(test=f"PMK pass DRAM budget per bank, L{LAYERS[0]}-L{LAYERS[-1]}, 2x4 mock", status="mock", layers=LAYERS,
               n_layers=len(LAYERS), windows=WINDOWS, views=views, attach_mib=round(a - b, 1),
               forward_transient_mib=round(peak - a, 1) if peak else None, pmk_total_mib=round(peak - b, 1) if peak else None,
               forward_min_largest_free_mib=fw.get("min_largest_free_mib_per_bank"), forward_min_at_op=fw.get("min_at_op"),
               forward_peak_alloc_mib=peak, forward_ops=fw.get("ops"), per_layer_owned=per_layer, wall_s=round(time.time() - t0, 1),
               tag=tag)  # fmt: skip
    H.log(f"PMKDRAM {tag}: build {b} MiB/bank; PMK attach +{row['attach_mib']}; forward transients +{row['forward_transient_mib']} "
          f"(peak {peak}, min largest free {row['forward_min_largest_free_mib']} at {row['forward_min_at_op']})")  # fmt: skip
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    row = dict(status="fail", error=f"{type(e).__name__}: {e}"[:1500], views=views, tag=tag)
RESULT = row
