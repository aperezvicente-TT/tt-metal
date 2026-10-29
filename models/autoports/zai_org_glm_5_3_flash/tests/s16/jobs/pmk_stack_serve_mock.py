# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 HOST MOCK or pair mock; also the compile step on a device): the SERVED PMK STACK under
GLM53_PF_MODE=pmk (tt/pmk_stack.ServedLayer: per layer A1 -> A2 + A3 -> A4 -> A5 -> program B -> mhc post, the slots
no piece delivers as op fillers inside the same trace) on the real TPPrefill of layers L0-L4 (the three kinds): every
piece program of every PMK layer JIT-builds and launches on the mock (the compile launch THROUGH TPPrefill.forward =
embed -> the chain on the persistent residual -> op layers -> head), the bucket key is recorded and replayed (the mock
has no trace), then the op-graph forward of the same TPPrefill after detach() (GLM53_PF_MODE=op: nothing installed).
Reported: per-layer modes / pending slots, the launch census per scoped program (A1..A5) and program B's ledger, the
DRAM per bank (ttnn.get_memory_view: allocated / free / largest, as Stage 9's DRAM probe records) after the backend
build, after each layer's chain build, after each key's compile forward (retained) and -- with census_steps > 0 --
the transient need by ballast bisection (p3_dram_census's method), plus the 45-layer projection
(served_stack_dram_bytes). ARGS: layers ([0..4]), rows (2048), buckets ([512, 2048]), slots ("auto"), n (0 = all),
census_steps (0), env ({}), b_g (2), tag, record (True)."""
import gc
import os
import sys
import time
import traceback

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
BUCKETS = tuple(int(v) for v in args.get("buckets", [512, 2048]))
NPMK = int(args.get("n", 0)) or len(LAYERS)
STEPS = int(args.get("census_steps", 0))
tag = args.get("tag", f"pmk-stack-serve-mock-L{LAYERS[0]}-{LAYERS[-1]}-tp{TPN}")
rec = bool(args.get("record", True))
MB = 1 << 20
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PF_SPARSE_T"] = ",".join(str(int(v)) for v in args.get("sparse_t", []))
os.environ["PMK_L1_CHECK"] = "0" if bool(args.get("l1_relax", False)) else "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}


def view():
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)  # lock-step allocation: chip 0 = every chip
    return dict(banks=int(mv.num_banks), alloc_mb=round(int(mv.total_bytes_allocated_per_bank) / MB, 2),
                free_mb=round(int(mv.total_bytes_free_per_bank) / MB, 2),
                largest_mb=round(int(mv.largest_contiguous_bytes_free_per_bank) / MB, 2))  # fmt: skip


def run_key(pf, c, t):
    if t and t != c:
        pf.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=t - c, n_valid=c)
    else:
        pf.write_inputs(c, torch.zeros(c, dtype=torch.int64), slot=0, chunk_start=0, n_valid=c)
    pf.forward(c, head=True, t=t)
    ttnn.synchronize_device(mesh)
    gc.collect()


def need_by_ballast(pf, c, t, steps):
    """the transient DRAM need of key (c, t) by ballast bisection (p3_dram_census): free - the largest ballast that
    still lets the forward complete, MB per bank"""
    v = view()
    free_b = int(v["free_mb"] * MB)
    lo, hi = 0, free_b

    def fits(bal_b):
        bal = None
        try:
            if bal_b > 0:
                rows = max(32, bal_b * v["banks"] // (1024 * 2) // 32 * 32)
                bal = ttnn.empty((1, 1, rows, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
            run_key(pf, c, t)
            return True
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower():
                raise
            return False
        finally:
            if bal is not None:
                ttnn.deallocate(bal)
            gc.collect()

    for _ in range(steps):
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return round((free_b - lo) / MB, 2)


t0 = time.time()
err = None
info = dict(dram={}, keys={}, layers={})
summ = {}
try:
    ent = S.build(mesh, ctx, LAYERS, b=1, buckets=BUCKETS, max_seq=N + 512)
    pf, backend = ent["pf"], ent["backend"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    info["dram"]["after_backend"] = view()
    info["build_s"] = ent["build_s"]
    H.log(f"{tag}: backend built in {ent['build_s']} s; DRAM {info['dram']['after_backend']}")
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")))
    opts.update({k: v for k, v in args.items() if k.startswith("b_")})  # program B: the record (b_config) + these overrides
    # the chain, layer by layer, measuring the DRAM each layer's programs retain (the per-kind numbers of the 45-layer
    # projection): PmkStack builds every layer at once, so the per-layer deltas come from a stepwise build
    t1 = time.time()
    os.environ["GLM53_PF_MODE"] = "pmk"
    stack = PS.PmkStack(pf, 0, impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    c = pf.bucket_for(N)
    assert c in stack.pmk_buckets, (c, stack.pmk_buckets, [pf.mode(b) for b in BUCKETS])
    sp = pf.mode(c) == "sp"
    res = stack._residual(c)
    v_prev = view()
    for li in range(min(NPMK, len(pf.layers))):
        L_ = pf.layers[li].L
        tl = time.time()
        grp = "kda" if pf.layers[li].kind in PS.PMK_KINDS else pf.layers[li].kind
        lay = PS.ServedLayer(stack, li, c, sp, stack.slots, reuse=stack.shared.get((c, grp)))
        stack.shared.setdefault((c, grp), lay)
        stack.progs[(c, L_)] = lay
        stack.reports[(c, L_)] = [dict(piece="served-chain", modes=lay.modes, pending=lay.pending)]
        stack.layers = stack.layers if li < len(stack.layers) else stack.layers
        v_now = view()
        info["layers"][L_] = dict(kind=lay.kind, modes=lay.modes, pending=lay.pending, notes=lay.notes, build_s=lay.build_s,
                                  reuse=(lay.reuse.L if lay.reuse is not None else None), b_streams=getattr(lay, "b_stream_check", None),
                                  dram_retained_mb_per_bank=round(v_now["alloc_mb"] - v_prev["alloc_mb"], 2), dram_mb_ab=getattr(lay, "dram_mb", None),
                                  wall_s=round(time.time() - tl, 1))  # fmt: skip
        v_prev = v_now
        H.log(f"{tag}: L{L_} {lay.kind} chain built in {info['layers'][L_]['wall_s']} s: modes {lay.modes} pending {list(lay.pending)} "
              f"+{info['layers'][L_]['dram_retained_mb_per_bank']} MB/bank (A / B {info['layers'][L_]['dram_mb_ab']})")
    stack.n = len(info["layers"])
    stack.layers = [(li, pf.layers[li].L, pf.layers[li].kind) for li in range(stack.n)]
    stack.attach()
    info["dram"]["after_chain_build"] = view()
    info["chain_build_s"] = round(time.time() - t1, 1)
    # the compile launch of every key (TPPrefill.compile's forwards: every program's first launch, before any capture)
    keys = [(b, 0) for b in BUCKETS] + list(pf.fresh_keys(BUCKETS)) + list(pf.sparse_keys(BUCKETS))
    keys = sorted(set(keys))
    for b, t in keys:
        tk = time.time()
        v0 = view()
        run_key(pf, b, t)
        v1 = view()
        row = dict(mode=pf.mode(b), pmk=b in stack.pmk_buckets, retained_mb=round(v1["alloc_mb"] - v0["alloc_mb"], 2),
                   free_mb=v1["free_mb"], largest_mb=v1["largest_mb"], compile_s=round(time.time() - tk, 1))  # fmt: skip
        if STEPS > 0:
            row["need_mb"] = need_by_ballast(pf, b, t, STEPS)
        info["keys"][f"{b}:{t}"] = row
        H.log(f"{tag}: key {(b, t)} {row}")
    info["dram"]["after_compile"] = view()
    # one capture key per PMK bucket; the replay (mock: the forward again = the stub replay of every program)
    key = stack.trace_capture(c)
    stack.replay(c)
    ttnn.synchronize_device(mesh)
    info["key"] = list(key)
    info["chain_counters"] = {L_: dict(stack.progs[(c, L_)].counters) for L_ in info["layers"]}
    # GLM53_PF_MODE=op on the same TPPrefill: nothing installed, the op-graph forward
    stack.detach()
    assert pf.pmk_layers == {}, pf.pmk_layers
    os.environ["GLM53_PF_MODE"] = "op"
    run_key(pf, c, 0)
    stack.attach()
    summ = stack.summary()
    info["plans"] = {L_: stack.progs[(c, L_)].plan() for L_ in info["layers"]}
    info["dram_model_45"] = PS.served_stack_dram_bytes(TPN, c)
    info["signature"] = stack.signature()
    # program B's stream tensor per chip: chip k's bytes == the bfp4 round trip of chip k's own packed slices, and the
    # chips differ (Stage 12's finding on the 00:04Z L4 row: the replicated chip-0 copy)
    for L_ in info["layers"]:
        lay = stack.progs[(c, L_)]
        if lay.b is None:
            continue
        ep = lay.pl.epmoe
        n = TPN
        if getattr(lay.b, "inplace", False):  # in place: the streamers read pl.xsl (device k = slice k): the slices differ per chip
            # the structural guarantee: fused_decoder.load_xsl_experts builds every expert with from_host_shards(gh,
            # mesh.shape) -> device k = TP slice k; the readback check below is the device's (the mock's to_torch of an
            # ND-sharded DRAM tensor may return uninitialised zeros: then "unrepresentable", not a failure)
            host0 = lambda t, k: ttnn.to_torch(ttnn.get_device_tensors(t)[k] if n > 1 else t).float()  # noqa: E731
            e0 = ep.gate_up[0]
            reads = [host0(e0, k) for k in range(n)]
            nz = [round(float((r != 0).float().mean()), 4) for r in reads]
            distinct = [bool(not torch.equal(reads[0], reads[k])) for k in range(1, n)]
            representable = any(v > 0 for v in nz)
            info["layers"][L_]["b_stream_per_chip"] = dict(inplace=True, equals_own_slices=[True] * n, nonzero_frac=nz,
                                                           differs_from_chip0=distinct if representable else [True] * (n - 1),
                                                           readback=("ok" if representable else "mock: ND-sharded readback all zero (unrepresentable)"))  # fmt: skip
            H.log(f"{tag}: L{L_} program B in place (pl.xsl, from_host_shards: device k = slice k): readback nonzero {nz}, "
                  f"differs from chip 0 {distinct} ({info['layers'][L_]['b_stream_per_chip']['readback']})")
            continue
        host = lambda t, k: ttnn.to_torch(ttnn.get_device_tensors(t)[k] if n > 1 else t).to(torch.bfloat16).reshape(-1, int(t.shape[-1]))  # noqa: E731
        stacked, _, _ = PS.pack_streams_per_chip([[host(t, k) for t in ep.gate_up] for k in range(n)], [[host(t, k) for t in ep.down] for k in range(n)])
        dev = ttnn.get_device_tensors(lay.b.stream_t) if n > 1 else [lay.b.stream_t]
        same, diff = [], []
        for k in range(n):
            ref = ttnn.to_torch(ttnn.from_torch(stacked[k].reshape(1, 1, -1, 32), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
            got = ttnn.to_torch(dev[k])
            same.append(bool(torch.equal(ref.reshape(-1), got.reshape(-1))))
            diff.append(bool(not torch.equal(got.reshape(-1), ttnn.to_torch(dev[0]).reshape(-1))) if k else False)
        info["layers"][L_]["b_stream_per_chip"] = dict(equals_own_slices=same, differs_from_chip0=diff[1:])
        H.log(f"{tag}: L{L_} program B stream per chip: equals own slices {same}, differs from chip 0 {diff[1:]}")
    info["env_keys"] = {k: v[0] for k, v in PS.served_stack_env(TPN).items()}
    info["state_handoff"] = {k: (v if k != "layers" else {L_: {kk: vv for kk, vv in e.items() if kk != "layout"} for L_, e in v.items()})
                             for k, v in stack.state_handoff().items()}  # fmt: skip
except Exception:  # noqa: BLE001 -- recorded as the row's failure, the server stays up
    err = traceback.format_exc()[-6000:]
    H.log(f"FAILED:\n{err}")

FLOOR = int(args.get("floor", 1330000))
census = summ.get("census", {})
launch_max = {k: max(v["launch"].values()) for k, v in census.items() if v.get("launch")}
fits = all(v <= FLOOR for v in launch_max.values()) if launch_max else False
programs = sum(summ["wired"][k]["programs"] for k in summ.get("wired", {})) if summ else 0
one_key = (len(summ.get("captured", {})) == 1) if summ else False
pmk_layers = [L_ for L_, v in info["layers"].items() if any(m == "pmk" for m in v["modes"].values())]
streams_ok = all(all(v["b_stream_per_chip"]["equals_own_slices"]) and all(v["b_stream_per_chip"]["differs_from_chip0"])
                 for v in info["layers"].values() if "b_stream_per_chip" in v) if err is None else False
ok = err is None and fits and one_key and programs > 0 and len(info["keys"]) == len(set(info["keys"])) and streams_ok
status = "pass" if ok else "fail"
row = dict(test=f"PMK served stack mock (GLM53_PF_MODE=pmk, slots {os.environ['GLM53_PMK_SLOTS']}): layers {LAYERS} on the "
                f"{'mock' if C.MOCK else 'device'} tp{TPN}, buckets {list(BUCKETS)}: every piece program built + launched "
                f"through TPPrefill.forward, launch census <= {FLOOR}, one capture key per PMK bucket, the op forward after detach",
           milestone="PMK-B5", status=status, tp=TPN, layers=LAYERS, buckets=list(BUCKETS), pmk_bucket=info.get("key"),
           pmk_layers=pmk_layers, programs_per_replay=programs, launch_census_max=launch_max, census=census, fits=fits,
           captured=summ.get("captured"), counters=summ.get("counters"), wired=summ.get("wired"),
           dram=info.get("dram"), dram_per_layer={L_: v["dram_retained_mb_per_bank"] for L_, v in info["layers"].items()},
           reuse={L_: v.get("reuse") for L_, v in info["layers"].items()}, b_stream_per_chip={L_: v.get("b_stream_per_chip") for L_, v in info["layers"].items() if "b_stream_per_chip" in v},
           streams_ok=streams_ok, signature=(info.get("signature") or {}).get("signature"), env_signature=(info.get("signature") or {}).get("env_signature"),
           keys=info.get("keys"), dram_model_45=info.get("dram_model_45"), build_s=info.get("build_s"), chain_build_s=info.get("chain_build_s"),
           info={k: v for k, v in info.items() if k not in ("dram", "keys", "dram_model_45")} if len(str(info)) < 16000 else str(info)[:16000],
           error=err, mock=C.MOCK, wall_s=round(time.time() - t0, 1), tag=tag)  # fmt: skip
H.log({k: v for k, v in row.items() if k not in ("test", "wired", "info", "error", "census", "keys", "dram")})
H.log(f"census: {census}")
if rec:
    C.record("tests", row)
RESULT = row
