# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 PACKED prefill (mixed ISL, program 2026-10-22 03:15Z), first row: K FRESH
requests of DIFFERENT lengths (default 700 / 900 / 400 tokens: different slices of the reference prompt) in ONE replay
of the (T, K) packed program (TPPrefill.prefill_packed via StackBackend.prefill_packed_tp) vs the same requests one by
one through the production single-request path (StackBackend.prefill_chunk). Per request: the first token (device
sampled), every KDA layer's final state S and conv carry C, and the slot's latent cache rows of every sparse_dsa layer
(positions < n) -- bitwise flag, max |diff|, PCC. Then both paths as traced replays: the wall of the K serial requests
vs the packed one, and each trace's steady-state replay time.

ARGS: adopt ("": Stage 16's own build | "m5": the served backend of an earlier job, TPPrefill rebuilt with the packed
keys under args env), layers ([0..4]), lens ([700, 900, 400]), offs ([0, 600, 1500]: each request's first token in the reference
prompt), t (2048: the packed bucket), ref, reps (3), replays (20), the prefill knobs (s16common.set_attn + moe / moe_v2 /
kda_scan; default the service knobs: absorbed MLA, op scan, nq 4 / 640 KB), env, tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
from readiness_check.schema import load_reference  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [] if args.get("layers") == "all" else [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]  # adopt: the backend's
LENS = [int(v) for v in args.get("lens", [700, 900, 400])]
OFFS = [int(v) for v in args.get("offs", [0, 600, 1500])]
T = int(args.get("t", 2048))
K = len(LENS)
REPS = int(args.get("reps", 3))
REF = args.get("ref", "/opt/glm53/work-s16/refpt/prefill2048_L0-4.refpt")
tag = args.get("tag", "p4-packed")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()

prompt = load_reference(REF).entries[0].prompt_tokens.reshape(-1).long()
assert all(o + n <= prompt.numel() for o, n in zip(OFFS, LENS)), (OFFS, LENS, prompt.numel())
ADOPT = args.get("adopt", "")
if ADOPT == "m5":  # 14b's SERVED construction (M5.serving): an earlier job's, or built here under args env
    keys = [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "m5_serving"]
    if not keys:
        from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
        import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
        import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "s14b"))
        import m5common as M5

        for k_, v_ in (args.get("env") or {}).items():  # the served env with the packed keys OFF (the rebuild adds them)
            os.environ[str(k_)] = str(v_)
        os.environ["GLM53_PF_PACKED"] = ""
        cfg_ = hf_config()
        ids_ = list(range(cfg_.num_hidden_layers)) if args.get("layers") == "all" else LAYERS
        keys = [M5.serving(ctx, mesh, SM, G, B=int(args.get("b", 32)), ids=ids_, cfg=cfg_,
                           native=bool(args.get("native", True)), rev=args.get("rev", "t25b"), log=S.log)]  # fmt: skip
    backend, gen = ctx[keys[-1]]["backend"], ctx[keys[-1]]["gen"]
    LAYERS = [int(e["L"]) for e in backend.seq]
    for k_, v_ in (args.get("env") or {}).items():  # the served env as is (S.set_attn would reset the P3 knobs)
        os.environ[str(k_)] = str(v_)
    os.environ["GLM53_PF_PACKED"] = f"{T}:{K}"  # the packed keys (T, 2 .. K) join every other key of the rebuild
    os.environ["GLM53_PF_PROBE_LAYERS"] = ""
    binding = gen._own_binding()
    kv, io = binding.kv_cache, binding.io
    backend.tpp.release()
    backend.tpp = None
    backend.prepare_prefill(kv)  # every key compiled (packed included) BEFORE capture() records them all
    backend.tpp.capture()
    gen.reset()
    pf = backend.tpp
    assert C.MOCK or ("packed", T, K) in pf.traces, sorted(map(str, pf.traces))  # (a mock captures nothing)
else:
    os.environ["GLM53_PF_MHC"] = "comp"
    os.environ["GLM53_PF_MHC_POST"] = "fused"
    os.environ["GLM53_PF_DSA"] = "0"
    args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
    S.set_attn(args)
    ent = S.build(mesh, ctx, LAYERS, b=max(4, K), buckets=(T,), max_seq=2560)
    pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
reqs = [(prompt[o : o + n], k, n) for k, (o, n) in enumerate(zip(OFFS, LENS))]
offs_packed = pf.packed_layout(T, LENS)


def serial(tag_=""):
    first = []
    for tk, slot, n in reqs:
        first.append(int(backend.prefill_chunk(tk, kv_cache=kv, slot=slot, chunk_start=0, n_valid=n, logits_rows="last",
                                               io=io, sample=GREEDY)))  # fmt: skip
    return first


def packed():
    return backend.prefill_packed_tp(reqs, kv_cache=kv, io=io, t=T, sample=GREEDY)


def snapshot():
    """per request: KDA S / C of every KDA layer and the latent rows (< n) of every sparse_dsa layer (host)"""
    lat = {L: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()
           for L, t_ in backend.latent_pages(kv).items()}  # fmt: skip
    snap = []
    for tk, slot, n in reqs:
        row = pf.page_row(slot)
        kda = backend.export_kda_state(kv, slot)
        lr = {}
        for L, cache in lat.items():  # [blocks, 1, 64, 512]
            pages = [int(row[p]) for p in range(-(-n // 64))]
            lr[L] = torch.cat([cache[p, 0] for p in pages], dim=0)[:n]
        dsa = {}
        for pl in pf.layers:  # GLM53_PF_DSA=1: the slot's pooled indexer keys (full pools < n // 4) and ring rows 0..3
            idx = getattr(getattr(pl, "mla", None), "idx", None)
            if idx is None:
                continue
            host = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()  # noqa: E731
            pps = idx.pages_per_slot
            st = host(idx.store)[slot * pps : (slot + 1) * pps].reshape(-1, 128)  # [pps * 32 pools, 128]
            dsa[f"L{pl.L}.pool"] = st[: n // 4]
            dsa[f"L{pl.L}.ring"] = host(idx.ring)[slot, 0, :4]
        snap.append(dict(kda={e["L"]: (e["S"].float(), e["C"].float()) for e in kda}, lat=lr, dsa=dsa))
    return snap


def compare(a, b):
    out = []
    for k, (x, y) in enumerate(zip(a, b)):
        d = dict(req=k, n=LENS[k])
        diffs = []
        for L in x["kda"]:
            for nm, i in (("S", 0), ("C", 1)):
                u, v = x["kda"][L][i], y["kda"][L][i]
                diffs.append((f"L{L}.{nm}", float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten())))
        for L in x["lat"]:
            u, v = x["lat"][L], y["lat"][L]
            diffs.append((f"L{L}.lat", float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten())))
        for nm in x.get("dsa", {}):  # the DSA indexer's pooled keys / ring of the slot
            u, v = x["dsa"][nm], y["dsa"][nm]
            diffs.append((nm, float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten())))
        d["bitwise"] = all(m == 0.0 for _, m, _ in diffs)
        d["max_abs"] = round(max(m for _, m, _ in diffs), 6)
        d["pcc_min"] = round(min(p for _, _, p in diffs), 6)
        d["worst"] = min(diffs, key=lambda z: z[2])[0]
        d["per"] = {nm: [round(m, 6), round(p, 6)] for nm, m, p in diffs}
        out.append(d)
    return out


# ---- eager: every program compiles before any capture (single keys, the packed forward, the request paths' buffers)
t0 = time.time()
if ADOPT != "m5":  # (adopt: compiled + captured by the rebuild above; the "eager" requests below replay its traces)
    pf.compile((T,))
first_serial = serial()
snap_s = snapshot()
first_packed = packed()
snap_p = snapshot()
cmp = compare(snap_s, snap_p)
S.log(f"eager: serial first {first_serial}, packed first {first_packed}, {time.time() - t0:.1f} s; {cmp}")

# ---- traced: the single keys and the packed key, then the timed requests
traced = False
walls_s, walls_p, rs_s, rs_p = [], [], {}, {}
if not C.MOCK:
    pf.capture((T,))
    pf.capture_packed(T, K)
    traced = True
    eager0 = pf.counters["eager"]
    for rep in range(REPS + 1):
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        fs = serial()
        ttnn.synchronize_device(mesh)
        t2 = time.perf_counter()
        fp = packed()
        ttnn.synchronize_device(mesh)
        t3 = time.perf_counter()
        if rep:
            walls_s.append((t2 - t1) * 1e3)
            walls_p.append((t3 - t2) * 1e3)
    traced = pf.counters["eager"] == eager0
    rs_s = S.replay_stats(mesh, pf.traces[pf.plan(0, max(LENS))], replays=args.get("replays", 20), label="single")
    rs_p = S.replay_stats(mesh, pf.traces[("packed", T, K)], replays=args.get("replays", 20), label="packed")
    first_serial_t, first_packed_t = fs, fp
else:
    first_serial_t, first_packed_t = first_serial, first_packed
med = lambda v: sorted(v)[len(v) // 2] if v else None  # noqa: E731
ok = first_serial == first_packed and all(c_["pcc_min"] >= 0.99 for c_ in cmp)
row = dict(test=f"p4 PACKED prefill (mixed ISL): {K} fresh requests of {LENS} tokens in ONE {T}-row replay vs {K} serial "
                f"single-request prefills, first token on device, tp{TPN} layers {len(LAYERS)}",
           milestone="P4", status="pass" if ok and (traced or C.MOCK) else "fail", traced=traced, tp=TPN,
           layers=len(LAYERS), lens=LENS, offs=OFFS, packed_offsets=offs_packed, t=T,
           first_serial=first_serial, first_packed=first_packed, first_serial_traced=first_serial_t,
           first_packed_traced=first_packed_t, same_first=first_serial == first_packed,
           bitwise=[c_["bitwise"] for c_ in cmp], pcc_min=[c_["pcc_min"] for c_ in cmp],
           max_abs=[c_["max_abs"] for c_ in cmp], worst=[c_["worst"] for c_ in cmp],
           serial_ms=round(med(walls_s), 2) if walls_s else None, packed_ms=round(med(walls_p), 2) if walls_p else None,
           single_replay_us=rs_s.get("median_us"), packed_replay_us=rs_p.get("median_us"),
           replays=[rs_s.get("replays"), rs_p.get("replays")], knobs=dict(mla_form=os.environ.get("GLM53_PF_MLA_FORM"),
           kda_scan=pf.kda_scan, moe_ffn=os.environ.get("GLM53_PF_MOE_FFN", "")), detail=cmp, tag=tag)  # fmt: skip
S.log({k_: v for k_, v in row.items() if k_ != "detail"})
if rec:
    C.record("pcc", row)
RESULT = [row]
