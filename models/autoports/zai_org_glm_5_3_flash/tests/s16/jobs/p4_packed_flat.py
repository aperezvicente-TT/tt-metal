# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 FLAT packed step (program 2026-10-22 07:59:27Z): ONE packed replay holding a
FRESH segment, a CONTINUATION segment (chunk start > 0: attends the slot's cached prefix, KDA from the slot's state)
and a HEAD-LESS intermediate segment (state only, no sample) vs each segment's serial single-request path, on separate
slots (serial: slots 0 / 1 / 2, packed: 3 / 4 / 5; the continuation's prefix is prefilled identically on both slots
first). Per segment: the first token (head segments), the KDA S / C of every KDA layer, the latent rows < end of every
sparse_dsa layer and (GLM53_PF_DSA=1) the slot's pooled indexer keys (full pools) and ring rows -- bitwise flags,
max |diff|, PCC. Everything traced (the eager forwards are the compile, before the captures).

ARGS: adopt ("" | "m5": 14b's served construction, TPPrefill rebuilt with GLM53_PF_PACKED=T:3 + the continuation
twin), layers ([0..4]), fresh (600), cont ([1408, 600]: prefix rows, continuation rows), inter (640: the head-less
segment), t (2048), env (the knobs; GLM53_PF_PACKED_CONT=1 is set here), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
NF = int(args.get("fresh", 600))
PRE, NC = (int(v) for v in args.get("cont", [1408, 600]))
NI = int(args.get("inter", 640))
T = int(args.get("t", 2048))
tag = args.get("tag", "p4-packed-flat")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()
GOLD = "/opt/glm53/kit/goldens"

prompt = torch.load(f"{GOLD}/prompts/2048.pt", map_location="cpu", weights_only=True).reshape(-1).long()
assert PRE + NC <= prompt.numel() and 100 + NI <= prompt.numel()
ADOPT = args.get("adopt", "")
if ADOPT == "m5":  # 14b's SERVED construction (an earlier job's, or built here); its TPPrefill rebuilt with the twins
    keys = [k_ for k_ in ctx if isinstance(k_, tuple) and k_ and k_[0] == "m5_serving"]
    for k_, v_ in (args.get("env") or {}).items():
        os.environ[str(k_)] = str(v_)
    if not keys:
        from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
        import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
        import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "s14b"))
        import m5common as M5

        os.environ["GLM53_PF_PACKED"] = ""
        cfg_ = hf_config()
        ids_ = list(range(cfg_.num_hidden_layers)) if args.get("layers") == "all" else LAYERS
        keys = [M5.serving(ctx, mesh, SM, G, B=int(args.get("b", 32)), ids=ids_, cfg=cfg_,
                           native=bool(args.get("native", True)), rev=args.get("rev", "t25b"), log=S.log)]  # fmt: skip
    backend, gen = ctx[keys[-1]]["backend"], ctx[keys[-1]]["gen"]
    LAYERS = [int(e["L"]) for e in backend.seq]
    os.environ["GLM53_PF_PACKED"] = f"{T}:3"  # the packed keys (fresh + continuation twins) join the rebuild
    os.environ["GLM53_PF_PACKED_CONT"] = "1"
    os.environ["GLM53_PF_PROBE_LAYERS"] = ""
    binding = gen._own_binding()
    kv, io = binding.kv_cache, binding.io
    backend.tpp.release()
    backend.tpp = None
    backend.prepare_prefill(kv)  # every key incl. both packed twins compiled BEFORE capture() records them all
    gen.reset()
    pf = backend.tpp
else:
    os.environ["GLM53_PF_MHC"] = "comp"
    os.environ["GLM53_PF_MHC_POST"] = "fused"
    for k_, v_ in (args.get("env") or {}).items():
        os.environ[str(k_)] = str(v_)
    os.environ["GLM53_PF_PACKED_CONT"] = "1"
    args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
    S.set_attn(args)
    ent = S.build(mesh, ctx, LAYERS, b=8, buckets=(T,), max_seq=2560)
    pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
assert pf.packed_cont, "GLM53_PF_PACKED_CONT=1 at build"

# (tokens, n, chunk start, head) per segment: fresh, continuation, head-less intermediate
SEGS = [(prompt[:NF], NF, 0, True), (prompt[PRE : PRE + NC], NC, PRE, True), (prompt[100 : 100 + NI], NI, 0, False)]
SER, PAK = [0, 1, 2], [3, 4, 5]


def prefix(slot):  # the continuation's cached prefix, the same single-request prefill on both slots
    backend.prefill_chunk(prompt[:PRE], kv_cache=kv, slot=slot, chunk_start=0, n_valid=PRE, logits_rows="last", io=io,
                          sample=GREEDY)  # fmt: skip


def serial():
    first = []
    for (tk, n, st, hd), slot in zip(SEGS, SER):
        f = backend.prefill_chunk(tk, kv_cache=kv, slot=slot, chunk_start=st, n_valid=n, logits_rows="last", io=io,
                                  sample=GREEDY)  # fmt: skip
        first.append(int(f) if hd else None)
    return first


def packed():
    reqs = [(tk, slot, n, st, hd) for (tk, n, st, hd), slot in zip(SEGS, PAK)]
    return backend.prefill_packed_tp(reqs, kv_cache=kv, io=io, t=T, sample=[GREEDY if hd else None for *_, hd in SEGS])


def snapshot(slots):
    host = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()  # noqa: E731
    lat = {L: host(t_) for L, t_ in backend.latent_pages(kv).items()}
    out = []
    for (tk, n, st, hd), slot in zip(SEGS, slots):
        end = st + n
        row = pf.page_row(slot)
        kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, slot)}
        lr = {L: torch.cat([c_[int(row[p]), 0] for p in range(-(-end // 64))], dim=0)[:end] for L, c_ in lat.items()}
        dsa = {}
        for pl in pf.layers:
            idx = getattr(getattr(pl, "mla", None), "idx", None)
            if idx is None:
                continue
            pps = idx.pages_per_slot
            dsa[f"L{pl.L}.pool"] = host(idx.store)[slot * pps : (slot + 1) * pps].reshape(-1, 128)[: end // 4]
            dsa[f"L{pl.L}.ring"] = host(idx.ring)[slot, 0, :4]
        out.append(dict(kda=kda, lat=lr, dsa=dsa))
    return out


def compare(a, b):
    res = []
    for k, (x, y) in enumerate(zip(a, b)):
        diffs = []
        for L in x["kda"]:
            for nm, i in (("S", 0), ("C", 1)):
                u, v = x["kda"][L][i], y["kda"][L][i]
                diffs.append((f"L{L}.{nm}", float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten())))
        for grp in ("lat", "dsa"):
            for nm in x[grp]:
                u, v = x[grp][nm], y[grp][nm]
                diffs.append((f"{nm}" if grp == "dsa" else f"L{nm}.lat", float((u - v).abs().max()),
                              C.pcc(u.flatten(), v.flatten())))  # fmt: skip
        res.append(dict(seg=["fresh", "continuation", "head-less"][k], bitwise=all(m == 0.0 for _, m, _ in diffs),
                        max_abs=round(max(m for _, m, _ in diffs), 6), pcc_min=round(min(p for _, _, p in diffs), 6),
                        worst=min(diffs, key=lambda z: z[2])[0],
                        per={nm: [round(m, 6), round(p, 6)] for nm, m, p in diffs}))  # fmt: skip
    return res


# ---- compile (eager, every key incl. the packed twin) -> capture -> the traced comparison
t0 = time.time()
if ADOPT != "m5":  # (adopt: compiled by the rebuild; the warm pass below then runs before its captures)
    pf.compile((T,))
# the whole request sequence once EAGER before any capture: every program AND the request path's host-side helper ops
# (decode-row position / sampler writes: their copy programs keep program-cache buffers) compile before the captures
prefix(SER[1])
prefix(PAK[1])
serial()
packed()
if not C.MOCK:
    if ADOPT == "m5":
        pf.capture()  # every key of the rebuild (single buckets, batch, both packed twins)
    else:
        pf.capture((T,))
        pf.capture_packed(T, 3, cont=True)  # (the fresh-only twin is not replayed here)
S.log(f"compiled + captured in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}")
eager0 = pf.counters["eager"]
prefix(SER[1])
prefix(PAK[1])
first_s = serial()
first_p = packed()
traced = pf.counters["eager"] == eager0
cmp = compare(snapshot(SER), snapshot(PAK))
ok = first_s == first_p and all(c_["bitwise"] or c_["pcc_min"] >= 0.9999 for c_ in cmp)
row = dict(test=f"p4 FLAT packed step: ONE {T}-row replay of a fresh {NF} + a continuation {NC}@{PRE} + a head-less "
                f"intermediate {NI} vs each segment's serial single-request path (separate slots), first tokens, KDA "
                f"S / C, latent rows, DSA pooled keys + ring, tp{TPN} layers {len(LAYERS)}", milestone="P4",
           status=("pass" if ok and (traced or C.MOCK) else "fail"), traced=traced, tp=TPN, layers=len(LAYERS),
           first_serial=first_s, first_packed=first_p, bitwise=[c_["bitwise"] for c_ in cmp],
           pcc_min=[c_["pcc_min"] for c_ in cmp], max_abs=[c_["max_abs"] for c_ in cmp],
           worst=[c_["worst"] for c_ in cmp], caps=backend.prefill_packed_caps(), detail=cmp, mock=C.MOCK,
           tag=tag)  # fmt: skip
S.log({k_: v_ for k_, v_ in row.items() if k_ != "detail"})
if rec:
    C.record("pcc", row)
RESULT = [row]
