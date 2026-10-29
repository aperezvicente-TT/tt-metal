# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP pair): Stage 16 PACKED ADMISSION invariance at the served co-admission shape (program
2026-10-22 13:50:40Z, after 14b's 13:45:49Z: the decode MoE is bitwise row-invariant, so Stage 9's abortmix 13/24
non-bitwise rows come from the PREFILL side): FOUR head segments in ONE packed replay -- fresh 500 (the target),
fresh 400, a CONTINUATION 400@1,024 (attends its slot's cached 1,000-token prefix, KDA from the slot's state) and fresh
500 (4 x 512 aligned rows = the 2,048 bucket) -- vs each segment's serial single-request path on separate slots (serial 0..3,
packed 4..7; the continuation's prefix is prefilled identically on both slots first). Per segment: the FULL logits row
of its last token (bitwise, max |d|, PCC), the first token, the KDA S / C of every KDA layer, the latent rows < end of
every sparse_dsa layer and the DSA pooled keys / ring -- bitwise flags. Everything traced (the eager pass compiles).
ARGS: layers ([0..4]), t (2048), env (the knobs; GLM53_PF_PACKED = "t:4" and GLM53_PF_PACKED_CONT=1 are set here),
tag, record (True)."""
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
T = int(args.get("t", 2048))
PRE = 1024  # the continuation's cached prefix (a chunk start is a multiple of 64)
tag = args.get("tag", "p4-packed-quad")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()
GOLD = "/opt/glm53/kit/goldens"

prompt = torch.load(f"{GOLD}/prompts/2048.pt", map_location="cpu", weights_only=True).reshape(-1).long()
ADOPT = ""  # the own build (the served construction's packing code is the same TPPrefill.prefill_packed)
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
    os.environ["GLM53_PF_PACKED"] = f"{T}:4"  # the packed keys for FOUR segments (fresh + continuation twins)
    args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
    S.set_attn(args)
    ent = S.build(mesh, ctx, LAYERS, b=8, buckets=(T,), max_seq=2560)
    pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
assert pf.packed_cont, "GLM53_PF_PACKED_CONT=1 at build"

# (tokens, n, chunk start, head) per segment: the target fresh 500, fresh 400, a continuation 400@1,024, fresh 500 (4 x 512 aligned rows = the 2,048 bucket)
SEGS = [(prompt[:500], 500, 0, True), (prompt[600:1000], 400, 0, True), (prompt[PRE : PRE + 400], 400, PRE, True),
        (prompt[1424:1924], 500, 0, True)]  # fmt: skip
NAMES = ["fresh500 (target)", "fresh400", "continuation400@1024", "fresh500b"]
SER, PAK = [0, 1, 2, 3], [4, 5, 6, 7]
CONT = 2  # the continuation segment's index


def prefix(slot):  # the continuation's cached prefix, the same single-request prefill on both slots
    backend.prefill_chunk(prompt[:PRE], kv_cache=kv, slot=slot, chunk_start=0, n_valid=PRE, logits_rows="last", io=io,
                          sample=GREEDY)  # fmt: skip


def logits_row(slot):  # the slot's logits row (its segment's last token), read right after the call that wrote it
    return backend.read(io.logits)[slot].float().clone()


def serial():
    first, lg = [], []
    for (tk, n, st, hd), slot in zip(SEGS, SER):
        f = backend.prefill_chunk(tk, kv_cache=kv, slot=slot, chunk_start=st, n_valid=n, logits_rows="last", io=io,
                                  sample=GREEDY)  # fmt: skip
        first.append(int(f) if hd else None)
        lg.append(logits_row(slot))
    return first, lg


def packed():
    reqs = [(tk, slot, n, st, hd) for (tk, n, st, hd), slot in zip(SEGS, PAK)]
    first = backend.prefill_packed_tp(reqs, kv_cache=kv, io=io, t=T, sample=[GREEDY if hd else None for *_, hd in SEGS])
    return first, [logits_row(slot) for slot in PAK]


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
        res.append(dict(seg=NAMES[k], bitwise=all(m == 0.0 for _, m, _ in diffs),
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
prefix(SER[CONT])
prefix(PAK[CONT])
serial()
packed()
if not C.MOCK:
    if ADOPT == "m5":
        pf.capture()  # every key of the rebuild (single buckets, batch, both packed twins)
    else:
        pf.capture((T,))
        pf.capture_packed(T, 4, cont=True)  # (the fresh-only twin is not replayed here)
S.log(f"compiled + captured in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}")
eager0 = pf.counters["eager"]
prefix(SER[CONT])
prefix(PAK[CONT])
first_s, lg_s = serial()
first_p, lg_p = packed()
traced = pf.counters["eager"] == eager0
cmp = compare(snapshot(SER), snapshot(PAK))
lgc = [dict(bitwise=bool(torch.equal(a, b)), max_abs=round(float((a - b).abs().max()), 6), pcc=round(C.pcc(a, b), 7))
       for a, b in zip(lg_s, lg_p)]  # fmt: skip
ok = first_s == first_p and all(c_["bitwise"] for c_ in cmp) and all(l_["bitwise"] for l_ in lgc)
row = dict(test=f"p4 PACKED ADMISSION (served co-admission shape): ONE {T}-row replay of FOUR head segments (fresh 500 "
                f"target + fresh 400 + continuation 400@{PRE} + fresh 500) vs each segment's serial single-request "
                f"path (separate slots): logits rows, first tokens, KDA S / C, latent rows, DSA pooled keys + ring "
                f"BITWISE, tp{TPN} layers {len(LAYERS)}", milestone="P4",
           status=("pass" if ok and (traced or C.MOCK) else "fail"), traced=traced, tp=TPN, layers=len(LAYERS),
           first_serial=first_s, first_packed=first_p, logits=lgc, bitwise=[c_["bitwise"] for c_ in cmp],
           pcc_min=[c_["pcc_min"] for c_ in cmp], max_abs=[c_["max_abs"] for c_ in cmp],
           worst=[c_["worst"] for c_ in cmp], caps=backend.prefill_packed_caps(), detail=cmp, mock=C.MOCK,
           tag=tag)  # fmt: skip
S.log({k_: v_ for k_, v_ in row.items() if k_ != "detail"})
if rec:
    C.record("pcc", row)
RESULT = [row]
