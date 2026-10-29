# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP pair): Stage 16 CO-ADMISSION LEAK unit (program 2026-10-22 15:33:21Z, 14b's x01 bisection at TP8
int27: a 97-token prompt's first token 444 -> 547, |dlogit| 28, when admitted PACKED or BATCHED with [1024, 33] vs alone;
per-user bitwise; independent of the co-admitted lengths). One build, buckets 128 / 512 / 2048, the served prefill knobs
(args env), GLM53_PF_REP_BUCKETS (default "128": the pair runs the 128 bucket REPLICATED as TP8 does -- 128 is not a
multiple of 32 x 8 -- while 512 / 2048 are sequence-parallel on both). Routes, each on its own slots, all traced:
  solo      every prompt alone through the single-request path (97 / 33 -> bucket 128, 1024 -> 2048)
  solo2048  the same prompts alone in the 2,048 bucket (sp): separates "bucket / mode" from "co-admission"
  packed    [97, 1024, 33] in ONE packed 2,048 replay (GLM53_PF_PACKED 2048:3)
  batched   [97, 1024, 33] in ONE batched replay (GLM53_PF_BATCH 3: 3 x 2,048 rows)
Per prompt and route vs solo: the logits row of the last token (bitwise, max |d|), the first token, every KDA layer's
S / C, the sparse_dsa layers' latent rows < n and pooled keys -- the first layer that differs names the stage.
ARGS: layers ([0..4]), rep ("128"), lens ([97, 1024, 33]), env (the knobs), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.environ.get("TT_METAL_HOME", ""), "models/autoports/zai_org_glm_5_3_flash/tests/s16"))  # (an out-of-tree copy)
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
LENS = [int(v) for v in args.get("lens", [97, 1024, 33])]
T = 2048
tag = args.get("tag", "p4-coadmit-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()
GOLD = "/opt/glm53/kit/goldens"
prompt = torch.load(f"{GOLD}/prompts/2048.pt", map_location="cpu", weights_only=True).reshape(-1).long()
# distinct prompts (the 97-token target first, as in 14b's A97 case)
starts = [0, 200, 1500]
PROMPTS = [prompt[s_ : s_ + n] for s_, n in zip(starts, LENS)]
if LENS[0] <= 128:  # the target = the 128-token golden prompt's prefix: layer_io/L*/128 rows < n are its CPU reference
    PROMPTS[0] = torch.load(f"{GOLD}/prompts/128.pt", map_location="cpu", weights_only=True).reshape(-1).long()[: LENS[0]]
NP = len(LENS)

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
os.environ["GLM53_PF_PACKED"] = f"{T}:{NP}"
os.environ["GLM53_PF_BATCH"] = str(NP)
os.environ["GLM53_PF_PACKED_CONT"] = "0"
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
# set_attn WRITES GLM53_PF_REP_BUCKETS from args["rep_buckets"] (an env value would be overwritten)
args["rep_buckets"] = [int(v) for v in str(args.get("rep", "128")).split(",") if v.strip()]
S.set_attn(args)
ROUTES = ["solo", "solo2048", "packed", "batched"]
SLOTS = {r: [i * NP + j for j in range(NP)] for i, r in enumerate(ROUTES)}
ent = S.build(mesh, ctx, LAYERS, b=len(ROUTES) * NP, buckets=(128, 512, T), max_seq=2560)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf
S.log(f"modes: 128 {pf.mode(128)}, 512 {pf.mode(512)}, 2048 {pf.mode(T)}; batch keys {pf.batch_keys}; packed keys "
      f"{pf.packed_keys}; align {backend.prefill_packed_align()}")


def logits_row(slot):
    return backend.read(io.logits)[slot].float().clone()


def run_solo(slots, force=None):
    first, lg = [], []
    keep = pf.buckets
    if force:
        pf.buckets = (force,)
    try:
        for tk, slot in zip(PROMPTS, slots):
            f = backend.prefill_chunk(tk, kv_cache=kv, slot=slot, chunk_start=0, n_valid=int(tk.numel()),
                                      logits_rows="last", io=io, sample=GREEDY)  # fmt: skip
            first.append(int(f))
            lg.append(logits_row(slot))
    finally:
        pf.buckets = keep
    return first, lg


def run_packed(slots):
    reqs = [(tk, slot, int(tk.numel())) for tk, slot in zip(PROMPTS, slots)]
    first = backend.prefill_packed_tp(reqs, kv_cache=kv, io=io, t=T, sample=[GREEDY] * NP)
    return first, [logits_row(s_) for s_ in slots]


def run_batched(slots):
    reqs = [(tk, slot, int(tk.numel())) for tk, slot in zip(PROMPTS, slots)]
    first = backend.prefill_batch_tp(reqs, kv_cache=kv, io=io, sample=GREEDY)
    return first, [logits_row(s_) for s_ in slots]


RUN = {"solo": lambda: run_solo(SLOTS["solo"]), "solo2048": lambda: run_solo(SLOTS["solo2048"], force=T),
       "packed": lambda: run_packed(SLOTS["packed"]), "batched": lambda: run_batched(SLOTS["batched"])}


def snapshot(slots):
    host = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()  # noqa: E731
    lat = {L: host(t_) for L, t_ in backend.latent_pages(kv).items()}
    out = []
    for tk, slot in zip(PROMPTS, slots):
        end = int(tk.numel())
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
        out.append(dict(kda=kda, lat=lr, dsa=dsa))
    return out


def compare(a, b):
    res = []
    for x, y in zip(a, b):
        diffs = []
        for L in sorted(x["kda"]):
            for nm, i in (("S", 0), ("C", 1)):
                u, v = x["kda"][L][i], y["kda"][L][i]
                diffs.append((f"L{L}.{nm}", float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten())))
        for L in sorted(x["lat"]):
            u, v = x["lat"][L], y["lat"][L]
            diffs.append((f"L{L}.lat", float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten())))
        for nm in sorted(x["dsa"]):
            u, v = x["dsa"][nm], y["dsa"][nm]
            diffs.append((nm, float((u - v).abs().max()), C.pcc(u.flatten(), v.flatten()) if u.numel() > 1 else 1.0))
        first_bad = next((nm for nm, m, _ in sorted(diffs, key=lambda z: (int(z[0][1:].split(".")[0]), z[0])) if m > 0),
                         None)  # fmt: skip
        res.append(dict(bitwise=all(m == 0.0 for _, m, _ in diffs), first_bad=first_bad,
                        per={nm: [round(m, 6), round(p, 6)] for nm, m, p in diffs}))  # fmt: skip
    return res


# ---- compile (eager, every key incl. batch + packed) -> every route once eager (host helper programs) -> capture
t0 = time.time()
pf.compile()
for r in ROUTES:
    RUN[r]()
if not C.MOCK:
    pf.capture()
S.log(f"compiled + captured in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}")
eager0 = pf.counters["eager"]
got = {r: RUN[r]() for r in ROUTES}
traced = pf.counters["eager"] == eager0
snaps = {r: snapshot(SLOTS[r]) for r in ROUTES}
rows = []
for r in ROUTES[1:]:
    cmp = compare(snaps["solo"], snaps[r])
    for j, n in enumerate(LENS):
        a, b = got["solo"][1][j], got[r][1][j]
        rows.append(dict(route=r, n=n, first_solo=got["solo"][0][j], first_route=got[r][0][j],
                         logits_bitwise=bool(torch.equal(a, b)), logits_max_abs=round(float((a - b).abs().max()), 5),
                         state_bitwise=cmp[j]["bitwise"], first_bad=cmp[j]["first_bad"], per=cmp[j]["per"]))  # fmt: skip
        S.log({k_: v_ for k_, v_ in rows[-1].items() if k_ != "per"})
ok = all(x["logits_bitwise"] and x["state_bitwise"] for x in rows if x["route"] in ("packed", "batched"))
row = dict(test=f"p4 CO-ADMISSION leak unit: prompts {LENS} alone (rep buckets {os.environ['GLM53_PF_REP_BUCKETS']}) vs "
                f"alone@2048 vs packed vs batched: logits, first tokens, KDA S / C, latents, DSA keys, tp{TPN} layers "
                f"{len(LAYERS)}", milestone="P4", status=("pass" if ok and (traced or C.MOCK) else "fail"),
           traced=traced, tp=TPN, layers=len(LAYERS), modes={c_: pf.mode(c_) for c_ in (128, 512, T)}, rows=rows,
           mock=C.MOCK, tag=tag)  # fmt: skip
S.log({k_: v_ for k_, v_ in row.items() if k_ != "rows"})
if rec and not C.MOCK:
    C.record("tests", row)
RESULT = [row]
