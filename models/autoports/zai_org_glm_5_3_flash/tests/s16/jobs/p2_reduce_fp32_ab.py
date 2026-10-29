# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP pair): GLM53_PF_REDUCE_FP32 A/B (program 2026-10-22 19:43Z, operator question: is the fp32
cross-chip reduction necessary?). One build (layers L0-L4, buckets 256 / 512 / 2,048, packed 2,048:3), per config
v in VALUES (default 2 then 0): pf.reduce_fp32 = v, every config warmed (eager compile of every key + every route) BEFORE
the first capture, then captured and run traced:
  (a) TF proxy: the 2,048 golden's teacher-forced positions 2,016 + i (i < GEN) through the prefill alone (first token
      sampled on device) vs the L0-L4 CPU fp32 reference's top-1 / top-5 (refpt prefill2048_L0-4)
  (b) the 32-token collapse prompt golden_ids(2048)[:32] (first token; no L0-L4 CPU reference at n = 32: compared across
      configs) + the L0-L4 CPU fp32 prefix logits (prefix_L0-4_golden_logits, n = 64 / 96 / 97 / 128 / 160 / 161 / 162 /
      2,016): first token = the CPU fp32 top-1, the CPU top-1's device rank, max |dlogit|
  (c) A97 / A300 solo vs packed [97, 300, 1024] in one 2,048 replay: first tokens + max |dlogit| of the last row
  (d) the 2,048-row bucket: ms per traced replay (S.replay_stats, REPLAYS)
The pair sums 2 partials (=0: one bf16 rounding per reduce); TP8 sums 8 over two mesh axes (up to 7 roundings), so the
pair bounds the effect from below only. ARGS: layers ([0..4]), values ([2, 0]), gen (32), replays (500), env, tag,
record (True)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.environ.get("TT_METAL_HOME", ""), "models/autoports/zai_org_glm_5_3_flash/tests/s16"))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
VALUES = [int(v) for v in args.get("values", [2, 0])]
GEN = int(args.get("gen", 32))
REPLAYS = int(args.get("replays", 500))
T = 2048
tag = args.get("tag", "p2-reduce-fp32-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()
GOLD = "/opt/glm53/kit/goldens"
REFD = args.get("refdir", "/opt/glm53/work-s16/refpt")
g2048 = torch.load(f"{GOLD}/prompts/2048.pt", map_location="cpu", weights_only=True).reshape(-1).long()
REF = torch.load(f"{REFD}/prefill2048_L0-4.refpt", map_location="cpu", weights_only=False)
ent0 = REF["entries"][0]
P = int(ent0["prompt_tokens"].shape[-1])
REF_TOPK = ent0["topk_tokens"].long()  # [G, K]
PFX = torch.load(f"{REFD}/prefix_L0-4_golden_logits.pt", map_location="cpu", weights_only=False)
assert torch.equal(PFX["prompt"].reshape(-1).long()[:P], g2048[:P]), "prefix logits prompt != goldens/prompts/2048.pt"
PFX_NS = [int(n) for n in PFX["ns"]]
CO_LENS = [97, 300, 1024]
CO = [g2048[0:97], g2048[200:500], g2048[600:1624]]

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
os.environ["GLM53_PF_PACKED"] = f"{T}:{len(CO)}"
os.environ["GLM53_PF_PACKED_CONT"] = "0"
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
args["rep_buckets"] = [int(v) for v in str(args.get("rep", "")).split(",") if v.strip()]
S.set_attn(args)
# slots: 0 = TF / prefix / collapse (each request restarts at chunk 0), 1..3 = solo co-admission prompts, 4..6 = packed
ent = S.build(mesh, ctx, LAYERS, b=7, buckets=(256, 512, T), max_seq=2560)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf
S.log(f"modes: 256 {pf.mode(256)}, 512 {pf.mode(512)}, 2048 {pf.mode(T)}; packed keys {pf.packed_keys}; "
      f"reduce_fp32 default {pf.reduce_fp32}, reduce_rows {getattr(pf, 'reduce_rows', None)}")


def last_logits(slot):
    return backend.read(io.logits)[slot].float().clone()


def solo(tk, slot):
    f = backend.prefill_chunk(tk, kv_cache=kv, slot=slot, chunk_start=0, n_valid=int(tk.numel()), logits_rows="last",
                              io=io, sample=GREEDY)  # fmt: skip
    return int(f), last_logits(slot)


def packed():
    reqs = [(tk, 4 + j, int(tk.numel())) for j, tk in enumerate(CO)]
    first = backend.prefill_packed_tp(reqs, kv_cache=kv, io=io, t=T, sample=[GREEDY] * len(CO))
    return [int(f) for f in first], [last_logits(4 + j) for j in range(len(CO))]


def run_all(light: bool = False):
    """every route once; light = one request per route (the eager warm of the host-side helper programs)"""
    out = {}
    tf_ns = [P + i for i in range(1 if light else GEN)]
    out["tf"] = [solo(g2048[:n], 0)[0] for n in tf_ns]
    out["pfx"] = {n: solo(g2048[:n], 0) for n in (PFX_NS[:1] if light else PFX_NS)}
    out["collapse"] = solo(g2048[:32], 0)
    out["solo"] = [solo(tk, 1 + j) for j, tk in enumerate(CO)]
    out["packed"] = packed()
    return out


def score(o):
    tf = [dict(i=i, tok=t_, top1=bool(t_ == int(REF_TOPK[i, 0])), top5=bool(t_ in REF_TOPK[i, :5].tolist()))
          for i, t_ in enumerate(o["tf"])]  # fmt: skip
    pfx = {}
    for j, n in enumerate(PFX_NS):
        f, lg = o["pfx"][n]
        g = PFX["logits"][j].float()
        gt = int(g.argmax())
        v = lg[: g.numel()]
        pfx[n] = dict(first=f, cpu_top1=gt, match=f == gt, cpu_top1_rank=int((v > v[gt]).sum()) + 1,
                      max_dlogit=round(float((v - g).abs().max()), 4))  # fmt: skip
    co = []
    for j, n in enumerate(CO_LENS):
        (fs, ls), fp, lp = o["solo"][j], o["packed"][0][j], o["packed"][1][j]
        co.append(dict(n=n, first_solo=fs, first_packed=fp, same=fs == fp, bitwise=bool(torch.equal(ls, lp)),
                       max_dlogit=round(float((ls - lp).abs().max()), 5)))  # fmt: skip
    return dict(tf_top1=round(sum(x["top1"] for x in tf) / len(tf), 4), tf_top5=round(sum(x["top5"] for x in tf) / len(tf), 4),
                tf_misses=[x for x in tf if not x["top1"]], pfx=pfx, pfx_match=sum(x["match"] for x in pfx.values()),
                collapse_first=o["collapse"][0], coadmit=co)  # fmt: skip


def bucket_trace(c):
    ks = [k for k in pf.traces if isinstance(k, tuple) and len(k) == 2 and k[0] == c]
    return pf.traces[ks[0]] if ks else None


def set_fp32(v):
    pf.reduce_fp32 = v
    os.environ["GLM53_PF_REDUCE_FP32"] = str(v)


# ---- warm every config (eager compile of every key + every route) before the first capture
t0 = time.time()
for v in VALUES:
    set_fp32(v)
    pf.compile()
    run_all(light=True)
S.log(f"warm ({VALUES}) {time.time() - t0:.1f} s")
res, outs = {}, {}
for v in VALUES:
    set_fp32(v)
    pf.release()
    if not C.MOCK:
        pf.capture()
    eager0 = pf.counters["eager"]
    t1 = time.time()
    o = run_all()
    traced = pf.counters["eager"] == eager0
    rs = S.replay_stats(mesh, bucket_trace(T), replays=REPLAYS, label=f"2048 fp32={v}")
    sc = score(o)
    outs[v] = o
    res[v] = dict(reduce_fp32=v, traced=traced, wall_s=round(time.time() - t1, 1),
                  replay_ms_2048=round(rs["median_us"] / 1e3, 3) if rs else None, replays=rs.get("replays"), **sc)
    S.log({k_: v_ for k_, v_ in res[v].items() if k_ not in ("pfx", "tf_misses")})
    S.log({"pfx": res[v]["pfx"], "tf_misses": res[v]["tf_misses"]})
a, b = VALUES[0], VALUES[-1]
cross = dict(tf_tokens_equal=outs[a]["tf"] == outs[b]["tf"], collapse_equal=outs[a]["collapse"][0] == outs[b]["collapse"][0],
             collapse_max_dlogit=round(float((outs[a]["collapse"][1] - outs[b]["collapse"][1]).abs().max()), 4),
             pfx_first_equal={n: outs[a]["pfx"][n][0] == outs[b]["pfx"][n][0] for n in PFX_NS})  # fmt: skip
ok = all(r_["traced"] for r_ in res.values()) or C.MOCK
row = dict(test=f"p2 REDUCE_FP32 A/B ({' vs '.join(map(str, VALUES))}): TF proxy (2,016 + i < {GEN}, L0-L4 CPU fp32 "
                f"refpt), 32-token collapse + prefix CPU fp32 logits, A97 / A300 solo vs packed, 2,048 replay ms; "
                f"tp{TPN} layers {len(LAYERS)}", milestone="P2", status="pass" if ok else "fail",
           traced=all(r_["traced"] for r_ in res.values()), tp=TPN, layers=len(LAYERS), by_value=res, cross=cross,
           mock=C.MOCK, tag=tag)  # fmt: skip
S.log({k_: v_ for k_, v_ in row.items() if k_ != "by_value"})
if rec and not C.MOCK:
    C.record("tests", row)
RESULT = [row]
