# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP pair): the routed-MoE ROW GROUPS A/B (program 2026-10-22 18:39Z: 9's 8,192 prefill bucket OOMed
on EpMoe's [cap, 4096] buffers; EpMoe now splits rows above GLM53_PF_MOE_GROUP_ROWS into groups inside the replay).
One build (buckets: the big one only), one prompt (n_valid < bucket: padding rows included), the SAME bucket traced twice:
  A  group_rows = 0     (the whole bucket in one routed-MoE pass: today's behaviour)
  B  group_rows = GROUP (row groups)
on separate slots; both configurations are warmed (eager compile forwards) BEFORE the first capture. Compared: the last
row's logits, the first token, every KDA layer's S / C, the sparse_dsa layers' latent rows -- BITWISE expected (per-row
math is unchanged). what = "reduce" (program 19:39Z): the same A/B for the sequence-parallel REDUCE row groups
(GLM53_PF_REDUCE_ROWS; the MoE groups stay at their default). ARGS: layers ([0..4]), bucket (8192), n (8000), group
(2048), what ("moe" | "reduce"), env (knobs), tag, record (True)."""
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
BUCKET = int(args.get("bucket", 8192))
N = int(args.get("n", 8000))
GROUP = int(args.get("group", 2048))
WHAT = args.get("what", "moe")
assert WHAT in ("moe", "reduce"), WHAT
tag = args.get("tag", "p2-moe-groups-ab")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GOLD = "/opt/glm53/kit/goldens"
prompt = torch.load(f"{GOLD}/prompts/32768.pt", map_location="cpu", weights_only=True).reshape(-1).long()[:N]

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
os.environ["GLM53_PF_MOE_GROUP_ROWS" if WHAT == "moe" else "GLM53_PF_REDUCE_ROWS"] = str(GROUP)
# an n > 2,050 chunk at position 0 is a SPARSE chunk (DSA): its window key (bucket, bucket) must be captured too
os.environ["GLM53_PF_SPARSE_T"] = str(int(args.get("window", BUCKET)))
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=3, buckets=(BUCKET,), max_seq=BUCKET + 512)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf
pf.sparse_ts = (int(args.get("window", BUCKET)),)
GREEDY = SamplingParams()


def set_groups(v: int):
    if WHAT == "reduce":
        os.environ["GLM53_PF_REDUCE_ROWS"] = str(v)
        pf.reduce_rows = v
        return 1
    os.environ["GLM53_PF_MOE_GROUP_ROWS"] = str(v)  # EpMoe objects built later read it
    n_ = 0
    for pl in pf.layers:
        ep = getattr(pl, "epmoe", None)
        if ep is not None:
            ep.group_rows = v
            n_ += 1
    return n_


def run(slot):
    f = backend.prefill_chunk(prompt, kv_cache=kv, slot=slot, chunk_start=0, n_valid=N, logits_rows="last", io=io,
                              sample=GREEDY)  # fmt: skip
    return int(f), backend.read(io.logits)[slot].float().clone()


def snapshot(slot):
    host = lambda t_: ttnn.to_torch(ttnn.get_device_tensors(t_)[0] if TPN > 1 else t_).float()  # noqa: E731
    lat = {L: host(t_) for L, t_ in backend.latent_pages(kv).items()}
    row = pf.page_row(slot)
    kda = {e["L"]: (e["S"].float(), e["C"].float()) for e in backend.export_kda_state(kv, slot)}
    lr = {L: torch.cat([c_[int(row[p]), 0] for p in range(-(-N // 64))], dim=0)[:N] for L, c_ in lat.items()}
    return kda, lr


# warm BOTH configurations before any capture (a program first compiled after a capture keeps a live buffer)
t0 = time.time()
for v in (GROUP, 0):
    set_groups(v)
    pf.compile((BUCKET,))  # the bucket's eager compile forward (inputs written by compile) under this config
run(0)  # the request path's host-side helper programs, eager, before the captures
ttnn.synchronize_device(mesh)
S.log(f"warm (both configs) {time.time() - t0:.1f} s")
res = {}
# A twice (the CONTROL: run-to-run bitwise of the whole-bucket config) and B
for name, v, slot in (("A_off", 0, 0), ("A_off2", 0, 2), ("B_groups", GROUP, 1)):
    n_ep = set_groups(v)
    pf.release()  # the previous config's traces
    if not C.MOCK:
        pf.capture((BUCKET,))
    eager0 = pf.counters["eager"]
    first, lg = run(slot)
    res[name] = dict(first=first, logits=lg, traced=pf.counters["eager"] == eager0, n_ep=n_ep, snap=snapshot(slot))
    S.log(f"{name}: {WHAT} group_rows {v} ({n_ep} objects), first token {first}, traced {res[name]['traced']}")
a, b, a2 = res["A_off"], res["B_groups"], res["A_off2"]
control_bitwise = bool(torch.equal(a["logits"], a2["logits"]))
control_max_abs = round(float((a["logits"] - a2["logits"]).abs().max()), 6)
diffs = []
for L in sorted(a["snap"][0]):
    for nm, i in (("S", 0), ("C", 1)):
        u, w = a["snap"][0][L][i], b["snap"][0][L][i]
        diffs.append((f"L{L}.{nm}", float((u - w).abs().max())))
for L in sorted(a["snap"][1]):
    u, w = a["snap"][1][L], b["snap"][1][L]
    diffs.append((f"L{L}.lat", float((u - w).abs().max())))
lg_bitwise = bool(torch.equal(a["logits"], b["logits"]))
ok = lg_bitwise and all(m == 0.0 for _, m in diffs) and a["first"] == b["first"]
row = dict(test=f"p2 {'MoE' if WHAT == 'moe' else 'REDUCE'} ROW GROUPS A/B: bucket {BUCKET} (n {N}) whole-bucket "
                f"{'routed MoE' if WHAT == 'moe' else 'sequence-parallel reduce'} vs {GROUP}-row groups in the same "
                f"replay, tp{TPN} layers {len(LAYERS)}: logits, first token, KDA S / C, latents", milestone="P3", what=WHAT,
           status=("pass" if ok and (a["traced"] and b["traced"] or C.MOCK) else "fail"), traced=a["traced"] and b["traced"],
           control_bitwise=control_bitwise, control_max_abs=control_max_abs,
           tp=TPN, layers=len(LAYERS), bucket=BUCKET, n=N, group=GROUP, first=[a["first"], b["first"]],
           logits_bitwise=lg_bitwise, logits_max_abs=round(float((a["logits"] - b["logits"]).abs().max()), 6),
           state_max_abs={k_: v_ for k_, v_ in diffs}, n_epmoe=b["n_ep"], mock=C.MOCK, tag=tag)  # fmt: skip
S.log(row)
if rec and not C.MOCK:
    C.record("tests", row)
RESULT = [row]
