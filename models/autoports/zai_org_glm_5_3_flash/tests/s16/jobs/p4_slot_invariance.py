# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair / mesh): SLOT INVARIANCE of the TP prefill's state hand-off (program 2026-10-23 12:08:34Z,
with 14b: a 128-token prompt in slot 4 decodes degenerately on every served line, 10-13 nats off fp32). The SAME prompt
(golden2048[:n]) is prefilled ALONE into each slot of `slots` (single-request path, StackBackend.prefill_chunk, one
after another, every other decode row idle), then the slots decode `decode_steps` greedy steps together through the
captured width-B decode trace (inputs as the generator sets them: token = the slot's first token, cur_pos = n). Per slot
vs the first slot of the list: the prefill's last-row logits and first token, the slot's exported KDA state (S / C of
every KDA layer: what state_io wrote) and, per decode step, the sampled token + the logits row -- all bitwise. A
slot-dependent hand-off (KDA state row, latent pages, DSA ring / pools, positions) shows as the first slot that differs
and the first thing that differs. Everything traced (eager = the one-time compile before the captures).
ARGS: layers ([0..4] | "all"), n (128), buckets ("512,2048"), slots ([0, 4, 7]), decode_steps (8), b (8), env, tag,
record (True)."""
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
LAYERS = list(range(int(args.get("n_layers", 45)))) if args.get("layers") == "all" else [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = int(args.get("n", 128))
TS = tuple(int(v) for v in str(args.get("buckets", "512,2048")).split(","))
SLOTS = [int(v) for v in args.get("slots", [0, 4, 7])]
DSTEPS = int(args.get("decode_steps", 8))
B = int(args.get("b", 8))
assert max(SLOTS) < B, (SLOTS, B)
tag = args.get("tag", "p4-slot-invariance")
rec = bool(args.get("record", True))
TPN = int(mesh.get_num_devices())
GREEDY = SamplingParams()
GOLD = "/opt/glm53/kit/goldens"
prompt = torch.load(f"{GOLD}/prompts/2048.pt", map_location="cpu", weights_only=True).reshape(-1).long()

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
for k_, v_ in (args.get("env") or {}).items():
    os.environ[str(k_)] = str(v_)
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_REV"] = "slotinv"
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, **args}
S.set_attn(args)
ent = S.build(mesh, ctx, LAYERS, b=B, buckets=TS, max_seq=max(TS) + 512 + 64)
pf, kv, io, backend = ent["pf"], ent["kv"], ent["io"], ent["backend"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf


def rows():
    return [int(v) for v in backend.read(io.tokens).reshape(-1)[:B]], [int(v) for v in backend.read(io.cur_pos).reshape(-1)[:B]]


def prefill(slot):
    f = backend.prefill_chunk(prompt[:N], kv_cache=kv, slot=slot, chunk_start=0, n_valid=N, logits_rows="last", io=io,
                              sample=GREEDY)  # fmt: skip
    return int(f), backend.read(io.logits)[slot].float().clone()


def kda(slot):
    return {e["L"]: (e["S"].float().clone(), e["C"].float().clone()) for e in backend.export_kda_state(kv, slot)}


# ---- compile: every prefill program + the decode step + the host row helpers, eagerly, before any capture
t0 = time.time()
pf.compile(TS)  # every prefill program (this tree replays captured keys only: no eager prefill_chunk before capture)
for s in SLOTS:
    kda(s)  # the export programs
rows()
backend.write(io.tokens, backend.read(io.tokens))
backend.write(io.cur_pos, backend.read(io.cur_pos))
backend.prepare_decode(io, kv, [B])
dtrace = backend.capture_decode(io, kv, B)  # the decode trace first (its capture compiles helpers of its own)
if not C.MOCK:
    pf.capture(TS)
S.log(f"{tag}: compiled + captured in {time.time() - t0:.1f} s; traces {sorted(map(str, pf.traces))}")

# ---- the measured pass (traced): each slot alone, then the slots' decode together
eager0 = pf.counters["eager"]
backend.write(io.cur_pos, torch.full_like(backend.read(io.cur_pos), -1))
pre = {}
for s in SLOTS:
    first, lg = prefill(s)
    pre[s] = dict(first=first, logits=lg, kda=kda(s))
traced = pf.counters["eager"] == eager0
tok0 = backend.read(io.tokens).clone()
pos0 = torch.full_like(backend.read(io.cur_pos), -1)
for s in SLOTS:
    tok0.reshape(-1)[s] = pre[s]["first"]
    pos0.reshape(-1)[s] = N
backend.write(io.tokens, tok0)
backend.write(io.cur_pos, pos0)
steps = []
for i in range(DSTEPS):
    if dtrace is not None:
        backend.replay(dtrace)
    ttnn.synchronize_device(mesh)
    tk = [int(v) for v in backend.read(io.tokens).reshape(-1)[:B]]
    lg = backend.read(io.logits)
    steps.append(dict(tokens={s: tk[s] for s in SLOTS}, logits={s: lg[s].float().clone() for s in SLOTS}))
pos_end = [int(v) for v in backend.read(io.cur_pos).reshape(-1)[:B]]

ref = SLOTS[0]
per_slot = []
for s in SLOTS[1:]:
    kd = {}
    for L in pre[ref]["kda"]:
        for nm, j in (("S", 0), ("C", 1)):
            u, v = pre[s]["kda"][L][j], pre[ref]["kda"][L][j]
            kd[f"L{L}.{nm}"] = round(float((u - v).abs().max()), 6)
    first_kda = next((k for k, v in kd.items() if v != 0.0), None)
    st_div = next((i for i, st in enumerate(steps) if st["tokens"][s] != st["tokens"][ref]
                   or not torch.equal(st["logits"][s], st["logits"][ref])), None)  # fmt: skip
    per_slot.append(dict(slot=s, first=pre[s]["first"], first_equal=pre[s]["first"] == pre[ref]["first"],
                         prefill_logits_bitwise=bool(torch.equal(pre[s]["logits"], pre[ref]["logits"])),
                         prefill_logits_max_abs=round(float((pre[s]["logits"] - pre[ref]["logits"]).abs().max()), 6),
                         kda_bitwise=first_kda is None, kda_first_differing=first_kda,
                         kda_max_abs=max(kd.values()) if kd else 0.0,
                         decode_first_divergent_step=st_div,
                         decode_tokens=[st["tokens"][s] for st in steps],
                         decode_logits_max_abs=[round(float((st["logits"][s] - st["logits"][ref]).abs().max()), 6) for st in steps]))  # fmt: skip
ok = all(p["first_equal"] and p["prefill_logits_bitwise"] and p["kda_bitwise"] and p["decode_first_divergent_step"] is None
         for p in per_slot)  # fmt: skip
row = dict(test=f"p4 SLOT invariance of the TP prefill hand-off: golden2048[:{N}] prefilled alone into slots {SLOTS} "
                f"(buckets {list(TS)}), last-row logits + KDA state + {DSTEPS} greedy decode steps bitwise vs slot {ref}, "
                f"traced, tp{TPN} layers {len(LAYERS)}", milestone="P4",
           status=("pass" if ok and (traced or C.MOCK) else "fail"), traced=traced, tp=TPN, layers=len(LAYERS), n=N,
           slots=SLOTS, ref_first=pre[ref]["first"], ref_decode_tokens=[st["tokens"][ref] for st in steps],
           positions_end={s: pos_end[s] for s in SLOTS}, per_slot=per_slot, mock=C.MOCK, tag=tag)  # fmt: skip
S.log({k_: v_ for k_, v_ in row.items() if k_ != "per_slot"} | {"per_slot": [{k: v for k, v in p.items() if k not in ("decode_logits_max_abs",)} for p in per_slot]})
if rec and not C.MOCK:
    C.record("tests", row)
RESULT = [row]
