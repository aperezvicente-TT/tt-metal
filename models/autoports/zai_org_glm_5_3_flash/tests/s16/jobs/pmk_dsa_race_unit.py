# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair): the sparse_dsa layer's A1 / A5 (GLM53_PMK_DSA_MHC=1) hand-off defect (program
2026-10-23 16:54:29Z ruling b: 14b saw L3 0.99937 vs 0.99984 for one config). Two causes give that signature and
this unit separates them, entirely under trace (the chain's captured forward, replayed):
  * COVERAGE (an uninitialised read): before every replay the DSA layer's hand-off tensors -- A1's hn[0] / posts[0] /
    combs[0], A5's hn[1] / posts[1] / combs[1], the y landing ys[0] -- are filled with a SENTINEL (host writes, outside
    the capture); after the replay every element still equal to the sentinel is a tile the producer never wrote (and
    every consumer of it read the sentinel: the residual then breaks at that layer);
  * RACE: the replay is repeated `reps` times on identical inputs; every probed layer's output residual and the
    logits must be BITWISE equal across replays (a timing race moves them).
Also reported per replay: the residual PCC vs the all-op chain (GLM53_PF_MODE=op, the same TPPrefill, same inputs).
ARGS: layers ([0..4]), rows (2048), reps (10), env ({}), b_* (program B levers), dump (a .pt path: rep 0's probes +
logits, for a bitwise compare across heads), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

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
REPS = int(args.get("reps", 10))
tag = args.get("tag", f"pmk-dsa-race-L{LAYERS[0]}-{LAYERS[-1]}-tp{TPN}")
rec = bool(args.get("record", True))
import pmk45_alias as A45  # noqa: E402

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(v) for v in LAYERS)
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PMK_DSA_MHC"] = "1"
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
SENT = {ttnn.bfloat16: 3.0e38, ttnn.float32: 3.0e38}  # a finite value no layer produces

ent = S.build(mesh, ctx, LAYERS, b=2, buckets=(N,), max_seq=N + 512)
pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf
pf.probe_layers = set(LAYERS)
A45.publish_page_table(backend, kv, io, pf, rows=2)
c = pf.bucket_for(N)
prompt = S.golden_prompt(N).reshape(-1).long()
opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
opts.update({k: v for k, v in args.items() if k.startswith("b_")})
os.environ["GLM53_PF_MODE"] = "pmk"
stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
dsa = [(L_, stack.progs[(c, L_)]) for _, L_, kind in stack.layers if kind == PS.SPARSE_DSA]
assert dsa and all(sl.a is not None for _, sl in dsa), ("a sparse_dsa layer with the mhc-only program A", [k for _, _, k in stack.layers])


def handoffs(sl):
    a = sl.a
    out = {"hn0": a.hns[0], "hn1": a.hns[1], "ys0": a.ys[0]}
    for i_, t_ in enumerate(getattr(a, "posts", []) or []):
        out[f"post{i_}"] = t_
    for i_, t_ in enumerate(getattr(a, "combs", []) or []):
        out[f"comb{i_}"] = t_
    return out


def fill(t_, v):
    """host write of the sentinel into a device tensor (outside any capture)"""
    shp = tuple(int(d) for d in t_.shape)
    full = torch.full((TPN * shp[0],) + shp[1:] if TPN > 1 else shp, float(v), dtype=torch.float32)
    host = ttnn.from_torch(full, dtype=t_.dtype, layout=t_.layout,
                           mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if TPN > 1 else None)  # fmt: skip
    ttnn.copy_host_to_device_tensor(host, t_)


def count_sent(t_, v):
    parts = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
    return [int((ttnn.to_torch(p_).float() >= 0.5 * v).sum()) for p_ in parts]


def forward_capture(chain_on: bool):
    if chain_on:
        os.environ["GLM53_PF_MODE"] = "pmk"
        stack.attach()
    else:
        stack.detach()
        os.environ["GLM53_PF_MODE"] = "op"
    pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        pf.forward(c)
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    return tid


def outputs():
    probes = {L_: pf.probe(c, L_).float().clone() for L_ in LAYERS}
    lg = backend.read(io.logits)[0:1].float().clone() if not C.MOCK else torch.zeros(1, 8)
    return probes, lg


t0 = time.time()
# every program compiled before the first capture: the op chain, then the PMK chain (both compile launches)
stack.detach()
os.environ["GLM53_PF_MODE"] = "op"
pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
pf.forward(c)
stack.attach()
os.environ["GLM53_PF_MODE"] = "pmk"
pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
pf.forward(c)
for L_, sl in dsa:  # the sentinel fill's host programs (copy_host_to_device has none) and the readback: warm now
    for nm, t_ in handoffs(sl).items():
        fill(t_, SENT.get(t_.dtype, 3.0e38))
        count_sent(t_, SENT.get(t_.dtype, 3.0e38))
ttnn.synchronize_device(mesh)
H.log(f"{tag}: compiled in {time.time() - t0:.1f} s; DSA layers {[L_ for L_, _ in dsa]}; hand-offs {[list(handoffs(sl)) for _, sl in dsa]}")
row = dict(test=f"PMK sparse_dsa A1 / A5 hand-off defect: coverage (sentinel) + determinism ({REPS} replays) of the L{LAYERS[0]}-L{LAYERS[-1]} chain, tp{TPN}",
           milestone="PMK-B5", tp=TPN, layers=LAYERS, reps=REPS, mock=C.MOCK, tag=tag)  # fmt: skip
if C.MOCK:
    row["status"] = "mock"
else:
    tid_op = forward_capture(False)
    ttnn.execute_trace(mesh, tid_op, cq_id=0, blocking=True)
    p_op, l_op = outputs()
    ttnn.release_trace(mesh, tid_op)
    tid = forward_capture(True)
    reps, first = [], None
    for r_ in range(REPS):
        for L_, sl in dsa:
            for nm, t_ in handoffs(sl).items():
                fill(t_, SENT.get(t_.dtype, 3.0e38))
        pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        p_, l_ = outputs()
        cov = {f"L{L_}.{nm}": count_sent(t_, SENT.get(t_.dtype, 3.0e38)) for L_, sl in dsa for nm, t_ in handoffs(sl).items()}
        pcc = {L_: H.cmp(p_[L_].reshape(N, -1), p_op[L_].reshape(N, -1))["pcc"] for L_ in LAYERS}
        if first is None:
            first = (p_, l_)
            bit = None
            if args.get("dump"):  # rep 0's probed residuals + logits (and the all-op chain's): a cross-HEAD bitwise compare
                torch.save(dict(pmk=p_, logits=l_, op=p_op, op_logits=l_op, layers=LAYERS, code=os.environ.get("GLM53_CODE_SHA")),
                           str(args["dump"]))  # fmt: skip
                H.log(f"{tag}: rep 0 dumped to {args['dump']}")
        else:
            bit = {L_: bool(torch.equal(p_[L_], first[0][L_])) for L_ in LAYERS}
            bit["logits"] = bool(torch.equal(l_, first[1]))
        reps.append(dict(rep=r_, sentinel_left=cov, pcc_vs_op=pcc, bitwise_vs_rep0=bit,
                         logits_pcc_vs_op=H.cmp(l_.reshape(-1), l_op.reshape(-1))["pcc"]))  # fmt: skip
        H.log(f"{tag}: rep {r_}: {reps[-1]}")
    ttnn.release_trace(mesh, tid)
    uncovered = sorted({k for r_ in reps for k, v in r_["sentinel_left"].items() if sum(v) > 0})
    deterministic = all(all(r_["bitwise_vs_rep0"].values()) for r_ in reps[1:])
    row.update(reps=reps, uncovered=uncovered, deterministic=deterministic,
               pcc_min={L_: min(r_["pcc_vs_op"][L_] for r_ in reps) for L_ in LAYERS},
               pcc_max={L_: max(r_["pcc_vs_op"][L_] for r_ in reps) for L_ in LAYERS},
               verdict=("COVERAGE: tiles never written " + str(uncovered)) if uncovered else
                       ("RACE: replays differ" if not deterministic else "covered and deterministic"),
               status="pass" if (not uncovered and deterministic) else "fail")  # fmt: skip
H.log(f"{tag}: RESULT {{k: v for k, v in row.items() if k != 'reps'}}")
H.log({k: v for k, v in row.items() if k != "reps"})
if rec and not C.MOCK:
    C.record("tests", {k: v for k, v in row.items() if k != "reps"})
RESULT = row
