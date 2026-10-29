# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair; A lever 2, program 19:18:59Z): program A of layer L in the default scopes (5
launches: [P1] [P2 P3] [P4 P6] [P7 P8] [P10]) vs PMK_A_FOLD=1 (3 launches: [P1 P2 P3] [P4 P6] [P7 P8 P10]) on ONE backend,
same inputs, two PmkProgramA instances:
  * bitwise: every output program A writes (x-hat / y / x1 / x-hat_f / S / C / the A2 stores) per chip, base vs fold
    (the same phases in the same order: equal by construction);
  * the per-scope launch census (CB end per role vs the ceiling, pmk_layer.CB_END_CEILING) of each instance;
  * replay time per mode (median of `replays`).
Every program of both instances compiled before the first capture; one live trace at a time.
ARGS: layer (4), rows (2048), replays (50), a_inplace (True), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_layer as PMK  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 50))
tag = args.get("tag", f"pmk-fold-ab-L{L}")
NDEV = int(mesh.get_num_devices())
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = "fused"
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ""
S.set_attn(args)
ent = S.build(mesh, ctx, [L], b=1, buckets=(N,), max_seq=N + 512)
pf = ent["pf"]
pl = pf.layers[0]
c = pf.bucket_for(N)
sp = pf.mode(c) == "sp"
ci = pf.chunk_inputs(c)
pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
rows, _ = S.golden_layer(L, N)
rows = rows.reshape(N, 4, 4096).float()
x0 = pf.streams_to_device(rows, sp)
x_p = pf.streams_to_device(rows, sp)
state_p = pl.kda.allocate_state(1)
ttnn.copy(ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask)
progs = {}
for mode, fold in (("base", "0"), ("fold", "1")):
    os.environ["PMK_A_FOLD"] = fold  # read at construction (the scopes are fixed there)
    progs[mode] = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, probe=False,
                                  a_inplace=bool(args.get("a_inplace", True)), pfs_park=int(args.get("pfs_park", 4)))  # fmt: skip
os.environ["PMK_A_FOLD"] = "0"
_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(_zeros[key], t_)


def fn(prog):
    ttnn.copy(x0, x_p)
    zero(state_p.S)
    zero(state_p.C)
    prog(x_p, ci)


per_chip = lambda t_: [ttnn.to_torch(d).float().clone() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731


def outputs(prog):
    out = dict(xhat=per_chip(prog.hns[0]), y=per_chip(prog.ys[0]), x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]),
               S=per_chip(state_p.S), C=per_chip(state_p.C))  # fmt: skip
    for k_, t_ in prog.stores().items():
        out[f"store_{k_}"] = per_chip(t_)
    return out


t0 = time.time()
for mode, prog in progs.items():  # every program of both instances compiled before the first capture
    fn(prog)
    ttnn.synchronize_device(mesh)
census = {m: dict(getattr(p_.layer, "launch_census", {}) or {}) for m, p_ in progs.items()}
over = {m: {sc: {r: v for r, v in d.items() if int(v) > PMK.CB_END_CEILING} for sc, d in cs.items()} for m, cs in census.items()}
over = {m: {sc: d for sc, d in cs.items() if d} for m, cs in over.items()}
S.log(f"{tag}: scopes base {progs['base'].scopes} fold {progs['fold'].scopes}; census {census}; over the ceiling "
      f"({PMK.CB_END_CEILING}) {over}; compiled in {time.time() - t0:.1f} s")  # fmt: skip
row = dict(test=f"PMK program A L{L} tp{NDEV}: PMK_A_FOLD (A1 into A2, A5 into A4: 5 -> 3 launches) vs the default scopes "
                f"(bitwise per chip, the per-scope launch census, replay us)", milestone="PMK-A-fold", layer=L, tp=NDEV, bucket=c,
           scopes={m: [list(s_) for s_ in p_.scopes] for m, p_ in progs.items()}, census=census, over_ceiling=over,
           ceiling=PMK.CB_END_CEILING, mock=C.MOCK, tag=tag)  # fmt: skip
if C.MOCK:
    row["status"] = "mock" if not over["fold"] else "fail"
else:
    res = {}
    for mode, prog in progs.items():
        n0 = int(mesh.num_program_cache_entries())
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn(prog)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        assert int(mesh.num_program_cache_entries()) == n0, f"{mode}: a program compiled inside the capture"
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        outs = outputs(prog)
        st = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} {mode}")
        ttnn.release_trace(mesh, tid)
        res[mode] = dict(outs=outs, us=st["median_us"])
        S.log(f"{tag}: {mode}: {len(prog.scopes)} launches, replay {st['median_us']} us")
    cmp = {k_: [bool(torch.equal(x_, y_)) for x_, y_ in zip(res["base"]["outs"][k_], res["fold"]["outs"][k_])] for k_ in res["base"]["outs"]}
    bit = all(all(v) for v in cmp.values())
    row.update(bitwise=bit, bitwise_per_output=cmp, us={m: res[m]["us"] for m in res},
               saved_us=round(res["base"]["us"] - res["fold"]["us"], 1), status="pass" if bit else "fail")  # fmt: skip
    if bool(args.get("record", True)):
        C.record("tests", {k: v for k, v in row.items() if k != "bitwise_per_output"})
S.log(f"{tag}: RESULT {row}")
RESULT = row
