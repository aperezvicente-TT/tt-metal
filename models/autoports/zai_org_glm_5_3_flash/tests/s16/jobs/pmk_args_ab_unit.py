# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, host-p300 pair / 2x4 mesh; trace-shrink ruling 2026-10-23 16:33:48Z item 3): program A of layer L
(A1..A5, the five scoped programs) with its runtime args in the LAUNCH (GLM53_PMK_ARGS=trace) vs in the per-program
DRAM TABLE (GLM53_PMK_ARGS=table: 2-word headers, pmk::load_arg_table) on ONE backend, same inputs:
  * bitwise: every output program A writes (x-hat / y / x1 / x-hat_f / S / C / the A2 stores) per chip, trace vs table
    (the bodies read the same arg values: equal by construction);
  * trace bytes: the trace region in use (MiB per chip) after capturing program A in each mode;
  * replay time per mode (median of `replays`).
Order: both modes' programs compiled (eager compile launches; the table filled there) BEFORE the first capture; then
capture + replay trace mode, release; capture + replay table mode (its build re-checks the rows: no host write inside
the capture), release. ARGS: layer (4), rows (2048), replays (20), a_inplace (True), tag, record (True)."""
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 20))
tag = args.get("tag", f"pmk-args-ab-L{L}")
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
prog = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, probe=False, a_inplace=bool(args.get("a_inplace", True)),
                       pfs_park=int(args.get("pfs_park", 4)))  # fmt: skip
_zeros = {}


def zero(t_):
    key = (tuple(t_.shape), t_.dtype, t_.layout)
    if key not in _zeros:
        _zeros[key] = ttnn.zeros(tuple(t_.shape), dtype=t_.dtype, layout=t_.layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.copy(_zeros[key], t_)


def fn():
    ttnn.copy(x0, x_p)
    zero(state_p.S)
    zero(state_p.C)
    prog(x_p, ci)


per_chip = lambda t_: [ttnn.to_torch(d).float().clone() for d in (ttnn.get_device_tensors(t_) if NDEV > 1 else [t_])]  # noqa: E731


def outputs():
    out = dict(xhat=per_chip(prog.hns[0]), y=per_chip(prog.ys[0]), x1=per_chip(x_p), xhat_f=per_chip(prog.hns[1]),
               S=per_chip(state_p.S), C=per_chip(state_p.C))  # fmt: skip
    for k_, t_ in prog.stores().items():
        out[f"store_{k_}"] = per_chip(t_)
    return out


def trace_mib():
    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return round(int(v.total_bytes_allocated_per_bank) * int(v.num_banks) / 2**20, 3)
    except Exception as ex:  # noqa: BLE001
        return repr(ex)[:80]


# the trace-light levers as modes (env knobs read at every descriptor build): (1) GLM53_PMK_RECT_MERGE, (2) GLM53_PMK_ARGS
MODES = {k: dict(v) for k, v in (args.get("modes") or {
    "base": {"GLM53_PMK_RECT_MERGE": "0", "GLM53_PMK_ARGS": "trace"},
    "rect": {"GLM53_PMK_RECT_MERGE": "1", "GLM53_PMK_ARGS": "trace"},
    "rect_table": {"GLM53_PMK_RECT_MERGE": "1", "GLM53_PMK_ARGS": "table"}}).items()}
modes = list(MODES)


def set_mode(m):
    for k_, v_ in MODES[m].items():
        os.environ[k_] = str(v_)


t0 = time.time()
for m in modes:  # every program of every mode compiled before the first capture (the table filled here)
    set_mode(m)
    fn()
    ttnn.synchronize_device(mesh)
    S.log(f"{tag}: {m} compile launch done; audit rt_bytes {[(k, v.get('rt_bytes', {}).get('total')) for k, v in prog.engine.arg_audits.items()][:6]}")
tables = getattr(prog.engine, "arg_tables", {})
S.log(f"{tag}: arg tables {[(k, v['nrows'], v['addr']) for k, v in tables.items()]}; compiled both modes in {time.time() - t0:.1f} s")
res, row = {}, dict(test=f"PMK program A L{L} tp{NDEV}: the trace-light levers (rect merge / arg table) vs the base build (bitwise per chip, trace MiB, replay us)",
                    milestone="PMK-trace-shrink", layer=L, tp=NDEV, bucket=c, mock=C.MOCK, tag=tag)  # fmt: skip
if C.MOCK:
    row["status"] = "mock"
    row["arg_tables"] = {"+".join(k): dict(rows=v["nrows"], bytes_per_chip=v["nrows"] * 1024) for k, v in tables.items()}
else:
    for m in modes:
        set_mode(m)
        before = trace_mib()
        n0 = int(mesh.num_program_cache_entries())
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        assert int(mesh.num_program_cache_entries()) == n0, f"{m}: a program compiled inside the capture"
        after = trace_mib()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        outs = outputs()
        st = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} {m}")
        ttnn.release_trace(mesh, tid)
        res[m] = dict(outs=outs, trace_mib=(before, after), us=st["median_us"])
        S.log(f"{tag}: {m}: trace MiB {before} -> {after}, replay {st['median_us']} us")
    cmp = {}
    for m in modes[1:]:  # every mode vs the first (base), per output, per chip
        cmp[m] = {k_: [bool(torch.equal(x_, y_)) for x_, y_ in zip(res[modes[0]]["outs"][k_], res[m]["outs"][k_])]
                  for k_ in res[modes[0]]["outs"]}  # fmt: skip
    bit = all(all(v) for d_ in cmp.values() for v in d_.values())
    tm = {m: res[m]["trace_mib"] for m in modes}
    tm_used = {m: (round(a_[1] - a_[0], 3) if isinstance(a_[0], float) and isinstance(a_[1], float) else a_) for m, a_ in tm.items()}
    row.update(bitwise=bit, bitwise_per_mode={m: all(all(v) for v in d_.values()) for m, d_ in cmp.items()}, bitwise_per_output=cmp,
               trace_mib=tm, trace_mib_program_a=tm_used, modes=MODES, us={m: res[m]["us"] for m in modes},
               arg_tables={"+".join(k): dict(rows=v["nrows"], bytes_per_chip=v["nrows"] * 1024) for k, v in tables.items()},
               status="pass" if bit else "fail")  # fmt: skip
    if bool(args.get("record", True)):
        C.record("tests", {k: v for k, v in row.items() if k != "bitwise_per_output"})
S.log(f"{tag}: RESULT {row}")
RESULT = row
