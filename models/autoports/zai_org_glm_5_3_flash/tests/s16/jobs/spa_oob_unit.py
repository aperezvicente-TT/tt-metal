# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, pair or 2x4): SpaUnion fed OUT-OF-WINDOW input (14b's realgate7 hang, 2026-10-25 12:01Z: after compile()
the bucket's chunk inputs held another key's chunk start; the (2048, 2048) key's warm forward fed ids far past its window
and spa_um indexed L1 out of bounds). The kernel must drop such input (no bitmap bit, compact row 0), count it (the row
lane's DROP word, SpaUnion.status()) and complete. Selections: random, expand_rows-shaped (row position p = start + c R + r:
min(n_closed, TOPK) closed pools + the open pool's tokens).
  token mode (pools 0): the first `oob_rows` rows of every chip carry ids + 4 T (past the window);
  POOLS mode (pools 1): the ODD chips get a stale chunk start (start + 4 T) and their selections + T (a later chunk's
  inputs: everything past the window, as realgate7); the even chips' first `oob_rows` rows select pools + T (>= n_closed:
  the expand_rows rule maps them to the row's own pool -- in the window).
Checks (exact): ids_new = the host mirror of the kernel's rule, the compact rows = the in-window union's latents, the
DROP count = launches x the mirror's per-launch drops. Traced: compile launch, capture, replays; eager guard.
The 1M bitmap item (program 2026-10-25 19:14:30Z): the row also carries the L1_REPORT lowest block / bytes before the
SpaUnion and after its compile launch (the persistent L1 it adds), the SPA program's CB bytes per role (descriptor sums),
and sha1s of ids_new / compact per chip -- the same job on two trees is the bitwise A/B.
ARGS: window (2048), rows (2048), oob_rows (8), pools (0 / 1), replays (20), seed, l1 (1), tag, record."""
import hashlib
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spa_union import SpaUnion  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import l1_report  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
T = int(args.get("window", 2048))
CR = int(args.get("rows", 2048))
OOB = int(args.get("oob_rows", 8))
POOLS = bool(int(args.get("pools", 0)))
REPLAYS = int(args.get("replays", 20))
L1 = bool(int(args.get("l1", 1)))
tag = args.get("tag", f"spa-oob-T{T}-{'pools' if POOLS else 'tok'}")
N = int(mesh.get_num_devices())
R = CR // N
W, TOPK, UNIT, D, K4 = PF.SPARSE_W, PF.TOPK_POOLS, 128, 512, 4
P = T // K4  # the window's pools (spa_um NPOOLS = T / 128 words x 32)
M32 = 0xFFFFFFFF
owner = lambda t: (t // UNIT) % N  # noqa: E731
lrow = lambda t: (t // UNIT // N) * UNIT + t % UNIT  # noqa: E731
g = torch.Generator().manual_seed(int(args.get("seed", 3)))
kv = torch.randn(T, D, generator=g).to(torch.bfloat16)
lat = torch.zeros(N, 1, T // N, D, dtype=torch.bfloat16)
tok = torch.arange(T)
for c in range(N):
    m = owner(tok) == c
    lat[c, 0, lrow(tok[m])] = kv[m]
start = T - CR  # the chunk = the window's last CR rows
stale = [POOLS and c % 2 == 1 for c in range(N)]
starts = [start + (4 * T if stale[c] else 0) for c in range(N)]

sel = torch.full((N, R, TOPK), -1, dtype=torch.int64)  # POOLS input
ids = torch.full((N, R, W), -1, dtype=torch.int64)  # token input
for c in range(N):
    for r in range(R):
        p = start + c * R + r
        n_closed = (p + 1) // K4
        nsel = min(n_closed, TOPK)
        s_ = torch.randperm(n_closed, generator=g)[:nsel]
        row = torch.cat([(K4 * s_.reshape(-1, 1) + torch.arange(K4)).reshape(-1), torch.arange(K4 * n_closed, p + 1)])
        if not POOLS and r < OOB:
            row = row + 4 * T
        if POOLS and (stale[c] or r < OOB):
            s_ = s_ + T
        sel[c, r, :nsel] = s_
        ids[c, r, :row.numel()] = row


cap = min(P, R * (TOPK + 1))


def mirror(c):
    """spa_um.cpp on chip c's rows -> (union pools, ids_new [R, W] (uint32 values), drops per launch)"""
    drops = 0
    bits = torch.zeros(P + 1, dtype=torch.bool)  # [P] = the sink of dropped pools (not counted as a bit)
    if POOLS:
        geo = []
        for r in range(R):
            p = starts[c] + c * R + r
            n_closed = (p + 1) // K4
            nsel = min(n_closed, TOPK)
            s_ = sel[c, r, :nsel] & M32
            isp = s_ < n_closed
            pool = torch.where(isp, s_, torch.full_like(s_, p >> 2))
            if K4 * n_closed <= p:
                pool = torch.cat([pool, torch.tensor([n_closed])])
            drops += int((pool >= P).sum())
            bits[pool.clamp(max=P)] = True
            geo.append((p, n_closed, s_, isp))
    else:
        v = ids[c]
        valid = v >= 0
        inw = valid & ((v >> 2) < P)
        drops = int((valid & ~inw).sum())
        bits[(v[inw] >> 2)] = True
    bits[P] = False
    pools_ = torch.nonzero(bits).reshape(-1)
    rank = torch.cumsum(bits.long(), 0) - bits.long()
    rank = torch.where(rank < cap, rank, torch.zeros_like(rank))

    def crow(t):  # t: int64 tensor of token ids (uint32 values)
        pl = t >> 2
        return torch.where(pl < P, 4 * rank[pl.clamp(max=P)] + (t & 3), torch.zeros_like(t))

    if not POOLS:
        out = torch.full((R, W), M32, dtype=torch.int64)
        out[valid] = crow(v[valid])
        return pools_, out, drops
    out = torch.empty((R, W), dtype=torch.int64)
    ar = torch.arange(K4)
    for r, (p, n_closed, s_, isp) in enumerate(geo):
        cp = crow(torch.tensor([p]))
        ent = torch.where(isp.reshape(-1, 1), crow(K4 * torch.where(isp, s_, torch.zeros_like(s_))).reshape(-1, 1) + ar,
                          cp.expand(K4).reshape(1, K4)).reshape(-1)  # fmt: skip
        t = K4 * n_closed + torch.arange(W - ent.numel())
        out[r] = torch.cat([ent, torch.where(t <= p, crow(t), torch.full_like(t, M32))])
    return pools_, out, drops


unions, ref_new, drops = zip(*[mirror(c) for c in range(N)])
shard = ttnn.ShardTensorToMesh(mesh, dim=0)
rep = ttnn.ReplicateTensorToMesh(mesh)
DRAM = ttnn.DRAM_MEMORY_CONFIG
put = lambda t, dt, mp: ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=mp)  # noqa: E731
d_lat = put(lat, ttnn.bfloat16, shard)
d_in = put((sel if POOLS else ids).reshape(N, 1, R, -1).to(torch.int32), ttnn.uint32, shard)
d_new = put(torch.zeros(N, 1, R, W, dtype=torch.int32), ttnn.uint32, shard)
d_cmp = put(torch.zeros(1, 1, 4 * cap, D, dtype=torch.bfloat16), ttnn.bfloat16, rep)
d_start = put(torch.tensor(starts, dtype=torch.int32).reshape(N, 1), ttnn.int32, shard)
d_base = put(torch.tensor([c * R for c in range(N)], dtype=torch.int32).reshape(N, 1), ttnn.int32, shard)
l1b = l1_report(mesh, f"{tag} before SpaUnion") if L1 else []
spu = SpaUnion(mesh, d_cmp, R, T, TOPK if POOLS else W, cap=cap, **(dict(pools=True, width_out=W) if POOLS else {}))  # (token mode:
# also the pre-POOLS SpaUnion of 14b's realgate7 tree)
kw = dict(start=d_start, base=d_base) if POOLS else {}
fn = lambda: spu(d_lat, d_in, d_new, **kw)  # noqa: E731
S.log(f"{tag}: mesh {tuple(mesh.shape)} T {T} rows {CR} ({R} per chip) oob_rows {OOB} pools {POOLS} stale chips "
      f"{[c for c in range(N) if stale[c]]}; mirror union pools {[int(u.numel()) for u in unions]} drops per launch {list(drops)}")  # fmt: skip
fn()  # the compile launch
ttnn.synchronize_device(mesh)
l1a = l1_report(mesh, f"{tag} after SpaUnion + its compile launch") if L1 else []


def cb_bytes():
    """the SPA program's CB bytes per role (descriptor sums on chip (0, 0)): lane, column lane, member 0, member 1"""
    try:
        prog = spu.program(d_lat, d_in, d_new, 0, 0, **kw)
        roles = {"lane": spu.lane, "col_lane": spu.col_lane, "member0": spu.members[0], "member1": spu.members[1]}
        out = {}
        for name, c in roles.items():
            x, y = int(c[0]) if isinstance(c, tuple) else int(c.x), int(c[1]) if isinstance(c, tuple) else int(c.y)
            out[name] = sum(int(d.total_size) for d in prog.cbs if d.core_ranges.contains(ttnn.CoreCoord(x, y)))
        return out
    except Exception as e:  # noqa: BLE001 (a report, never a failure)
        return f"n/a ({type(e).__name__}: {e})"


row = dict(test=f"SpaUnion with out-of-window input (window {T}, {'POOLS: stale chunk start on the odd chips' if POOLS else f'{OOB} rows per chip past it'})",
           milestone="P3", window=T, rows=CR, tp=N, oob_rows=OOB, pools=POOLS, stale_chips=[c for c in range(N) if stale[c]],
           union_pools=[int(u.numel()) for u in unions], drops_per_launch=list(drops), tag=tag,
           l1_lowest_before=min((r["address"] for r in l1b), default=None), l1_bytes_before=sum(r["size"] for r in l1b),
           l1_lowest_after=min((r["address"] for r in l1a), default=None), l1_bytes_after=sum(r["size"] for r in l1a),
           spa_l1_persistent=sum(r["size"] for r in l1a) - sum(r["size"] for r in l1b), spa_cb_bytes=cb_bytes(),
           status="mock" if C.MOCK else "info")  # fmt: skip
tid = None
if not C.MOCK:
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
else:
    fn()  # the mock runs no trace: the path once more for its allocations
assert spu.calls == 2, ("eager guard / capture count", spu.calls)
if C.MOCK:
    got = [torch.zeros(R, W, dtype=torch.int64)] * N
    cc = [torch.zeros(4 * cap, D)] * N
    st = None
else:
    got = [ttnn.to_torch(t_).reshape(R, W).to(torch.int64) & M32 for t_ in ttnn.get_device_tensors(d_new)]
    cc = [ttnn.to_torch(t_).reshape(4 * cap, D) for t_ in ttnn.get_device_tensors(d_cmp)]  # bf16 (exact compare)
    st = spu.status()  # after 2 launches (compile + the first replay)
ids_ok = [bool(torch.equal(got[c], ref_new[c])) for c in range(N)]
cmp_ok = []
for c in range(N):
    exp = kv[(unions[c].reshape(-1, 1) * 4 + torch.arange(4)).reshape(-1)]
    cmp_ok.append(bool(torch.equal(cc[c][:exp.shape[0]].to(exp.dtype), exp)))
sha = lambda t: hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]  # noqa: E731
row.update(ids_new_exact=ids_ok, compact_rows_exact=cmp_ok, sha_ids_new=[sha(got[c].to(torch.int64)) for c in range(N)],
           sha_compact=[sha(cc[c].to(torch.bfloat16)) for c in range(N)])  # fmt: skip
if st is not None:
    drop_ok = [st[c][2] == 2 * drops[c] for c in range(N)]
    row.update(status_after_2=st, drops_exact=drop_ok, overflow_free=[st[c][0] == 0 and st[c][1] == 0 for c in range(N)])
    if not all(ids_ok):
        bad = [(c, int((got[c] != ref_new[c]).sum())) for c in range(N) if not ids_ok[c]]
        row["ids_defects"] = bad
    perf = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} union")
    st2 = spu.status()
    row.update(us=perf.get("median_us"), replays=perf.get("replays"),
               drops_multiple=[drops[c] == 0 and st2[c][2] == 0 or drops[c] > 0 and st2[c][2] % drops[c] == 0 for c in range(N)])  # fmt: skip
    assert spu.calls == 2, ("eager guard: the union launched outside the trace", spu.calls)
    ttnn.release_trace(mesh, tid)
    row["status"] = "pass" if all(ids_ok) and all(cmp_ok) and all(drop_ok) and all(row["drops_multiple"]) else "fail"
S.log(f"{tag}: {row}")
if args.get("record", False):
    C.record("pcc", row)
RESULT = [row]
print("SPA_OOB_UNIT", row, flush=True)
