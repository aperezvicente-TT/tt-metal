# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the 2x4 MOCK, host only): the FABRIC-CONNECTION word audit of program A (program 2026-10-23
17:49:57Z, after 14b's watched TP8 hang: the first op fabric all_gather after L0's PMK sequence never completes on mesh
row 0 while every PMK core is done). A lane's fabric connection keeps two words in the LANE core's semaphore space:
the TEARDOWN ack (close_finish waits on it) and the BUFFER-INDEX / producer cursor (open_finish adopts it, close_start
persists it into the EDM for the next opener). setup_fabric_connection allocates both with find_available_semaphore_id
when FabricStream.connect runs (pmk_layer post_build, after the program's own SemaphoreDescriptors), so their ids depend
on how many connections the lane opens (TP8 middle chip 3 -> ids 0..5, end chip 2 -> 0..3, the pair 1 -> 0..1).
Per program A launch (A1..A5) and per mesh rank this job lists:
  * the semaphore ids on every lane core, split into the PMK's own declarations and the fabric connections' (appended
    by the hooks);
  * every declared NoC target of every registered part (part['noc_xy']) and of the engine tails (XS-FREE credits, the
    lane triple) that is a LANE core, with the part's leading runtime words (a pmk_sig part: [flag, sem id, n, ...]);
  * every pair of consecutive declared NoC pairs whose rectangle contains a lane core (multicast corners: PfStream
    PARK / VALID, pmk_agg) -- candidates, the part name says whether the pair is a rectangle.
Nothing launches on a device (the program is built by one eager call under the mock).
ARGS: layer (0), rows (2048), env ({}), tag, record (False)."""
import os
import sys

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp import BF16  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
L = int(args.get("layer", 0))
N = int(args.get("rows", 2048))
tag = args.get("tag", f"pmk-lane-sem-audit-L{L}")
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
x_p = pf.streams_to_device(torch.zeros(N, 4, 4096), sp)
state_p = pl.kda.allocate_state(1)
ttnn.copy(ttnn.ones(tuple(state_p.slot_mask.shape), dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh), state_p.slot_mask)
prog = ASM.PmkProgramA(pf, 0, c, sp, x_p, state_p, ci.valid_end, probe=False, a_inplace=bool(args.get("a_inplace", True)),
                       pfs_park=int(args.get("pfs_park", 4)))  # fmt: skip
E = type(prog.engine)
orig_audit, orig_desc = E.audit_args, E.descriptor
cap = []


def audit_w(self, device, comps, rank):
    self._lane_audit_comps = comps
    return orig_audit(self, device, comps, rank)


def desc_w(self, device, io, rank=0, **kw):
    n_own = len(self.sems)  # the descriptor declares one SemaphoreDescriptor per self.sems entry, then the hooks append
    p_ = orig_desc(self, device, io, rank=rank, **kw)
    scope = "+".join(sorted(self.scope, key=lambda q: int(q[1:]))) if self.scope else "all"
    cap.append(dict(scope=scope, rank=rank, comps=self._lane_audit_comps, prog=p_, n_own=n_own))
    return p_


E.audit_args, E.descriptor = audit_w, desc_w
try:
    prog(x_p, ci)  # builds every program A launch (A1..A5) on every rank; the mock enqueues are no-ops
    ttnn.synchronize_device(mesh)
finally:
    E.audit_args, E.descriptor = orig_audit, orig_desc

P = prog.engine.place
lanes = [tuple(v) for v in P.role_cores()["lane"]]


def noc(xy):
    w = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
    return int(w.x), int(w.y)


lane_noc = {noc(v): v for v in lanes}


def covers(crs, core):
    for r_ in crs.ranges():
        if r_.start.x <= core[0] <= r_.end.x and r_.start.y <= core[1] <= r_.end.y:
            return True
    return False


rows = []
for e_ in cap:
    p_ = e_["prog"]
    sems = list(p_.semaphores)
    lane_ids = {f"{v}": dict(pmk=sorted(int(s.id) for s in sems[: e_["n_own"]] if covers(s.core_ranges, v)),
                             fabric=sorted(int(s.id) for s in sems[e_["n_own"]:] if covers(s.core_ranges, v)))
                for v in lanes}  # fmt: skip
    hits, rects = [], []
    for role, risc, rt_d, ct, layout, tails, src in e_["comps"]:
        for pid, ct0, rt0, part in layout:
            for core, offs in (part.get("noc_xy") or {}).items():
                vals = rt_d[tuple(core)]
                xys = [(int(i), tuple(int(a) for a in vals[rt0 + i : rt0 + i + 2])) for i in offs]
                for i, xy in xys:
                    if xy in lane_noc:
                        hits.append(dict(role=role, risc=risc, pid=pid, part=part.get("name"), core=list(core), arg=rt0 + i,
                                         lane=list(lane_noc[xy]), head=[int(a) for a in vals[rt0 : rt0 + 4]]))  # fmt: skip
                for (i0, a_), (i1, b_) in zip(xys, xys[1:]):
                    lo, hi = (min(a_[0], b_[0]), min(a_[1], b_[1])), (max(a_[0], b_[0]), max(a_[1], b_[1]))
                    inside = [list(v) for xy, v in lane_noc.items() if lo[0] <= xy[0] <= hi[0] and lo[1] <= xy[1] <= hi[1]]
                    if inside and a_ != b_:
                        rects.append(dict(role=role, risc=risc, pid=pid, part=part.get("name"), core=list(core),
                                          args=[rt0 + i0, rt0 + i1], lo=list(lo), hi=list(hi), lanes=inside))  # fmt: skip
        for tname, tv in (tails or {}).items():
            pass  # the XS-FREE tail targets members' lane WORDS (sync tensor), listed by the engine audit itself
    rows.append(dict(scope=e_["scope"], rank=e_["rank"], lane_sem_ids=lane_ids, lane_targets=hits[:40], n_lane_targets=len(hits),
                     rect_candidates=rects[:40], n_rect_candidates=len(rects)))  # fmt: skip
    S.log(f"{tag}: {e_['scope']} rank {e_['rank']}: lane sems {lane_ids}; {len(hits)} lane NoC targets "
          f"{sorted({(h['pid'], h['part'], h['role'], h['risc']) for h in hits})}; {len(rects)} rect candidates "
          f"{sorted({(r_['pid'], r_['part'], r_['role'], r_['risc']) for r_ in rects})}")  # fmt: skip
collide = []
for r_ in rows:
    for lane, ids in r_["lane_sem_ids"].items():
        both = sorted(set(ids["pmk"]) & set(ids["fabric"]))
        if both:
            collide.append(dict(scope=r_["scope"], rank=r_["rank"], lane=lane, ids=both))
row = dict(test=f"PMK program A L{L} tp{NDEV}: lane-core fabric connection words (teardown / buffer index) vs every PMK NoC target / multicast rectangle (mock audit)",
           milestone="PMK-fabric-audit", layer=L, tp=NDEV, mock=C.MOCK, tag=tag, lanes=[list(v) for v in lanes],
           lane_noc={f"{k}": list(v) for k, v in lane_noc.items()}, programs=rows, double_declared=collide,
           status="mock" if C.MOCK else "done")  # fmt: skip
S.log(f"{tag}: RESULT double_declared {collide}; per program lane targets {[(r_['scope'], r_['rank'], r_['n_lane_targets'], r_['n_rect_candidates']) for r_ in rows]}")
RESULT = row
