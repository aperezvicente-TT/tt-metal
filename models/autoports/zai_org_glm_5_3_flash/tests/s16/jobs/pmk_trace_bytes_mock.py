# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, the 2x4 / pair MOCK; trace-shrink ruling 2026-10-23 16:33:48Z item 1): the per-launch RUNTIME-ARG
bytes of the served PMK chain's programs, per program and per part, at the job's TP. Builds one layer of each kind
(default L0 dense_kda, L3 sparse_dsa, L4 sparse_kda) through PmkStack(impl="served"), runs the compile forward (every
program's descriptors are built there, the engine's rt_bytes report fills in), and reports:
  * program A (the family engine, per scope = A1..A5): pmk_layer.rt_bytes per rank -- total, per (role, RISC)
    composite, the top parts;
  * program B (pmk_moe2): the runtime args of each of its kernels (recorded at ProgramB / PmkMoe.kernel);
  * per layer and per bucket replay: the sum over its programs, x 45 layers by kind (31 sparse_kda / 11 sparse_dsa /
    3 dense_kda).
The fabric connection args the post-build hooks append (a few dozen words per lane core) are not counted. Binaries,
CB / semaphore configs and launch messages are NOT runtime args (the pair unit measures the trace region itself).
ARGS: layers ([0, 3, 4]), rows (2048), env ({}), b_* (program B levers), tag."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe as M1  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_moe2 as M2  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
LAYERS = [int(v) for v in args.get("layers", [0, 3, 4])]
N = int(args.get("rows", 2048))
tag = args.get("tag", f"pmk-trace-bytes-tp{TPN}")
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}

B_REC = {}  # (id(program B object), kernel name) -> dict(cores, width, bytes)


def _wrap(cls):
    orig = cls.kernel

    def kernel(self, name, ct, rt, defs, cfg):
        vals = list(rt.values()) if isinstance(rt, dict) else []
        B_REC[(id(self), name)] = dict(cores=len(vals), width=max((len(v) for v in vals), default=0),
                                       bytes=4 * sum(len(v) for v in vals), ct=len(ct))  # fmt: skip
        return orig(self, name, ct, rt, defs, cfg)

    cls.kernel = kernel


_wrap(M1.PmkMoe)
_wrap(M2.ProgramB)
t0 = time.time()
ent = S.build(mesh, ctx, LAYERS, b=2, buckets=(N,), max_seq=N + 512)
pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
pf.moe_impl = args.get("moe", "ep")
S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
pf.kda_scan = args.get("kda_scan", "op")
backend.tpp = pf
c = pf.bucket_for(N)
opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=False)
opts.update({k: v for k, v in args.items() if k.startswith("b_")})
os.environ["GLM53_PF_MODE"] = "pmk"
stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts).attach()
pf.compile((c,))  # the compile forward: every program's descriptors built (the engine audits fill in)
H.log(f"{tag}: built + compiled in {time.time() - t0:.1f} s")

rows = {}
for li, L, kind in stack.layers:
    sl = stack.progs[(c, L)]
    r_ = dict(kind=kind, modes=dict(sl.modes), A={}, B={})
    a = getattr(sl, "a", None)
    if a is not None:
        for key, au in sorted(a.engine.arg_audits.items()):
            scope, rank = key.rsplit(":", 1) if ":" in key else ("all", key)
            rb = au.get("rt_bytes") or {}
            d = r_["A"].setdefault(scope, dict(bytes_rank0=None, bytes_max=0, composites=None, top_parts=None))
            d["bytes_max"] = max(d["bytes_max"], int(rb.get("total", 0)))
            if rank == "0":
                d.update(bytes_rank0=int(rb.get("total", 0)), composites={k: v["bytes"] for k, v in rb.get("composites", {}).items()},
                         top_parts=rb.get("top_parts"))  # fmt: skip
    b = getattr(sl, "b", None)
    if b is not None:
        bobjs = {id(b)} | {id(v) for v in vars(b).values() if isinstance(v, M1.PmkMoe)}  # B itself or the wrapper's B
        r_["B"] = {name: v for (oid, name), v in B_REC.items() if oid in bobjs}
    r_["A_total"] = sum(d["bytes_max"] for d in r_["A"].values())
    r_["B_total"] = sum(v["bytes"] for v in r_["B"].values())
    rows[L] = r_
    H.log(f"{tag}: L{L} {kind}: A {r_['A_total']} B {r_['B_total']} B per program; A per scope "
          f"{ {k: v['bytes_max'] for k, v in r_['A'].items()} }; B kernels {r_['B']}")
    for scope, d in r_["A"].items():
        H.log(f"{tag}: L{L} {scope}: composites {d['composites']}; top parts {d['top_parts']}")
by_kind = {}
for L, r_ in rows.items():
    by_kind.setdefault(r_["kind"], r_["A_total"] + r_["B_total"])
counts = dict(sparse_kda=31, sparse_dsa=11, dense_kda=3)
per_replay = sum(by_kind.get(k, 0) * n for k, n in counts.items())
summary = dict(tp=TPN, per_layer_bytes=by_kind, per_45_layer_replay_bytes=per_replay, per_45_layer_replay_mib=round(per_replay / 2**20, 2),
               unmatched_B_records=len(B_REC) - sum(len(r_["B"]) for r_ in rows.values()), tag=tag)  # fmt: skip
H.log(f"{tag}: SUMMARY {summary}")
RESULT = dict(summary=summary, layers=rows, mock=S.C.MOCK, status="mock" if S.C.MOCK else "info")
