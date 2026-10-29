# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, TP mesh): Stage 16 P3 TTFT of contexts the decode stack cannot hold (8,192: MkMla's dense attention
holds <= 6,144 cache rows) on the backend p2_full built EARLIER IN THIS PROCESS -- one full-model build per process
(a second backend build lowers the L1 floor). Run it after p2_full in the same server; the server closes after it
(the generator's prefill traces are released here: its decode is unusable afterwards).

Steps: the "s16-full" entry's backend -> release the generator's prefill traces (trace region) -> a B=1 kv cache of
`max_seq` rows + decode io on the SAME backend (the stacks' L1 hand-off buffers are shared: persistent_buffers is keyed)
-> registered as s16common.build's entry with a stale prefill, so p2_ttft (run here with `ttft` args) builds its
TPPrefill on that cache, compiles / captures the prefill-only keys (warmup "prefill") and records its rows.

ARGS: max_seq (8192), ttft ({}: p2_ttft args; defaults below: buckets [2048], ns [2048, 8192], dsa True, sparse_ts
[4096, 8192], warmup "prefill", milestone "P3", gate_n 8192), tag."""
import gc
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
MAXSEQ = int(args.get("max_seq", 8192))
TPN = int(mesh.get_num_devices())

# the backend of an earlier job of this process: Stage 16's p2_full ("s16-full") or 14b's served construction
# (M5.serving, "m5_serving": e.g. p2_tf_m5 / p2_layer_golden construction m5)
full = [k for k in ctx if isinstance(k, tuple) and k and k[0] in ("s16-full", "m5_serving")]
if not full:
    raise RuntimeError("p3_long_on_full: no p2_full / served backend in this process (run one first, same server)")
ent_full = ctx[full[-1]]
backend, layers = ent_full["backend"], list(full[-1][2])


class _Stale:
    """a placeholder prefill: every attribute s16common.build compares differs -> it builds the TPPrefill itself"""

    mhc_mode = mhc_post = dsa = None

    def release(self):
        pass


if getattr(backend, "tpp", None) is not None:
    backend.tpp.release()  # the generator's prefill traces (the trace region is shared with the new keys)
gc.collect()
kv = backend.allocate_kv_cache(max_batch=1, max_seq_len=MAXSEQ)
io = backend.allocate_decode_io(kv)
backend._pt_host = None  # the generator's host page table (its max_seq): the new cache's identity mapping instead
ctx[S.backend_key(layers, 1, TPN, False)] = dict(backend=backend, kv=kv, io=io, pf=_Stale(), build_s=0.0,
                                                 backend_s=0.0, pf_rev=None, pf_attn=None)  # fmt: skip
S.log(f"p3_long_on_full: kv cache max_seq {MAXSEQ} on the p2_full backend ({len(layers)} layers)")

targs = dict(layers=layers if len(layers) != 45 else "all", buckets=[2048], ns=[2048, MAXSEQ], max_seq=MAXSEQ,
             dsa=True, sparse_ts=[4096, MAXSEQ], warmup="prefill", milestone="P3", gate_n=MAXSEQ, reps=3,
             tag=args.get("tag", "p3-long-on-full"), record=args.get("record", True))  # fmt: skip
targs.update(args.get("ttft", {}))
ns = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2_ttft.py"),
                    init_globals={"MESH": mesh, "ARGS": targs, "CTX": ctx}, run_name="__job__")  # fmt: skip
RESULT = ns.get("RESULT")
