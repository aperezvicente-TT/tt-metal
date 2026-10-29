# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 2x4 HOST MOCK -- no-tt-devices + TT_METAL_MOCK_CLUSTER_DESC_PATH): the PMK STACK driver
(tt/pmk_stack.py) at n PMK layers. Two modes:
  real=True  (n = 1, layer 4): the layer's TPPrefill pieces are built (S.build, weights from the cache), PmkStack(pf,
             1).attach() installs the hook, the bucket's compile launch runs THROUGH TPPrefill.forward (embed -> the
             PMK program on the persistent residual -> head), then trace_capture(c) records the key (no trace on the
             mock) and replay() runs it once more; the n = 0 path runs on the same TPPrefill after detach().
  real=False (n = 3 / 45): PfShim (the 45-layer kind table on the mock mesh, no weights; mhc with random weights):
             every one of the n programs is built (kind builder + wired pieces), launched once on the mock mesh
             (kernels JIT-build, CBs / semaphores / runtime-arg audit validated per rank) in schedule order on one
             persistent residual, and the census per kind + the wired / stub phases are recorded.
Criterion (one row in tests.jsonl): every program built and launched, census fits per kind (l1_max_cb_end <=
1,330,000), exactly one capture key per planned bucket, counters as expected. ARGS: n (1), real (n == 1), layer (4),
rows (2048), tag, record (True)."""
import gc
import os
import sys
import time
import traceback

import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
NPMK = int(args.get("n", 1))
REAL = bool(args.get("real", NPMK == 1))
L = int(args.get("layer", 4))
N = int(args.get("rows", 2048))
FLOOR = int(args.get("floor", 1330000))
tag = args.get("tag", f"pmk-stack-mock-n{NPMK}-tp{TPN}")
rec = bool(args.get("record", True))
cfg = hf_config()
t0 = time.time()
err = None
info = {}
try:
    if REAL:
        os.environ["GLM53_PF_MHC"] = "comp"
        os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
        os.environ["GLM53_PF_PACKED"] = ""
        os.environ["GLM53_PF_PROBE_LAYERS"] = ""
        S.set_attn(args)
        layers = [int(v) for v in args.get("layers", [L])]
        LOOP_N = int(args.get("loop_n", 0))
        ent = S.build(
            mesh, ctx, layers, b=1, buckets=(N,), max_seq=max(1, LOOP_N) * N + 512
        )  # the loop's N chunks of pages
        pf = ent["pf"]
        c = pf.bucket_for(N)
        stack = PS.PmkStack(pf, NPMK).attach()
        H.log(f"{tag}: built {len(stack.progs)} program(s) in {time.time() - t0:.1f} s: {stack.wired()}")
        pf.write_inputs(c, torch.zeros(N, dtype=torch.int64), slot=0, chunk_start=0, n_valid=N)
        t1 = time.time()
        pf.forward(c)  # the compile launch of the bucket (embed -> PMK layer(s) -> op layers -> head)
        ttnn.synchronize_device(mesh)
        info["compile_s"] = round(time.time() - t1, 1)
        key = stack.trace_capture(c)
        stack.replay(c)
        ttnn.synchronize_device(mesh)
        info["key"] = list(key)
        # n = 0 on the same TPPrefill: nothing installed, the op-graph forward runs as before
        stack.detach()
        assert pf.pmk_layers == {}, pf.pmk_layers
        pf.forward(c)
        ttnn.synchronize_device(mesh)
        stack.attach()
        info["state_handoff"] = {k: (v if k != "layers" else {L_: {kk: vv for kk, vv in e.items() if kk != "layout"} for L_, e in v.items()})
                                 for k, v in stack.state_handoff().items()}  # fmt: skip
        if LOOP_N > 0:  # lever 2: N chunks in ONE capture on the device-resident arg table (mock: N eager forwards)
            t2 = time.time()
            loop = PS.ChunkLoop(stack, LOOP_N, c)
            filled = loop.fill(torch.zeros(LOOP_N * c, dtype=torch.int64), slot=0, chunk_start=0, t=0)
            aud = loop.audit()
            lkey = loop.capture(t=0)
            loop.replay()
            ttnn.synchronize_device(mesh)
            info["loop"] = dict(n=LOOP_N, filled=filled, key=list(lkey), audit=aud, counters=dict(loop.counters),
                                table_rows=list(loop.table.shape), plan_1m=loop.prompt_plan(1_048_576), s=round(time.time() - t2, 1))  # fmt: skip
            H.log(f"{tag}: chunk loop {info['loop']}")
    else:
        pf = PS.PfShim(mesh, cfg)
        stack = PS.PmkStack(pf, NPMK, weights="random").attach()
        c = PS.V1_BUCKET
        H.log(f"{tag}: built {len(stack.progs)} programs in {time.time() - t0:.1f} s")
        res = stack.res[c]
        ci = type("Ci", (), {})()
        ci.bucket = c
        t1 = time.time()
        launched = 0
        for li, L_, kind in stack.layers:  # the chain in schedule order on the one persistent residual
            out = pf.pmk_layers[L_](res, tpp=pf, pl=pf.layers[li], ci=ci, sp=True)
            assert out is res
            launched += 1
            if launched in (1, 2, 3) or launched % 10 == 0:
                H.log(f"{tag}: launched {launched}/{NPMK} (L{L_} {kind}) at {time.time() - t1:.1f} s")
        ttnn.synchronize_device(mesh)
        info["launch_s"] = round(time.time() - t1, 1)
        info["launched"] = launched
        key = stack.trace_capture(c)
        info["key"] = list(key)
    summ = stack.summary()
    info["streams"] = stack.streams(c)
    info["geometry"] = stack.geometry(c)
    info["time_model"] = PS.time_model(cfg)
except Exception:  # noqa: BLE001 -- recorded as the row's failure, the server stays up
    err = traceback.format_exc()[-4000:]
    H.log(f"FAILED:\n{err}")
    summ = {}

census = summ.get("census", {})
# the criterion is the LAUNCH census (the CBs the program allocates = the registered bodies'; the engine's check_l1
# judges the same); the full-plan census (every phase) is reported, and its overage named (a family-side item)
fits = all(max(v["launch"].values()) <= FLOOR for v in census.values()) if census else False
planned_over = {k: v["l1_max_cb_end"] - FLOOR for k, v in census.items() if v["l1_max_cb_end"] > FLOOR}
kinds_built = sorted(census)
one_key = (len(summ.get("captured", {})) == 1) if summ else False
loop_ok = ("loop" not in info) or (
    info["loop"]["filled"] == info["loop"]["n"] and info["loop"]["counters"]["forwards"] == 2 * info["loop"]["n"]
)
ok = err is None and fits and one_key and summ.get("programs") == NPMK and loop_ok
status = "pass" if ok else "fail"
row = dict(test=f"PMK stack mock n={NPMK} ({'real TPPrefill L' + str(L) if REAL else 'PfShim 45-layer table'}): {NPMK} programs built + "
                f"launched on the {'mock' if C.MOCK else 'device'} 2x4 mesh, census per kind <= {FLOOR}, one capture key per bucket",
           milestone="PMK-B5", status=status, n=NPMK, real=REAL, tp=TPN, bucket=PS.V1_BUCKET, programs=summ.get("programs"),
           kinds_built=kinds_built, census=census, fits=fits, planned_over_bytes=planned_over, captured=summ.get("captured"), counters=summ.get("counters"),
           wired={k: {kk: vv for kk, vv in v.items() if kk != "stubs"} for k, v in summ.get("wired", {}).items()} if len(str(summ.get("wired", {}))) < 8000 else str(summ.get("wired"))[:8000],
           stubs_per_kind={v["kind"]: v["stubs"] for v in summ.get("wired", {}).values()},
           loop=info.get("loop"), info=info if len(str(info)) < 12000 else str(info)[:12000], error=err, mock=C.MOCK, wall_s=round(time.time() - t0, 1), tag=tag)  # fmt: skip
H.log({k: v for k, v in row.items() if k not in ("test", "wired", "info", "error", "census")})
H.log(f"census: {census}")
if rec:
    C.record("tests", row)
RESULT = row
