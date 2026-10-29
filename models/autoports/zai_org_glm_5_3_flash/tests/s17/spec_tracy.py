# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 Tracy / device-profiler run (run under `python -m tracy -r -p -v --no-web-server -o <dir>`; a pair): the
speculative step's traced replays in signposted windows for tt-perf-report — SPEC_STEP_K<K> (the whole step trace),
SPEC_BOOT_K<K> (the bootstrap trace) and PLAIN_W<w> (the plain decode trace of the same backend), each `reps`
replays, every replay synchronized and the device profiler read (ReadDeviceProfiler) so no buffer overflows. The
model prefix (default L0-L4) and K as the pair units; the prompt is teacher-forced with the MTP cache kept
(s17common.mtp_prefill). No measured wall times here (the profiler perturbs them): the device-kernel durations come
from the tt-perf-report windows; the wall-clock rows are the unwatched 2,000-replay jobs.
usage: spec_tracy.py [--layers 0,1,2,3,4] [--b 4] [--k 3] [--reps 4] [--prompt aime24_04]"""
import argparse
import json
import os
import sys
import time

import torch

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "s14b"))
import s14common as C  # noqa: E402
import s17common as S  # noqa: E402

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_a, **_k):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,2,3,4")
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--prompt", default="aime24_04")
    a = ap.parse_args()
    profiling = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))
    mesh, open_s = C.open_mesh(ttnn, "pair", trace_region=512 << 20, worker_l1_size=1264512)
    ctx = {"cmap": C.chip_map(mesh), "mgd": "pair", "open_s": open_s, "worker_l1": 1264512, "payload": 0}
    try:
        ids = [int(v) for v in a.layers.split(",")]
        key, be, gen = S.serving(ctx, mesh, B=a.b, ids=ids, spec_k=a.k)
        sp, kv = be.spec, be._kv
        io = kv.extra["io"]
        pt = (torch.arange(a.b * kv.blocks_per_seq) % kv.num_blocks).reshape(a.b, -1)
        be.write(io.page_table, pt)
        wts = kv.extra.get("decode_traces", {})
        w0 = min([w for w in be.native_widths + be.narrow_widths if w in wts] or [a.b])
        plain = wts.get(w0, kv.extra.get("decode_trace"))
        ws = "/opt/glm53/kit"
        prompts = {q["id"]: q for q in json.load(open(os.path.join(ws, "doc/prework/p5_mtp_acceptance/prompts.json")))["prompts"]}
        prompt = [int(v) for v in prompts[a.prompt]["ids"]]
        flush = (lambda: ttnn.ReadDeviceProfiler(mesh)) if profiling else (lambda: None)
        S.mtp_prefill(be, sp, io, prompt, plain, pt[0], a.b)
        sp.enter(pt[0])
        tr = sp.capture()
        if C.MOCK:  # (the mock cluster cannot capture: the step and bootstrap run eagerly once)
            tr["step"][1]()
            tr["boot"][1]()
            print("S17-TRACY mock ok", flush=True)
            return
        for _ in range(2):  # warm
            ttnn.execute_trace(mesh, tr["step"], cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        flush()
        for name, tid in ((f"SPEC_STEP_K{a.k}", tr["step"]), (f"SPEC_BOOT_K{a.k}", tr["boot"]), (f"PLAIN_W{w0}", plain)):
            signpost(name)
            t0 = time.perf_counter()
            for _ in range(a.reps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                flush()
            print(f"S17-TRACY {name} reps {a.reps} wall/replay (profiler on) {(time.perf_counter() - t0) / a.reps * 1e3:.2f} ms",
                  flush=True)
            signpost(f"{name}_END")
        sp.leave()
        print(f"S17-TRACY done code={os.environ.get('GLM53_CODE_SHA')} B={a.b} K={a.k} plain_width={w0}", flush=True)
    finally:
        for k, v in list(ctx.items()):
            rel = getattr(v, "release", None)
            if callable(rel):
                try:
                    rel()
                except Exception as e:  # noqa: BLE001
                    print(f"release {k}: {e!r}", flush=True)
        C.close_mesh(ttnn, mesh, fabric=True)
        print("S17-TRACY closed", flush=True)


if __name__ == "__main__":
    main()
