# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Segmented prefill traces end to end over HTTP (Stage 12 item 10, program 2026-10-27T12:50:09Z):
`vllm serve` with the TT plugin's alternating loop and this adapter's vLLM face (segment_rows / prefill_segments /
prefill_abort / open_chunk) over the CPU mock (mock_server.py, the engine core in its own process).

- A decoder runs while a 1,300-token prompt prefills. Its chunks run as S = 3 segments, one per engine step, with the
  decoder's steps between them (the plugin stamps them; the mock records every segment in the marker).
- Both requests' tokens equal the mock oracle's greedy ids: the plugin's chunk-minus-one + 1-token release accounting
  and the adapter's segment-0 record / re-send / commit-at-the-last-segment keep the sequence exactly today's.
- The identity: S = 1 (no plugin stamp) gives the same ids and no segment ran.

The plugin with the loop (glm53/glm53-s12-segdefer >= 4aaf584) is taken from GLM53_SEG_PLUGIN_SRC when set."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness import SNAPSHOT
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import make_backend, prompt_ids, reference_greedy

MODEL = "zai-org/GLM-5.3-Flash"
PLUGIN = os.environ.get("GLM53_SEG_PLUGIN_SRC", "")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _serve(tmp: Path, n_seg: int):
    port, marker, log = _free_port(), tmp / f"marker_s{n_seg}.jsonl", tmp / f"server_s{n_seg}.log"
    tt = {"sample_on_device_mode": "all", "trace_mode": "all", "mixed_prefill_decode_steps": True,
          "prefill_segments": n_seg}  # fmt: skip
    args = [SNAPSHOT, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(port), "--block-size", "64",
            "--max-num-seqs", "8", "--max-num-batched-tokens", "512", "--no-enable-prefix-caching",
            "--shutdown-timeout", "30", "--additional-config", json.dumps({"tt": tt})]  # fmt: skip
    env = dict(os.environ, GLM53_MOCK_MARKER=str(marker), HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="60", GLM53_SELFTEST="off",
               GLM53_PF_SEGMENTS="3",  # the model (mock) segments 2,048-bucket-like chunks; the PLUGIN's S decides
               GLM53_PREFILL_SEGMENT_MIN_TOKENS="256")  # the suite's 512-token chunks are segmentable  # fmt: skip
    if PLUGIN:
        env["PYTHONPATH"] = PLUGIN + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    f = log.open("w")
    proc = subprocess.Popen([sys.executable, "-m", "models.autoports.zai_org_glm_5_3_flash.tests.vllm.mock_server", *args],
                            stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            pytest.fail(f"mock server exited rc={proc.returncode}; log {log}:\n{log.read_text()[-4000:]}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
        pytest.fail(f"mock server not healthy in 300 s; log {log}")
    return proc, f, base, marker, log


def _stop(proc, f):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.fail("mock server did not shut down within 120 s of SIGINT")
    f.close()


def _workload(base):
    """a decoder (40-token prompt, 96 tokens out) and, once it decodes, a 1,300-token prompt (16 tokens out)"""
    a_ids, b_ids = prompt_ids(11, 40), prompt_ids(12, 1300)
    out = {}

    def run(name, ids, n):
        out[name] = _post(base + "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": n, "temperature": 0,
                                                       "ignore_eos": True, "return_token_ids": True})["choices"][0]  # fmt: skip

    ta = threading.Thread(target=run, args=("a", a_ids, 96))
    ta.start()
    time.sleep(1.0)  # the decoder is running when the long prompt's chunks arrive
    run("b", b_ids, 16)
    ta.join(600)
    return {"a": (a_ids, out["a"]), "b": (b_ids, out["b"])}


@pytest.mark.parametrize("n_seg", [3, 1])
def test_segmented_chunks_beside_a_decoder_keep_the_greedy_ids(tmp_path, n_seg):
    proc, f, base, marker, log = _serve(tmp_path, n_seg)
    try:
        got = _workload(base)
    finally:
        _stop(proc, f)
    oracle = make_backend(max_batch=8)
    for name, (ids, choice) in got.items():
        n = len(choice["token_ids"])
        assert choice["token_ids"] == reference_greedy(oracle, ids, n), (name, n_seg, choice["token_ids"])
    events = [json.loads(line) for line in marker.read_text().splitlines()]
    segs = [e for e in events if e.get("event") == "prefill_segment"]
    if n_seg == 1:
        assert not segs, f"S = 1 ran segments: {segs[:4]}"
    else:
        by_chunk = {}
        for e in segs:
            by_chunk.setdefault((e["slot"], e["start"], e["n"]), []).append(e["segment"])
        assert by_chunk, f"no chunk ran segmented (log {log})"
        assert all(v == [0, 1, 2] for v in by_chunk.values()), by_chunk
    closed = [e for e in events if e.get("event") == "mesh_closed"]
    assert closed, "the engine process did not close its mesh"
    assert "Traceback" not in log.read_text(), log.read_text()[-3000:]
