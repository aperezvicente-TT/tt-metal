# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving robustness over HTTP (SRV lifecycle): `vllm serve` with the TT plugin, the GLM-5.3-Flash adapter and the
CPU mock generator, the engine core in its own process (mock_server.py), no TT device.

5  streaming: one SSE chunk per token, the first token streamed right after prefill (TTFT << e2e)
1  client disconnects mid-stream (the API server's abort path) among concurrent users: survivors bitwise vs the
   same prompts served alone, the aborted users' slots reused by a following wave (oracle-exact), server healthy
7  a device exception in a served step: the engine core dies, the API server exits non-zero (status 70), the
   engine process closed its mesh exactly once, nothing re-opened it
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest
import requests

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness import SNAPSHOT
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    FAULT_DECODE_STEP_ENV,
    make_backend,
    prompt_ids,
    reference_greedy,
)

MODEL = "zai-org/GLM-5.3-Flash"
_ORACLE = None


def _oracle():
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = make_backend(max_batch=8)
    return _ORACLE


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(tmp, *, max_num_seqs=8, env_extra=None):
    port = _free_port()
    marker = tmp / "mock_built.jsonl"
    log = tmp / "server.log"
    args = [SNAPSHOT, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(port), "--block-size", "64",
            "--max-num-seqs", str(max_num_seqs), "--max-num-batched-tokens", "512", "--no-enable-prefix-caching",
            "--shutdown-timeout", "30", "--max-model-len", "2051",
            "--additional-config",
            json.dumps({"tt": {"sample_on_device_mode": "all", "trace_mode": "all"}})]  # fmt: skip
    env = dict(os.environ, GLM53_MOCK_MARKER=str(marker), HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="60", **(env_extra or {}))  # fmt: skip
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    f = log.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "models.autoports.zai_org_glm_5_3_flash.tests.vllm.mock_server", *args],
        stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 300
    while time.time() < deadline:
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
    return {"base": base, "proc": proc, "marker": marker, "log": log, "file": f}


def _stop(srv, *, expect_rc=0):
    proc = srv["proc"]
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            pytest.fail("mock server did not shut down within 180 s of SIGINT")
    srv["file"].close()
    events = [json.loads(line) for line in srv["marker"].read_text().splitlines()]
    built = [e for e in events if "max_batch" in e]
    closed = [e for e in events if e.get("event") == "mesh_closed"]
    assert len(built) == 1, f"the mesh was opened {len(built)} times: {events}"
    assert len(closed) == 1 and closed[0]["pid"] == built[0]["pid"], f"mesh not closed once by the engine: {events}"
    assert "force killing" not in srv["log"].read_text(), "shutdown force-killed a process"
    assert proc.returncode == expect_rc, (proc.returncode, srv["log"].read_text()[-3000:])
    return events


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = _launch(tmp_path_factory.mktemp("srv_http"))
    yield srv
    _stop(srv)


def _stream(base, prompt, max_tokens, *, stop_after=None, timeout=600):
    """Streamed completion with token ids; `stop_after` = close the connection after that many tokens (the client
    disconnect the API server turns into an engine abort). Returns dict(tokens, stamps, finish, status)."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "return_token_ids": True, "stream": True}  # fmt: skip
    t0 = time.perf_counter()
    toks, stamps, finish = [], [], None
    with requests.post(f"{base}/v1/completions", json=body, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            return {"status": r.status_code, "error": r.text[:300]}
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            msg = json.loads(line[6:])
            for c in msg.get("choices", []):
                ids = c.get("token_ids") or []
                if ids:
                    stamps.append((time.perf_counter() - t0, len(ids)))
                toks += ids
                finish = c.get("finish_reason") or finish
            if stop_after is not None and len(toks) >= stop_after:
                r.close()  # the disconnect
                return {"status": 200, "tokens": toks, "stamps": stamps, "finish": "client_closed"}
    return {"status": 200, "tokens": toks, "stamps": stamps, "finish": finish, "e2e_s": time.perf_counter() - t0}


def _running(base) -> float:
    txt = requests.get(f"{base}/metrics", timeout=30).text
    vals = [float(l.split()[-1]) for l in txt.splitlines() if l.startswith("vllm:num_requests_running")]
    return max(vals) if vals else -1.0


def test_http_streaming_emits_one_chunk_per_token_with_an_early_first_token(server, evidence):
    p = prompt_ids(5, 400)
    r = _stream(server["base"], p, 48)
    assert r["status"] == 200 and r["finish"] == "length", r
    assert r["tokens"] == reference_greedy(_oracle(), p, 48)
    chunks = len(r["stamps"])
    assert chunks == 48 and all(n == 1 for _, n in r["stamps"]), "a token was held back and batched into a later chunk"
    ttft = r["stamps"][0][0]
    itl = [b[0] - a[0] for a, b in zip(r["stamps"], r["stamps"][1:])]
    # the first token is streamed as soon as prefill + one sample is done: well before the 48-token tail
    assert ttft < 0.5 * r["e2e_s"], (ttft, r["e2e_s"])
    assert ttft < 3 * max(itl) + 2.0, (ttft, max(itl))
    evidence(chunks=chunks, ttft_s=round(ttft, 4), e2e_s=round(r["e2e_s"], 4), itl_s_max=round(max(itl), 4))


def test_http_client_disconnects_abort_without_disturbing_the_others(server, evidence):
    """8 concurrent users; 3 close their stream after 2-5 tokens. The 5 others match the same prompts served alone
    (bitwise) and the oracle; the server reports 0 running requests afterwards; a following wave of 8 fresh users
    (reusing the aborted slots) is oracle-exact."""
    import concurrent.futures as cf

    base = server["base"]
    prompts = [prompt_ids(100 + i, 30 + 97 * i) for i in range(8)]
    n_out = 24
    victims = {1: 2, 4: 5, 6: 3}
    alone = {i: reference_greedy(_oracle(), prompts[i], n_out) for i in range(8)}
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: _stream(base, prompts[i], n_out, stop_after=victims.get(i)), range(8)))
    for i, r in enumerate(res):
        assert r["status"] == 200, (i, r)
        if i in victims:
            assert r["finish"] == "client_closed" and len(r["tokens"]) >= victims[i], (i, r)
            assert r["tokens"] == alone[i][: len(r["tokens"])]
        else:
            assert r["finish"] == "length" and r["tokens"] == alone[i], (i, r["tokens"][:8], alone[i][:8])
    deadline = time.time() + 60
    while _running(base) > 0 and time.time() < deadline:
        time.sleep(0.5)
    assert _running(base) == 0, "aborted requests still counted as running"
    with cf.ThreadPoolExecutor(8) as ex:
        wave = list(ex.map(lambda i: _stream(base, prompts[i], n_out), range(8)))
    assert all(w["finish"] == "length" and w["tokens"] == alone[i] for i, w in enumerate(wave)), [
        w["finish"] for w in wave
    ]
    with urllib.request.urlopen(base + "/health", timeout=10) as h:
        assert h.status == 200
    evidence(aborted_after_tokens=victims, survivors_bitwise=5, second_wave_oracle_exact=8)


def test_http_device_exception_exits_the_server_non_zero(tmp_path, evidence):
    """GLM53_MOCK_FAULT_DECODE_STEP=3 in the engine process: the 3rd decode step raises. The in-flight request gets
    an error, the engine core dies, the API server exits with status 70 (vllm_tt_plugin TT_ENGINE_DEAD_EXIT_CODE),
    the engine process closed its mesh exactly once and nothing re-opened a mesh."""
    # vllm_tt_plugin.platform.TT_ENGINE_DEAD_EXIT_CODE (not imported here: the platform module cannot be imported
    # in a test process before vLLM resolved its plugins)
    TT_ENGINE_DEAD_EXIT_CODE = 70

    srv = _launch(tmp_path, env_extra={FAULT_DECODE_STEP_ENV: "3", "GLM53_SELFTEST": "off"})  # a request-time fault: the startup self-test (default strict) would take decode steps 1-3
    try:
        r = _stream(srv["base"], prompt_ids(1, 20), 16, timeout=120)
    except requests.exceptions.RequestException as exc:  # the stream may be cut when the engine dies
        r = {"status": "cut", "error": str(exc)[:200]}
    deadline = time.time() + 240
    while srv["proc"].poll() is None and time.time() < deadline:
        time.sleep(1)
    assert srv["proc"].poll() is not None, "the API server stayed up after the engine died"
    events = _stop(srv, expect_rc=TT_ENGINE_DEAD_EXIT_CODE)
    log = srv["log"].read_text()
    assert "injected device fault" in log
    assert "exiting with status %d" % TT_ENGINE_DEAD_EXIT_CODE in log
    evidence(client_result=r, exit_status=srv["proc"].returncode, marker_events=events)
