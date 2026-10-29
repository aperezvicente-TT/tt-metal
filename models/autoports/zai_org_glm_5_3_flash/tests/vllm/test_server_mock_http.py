# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP checks against `vllm serve` with the TT plugin, the GLM-5.3-Flash adapter and the CPU mock
generator (mock_server.py): the production launch path with the engine core in its own process. Covers the
plugin registration resolving to this adapter inside the engine process, a non-aligned prompt over HTTP, the
served max_model_len, the DSA request limit as HTTP 400, and the chat template path."""

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
from pathlib import Path

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness import SNAPSHOT
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    make_backend,
    prompt_ids,
    reference_greedy,
)

MODEL = "zai-org/GLM-5.3-Flash"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    port = _free_port()
    tmp = tmp_path_factory.mktemp("mock_server")
    marker = tmp / "mock_built.jsonl"
    log = Path(os.environ.get("GLM53_09A_SERVER_LOG", tmp / "server.log"))
    args = [
        SNAPSHOT,
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--block-size",
        "64",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "512",
        "--no-enable-prefix-caching",
        "--shutdown-timeout",
        "30",
        "--additional-config",
        json.dumps({"tt": {"sample_on_device_mode": "all", "trace_mode": "all"}}),
    ]
    # the engine core's graceful-shutdown budget as in the device launch (scripts/env.sh raises it to 900 s): vLLM's 5 s
    # default force-kills a slow-but-orderly engine core on a loaded host, which on device is an unclean death
    env = dict(
        os.environ,
        GLM53_MOCK_MARKER=str(marker),
        HF_HUB_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="60",
        # the served plugin (srv/stack) runs a strict startup self-test against fp32 goldens before the socket binds
        # (default mode strict); the mock generator's logits are synthetic, so the self-test is the device gate, not
        # this one: off here (the device launches set GLM53_SELFTEST=strict explicitly)
        GLM53_SELFTEST="off",
    )
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    f = log.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "models.autoports.zai_org_glm_5_3_flash.tests.vllm.mock_server", *args],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
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
    yield {"base": base, "proc": proc, "marker": marker, "log": log, "args": args}
    # orderly shutdown: signal only the API server; it stops the engine core (which closes the mesh) itself
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.fail("mock server did not shut down within 120 s of SIGINT")
    f.close()
    events = [json.loads(l) for l in marker.read_text().splitlines()]
    built = [e for e in events if "max_batch" in e]
    closed = [e for e in events if e.get("event") == "mesh_closed"]
    assert closed and closed[0]["pid"] == built[0]["pid"], f"engine process did not close its mesh: {events}"
    assert "force killing" not in log.read_text(), "shutdown force-killed a process"


_ORACLE = None


def _oracle():
    """The server's generator is built in the engine process; the same deterministic MockModel weights here give the
    oracle (glm53_mock_stack.make_backend: fixed seeds, the server's dsa_mode)."""
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = make_backend(max_batch=8)
    return _ORACLE


def _prompt(seed: int, n: int) -> list[int]:
    return prompt_ids(seed, n)


def test_server_registration_resolves_adapter_in_engine_process(server, evidence):
    built = [json.loads(l) for l in server["marker"].read_text().splitlines()]
    assert len(built) == 1 and built[0]["pid"] != server["proc"].pid, built
    assert built[0]["max_batch"] == 8 and built[0]["max_seq_len"] == 1048576 and built[0]["dsa_mode"] == "dense_exact"
    with urllib.request.urlopen(server["base"] + "/v1/models") as r:
        models = json.loads(r.read())
    card = models["data"][0]
    assert card["id"] == MODEL and card["max_model_len"] == 1048576
    log = server["log"].read_text()
    assert "Resolved architecture: Glm5NextForConditionalGeneration" in log
    evidence(
        engine_pid=built[0]["pid"],
        api_pid=server["proc"].pid,
        served_max_model_len=card["max_model_len"],
        serve_args=server["args"],
    )


def test_server_nonaligned_prompt_http(server, evidence):
    got = {}
    for n in (1, 33, 131, 1013):
        status, body = _post(
            server["base"] + "/v1/completions",
            {
                "model": MODEL,
                "prompt": _prompt(n, n),
                "max_tokens": 12,
                "temperature": 0.0,
                "ignore_eos": True,
                "return_token_ids": True,
            },
        )
        assert status == 200, body
        toks = body["choices"][0]["token_ids"]
        assert toks == reference_greedy(_oracle(), _prompt(n, n), 12), n
        assert body["usage"]["prompt_tokens"] == n and body["usage"]["completion_tokens"] == 12
        got[n] = toks[:4]
    evidence(prompt_lengths=list(got), first_tokens=got)


def test_server_request_context_limit_is_http_400(server, evidence):
    """A prompt with no room under the DSA dense-exact limit gets HTTP 400 naming the limit; an explicit max_tokens
    past the limit is clamped by the OpenAI frontend exactly like one past max_model_len (finish_reason length);
    the default max_tokens stops at the limit; the server stays healthy."""
    status, body = _post(
        server["base"] + "/v1/completions",
        {"model": MODEL, "prompt": _prompt(7, 2100), "max_tokens": 4, "temperature": 0.0},
    )
    assert status == 400 and "request context limit of 2051" in json.dumps(body), (status, body)
    # explicit max_tokens past the limit: clamped to the room (2051 - 2000) like one past max_model_len
    status2, body2 = _post(
        server["base"] + "/v1/completions",
        {
            "model": MODEL,
            "prompt": _prompt(8, 2000),
            "max_tokens": 100,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    assert status2 == 200, body2
    choice = body2["choices"][0]
    assert len(choice["token_ids"]) == 51 and choice["finish_reason"] == "length", choice
    # chat without max_tokens defaults to the remaining context: it stops exactly at the limit
    status3, body3 = _post(
        server["base"] + "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Write a long story."}],
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert status3 == 200, body3
    n_prompt = len(body3["prompt_token_ids"])
    n_out = len(body3["choices"][0]["token_ids"])
    assert n_prompt + n_out == 2051 and body3["choices"][0]["finish_reason"] == "length", (n_prompt, n_out)
    lens = {
        "completions_max_tokens_100_at_2000": len(choice["token_ids"]),
        "chat_default": {"prompt": n_prompt, "completion": n_out},
    }
    with urllib.request.urlopen(server["base"] + "/health") as r:
        assert r.status == 200
    evidence(over_limit_status=status, over_limit_message=json.dumps(body)[:300], completion_tokens_at_2000=lens)


def test_server_chat_template_path(server, evidence):
    status, body = _post(
        server["base"] + "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert status == 200, body
    chat_ids = body["prompt_token_ids"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 5."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    expected = tok(rendered, add_special_tokens=False)["input_ids"]
    assert chat_ids == expected, (tok.decode(chat_ids), rendered)
    assert body["choices"][0]["token_ids"] == reference_greedy(_oracle(), chat_ids, 8)
    evidence(chat_prompt_tokens=len(chat_ids), rendered=rendered)
