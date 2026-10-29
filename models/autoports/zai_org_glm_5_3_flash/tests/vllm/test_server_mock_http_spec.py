# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding (method mtp) x logprobs over HTTP: `vllm serve` with the TT plugin, the GLM-5.3-Flash adapter
and the CPU mock generator (mock_server.py) with a host speculative step (glm53_mock_stack.HostSpec, GLM53_SPEC_K=1),
against the same server without speculation.

The served stack's strict startup self-test asks for logprobs=100 on every request (teacher-forced top-5 against the
fp32 goldens); a plugin that refused logprobs under speculation failed it on every prompt and the server never bound.
Here the self-test's own plan (vllm_tt_plugin.selftest: the pack's AIME / g128 / g512 prompts, 32 teacher-forced
steps) runs over HTTP on both servers: every observation (token ids and the top logprobs at every position) must be
BITWISE the plain server's -- a logprobs request decodes without speculation. A greedy request without logprobs still
speculates on the MTP server (the spec-decode counters move) and returns the plain server's tokens.

The mock's logits are synthetic, so the self-test's golden scores are not the point here (the device gate is);
GLM53_SELFTEST stays off at startup and the comparison is MTP on vs off."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from models.autoports.zai_org_glm_5_3_flash.tests.vllm.engine_harness import SNAPSHOT

MODEL = "zai-org/GLM-5.3-Flash"
SPEC_K = int(os.environ.get("GLM53_TEST_SPEC_K", "1"))  # the speculating server's K (MTP: K=1 / K=3; dflash: K=7)
# the speculative method: "mtp" (the MTP head) or "dflash" (the DFlash2 kind: GLM53_SPEC_KIND=dflash2; in the CPU mock
# both are the host speculative step, so the cell checks the plugin route / adapter contract of each method)
SPEC_METHOD = os.environ.get("GLM53_TEST_SPEC_METHOD", "mtp")
SPEC_FACTORY = "models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack:build_mock_spec_generator"

selftest_runner = pytest.importorskip("vllm_tt_plugin.selftest.runner")
selftest_score = pytest.importorskip("vllm_tt_plugin.selftest.score")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(tmp: Path, name: str, *, spec: bool):
    port = _free_port()
    log = tmp / f"server_{name}.log"
    args = [SNAPSHOT, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(port), "--block-size", "64",
            "--max-num-seqs", "8", "--max-num-batched-tokens", "512", "--no-enable-prefix-caching",
            "--shutdown-timeout", "30", "--max-logprobs", "-1",
            "--additional-config", json.dumps({"tt": {"sample_on_device_mode": "all", "trace_mode": "all"}})]  # fmt: skip
    # GLM53_TEST_ASYNC_SCHED=1: asynchronous scheduling on (a GLM53_SPEC_ASYNC=1 model declares
    # supports_async_spec_decode; without it the speculating launch must run --no-async-scheduling, as served)
    if os.environ.get("GLM53_TEST_ASYNC_SCHED") != "1":
        args.append("--no-async-scheduling")
    # GLM53_TEST_SERVER_EXTRA: the served flags to reproduce (n29df2e: "--scheduling-policy priority"; a later
    # --max-num-seqs overrides the 8 above)
    args += os.environ.get("GLM53_TEST_SERVER_EXTRA", "").split()
    env = dict(os.environ, GLM53_MOCK_MARKER=str(tmp / f"marker_{name}.jsonl"), HF_HUB_OFFLINE="1",
               GLM53_VLLM_EVENTS=str(tmp / f"events_{name}.jsonl"),
               VLLM_NO_USAGE_STATS="1", VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="60", GLM53_SELFTEST="off")  # fmt: skip
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    if spec:
        args += ["--speculative-config", json.dumps({"method": SPEC_METHOD, "num_speculative_tokens": SPEC_K})]
        env.update(GLM53_SPEC_K=str(SPEC_K), GLM53_VLLM_GENERATOR_FACTORY=SPEC_FACTORY)
        if SPEC_METHOD == "dflash":
            env.update(GLM53_SPEC_KIND="dflash2")
    else:
        env.pop("GLM53_SPEC_K", None)
    f = log.open("w")
    proc = subprocess.Popen([sys.executable, "-m", "models.autoports.zai_org_glm_5_3_flash.tests.vllm.mock_server", *args],
                            stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)  # fmt: skip
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"mock server {name} exited rc={proc.returncode}; log {log}:\n{log.read_text()[-4000:]}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
        pytest.fail(f"mock server {name} not healthy in 300 s; log {log}")
    return dict(base=base, proc=proc, log=log, f=f, events=tmp / f"events_{name}.jsonl")


def _adapter_counter(s, key: str) -> int:
    """the adapter's counter `key` in the server's latest event report (GLM53_VLLM_EVENTS rows carry
    report.adapter: e.g. spec_plain_verifies = verify blocks computed by plain sub-steps)"""
    best = 0
    if s["events"].exists():
        for line in s["events"].read_text().splitlines():
            try:
                rep = json.loads(line).get("report") or {}
            except ValueError:
                continue
            best = max(best, int((rep.get("adapter") or {}).get(key, 0) or 0))
    return best


def _stop(s):
    s["proc"].send_signal(signal.SIGINT)
    try:
        s["proc"].wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(s["proc"].pid, signal.SIGKILL)
        pytest.fail(f"mock server did not shut down within 120 s of SIGINT ({s['log']})")
    s["f"].close()


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mock_spec_servers")
    plain = _start(tmp, "plain", spec=False)
    mtp = _start(tmp, "mtp", spec=True)
    yield dict(plain=plain, mtp=mtp)
    _stop(mtp)
    _stop(plain)


def _metric(base: str, needle: str) -> float:
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        text = r.read().decode()
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
               if line and not line.startswith("#") and needle in line.split("{")[0].split(" ")[0])  # fmt: skip


def _post(base: str, body: dict) -> dict:
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def test_selftest_plan_logprobs_are_bitwise_the_plain_server_s(servers):
    """The self-test's own requests (logprobs=100, greedy, teacher-forced) on the MTP server == the plain server."""
    cfg = selftest_runner.Config(mode="strict", concurrency=1, request_timeout_s=600.0)
    plan = selftest_score.build_plan(selftest_score.load_pack(cfg.pack_path), tf_steps=cfg.tf_steps)
    obs = {}
    for name in ("plain", "mtp"):
        o, _ = selftest_runner.run_http(servers[name]["base"], MODEL, plan, cfg, logprobs=100)
        obs[name] = o
    assert set(obs["plain"]) == set(obs["mtp"]) == {r.name for r in plan}
    errors = {k: v.error for k, v in obs["mtp"].items() if v.error}
    assert not errors, f"the MTP server refused / failed self-test requests: {errors}"
    for k in obs["plain"]:
        p, m = obs["plain"][k], obs["mtp"][k]
        assert p.error is None
        assert m.token_ids == p.token_ids, k
        assert m.top == p.top, k  # every position's top-100 (ids, logprob): bitwise
    assert len(plan) >= 30


def test_greedy_without_logprobs_still_speculates_and_matches(servers):
    """Drafting is withheld only from logprobs requests: a plain greedy request speculates on the MTP server."""
    body = dict(model=MODEL, prompt=list(range(1000, 1161)), max_tokens=24, temperature=0.0, ignore_eos=True,
                return_token_ids=True)  # fmt: skip
    before = _metric(servers["mtp"]["base"], "spec_decode_num_draft_tokens")
    outs = {name: _post(servers[name]["base"], body) for name in ("plain", "mtp")}
    after = _metric(servers["mtp"]["base"], "spec_decode_num_draft_tokens")
    assert outs["mtp"]["choices"][0]["text"] == outs["plain"]["choices"][0]["text"]
    assert after > before, "the MTP server did not draft for a greedy request"


def test_drafting_stops_for_a_second_user_and_resumes_alone(servers):
    """The width policy: the adapter speculates for a LONE user only (spec_enter needs slot 0 to be the only live
    slot; propose offers no drafts with a second live row), so a speculating step has K + 1 <= 4 verify rows and any
    two or more users decode plainly. User A runs alone (drafts), B arrives mid-stream (drafting off for both), B
    finishes and A continues alone (drafting back on): both outputs equal the plain server's, token for token. B's
    arrival is the n29df2e crash shape (2026-10-23 07:11Z): the runner may schedule A's next block from a proposal
    made while A was alone; the adapter binds each row's own drafts (tt/generator_vllm.py _verify_plain)."""
    import concurrent.futures as cf

    a = dict(model=MODEL, prompt=list(range(2000, 2161)), max_tokens=96, temperature=0.0, ignore_eos=True,
             return_token_ids=True)  # fmt: skip
    b = dict(model=MODEL, prompt=list(range(3000, 3100)), max_tokens=16, temperature=0.0, ignore_eos=True,
             return_token_ids=True)  # fmt: skip
    outs = {}
    for name in ("plain", "mtp"):
        base = servers[name]["base"]
        before = _metric(base, "spec_decode_num_draft_tokens") if name == "mtp" else 0.0
        with cf.ThreadPoolExecutor(2) as ex:
            fa = ex.submit(_post, base, a)
            time.sleep(1.5)  # A decodes alone first
            fb = ex.submit(_post, base, b)
            outs[name] = (fa.result()["choices"][0], fb.result()["choices"][0])
        if name == "mtp":
            assert _metric(base, "spec_decode_num_draft_tokens") > before, "A never speculated"
    (pa, pb), (ma, mb) = outs["plain"], outs["mtp"]
    assert ma["text"] == pa["text"] and mb["text"] == pb["text"]
    assert ma.get("token_ids") == pa.get("token_ids") and mb.get("token_ids") == pb.get("token_ids")


def test_sixteen_staggered_users_match_the_plain_server(servers):
    """8-16 concurrent users (the re-scoped serve, 2026-10-23): user 0 speculates alone first, then 15 more arrive
    staggered (more than --max-num-seqs 8: arrivals, finishes and slot remaps while blocks with drafts are in flight),
    every 3rd with logprobs (a plain row inside any verify block). Every output == the plain server's, token for
    token; the speculating server stays up (no 'verify block' failure) and drafted while user 0 was alone."""
    import concurrent.futures as cf

    bodies = []
    for u in range(16):
        b = dict(model=MODEL, prompt=list(range(4000 + 37 * u, 4000 + 37 * u + 60 + 7 * u)), max_tokens=24 + 3 * (u % 5),
                 temperature=0.0, ignore_eos=True, return_token_ids=True)  # fmt: skip
        if u % 3 == 2:
            b["logprobs"] = 2
        bodies.append(b)
    bodies[0]["max_tokens"] = 80
    outs = {}
    for name in ("plain", "mtp"):
        base = servers[name]["base"]
        before = _metric(base, "spec_decode_num_draft_tokens") if name == "mtp" else 0.0
        with cf.ThreadPoolExecutor(16) as ex:
            futs = [ex.submit(_post, base, bodies[0])]
            time.sleep(1.5)  # user 0 decodes alone first (drafts)
            for b in bodies[1:]:
                futs.append(ex.submit(_post, base, b))
                time.sleep(0.15)
            outs[name] = [f.result()["choices"][0] for f in futs]
        if name == "mtp":
            assert _metric(base, "spec_decode_num_draft_tokens") > before, "user 0 never speculated"
            assert servers[name]["proc"].poll() is None, "the speculating server died"
    for u, (p, m) in enumerate(zip(outs["plain"], outs["mtp"])):
        assert m.get("token_ids") == p.get("token_ids") and m["text"] == p["text"], f"user {u} differs"
    # which verify path ran (review round 4 P2-4): blocks computed by plain sub-steps (an arrival after the lone
    # proposal, a remap) -- reported; GLM53_TEST_REQUIRE_PLAIN_VERIFY=1 makes it a bar
    pv = _adapter_counter(servers["mtp"], "spec_plain_verifies")
    print(f"[{SPEC_METHOD} K={SPEC_K}] spec_plain_verifies={pv} spec_steps={_adapter_counter(servers['mtp'], 'spec_steps')}")
    if os.environ.get("GLM53_TEST_REQUIRE_PLAIN_VERIFY") == "1":
        assert pv > 0, "no verify block went through _verify_plain"


def test_drafting_off_ends_and_multi_to_lone_keep_the_server_up(servers):
    """(the k4b twins, 2026-10-27 09:29Z / 09:42Z: the speculating engine died under async scheduling on propose's
    guard) the two shapes that killed it: (1) a drafting-off request (temperature 1, top_k 1: argmax, but sampled for
    the plugin) decoded on overlapped plain steps to its max_tokens after greedy speculating requests; (2) greedy users
    started together whose last survivor goes lone after overlapped plain steps. Run with GLM53_TEST_ASYNC_SCHED=1 and
    GLM53_SPEC_ASYNC=1 (the served k4 line). Every output == the plain server's; the speculating server stays up and
    drafted again after both shapes."""
    import concurrent.futures as cf

    g = dict(model=MODEL, max_tokens=40, temperature=0.0, ignore_eos=True, return_token_ids=True)
    seq = [dict(g, prompt=list(range(5000, 5161))),
           dict(g, prompt=list(range(5200, 5361)), temperature=1.0, top_k=1, max_tokens=72),
           dict(g, prompt=list(range(5400, 5561)))]  # fmt: skip
    multi = [dict(g, prompt=list(range(6000 + 211 * u, 6000 + 211 * u + 120 + 9 * u)), max_tokens=24 + 16 * u)
             for u in range(4)]  # fmt: skip
    outs = {}
    for name in ("plain", "mtp"):
        base = servers[name]["base"]
        o = [_post(base, b)["choices"][0] for b in seq]
        with cf.ThreadPoolExecutor(4) as ex:
            o += [f.result()["choices"][0] for f in [ex.submit(_post, base, b) for b in multi]]
        before = _metric(base, "spec_decode_num_draft_tokens") if name == "mtp" else 0.0
        o.append(_post(base, dict(g, prompt=list(range(7000, 7161))))["choices"][0])  # a lone greedy request after
        if name == "mtp":
            assert servers[name]["proc"].poll() is None, "the speculating server died"
            assert _metric(base, "spec_decode_num_draft_tokens") > before, "no drafting after the two shapes"
        outs[name] = o
    for i, (p, m) in enumerate(zip(outs["plain"], outs["mtp"])):
        assert m.get("token_ids") == p.get("token_ids") and m["text"] == p["text"], f"request {i} differs"
    print(f"[{SPEC_METHOD} K={SPEC_K}] spec_declines_inflight={_adapter_counter(servers['mtp'], 'spec_declines_inflight')}")
