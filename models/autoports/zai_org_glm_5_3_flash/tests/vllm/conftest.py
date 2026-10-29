# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only vLLM adapter tests (09a). Run with the venv-vllm interpreter under no-tt-devices and
`--confcutdir` set to this directory (the tt-metal root conftest opens devices).

Each test's outcome plus the evidence it records with `evidence(**fields)` is appended to the JSONL file named
by GLM53_09A_TESTS_JSONL (one row per test: test, nodeid, status, duration_s, details)."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger

# The host suites model the Stage 6 generator's dense-exact DSA regime (the mock stack serves 2,051 positions per
# request, glm53_mock_stack.py): they run against that regime's program contract, pinned here before any test imports
# tt/generator_vllm.py (the class capability is read at import, also by the HTTP tests' server subprocesses). The served
# contract, doc/context_contract.json (32,768 tokens, device-DRAM bound), has its own tests
# (test_generator_vllm.py::test_served_context_contract_*). GLM53_TEST_CONTEXT_CONTRACT selects another contract.
DENSE_EXACT_CONTRACT = Path(__file__).parent / "contracts" / "context_contract_dense_exact_2051.json"
os.environ["GLM53_CONTEXT_CONTRACT"] = os.environ.get("GLM53_TEST_CONTEXT_CONTRACT", str(DENSE_EXACT_CONTRACT))

SNAPSHOT = os.environ.get(
    "GLM53_HF_SNAPSHOT",
    "/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/"
    "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
)


@pytest.fixture(scope="session")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(SNAPSHOT)


@pytest.fixture(scope="session")
def text_config(hf_config):
    return hf_config.text_config


@pytest.fixture
def expect_error():
    """The tt-metal root conftest's `expect_error` (that conftest is cut off here: it queries the device arch).
    Marks expected errors in the log for CI triage; `message` is a regex matched against the error text."""

    @contextlib.contextmanager
    def expect_error_(error, message):
        names = ", ".join(e.__name__ for e in (error if isinstance(error, tuple) else (error,)))
        logger.info(f'[EXPECTED_ERROR BEGIN] {names} message="{message}"')
        try:
            with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture itself
                yield exc_info
        finally:
            logger.info(f'[EXPECTED_ERROR END] {names} message="{message}"')

    return expect_error_


@pytest.fixture
def lifted_contract(monkeypatch, tmp_path):
    """doc/context_contract.json as it will read once the sparse DSA path lands (current_supported_context = target,
    no request limit): the adapter's contract loader and the class capability both see it, in this process."""
    from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv

    data = json.loads(Path(gv.CONTEXT_CONTRACT_PATH).read_text())
    data["current_supported_context"] = data["target_context"]
    path = tmp_path / "context_contract_lifted.json"
    path.write_text(json.dumps(data))
    load = gv.load_context_contract
    monkeypatch.setattr(gv, "load_context_contract", lambda p=path: load(p))
    monkeypatch.setitem(gv.Glm5NextForCausalLM.model_capabilities, "tt_request_context_limit", 0)
    return path


@pytest.fixture
def evidence(request):
    """Attach JSON-serialisable evidence to this test's tests.jsonl row."""
    store: dict = {}
    request.node.user_properties.append(("evidence", store))

    def add(**fields):
        store.update(fields)

    return add


def pytest_runtest_logreport(report):
    path = os.environ.get("GLM53_09A_TESTS_JSONL")
    if not path:
        return
    if report.when == "call" or (report.when in ("setup", "teardown") and report.outcome != "passed"):
        details = {}
        for key, value in report.user_properties:
            if key == "evidence":
                details.update(value)
        name = report.nodeid.split("::")[-1] + ("" if report.when == "call" else f"[{report.when}]")
        row = {
            "test": name,
            "nodeid": report.nodeid,
            "status": "passed" if report.passed else ("skipped" if report.skipped else "failed"),
            "duration_s": round(report.duration, 3),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suite": os.environ.get("GLM53_09A_SUITE", "adapter"),
            # code under test: adapter / 06a generator / plugin / vLLM revisions (scripts/run_09a_tests.sh)
            "provenance": json.loads(os.environ.get("GLM53_09A_PROVENANCE", "{}")),
            "details": details,
        }
        if report.failed:
            row["error"] = str(report.longrepr)[-2000:]
        with Path(path).open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
