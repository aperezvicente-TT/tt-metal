# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-loop timing of one served decode step, in the EngineCore process (program 2026-10-20 23:32:19Z: where do
the host milliseconds of a served B=1 step go?). Installed by tests/vllm/stack_prefix_backend.py when
GLM53_HOSTLOOP_PROBE=1; one JSONL row per wrapped call (name, start, wall s, thread) to $GLM53_HOSTLOOP_JSONL.

Wrapped (whatever exists): vLLM EngineCore.step / step_with_batch_queue (the whole engine iteration), the TT
scheduler's schedule / update_from_output, the TT model runner's _prepare_model_inputs / execute_model / sample_tokens,
the adapter's decode_forward / read_decode_output, Glm53Generator.decode_forward / read_decode_output, and the
backend's replay (trace enqueue), read / read_async / finish_read (device -> host copies; a blocking read includes
waiting for the trace). On the mock cluster the backend's replay / read stand for device work done on the CPU, so
only the host-side rows are representative there."""

from __future__ import annotations

import functools
import json
import os
import threading
import time

PROBE_ENV = "GLM53_HOSTLOOP_PROBE"
JSONL_ENV = "GLM53_HOSTLOOP_JSONL"
_lock = threading.Lock()
_installed = False


def _emit(row: dict) -> None:
    path = os.environ.get(JSONL_ENV)
    if path:
        with _lock, open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _wrap(cls, name: str, tag: str, done: list) -> None:
    fn = cls.__dict__.get(name) or getattr(cls, name, None)
    if fn is None or getattr(fn, "_glm53_hostloop", False):
        return

    @functools.wraps(fn)
    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _emit({"name": tag, "t0": t0, "wall_s": time.perf_counter() - t0, "thread": threading.current_thread().name})

    wrapper._glm53_hostloop = True
    setattr(cls, name, wrapper)
    done.append(tag)


def install() -> list[str]:
    global _installed
    if _installed:
        return []
    _installed = True
    done, errors = [], {}
    targets = (
        ("vllm.v1.engine.core", "EngineCore", ("step", "step_with_batch_queue")),
        ("vllm_tt_plugin.scheduler", "TTScheduler", ("schedule", "update_from_output")),
        ("vllm_tt_plugin.model_runner", "TTModelRunner", ("_prepare_model_inputs", "execute_model", "sample_tokens")),
        ("models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm", "Glm5NextForCausalLM",
         ("decode_forward", "read_decode_output")),
        ("models.autoports.zai_org_glm_5_3_flash.tt.generator", "Glm53Generator", ("decode_forward", "read_decode_output")),
        ("models.autoports.zai_org_glm_5_3_flash.tt.stack_model", "StackBackend",
         ("replay", "read", "read_async", "finish_read")),
    )  # fmt: skip
    import importlib

    for mod, cls_name, names in targets:
        try:
            cls = getattr(importlib.import_module(mod), cls_name)
            for n in names:
                _wrap(cls, n, f"{cls_name}.{n}", done)
        except Exception as e:  # never stop the server
            errors[f"{mod}.{cls_name}"] = repr(e)[:160]
    _emit({"name": "installed", "wrapped": done, "errors": errors, "pid": os.getpid(), "t0": time.perf_counter()})
    return done


def install_from_env() -> list[str]:
    return install() if os.environ.get(PROBE_ENV) == "1" else []
