# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 cf3f (program 2026-10-23T18:11:19Z): StackBackend.capture_decode captures the speculative traces
(SpecMTP.capture) at the FIRST decode capture -- after the prefill keys' eager warm, ahead of every decode trace and
every TP-prefill key -- instead of lazily at the first speculative request (n29h7s2 cf3e run 1756: the lazy spec
program compiled after every other capture and its static CBs met an L1 buffer allocated after the decode captures).
Host only: the backend's capture path on recording fakes (no device, no trace)."""
import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM


class _Spec:
    def __init__(self, log):
        self.log, self.traces = log, {}

    def capture(self):
        if not self.traces:
            self.log.append("spec")
            self.traces = {"step": 1, "boot": 2}
        return self.traces


class _Tpp:
    def __init__(self, log, kv):
        self.log, self.kv, self._warmed, self.traces = log, kv, False, {}

    def warm(self):
        self.log.append("prefill warm")
        self._warmed = True

    def capture(self, *a, **k):
        self.log.append("prefill capture")
        self.traces = {"k": 1}


def _backend(log, kv, *, spec=True, tpp=True):
    be = object.__new__(SM.StackBackend)
    be.native_widths, be.narrow_widths = [8, 16], [1]
    be.n_chunks, be.prefill_traces, be.traces, be.mesh = 1, False, [], None
    be.spec = _Spec(log) if spec else None
    be.tpp = _Tpp(log, kv) if tpp else None
    be._capture = lambda kv_, chunks, width=None: log.append(f"decode w{width}") or f"tr{width}"
    return be


@pytest.fixture(autouse=True)
def _device_mode(monkeypatch):
    monkeypatch.setattr(SM, "MOCK", False)  # the device order (the mock records spec traces without capturing)
    monkeypatch.delenv("GLM53_L1_REPORT", raising=False)
    monkeypatch.delenv("GLM53_PF_CAPTURE_REPORT", raising=False)


def test_spec_traces_captured_before_every_decode_trace_and_prefill_key():
    log, kv = [], type("KV", (), {"extra": {}})()
    be = _backend(log, kv)
    for w in (1, 8, 16):  # the 16-seat launch: widths 1 / 8 / 16, the prefill keys after the largest native width
        be.capture_decode(None, kv, w)
    assert log == ["prefill warm", "spec", "decode w1", "decode w8", "decode w16", "prefill capture"], log


def test_spec_capture_once_and_no_spec_backend_unchanged():
    log, kv = [], type("KV", (), {"extra": {}})()
    be = _backend(log, kv)
    be.capture_decode(None, kv, 1)
    be.capture_decode(None, kv, 8)
    assert log.count("spec") == 1, log
    log2, kv2 = [], type("KV", (), {"extra": {}})()
    be2 = _backend(log2, kv2, spec=False)
    for w in (1, 8, 16):
        be2.capture_decode(None, kv2, w)
    assert log2 == ["prefill warm", "decode w1", "decode w8", "decode w16", "prefill capture"], log2


def test_spec_capture_without_tp_prefill():
    log, kv = [], type("KV", (), {"extra": {}})()
    be = _backend(log, kv, tpp=False)
    be.capture_decode(None, kv, 8)
    assert log == ["spec", "decode w8"], log
