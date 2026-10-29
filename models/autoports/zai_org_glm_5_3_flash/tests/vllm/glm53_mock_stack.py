# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The generator stack the 09a host-only vLLM tests serve: 06a's `tt/generator.py::Glm53Generator` (the real generator
the adapter delegates to) over 06a's CPU `tests/mock_model.py::MockModel` backend (Stage-1-contract mock decoders,
paged MLA latents, per-slot KDA state, mock trace capture/replay that raises on any compile while a trace is live,
NaN-poisoned inactive rows, and an independent unpaged `reference_greedy` oracle).

`RecordingMockModel` only adds an event log (prefill chunks, decode steps, slot copies) for the mixed-shape evidence.
`build_mock_generator` has the signature of `tt/generator.py::build_generator` and is what
`GLM53_VLLM_GENERATOR_FACTORY` points at. The generator and the mocks come from the same tree as the adapter
(Stage 9 runner: doc/ttnn/09-vllm/scripts/run_host_tests.sh).
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch

from models.autoports.zai_org_glm_5_3_flash.tests.mock_model import TOKENIZER_VOCAB, MockModel
from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator

KINDS = ("dense_kda", "sparse_dsa", "sparse_kda")
# Default: the generator in its current regime (dense-exact DSA, 2051 positions per request, doc/context_contract.json)
# with its latent pool sized from vLLM's num_blocks; "unbounded" = the lifted bound (sparse DSA, long-context tests).
DSA_MODE_ENV = "GLM53_MOCK_DSA_MODE"
MARKER_ENV = "GLM53_MOCK_MARKER"
# "1": the mock declares full-width decode traces like the 8-chip pipeline (tt/model.py width_bucketed_decode False)
FULL_WIDTH_ENV = "GLM53_MOCK_FULL_WIDTH"
# e.g. "1,2,4": the mock declares per-width traces like tt/per_width.py PerWidthBackend (copy_slot re-seeds)
PER_WIDTH_ENV = "GLM53_MOCK_PER_WIDTH"
# "1": the decode trace also samples (ModelBackend.samples_in_decode_trace: one replay per step, the Stage 14 shape)
FUSED_SAMPLING_ENV = "GLM53_MOCK_FUSED_SAMPLING"
# "1": a greedy-only device sampler (max_device_top_k 1; a row with k > 1 and temp > 0 is refused, as the Stage 14b stack)
GREEDY_ONLY_ENV = "GLM53_MOCK_GREEDY_ONLY"
# "<bucket>,<nb>": the mock reports a captured batched-admission trace like Stage 16's StackBackend (GLM53_PF_BATCH):
# prefill_batch_size returns it and prefill_batch_tp prefills a group of nb fresh requests in ONE call (the mock runs
# each request's own chunk plan in turn, so its KV writes and first tokens equal the per-user path's)
PF_BATCH_ENV = "GLM53_MOCK_PF_BATCH"
PF_PACKED_ENV = "GLM53_MOCK_PF_PACKED"  # "T,KMAX": StackBackend.prefill_packed_size (Stage 16's GLM53_PF_PACKED=T:KMAX)
# "continuation,no_head": StackBackend.prefill_packed_caps -- packed segments may continue a slot's cached prefix and
# may be head-less (a long prompt's intermediate chunk): the whole vLLM step in one packed replay
PF_PACKED_CAPS_ENV = "GLM53_MOCK_PF_PACKED_CAPS"
# "<n>": the mock raises MockDeviceError at its n-th decode step (1-based) -- a device fault in a served step
# (serving robustness: the plugin must latch unhealthy, the engine must exit non-zero, nothing may retry)
FAULT_DECODE_STEP_ENV = "GLM53_MOCK_FAULT_DECODE_STEP"

REGISTRY: list = []  # every generator built in this process (in-process engine tests read the live one)
_TOKENIZER = None


class SlotLedgerError(AssertionError):
    pass


class SlotLedger:
    """The mock device's own record of which slots hold a live request, independent of the generator's and the
    plugin's bookkeeping (serving robustness: no slot leak, no double release, no fresh prefill into a held slot).
    `occupy` at a prefill from position 0 (the slot's KDA state was just zeroed: `resets` counts it), `move` on a
    compaction copy, `release` on the adapter's release. A violation raises at once (the test fails at the step
    that leaked) and is kept in `violations`."""

    def __init__(self):
        self.held: dict[int, dict] = {}  # slot -> {"key": first tokens, "step": serial}
        self.serial = 0
        self.occupies = self.releases = self.moves = self.resets = 0
        self.violations: list[str] = []
        self.history: list[tuple] = []

    def _fail(self, msg):
        self.violations.append(msg)
        raise SlotLedgerError(msg)

    def occupy(self, slot: int, key, reset_seen: bool):
        self.serial += 1
        if slot in self.held:
            self._fail(f"fresh prefill into slot {slot} still held by {self.held[slot]} (not released): {key}")
        if not reset_seen:
            self._fail(f"fresh prefill into slot {slot} without zeroing its KDA state first")
        self.held[slot] = {"key": key, "serial": self.serial}
        self.occupies += 1
        self.history.append(("occupy", slot, key))

    def release(self, slot: int):
        if slot not in self.held:
            self._fail(f"double release of slot {slot} (held: {sorted(self.held)})")
        self.history.append(("release", slot, self.held.pop(slot)["key"]))
        self.releases += 1

    def move(self, src: int, dst: int):
        if src not in self.held:
            self._fail(f"compaction move from an unheld slot {src}")
        if dst in self.held:
            self._fail(f"compaction move {src} -> {dst} onto a held slot")
        self.held[dst] = self.held.pop(src)
        self.moves += 1
        self.history.append(("move", src, dst))

    def clear(self):
        self.held.clear()

    def summary(self) -> dict:
        return {"held": sorted(self.held), "occupies": self.occupies, "releases": self.releases, "moves": self.moves,
                "resets": self.resets, "violations": list(self.violations)}  # fmt: skip


class RecordingMockModel(MockModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[dict] = []
        self.ledger = SlotLedger()
        self._reset_pending: set[int] = set()  # slots zeroed since their last occupy
        self._imported: set[int] = set()  # slots whose state a connector / host-tier import just wrote
        self._decode_steps = 0
        v = os.environ.get(FAULT_DECODE_STEP_ENV, "")
        self.fault_decode_step = int(v) if v else 0
        # the zeroing of a slot's KDA state happens per layer (MockDecoder.reset_slot, called by begin_prefill_chunk
        # at chunk_start 0 and by MockModel.reset_slot / reset_all): observe it there, so the ledger can require it
        # before every fresh occupy
        for layer in self.layers:
            layer.reset_slot = self._observe_layer_reset(layer, layer.reset_slot)

    def _observe_layer_reset(self, layer, original):
        def reset_slot(cache, slot):
            original(cache, slot)
            if layer.is_kda:
                self.ledger.resets += 1
                self._reset_pending.add(int(slot))
                self.events.append({"event": "reset_slot", "slot": int(slot), "layer": layer.layer_idx})

        return reset_slot

    def reset_all(self, kv):
        super().reset_all(kv)
        self.ledger.clear()

    def _occupy(self, slot, tokens, n_valid, reset_seen=None):
        """the ledger's fresh occupy of `slot` (every path that starts a request: the per-chunk prefill, the batched
        and packed admissions, a prefix-cache attach / connector import whose state overwrites the slot's wholesale)"""
        key = tuple(int(t) for t in tokens[: min(8, int(n_valid))].tolist())
        seen = int(slot) in self._reset_pending if reset_seen is None else reset_seen
        self.ledger.occupy(int(slot), key, seen)
        self._reset_pending.discard(int(slot))

    def release_slot(self, kv, slot):
        self.events.append({"event": "release_slot", "slot": int(slot)})
        self.ledger.release(int(slot))

    def latent_pages(self, kv_cache):
        # the mock's own layer indices (3 layers: L1 is its sparse_dsa); latent [num_blocks, 64, dl] torch
        return {i: c.latent for i, (l, c) in enumerate(zip(self.layers, kv_cache.caches)) if not l.is_kda}

    def export_kda_state(self, kv_cache, slot, n_tokens=None, source="live"):  # StackBackend's layout: [{L, S, C}]
        """source="snapshot" (one-replay store A): the KDA state at the cut n_tokens that prefill_chunk(snap_at=) kept,
        consumed by the export (StackBackend's kda_snap_at tag); "live": the slot's current state"""
        if source == "snapshot":
            held = self._kda_snap.get(int(slot))
            if held is None or held[0] != int(n_tokens):
                raise ValueError(f"slot {slot}: no in-replay snapshot at {n_tokens} (held {held and held[0]})")
            del self._kda_snap[int(slot)]
            return held[1]
        return [
            dict(L=i, S=c.state[slot].clone(), C=c.conv[slot].clone())
            for i, (l, c) in enumerate(zip(self.layers, kv_cache.caches))
            if l.is_kda
        ]

    @property
    def kda_checkpoint_in_replay(self) -> bool:
        """one-replay store A on the mock (GLM53_PF_KDA_SNAPSHOT=1, as StackBackend): prefill_chunk(snap_at=) keeps a
        storing row's KDA state at its checkpoint cut, export_kda_state(source="snapshot") returns it"""
        return os.environ.get("GLM53_PF_KDA_SNAPSHOT", "0") == "1"

    def kda_snapshot_holds(self, kv_cache, slot, n_tokens):
        held = self._kda_snap.get(int(slot))
        return held is not None and held[0] == int(n_tokens)

    @property
    def _kda_snap(self) -> dict:
        return self.__dict__.setdefault("_kda_snap_held", {})

    def import_kda_state(self, kv_cache, slot, states):
        self._imported.add(int(slot))  # a fresh request's state, written wholesale (a virtual slot never prefills)
        by_l = {int(d["L"]): d for d in states}
        for i, (l, c) in enumerate(zip(self.layers, kv_cache.caches)):
            if l.is_kda:
                c.state[slot].copy_(by_l[i]["S"])
                c.conv[slot].copy_(by_l[i]["C"])

    def prefill_batch_size(self):
        """(bucket, nb) of the batched-admission trace (GLM53_MOCK_PF_BATCH), else (0, 0): StackBackend's contract"""
        v = os.environ.get(PF_BATCH_ENV, "")
        return tuple(int(x) for x in v.split(",")) if v else (0, 0)

    def prefill_batch_tp(self, reqs, *, kv_cache, io, sample=None):
        """StackBackend.prefill_batch_tp's contract on the mock: reqs = [(tokens, slot, n_valid)] fresh requests (chunk
        start 0, n_valid <= the bucket, distinct slots), `sample` one SamplingParams per request (or one for all) ->
        each request's first token, sampled at its last position into its decode token row"""
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import plan_prefill

        cap, nb = self.prefill_batch_size()
        slots = [int(s) for _, s, _ in reqs]
        assert len(reqs) == nb and len(set(slots)) == nb, (slots, nb)
        assert all(1 <= int(nv) <= cap for _, _, nv in reqs), [int(nv) for _, _, nv in reqs]
        sps = list(sample) if isinstance(sample, (list, tuple)) else [sample] * len(reqs)
        self.events.append({"event": "prefill_batch", "slots": slots, "n": [int(nv) for _, _, nv in reqs],
                            "temperature": [float(getattr(sp, "temperature", 0.0) or 0.0) for sp in sps],
                            "seed": [getattr(sp, "seed", None) for sp in sps]})  # fmt: skip
        first = []
        for (tk, slot, nv), sp in zip(reqs, sps):
            plans = plan_prefill(int(nv), 0, self.prefill_chunk_tokens, self.prefill_align)
            for i, pl in enumerate(plans):
                last = i == len(plans) - 1
                chunk = torch.zeros(pl.physical, dtype=torch.int64)
                chunk[: pl.n_valid] = tk[pl.offset : pl.offset + pl.n_valid]
                out = super().prefill_chunk(chunk, kv_cache=kv_cache, slot=int(slot), chunk_start=pl.start,
                                            n_valid=pl.n_valid, logits_rows="last" if last else "none",
                                            sample=sp if last else None, io=io)  # fmt: skip
                if pl.start == 0:  # the request starts in its slot (zeroed by begin_prefill_chunk)
                    self._occupy(slot, chunk, pl.n_valid)
            first.append(int(out))
        return first

    def prefill_packed_size(self):
        """(T, KMAX) of the packed-prefill traces (GLM53_MOCK_PF_PACKED), else (0, 0): StackBackend's contract"""
        v = os.environ.get(PF_PACKED_ENV, "")
        return tuple(int(x) for x in v.split(",")) if v else (0, 0)

    def prefill_packed_caps(self):
        """the packed call's segment kinds beyond fresh + headed (GLM53_MOCK_PF_PACKED_CAPS): StackBackend's contract"""
        return {c: True for c in os.environ.get(PF_PACKED_CAPS_ENV, "").split(",") if c}

    def prefill_packed_tp(self, reqs, *, kv_cache, io, t=2048, sample=None, starts=None, heads=None):
        """StackBackend.prefill_packed_tp's contract on the mock: reqs = [(tokens, slot, n_valid)] fresh requests of
        DIFFERENT lengths packed back to back (each segment rounded up to 32 rows, the sum <= t, 2 <= K <= KMAX,
        distinct slots), `sample` one SamplingParams per request (or one for all) -> each request's first token, sampled
        at its last position into its decode token row (computed per request: the packed attention is block-diagonal)"""
        tt_, kmax = self.prefill_packed_size()
        caps = self.prefill_packed_caps()
        starts = [int(x) for x in (starts or [0] * len(reqs))]
        heads = [bool(x) for x in (heads or [True] * len(reqs))]
        assert caps.get("continuation") or not any(starts), ("continuation segments need the capability", starts)
        assert caps.get("no_head") or all(heads), ("head-less segments need the capability", heads)
        slots = [int(s) for _, s, _ in reqs]
        assert t == tt_ and 2 <= len(reqs) <= kmax and len(set(slots)) == len(reqs), (t, tt_, slots, kmax)
        assert sum(-(-int(nv) // 32) * 32 for _, _, nv in reqs) <= t, [int(nv) for _, _, nv in reqs]
        sps = list(sample) if isinstance(sample, (list, tuple)) else [sample] * len(reqs)
        self.events.append({"event": "prefill_packed", "slots": slots, "n": [int(nv) for _, _, nv in reqs], "t": t,
                            "starts": starts, "heads": heads,
                            "temperature": [float(getattr(sp, "temperature", 0.0) or 0.0) for sp in sps]})  # fmt: skip
        first = []
        for (tk, slot, nv), sp, st, hd in zip(reqs, sps, starts, heads):
            chunk = torch.zeros(-(-int(nv) // self.prefill_align) * self.prefill_align, dtype=torch.int64)
            chunk[: int(nv)] = tk[: int(nv)]
            out = super().prefill_chunk(chunk, kv_cache=kv_cache, slot=int(slot), chunk_start=st, n_valid=int(nv),
                                        logits_rows="last" if hd else "none", sample=sp if hd else None, io=io)  # fmt: skip
            if st == 0:  # a fresh segment starts its request (a continuation segment keeps the held slot)
                self._occupy(slot, chunk, nv)
            first.append(int(out) if hd else -1)
        return first

    def prefill_segments(self, chunk_start, n_valid):
        """segtrace on the mock (GLM53_PF_SEGMENTS = S): a chunk of at least GLM53_PREFILL_SEGMENT_MIN_TOKENS (default 513,
        the plugin's floor: the 2,048-bucket keys) runs in S segments, a shorter one whole"""
        n_seg = int(os.environ.get("GLM53_PF_SEGMENTS", "1") or 1)
        floor = int(os.environ.get("GLM53_PREFILL_SEGMENT_MIN_TOKENS", "513") or 513)
        return n_seg if n_seg > 1 and int(n_valid) >= floor else 1

    def _segment(self, tokens, slot, chunk_start, n_valid, segment, state_src, state_snap, snap_at) -> bool:
        """a segment of a segmented chunk: True = a non-final one (recorded, nothing computed). The mock computes the
        whole chunk at its last segment, so a segmented chunk's numerics are the unsegmented one's by construction
        (the device's bitwise equality is 14b's gate 2). Checks the contract the device stack asserts: every segment
        re-sends segment 0's chunk and its state_src / state_snap / snap_at; one open chunk at a time."""
        n_seg = self.prefill_segments(chunk_start, n_valid)
        open_ = self.__dict__.setdefault("_seg_open", {})
        key = (int(slot), int(chunk_start), int(n_valid), state_src, state_snap, snap_at)
        if segment == 0:
            assert not open_ or n_seg == 1, ("one open segmented chunk at a time", open_)
            if n_seg > 1:
                open_[int(slot)] = key
        else:
            assert open_.get(int(slot)) == key, ("a later segment must re-send segment 0's chunk", open_, key)
        rec = {"event": "prefill_segment", "slot": int(slot), "segment": int(segment), "n_seg": n_seg,
               "start": int(chunk_start), "n": int(n_valid)}  # fmt: skip
        self.events.append(rec)
        marker = os.environ.get(MARKER_ENV)
        if marker:
            with open(marker, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if int(segment) < n_seg - 1:
            return True
        open_.pop(int(slot), None)
        return False

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None,
                      state_src=None, state_snap=None, snap_at=None, segment=None):  # fmt: skip
        if segment is not None and self._segment(tokens, slot, chunk_start, n_valid, int(segment), state_src,
                                                 state_snap, snap_at):  # fmt: skip
            return None  # a non-final segment: its trace only on the device; the generator returns the placeholder
        self.events.append(
            {
                "event": "prefill_chunk",
                "slot": slot,
                "start": chunk_start,
                "n": n_valid,
                "bucket": int(tokens.numel()),
                "rows": logits_rows,
                "device_sample": sample is not None,
                "state_src": state_src,
                "state_snap": state_snap,
                "snap_at": snap_at,
            }
        )
        if snap_at is not None:
            out = self._prefill_chunk_snap(tokens, kv_cache, slot, chunk_start, n_valid, logits_rows, sample, io,
                                           state_src, state_snap, int(snap_at))  # fmt: skip
        else:
            out = super().prefill_chunk(
                tokens,
                kv_cache=kv_cache,
                slot=slot,
                chunk_start=chunk_start,
                n_valid=n_valid,
                logits_rows=logits_rows,
                sample=sample,
                io=io,
                state_src=state_src,
                state_snap=state_snap,
            )
        if chunk_start == 0:  # a fresh request: the slot's state was zeroed by begin_prefill_chunk (reset_slot)
            self._occupy(slot, tokens, n_valid)
        elif state_src is not None or int(slot) in self._imported:
            # a fresh request continuing a cached prefix: its state came wholesale from a snapshot entry (prefix-cache
            # attach) or a connector checkpoint (import_kda_state), not from the slot's previous owner
            self._imported.discard(int(slot))
            self._occupy(slot, tokens, n_valid, reset_seen=True)
        return out

    def _prefill_chunk_snap(self, tokens, kv_cache, slot, chunk_start, n_valid, logits_rows, sample, io, state_src,
                            state_snap, cut):  # fmt: skip
        """one-replay store A on the mock: the chunk [chunk_start, chunk_start + n_valid) with the KDA state at `cut`
        kept for export (the CPU recurrence run in two pieces around the cut: the same pages, state and head output)"""
        if not self.kda_checkpoint_in_replay:
            raise ValueError("prefill_chunk(snap_at=) needs GLM53_PF_KDA_SNAPSHOT=1")
        end = chunk_start + n_valid
        if not chunk_start < cut <= end:
            raise ValueError(f"snap_at {cut} outside the chunk ({chunk_start}, {end}]")
        if logits_rows == "all" and cut < end:
            raise NotImplementedError("the mock's snapshot split returns the head of the second piece only")
        live = lambda: [dict(L=i, S=c.state[slot].clone(), C=c.conv[slot].clone())  # noqa: E731
                        for i, (l, c) in enumerate(zip(self.layers, kv_cache.caches)) if l.is_kda]  # fmt: skip
        if cut == end:
            out = super().prefill_chunk(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start,
                                        n_valid=n_valid, logits_rows=logits_rows, sample=sample, io=io,
                                        state_src=state_src, state_snap=state_snap)  # fmt: skip
            self._kda_snap[int(slot)] = (cut, live())
            return out

        def piece(start, n):
            t = torch.zeros(-(-n // self.prefill_align) * self.prefill_align, dtype=torch.int64)
            t[:n] = tokens[start - chunk_start : start - chunk_start + n]
            return t

        super().prefill_chunk(piece(chunk_start, cut - chunk_start), kv_cache=kv_cache, slot=slot,
                              chunk_start=chunk_start, n_valid=cut - chunk_start, logits_rows="none", sample=None,
                              io=io, state_src=state_src, state_snap=None)  # fmt: skip
        self._kda_snap[int(slot)] = (cut, live())
        return super().prefill_chunk(piece(cut, end - cut), kv_cache=kv_cache, slot=slot, chunk_start=cut,
                                     n_valid=end - cut, logits_rows=logits_rows, sample=sample, io=io,
                                     state_src=None, state_snap=state_snap)  # fmt: skip

    def snapshot_slot(self, kv, slot, entry, **kw):
        self.events.append({"event": "snapshot_slot", "slot": int(slot), "entry": int(entry)})
        super().snapshot_slot(kv, slot, entry, **kw)

    def restore_slot(self, kv, entry, slot, **kw):
        self.events.append({"event": "restore_slot", "slot": int(slot), "entry": int(entry)})
        super().restore_slot(kv, entry, slot, **kw)

    def decode(self, io, kv, width):
        self._decode_steps += 1
        if self.fault_decode_step and self._decode_steps == self.fault_decode_step:
            from models.autoports.zai_org_glm_5_3_flash.tests.mock_model import MockDeviceError

            self.events.append({"event": "device_fault", "decode_step": self._decode_steps})
            raise MockDeviceError(
                f"injected device fault at decode step {self._decode_steps} ({FAULT_DECODE_STEP_ENV})"
            )
        pos = io.cur_pos._t.tolist()
        self.events.append(
            {"event": "decode_step", "active": sum(p >= 0 for p in pos), "positions": pos, "width": width}
        )
        out = super().decode(io, kv, width)
        if self.samples_in_decode_trace:  # like the Stage 14b stack: every decode step (eager too) ends in the sampler
            self._sampler(io, "sample")  # parameter-driven: greedy rows carry k=1
        return out

    def sample(self, io, mode):
        if not self.samples_in_decode_trace:  # fused: the decode step sampled (StackBackend.sample does nothing)
            super().sample(io, mode)

    def capture_decode(self, io, kv, width):
        if not self.samples_in_decode_trace:
            return super().capture_decode(io, kv, width)
        bound, rows = self._bind(io), io.tokens.shape[0]
        return self.device.capture(  # one program: decode + the sampler
            lambda: self.decode(bound, kv, width),
            self._decode_programs(width) + [("sampler", "sample", rows)],
            f"decode+sample-w{width}",
        )

    def write(self, buf, value):
        super().write(buf, value)
        if getattr(self, "greedy_only", False):  # the Stage 14b stack's refusal (tt/stack_model.py write)
            if buf.name == "k":
                self._last_k = torch.as_tensor(value).clone()
            elif buf.name == "temp" and bool(((self._last_k > 1) & (torch.as_tensor(value) > 0)).any()):
                raise NotImplementedError("greedy-only device sampler: a sampled row reached the device")

    def copy_slot(self, kv, src, dst):
        self.events.append({"event": "copy_slot", "src": src, "dst": dst})
        super().copy_slot(kv, src, dst)
        self.ledger.move(int(src), int(dst))
        if not self.copy_slot_keeps_sampler_rng:  # like the 8-chip pipeline: the destination row is re-seeded
            self._seed_counter += 1
            self.rng[dst] = torch.Generator().manual_seed(0xABCDEF + 7919 * self._seed_counter + dst)

    def read_async(self, buf):
        """Like the device's non-blocking copy: the value at the enqueue point, delivered by finish_read later (the
        CPU mock completes at once, so there is no device event for the plugin to wait on)."""
        self.events.append({"event": "read_async", "buffer": buf.name})
        return self.read(buf), []

    def finish_read(self, buf, pending):
        self.events.append({"event": "finish_read", "buffer": buf.name})
        return pending

    def close(self):
        self.events.append({"event": "backend_close"})


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

        _TOKENIZER = load_tokenizer()
    return _TOKENIZER


def make_backend(
    max_batch: int,
    dsa_mode: str | None = None,
    seed: int = 0,
    full_width: bool | None = None,
    per_width: tuple | None = None,
    fused_sampling: bool | None = None,
) -> RecordingMockModel:
    """full_width: every width replays one trace (tt/model.py Glm53PipelineBackend); per_width: only these widths have
    their own traces and copy_slot re-seeds the destination's sampler row (tt/per_width.py PerWidthBackend)."""
    backend = RecordingMockModel(
        KINDS, max_batch=max_batch, dsa_mode=dsa_mode or os.environ.get(DSA_MODE_ENV, "dense_exact"), seed=seed
    )
    if full_width if full_width is not None else os.environ.get(FULL_WIDTH_ENV) == "1":
        backend.width_bucketed_decode = False
    if per_width is None and os.environ.get(PER_WIDTH_ENV):
        per_width = tuple(int(w) for w in os.environ[PER_WIDTH_ENV].split(",") if w)
    if per_width:
        backend.width_bucketed_decode = tuple(per_width)
        backend.copy_slot_keeps_sampler_rng = False
    if fused_sampling if fused_sampling is not None else os.environ.get(FUSED_SAMPLING_ENV) == "1":
        backend.samples_in_decode_trace = True
    if os.environ.get(GREEDY_ONLY_ENV) == "1":
        backend.max_device_top_k, backend.greedy_only = 1, True
    return backend


def build_mock_generator(
    model_dir,
    mesh_device,
    *,
    max_batch,
    max_seq_len,
    optimizations=None,
    hf_config=None,
    dsa_mode=None,
    full_width=None,
    per_width=None,
    fused_sampling=None,
    **kwargs,
):
    """`tt/generator.py::build_generator` over the CPU MockModel backend (GLM53_VLLM_GENERATOR_FACTORY target)."""
    backend = make_backend(
        max_batch, dsa_mode, full_width=full_width, per_width=per_width, fused_sampling=fused_sampling
    )
    gen = build_generator(
        model_dir,
        mesh_device,
        backend=backend,
        tokenizer=tokenizer(),
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        stop_token_ids=(),
    )
    REGISTRY.append(gen)
    marker = os.environ.get(MARKER_ENV)
    if marker:  # lets an out-of-process test see where (which process) the adapter built its generator
        with open(marker, "a") as f:
            f.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "max_batch": max_batch,
                        "max_seq_len": max_seq_len,
                        "dsa_mode": backend.layers[1].dsa_mode,
                    }
                )
                + "\n"
            )
    return gen


def reference_greedy(backend: MockModel, prompt: Sequence[int], n: int) -> list[int]:
    """06a's independent oracle: unpaged, unchunked full-sequence forward of the same mock weights."""
    return backend.reference_greedy(list(prompt), n)


def prompt_ids(seed: int, n: int) -> list[int]:
    """Deterministic token ids the tokenizer can produce (the mock head never emits ids >= TOKENIZER_VOCAB)."""
    return [(seed * 7919 + k * 104729) % (TOKENIZER_VOCAB - 1000) + 17 for k in range(n)]


class HostSpec:
    """The speculative step's host API (tt/spec_mtp.SpecMTP: K, set_page_table, enter, step, read, leave, sel, mesh)
    on the CPU MockModel, for the served-path mocks (Stage 17 x 9: MTP with logprobs requests). The verify consumes
    [bonus, d1..dK] as plain decode steps of slot 0 at positions c..c+K and stops at the first rejection, so the
    committed state is exactly the accepted prefix (the device's candidate rows + selector) and every committed token
    is the plain greedy token. Drafts are a deterministic stand-in for the MTP head: the next token's guess repeated
    (some steps accept, some reject). `kv` is bound by the generator wrappers of build_mock_spec_generator."""

    def __init__(self, backend, K: int):
        self.be, self.K, self.kv = backend, int(K), None
        self.rec = None
        self.sel, self.mesh = None, None
        self.calls = dict(enter=0, step=0, leave=0)

    def set_page_table(self, row):
        pass

    def _io(self):
        return self.kv.extra["io"]

    def _drafts(self, last: int) -> list:
        return [int(last)] * self.K

    def enter(self, page_row):
        io = self._io()
        c = int(io.cur_pos._t[0]) - 1
        b = int(io.tokens._t[0])
        d = self._drafts(b)
        self.rec = dict(n=0, c=c, a=[b] * (self.K + 1), committed=[b], drafts=d, steps=0)
        io.tokens._t[1 : 1 + self.K] = torch.tensor(d, dtype=io.tokens._t.dtype)
        self.calls["enter"] += 1

    def step(self):
        io, rec, K = self._io(), self.rec, self.K
        base = rec["c"] + rec["n"] + 1
        ins = [rec["a"][rec["n"]]] + list(rec["drafts"])
        a = []
        for j in range(K + 1):
            io.tokens._t[0] = ins[j]
            io.cur_pos._t[:] = -1
            io.cur_pos._t[0] = base + j
            self.be.decode(io, self.kv, 1)
            row = io.logits._t[0, : self.be.vocab_size].double()
            a.append(int(torch.nonzero(row == row.max())[0]))  # greedy, lowest id on ties
            if j == K or a[j] != ins[j + 1]:
                break
        n = len(a) - 1
        io.tokens._t[0] = a[n]
        io.cur_pos._t[:] = -1
        io.cur_pos._t[0] = base + n + 1
        d = self._drafts(a[n])
        self.rec = dict(n=n, c=base, a=a + [a[-1]] * (K + 1 - len(a)), committed=a[: n + 1], drafts=d,
                        steps=rec["steps"] + 1)  # fmt: skip
        self.calls["step"] += 1

    def read(self):
        return dict(self.rec)

    def leave(self):
        self.calls["leave"] += 1
        return dict(self.rec)


def build_mock_spec_generator(*args, **kwargs):
    """build_mock_generator + the HostSpec step when GLM53_SPEC_K > 0 (the served-path mocks of speculative decoding):
    the generator's spec entry points bind the KV cache to the step, and spec_abort only drops the speculation (its
    device selector write has no mock counterpart). (The multi-user step, MultiHostSpec, is the spec-multi line's.)"""
    gen = build_mock_generator(*args, **kwargs)
    k = int(os.environ.get("GLM53_SPEC_K", "0") or 0)
    if k <= 0:
        return gen
    spec = HostSpec(gen.backend, k)
    gen.backend.spec = spec
    names = ("spec_enter", "spec_step", "spec_step_submit", "spec_step_collect", "spec_leave")  # (+ GLM53_SPEC_ASYNC's split)
    for name in names:
        orig = getattr(gen, name)

        def bound(kv_cache, *a, _orig=orig, **kw):
            spec.kv = kv_cache
            return _orig(kv_cache, *a, **kw)

        setattr(gen, name, bound)

    def abort(kv_cache):  # (as the generator's: a pre-enqueued step's record is dropped with the speculation)
        b = gen._binding(kv_cache)
        b.spec = None
        b.spec_pending = None

    gen.spec_abort = abort
    return gen

