# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only CPU mocks that drive `tt/generator.py` exactly like the device model will (stage 06a).

* `MockDevice` / `DeviceTensor` / `MockTrace`: a mock device with a program cache and trace capture/replay. Capture
  records a closure over persistent buffers (replay re-reads those exact buffers, so a host value written into a NEW
  buffer instead of the bound one is stale, as on device). Rules mirroring tt-metal: every program a trace needs must
  be compiled before capture, and compiling any program while a trace is live raises (the unsafe program-cache
  allocation `TT_METAL_TRACE_ALLOC_TRACKING` flags), so warmup ordering is tested.
* `MockDecoder`: one decoder layer honouring Stage 1's `FunctionalDecoder` contract (tt/functional_decoder.py,
  doc/context_contract.json): streams [1, 1, T, 4h] in/out, `allocate_cache`, `make_page_table` / `update_page_table`,
  `begin_prefill_chunk` / `prefill_chunk` (physical rows must equal round_up(n_valid, 32 KDA | 64 MLA); padded rows
  are identity steps for KDA and land in the slot's own not-yet-valid MLA pages), `prefill_forward`, device-only
  `decode_forward` (cur_pos -1 = inactive slot left bit-identical), `capture_decode_trace`, `snapshot_state` /
  `restore_state`, `reset_slot`, `check_decode_positions`, `prepare_prefill`; the MLA/DSA kind enforces the
  dense-exact `kv_len <= 2051` guard (`dsa_mode="dense_exact"`) or models Stage 5's lifted bound (`"unbounded"`).
  Padded and inactive output rows are NaN-poisoned, so any consumer that reads them is caught. `bug=` injects a
  defect for negative controls: "pad_leak" (KDA padded rows update the state), "causal_leak" (MLA prefill queries
  see later tokens of their chunk).
* `MockModel(ModelBackend)`: embedding (4 replicated streams) -> MockDecoders (per-kind physical rows) -> mean of
  streams -> RMSNorm -> lm_head -> on-device sampler model; plus `reference_logits` / `reference_greedy`, an
  independent unpaged, unchunked full-sequence forward used as the oracle (float64 math, float32 logits).
* `GoldenReplayModel(ModelBackend)`: returns the fp32 golden logits (goldens_gpu_fp32 per its manifest) for the
  token prefix it finds in its own paged token cache - correct only if the generator's positions, page tables, slot
  routing and token feedback are right; `perturb_every=k` corrupts every k-th continuation position (negative control).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV, round_up
from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    MODEL_PREFILL_ALIGN,
    PREFILL_ALIGN,
    PREFILL_CHUNK,
    SEED_SKIP,
    SEED_UNSEEDED,
    DecodeIO,
    ModelBackend,
    blocks_per_seq,
)
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import (
    SamplingParams,
    candidate_sample_reference,
    vocab_splits_single_device,
)

VOCAB = 154880
TOKENIZER_VOCAB = 154856
NAN = float("nan")


class MockDeviceError(RuntimeError):
    pass


class ContractViolation(AssertionError):
    pass


# ================================================================================================ mock device
class MockDevice:
    def __init__(self):
        self.program_cache: set = set()
        self.live_traces: set = set()
        self.compile_log: list = []
        self.n_writes = 0
        self.n_reads = 0

    def compile(self, sig: tuple):
        if sig in self.program_cache:
            return
        if self.live_traces:
            raise MockDeviceError(
                f"program {sig} compiled while {len(self.live_traces)} trace(s) are live "
                "(unsafe program-cache allocation; prepare every variant before the first capture)"
            )
        self.program_cache.add(sig)
        self.compile_log.append(sig)

    def capture(self, fn: Callable[[], None], needed: Sequence[tuple], label: str) -> "MockTrace":
        missing = [s for s in needed if s not in self.program_cache]
        if missing:
            raise MockDeviceError(f"trace '{label}': programs {missing} not compiled before capture")
        t = MockTrace(self, fn, label, tuple(needed))
        self.live_traces.add(t)
        return t


class MockTrace:
    def __init__(self, device: MockDevice, fn, label: str, programs: tuple):
        self.device, self.fn, self.label, self.programs = device, fn, label, programs
        self.replays = 0
        self.released = False

    def replay(self):
        if self.released:
            raise MockDeviceError(f"replay of released trace {self.label}")
        for s in self.programs:
            if s not in self.device.program_cache:
                raise MockDeviceError(f"trace {self.label}: program {s} evicted")
        self.replays += 1
        self.fn()

    def release(self):
        self.released = True
        self.device.live_traces.discard(self)


class DeviceTensor:
    """Mock persistent device buffer: the host reaches its contents only through backend.write / backend.read."""

    __slots__ = ("_t", "name")

    def __init__(self, t: torch.Tensor, name: str = ""):
        self._t = t
        self.name = name

    @property
    def shape(self):
        return tuple(self._t.shape)

    @property
    def dtype(self):
        return self._t.dtype


def _host_write(device: MockDevice, buf: DeviceTensor, value: torch.Tensor):
    value = torch.as_tensor(value)
    if tuple(value.shape) != buf.shape:
        raise MockDeviceError(f"copy_host_to_device_tensor shape {tuple(value.shape)} != {buf.shape} ({buf.name})")
    buf._t.copy_(value.to(buf.dtype))
    device.n_writes += 1


# ================================================================================================ mock decoder
@dataclass
class MockDecoderCache:
    max_batch: int
    max_seq_len: int
    # KDA
    state: Optional[torch.Tensor] = None  # [slots, dk] (stands for S [slots, 64, 128, 128] fp32)
    conv: Optional[torch.Tensor] = None  # [slots, 3, dk] (the last 3 pre-conv rows)
    # MLA
    latent: Optional[torch.Tensor] = None  # [slots * blocks_per_seq, 64, dl] paged latent cache
    blocks_per_seq: int = 0
    page_table_host: Optional[torch.Tensor] = None
    chunk: dict = field(default_factory=dict)  # per-chunk "device scalars" written by begin_prefill_chunk


class MockDecoder:
    """CPU stand-in for one GLM-5.3-Flash decoder layer with Stage 1's public contract (see module docstring)."""

    def __init__(
        self,
        kind: str,
        layer_idx: int,
        device: MockDevice,
        *,
        hidden: int = 16,
        dk: int = 8,
        ffn: int = 32,
        seed: int = 0,
        dsa_mode: str = "dense_exact",
        bug: Optional[str] = None,
    ):
        if kind not in ("dense_kda", "sparse_kda", "sparse_dsa"):
            raise ValueError(kind)
        self.kind = kind
        self.layer_idx = layer_idx
        self.device = device
        self.is_kda = kind != "sparse_dsa"
        self.attention = "kda" if self.is_kda else "mla"
        self.prefill_align = PREFILL_ALIGN[self.attention]
        self.prefill_buckets = tuple(range(self.prefill_align, PREFILL_CHUNK + 1, self.prefill_align))
        self.block_size = BLOCK_SIZE
        self.width = 4 * hidden
        self.dsa_mode = dsa_mode
        self.bug = bug
        g = torch.Generator().manual_seed(1000 + 17 * layer_idx + seed)
        r = lambda *s, sc=1.0: torch.randn(*s, generator=g, dtype=torch.float64) * sc
        W = self.width
        self.A = r(W, dk, sc=1.0 / math.sqrt(W))
        self.Bo = r(dk, W, sc=1.5 / math.sqrt(dk))
        self.F1 = r(W, ffn, sc=1.0 / math.sqrt(W))
        self.F2 = r(ffn, W, sc=0.5 / math.sqrt(ffn))
        if self.is_kda:
            self.lam = 0.6 + 0.38 * torch.rand(dk, generator=g, dtype=torch.float64)
            self.taps = r(4, sc=0.5)
        else:
            self.Q = r(W, dk, sc=2.0 / math.sqrt(W))
        self.dk = dk
        self.supported_context = 1048576 if (self.is_kda or dsa_mode == "unbounded") else DSA_DENSE_EXACT_MAX_KV

    # ---------------------------------------------------------------------------------------------- shared math
    def _u(self, x):  # [n, W] -> [n, dk]
        return torch.tanh(x @ self.A)

    def _kda_rows(self, x, state, conv):
        """Sequential recurrence over valid rows x [n, W]; state [dk], conv [3, dk] updated in place."""
        out = torch.empty_like(x)
        u = self._u(x)
        for t in range(x.shape[0]):
            c = self.taps[0] * conv[0] + self.taps[1] * conv[1] + self.taps[2] * conv[2] + self.taps[3] * u[t]
            state.mul_(self.lam).add_(c)
            conv[0], conv[1], conv[2] = conv[1].clone(), conv[2].clone(), u[t]
            out[t] = x[t] + torch.tanh(state + c) @ self.Bo
        return out

    def _mla_rows(self, x, lat_keys, q_pos):
        """x [n, W] queries at absolute positions q_pos [n]; lat_keys [kv_len, dk] (all finite, causal by q_pos)."""
        q = x @ self.Q
        scores = (q @ lat_keys.T) / math.sqrt(self.dk)
        kpos = torch.arange(lat_keys.shape[0])
        scores = scores.masked_fill(kpos[None, :] > q_pos[:, None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        return x + (p @ lat_keys) @ self.Bo

    def _ffn(self, y):
        return y + torch.tanh(y @ self.F1) @ self.F2

    # ---------------------------------------------------------------------------------------------- state
    def _check_dense_exact(self, kv_len):
        if not self.is_kda and self.dsa_mode == "dense_exact" and kv_len > DSA_DENSE_EXACT_MAX_KV:
            raise NotImplementedError(
                f"MLA/DSA layer {self.layer_idx}: kv_len {kv_len} > {DSA_DENSE_EXACT_MAX_KV} needs the DSA indexer path"
            )

    def allocate_cache(self, *, max_batch: int, max_seq_len: int, num_blocks: Optional[int] = None) -> MockDecoderCache:
        if max_seq_len > 1048576:
            raise ValueError("max_seq_len > advertised context")
        c = MockDecoderCache(max_batch=max_batch, max_seq_len=max_seq_len)
        if self.is_kda:
            c.state = torch.zeros(max_batch, self.dk, dtype=torch.float64)
            c.conv = torch.zeros(max_batch, 3, self.dk, dtype=torch.float64)
        else:
            self._check_dense_exact(max_seq_len)
            c.blocks_per_seq = round_up(round_up(max_seq_len + self.prefill_align - 1, BLOCK_SIZE) // BLOCK_SIZE, 8)
            c.latent = torch.full(
                (num_blocks or max_batch * c.blocks_per_seq, BLOCK_SIZE, self.dk), NAN, dtype=torch.float64
            )
        return c

    def make_page_table(self, cache, permutation=None) -> DeviceTensor:
        n = cache.max_batch * cache.blocks_per_seq
        perm = torch.arange(n) if permutation is None else torch.as_tensor(permutation)
        cache.page_table_host = perm.reshape(cache.max_batch, cache.blocks_per_seq).to(torch.int32)
        return DeviceTensor(cache.page_table_host.clone(), "page_table")

    def update_page_table(self, cache, page_table: DeviceTensor, permutation):
        n = cache.max_batch * cache.blocks_per_seq
        perm = torch.as_tensor(permutation).reshape(-1).to(torch.int32)
        if perm.numel() != n or sorted(perm.tolist()) != list(range(n)):
            raise ValueError(f"page table must be a permutation of the {n} physical blocks")
        cache.page_table_host = perm.reshape(cache.max_batch, cache.blocks_per_seq)
        _host_write(self.device, page_table, cache.page_table_host)

    def snapshot_state(self, cache):
        return (cache.state.clone(), cache.conv.clone()) if self.is_kda else None

    def restore_state(self, cache, snap):
        if self.is_kda:
            cache.state.copy_(snap[0])
            cache.conv.copy_(snap[1])

    def reset_slot(self, cache, slot: int):
        if self.is_kda:
            cache.state[slot].zero_()
            cache.conv[slot].zero_()

    def check_decode_positions(self, positions, cache=None):
        live = [p for p in positions if p != -1]
        if cache is not None:
            if len(positions) != cache.max_batch:
                raise ValueError(f"{len(positions)} positions for {cache.max_batch} decode slots")
            if min(positions) < -1 or (live and max(live) >= cache.max_seq_len):
                raise ValueError(f"decode positions {positions} outside [0, {cache.max_seq_len})")
        if live:
            self._check_dense_exact(max(live) + 1)

    # ---------------------------------------------------------------------------------------------- prefill
    def prepare_prefill(self, cache):
        for p in self.prefill_buckets:
            self.device.compile(("prefill", self.kind, p))
        self.reset_slot(cache, 0)

    def begin_prefill_chunk(self, cache, *, user_slot: int, chunk_start: int, n_valid: int):
        if not 0 <= user_slot < cache.max_batch:
            raise ValueError(f"user_slot {user_slot} outside [0, {cache.max_batch})")
        if not 1 <= n_valid <= PREFILL_CHUNK or chunk_start < 0 or chunk_start + n_valid > cache.max_seq_len:
            raise ValueError(
                f"chunk [{chunk_start}, {chunk_start + n_valid}) invalid for max_seq_len {cache.max_seq_len}"
            )
        if not self.is_kda:
            if chunk_start % BLOCK_SIZE:
                raise ValueError(f"MLA chunk start {chunk_start} must be a multiple of {BLOCK_SIZE}")
            self._check_dense_exact(chunk_start + n_valid)
            if cache.page_table_host is None:
                raise ValueError("bind a page table before prefill")
        cache.chunk = {"slot": user_slot, "start": chunk_start, "n_valid": n_valid}
        if chunk_start == 0:
            self.reset_slot(cache, user_slot)

    def prefill_chunk(self, x: DeviceTensor, *, cache, user_slot: int = 0) -> DeviceTensor:
        ch = cache.chunk
        if ch.get("slot") != user_slot:
            raise ContractViolation("prefill_chunk without a matching begin_prefill_chunk")
        n = ch["n_valid"]
        P = x.shape[2]
        if P != round_up(n, self.prefill_align):
            raise ContractViolation(
                f"layer {self.layer_idx} ({self.kind}): physical rows {P} != round_up({n}, {self.prefill_align})"
            )
        if x.shape[3] != self.width:
            raise ContractViolation(f"stream width {x.shape[3]} != {self.width}")
        self.device.compile(("prefill", self.kind, P))
        rows = x._t.reshape(P, self.width)
        out = torch.full_like(rows, NAN)
        if self.is_kda:
            valid = rows if self.bug == "pad_leak" else rows[:n]
            y = self._kda_rows(valid, cache.state[user_slot], cache.conv[user_slot])
            out[:n] = y[:n]
        else:
            start = ch["start"]
            lat = self._u(rows)  # padded rows too: they land in the slot's own not-yet-valid pages (Stage 1)
            pt = cache.page_table_host[user_slot].long()
            pos = torch.arange(start, start + P)
            cache.latent[pt[pos // BLOCK_SIZE], pos % BLOCK_SIZE] = lat
            kv_end = start + n
            kpos = torch.arange(kv_end)
            keys = cache.latent[pt[kpos // BLOCK_SIZE], kpos % BLOCK_SIZE]
            q_pos = torch.arange(start, kv_end)
            if self.bug == "causal_leak":  # negative control: queries see later tokens of the chunk
                q_pos = torch.full_like(q_pos, kv_end - 1)
            out[:n] = self._mla_rows(rows[:n], keys, q_pos)
        out[:n] = self._ffn(out[:n])
        cache.chunk = {}
        return DeviceTensor(out.reshape(1, 1, P, self.width), "streams")

    def prefill_forward(self, streams: torch.Tensor, *, cache, user_slot: int = 0, start_pos: int = 0) -> torch.Tensor:
        rows = streams.reshape(-1, self.width).double()
        seq = rows.shape[0]
        outs = []
        for off in range(0, seq, PREFILL_CHUNK):
            n = min(PREFILL_CHUNK, seq - off)
            P = round_up(n, self.prefill_align)
            chunk = torch.zeros(P, self.width, dtype=torch.float64)
            chunk[:n] = rows[off : off + n]
            self.begin_prefill_chunk(cache, user_slot=user_slot, chunk_start=start_pos + off, n_valid=n)
            y = self.prefill_chunk(DeviceTensor(chunk.reshape(1, 1, P, self.width)), cache=cache, user_slot=user_slot)
            outs.append(y._t.reshape(P, self.width)[:n])
        return torch.cat(outs).reshape(1, seq, self.width)

    # ---------------------------------------------------------------------------------------------- decode
    def decode_forward(
        self, streams: DeviceTensor, *, cache, page_table: DeviceTensor = None, cur_pos: DeviceTensor = None
    ):
        if cur_pos is None:
            raise ValueError("decode_forward needs cur_pos [B] int32 (-1 = inactive)")
        B = streams.shape[2]
        if B > cache.max_batch:
            raise ContractViolation(f"decode of {B} rows > max_batch={cache.max_batch} slots")
        # rows map to slots 0..B-1 (width-B bucketed decode; Stage 2's FusedKDA takes exactly max_batch rows today, so
        # Stage 6 needs a slot-prefix view of its [slots, ...] state and [1, 3, slots, 24576] conv history)
        self.device.compile(("decode", self.kind, B))
        rows = streams._t.reshape(B, self.width)
        pos = cur_pos._t.long()
        out = torch.full_like(rows, NAN)
        for b in range(B):
            p = int(pos[b])
            if p < 0:
                continue  # inactive slot: state bit-identical, output row don't-care (NaN)
            x = rows[b : b + 1]
            if self.is_kda:
                y = self._kda_rows(x, cache.state[b], cache.conv[b])
            else:
                pt = page_table._t[b].long()
                cache.latent[pt[p // BLOCK_SIZE], p % BLOCK_SIZE] = self._u(x)[0]
                kpos = torch.arange(p + 1)
                keys = cache.latent[pt[kpos // BLOCK_SIZE], kpos % BLOCK_SIZE]
                y = self._mla_rows(x, keys, torch.tensor([p]))
            out[b] = self._ffn(y)[0]
        return DeviceTensor(out.reshape(1, 1, B, self.width), "streams")

    def capture_decode_trace(self, streams: DeviceTensor, *, cache, page_table=None, cur_pos=None):
        out = DeviceTensor(torch.full_like(streams._t, NAN), "streams_out")

        def run():
            out._t.copy_(self.decode_forward(streams, cache=cache, page_table=page_table, cur_pos=cur_pos)._t)

        trace = self.device.capture(run, [("decode", self.kind, streams.shape[2])], f"decode L{self.layer_idx}")
        return MockDecodeTrace(trace, streams, out, cur_pos, page_table)


@dataclass
class MockDecodeTrace:
    trace: MockTrace
    streams_in: DeviceTensor
    streams_out: DeviceTensor
    cur_pos: Any
    page_table: Any

    def replay(self, blocking: bool = False):
        self.trace.replay()
        return self.streams_out

    def release(self):
        self.trace.release()


# ================================================================================================ backends
@dataclass
class MockKVCache:
    caches: list
    max_batch: int
    max_seq_len: int
    blocks_per_seq: int
    extra: dict = field(default_factory=dict)
    num_blocks: int = 0


class MockBackendBase(ModelBackend):
    """Shared mock-device plumbing: DecodeIO buffers, host write/read, the on-device sampler model, traces."""

    def __init__(
        self,
        *,
        max_batch: int,
        vocab_size: int = VOCAB,
        logits_width: int = VOCAB,
        supported_context: int,
        sampler_shards: Optional[int] = None,
    ):
        self.device = MockDevice()
        self.vocab_size = vocab_size
        self.logits_width = logits_width
        self.max_batch = max_batch
        self.supported_context = supported_context
        self.sampler_shards = sampler_shards or vocab_splits_single_device(logits_width) or 1
        self.rng = [torch.Generator().manual_seed(0x5EED + i) for i in range(max_batch)]
        self._seed_counter = 0
        self.decode_calls = 0
        self.decode_log: Optional[list] = None  # set to [] to record (cur_pos, tokens, logits) of every decode step

    def _log_decode(self, io, width):
        if self.decode_log is not None:  # rows >= width were not decoded this step: logged as inactive
            pos = io.cur_pos._t.clone()
            pos[width:] = -1
            self.decode_log.append((pos, io.tokens._t.clone(), io.logits._t.clone()))

    # ---- buffers
    def allocate_decode_io(self, kv) -> DecodeIO:
        B = kv.max_batch
        bps = kv.blocks_per_seq
        z = lambda shape, dt, name: DeviceTensor(torch.zeros(shape, dtype=dt), name)
        io = DecodeIO(
            tokens=z((B,), torch.int32, "tokens"),
            cur_pos=DeviceTensor(torch.full((B,), -1, dtype=torch.int32), "cur_pos"),
            page_table=DeviceTensor(
                (torch.arange(B * bps, dtype=torch.int32) % (kv.num_blocks or B * bps)).reshape(B, bps), "page_table"
            ),
            logits=z((B, self.logits_width), torch.float32, "logits"),
            sampled=z((B,), torch.int32, "sampled"),
            k=DeviceTensor(torch.ones(B, dtype=torch.int32), "k"),
            p=z((B,), torch.float32, "p"),
            temp=DeviceTensor(torch.ones(B, dtype=torch.float32), "temp"),
            seeds=DeviceTensor(torch.full((B,), SEED_SKIP, dtype=torch.int64), "seeds"),
        )
        kv.extra["io"] = io
        return io

    def sampler_rng_rows(self, slots):
        return {int(s): self.rng[int(s)].get_state() for s in slots}

    def set_sampler_rng_rows(self, rows: dict) -> None:
        for s, st in rows.items():
            self.rng[int(s)].set_state(st)

    def write(self, buf: DeviceTensor, value: torch.Tensor) -> None:
        _host_write(self.device, buf, value)
        if buf.name == "seeds":  # ttnn.manual_seed at the request boundary
            for s, v in enumerate(buf._t.tolist()):
                if v == SEED_SKIP:
                    continue
                if v == SEED_UNSEEDED:
                    self._seed_counter += 1
                    v = 0xABCDEF + 7919 * self._seed_counter + s
                self.rng[s] = torch.Generator().manual_seed(int(v))
            buf._t.fill_(SEED_SKIP)

    def read(self, buf: DeviceTensor) -> torch.Tensor:
        self.device.n_reads += 1
        return buf._t.clone()

    # ---- sampler model (models/common/sampling semantics: split candidates, k/p/temp, lowest-id greedy tie-break)
    def _sample_row(
        self, logits_row: torch.Tensor, k: int, p: float, temp: float, slot: int, mode: str, active: bool = True
    ) -> int:
        row = logits_row.double()
        if not torch.isfinite(row[: self.vocab_size]).all():
            if not active:  # backend contract (contract.md 5): rows with cur_pos == -1 reach the sampler masked
                raise MockDeviceError(f"sampler input row {slot} (inactive) is not finite: the backend must mask it")
            return 0  # an active row the model corrupted (negative controls): undefined token, caught by the oracle
        params = (
            SamplingParams()
            if (mode == "greedy" or k <= 1)
            else SamplingParams(temperature=float(temp), top_k=int(k), top_p=float(p) if p > 0 else 1.0)
        )
        tok, _, _ = candidate_sample_reference(
            row, params, n_shards=self.sampler_shards, generator=self.rng[slot], vocab_size=self.vocab_size
        )
        return int(tok)

    def _sampler(self, io: DecodeIO, mode: str):
        B = io.tokens.shape[0]
        self.device.compile(("sampler", mode, B))
        k, p, temp = io.k._t.tolist(), io.p._t.tolist(), io.temp._t.tolist()
        act = (io.cur_pos._t >= 0).tolist()  # after the position advance: inactive rows stay -1
        toks = [self._sample_row(io.logits._t[s], k[s], p[s], temp[s], s, mode, act[s]) for s in range(B)]
        t = torch.tensor(toks, dtype=torch.int32)
        io.tokens._t.copy_(t)  # tt_out_tok: the next decode's token input
        io.sampled._t.copy_(t)

    def sample(self, io: DecodeIO, mode: str) -> None:
        self._sampler(io, mode)

    @staticmethod
    def _bind(io: DecodeIO) -> DecodeIO:
        """A trace binds the buffers that exist at capture (device addresses), not the host container: replacing a
        DecodeIO field with a new buffer afterwards is invisible to replay (stale input), as on device."""
        return dataclasses.replace(io)

    def capture_sampling(self, io: DecodeIO, mode: str):
        bound = self._bind(io)
        return self.device.capture(
            lambda: self._sampler(bound, mode), [("sampler", mode, io.tokens.shape[0])], f"sampler-{mode}"
        )

    def _sample_prefill_row(self, logits_row, slot, params: SamplingParams, io: DecodeIO) -> int:
        mode = params.device_mode
        self.device.compile(("sample_row", mode))
        k, p, temp = params.device_values()
        tok = self._sample_row(logits_row, k, p, temp, slot, mode)
        io.tokens._t[slot] = tok
        return tok

    def replay(self, trace):
        trace.replay()

    def release(self, trace):
        trace.release()

    def _decode_programs(self, B):
        raise NotImplementedError

    def prepare_decode(self, io, kv, widths):
        B = io.tokens.shape[0]
        for w in widths:
            for sig in self._decode_programs(w):
                self.device.compile(sig)
        for mode in ("greedy", "sample"):
            self.device.compile(("sampler", mode, B))
            self.device.compile(("sample_row", mode))

    def capture_decode(self, io, kv, width):
        bound = self._bind(io)
        return self.device.capture(
            lambda: self.decode(bound, kv, width), self._decode_programs(width), f"decode-w{width}"
        )

    def _advance_positions(self, io, width):
        cp = io.cur_pos._t[:width]
        cp.copy_(torch.where(cp >= 0, cp + 1, cp))  # ttnn.plus_one(skip_negative_entries=True) on rows 0..width-1


class MockModel(MockBackendBase):
    """embedding -> MockDecoders -> head, with an independent full-sequence reference."""

    def __init__(
        self,
        kinds: Sequence[str] = ("dense_kda", "sparse_dsa", "sparse_kda"),
        *,
        max_batch: int = 8,
        hidden: int = 16,
        seed: int = 0,
        dsa_mode: str = "dense_exact",
        logits_width: int = VOCAB,
        bug: Optional[str] = None,
    ):
        layers_ctx = [
            1048576 if (k != "sparse_dsa" or dsa_mode == "unbounded") else DSA_DENSE_EXACT_MAX_KV for k in kinds
        ]
        super().__init__(max_batch=max_batch, logits_width=logits_width, supported_context=min(layers_ctx))
        self.layer_kinds = tuple(kinds)
        self.mask_inactive = True  # backend requirement (contract.md 5); False = negative control
        self.hidden = hidden
        self.width = 4 * hidden
        self.layers = [
            MockDecoder(k, i, self.device, hidden=hidden, seed=seed, dsa_mode=dsa_mode, bug=bug)
            for i, k in enumerate(kinds)
        ]
        g = torch.Generator().manual_seed(4242 + seed)
        self.embed = torch.randn(VOCAB, hidden, generator=g, dtype=torch.float64)
        self.norm_w = 0.5 + torch.rand(hidden, generator=g, dtype=torch.float64)
        self.lm_head = torch.randn(hidden, VOCAB, generator=g, dtype=torch.float64) * 3.0
        self.lm_bias = torch.zeros(VOCAB, dtype=torch.float64)
        self.lm_bias[TOKENIZER_VOCAB:] = -1e3  # ids the tokenizer never produces (a trained head never picks them)

    # ---- math shared by the paged path and the reference
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        e = self.embed[ids.long()]
        return e.repeat(1, 4)  # X_i = E[ids] for the 4 mHC streams

    def _head(self, streams: torch.Tensor) -> torch.Tensor:
        h = streams.reshape(-1, 4, self.hidden).mean(dim=1)
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-5) * self.norm_w
        logits = (h @ self.lm_head + self.lm_bias).float()
        if self.logits_width > VOCAB:  # TP-padded lm_head columns hold garbage until the sampler masks them
            pad = torch.full((logits.shape[0], self.logits_width - VOCAB), 1e6, dtype=torch.float32)
            logits = torch.cat([logits, pad], dim=-1)
        return logits

    # ---- ModelBackend: state
    # GLM53_MOCK_DSA_SHARD=C (> 1): the sequence-sharded DSA pool's block contract (tt/owner_blocks.py) on the CPU:
    # the latent pool is C x the per-chip physical blocks (physical g = chip g % C, local block g // C, every chip's
    # local 0 its null block) and every page table the generator hands over must be owner-aware (column j's block on
    # chip (j // 2) % C). A wrong logical -> physical map (two live blocks on one physical block, a block read at
    # another position) changes the tokens against the unsharded run and the reference.
    dsa_shard = int(os.environ.get("GLM53_MOCK_DSA_SHARD", "1"))

    def allocate_kv_cache(self, *, max_batch, max_seq_len, num_blocks=None):
        if max_batch > self.max_batch:
            raise ValueError(f"max_batch {max_batch} > {self.max_batch}")
        bps = blocks_per_seq(max_seq_len)
        nb = num_blocks or max_batch * bps
        phys, extra = nb, {}
        if self.dsa_shard > 1:
            from models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard import UNIT
            from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap

            bpu = UNIT // BLOCK_SIZE
            nb_local = OwnerBlockMap.local_blocks_for(self.dsa_shard, nb, max_batch, bpu)
            phys, extra = self.dsa_shard * nb_local, {"dsa_shard": (self.dsa_shard, nb_local, bpu)}
        caches = [l.allocate_cache(max_batch=max_batch, max_seq_len=max_seq_len, num_blocks=phys) for l in self.layers]
        return MockKVCache(caches, max_batch, max_seq_len, bps, extra=extra, num_blocks=nb)

    def dsa_shard_owners(self, kv):
        """(C, physical blocks per chip, blocks per unit) of a sharded cache, else None (Glm53Generator's block map)"""
        return kv.extra.get("dsa_shard")

    def _check_owner_table(self, kv, table):
        sh = self.dsa_shard_owners(kv)
        if sh is None:
            return
        C, nb_local, bpu = sh
        t = torch.as_tensor(table).long().reshape(-1, table.shape[-1])
        col_owner = (torch.arange(t.shape[1]) // bpu) % C
        if ((t % C) != col_owner).any() or (t < 0).any() or (t >= C * nb_local).any():
            bad = torch.nonzero((t % C) != col_owner)[:4].tolist()
            raise ContractViolation(f"sharded DSA pool: page-table entries off their column's owner chip at {bad}")

    def write(self, buf, value):
        if buf.name == "page_table" and self.dsa_shard > 1:
            kv = next((k for k in getattr(self, "_kvs", ()) if k.extra.get("io") is not None and k.extra["io"].page_table is buf), None)
            if kv is not None:
                self._check_owner_table(kv, torch.as_tensor(value))
        super().write(buf, value)

    def allocate_decode_io(self, kv):
        io = super().allocate_decode_io(kv)
        self._kvs = getattr(self, "_kvs", []) + [kv]
        return io

    def set_page_table_host(self, kv, page_table):
        self._check_owner_table(kv, page_table)
        for l, c in zip(self.layers, kv.caches):
            if not l.is_kda:
                c.page_table_host = page_table.to(torch.int32).clone()

    def reset_slot(self, kv, slot):
        for l, c in zip(self.layers, kv.caches):
            l.reset_slot(c, slot)

    def copy_slot(self, kv, src, dst):
        for l, c in zip(self.layers, kv.caches):
            if l.is_kda:
                c.state[dst].copy_(c.state[src])
                c.conv[dst].copy_(c.conv[src])
        self.rng[dst].set_state(self.rng[src].get_state())

    def reset_all(self, kv):
        for s in range(kv.max_batch):
            self.reset_slot(kv, s)

    # ---- prefix-cache snapshot pool (ModelBackend.snapshot_slot / restore_slot; the stack's virtual slots are a dict
    # of entries here): entry -> {layer index: (state, conv)} of the KDA layers. GLM53_MOCK_SNAPSHOT_ENTRIES (32)
    snapshot_entries = int(os.environ.get("GLM53_MOCK_SNAPSHOT_ENTRIES", "32"))
    prefill_state_io = True  # prefill_chunk takes state_src / state_snap (like the TP prefill's in-trace state_io)

    # (no mock-device compile: on the device these are the SlotOps / state_io programs warm-up already compiled)
    def _pool(self, kv):
        return kv.extra.setdefault("snapshots", {})

    def _check_entry(self, entry):
        if not 0 <= int(entry) < self.snapshot_entries:
            raise ContractViolation(f"snapshot entry {entry} outside [0, {self.snapshot_entries})")

    _tee_pairs: tuple = ()

    def decode_tee(self, kv) -> bool:
        """GLM53_DECODE_TEE=1: the decode step itself snapshots the pairs set before it (stack_model.DecodeTee)"""
        return os.environ.get("GLM53_DECODE_TEE", "0") == "1" and self.snapshot_entries > 0

    def set_decode_tee(self, kv, pairs):
        self._tee_pairs = tuple((int(s_), int(e_), int(a_), int(b_)) for s_, e_, a_, b_ in pairs)

    def snapshot_slot(self, kv, slot, entry, upto=None):  # upto: paged per-token state only (none here)
        self._check_entry(entry)
        self._pool(kv)[int(entry)] = {
            i: (c.state[slot].clone(), c.conv[slot].clone())
            for i, (l, c) in enumerate(zip(self.layers, kv.caches))
            if l.is_kda
        }

    def restore_slot(self, kv, entry, slot, upto=None):
        self._check_entry(entry)
        snap = self._pool(kv).get(int(entry))
        if snap is None:
            raise ContractViolation(f"snapshot entry {entry} was never written")
        for i, (l, c) in enumerate(zip(self.layers, kv.caches)):
            if l.is_kda:
                c.state[slot].copy_(snap[i][0])
                c.conv[slot].copy_(snap[i][1])

    def export_snapshot(self, kv, entry):
        self._check_entry(entry)
        snap = self._pool(kv)[int(entry)]
        return [dict(L=i, S=v[0].clone(), C=v[1].clone()) for i, v in sorted(snap.items())]

    def import_snapshot(self, kv, entry, states):
        self._check_entry(entry)
        self._pool(kv)[int(entry)] = {int(d["L"]): (d["S"].clone(), d["C"].clone()) for d in states}

    # ---- prefill
    def prepare_prefill(self, kv):
        for l, c in zip(self.layers, kv.caches):
            l.prepare_prefill(c)
        for P in range(MODEL_PREFILL_ALIGN, PREFILL_CHUNK + 1, MODEL_PREFILL_ALIGN):
            self.device.compile(("embed", P))
            self.device.compile(("head_all", P))
            for a in {l.prefill_align for l in self.layers}:
                for n in (P - MODEL_PREFILL_ALIGN + 1, P):  # both physical row counts a P-row chunk can map to
                    Pl = round_up(n, a)
                    if Pl != P:
                        self.device.compile(("slice_rows", P, Pl))
                        self.device.compile(("pad_rows", Pl, P))
        self.device.compile(("head_last",))
        for mode in ("greedy", "sample"):
            self.device.compile(("sample_row", mode))

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None,
                      state_src=None, state_snap=None):  # fmt: skip
        P = tokens.numel()
        if P != round_up(n_valid, MODEL_PREFILL_ALIGN) or not 1 <= n_valid <= PREFILL_CHUNK:
            raise ContractViolation(f"model chunk of {P} rows for {n_valid} valid tokens")
        if state_src is not None:  # a prefix-cache attach: the chunk starts from the entry's state (state_io STATE_IN)
            if chunk_start == 0:
                raise ContractViolation("an attach continues a cached prefix (chunk_start > 0)")
            self.restore_slot(kv_cache, state_src, slot)
        out = self._prefill_chunk(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start, n_valid=n_valid,
                                  logits_rows=logits_rows, sample=sample, io=io)  # fmt: skip
        if state_snap is not None:  # the state after the chunk also goes to the entry (state_io STATE_OUT)
            self.snapshot_slot(kv_cache, slot, state_snap)
        return out

    def _prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None):
        P = tokens.numel()
        self.device.compile(("embed", P))
        x = self._embed(tokens)
        for l, c in zip(self.layers, kv_cache.caches):
            Pl = round_up(n_valid, l.prefill_align)
            xl = x
            if Pl != P:
                self.device.compile(("slice_rows", P, Pl))
                xl = x[:Pl]
            l.begin_prefill_chunk(c, user_slot=slot, chunk_start=chunk_start, n_valid=n_valid)
            y = l.prefill_chunk(DeviceTensor(xl.reshape(1, 1, Pl, self.width)), cache=c, user_slot=slot)._t.reshape(
                Pl, self.width
            )
            if Pl != P:
                self.device.compile(("pad_rows", Pl, P))
                y = torch.cat([y, torch.zeros(P - Pl, self.width, dtype=y.dtype)])
            x = y
        if logits_rows == "all":
            self.device.compile(("head_all", P))
            return self._head(x)
        if logits_rows == "last":
            self.device.compile(("head_last",))
            row = self._head(x[n_valid - 1 : n_valid])
            if sample is not None:
                return self._sample_prefill_row(row[0], slot, sample, io)
            return row
        return None

    # ---- decode
    def _decode_programs(self, B):
        return (
            [("embed_decode", B)] + [("decode", l.kind, B) for l in self.layers] + [("head_decode", B), ("plus_one", B)]
        )

    def decode(self, io, kv, width):
        B = width
        self.device.compile(("embed_decode", B))
        x = DeviceTensor(self._embed(io.tokens._t[:B].clamp(min=0)).reshape(1, 1, B, self.width), "streams")
        pt, cp = DeviceTensor(io.page_table._t[:B], "page_table"), DeviceTensor(io.cur_pos._t[:B], "cur_pos")  # views
        inactive = io.cur_pos._t[:B] < 0
        for l, c in zip(self.layers, kv.caches):
            x = l.decode_forward(x, cache=c, page_table=pt, cur_pos=cp)
            if self.mask_inactive:  # layer-boundary mask of inactive rows (decoders leave them undefined: NaN here)
                x._t[0, 0, inactive] = 0.0
        for s_, e_, _, _ in self._tee_pairs:  # the decode trace's tee (stack_model.DecodeTee): the post-step state
            self.snapshot_slot(kv, s_, e_)
        self.device.compile(("head_decode", B))
        io.logits._t[:B].copy_(self._head(x._t.reshape(B, self.width)))
        self._log_decode(io, B)
        self.device.compile(("plus_one", B))
        self._advance_positions(io, B)
        self.decode_calls += 1

    # ---- independent reference (no pages, no chunks, no slots)
    def reference_logits(self, ids: Sequence[int], last_only: bool = False) -> torch.Tensor:
        ids = torch.as_tensor(list(ids), dtype=torch.int64)
        T = ids.numel()
        x = self._embed(ids)
        for l in self.layers:
            if l.is_kda:
                st = torch.zeros(l.dk, dtype=torch.float64)
                cv = torch.zeros(3, l.dk, dtype=torch.float64)
                y = l._kda_rows(x, st, cv)
            else:
                if l.dsa_mode == "dense_exact" and T > DSA_DENSE_EXACT_MAX_KV:
                    raise NotImplementedError("reference beyond the dense-exact regime")
                y = l._mla_rows(x, l._u(x), torch.arange(T))
            x = l._ffn(y)
        return self._head(x[-1:] if last_only else x)[:, :VOCAB]

    def reference_greedy(self, ids: Sequence[int], n: int, stop_ids=()) -> list[int]:
        seq = list(ids)
        out = []
        for _ in range(n):
            t = int(torch.argmax(self.reference_logits(seq, last_only=True)[-1]))
            out.append(t)
            seq.append(t)
            if t in stop_ids:
                break
        return out


# ================================================================================================ golden replay
@dataclass
class GoldenEntry:
    name: str
    prompt_ids: torch.Tensor  # [T]
    targets: torch.Tensor  # [G] greedy continuation (targets[i] = argmax(cont[i]))
    cont: torch.Tensor  # [G, V] fp32: cont[0] = last prompt logits, cont[i] after feeding targets[i-1]
    top32_values: Optional[torch.Tensor] = None  # [T, 32] prompt positions
    top32_indices: Optional[torch.Tensor] = None

    @property
    def input_ids(self) -> torch.Tensor:  # prompt + targets[:-1]: the tokens whose logits we know
        return torch.cat([self.prompt_ids, self.targets[:-1]])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_golden_entries(
    goldens_dir: str, prompts: Sequence[int | str], *, verify_sha: bool = True
) -> list[GoldenEntry]:
    """Teacher-forcing goldens per `<goldens_dir>/manifest.json`: teacher_forcing/<T>.pt (prompt_ids, targets) and
    logits/<T>.pt (continuation_logits [G, V], prompt_top32_*). sha256 verified against the manifest."""
    man = json.load(open(os.path.join(goldens_dir, "manifest.json")))
    files = man["files"]
    out = []
    for T in prompts:
        tf_rel, lg_rel = f"teacher_forcing/{T}.pt", f"logits/{T}.pt"
        for rel in (tf_rel, lg_rel):
            if rel not in files:
                raise FileNotFoundError(f"{rel} not in {goldens_dir}/manifest.json")
            if verify_sha and _sha256(os.path.join(goldens_dir, rel)) != files[rel]["sha256"]:
                raise ValueError(f"sha256 mismatch for {goldens_dir}/{rel}")
        tf = torch.load(os.path.join(goldens_dir, tf_rel))
        lg = torch.load(os.path.join(goldens_dir, lg_rel))
        cont = lg["continuation_logits"].float()
        tg = tf["targets"].reshape(-1).long()
        if not torch.equal(torch.argmax(cont, dim=-1), tg):
            raise ValueError(f"{goldens_dir} T={T}: targets != argmax(continuation_logits)")
        out.append(
            GoldenEntry(
                str(T),
                tf["prompt_ids"].reshape(-1).long(),
                tg,
                cont,
                lg.get("prompt_top32_values"),
                lg.get("prompt_top32_indices"),
            )
        )
    return out


class GoldenReplayModel(MockBackendBase):
    """Model backend that returns the fp32 golden logits of the token prefix found in its paged token cache.

    The token cache uses the same slot / page-table / position semantics as the latent cache, so the replay is right
    only when the generator prefilled the right tokens at the right positions into the right slot and fed the right
    token back at every decode step. A prefix that matches no golden entry yields deterministic garbage logits and
    is counted in `mismatches`. `perturb_every=k` replaces every k-th continuation row with logits whose argmax is a
    token outside the golden top-100 (negative control)."""

    def __init__(
        self,
        entries: Sequence[GoldenEntry],
        *,
        max_batch: int = 4,
        supported_context: int = 1048576,
        perturb_every: int = 0,
    ):
        super().__init__(max_batch=max_batch, supported_context=supported_context)
        self.entries = list(entries)
        self.layer_kinds = ("golden_replay",)
        self.perturb_every = perturb_every
        self.mismatches = 0
        self.beyond = 0
        self.lookups = 0
        self.perturbed_positions: list = []

    def allocate_kv_cache(self, *, max_batch, max_seq_len, num_blocks=None):
        bps = blocks_per_seq(max_seq_len)
        nb = num_blocks or max_batch * bps
        tok = torch.full((nb, BLOCK_SIZE), -7, dtype=torch.int64)
        return MockKVCache(
            [],
            max_batch,
            max_seq_len,
            bps,
            extra={"tokens": tok, "pt": None, "entry": [None] * max_batch},
            num_blocks=nb,
        )

    def set_page_table_host(self, kv, page_table):
        kv.extra["pt"] = page_table.to(torch.int64).clone()

    def reset_slot(self, kv, slot):
        kv.extra["entry"][slot] = None

    def copy_slot(self, kv, src, dst):
        kv.extra["entry"][dst] = kv.extra["entry"][src]
        self.rng[dst].set_state(self.rng[src].get_state())

    def reset_all(self, kv):
        kv.extra["tokens"].fill_(-7)
        kv.extra["entry"] = [None] * kv.max_batch

    def prepare_prefill(self, kv):
        for P in range(MODEL_PREFILL_ALIGN, PREFILL_CHUNK + 1, MODEL_PREFILL_ALIGN):
            self.device.compile(("prefill", P))
            self.device.compile(("head_all", P))
        self.device.compile(("head_last",))
        for mode in ("greedy", "sample"):
            self.device.compile(("sample_row", mode))

    def _write_tokens_paged(self, kv, pt_row, start, toks):
        pos = torch.arange(start, start + toks.numel())
        kv.extra["tokens"][pt_row[pos // BLOCK_SIZE], pos % BLOCK_SIZE] = toks.long()

    def _prefix(self, kv, pt_row, upto):
        pos = torch.arange(upto + 1)
        return kv.extra["tokens"][pt_row[pos // BLOCK_SIZE], pos % BLOCK_SIZE]

    def _garbage(self, pos):
        g = torch.Generator().manual_seed(99991 + pos)
        return torch.randn(self.logits_width, generator=g)

    def _match(self, prefix: torch.Tensor, pos: int):
        for e in self.entries:
            L = e.input_ids.numel()
            if pos < L and torch.equal(prefix, e.input_ids[: pos + 1]):
                return e, "known"
            if pos >= L and torch.equal(prefix[:L], e.input_ids):
                return e, "beyond"  # past the last golden logits row (e.g. the final token of prompt + continuation)
        return None, "mismatch"

    def _logits_at(self, kv, slot, pt_row, pos) -> torch.Tensor:
        self.lookups += 1
        prefix = self._prefix(kv, pt_row, pos)
        e = kv.extra["entry"][slot]
        state = (
            "known"
            if (e is not None and pos < e.input_ids.numel() and torch.equal(prefix, e.input_ids[: pos + 1]))
            else None
        )
        if state is None:
            e, state = self._match(prefix, pos)
            kv.extra["entry"][slot] = e
        if state == "mismatch":
            self.mismatches += 1
            return self._garbage(pos)
        if state == "beyond":
            self.beyond += 1
            return self._garbage(pos)
        T = e.prompt_ids.numel()
        if pos >= T - 1:
            j = pos - (T - 1)
            row = e.cont[j].clone()
            if self.perturb_every and j % self.perturb_every == self.perturb_every - 1:
                top = set(torch.topk(row, 100).indices.tolist())
                bad = next(t for t in torch.argsort(row).tolist() if t not in top and t < TOKENIZER_VOCAB)
                row[bad] = row.max() + 5.0
                self.perturbed_positions.append(pos)
            return row
        row = torch.full((self.logits_width,), float(e.top32_values[pos, -1]) - 50.0)
        row[e.top32_indices[pos]] = e.top32_values[pos].float()
        return row

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None):
        P = tokens.numel()
        if P != round_up(n_valid, MODEL_PREFILL_ALIGN):
            raise ContractViolation(f"model chunk of {P} rows for {n_valid} valid tokens")
        if chunk_start % BLOCK_SIZE:
            raise ContractViolation(f"chunk start {chunk_start} not a page multiple")
        self.device.compile(("prefill", P))
        pt_row = kv_cache.extra["pt"][slot]
        if chunk_start == 0:
            kv_cache.extra["entry"][slot] = None
        self._write_tokens_paged(kv_cache, pt_row, chunk_start, tokens)  # padded rows land in own pages
        if logits_rows == "all":
            self.device.compile(("head_all", P))
            rows = [self._logits_at(kv_cache, slot, pt_row, chunk_start + i) for i in range(n_valid)]
            pad = [torch.full((self.logits_width,), NAN)] * (P - n_valid)
            return torch.stack(rows + pad)
        if logits_rows == "last":
            self.device.compile(("head_last",))
            row = self._logits_at(kv_cache, slot, pt_row, chunk_start + n_valid - 1)[None]
            if sample is not None:
                return self._sample_prefill_row(row[0], slot, sample, io)
            return row
        return None

    def _decode_programs(self, B):
        return [("decode", B), ("plus_one", B)]

    def decode(self, io, kv, width):
        B = width
        self.device.compile(("decode", B))
        pos = io.cur_pos._t.long()
        for s in range(B):
            p = int(pos[s])
            if p < 0:
                io.logits._t[s] = 0.0  # masked inactive row (finite, never consumed)
                continue
            pt_row = io.page_table._t[s].long()
            self._write_tokens_paged(kv, pt_row, p, io.tokens._t[s : s + 1])
            io.logits._t[s] = self._logits_at(kv, s, pt_row, p)
        self._log_decode(io, B)
        self.device.compile(("plus_one", B))
        self._advance_positions(io, B)
        self.decode_calls += 1
