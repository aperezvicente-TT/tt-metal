# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter for zai-org/GLM-5.3-Flash (`glm5_next`) on the standalone TT vLLM plugin.

The adapter translates vLLM scheduler inputs into the low-level API of the full-model generator
(`tt/generator.py::Glm53Generator`, contract: bring-up workspace doc/ttnn/06a-full-model-prep/contract.md section 2.1)
and owns nothing else: no model math, no sampling (the generator's split-sampling path samples on device; vLLM's host
sampler handles requests the device sampler cannot), no trace capture (the generator owns the two-phase warm-up and
every trace). Mapping and remaining contract gaps: doc/ttnn/09a-vllm-prep/generator_interface.md.

Cache ownership in serving mode
-------------------------------
* MLA/DSA layers (11: 3, 7, ..., 43): one `MLAAttentionSpec` per layer -- 64-token pages, 1 KV head of 512 values
  (kv_lora_rank; NoPE so no rope part), bf16 = 1024 B per token per layer; vLLM owns the page pool and the block
  tables, the generator holds the latent pages ([blocks, 1, 64, 512], Stage 1 `cache_semantics.mla`) and the DSA
  pooled-key companion (128 bf16 per 4 tokens) under the same block ids.
* KDA layers (34): no KV pages. Per-slot recurrent state [64, 128, 128] fp32 + the last 3 pre-conv q|k|v rows [3, 24576]
  bf16 (Stage 1 `cache_semantics.kda`) live in the generator, one per decode slot. vLLM's state-slot lifecycle drives
  them: `empty_slots` (prefill) places a request, `slot_remap` (decode) moves requests between rows, `release_request`
  frees one. The adapter applies `slot_remap` as a host-side indirection (vLLM slot -> generator slot, passed as the
  generator's `slots=`), so no device state is ever copied for a remap. Live requests are packed into the lowest
  generator slots only when the generator has several decode trace widths (`decode_buckets`): the served TP8 stack
  (tt/stack_model.py) has the native widths of GLM53_STACK_NATIVE_WIDTHS (e.g. 8 / 16) plus the narrow width-1 trace
  (GLM53_STACK_NARROW_WIDTHS=1), so `_compact` moves live requests down into the smallest width;
  a full-width backend (the Stage 4/6 layer pipeline, tt/model.py) never moves state.

Device (host-8chip 2x4 Blackhole mesh)
-------------------------------------
The plugin opens the parent mesh from `MESH_DEVICE="(2, 4)"` with the fabric this class declares
(`model_capabilities["fabric_config"]`); the launch environment names the 8-chip mesh graph descriptor
(`TT_MESH_GRAPH_DESC_PATH`, checked at startup). The serving backend module (`GLM53_BACKEND_MODULE`) builds on that
parent: the served line is Stage 14b's TP8 stack (tt/stack_model.py, restored from its DRAM image by tt/tp_image.py);
the default module is the Stage 4/6 layer pipeline (1x1 submeshes + socket line, tt/multichip_decoder.py).

Context
-------
Served `max_model_len` defaults to `target_context` of doc/context_contract.json (1,048,576 = the HF advertised context;
KDA layers have no positional limit). While the contract records a smaller `current_supported_context` from a
stage-scope directive (e.g. the MLA/DSA dense-exact regime, kv_len <= 2051, before the sparse DSA path), the class
declares it as `model_capabilities["tt_request_context_limit"]`; the plugin rejects longer requests at admission (HTTP
400) instead of serving them wrong or killing the engine. A hard physical limit (`capability_reduction.
hard_physical_limit`, with its capacity evidence) lowers the served `max_model_len` instead: the default becomes
`vllm_serving.served_max_model_len` (<= current_supported_context, no request limit) and a launch asking for more is
refused. doc/context_contract.json records the served request limit and why it is an implementation limit.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import inspect
import json
import math
import os
from models.autoports.zai_org_glm_5_3_flash.tt import k4_defaults  # noqa: F401,E402 -- the k4 knob defaults (setdefault)
from models.autoports.zai_org_glm_5_3_flash.tt import grid_pin  # noqa: F401,E402 -- GLM53_GRID_X / Y (the cross-host grid pin, program 02:12:31Z)
import queue
import threading
import time
import weakref
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import torch
from loguru import logger

from models.autoports.zai_org_glm_5_3_flash.tt import prefix_cache as PC

MODEL_DIR = Path(__file__).resolve().parents[1]
# GLM53_CONTEXT_CONTRACT: a context-ladder rung's contract (ladder_contract.py: the program contract with the rung's
# current_supported_context = what the stack serves at that rung), read the same way at import (the class's
# tt_request_context_limit) and at initialize_vllm_model; unset = the program contract doc/context_contract.json
CONTEXT_CONTRACT_ENV = "GLM53_CONTEXT_CONTRACT"
CONTEXT_CONTRACT_PATH = Path(os.environ.get(CONTEXT_CONTRACT_ENV) or MODEL_DIR / "doc" / "context_contract.json")

# Stage-1 cache contract (doc/context_contract.json `cache_semantics`).
BLOCK_SIZE = 64  # latent page = MLA prefill alignment = continuation granularity
LATENT_DIM = 512  # kv_lora_rank; qk_rope_head_dim = 0 (NoPE)
LATENT_DTYPE = torch.bfloat16
POOL_TOKENS = 4  # index_kpool
POOLED_KEY_DIM = 128  # index_head_dim
KDA_STATE_DTYPE = torch.float32
KDA_CONV_DTYPE = torch.bfloat16
MAX_STATE_SLOTS = 32  # FusedKDA.allocate_state cap (Stage 2 context_contract fused_decoder.max_kda_decode_slots)

KDA_LAYER_TYPES = frozenset({"linear_attention"})
# The checkpoint spells MLA/DSA `deepseek_sparse_attention`; transformers >= 5.16 normalises it to
# `indexed_attention` (configuration_utils `_LEGACY_LAYER_TYPE_REMAP`).
MLA_LAYER_TYPES = frozenset({"deepseek_sparse_attention", "indexed_attention"})

# Opt-in factory override for the generator (the host-only tests build Glm53Generator over 06a's CPU MockModel), as
# "package.module:function" with the signature of tt/generator.py::build_generator.
GENERATOR_FACTORY_ENV = "GLM53_VLLM_GENERATOR_FACTORY"
# KV pool size in tokens (shared by all requests); default in `get_max_tokens_all_users`.
KV_POOL_TOKENS_ENV = "GLM53_VLLM_KV_POOL_TOKENS"
# Optional JSONL sink of serving events (warm-up end, every prefill call, every EVENT_DECODE_EVERY-th decode call,
# shutdown), each with the generator's trace report: trace captures / replays / refreshes and, on device, per-chip trace
# ids and program-cache entries -- the mixed-shape trace-reuse evidence HTTP responses cannot expose. Off the serving
# hot path: prefill / decode rows carry counter copies only (`_light_report`, no
# device query) and a writer thread serializes and appends them; warm-up / startup / shutdown rows carry the full
# trace_report() and wait until written.
EVENTS_ENV = "GLM53_VLLM_EVENTS"
# Stage 17 (doc/ttnn/17-mtp-spec-decode): K of the backend's speculative step (tt/spec_mtp.py, built by StackBackend
# when GLM53_SPEC_K is set before the model loads); 0 = no speculative decoding
SPEC_K = int(os.environ.get("GLM53_SPEC_K", "0"))
# the drafter kind of that step (tt/stack_model.py: GLM53_SPEC_KIND, alias GLM53_SPEC_DRAFTER): "mtp" (the model's own
# MTP head, K <= 3) or "dflash2" (the Apache-2.0 DFlash2 block drafter inside the step, K <= 7; vLLM method "dflash")
SPEC_KIND = os.environ.get("GLM53_SPEC_KIND") or os.environ.get("GLM53_SPEC_DRAFTER", "mtp")
# Stage 9 (program 2026-10-26T23:57:28Z): the speculative step writes candidate KDA rows / conv histories into rows
# 1..K of the shared 32-row state, i.e. into OTHER generator slots' rows. 1 = speculate only while no other generator
# slot holds live state (a mid-prefill request between its chunks, a decoding or held request): the adapter checks
# its own slot mirror and the generator binding before every proposal and every drafted verify (the generator's
# spec_enter checks the binding as well). 0 = only spec_enter's binding check (the pre-guard behaviour).
SPEC_SOLE_STATE = os.environ.get("GLM53_SPEC_SOLE_STATE", "1") != "0"
# Stage 17 async overlap (program 2026-10-23 10:23:07Z / 10:30:22Z): after a lone speculating step's record the next
# step is enqueued at once (its inputs -- the bonus and the drafts -- are already on device), so the device runs it during
# the runner's host turn; the next verify block reads its record. A deviating call commits nothing on device: the
# tokens the pre-enqueued step committed are HELD and answer slot 0's next inputs until the runner reaches the device's
# position. The launch may then drop --no-async-scheduling (the plain multi-user steps keep their overlap).
SPEC_ASYNC = os.environ.get("GLM53_SPEC_ASYNC", "0") == "1"
EVENT_DECODE_EVERY = 256
# Host overhead per decode step (the serving target: TPOT = the device step + <= 1 ms of host). Every
# decode call times the wall since the previous call; on a STEADY step (decode right after decode, resident inputs, no
# prefill / probe / speculative step in between) that interval is the served step time, kept per live-row count.
# GLM53_DECODE_PROBE_EVERY=N (0 = off) drains the device before every N-th steady step and synchronizes after its
# submit: the probe times the step alone (nothing overlapped), so host overhead per step = steady wall - probe device
# (trace_report()["host_step"]). A probe costs one pipeline bubble (about the host's per-step work) on that step.
DECODE_PROBE_ENV = "GLM53_DECODE_PROBE_EVERY"
HOST_STEP_WINDOW = 1024  # the most recent steady intervals / probes kept per live-row count
# Engine gaps (a 2,048-token request's first token once came 1.56 s after its prefill returned, 0.10 s on a fresh
# server): the time between the end of one model call and the start of the next while a
# request is live; >= ENGINE_GAP_EVENT_S -> an `engine_gap` event (with the gen-2 GC pause time seen meanwhile).
ENGINE_GAP_EVENT_S = 0.5
# Stall sampler (a 1.29-s gap was once seen between two decode calls of a lone request, no gen-2 GC, outside the
# adapter): with the events sink on, a daemon thread samples the engine thread's Python stack when no model call has run
# for STALL_SAMPLE_S while a request is live (an `engine_stall_stack` event, at most STALL_SAMPLES per stall, every
# STALL_SAMPLE_S) -> where the engine waits (a device event wait, the scheduler, output processing). GLM53_STALL_SAMPLER=0
# off.
STALL_SAMPLE_S = 0.4
STALL_SAMPLES = 3
# Design point of the 8-chip DRAM budget (doc/architecture.md section 9): 32 users x 32K tokens.
DEFAULT_TOKENS_PER_USER = 32768
# The 8-chip parent mesh needs its graph descriptor before the plugin opens it (multichip_decoder.open_pipeline_mesh).
MESH_GRAPH_DESC_ENV = "TT_MESH_GRAPH_DESC_PATH"
MESH_GRAPH_DESCRIPTORS = {(2, 4): "glm8_2x4_mesh_graph_descriptor.textproto"}
# Stage 12 fast start: GLM53_WEIGHT_SOURCE=image|tensors. The layer pipeline (the default backend module):
# tt/fast_start.py's generator factory (DRAM-image weight pull with fallback to the tensor loader; its own timeline JSON at
# GLM53_TIMELINE_PATH). A TP stack module (tt/generator.py TP_IMAGE_MODULES): tt/generator.py with the tt/tp_image.py
# restore (GLM53_IMAGE_REQUIRED=1: no tensor load; GLM53_IMAGE_PIN: one image). Unset = tt/generator.py.
WEIGHT_SOURCE_ENV = "GLM53_WEIGHT_SOURCE"
FAST_START_FACTORY = "models.autoports.zai_org_glm_5_3_flash.tt.fast_start:build_generator"
FAST_START_TIMELINE_ENV = "GLM53_TIMELINE_PATH"
# JSON file the serving startup timeline is written to at model ready (serve_window.sh: $RUN/startup_timeline.json).
STARTUP_TIMELINE_ENV = "GLM53_STARTUP_TIMELINE"
# The serving decoder (tt/model.py build_model_backend `decoder=`): unset = today's op-chain decoder ("optimized", one
# decode step in flight = K 1); "persistent" = Stage 7a's per-layer persistent programs once tt/model.py lists them in
# SERVING_DECODERS. A decoder this tree lacks falls back to "optimized".
DECODER_ENV = "GLM53_DECODER"
DEFAULT_DECODER = "optimized"


class _PipelineFabric(Mapping):
    """`model_capabilities["fabric_config"]`: the set_fabric_config keywords the serving backend was validated with,
    from its module (tt/generator.py GLM53_BACKEND_MODULE; default tt/model.py pipeline_fabric_config = Stage 4
    open_pipeline_mesh: FABRIC_1D, router payload FABRIC_PAYLOAD_BYTES; a module may define serving_fabric_config()
    with the same two keys). The plugin's default reliability mode STRICT_INIT is the device default too. Built on
    first access in the worker, so importing this module needs no device stack."""

    def _kwargs(self) -> dict:
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import BACKEND_MODULE_ENV, DEFAULT_BACKEND_MODULE

        mod = importlib.import_module(os.environ.get(BACKEND_MODULE_ENV) or DEFAULT_BACKEND_MODULE)
        fn = getattr(mod, "serving_fabric_config", None) or getattr(mod, "pipeline_fabric_config")
        # None = the binding's default (ttnn.set_fabric_config takes no None for router_config)
        return {k: v for k, v in fn().items() if v is not None}

    def __getitem__(self, key):
        return self._kwargs()[key]

    def __iter__(self):
        return iter(self._kwargs())

    def __len__(self):
        return len(self._kwargs())

    def __repr__(self):
        return f"_PipelineFabric({os.environ.get('GLM53_BACKEND_MODULE') or 'tt/model.py pipeline_fabric_config'})"


def _states_dsa(states) -> bool:
    """a prefix checkpoint (export_kda_state's list) carries the slot's DSA indexer rows (sparse regime)"""
    return isinstance(states, (list, tuple)) and any(isinstance(d, dict) and d.get("dsa") is not None for d in states)


def _states_pools_less(states) -> bool:
    """a prefix checkpoint whose DSA record has no pooled-key rows (GLM53_KV_POOLED_PAGES: they ride in the LMCache
    pages; the plugin attaches them as blob["pooled"])"""
    return isinstance(states, (list, tuple)) and any(
        isinstance(d, dict) and isinstance(d.get("dsa"), dict) and d["dsa"].get("pools") is False for d in states)


def _hit_restart(blob, start: int, paged: bool, hit_dma) -> str | None:
    """why a connector hit continuing at `start` (sparse regime) cannot resume from its checkpoint `blob` (None: it can):
    "no_dsa" -- the record has no DSA rows (the slot's pooled keys of [0, start) would be its previous owner's);
    "no_pooled" -- the record is ring-only (GLM53_KV_POOLED_PAGES) or the store is paged by latent block
    (GLM53_DSA_PAGED_POOLS: the hit's freshly admitted blocks hold stale keys), and the hit does not bring its pages'
    pooled keys up to `start` over the page-DMA. Under a paged store the CACHE decides, not the record: a checkpoint
    written before Stage 12's 2026-10-23 fix says pools=True and carries the TP prefill alias's rows."""
    states = blob.get("states")
    if not _states_dsa(states):
        return "no_dsa"
    ring_only = _states_pools_less(states)
    if not (paged or ring_only):
        return None
    p = blob.get("pooled")
    if ring_only and p is not None and int(p["units"]) * 128 >= int(start) and hit_dma is not None:
        return None
    return "no_pooled"


def _blob_bytes(obj) -> int:
    """host bytes of a checkpoint / host-tier blob (every tensor in it)"""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, Mapping):
        return sum(_blob_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_blob_bytes(v) for v in obj)
    return 0


def _check_device_mesh(mesh_device) -> None:
    """The device generator builds the layer pipeline on the mesh the plugin opened: it must be the 2x4 parent mesh,
    opened with its graph descriptor (MESH_DEVICE="(2, 4)", TT_MESH_GRAPH_DESC_PATH in the launch environment)."""
    shape = (int(mesh_device.shape[0]), int(mesh_device.shape[1]))
    want = MESH_GRAPH_DESCRIPTORS.get(shape)
    if want is None:
        raise ValueError(
            f'mesh {shape}: the GLM-5.3-Flash layer pipeline serves the 2x4 Blackhole mesh (MESH_DEVICE="(2, 4)")'
        )
    path = os.environ.get(MESH_GRAPH_DESC_ENV, "")
    if Path(path).name != want or not Path(path).is_file():
        raise ValueError(
            f"{MESH_GRAPH_DESC_ENV}={path!r}: launch with {MESH_GRAPH_DESC_ENV}={MODEL_DIR / 'mesh' / want} "
            "(the plugin opens the parent mesh before the model loads)"
        )


# --------------------------------------------------------------------------------------------- context contract
@dataclass(frozen=True)
class ContextContract:
    """The serving-relevant fields of doc/context_contract.json."""

    advertised: int
    target: int
    supported: int
    supported_by_kind: dict[str, int]
    reduction: dict[str, Any] | None
    # vllm_serving.served_max_model_len when a hard physical limit lowers the served context (None: the target)
    served: int | None = None

    @property
    def served_max_model_len(self) -> int:
        return self.target if self.served is None else self.served

    @property
    def request_context_limit(self) -> int:
        """Largest prompt + output length the current model computes faithfully, or 0 when that is the full
        served context. Only a stage-scope directive may sit here; a hard physical limit lowers the served
        `max_model_len` instead (vllm_serving.served_max_model_len, with evidence in the contract)."""
        if self.supported >= self.served_max_model_len:
            return 0
        if self.reduction is not None and self.reduction.get("hard_physical_limit"):
            raise ValueError(
                f"context contract records a hard physical limit ({self.supported} < {self.target}); serve "
                "max_model_len = the largest feasible value instead of a request limit"
            )
        return self.supported


def load_context_contract(path: Path | str = CONTEXT_CONTRACT_PATH) -> ContextContract:
    data = json.loads(Path(path).read_text())
    reduction = data.get("capability_reduction")
    supported = int(data["current_supported_context"])
    served = None
    if reduction is not None and reduction.get("hard_physical_limit"):
        # a hard physical limit lowers the served max_model_len: the contract names the value it serves
        served = (data.get("vllm_serving") or {}).get("served_max_model_len")
        if served is None or not 1 <= int(served) <= supported:
            raise ValueError(
                f"{path}: capability_reduction.hard_physical_limit needs vllm_serving.served_max_model_len in "
                f"[1, current_supported_context {supported}] (got {served})"
            )
        served = int(served)
    return ContextContract(
        advertised=int(data["hf_advertised_context"]),
        target=int(data["target_context"]),
        supported=supported,
        supported_by_kind={k: int(v) for k, v in data.get("supported_context_by_layer_kind", {}).items()},
        reduction=reduction,
        served=served,
    )


def _contract_request_limit() -> int:
    """The class capability, evaluated at import. Without the contract file the module still imports (0 here), but
    initialize_vllm_model loads the contract and refuses to start, so a missing contract never serves unlimited."""
    try:
        return load_context_contract().request_context_limit
    except FileNotFoundError:
        return 0


# ------------------------------------------------------------------------------------------------- cache spec
def text_config_of(hf_config: Any) -> Any:
    """The glm5_next text config (the checkpoint nests it under `text_config`)."""
    return getattr(hf_config, "text_config", None) or hf_config


def layer_kinds(text_config: Any) -> list[str]:
    """`layer_types` -> ['kda' | 'mla'] per decoder layer (MTP layer excluded)."""
    types = list(getattr(text_config, "layer_types", None) or ())
    n = int(text_config.num_hidden_layers)
    if len(types) != n:
        raise ValueError(f"layer_types has {len(types)} entries for {n} decoder layers")
    kinds = []
    for i, t in enumerate(types):
        if t in KDA_LAYER_TYPES:
            kinds.append("kda")
        elif t in MLA_LAYER_TYPES:
            kinds.append("mla")
        else:
            raise ValueError(f"layer {i}: unsupported layer type {t!r} (expected KDA or MLA/DSA)")
    return kinds


@dataclass(frozen=True)
class LayerCacheSpec:
    layer_idx: int
    kind: str  # "mla" | "kda"
    shape: tuple[int, ...]  # mla: latent pages [num_blocks, 1, 64, 512]; kda: recurrent state per slot set
    dtype: torch.dtype
    aux_shape: tuple[int, ...]  # mla: pooled-key pages [num_blocks, 1, 16, 128]; kda: conv tail [slots, 3, C]
    aux_dtype: torch.dtype

    @property
    def bytes(self) -> int:
        size = lambda shape, dtype: math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        return size(self.shape, self.dtype) + size(self.aux_shape, self.aux_dtype)


@dataclass(frozen=True)
class ServingCacheSpec:
    """Everything the generator allocates for vLLM serving: vLLM-paged latent caches for the MLA layers
    (`num_blocks` pages of `block_size` tokens) and model-owned KDA state for `max_slots` slots."""

    num_blocks: int
    block_size: int
    max_slots: int
    layers: tuple[LayerCacheSpec, ...]

    @property
    def mla_layers(self) -> list[int]:
        return [s.layer_idx for s in self.layers if s.kind == "mla"]

    @property
    def kda_layers(self) -> list[int]:
        return [s.layer_idx for s in self.layers if s.kind == "kda"]

    def summary(self) -> dict[str, Any]:
        mla = [s for s in self.layers if s.kind == "mla"]
        kda = [s for s in self.layers if s.kind == "kda"]
        elem = lambda dtype: torch.empty((), dtype=dtype).element_size()
        latent_tok = math.prod(mla[0].shape[1:]) * elem(mla[0].dtype) // self.block_size if mla else 0
        pooled_tok = math.prod(mla[0].aux_shape[1:]) * elem(mla[0].aux_dtype) // self.block_size if mla else 0
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "pool_tokens": self.num_blocks * self.block_size,
            "max_slots": self.max_slots,
            "mla_layers": len(mla),
            "kda_layers": len(kda),
            "latent_bytes_per_token_per_layer": latent_tok,
            "pooled_key_bytes_per_token_per_layer": pooled_tok,
            "mla_bytes_total": sum(s.bytes for s in mla),
            "kda_bytes_total": sum(s.bytes for s in kda),
            "kda_bytes_per_slot": sum(s.bytes for s in kda) // max(self.max_slots, 1),
        }


def build_serving_cache_spec(
    text_config: Any, *, num_blocks: int, block_size: int = BLOCK_SIZE, max_slots: int
) -> ServingCacheSpec:
    if block_size != BLOCK_SIZE:
        raise ValueError(f"GLM-5.3-Flash latent pages are {BLOCK_SIZE} tokens, vLLM block_size is {block_size}")
    if not 1 <= max_slots <= MAX_STATE_SLOTS:
        raise ValueError(f"max_slots {max_slots} outside [1, {MAX_STATE_SLOTS}] (KDA state slots)")
    if num_blocks < 1:
        raise ValueError(f"num_blocks {num_blocks} < 1")
    kda_cfg = getattr(text_config, "linear_attn_config", None) or {}
    heads = int(kda_cfg.get("num_heads", 64))
    head_dim = int(kda_cfg.get("head_dim", 128))
    conv_rows = int(kda_cfg.get("short_conv_kernel_size", 4)) - 1
    qkv = 3 * heads * head_dim
    latent_dim = int(getattr(text_config, "kv_lora_rank", LATENT_DIM)) + int(
        getattr(text_config, "qk_rope_head_dim", 0) or 0
    )
    pool = int(getattr(text_config, "index_kpool", POOL_TOKENS))
    pooled_dim = int(getattr(text_config, "index_head_dim", POOLED_KEY_DIM))
    if block_size % pool:
        raise ValueError(f"block_size {block_size} is not a multiple of index_kpool {pool}")
    layers = []
    for i, kind in enumerate(layer_kinds(text_config)):
        if kind == "mla":
            layers.append(
                LayerCacheSpec(
                    i,
                    "mla",
                    (num_blocks, 1, block_size, latent_dim),
                    LATENT_DTYPE,
                    (num_blocks, 1, block_size // pool, pooled_dim),
                    LATENT_DTYPE,
                )
            )
        else:
            layers.append(
                LayerCacheSpec(
                    i,
                    "kda",
                    (max_slots, heads, head_dim, head_dim),
                    KDA_STATE_DTYPE,
                    (max_slots, conv_rows, qkv),
                    KDA_CONV_DTYPE,
                )
            )
    return ServingCacheSpec(num_blocks=num_blocks, block_size=block_size, max_slots=max_slots, layers=tuple(layers))


# ---------------------------------------------------------------------------------------------- slot bookkeeping
@dataclass
class _SlotState:
    key: int  # first latent page of the owning request: unique among live requests, stable across row moves
    next_pos: int  # tokens already folded into the slot's KDA state (= position of the next token it consumes)


@dataclass
class StateSlotTable:
    """Host mirror of which request's KDA state each generator slot holds and how many tokens it has folded in. It
    moves no device state; it checks every prefill continuation and every reloading decode against the plugin's slot
    bookkeeping (a wrong slot would silently continue another request's recurrence) and supplies the resident
    positions of a non-reloading decode, whose host positions are stale by contract."""

    n_slots: int
    slots: dict[int, _SlotState] = field(default_factory=dict)
    #: prefix caching: live requests may share their first latent page (a cached prefix), so the key only confirms
    #: a continuation / decode row against its slot and no longer proves uniqueness across slots
    shared_pages: bool = False

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.n_slots:
            raise ValueError(f"state slot {slot} outside [0, {self.n_slots})")

    def check_prefill(self, slot: int, key: int, start: int, end: int, *, fresh: bool = False) -> None:
        """Validate one prefill row before the generator runs it (the mirror changes only once it is accepted).
        `fresh`: a new request that starts past 0 because a KV connector loaded its prefix (with its KDA checkpoint)."""
        self._check_slot(slot)
        if start == 0 or fresh:
            # fresh request (new, or resumed after preemption: replays its whole history)
            if slot in self.slots:
                raise RuntimeError(
                    f"fresh prefill into slot {slot}, which still holds a live request {self.slots[slot]}: a finished "
                    f"or preempted request must be released first (slots={self.describe()})"
                )
            for other, st in self.slots.items():
                if other != slot and st.key == key and not self.shared_pages:
                    raise RuntimeError(
                        f"fresh prefill at slot {slot} reuses page {key}, still recorded as live at slot {other}: "
                        f"slots={self.describe()}"
                    )
        else:
            st = self.slots.get(slot)
            if st is None or st.key != key or st.next_pos != start:
                holder = [s for s, v in self.slots.items() if v.key == key]
                raise RuntimeError(
                    f"prefill continuation [{start}, {end}) at slot {slot} for the request owning page {key}: the "
                    f"slot holds {st}; the request's state is at {holder or 'no slot'}. The continuation must run "
                    f"in the slot that holds the request's KDA state (slots={self.describe()})"
                )

    def commit_prefill(self, slot: int, key: int, end: int) -> None:
        self.slots[slot] = _SlotState(key=key, next_pos=end)

    def begin_prefill(self, slot: int, key: int, start: int, end: int) -> None:
        self.check_prefill(slot, key, start, end)
        self.commit_prefill(slot, key, end)

    def check_decode_inputs(self, slots: Sequence[int], positions: Sequence[int], keys: Sequence[int]) -> None:
        """Reloading decode: row i (generator slot slots[i]) consumes the token at positions[i] (-1 = inactive)."""
        for row, (slot, pos, key) in enumerate(zip(slots, positions, keys)):
            if pos < 0:
                continue
            st = self.slots.get(slot)
            if st is None or st.key != key or st.next_pos != pos:
                raise RuntimeError(
                    f"decode row {row} consumes position {pos} of the request owning page {key}, but its generator "
                    f"slot {slot} holds {st} (slots={self.describe()})"
                )

    def resident_positions(self, slots: Sequence[int], active: set[int]) -> list[int]:
        return [self.slots[s].next_pos if s in active else -1 for s in slots]

    def advance(self, active: Sequence[int]) -> None:
        for slot in active:
            self.slots[slot].next_pos += 1

    def release(self, slot: int) -> None:
        self.slots.pop(int(slot), None)

    def move(self, src: int, dst: int) -> None:
        self._check_slot(dst)
        if dst in self.slots:
            raise RuntimeError(f"slot move {src} -> {dst}: the destination holds {self.slots[dst]}")
        self.slots[dst] = self.slots.pop(src)

    def describe(self) -> dict[int, tuple[int, int]]:
        return {s: (v.key, v.next_pos) for s, v in sorted(self.slots.items())}


# ------------------------------------------------------------------------------------------- generator interface
class ServingGenerator(Protocol):
    """The part of `tt/generator.py::Glm53Generator` (06a contract.md sections 2.1 and 9) the adapter calls.

    `allocate_kv_cache(max_batch, max_seq_len, num_blocks)` (latent pool sized from vLLM's num_blocks);
    `warmup_serving(kv, phase="prepare" | "capture", prefill=, decode=)` (two-phase, idempotent);
    `prefill_forward(tokens [B, P], *, page_table [B, blocks], kv_cache, prompt_lens, empty_slots, start_pos,
    sampling_params: list[tt.sampling.SamplingParams] | None)` -> logits [B, 1, V] or first tokens [B];
    `decode_forward(tokens [B, 1] (-1 = the slot's device-resident next token), start_pos [B] (-1 = inactive; else the
    slot's cache length), *, page_table [B, blocks] | None (None = unchanged), kv_cache, slots, enable_trace,
    sampling_params, read_from_device)` -> logits [B, V], tokens [B] or a device-resident `DecodeOutput`; it writes only
    the inputs that differ from its device copies; `read_decode_output(out)` -> tokens [B], or with `async_read=True`
    (pending, events) completed by `process_decode_output_host(pending)` -> tokens [B, 1];
    `remap_slots(kv, {src: dst})` (moves a live request's state to a free slot); `release_slot(kv, slot)`;
    `trace_report()`; `teardown()`; optional `close()` (teardown + the backend's device resources, mesh still open).
    """

    max_batch: int
    max_seq_len: int
    supported_context: int
    decode_buckets: tuple[int, ...]  # decode trace widths; non-empty -> the adapter packs live requests low

    def allocate_kv_cache(
        self,
        max_batch: int | None = None,
        max_seq_len: int | None = None,
        num_blocks: int | None = None,
        null_block: int | None = None,
    ) -> Any:
        ...

    def warmup_serving(self, kv_cache: Any, *, phase: str, prefill: bool, decode: bool) -> None:
        ...

    def prefill_forward(self, tokens: torch.Tensor, **kwargs: Any) -> Any:
        ...

    def decode_forward(self, tokens: torch.Tensor, start_pos: torch.Tensor, **kwargs: Any) -> Any:
        ...

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        ...

    def process_decode_output_host(self, host_out: Any, is_tokens: bool = True) -> Any:
        ...

    def remap_slots(self, kv_cache: Any, mapping: dict[int, int]) -> None:
        ...

    def release_slot(self, kv_cache: Any, slot: int) -> None:
        ...

    def trace_report(self) -> dict:
        ...

    def teardown(self) -> None:
        ...


SERVING_API = (
    "allocate_kv_cache",
    "warmup_serving",
    "prefill_forward",
    "decode_forward",
    "read_decode_output",
    "process_decode_output_host",
    "remap_slots",
    "release_slot",
    "trace_report",
    "teardown",
)


def _resolve_factory(spec: str | None):
    if not spec:
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator

        return build_generator
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"{GENERATOR_FACTORY_ENV}={spec!r}: expected 'package.module:function'")
    return getattr(importlib.import_module(module), name)


def _process_start_epoch() -> float | None:
    """This process's start time (Linux /proc): the EngineCore launch, before vLLM imports the model."""
    try:
        ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        boot = next(float(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime"))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


_STARTUP_GAUGE = None
_CONTEXT_GAUGE = None
# Concurrent-user counts the endpoint advertises its max ISL + OSL per user for; max_num_seqs itself is always
# included.
CONTEXT_PROFILE_USERS = (1, 2, 4, 8, 16, 32)


def context_by_users(num_blocks: int, max_model_len: int, request_limit: int, max_num_seqs: int) -> dict[int, int]:
    """Largest prompt + output length that N concurrent users can each hold at once: the shared latent pool (vLLM's
    `num_blocks` pages of 64 tokens minus its null block 0) split N ways in whole pages, capped by the served
    `max_model_len` and the request context limit (0 = none). vLLM admits longer requests while the pool has room and
    preempts (recompute) when decode growth exhausts it; this table is the no-preemption guarantee the endpoint
    advertises (doc/ttnn/09-vllm/isl_osl_contract.md)."""
    pages = int(num_blocks) - 1
    cap = min(int(max_model_len), int(request_limit) or int(max_model_len))
    users = sorted({n for n in CONTEXT_PROFILE_USERS if n <= int(max_num_seqs)} | {int(max_num_seqs)})
    return {n: min(cap, pages // n * BLOCK_SIZE) for n in users}


class StartupTimeline:
    """Serving startup timeline: wall-clock marks from the EngineCore process
    start to the model being ready to serve (the end of the last warm-up call), as phases: engine_start (process start
    -> initialize_vllm_model), load (the generator factory: pipeline plan, sockets, weights; GLM53_WEIGHT_SOURCE selects
    Stage 12's loader), kv_cache, warmup (phase-1 preparation of prefill + decode), capture (phase-2 trace captures),
    total. Emitted once, at model ready: printed (`[glm53-serving] startup ...`), as the adapter event `startup`, as
    JSON at $GLM53_STARTUP_TIMELINE (Stage 12's $GLM53_TIMELINE_PATH timeline merged in when present), and as the
    Prometheus gauge `glm53_startup_seconds{phase}` when the API server collects multiprocess metrics
    (PROMETHEUS_MULTIPROC_DIR set at launch). The server's /health time is the launcher's (serve_window.sh)."""

    PHASES = (
        ("engine_start", "process_start", "model_init"),
        ("load", "model_init", "model_built"),
        ("kv_cache", "model_built", "kv_cache_allocated"),
        ("warmup", "kv_cache_allocated", "warmup_done"),
        ("capture", "warmup_done", "model_ready"),
        ("total", "process_start", "model_ready"),
        # the warm-up / capture halves (restart_time.md): prefill prepare, decode prepare, prefill capture, decode capture
        ("warmup_prefill", "kv_cache_allocated", "warmup_prefill_done"),
        ("warmup_decode", "warmup_prefill_done", "warmup_done"),
        ("capture_prefill", "warmup_done", "capture_prefill_done"),
        ("capture_decode", "capture_prefill_done", "model_ready"),
    )

    def __init__(self):
        self.marks: dict[str, float] = {}
        start = _process_start_epoch()
        if start is not None:
            self.marks["process_start"] = start
        self.info: dict[str, Any] = {}
        self.emitted = False

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.time())

    def phases(self) -> dict[str, float]:
        m = self.marks
        return {p: round(m[b] - m[a], 3) for p, a, b in self.PHASES if a in m and b in m}

    def report(self) -> dict:
        utc = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"
        return {"phases_s": self.phases(), "marks_utc": {k: utc(v) for k, v in self.marks.items()}, **self.info}

    def emit(self) -> dict:
        """At model ready: print, write $GLM53_STARTUP_TIMELINE, export the gauges; returns the report."""
        self.emitted = True
        rep = self.report()
        fast = os.environ.get(FAST_START_TIMELINE_ENV)
        if fast and Path(fast).is_file():
            try:
                rep["fast_start"] = json.loads(Path(fast).read_text())
            except ValueError:
                rep["fast_start"] = {"unreadable": fast}
        rep["prometheus"] = self._export(rep["phases_s"], self.info.get("context_by_users"))
        print(f"[glm53-serving] startup {json.dumps(rep['phases_s'])} {json.dumps(self.info)}", flush=True)
        path = os.environ.get(STARTUP_TIMELINE_ENV)
        if path:
            Path(path).write_text(json.dumps(rep, indent=1) + "\n")
        return rep

    @staticmethod
    def _export(phases: dict, context: dict | None = None) -> bool:
        global _STARTUP_GAUGE, _CONTEXT_GAUGE
        if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            return False
        try:
            from prometheus_client import Gauge
        except ImportError:
            return False
        if _STARTUP_GAUGE is None:
            _STARTUP_GAUGE = Gauge(
                "glm53_startup_seconds",
                "GLM-5.3-Flash serving startup phase durations in seconds (EngineCore process start to model ready)",
                ["phase"],
                multiprocess_mode="max",
            )
        for phase, value in phases.items():
            _STARTUP_GAUGE.labels(phase=phase).set(value)
        if context:
            if _CONTEXT_GAUGE is None:
                _CONTEXT_GAUGE = Gauge(
                    "glm53_max_context_tokens",
                    "GLM-5.3-Flash max prompt + output tokens per user when this many users are served at once",
                    ["concurrent_users"],
                    multiprocess_mode="max",
                )
            for users, tokens in context.items():
                _CONTEXT_GAUGE.labels(concurrent_users=str(users)).set(tokens)
        return True


@contextlib.contextmanager
def _phase_profile(name: str):
    """GLM53_STARTUP_PROFILE=1 (host mocks / diagnostics): cProfile of one startup phase -> startup_profile_<name>.txt
    (top 45 by cumulative and by tottime) in GLM53_STARTUP_PROFILE_DIR (default: the $GLM53_STARTUP_TIMELINE dir).
    Adds ~10-20 % to the phase: never in a served run."""
    if os.environ.get("GLM53_STARTUP_PROFILE") != "1":
        yield
        return
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        out = (
            os.environ.get("GLM53_STARTUP_PROFILE_DIR")
            or os.path.dirname(os.environ.get(STARTUP_TIMELINE_ENV, ""))
            or "."
        )
        path = os.path.join(out, f"startup_profile_{name}.txt")
        try:
            with open(path, "w") as f:
                for key in ("cumulative", "tottime"):
                    pstats.Stats(pr, stream=f).sort_stats(key).print_stats(45)
            print(f"[glm53-serving] startup profile {name}: {path}", flush=True)
        except OSError as e:
            print(f"[glm53-serving] startup profile {name}: not written ({e})", flush=True)


def _weight_source_factory(info: dict):
    """GLM53_WEIGHT_SOURCE set: for the layer pipeline (the default backend module), Stage 12's tt/fast_start.py
    generator factory (same contract as tt/generator.py's; when this tree has no tt/fast_start.py the tensor loader
    serves and the timeline says so); for a TP stack module (tt/generator.py TP_IMAGE_MODULES), tt/generator.py itself
    (tt/tp_image.py restore); any other backend module has no image path (raises: fast_start would build the
    pipeline)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
        BACKEND_MODULE_ENV,
        DEFAULT_BACKEND_MODULE,
        TP_IMAGE_MODULES,
        check_image_required,
    )

    info["weight_source_requested"] = source = os.environ.get(WEIGHT_SOURCE_ENV)
    module = os.environ.get(BACKEND_MODULE_ENV) or DEFAULT_BACKEND_MODULE
    if module in TP_IMAGE_MODULES:  # (tt/generator.py checks GLM53_IMAGE_REQUIRED against the source)
        how = "tt/tp_image.py restore" if source == "image" else f"{WEIGHT_SOURCE_ENV}={source}: tensor loader"
        info["generator_factory"] = f"tt/generator.py:build_generator ({how})"
        return _resolve_factory(None)
    check_image_required(False, f"backend module {module} (tt/fast_start.py: the layer pipeline)")
    if module != DEFAULT_BACKEND_MODULE:
        raise ValueError(f"{WEIGHT_SOURCE_ENV}={info['weight_source_requested']}: no DRAM-image path for backend "
                         f"module {module} ({BACKEND_MODULE_ENV}; tt/fast_start.py builds the layer pipeline only)")  # fmt: skip
    try:
        factory = _resolve_factory(FAST_START_FACTORY)
        info["generator_factory"] = FAST_START_FACTORY
        return factory
    except (ImportError, AttributeError) as e:
        print(
            f"[glm53-serving] {WEIGHT_SOURCE_ENV}={info['weight_source_requested']} but {FAST_START_FACTORY} is not "
            f"available ({e!r}): tensor loader (tt/generator.py)",
            flush=True,
        )
        info["generator_factory"] = "tt/generator.py:build_generator (fast_start unavailable)"
        return _resolve_factory(None)


def _serving_decoder(info: dict) -> str | None:
    """GLM53_DECODER -> the `decoder=` the generator factory forwards to tt/model.py (None when unset: the factory's
    default). Checked against tt/model.py SERVING_DECODERS before anything loads; a decoder this tree does not provide
    serves as "optimized" (printed, and recorded in the startup timeline)."""
    want = os.environ.get(DECODER_ENV)
    if not want:
        return None
    from models.autoports.zai_org_glm_5_3_flash.tt import model as tt_model

    have = tuple(getattr(tt_model, "SERVING_DECODERS", (DEFAULT_DECODER,)))
    active = want if want in have else DEFAULT_DECODER
    if active != want:
        print(
            f"[glm53-serving] {DECODER_ENV}={want} is not in tt/model.py SERVING_DECODERS {have}: serving "
            f"{DEFAULT_DECODER} (the K=1 op-chain decoder)",
            flush=True,
        )
    info.update(decoder_requested=want, decoder_active=active)
    return active


def vllm_page_rows(page_table: torch.Tensor) -> torch.Tensor:
    """The runner's block-table rows -> the generator's page-table rows. vLLM pads each row with block 0, its null
    block (`BlockPool.null_block`, never a request's page); the generator treats the entries before the first negative
    one as the request's pages (its coverage). Marking the padding -1 makes the coverage the request's allocation, so
    a position outside it (a stale page table) is refused instead of addressing the null block."""
    table = page_table.to(torch.int32)
    return torch.where(table == 0, torch.full_like(table, -1), table)


def _as_list(x: Any) -> list[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.reshape(-1).tolist()]
    return [int(v) for v in x]


def _field_rows(params: Any, name: str, rows: int, default):
    value = getattr(params, name, None)
    if value is None:
        return [default] * rows
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value] * rows
    value = list(value)[:rows]
    return value + [default] * (rows - len(value))


def generator_sampling_params(tt_params: Any, rows: int) -> list:
    """vLLM-side TTSamplingParams (per-row lists) -> the generator's per-row `tt.sampling.SamplingParams`. Penalties
    are host-only for this model (`supports_device_penalties=False`): a device step that carries them is refused."""
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

    for name, neutral in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
        bad = [v for v in _field_rows(tt_params, name, rows, neutral) if v is not None and float(v) != neutral]
        if bad:
            raise ValueError(f"device sampling step carries {name}={bad}; penalties are host-sampled for this model")
    temps = _field_rows(tt_params, "temperature", rows, 0.0)
    top_k = _field_rows(tt_params, "top_k", rows, 0)
    top_p = _field_rows(tt_params, "top_p", rows, 1.0)
    seeds = _field_rows(tt_params, "seed", rows, None)
    out = []
    for t, k, p, seed in zip(temps, top_k, top_p, seeds):
        k = int(k) if k is not None else 0
        out.append(
            SamplingParams(
                temperature=float(t or 0.0),
                top_k=k if k > 0 else None,
                top_p=float(p) if p is not None and 0.0 < float(p) <= 1.0 else 1.0,
                seed=None if seed is None or int(seed) < 0 else int(seed),
            )
        )
    return out


def _device_seeded(params: Any) -> bool:
    """A device-sampled request whose stream depends on its explicit seed (greedy rows draw nothing)."""
    return params is not None and params.seed is not None and not params.is_greedy


# ---------------------------------------------------------------------------------------------------- adapter
class Glm5NextForCausalLM:
    """TT vLLM model class for zai-org/GLM-5.3-Flash (text only), delegating to `Glm53Generator`."""

    model_capabilities = {
        # vLLM automatic prefix caching shares the latent pages; the adapter's PrefixStore (tt/prefix_cache.py)
        # snapshots the KDA / DSA slot state at every 128-token chunk boundary and at the end of every generation and
        # attaches a hit to the longest cached boundary (GLM53_PREFIX_CACHE=0 declares it off)
        "supports_prefix_caching": PC.enabled(),
        "supports_chunked_prefill": True,
        # chunk pipelining (doc/ttnn/09-vllm/chunk_pipelining.md): a step whose rows are intermediate for their
        # prompts keeps device sampling; the plugin passes `sample_rows` and the intermediate rows run head-less
        # (no head, no sample, no RNG advance, no host wait) -- their placeholder (-1) is dropped by the plugin
        "supports_partial_prefill_sampling": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        # models/common/sampling TTSampling: top-k candidates per row (06a contract section 7); larger / unset top_k
        # requests take vLLM's host sampler (exact) instead of the top-32 approximation
        "max_device_top_k": 32,
        # penalties are not part of the generator's sampler contract (06a contract section 7): host sampler
        "supports_device_penalties": False,
        # the device sampler returns tokens only (06a contract section 7): logprob requests take the host sampler
        "supports_device_logprobs": False,
        # a stage-scope request limit from the context contract (current_supported_context below the served
        # max_model_len without a hard physical limit, e.g. the old dense-exact DSA bound); 0 = no limit beyond
        # max_model_len (the served 32K contract)
        "tt_request_context_limit": _contract_request_limit(),
        # the fabric keywords the serving backend module was validated with (its serving_fabric_config(): the TP8
        # stack's FABRIC_1D + default router config; the Stage 4 pipeline's pipeline_fabric_config()); not on the
        # launch line
        "fabric_config": _PipelineFabric(),
        # Stage 17: the model's own drafter drafts on device inside the decode step (method "mtp" = the MTP head,
        # method "dflash" = the Apache-2.0 DFlash2 block drafter kind, GLM53_SPEC_KIND=dflash2; or the model-owned
        # "custom_class"): verify = decode_forward(spec_mode="argmax_ids"), propose = propose_draft_tokens; the
        # hidden state never leaves the device; greedy only; the verify / propose split is synchronous
        # (--no-async-scheduling)
        "supports_spec_decode": SPEC_K > 0,
        # a prefill step holding intermediate chunks keeps device sampling: the runner passes sample_rows (False =
        # an intermediate chunk, its token discarded) and those rows are prefilled without a head or a sample, so
        # the other rows still sample (and batch) on device (mixed input lengths)
        "supports_partial_prefill_sampling": True,
        "spec_requirements": ("device_propose", "hidden_feed"),
        "spec_hidden_handoff": ("on_device",),
        # GLM53_SPEC_ASYNC: a verify still returns its argmax ids synchronously (the runner drains around it); the
        # overlap of the speculative steps is the adapter's own pre-enqueued step
        "supports_async_spec_decode": SPEC_ASYNC,
    }
    decode_input_update_contract = 1

    def __init__(
        self, generator: ServingGenerator, *, max_batch_size: int, max_seq_len: int, text_config=None, vllm_config=None
    ):
        # vLLM's model inspection may construct the class with `vllm_config`; serving goes through
        # initialize_vllm_model.
        if vllm_config is not None:
            raise RuntimeError("Initialize the TT model through initialize_vllm_model")
        self.generator = generator
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.text_config = text_config
        self.kv_cache = None
        self.cache_spec: ServingCacheSpec | None = None
        self._traces_captured = False  # the generator's serving traces exist for self.kv_cache
        # pack live requests into the lowest generator slots (fresh requests take the lowest free slot; survivors move
        # down before the next decode step) when the generator decodes pow2 bucket widths (`decode_buckets`, DECISIONS
        # 2026-10-17): the bucket covering the highest live slot is the decode width. With one width (a
        # full-width trace) packing saves nothing and a move would only copy KDA state.
        self.pack_slots = len(getattr(generator, "decode_buckets", None) or ()) > 1
        self._reset_slots()
        # adapter-side counters (the generator keeps the device-side ones: trace_report()); decode_submit_s = host wall
        # time of decode_forward (adapter + generator input handling + trace enqueue), the serving-side decode overhead;
        # compaction_s = the part of it spent in slot moves (backend copy_slot, waits for the previous replay)
        self.counters = {
            "prefill_calls": 0,
            "decode_calls": 0,
            "remaps": 0,
            "slot_releases": 0,
            "slot_swaps": 0,
            "compaction_moves": 0,
            "decode_submit_s_total": 0.0,
            "decode_submit_s_max": 0.0,
            "compaction_s_total": 0.0,
            "compaction_s_max": 0.0,
            "kda_exports": 0,
            "kda_imports": 0,
            "decode_steady_steps": 0,
            "decode_steady_wall_s_total": 0.0,
            "decode_probes": 0,
            "decode_probe_device_s_total": 0.0,
            "engine_gaps": 0,
            "engine_gap_s_max": 0.0,
            "gc_gen2": 0,
            "gc_pause_s_total": 0.0,
            "gc_pause_s_max": 0.0,
        }
        self._events_path = os.environ.get(EVENTS_ENV) or None
        self._event_q: queue.Queue | None = None
        self._event_thread: threading.Thread | None = None
        self._probe_every = max(0, int(os.environ.get(DECODE_PROBE_ENV, "0") or 0))
        self._prev_decode_t: float | None = None  # entry time of the previous decode call (None: not a steady pair)
        self._steady_wall: dict[int, deque] = {}  # live rows -> recent steady step intervals (s)
        self._probe_dev: dict[int, deque] = {}  # live rows -> probe device times (s): submit return -> device idle
        self._probe_total: dict[int, deque] = {}  # live rows -> probe submit + device (s), the drained step end to end
        self._last_exit_t: float | None = None  # end of the last prefill / decode call (engine gaps)
        self._in_call = False
        self._engine_tid: int | None = None  # the thread that calls the model (vLLM's engine core loop)
        self._stall_thread: threading.Thread | None = None
        self._last_kind: str | None = None
        self._gc_t0: float | None = None
        gc_ref = weakref.WeakMethod(self._gc_callback)

        def _gc_cb(phase, info, _ref=gc_ref):  # weak: a collected adapter leaves an inert callback
            m = _ref()
            if m is not None:
                m(phase, info)

        gc.callbacks.append(_gc_cb)
        self.startup = StartupTimeline()  # initialize_vllm_model replaces it with the one that saw the model load

    def _event(self, kind: str, wait: bool = False, **fields) -> None:
        """Queue one event row for the writer thread; `wait` = return after it (and every earlier row) is written."""
        if self._events_path is None:
            return
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "t": time.time(), "event": kind, **fields}
        if self._event_thread is None or not self._event_thread.is_alive():
            self._event_q = queue.Queue()
            self._event_thread = threading.Thread(target=self._event_writer, name="glm53-events", daemon=True)
            self._event_thread.start()
        self._event_q.put(row)
        if wait:
            self._event_q.join()

    def _event_writer(self) -> None:
        q = self._event_q
        while True:
            row = q.get()
            try:
                with open(self._events_path, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
            except Exception as exc:  # a full disk must not stop serving; the row is lost, the error logged
                logger.warning("GLM-5.3-Flash events: {} row not written: {}", row.get("event"), exc)
            finally:
                q.task_done()

    def _gc_callback(self, phase: str, info: dict) -> None:
        """Times gen-2 collections (every thread of the process stops for them)."""
        if info.get("generation") != 2:
            return
        if phase == "start":
            self._gc_t0 = time.perf_counter()
        elif self._gc_t0 is not None:
            dt, self._gc_t0 = time.perf_counter() - self._gc_t0, None
            self.counters["gc_gen2"] += 1
            self.counters["gc_pause_s_total"] += dt
            self.counters["gc_pause_s_max"] = max(self.counters["gc_pause_s_max"], dt)

    def _call_enter(self, kind: str) -> None:
        """A model call starts: the gap since the last one ended, while a request is live (engine-side stall)."""
        self._in_call = True
        if self._engine_tid is None:
            self._engine_tid = threading.get_ident()
            if self._events_path and os.environ.get("GLM53_STALL_SAMPLER", "1") == "1":
                self._stall_thread = threading.Thread(target=self._stall_sampler, name="glm53-stalls", daemon=True)
                self._stall_thread.start()
        if self._last_exit_t is None or not self.slot_table.slots:
            return
        gap = time.perf_counter() - self._last_exit_t
        self.counters["engine_gap_s_max"] = max(self.counters["engine_gap_s_max"], gap)
        if gap >= ENGINE_GAP_EVENT_S:
            self.counters["engine_gaps"] += 1
            self._event("engine_gap", gap_s=round(gap, 4), after=self._last_kind, before=kind,
                        live=len(self.slot_table.slots), gc_pause_s_total=round(self.counters["gc_pause_s_total"], 4))  # fmt: skip

    def _call_exit(self, kind: str) -> None:
        self._last_exit_t, self._last_kind = time.perf_counter(), kind
        self._in_call = False

    def _stall_sampler(self) -> None:
        """Daemon: while a request is live and no model call has run for STALL_SAMPLE_S, sample the engine thread's
        stack (up to STALL_SAMPLES per stall, STALL_SAMPLE_S apart) into an `engine_stall_stack` event."""
        import sys
        import traceback

        sampled_for, n = None, 0
        while True:
            time.sleep(0.1)
            t0 = self._last_exit_t
            if t0 is None or self._in_call or not self.slot_table.slots:
                continue
            idle = time.perf_counter() - t0
            if sampled_for != t0:
                sampled_for, n = t0, 0
            if n >= STALL_SAMPLES or idle < STALL_SAMPLE_S * (n + 1):
                continue
            frame = sys._current_frames().get(self._engine_tid)
            stack = [ln.strip() for ln in traceback.format_stack(frame)[-14:]] if frame is not None else []
            n += 1
            self._event("engine_stall_stack", idle_s=round(idle, 3), sample=n, after=self._last_kind,
                        live=len(self.slot_table.slots), stack=stack)  # fmt: skip

    def _light_report(self) -> dict:
        """The hot-path events' report: copies of the adapter / generator / backend counters, no device query (+ the
        prefix store's host-side report when it is on)."""
        light = getattr(self.generator, "counters_report", None)
        gen = light() if callable(light) else {"counters": dict(getattr(self.generator, "counters", None) or {})}
        rep = {"generator": gen, "adapter": dict(self.counters)}
        if self.prefix is not None:
            rep["prefix_cache"] = self.prefix.report()
        return rep

    def _device_sync(self) -> bool:
        """Block until the device finished every enqueued command (decode probes; the generator's device_sync);
        False = no device to wait on."""
        sync = getattr(self.generator, "device_sync", None)
        return bool(sync()) if callable(sync) else False

    def host_step_report(self) -> dict:
        """Per live-row count: steady step wall (p50 / p90), probe device time and the host overhead per step
        (steady wall p50 - probe device p50); plus the mean host submit time of decode_forward."""

        def pct(xs, q):
            if not xs:
                return None
            s = sorted(xs)
            return round(1e3 * s[min(len(s) - 1, int(q * len(s)))], 3)

        rows = {}
        for n in sorted(set(self._steady_wall) | set(self._probe_dev)):
            wall, dev, tot = self._steady_wall.get(n, ()), self._probe_dev.get(n, ()), self._probe_total.get(n, ())
            r = {"steady_steps": len(wall), "wall_ms_p50": pct(wall, 0.5), "wall_ms_p90": pct(wall, 0.9),
                 "probes": len(dev), "probe_device_ms_p50": pct(dev, 0.5), "probe_total_ms_p50": pct(tot, 0.5)}  # fmt: skip
            if r["wall_ms_p50"] is not None and r["probe_device_ms_p50"] is not None:
                r["host_overhead_ms_p50"] = round(r["wall_ms_p50"] - r["probe_device_ms_p50"], 3)
            rows[str(n)] = r
        calls = max(1, self.counters["decode_calls"] - self.counters["decode_probes"])  # probe submits are not counted
        return {"probe_every": self._probe_every, "submit_ms_mean": round(1e3 * self.counters["decode_submit_s_total"] / calls, 3),
                "by_live_rows": rows}  # fmt: skip

    def _reset_slots(self) -> None:
        self.gen_slot = list(range(self.max_batch_size))  # vLLM state slot -> generator slot (slot_remap indirection)
        self.slot_table = StateSlotTable(self.max_batch_size)  # keyed by generator slot
        self._resident: set[int] = set()  # generator slots that decoded in the last accepted step
        self._last_moves: dict[int, int] | None = None
        self._params: dict[int, Any] = {}  # generator slot -> sampling params last uploaded (device sampling)
        self._prefix_states: dict[int, Any] = {}  # vLLM slot -> KDA checkpoint of a connector hit (import_kda_state)
        self._pooled_pages = False  # GLM53_KV_POOLED_PAGES, enabled by the plugin (kv_pooled_pages_enable)
        self._spec = None  # Stage 17: {bonus, drafts, pos} of the speculating request (generator slot 0), or None
        # GLM53_SPEC_ASYNC: the pre-enqueued step {pos, block = [bonus] + drafts} whose record nobody read yet, and the
        # held tokens {base, inputs, outs} slot 0's device committed past the runner's position (see _spec_resolve)
        self._pend = None
        self._held = None
        self._spec_tl = None  # the lone step's (record wait start, record in hand) in the current verify call
        self._spec_t_exit = None  # when the last call, a lone speculative verify, returned (the runner's turn since)
        # the prefix store (built with the cache, allocate_kv_cache_per_layer) survives a slot reset: the snapshot
        # pool is not part of the slots; only the slots' attachments / trailing snapshots go
        if getattr(self, "prefix", None) is not None:
            for g in range(self.max_batch_size):
                self.prefix.release_slot(g)
            self.slot_table.shared_pages = True
        else:
            self.prefix: PC.PrefixStore | None = None
        self._slot_tokens: dict[int, list] = {}  # generator slot -> the prompt ids prefilled so far (hash chain input)

    # vLLM interface shims: the TT path never calls these.
    def embed_input_ids(self, input_ids):
        raise RuntimeError("GLM-5.3-Flash runs through prefill_forward / decode_forward")

    def forward(self, input_ids, positions, **kwargs):
        raise RuntimeError("GLM-5.3-Flash runs through prefill_forward / decode_forward")

    def compute_logits(self, hidden_states, **kwargs):
        raise RuntimeError("The TT generator owns the final norm and LM head")

    def modules(self):
        # vLLM's in-process LLMEngine cleanup walks model.modules() for torch.compile wrappers; the TT model
        # holds no torch modules.
        return iter(())

    # ------------------------------------------------------------------------------------------ construction
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=None,
        tt_data_parallel=1,
        optimizations=None,
        *,
        generator_factory=None,
        **kwargs,
    ):
        text = text_config_of(hf_config)
        expected = {"num_hidden_layers": 45, "hidden_size": 4096, "vocab_size": 154880, "kv_lora_rank": 512}
        found = {k: getattr(text, k, None) for k in expected}
        if found != expected:
            raise ValueError(f"not the zai-org/GLM-5.3-Flash text model: {found} != {expected}")
        layer_kinds(text)  # validates layer_types
        if tt_data_parallel != 1:
            raise ValueError(
                f"tt_data_parallel={tt_data_parallel}: GLM-5.3-Flash serves one pipeline over the whole mesh (DP 1)"
            )
        if not 1 <= max_batch_size <= MAX_STATE_SLOTS:
            raise ValueError(f"max_num_seqs {max_batch_size} outside [1, {MAX_STATE_SLOTS}] (KDA state slots)")
        contract = load_context_contract()
        max_seq_len = contract.served_max_model_len if max_seq_len is None else int(max_seq_len)
        if not 1 <= max_seq_len <= contract.advertised:
            raise ValueError(f"max_model_len {max_seq_len} outside [1, {contract.advertised}] (advertised context)")
        if contract.served is not None and max_seq_len > contract.served:
            raise ValueError(
                f"max_model_len {max_seq_len} above {CONTEXT_CONTRACT_PATH} vllm_serving.served_max_model_len "
                f"{contract.served} (hard physical limit: {contract.reduction.get('reduction_reason')})"
            )
        if optimizations not in (None, "performance"):
            raise ValueError(
                f"optimizations={optimizations!r}: GLM-5.3-Flash has one precision policy (the Stage 3/4/6 optimized "
                "decoder, tt/model.py); expected None or 'performance'"
            )
        startup = StartupTimeline()
        startup.mark("model_init")
        override = generator_factory or os.environ.get(GENERATOR_FACTORY_ENV)
        if override is None:
            _check_device_mesh(mesh_device)
        if generator_factory is not None or os.environ.get(GENERATOR_FACTORY_ENV):
            from models.autoports.zai_org_glm_5_3_flash.tt.generator import check_image_required

            check_image_required(False, "a generator factory override")
            factory = generator_factory or _resolve_factory(os.environ.get(GENERATOR_FACTORY_ENV))
            startup.info["generator_factory"] = getattr(factory, "__qualname__", str(factory))
        elif os.environ.get(WEIGHT_SOURCE_ENV):
            factory = _weight_source_factory(startup.info)
        else:
            factory = _resolve_factory(None)
            startup.info["generator_factory"] = "tt/generator.py:build_generator"
        # max_batch / max_seq_len are generator arguments; the factory forwards `decoder=` to tt/model.py. vLLM's
        # hf_config is not forwarded: tt/model.py reads the checkpoint's text config itself (tt/config).
        decoder = _serving_decoder(startup.info)
        build_kw = {"decoder": decoder} if decoder else {}
        # the stack / TP knob environment as launched (the Stage 14b build rewrites some per layer kind: set_knobs)
        launch_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_STACK_", "GLM53_TP_"))}
        with _phase_profile("load"):
            generator = factory(MODEL_DIR, mesh_device, max_batch=max_batch_size, max_seq_len=max_seq_len, **build_kw)
        startup.mark("model_built")
        if int(generator.max_batch) != max_batch_size:
            raise ValueError(f"generator built {generator.max_batch} slots, vLLM asked for {max_batch_size}")
        missing = [m for m in SERVING_API if not callable(getattr(generator, m, None))]
        if missing or "num_blocks" not in inspect.signature(generator.allocate_kv_cache).parameters:
            raise TypeError(
                f"{type(generator).__name__} lacks the serving API of 06a contract.md section 2.1 (missing {missing}, "
                "allocate_kv_cache(..., num_blocks=))"
            )
        if cls.model_capabilities["tt_request_context_limit"] != contract.request_context_limit:
            raise ValueError(
                f"class tt_request_context_limit {cls.model_capabilities['tt_request_context_limit']} != "
                f"{CONTEXT_CONTRACT_PATH} {contract.request_context_limit}: the module was imported against another "
                "contract"
            )
        limit = cls.model_capabilities["tt_request_context_limit"] or max_seq_len
        if int(generator.supported_context) < min(limit, max_seq_len):
            raise ValueError(
                f"the generator computes {generator.supported_context} tokens per request, but the class admits "
                f"{min(limit, max_seq_len)} (tt_request_context_limit {limit}, max_model_len {max_seq_len}): update "
                f"{CONTEXT_CONTRACT_PATH} current_supported_context to what the generator supports"
            )
        model = cls(generator, max_batch_size=max_batch_size, max_seq_len=max_seq_len, text_config=text)
        # a backend whose device sampler serves fewer candidates (its `max_device_top_k`, e.g. the Stage 14b stack built
        # without device sampling: 1) narrows the plugin's device-sampling route for this instance: sampled requests
        # then take vLLM's host sampler over the read-back logits instead of reaching a device sampler that cannot
        # serve them
        top_k = getattr(getattr(generator, "backend", None), "max_device_top_k", None)
        if top_k is not None and int(top_k) < int(cls.model_capabilities["max_device_top_k"]):
            model.model_capabilities = {**cls.model_capabilities, "max_device_top_k": int(top_k)}
        startup.info.update(max_num_seqs=max_batch_size, max_model_len=max_seq_len)
        # the serving profile as built: the backend module's extra keywords and the backend's row chunk (Stage 14b
        # stack: 32 = the native 32-row step, 4 = four-row chunks)
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import BACKEND_KW_ENV

        backend = getattr(generator, "backend", None)
        startup.info.update(backend=type(backend).__name__, backend_kw=os.environ.get(BACKEND_KW_ENV) or None,
                            backend_chunk=getattr(backend, "chunk", None),
                            stack_env=launch_env, weight_source=getattr(backend, "weight_source_info", None))  # fmt: skip
        model.startup = startup
        return model

    @classmethod
    def get_max_tokens_all_users(
        cls, *, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """Token capacity of the shared latent page pool (the plugin adds one page per user on top)."""
        max_model_len = int(max_model_len or load_context_contract().served_max_model_len)
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import blocks_per_seq

        users = int(max_num_seqs or 1)
        override = os.environ.get(KV_POOL_TOKENS_ENV)
        # floor: the generator's page-table width (blocks_per_seq pages: one max_model_len request plus the padded tail
        # of its last prefill chunk), so a small max_num_seqs still allocates (e.g. 1 seq at 2051: 40 pages, not 34)
        floor = blocks_per_seq(max_model_len) * BLOCK_SIZE
        pool = int(override) if override else max(floor, users * min(max_model_len, DEFAULT_TOKENS_PER_USER))
        if pool < max_model_len:
            raise ValueError(
                f"{KV_POOL_TOKENS_ENV}={pool} cannot hold one max_model_len={max_model_len} request; vLLM refuses "
                "to start (raise the pool, or serve a smaller max_model_len only with hard-limit evidence)"
            )
        return pool

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One MLAAttentionSpec per MLA/DSA layer (bf16 latent, 1024 B/token/layer); KDA layers have no pages."""
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        cache_config = vllm_config.cache_config
        # the adapter's PrefixStore follows vLLM's switch: without APC no page is ever cached, so snapshots would be
        # pure overhead (allocate_kv_cache_per_layer -> _prefix_init reads this)
        cls._vllm_prefix_caching = bool(cache_config.enable_prefix_caching)
        if cache_config.enable_prefix_caching and not PC.enabled():
            raise ValueError(
                "GLM-5.3-Flash serves vLLM prefix caching only with its slot-state snapshots (tt/prefix_cache.py, "
                f"{PC.ENABLED_ENV}=1): the KDA state is not paged. Launch with --no-enable-prefix-caching or enable them"
            )
        if cache_config.block_size != BLOCK_SIZE:
            raise ValueError(f"GLM-5.3-Flash latent pages are {BLOCK_SIZE} tokens: pass --block-size {BLOCK_SIZE}")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise ValueError(f"latent cache dtype is bf16, got --kv-cache-dtype {cache_config.cache_dtype}")
        text = text_config_of(vllm_config.model_config.hf_config)
        latent = int(text.kv_lora_rank) + int(getattr(text, "qk_rope_head_dim", 0) or 0)
        return {
            f"model.layers.{i}.self_attn": MLAAttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=latent, dtype=LATENT_DTYPE
            )
            for i, kind in enumerate(layer_kinds(text))
            if kind == "mla"
        }

    # ---------------------------------------------------------------------------------------------- KV cache
    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """`per_layer_specs`: [(shape, dtype, tensor_idx)] in vLLM attention-layer order -- one per decoder layer
        (45) when vLLM counts every layer as attention, or one per MLA layer (11). Only the MLA entries carry
        pages; their shape must be [num_blocks, 1, 64, 512] bf16."""
        kinds = layer_kinds(self.text_config)
        mla = [i for i, k in enumerate(kinds) if k == "mla"]
        if len(per_layer_specs) == len(kinds):
            entries = [per_layer_specs[i] for i in mla]
        elif len(per_layer_specs) == len(mla):
            entries = list(per_layer_specs)
        else:
            raise ValueError(
                f"{len(per_layer_specs)} per-layer KV specs; expected {len(kinds)} (all layers) or {len(mla)} (MLA)"
            )
        latent = int(self.text_config.kv_lora_rank) + int(getattr(self.text_config, "qk_rope_head_dim", 0) or 0)
        num_blocks = None
        for layer, (shape, dtype, _tensor_idx) in zip(mla, entries):
            shape = tuple(int(s) for s in shape)
            if shape[1:] != (1, BLOCK_SIZE, latent) or dtype != LATENT_DTYPE:
                raise ValueError(
                    f"layer {layer}: KV spec {shape} {dtype}, expected [N, 1, {BLOCK_SIZE}, {latent}] bf16"
                )
            if num_blocks not in (None, shape[0]):
                raise ValueError(f"layer {layer}: {shape[0]} blocks, other MLA layers have {num_blocks}")
            num_blocks = shape[0]
        self.cache_spec = build_serving_cache_spec(
            self.text_config, num_blocks=num_blocks, block_size=BLOCK_SIZE, max_slots=self.max_batch_size
        )
        # vLLM owns the page pool: the generator sizes its latent (+ pooled-key) pages from vLLM's num_blocks; block 0
        # is vLLM's null block (BlockPool.null_block, never a request's page; vllm_page_rows): the tail of every row,
        # so the TP prefill's pad pages land there instead of on a request's last page
        with _phase_profile("kv_cache"):
            self.kv_cache = self.generator.allocate_kv_cache(
                self.max_batch_size, self.max_seq_len, num_blocks=num_blocks, null_block=0
            )
        self._traces_captured = False
        self._reset_slots()
        self.startup.mark("kv_cache_allocated")
        self.startup.info["num_blocks"] = num_blocks
        # the KV phase's breakdown (StackBackend.allocate_kv_cache: attach_dsa, stacks, latent pool, DSA stores, pool)
        self.startup.info["kv_cache_sub"] = getattr(getattr(self.generator, "backend", None), "startup_sub", None)
        self.startup.info["context_by_users"] = context_by_users(
            num_blocks, self.max_seq_len, self.model_capabilities["tt_request_context_limit"], self.max_batch_size
        )
        self._prefix_init()
        return self.kv_cache

    # ------------------------------------------------------------------------------------------ prefix cache
    def _prefix_init(self) -> None:
        """The PrefixStore over the backend's snapshot pool (doc/ttnn/09-vllm/srv/prefixcache/README.md): keys are
        salted with the serving signature (model, image pin, code sha, knob set, mesh), so a changed image / code /
        knob set never serves an older snapshot. Registers the scheduler-side oracle (vllm_tt_plugin.prefix_oracle:
        the TT scheduler truncates vLLM's page hit to the longest boundary with a snapshot)."""
        self.prefix = None
        # serving always queries get_kv_cache_spec first; an adapter built without it (unit tests) keeps the strict
        # single-owner page checks
        if not PC.enabled() or not getattr(type(self), "_vllm_prefix_caching", False):
            return
        backend = self.generator.backend
        entries = int(getattr(backend, "snapshot_entries", 0))
        host_bytes = int(os.environ.get(PC.HOST_BYTES_ENV, "0"))
        if entries == 0 and host_bytes == 0:
            logger.info("prefix cache: the backend has no snapshot pool and no host tier is configured: off")
            return
        kinds = layer_kinds(self.text_config)
        tp = int(getattr(backend, "tp", 1))
        bytes_model = PC.SnapshotBytes(
            kda_layers=sum(k != "mla" for k in kinds), dsa_layers=sum(k == "mla" for k in kinds), tp=tp
        )
        sig = PC.signature(model=os.environ.get("GLM53_MODEL_NAME", getattr(self.text_config, "_name_or_path", "")),
                           image=os.environ.get("GLM53_IMAGE_PIN", ""),
                           code=os.environ.get("GLM53_CODE_SHA", ""), knobs=os.environ.get("GLM53_STACK_KNOB_SET", ""),
                           tp=tp, backend=type(backend).__name__, boundary=PC.BOUNDARY)  # fmt: skip
        self.prefix = PC.PrefixStore(max_batch=self.max_batch_size, device_entries=entries, host_bytes=host_bytes,
                                     sig=sig, entry_bytes=bytes_model.entry_per_chip(self.max_seq_len,
                                                                                    self.max_seq_len > 2051))  # fmt: skip
        if host_bytes > 0:
            self.prefix.spill_fn = self._prefix_spill_export  # Stage 12: LRU device entries go to the host tier
            self._ensure_dsa_export()  # sparse regime: the spill blob carries the DSA rows (staging before capture)
        self.slot_table.shared_pages = True  # cached prefixes: live requests may share their first page
        self.generator.shared_pages = True  # and alias the prefix's pages in their page-table rows
        self.startup.info["prefix_cache"] = dict(device_entries=entries, host_bytes=host_bytes, sig=sig,
                                                 entry_bytes_per_chip=self.prefix.entry_bytes,
                                                 kda_bytes_total=bytes_model.kda_total)  # fmt: skip
        try:
            from vllm_tt_plugin import prefix_oracle

            prefix_oracle.register(self)
        except ImportError:
            pass
        logger.info("prefix cache: {} device entries ({} bytes each per chip), host tier {} bytes, signature {}",
                    entries, self.prefix.entry_bytes, host_bytes, sig)  # fmt: skip

    def _prefix_spill_export(self, ent) -> Any:
        """PrefixStore spill hook (Stage 12 host KV tier): the LRU device entry's state -> a host blob, the backend's
        export_snapshot of the entry's virtual slot (what the host-attach path imports, like a connector KDA
        checkpoint). In the sparse regime the entry also holds the DSA pooled keys + ring rows: the blob carries them
        (StackBackend.export_dsa_state, the entry's first ceil(n_tokens / 128) units; host_kv_tier.md 5b) and its host
        bytes are the blob's; a blob without them is refused (None: the entry is dropped as before)."""
        if ent.device_entry is None:
            return None
        sparse = self._dsa_state()
        t0 = time.perf_counter()
        blob = self.generator.export_snapshot(self.kv_cache, int(ent.device_entry), n_tokens=int(ent.n_tokens))
        if sparse and not _states_dsa(blob):
            self.counters["prefix_spill_no_dsa"] = self.counters.get("prefix_spill_no_dsa", 0) + 1
            return None
        dt = time.perf_counter() - t0
        nbytes = _blob_bytes(blob) if sparse else self.prefix.entry_bytes
        c = self.counters
        c["prefix_spills"] = c.get("prefix_spills", 0) + 1
        c["prefix_spill_s"] = round(c.get("prefix_spill_s", 0.0) + dt, 6)  # T1 spill timing (host wall, blocking reads)
        c["prefix_spill_last_s"] = round(dt, 6)
        c["prefix_spill_bytes"] = c.get("prefix_spill_bytes", 0) + int(nbytes)
        return blob, nbytes

    def _ensure_dsa_export(self) -> None:
        """the backend's checkpoint staging (StackBackend.ensure_state_export: the KDA pack + the DSA rows), allocated
        only when a consumer exists (a host tier or a KV connector), before the first capture"""
        backend = self.generator.backend
        ensure = getattr(backend, "ensure_state_export", None) or (
            getattr(backend, "ensure_dsa_export", None) if self._dsa_state() else None)  # fmt: skip
        if callable(ensure) and self.kv_cache is not None:
            ensure(self.kv_cache)

    def _dsa_state(self) -> bool:
        """the serving cache keeps per-slot DSA indexer state (sparse regime): every prefix checkpoint must carry it"""
        if getattr(self, "_dsa_state_v", False):
            return True  # kept once seen
        if self.kv_cache is None:
            return False  # the answer comes with the cache
        has = getattr(self.generator.backend, "has_dsa_state", None)
        # not cached while False: a prefill-only DSA store appears at prepare_prefill
        self._dsa_state_v = bool(callable(has) and has(self.kv_cache))
        return self._dsa_state_v

    # the scheduler oracle (vllm_tt_plugin.prefix_oracle.PrefixOracle)
    boundary = PC.BOUNDARY

    def longest_cached(self, token_ids, max_tokens: int | None = None) -> int:
        """The longest cached boundary <= max_tokens of a prompt (0: none): what vLLM's page hit is truncated to."""
        if self.prefix is None:
            return 0
        return self.prefix.longest_cached(list(token_ids), max_tokens)[0]

    def note_finished(self, token_ids) -> bool:
        """Scheduler-side hint (prefix_oracle.note_finished, the TT scheduler's _free_request): a request with these
        prompt + output ids just finished. Its trailing generation snapshot is keyed NOW, one step before the runner's
        release_request reaches the adapter, so the next chat turn (scheduled before that release) hits it. The slot
        is found by its prefilled history (a prefix of the ids); an ambiguous match (identical prompts live in two
        slots) waits for release_request, which names the slot. Returns True when a snapshot was committed."""
        if self.prefix is None:
            return False
        ids = [int(t) for t in token_ids]
        cands = [g for g, (ent, pos) in self.prefix.trailing.items()
                 if pos <= len(ids) and g in self._slot_tokens and len(self._slot_tokens[g]) <= len(ids)
                 and ids[: len(self._slot_tokens[g])] == self._slot_tokens[g]]  # fmt: skip
        if len(cands) != 1:
            return False
        g = cands[0]
        b = self.generator._binding(self.kv_cache)
        pages = [int(p) for p in b.pt_shadow[g].tolist()]
        ent = self.prefix.commit_trailing(g, ids, pages)
        if ent is not None:
            self.counters["prefix_generation_entries"] = self.counters.get("prefix_generation_entries", 0) + 1
        return ent is not None

    def _prefix_plan(self, i: int, tokens, s: int, e: int, fresh: bool, connector_hit: bool) -> tuple:
        """One prefill row's prefix-cache decisions -> (start, attach entry, host blob, snapshot entry, snapshot key).
        A fresh request with s > 0 (vLLM's page hit) attaches to the longest cached boundary <= s: the device entry
        (state read inside the prefill trace) or a host blob (import); a shorter boundary restarts the row there
        (its pages are rewritten with identical content); no boundary restarts at 0. A row whose end is a boundary
        snapshots its state (a free pool entry, LRU-evicting) unless that key exists."""
        attach = blob = snap = key_e = None
        ids = tokens[i, :e].tolist()
        if fresh and s > 0 and not connector_hit:
            pos, ent = self.prefix.longest_cached(ids, s)
            if pos != s:
                self.counters["prefix_restarts"] = self.counters.get("prefix_restarts", 0) + 1
                logger.warning(
                    "prefill row {}: vLLM hit {} tokens, the longest snapshot is at {}: restarting there", i, s, pos
                )
                s = pos
            if s > 0:
                if ent.tier == "device":
                    attach = ent
                else:
                    blob = ent
        if e % self.prefix.boundary == 0 and e > 0:
            key_e = self.prefix.key(ids, e)
            if not self.prefix.has(key_e):
                snap = self.prefix.reserve_device()
                if snap is None:
                    self.counters["prefix_snapshot_skipped"] = self.counters.get("prefix_snapshot_skipped", 0) + 1
        return s, attach, blob, snap, key_e

    def kv_cache_connector_views(self) -> dict:
        """vllm-tt-plugin KV-connector hook (LMCache A1, `resolve_connector_kv_views`): {the vLLM layer name of
        `get_kv_cache_spec`, `model.layers.{i}.self_attn`: MLA/DSA layer i's paged latent tensor [num_blocks, 1, 64, 512]
        bf16 (page b = vLLM block b)}, from the backend (ModelBackend.latent_pages). Pages only: a prefix hit also
        needs the slot's KDA state after the prefix (doc/ttnn/15-lmcache/page_contract.md section 5); without it the
        hit's continuation is refused by StateSlotTable.check_prefill (no slot holds the request's state)."""
        if self.kv_cache is None:
            raise RuntimeError("kv_cache_connector_views before allocate_kv_cache_per_layer")
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import ModelBackend

        backend = type(self.generator.backend)
        if any(getattr(backend, m) is getattr(ModelBackend, m) for m in ("export_kda_state", "import_kda_state")):
            # a hit without the prefix's KDA state would continue a recurrence that never saw the prefix
            raise NotImplementedError(
                f"{backend.__name__} has no KDA checkpoint (export_kda_state / import_kda_state): no KV connector"
            )
        self._ensure_dsa_export()  # sparse regime: connector checkpoints carry the DSA rows (staging before capture)
        pages = self.generator.latent_pages(self.kv_cache)  # GLM53_DSA_SHARD: local pools, see kv_cache_connector_shard
        mla = [i for i, k in enumerate(layer_kinds(self.text_config)) if k == "mla"]
        if sorted(pages) != mla:
            raise ValueError(f"the backend exposes latent pages of layers {sorted(pages)}; vLLM's KV spec has {mla}")
        # GLM53_KV_T2=1 (program 2026-10-23T02:44:58Z; GLM53_LMCACHE_PAGE_DMA=1 is the same switch): LMCache hits page
        # in through the page-DMA (tt/page_pool.py LatentHitDma); default off until its mesh row
        t2 = "1" in (os.environ.get("GLM53_KV_T2", "0"), os.environ.get("GLM53_LMCACHE_PAGE_DMA", "0"))
        if t2 and getattr(self, "_hit_dma", None) is None:
            # the hit page-in's persistent kernel starts here: the connector views are resolved before warm-up, so the
            # launch (a program) and the pinned pool precede every capture
            from models.autoports.zai_org_glm_5_3_flash.tt.page_pool import LatentHitDma

            # GLM53_KDA_CKPT_DMA=1 (default with T2; Stage 12, program 03:54Z): the prompt-end KDA checkpoint is a
            # device snapshot the same page-DMA saves while the next steps run (export_kda_state_async)
            snap = []
            hook = getattr(self.generator.backend, "state_snapshot_tensors", None)
            if os.environ.get("GLM53_KDA_CKPT_DMA", "1") == "1" and callable(hook):
                snap = hook(self.kv_cache)
            # GLM53_KV_POOLED_PAGES=1 (Stage 12, ruling 2026-10-23T09:54:56Z): the DSA pooled-key stores are page-DMA
            # targets too -- the LMCache pages carry each 64-token page's pooled keys (paged_pooled_keys.md), so the
            # KDA checkpoints drop their pooled-key rows (kv_pooled_pages_enable, by the plugin)
            pools = []
            if os.environ.get("GLM53_KV_POOLED_PAGES", "0") == "1":
                hook = getattr(self.generator.backend, "pooled_store_tensors", None)
                pools = hook(self.kv_cache) if callable(hook) else []
                if len(pools) != len(mla):
                    logger.warning(f"GLM53_KV_POOLED_PAGES: {len(pools)} pooled-key stores for {len(mla)} DSA layers: "
                                   "off (the checkpoints keep their pooled-key rows)")  # fmt: skip
                    pools = []
            self._hit_dma = LatentHitDma(getattr(self.generator.backend, "mesh", None), [pages[i] for i in mla],
                                         gib=int(os.environ.get("GLM53_LMCACHE_PAGE_DMA_GIB", "1")),
                                         cores=int(os.environ.get("GLM53_LMCACHE_PAGE_DMA_CORES", "4")),
                                         state_tensors=snap, pool_tensors=pools,
                                         # GLM53_LMCACHE_PAGE_DMA_RING (Stage 12 lever (c); 2,048 descriptors per
                                         # service core = 64 KiB of its L1): past cores x ring a hit's push blocks on
                                         # ring space and the chips are fed one after another (d0 pair, a 32K hit per
                                         # chip: ring 256 -> 95 ms submit, 1,024 -> 55, 2,048 -> 24 + an async DMA)
                                         ring=int(os.environ.get("GLM53_LMCACHE_PAGE_DMA_RING", "256")),  # the k line keeps 256 by default
                                         # GLM53_LMCACHE_POOL_CPU_GB (lever (c) step 2, default 0 = off): LMCache's CPU
                                         # tier lives in the pool's top N GiB (the shared pool; the fork's external
                                         # buffer hook, page_pool.lmcache_external_buffer) -> zero-copy hits
                                         lmcache_gib=int(os.environ.get("GLM53_LMCACHE_POOL_CPU_GB", "0")),
                                         shared=True)  # fmt: skip
            self._hit_dma.start()
            self._hit_dma.layer_names = [f"model.layers.{i}.self_attn" for i in mla]
            logger.info(f"LMCache hit page-in over the page-DMA kernel: {len(mla)} latent caches, pool "
                        f"{self._hit_dma.pool.gib} GiB ({self._hit_dma.pool.backing}), {self._hit_dma.slots} staging pages; "
                        f"KDA checkpoint snapshots {'over the page-DMA (' + str(len(snap)) + ' buffers, ' + str(self._hit_dma.chip_state_bytes) + ' B per chip)' if snap else 'synchronous'}")  # fmt: skip
        return {f"model.layers.{i}.self_attn": pages[i] for i in mla}

    def prefill_plan_error(self, start_pos: int, n: int):
        """vllm-tt-plugin hook (Stage 12, program 2026-10-26T17:09:24Z): the generator's host-only prefill plan
        check (Glm53Generator.prefill_plan_error) -- the scheduler refuses a request whose prefill cannot plan"""
        fn = getattr(self.generator, "prefill_plan_error", None)
        return fn(int(start_pos), int(n)) if callable(fn) else None

    def kv_checkpoints_resumable(self) -> bool:
        """vllm-tt-plugin hook (Stage 12, n29h7p2 2026-10-23): False when no KDA checkpoint of this cache can ever resume
        a connector hit -- a paged pooled-key store (GLM53_DSA_PAGED_POOLS) without the pooled pages over the page-DMA
        (GLM53_KV_T2=1 + GLM53_KV_POOLED_PAGES=1), where _hit_restart refuses every hit. The plugin then exports no
        checkpoint, so the scheduler's lookup aligns every hit to 0: no retrieve that is thrown away (2-19 s per hit on
        1843), no prompt-end export on the next step. Ask after kv_pooled_pages_enable (the plugin's init)."""
        if self.kv_cache is None or not self._dsa_state():
            return True
        paged = getattr(self.generator.backend, "pooled_paged", lambda kv: False)(self.kv_cache)
        return not paged or (bool(self._pooled_pages) and getattr(self, "_hit_dma", None) is not None)

    def kv_pooled_pages_enable(self) -> bool:
        """vllm-tt-plugin hook (GLM53_KV_POOLED_PAGES): the plugin carries every page's pooled keys in the LMCache page
        (store: kv_read_pooled_pages; load: the hit's checkpoint blob["pooled"]), so from now on the KDA checkpoints
        omit the pooled-key rows (O(n) per checkpoint). False: the page-DMA has no pooled-key stores (the checkpoints
        keep them)."""
        dma = getattr(self, "_hit_dma", None)
        if dma is None or not getattr(dma, "pool_ids", None):
            return False
        self._pooled_pages = True
        return True

    def kv_read_pooled_pages(self, pairs, pages=None) -> list:
        """vllm-tt-plugin hook (GLM53_KV_POOLED_PAGES, the LMCache store): pairs [(vLLM slot, column = position // 64)]
        -> per DSA layer (the page-DMA's layer order) [n, 4,096] uint8: that page's pooled keys, device-native (rows
        16 (j % 2) .. + 16 of the slot's unit j // 2 page; page_pool.pooled_halves), read by the page-DMA from each
        unit's owner chip. GLM53_DSA_PAGED_POOLS: pages [n] = the pages' LOGICAL blocks -> their physical blocks' own
        store pages (rows 0-15). Call after the steps that wrote them finished."""
        targets, finish = _pooled_store_plan(self, pairs, pages)
        return finish(self._hit_dma.read_pooled(targets))

    def kv_pooled_store_plan(self, pairs, pages=None):
        """vllm-tt-plugin hook (Stage 12 store path P2): kv_read_pooled_pages split in two -> (targets, finish).
        targets [(chip, page)] = the pooled-key unit (or GLM53_DSA_PAGED_POOLS block) pages to read; finish(data) maps
        the page-DMA's 11 arrays [m, 8,192] of them to kv_read_pooled_pages' result. The plugin hands targets to
        LatentHitDma.save_resident_pooled, which reads them in the same device round trip as the store's latent
        pages."""
        return _pooled_store_plan(self, pairs, pages)

    def _load_prefix_pooled(self, rows: dict) -> None:
        """GLM53_KV_POOLED_PAGES: the hit rows' pooled keys (blob["pooled"] = {"units": U, "pages": the hit's pages, "data":
        per layer [U, 8,192] uint8 device-native unit pages, units 0..U-1}) -> their generator slots' units through the
        page-DMA, landed
        before the prefill that continues from them is enqueued. rows {i: (generator slot, pooled, logical block row)};
        GLM53_DSA_PAGED_POOLS: each 64-token half goes to its block's own page (the hit's blocks are admitted by the
        connector's load), padded (page_pool.block_page_of_half), built in the page-DMA ring (load_pooled_blocks)."""
        import numpy as np  # noqa: PLC0415

        t0 = time.perf_counter()
        dma, backend = self._hit_dma, self.generator.backend
        paged = getattr(backend, "pooled_paged", lambda kv: False)(self.kv_cache)
        obm = self.generator.dsa_shard_map(self.kv_cache) if paged else None
        targets, data, prows = [], [], []
        for _i, (gs, p, row) in sorted(rows.items()):
            u = int(p["units"])
            arrs = [np.asarray(a).view(np.uint8).reshape(u, -1) for a in p["data"]]
            if paged:  # the hit's pages only (blob "pages"; a unit's padding half past the hit has no block)
                blocks = [int(b) for b in list(row)[: int(p.get("pages", 2 * u))]]
                phys = [obm.physical(b) if hasattr(obm, "physical") else obm.phys.get(b) for b in blocks]
                if any(g is None for g in phys):
                    raise ValueError(f"row {_i}: hit blocks without a physical block (not admitted): {blocks}")
                targets += backend.pooled_block_pages(self.kv_cache, phys)
                prows.append((arrs, len(blocks)))  # padded block pages, built in the ring (page_pool.load_pooled_blocks)
            else:
                targets += backend.pooled_unit_pages(self.kv_cache, gs, range(u))
                data.append(arrs)
        t1 = time.perf_counter()
        st0 = dict(getattr(dma, "stats", None) or {})
        if targets:
            if paged:
                dma.load_pooled_blocks(prows, targets)
            else:
                dma.load_pooled(data[0] if len(data) == 1 else [np.concatenate(c) for c in zip(*data)], targets)
        t2 = time.perf_counter()
        if targets:
            dma.wait()
        t3 = time.perf_counter()
        # Stage 12 (program 2026-10-26T17:50:09Z WARM): the hit's pooled-unit load was untimed -- map (block ids ->
        # owner pages), push (load_pooled*: prewait = a previous page-in still queued, stage = the ring staging, the
        # descriptor push), wait (the device landing them, before the tail prefill is enqueued); scalar counters
        st1 = getattr(dma, "stats", None) or {}
        ph = dict(map=t1 - t0, push=t2 - t1, wait=t3 - t2,
                  **{k: float(st1.get(f"pooled_load_{k}_s", 0.0)) - float(st0.get(f"pooled_load_{k}_s", 0.0))
                     for k in ("prewait", "stage")})  # fmt: skip
        c = self.counters
        c["pooled_unit_loads"] = c.get("pooled_unit_loads", 0) + len(targets)
        c["pooled_loads"] = c.get("pooled_loads", 0) + 1
        c["pooled_load_s_total"] = round(c.get("pooled_load_s_total", 0.0) + t3 - t0, 6)
        c["pooled_load_s_max"] = round(max(c.get("pooled_load_s_max", 0.0), t3 - t0), 6)
        for k, v in ph.items():
            c[f"pooled_load_{k}_s_total"] = round(c.get(f"pooled_load_{k}_s_total", 0.0) + v, 6)
        if getattr(self, "_events_path", None):
            self._event("pooled_load", units=len(targets), rows=len(rows), s=round(t3 - t0, 6),
                        **{f"{k}_s": round(v, 6) for k, v in ph.items()})  # fmt: skip

    def latent_hit_dma(self):
        """vllm-tt-plugin hook (GLM53_LMCACHE_PAGE_DMA=1): the LatentHitDma the connector's h2d submits hit pages to
        (tt/page_pool.py; layer_names = the connector view order), None when off. Valid after kv_cache_connector_views."""
        return getattr(self, "_hit_dma", None)

    def kv_cache_connector_shard(self) -> Any:
        """vllm-tt-plugin KV-connector hook (Stage 12 x 14: LMCache pages for the sequence-sharded DSA pool,
        GLM53_DSA_SHARD; shard_tp8_design.md 6.3): None on the replicated pool. Else {"C", "nb_local", "bpu",
        "num_blocks", "map"}: the tensors of kv_cache_connector_views are then every chip's LOCAL pool [nb_local, 1,
        64, 512] indexed by LOCAL block, and the plugin (kv_connector.TTShardedPagedCacheHandle) translates vLLM's
        logical block b through map (tt/owner_blocks.py OwnerBlockMap: physical(b) for a STORE, admit([(b, column)])
        for a LOAD) to (owner chip g % C, local block g // C): a page is read from and written to its owner only."""
        if self.kv_cache is None:
            raise RuntimeError("kv_cache_connector_shard before allocate_kv_cache_per_layer")
        return self.generator.connector_shard_map(self.kv_cache)

    @property
    def kda_checkpoint_in_replay(self) -> bool:
        """one-replay store A: the backend writes a storing row's KDA state at its checkpoint cut inside the prefill
        replay (prefill_forward(kda_snapshot_at=)), so the plugin drops the tail split for storing requests and exports
        at the cut with export_kda_state(_async)(at=cut)"""
        return bool(getattr(self.generator.backend, "kda_checkpoint_in_replay", False))

    def _checkpoint_at(self, slot: int, at: Optional[int]) -> tuple:
        """(generator slot, checkpoint position, state source) of vLLM slot `slot`'s checkpoint at `at`: None (or the
        slot's own position) = its live state, which must sit on a page boundary; a page boundary BEFORE its position =
        the in-replay snapshot of the prefill that covered it (one-replay store: the row carried kda_snapshot_at)"""
        g = self.gen_slot[int(slot)]
        st = self.slot_table.slots.get(g)
        if st is None:
            raise ValueError(f"vLLM slot {slot}: no live request to checkpoint")
        if at is None or int(at) == st.next_pos:
            from models.autoports.zai_org_glm_5_3_flash.tt import generator as _gen  # noqa: PLC0415

            # GLM53_RESUME_AS_DECODE: a live checkpoint OFF the page grid (a request preempted mid-decode, Stage 12's
            # exact save) is resumable: its one-token resume decodes, at any position (prefill_forward)
            if st.next_pos % BLOCK_SIZE and not _gen.RESUME_AS_DECODE:
                raise ValueError(f"vLLM slot {slot}: no live request at a page boundary to checkpoint ({st})")
            return g, st.next_pos, "live"
        at = int(at)
        if at <= 0 or at % BLOCK_SIZE or at > st.next_pos:
            raise ValueError(f"vLLM slot {slot}: a checkpoint at {at} needs a page boundary <= {st.next_pos} tokens")
        return g, at, "snapshot"

    def kda_snapshot_ready(self, kv_cache, slot: int, at: int) -> bool:
        """True when the backend's in-replay snapshot holds vLLM slot `slot`'s KDA state at `at` tokens (the replay that
        covered the cut carried it and no export has consumed it yet): export_kda_state(at=) will succeed"""
        self._check_cache(kv_cache)
        fn = getattr(self.generator.backend, "kda_snapshot_holds", None)
        return bool(fn is not None and fn(kv_cache, self.gen_slot[int(slot)], int(at)))

    def export_kda_state(self, kv_cache, slot: int, at: Optional[int] = None) -> Any:
        """vllm-tt-plugin KDA checkpoint policy (LMCache join `KDACheckpointStore`): vLLM state slot `slot`'s KDA
        recurrent state after its prefill chunk that ended on a page boundary, as {"next_pos": tokens folded in,
        "states": the backend's export_kda_state host tensors}; the plugin saves it keyed by the prefix's token ids,
        import_kda_state + prefill_forward check `next_pos` against the hit. at (one-replay store): the cut, a page
        boundary before the slot's position, exported from the in-replay snapshot (_checkpoint_at); next_pos = at."""
        self._check_cache(kv_cache)
        g, pos, source = self._checkpoint_at(slot, at)
        states = self.generator.export_slot_state(kv_cache, g, n_tokens=pos, pools=not self._pooled_pages,
                                                  source=source)  # fmt: skip
        self.counters["kda_exports"] += 1
        if source != "live":
            self.counters["kda_exports_snapshot"] = self.counters.get("kda_exports_snapshot", 0) + 1
        return {"next_pos": pos, "states": states}

    def export_kda_state_async(self, kv_cache, slot: int, at: Optional[int] = None) -> Any:
        """export_kda_state off the first-token path (Stage 12, GLM53_KV_T2=1 + GLM53_KDA_CKPT_DMA=1): the slot's state
        is snapshotted on device now and saved by the page-DMA while the next steps run. Returns a zero-argument callable
        -> export_kda_state's value (callable from any thread: no device access), or None (no page-DMA, or the
        checkpoint does not fit the snapshot: the plugin calls export_kda_state instead). at: as export_kda_state."""
        self._check_cache(kv_cache)
        dma = getattr(self, "_hit_dma", None)
        if dma is None or not getattr(dma, "state", None):
            return None
        g, pos, source = self._checkpoint_at(slot, at)
        fn = self.generator.export_slot_state_async(kv_cache, g, n_tokens=pos, dma=dma, pools=not self._pooled_pages,
                                                    source=source)  # fmt: skip
        if fn is None:
            return None
        self.counters["kda_exports_async"] = self.counters.get("kda_exports_async", 0) + 1
        if source != "live":
            self.counters["kda_exports_snapshot"] = self.counters.get("kda_exports_snapshot", 0) + 1
        out = lambda: {"next_pos": pos, "states": fn()}  # noqa: E731
        # Stage 12 lever 1 (b): the backend's handle on the device snapshot (stack_model.KDASnapshotHandle): a hit at
        # this checkpoint while the snapshot buffers still hold it imports on device (kda_snapshot_states)
        out.kda_snapshot = getattr(fn, "kda_snapshot", None)
        out.next_pos = pos
        return out

    def kda_snapshot_states(self, producer: Any, fallback: Any) -> Any:
        """vllm-tt-plugin hook (GLM53_KDA_SNAP_IMPORT, Stage 12 lever 1 (b)): for a hit resuming at the checkpoint an
        export_kda_state_async `producer` exported, while the device snapshot still holds it: import_kda_state's input
        that restores the slot from the snapshot buffers on device ({"next_pos", "states": marker}); fallback() -> the
        host checkpoint (export_kda_state's value) when the snapshot goes stale before the prefill. None: no handle /
        stale now (the plugin loads the host checkpoint as before)."""
        h = getattr(producer, "kda_snapshot", None)
        if h is None or not h.valid():
            return None
        return {"next_pos": int(producer.next_pos), "states": h.states(lambda: fallback()["states"])}

    def import_kda_state(self, kv_cache, slot: int, states: Any) -> None:
        """The plugin hands a prefix hit's KDA checkpoint for vLLM state slot `slot` before the hit's continuation
        prefill: kept until that row's prefill_forward, which places the request (its generator slot is chosen there)
        and restores the state as part of the prefill (Glm53Generator.prefill_forward prefix_states)."""
        self._check_cache(kv_cache)
        if not isinstance(states, dict) or {"next_pos", "states"} - set(states):
            raise TypeError("import_kda_state takes what export_kda_state returned ({'next_pos', 'states'})")
        self._prefix_states[int(slot)] = states

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Legacy uniform allocator API: same validation as the per-layer path."""
        return self.allocate_kv_cache_per_layer([(tuple(kv_cache_shape), dtype, i) for i in range(num_layers)])

    def _check_cache(self, kv_cache) -> None:
        if self.kv_cache is None or kv_cache is not self.kv_cache:
            raise ValueError("serving must pass the cache allocated by allocate_kv_cache_per_layer")

    # ------------------------------------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        *,
        enable_trace=False,
        start_pos=None,
        empty_slots=None,
        sampling_params=None,
        page_tables_per_layer=None,
        sample_rows=None,
        kda_snapshot_at=None,
        segment_rows=None,
        **kwargs,
    ):
        """vLLM prefill step. `tokens` [B, >= end] holds each row's token history, `prompt_lens[i]` is the END of
        the chunk scheduled for row i and `start_pos[i]` its start (the scheduler's computed tokens), so the
        chunk is tokens[i, start:end] at absolute positions start..end-1. Rows are front-packed prefill rows;
        `empty_slots[i]` is row i's vLLM state slot. The generator pads each chunk to its physical buckets; logical
        lengths and positions pass through unchanged.

        `sample_rows[i] = False` (the plugin's intermediate-prefill mask under `supports_partial_prefill_sampling`):
        row i's chunk does not end its prompt, so the generator runs it head-less (no head, no sample, no RNG
        advance): a run of K chunks (tt.prefill_chunk_run, doc/ttnn/09-vllm/chunk_pipelining.md) is K bucket replays
        enqueued back to back with no host wait; the row's result is the -1 placeholder the plugin drops. The mask
        applies with `sampling_params` only (a host-sampled step reads every row).

        `kda_snapshot_at[i]` (one-replay store A, kda_checkpoint_in_replay): row i's checkpoint cut (absolute
        position, 0 / None = none) when its chunk covers it and the request stores: the replay also writes the KDA
        state at the cut, exported afterwards by export_kda_state(_async)(at=cut).

        `segment_rows` (segtrace, GLM53_PF_SEGMENTS; the plugin's alternating loop, Stage 12 item 10): [s] runs only
        segment s of the call's ONE row's chunk (prefill_segments(start, n) of them). Segment 0 does the row's whole
        setup here (the prefix plan, imports, the slot) and records it. Every later segment re-sends the same row: it
        runs segment 0's recorded arguments, and nothing is re-planned. The slot's mirror, the prefix snapshot and the
        counters are committed at the LAST segment. A non-final segment returns the head-less placeholder.
        prefill_abort(slot) drops an open chunk; open_chunk() names it."""
        if kwargs:
            raise TypeError(f"unsupported prefill arguments for GLM-5.3-Flash: {sorted(kwargs)}")
        head_rows = None if sample_rows is None else [bool(v) for v in _as_list(sample_rows)]
        if not self._traces_captured:
            raise RuntimeError(
                "GLM-5.3-Flash serves through the generator's traces, but none were captured: launch with "
                "tt.trace_mode all (or decode_only) and tt.enable_model_warmup true"
            )
        if page_tables_per_layer is not None and any(
            t is not None and t is not page_table for t in page_tables_per_layer[1:]
        ):
            raise ValueError("GLM-5.3-Flash has one KV cache group; per-layer page tables must all be the group's")
        self._check_cache(kv_cache)
        self._call_enter("prefill")
        if self._spec is not None:  # Stage 17: rows 1..K of the shared state are the speculating request's candidates
            self._spec_leave()
        ends = _as_list(prompt_lens)
        rows = len(ends)
        starts = _as_list(start_pos) if start_pos is not None else [0] * rows
        vslots = _as_list(empty_slots) if empty_slots is not None else list(range(rows))
        if len(starts) != rows or len(vslots) != rows or len(set(vslots)) != rows:
            raise ValueError(f"prefill rows: ends={ends} starts={starts} slots={vslots}")
        if tokens.shape[0] < rows or page_table.shape[0] < rows:
            raise ValueError(f"{rows} prefill rows, tokens {tuple(tokens.shape)}, page table {tuple(page_table.shape)}")
        from models.autoports.zai_org_glm_5_3_flash.tt import generator as _gen  # (GLM53_RESUME_AS_DECODE, read late)

        for i, (s, e) in enumerate(zip(starts, ends)):
            if not 0 <= s < e <= tokens.shape[1]:
                raise ValueError(f"row {i}: chunk [{s}, {e}) outside the token history of width {tokens.shape[1]}")
            blob = self._prefix_states.get(vslots[i])
            # a resumed request's one token at an exact checkpoint: the generator decodes it, at any start
            resume_row = (_gen.RESUME_AS_DECODE and e - s == 1 and blob is not None and blob.get("next_pos") == s
                          and (sample_rows is None or bool(_as_list(sample_rows)[i])))  # fmt: skip
            if s % BLOCK_SIZE and not resume_row:
                raise ValueError(
                    f"row {i}: chunk start {s} is not a multiple of the {BLOCK_SIZE}-token latent page; the plugin "
                    f"must align chunk ends (tt.prefill_chunk_alignment, a multiple of {BLOCK_SIZE})"
                )
            if e > self.max_seq_len:
                raise ValueError(f"row {i}: chunk end {e} > max_model_len {self.max_seq_len}")
        seg_s = None  # segtrace: this call's segment of the row's one chunk; None = the whole chunk (today)
        if segment_rows is not None and any(x is not None for x in _as_list(segment_rows)):
            segs = _as_list(segment_rows)
            if rows != 1 or len(segs) != 1:
                raise ValueError(f"segment_rows {segs}: a segmented chunk is its call's only row ({rows} rows)")
            seg_s = int(segs[0])
        seg_open = self.__dict__.setdefault("_seg_open", {})  # vLLM slot -> segment 0's recorded call
        if seg_s is not None and seg_s > 0:
            o = seg_open.get(vslots[0])
            if o is None or o["raw"] != (starts[0], ends[0]):
                raise ValueError(f"segment {seg_s} of [{starts[0]}, {ends[0]}) at slot {vslots[0]}: the open chunk is "
                                 f"{None if o is None else o['raw']}")  # fmt: skip
            if not seg_s < o["n_seg"]:
                raise ValueError(f"segment {seg_s} of a {o['n_seg']}-segment chunk")
            params = None if sampling_params is None else generator_sampling_params(sampling_params, rows)
            try:
                out = self.generator.prefill_forward(
                    o["chunk"], page_table=o["page_rows"], kv_cache=kv_cache, prompt_lens=o["lengths"],
                    empty_slots=o["gslots"], start_pos=o["starts"], sampling_params=params, enable_trace=enable_trace,
                    **o["extra"], segment_rows=[seg_s],
                    **({"sample_rows": list(sample_rows)} if sample_rows is not None and params is not None else {}),
                )  # fmt: skip
            except Exception:  # a failed segment leaves no open chunk behind (14's 13:07:08Z SHOULD 1); as the whole-chunk
                self.prefill_abort(vslots[0])  # path, the failed call is not closed (_call_exit)
                raise
            self.counters["prefill_segments"] = self.counters.get("prefill_segments", 0) + 1
            if seg_s == o["n_seg"] - 1:  # the chunk is complete: commit as one whole prefill would have
                del seg_open[vslots[0]]
                self._prefill_accepted(o["tokens"], o["table"], vslots, o["gslots"], o["keys"], o["starts"], ends,
                                       o["pc_attach"], o["pc_blob"], o["pc_snap"], o["pc_key"], params, sample_rows)  # fmt: skip
            self._call_exit("prefill")
            return out
        if seg_s == 0 and vslots[0] in seg_open:
            raise ValueError(f"segment 0 at slot {vslots[0]} while its chunk {seg_open[vslots[0]]['raw']} is open")
        raw0 = (starts[0], ends[0]) if seg_s is not None else None
        # prefix cache: a fresh row starting past 0 (vLLM's page hit) attaches to a snapshot (or restarts lower); a
        # row ending on a boundary snapshots. Decided before the chunk is sliced (a restart lowers the start).
        pc_attach, pc_blob, pc_snap, pc_key = {}, {}, {}, {}
        if self.prefix is not None:
            for i, (v, s, e) in enumerate(zip(vslots, starts, ends)):
                st = self.slot_table.slots.get(self.gen_slot[v])
                fresh = s == 0 or st is None or st.key != int(page_table[i, 0]) or st.next_pos != s
                s2, att, blob, snap, key_e = self._prefix_plan(i, tokens, s, e, fresh, v in self._prefix_states)
                starts[i] = s2
                if att is not None:
                    pc_attach[i] = att
                if blob is not None:
                    pc_blob[i] = blob
                if snap is not None:
                    pc_snap[i] = snap
                if key_e is not None:
                    pc_key[i] = key_e
        # a KV-connector hit whose checkpoint has no DSA rows cannot resume in the sparse regime (the slot's pooled keys of
        # [0, hit) would be its previous owner's: exact only while the context stays <= 2,051, wrong top-k past it):
        # the row restarts at 0 (its pages are rewritten with identical content)
        if self._prefix_states and self._dsa_state():
            paged = getattr(self.generator.backend, "pooled_paged", lambda kv: False)(self.kv_cache)
            for i, v in enumerate(vslots):
                blob = self._prefix_states.get(v)
                why = None if blob is None or starts[i] <= 0 else _hit_restart(blob, starts[i], paged,
                                                                               getattr(self, "_hit_dma", None))  # fmt: skip
                if why == "no_dsa":
                    logger.warning("prefill row {}: connector checkpoint at {} has no DSA rows: restarting at 0", i,
                                   starts[i])  # fmt: skip
                elif why == "no_pooled":  # a ring-only checkpoint (or a paged store) needs the hit pages' pooled keys
                    logger.warning("prefill row {}: connector checkpoint at {} has no pooled keys: restarting at 0",
                                   i, starts[i])  # fmt: skip
                if why is not None:
                    self.counters[f"kda_import_{why}"] = self.counters.get(f"kda_import_{why}", 0) + 1
                    starts[i] = 0
        lengths = [e - s for s, e in zip(starts, ends)]
        chunk = torch.zeros(rows, max(lengths), dtype=torch.int64)
        for i, (s, e) in enumerate(zip(starts, ends)):
            chunk[i, : e - s] = tokens[i, s:e].to(torch.int64)
        table = page_table[:rows].to(torch.int32)
        # KV-connector prefix hits: the plugin imported the prefix's KDA checkpoint for the row's vLLM slot just before
        # this call (import_kda_state); the row is a new request starting at its hit length, which must be the
        # checkpoint's (kept until the generator accepts the prefill, so a refused one can be retried)
        prefix, pooled = {}, {}
        for i, v in enumerate(vslots):
            if v in self._prefix_states:
                if starts[i] == 0:  # an abandoned hit: the request is prefilled from 0, its checkpoint is moot
                    del self._prefix_states[v]
                    continue
                blob = self._prefix_states[v]
                if blob["next_pos"] != starts[i]:
                    raise ValueError(
                        f"row {i}: KDA checkpoint after {blob['next_pos']} tokens for a hit continuing at {starts[i]}"
                    )
                prefix[i] = blob["states"]
                if blob.get("pooled") is not None:
                    pooled[i] = blob["pooled"]
        for i, e in pc_blob.items():  # a host-tier snapshot: imported like a connector checkpoint
            prefix[i] = e.host_blob
        for i, (v, s) in enumerate(zip(vslots, starts)):
            if (s == 0 or i in prefix or i in pc_attach) and self.pack_slots:
                self._take_lowest_free_slot(v, vslots)
        gslots = [self.gen_slot[v] for v in vslots]
        if pooled:  # GLM53_KV_POOLED_PAGES: the hits' pooled keys into their generator slots, before the prefill
            self._load_prefix_pooled({i: (gslots[i], p, table[i].tolist()) for i, p in pooled.items()})
        keys = [int(table[i, 0]) for i in range(rows)]
        for i, (slot, key, s, e) in enumerate(zip(gslots, keys, starts, ends)):
            self.slot_table.check_prefill(slot, key, s, e, fresh=i in prefix or i in pc_attach)
        params = None if sampling_params is None else generator_sampling_params(sampling_params, rows)
        attach_entries = {i: int(e.device_entry) for i, e in pc_attach.items()}
        snap_at = {i: int(v) for i, v in enumerate(list(kda_snapshot_at or [])[:rows]) if v}  # one-replay store A
        n_seg = 1
        if seg_s is not None:  # segment 0: the model's segment count for the chunk as the plan left it
            n_seg = int(self.prefill_segments(starts[0], lengths[0]))
            if raw0 != (starts[0], ends[0]) and n_seg != int(self.prefill_segments(*raw0[:1], raw0[1] - raw0[0])):
                raise ValueError(f"segtrace: the prefix plan moved the chunk {raw0} to start {starts[0]} and changed its "
                                 f"segment count; the scheduler's stamp no longer applies")  # fmt: skip
        try:
            out = self.generator.prefill_forward(
                chunk,
                page_table=vllm_page_rows(table),
                kv_cache=kv_cache,
                prompt_lens=lengths,
                empty_slots=gslots,
                start_pos=starts,
                sampling_params=params,
                enable_trace=enable_trace,
                **({"prefix_states": prefix} if prefix else {}),
                **({"prefix_attach": attach_entries} if attach_entries else {}),
                **({"snapshot_after": dict(pc_snap)} if pc_snap else {}),
                **({"sample_rows": list(sample_rows)} if sample_rows is not None and params is not None else {}),
                **({"snapshot_at": snap_at} if snap_at else {}),
                **({"segment_rows": [0]} if seg_s is not None else {}),
            )
        except Exception:
            for ent in pc_snap.values():  # reserved entries never written
                self.prefix.unreserve(ent)
            if seg_s is not None:  # segment 0 failed: the model's open chunk (if it got that far) goes too
                abort = getattr(self.generator, "prefill_abort", None)
                if callable(abort):
                    abort(int(gslots[0]))
            raise
        for i in prefix:
            self._prefix_states.pop(vslots[i], None)
        self.counters["kda_imports"] += len(prefix)
        if seg_s is not None and n_seg > 1:  # an open chunk: its later segments re-send this call; commit at the last
            seg_open[vslots[0]] = {
                "raw": raw0, "n_seg": n_seg, "chunk": chunk, "page_rows": vllm_page_rows(table), "lengths": lengths,
                "gslots": gslots, "starts": starts, "keys": keys, "tokens": tokens, "table": table,
                "pc_attach": pc_attach, "pc_blob": pc_blob, "pc_snap": pc_snap, "pc_key": pc_key,
                "extra": {**({"prefix_attach": attach_entries} if attach_entries else {}),
                          **({"snapshot_after": dict(pc_snap)} if pc_snap else {}),
                          **({"snapshot_at": snap_at} if snap_at else {})},
            }  # fmt: skip
            self.counters["prefill_segments"] = self.counters.get("prefill_segments", 0) + 1
            self._call_exit("prefill")
            return out
        self._prefill_accepted(tokens, table, vslots, gslots, keys, starts, ends, pc_attach, pc_blob, pc_snap, pc_key,
                               params, sample_rows)  # fmt: skip
        self._call_exit("prefill")
        return out

    def _prefill_accepted(self, tokens, table, vslots, gslots, keys, starts, ends, pc_attach, pc_blob, pc_snap, pc_key,
                          params, sample_rows) -> None:
        """the generator accepted the prefill (a whole chunk, or a segmented chunk's last segment): the slot mirror, the
        sampling params, the prefix snapshots, the counters and the event"""
        if sample_rows is not None and params is not None and not all(bool(v) for v in _as_list(sample_rows)):
            self.counters["prefill_headless_rows"] = self.counters.get("prefill_headless_rows", 0) + sum(
                1 for v in _as_list(sample_rows) if not bool(v)
            )
        # accepted: the generator now holds each row's state in its slot
        for i, (slot, key, e) in enumerate(zip(gslots, keys, ends)):
            self.slot_table.commit_prefill(slot, key, e)
            if params is not None:
                self._params[slot] = params[i]
        if self.prefix is not None:
            self._prefix_commit(tokens, table, gslots, starts, ends, pc_attach, pc_blob, pc_snap, pc_key)
        self.counters["prefill_calls"] += 1
        self._prev_decode_t = None  # the next decode interval holds this prefill: not a steady step
        if self._events_path:
            self._event(
                "prefill",
                rows=[
                    {
                        "vslot": v,
                        "gslot": g,
                        "start": s,
                        "end": e,
                        "n": e - s,
                        "attach": None
                        if i not in pc_attach
                        else {"entry": pc_attach[i].device_entry, "kind": pc_attach[i].kind},
                        "host_attach": i in pc_blob,
                        "snapshot": pc_snap.get(i),
                        # the chunk's first / last page (vLLM block ids): the cross-request isolation control compares a
                        # request's tail page with its predecessor's
                        "blocks": [int(table[i, s // BLOCK_SIZE]), int(table[i, (e - 1) // BLOCK_SIZE])],
                    }
                    for i, (v, g, s, e) in enumerate(zip(vslots, gslots, starts, ends))
                ],
                device_sampling=params is not None,
                report=self._light_report(),
            )

    # -- segtrace's vLLM face (Stage 12 item 10: the plugin's alternating loop; TTModelRunner._segment_prepare) ---------

    def prefill_segments(self, start: int, n: int) -> int:
        """S, the segments the model runs a chunk [start, start + n) in (1 = one replay, GLM53_PF_SEGMENTS unset)"""
        fn = getattr(self.generator.backend, "prefill_segments", None)
        return int(fn(int(start), int(n))) if callable(fn) else 1

    def prefill_abort(self, slot: int) -> None:
        """drop vLLM slot `slot`'s open segmented chunk (its request left inside the hold): the model's open chunk, the
        snapshot entries its segment 0 reserved and the recorded call"""
        o = self.__dict__.setdefault("_seg_open", {}).pop(int(slot), None)
        abort = getattr(self.generator, "prefill_abort", None)
        if callable(abort) and 0 <= int(slot) < len(self.gen_slot) and self.gen_slot[int(slot)] is not None:
            abort(int(self.gen_slot[int(slot)]))
        if o is not None and self.prefix is not None:
            for ent in o["pc_snap"].values():
                self.prefix.unreserve(ent)
        self.counters["prefill_segment_aborts"] = self.counters.get("prefill_segment_aborts", 0) + 1

    def open_chunk(self):
        """the vLLM slot whose segmented chunk is open, or None (the plugin refuses a whole-chunk prefill while one is)"""
        tpp = getattr(self.generator.backend, "tpp", None)
        fn = getattr(tpp, "open_chunk", None)
        g = fn() if callable(fn) else None
        if g is None:
            return None
        return next((v for v, gs in enumerate(self.gen_slot) if gs == g), None)  # (vLLM slots only: 14's SHOULD 2)

    def _prefix_commit(self, tokens, table, gslots, starts, ends, pc_attach, pc_blob, pc_snap, pc_key) -> None:
        """After the generator accepted the rows: readers of attached entries, the new snapshots' registrations, the
        per-slot token history (the hash chain of later boundaries) and the TTFT accounting."""
        for i, (g, s, e) in enumerate(zip(gslots, starts, ends)):
            if s == 0:
                self._slot_tokens[g] = []
                self.prefix.release_slot(g)
            ent = pc_attach.get(i) or pc_blob.get(i)
            if ent is not None:
                self.prefix.attach(ent.key, g)
                self.counters["prefix_attach_rows"] = self.counters.get("prefix_attach_rows", 0) + 1
                self.counters["prefix_tokens_saved"] = self.counters.get("prefix_tokens_saved", 0) + s
                self._slot_tokens[g] = tokens[i, :s].tolist()
            self.counters["prefix_tokens_prefilled"] = self.counters.get("prefix_tokens_prefilled", 0) + (e - s)
            hist = self._slot_tokens.setdefault(g, [])
            if len(hist) == s:
                hist.extend(tokens[i, s:e].tolist())
            else:  # a continuation whose history we did not see (resumed elsewhere): take the row's history
                self._slot_tokens[g] = tokens[i, :e].tolist()
            if i in pc_snap:
                pages = [int(p) for p in table[i, : -(-e // BLOCK_SIZE)].tolist()]
                self.prefix.register(pc_key[i], n_tokens=e, pages=pages, kind="P", device_entry=pc_snap[i])
                self.counters["prefix_snapshot_rows"] = self.counters.get("prefix_snapshot_rows", 0) + 1

    def _take_lowest_free_slot(self, vslot: int, batch: list[int]) -> None:
        """A fresh request takes the lowest free generator slot (a relabelling of two empty slots: nothing moves), so
        live state stays packed at the low end, where bucketed decode widths read it. A vLLM slot that still maps to
        live state keeps it, so `check_prefill` refuses the fresh prefill instead of the stale state changing owner."""
        if self.gen_slot[vslot] in self.slot_table.slots:
            return
        busy = set(self.slot_table.slots) | {self.gen_slot[v] for v in batch if v != vslot}
        free = [g for g in range(self.max_batch_size) if g not in busy]
        if not free or free[0] >= self.gen_slot[vslot]:
            return
        other = self.gen_slot.index(free[0])
        self.gen_slot[vslot], self.gen_slot[other] = self.gen_slot[other], self.gen_slot[vslot]
        self.counters["slot_swaps"] += 1

    def _compact(self) -> None:
        """Move live requests that sit at or above the smallest decode width covering their count (the generator's
        `decode_buckets`: pow2 widths or the backend's own trace widths below max_batch, plus max_batch) into free lower
        generator slots (one `remap_slots` move each, always into a free slot, so no cycles). Runs before every decode
        step's inputs are built: a finished request does not always make the runner reload, so a steady step can follow
        a release. The moves are physical (KDA state, page-table row and next-input token travel with the request; the
        destination's device sampler row is what the backend's copy_slot leaves there; the Stage 4/6 pipeline backend
        re-seeds it like a fresh request), so the indirection, the mirror and the resident set follow immediately. A
        full-width backend (`width_bucketed_decode = False`) has one decode bucket, so this never moves a request. When
        the backend's copy_slot does not carry the device RNG (`copy_slot_keeps_sampler_rng = False`), explicitly
        seeded device-sampled requests stay put (a move would restart their seeded stream) and the width covers them."""
        live = sorted(self.slot_table.slots)
        if self._spec is not None and live != [0]:  # a move must carry plain decode inputs (Stage 17)
            self._spec_leave()
        pinned = (
            []
            if getattr(self.generator.backend, "copy_slot_keeps_sampler_rng", True)
            else [g for g in live if _device_seeded(self._params.get(g))]
        )
        need = max(len(live), max(pinned, default=-1) + 1)
        width = next((w for w in sorted(self.generator.decode_buckets) if w >= need), self.max_batch_size)
        free = [g for g in range(width) if g not in self.slot_table.slots]
        for hi in [g for g in live if g >= width]:
            lo = free.pop(0)
            t0 = time.perf_counter()
            self.generator.remap_slots(self.kv_cache, {hi: lo})
            dt = time.perf_counter() - t0
            self.counters["compaction_s_total"] += dt
            self.counters["compaction_s_max"] = max(self.counters["compaction_s_max"], dt)
            v_hi, v_lo = self.gen_slot.index(hi), self.gen_slot.index(lo)
            self.gen_slot[v_hi], self.gen_slot[v_lo] = lo, hi
            self.slot_table.move(hi, lo)
            if self.prefix is not None:
                self.prefix.move_slot(hi, lo)
            if hi in self._slot_tokens:
                self._slot_tokens[lo] = self._slot_tokens.pop(hi)
            if hi in self._params:
                self._params[lo] = self._params.pop(hi)
            if hi in self._resident:
                self._resident.discard(hi)
                self._resident.add(lo)
            self.counters["compaction_moves"] += 1

    # ------------------------------------------------------------------------------------------------- decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        *,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        page_tables_per_layer=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        **kwargs,
    ):
        """One decode step over all `max_batch_size` rows (row i = vLLM slot i after `slot_remap`; padded rows have
        position -1). Version-1 decode-input contract, executed through the generator's inputs: `reload_inputs` ->
        the host tokens / positions / page table; otherwise token -1 (the token the device sampled last step), the
        resident positions (this adapter's mirror of the device's +1 per step) and page table None unless
        `reload_page_table`. On a non-reloading step `tokens` / `start_pos` are stale by design and are not read."""
        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract=1: the legacy reset_batch keyword is not accepted")
        if kwargs:
            raise TypeError(f"unsupported decode arguments for GLM-5.3-Flash: {sorted(kwargs)}")
        if spec_mode is None and (num_valid_drafts is not None or accepted_counts is not None):
            raise TypeError("num_valid_drafts / accepted_counts belong to a speculative verify step (spec_mode)")
        self._call_enter("decode")
        t_entry, prev_t = time.perf_counter(), self._prev_decode_t
        self._prev_decode_t = None  # set again only when this call completes as a plain decode step
        spec_prev, self._spec_t_exit = self._spec_t_exit, None
        if spec_mode is not None:  # Stage 17: a speculative verify block [S, 1+K]
            self._spec_tl = None
            out = self._verify(tokens, start_pos, page_table, kv_cache, num_valid_drafts=num_valid_drafts,
                               accepted_counts=accepted_counts, spec_mode=spec_mode, enable_trace=enable_trace,
                               sampling_params=sampling_params, slot_remap=slot_remap)  # fmt: skip
            if self._spec_tl is not None:
                self._spec_timeline(t_entry, *self._spec_tl, spec_prev)
            self._call_exit("verify")
            return out
        if self._spec is not None:  # an ordinary step after speculation: the plain decode inputs back first
            self._spec_leave()
        if not enable_trace:
            raise ValueError(
                "GLM-5.3-Flash serving decodes through the generator's trace (tt.trace_mode all|decode_only)"
            )
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already reloads page tables; reload_page_table must be False with it")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("reset_sampling_state requires reload_inputs (authoritative positions)")
        if page_tables_per_layer is not None and any(
            t is not None and t is not page_table for t in page_tables_per_layer[1:]
        ):
            raise ValueError("GLM-5.3-Flash has one KV cache group; per-layer page tables must all be the group's")
        if prompt_tokens is not None or output_tokens is not None:
            raise ValueError("penalty state reached a device-sampled decode; penalties are host-sampled for this model")
        self._check_cache(kv_cache)
        # steady pair: the previous call was an accepted plain decode, nothing ran in between, inputs stay resident
        live = len(self._resident)
        steady = prev_t is not None and not reload_inputs and not reload_page_table and live > 0
        probe = False
        if steady:
            dt_wall = t_entry - prev_t
            self.counters["decode_steady_steps"] += 1
            self.counters["decode_steady_wall_s_total"] += dt_wall
            self._steady_wall.setdefault(live, deque(maxlen=HOST_STEP_WINDOW)).append(dt_wall)
            probe = self._probe_every > 0 and self.counters["decode_steady_steps"] % self._probe_every == 0
            probe = probe and self._device_sync()  # drained: this step runs alone
        t0 = time.perf_counter()
        S = self.max_batch_size
        if tokens.shape[0] != S:
            raise ValueError(f"decode tokens {tuple(tokens.shape)}: expected {S} rows (one per slot)")
        if self.pack_slots:
            self._compact()
        gen_slot = list(self.gen_slot)
        moves = None
        if slot_remap is not None:
            remap = _as_list(slot_remap)
            if sorted(remap) != list(range(S)):
                raise ValueError(f"slot_remap {remap} is not a permutation of {S} slots")
            if remap != list(range(S)):
                if not reload_inputs:
                    raise ValueError(
                        "a non-identity slot_remap needs reload_inputs: moved rows have no resident inputs"
                    )
                # row i now reads the state vLLM held in slot remap[i]: re-point the indirection, move nothing
                gen_slot = [self.gen_slot[j] for j in remap]
                moves = {j: i for i, j in enumerate(remap) if i != j}
        if reload_inputs:
            host_pos = _as_list(start_pos)
            keys = [int(k) for k in page_table[:S, 0].tolist()]
            self.slot_table.check_decode_inputs(gen_slot, host_pos, keys)
            active = [gen_slot[i] for i, p in enumerate(host_pos) if p >= 0]
            dec_tokens, dec_pos, dec_table = tokens, start_pos, vllm_page_rows(page_table)
        else:
            active = sorted(self._resident)
            pos = self.slot_table.resident_positions(gen_slot, self._resident)
            dec_tokens = torch.full((S, 1), -1, dtype=torch.int64)
            dec_pos = torch.tensor(pos, dtype=torch.int64)
            dec_table = vllm_page_rows(page_table) if reload_page_table else None
        # (GLM53_SPEC_ASYNC) slot 0's held tokens: its input at a position the device already committed is answered
        # from them and the row sits this replay out (position -1); the device decodes it again from its own position
        held_row, dev_active = None, active
        if self._held is not None and 0 in active:
            r0 = gen_slot.index(0)
            x = int(torch.as_tensor(dec_tokens).reshape(S, -1)[r0, 0]) if reload_inputs else -1
            ans = self._held_out(int(torch.as_tensor(dec_pos).reshape(-1)[r0]), x)
            if ans is not None:
                held_row = (r0, int(ans))
                dec_pos = torch.as_tensor(dec_pos).reshape(-1).clone()
                dec_pos[r0] = -1
                dev_active = [g for g in active if g != 0]
        # reload_sampling_params (or reload_inputs): upload the scheduler's parameters; otherwise pass the ones last
        # uploaded per generator slot (the generator writes only differing values, so this writes nothing).
        # reset_sampling_state ("rebuild RNG state for the current request layout") is executed by the layout itself:
        # slot_remap only re-points the indirection (a request keeps its generator slot and that slot's device RNG),
        # and the generator seeds a request's slot when it is admitted (prefill from position 0; seed None = fresh
        # state). Compaction moves (width-bucketed backends only; never on the full-width pipeline) leave the
        # destination row's RNG to the backend's copy_slot (see _compact). Every row's RNG therefore already belongs to
        # the current layout; re-seeding here would restart the stream of every continuing seeded request at each
        # transition (test_engine_seeded_sampling_state_follows_the_request, with the re-seeding control).
        # generation (G) snapshots inside the decode trace (GLM53_DECODE_TEE): a slot whose step consumes position
        # 128 m - 1 tees its post-step state into its trailing entry; the DSA pooled-key pages since the entry's last
        # crossing ride along (all of them for a fresh entry). No crossing: the tee's ctl is cleared (one write).
        tee = self.prefix is not None and self.generator.decode_tee(kv_cache)
        if tee:
            pairs = []
            for g in dev_active:  # (a held row: the device crossed its boundaries ahead of the runner; no tee)
                p = self.slot_table.slots[g].next_pos + 1
                if p % self.prefix.boundary == 0:
                    prev = self.prefix.trailing.get(g)
                    ent = self.prefix.trailing_entry(g, p)
                    if ent is not None:
                        t0 = prev[1] if prev is not None and prev[0] == ent else 0  # tokens already in the entry
                        pairs.append((g, ent, t0, p))
            self.generator.set_decode_tee(kv_cache, pairs)
            if pairs:
                self.counters["prefix_trailing_snapshots"] = self.counters.get("prefix_trailing_snapshots", 0) + len(
                    pairs
                )
        params = None
        if sampling_params is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

            if reload_inputs or reload_sampling_params:
                params = generator_sampling_params(sampling_params, S)
            else:
                params = [self._params.get(g, SamplingParams()) for g in gen_slot]
        device_read = read_from_device or params is None  # the logits path is read back by the generator
        self.generator.spec_lone_width = self._spec_lone_width(dev_active, gen_slot, params, sampling_params)
        if held_row is not None and not dev_active:  # slot 0 alone, answered from its held tokens: no replay
            out = None
        else:
            out = self.generator.decode_forward(
                dec_tokens,
                dec_pos,
                page_table=dec_table,
                kv_cache=kv_cache,
                slots=gen_slot,
                enable_trace=True,
                sampling_params=params,
                read_from_device=device_read or held_row is not None,  # (a held row patches the host output)
            )
        if held_row is not None:
            out = self._held_patch(out, held_row, S, tokens_out=params is not None)
        if probe:
            t_sub = time.perf_counter()
            self._device_sync()
            t_done = time.perf_counter()
            self.counters["decode_probes"] += 1
            self.counters["decode_probe_device_s_total"] += t_done - t_sub
            self._probe_dev.setdefault(live, deque(maxlen=HOST_STEP_WINDOW)).append(t_done - t_sub)
            self._probe_total.setdefault(live, deque(maxlen=HOST_STEP_WINDOW)).append(t_done - t0)
        # accepted: commit the indirection and the mirror (one position consumed per active row)
        self.gen_slot = gen_slot
        self._last_moves = moves
        if params is not None:
            for row, g in enumerate(gen_slot):
                if g in active:
                    self._params[g] = params[row]
        self.slot_table.advance(active)
        snapped = 0  # trailing snapshots enqueued after this step (device copies: the next interval is not steady)
        if self.prefix is not None and not tee:  # generation snapshots without the tee (GLM53_DECODE_TEE): a slot
            # crossing a boundary copies its state after the step (plain DMA between steps; the tee copies it inside)
            for g in dev_active:
                p = self.slot_table.slots[g].next_pos
                if p > 0 and p % self.prefix.boundary == 0:
                    ent = self.prefix.trailing_entry(g, p)
                    if ent is not None:
                        self.generator.snapshot_slot(self.kv_cache, g, ent, upto=p)
                        snapped += 1
                        self.counters["prefix_trailing_snapshots"] = (
                            self.counters.get("prefix_trailing_snapshots", 0) + 1
                        )
        self._resident = set(active)
        self.counters["decode_calls"] += 1
        if moves:
            self.counters["remaps"] += 1
        dt = time.perf_counter() - t0
        if not probe:  # a probe's submit includes its device wait
            self.counters["decode_submit_s_total"] += dt
            self.counters["decode_submit_s_max"] = max(self.counters["decode_submit_s_max"], dt)
        # the step after a probe holds the drain bubble; after trailing snapshots it holds their state copies
        self._prev_decode_t = None if (probe or snapped) else t_entry
        if self._events_path and (reload_inputs or self.counters["decode_calls"] % EVENT_DECODE_EVERY == 0):
            self._event(
                "decode",
                reload_inputs=bool(reload_inputs),
                reload_page_table=bool(reload_page_table),
                active_slots=len(active),
                submit_s=round(dt, 6),
                probe=probe,
                report=self._light_report(),
            )
        self._call_exit("decode")
        if isinstance(out, torch.Tensor) and params is None:
            return out.reshape(S, 1, -1)  # logits [B, V] -> the plugin's [B, 1, V]
        return out

    # ------------------------------------------------------------------------------------------- Stage 17 spec
    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """the plugin's speculative plan (spec_admission.resolve_speculative_plan): the backend's step is built for
        ONE K (GLM53_SPEC_K) and drafts for a lone greedy request (every other step is the unchanged plain decode)"""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if SPEC_K <= 0:
            return SpecReject("the GLM-5.3-Flash backend was built without its speculative step (GLM53_SPEC_K)")
        if int(requested_k) < SPEC_K:
            return SpecReject(f"the device speculative step is built for K = {SPEC_K} drafts", supported_k=(SPEC_K,))
        # candidate KDA rows / conv histories live in the shared 32-row state (rows 1..K of the lone user's step):
        # no per-sequence bytes; per token: the MTP layer's latent cache 1 KiB, or the DFlash2 drafter's own paged
        # K / V cache (8 layers x 1 kv head per chip at TP8 x 128 x (K + V) bf16 = 4 KiB / chip) + its cos / sin
        # rows (512 B) -- the drafter's cache is sized for ONE speculating user (cap = max_seq_len + 32)
        per_token = 4096 + 512 if SPEC_KIND == "dflash2" else 1024
        return SpecPlan(effective_k=SPEC_K, lanes_per_request=1,
                        extra_bytes_per_seq=0, extra_bytes_per_token=per_token, accept_modes=("argmax_ids",),
                        drafter_state="internal", supports_narrow_decode=True)  # fmt: skip

    def _spec_lone_width(self, active, gen_slot, params, sampling_params) -> int:
        """Stage 17 (n29h6): the plain width of a step whose only decoding request is SPECULABLE (greedy, no logprobs:
        the plugin's _request_is_speculable) = the backend's speculative plain width (SpecMTP.plain_w, the smallest
        decode width >= K+1), so its plain steps and its verify share the programs; 0 otherwise (the w1 bucket serves
        non-speculable rows and MTP off)"""
        spw = int(getattr(getattr(self.generator.backend, "spec", None), "plain_w", 0) or 0)
        if SPEC_K <= 0 or not spw or len(active) != 1:
            return 0
        g = active[0]
        row = gen_slot.index(g) if g in gen_slot else -1
        p = params[row] if params is not None and 0 <= row < len(params) else self._params.get(g)
        if p is not None and not bool(getattr(p, "is_greedy", False)):
            return 0
        lp = getattr(sampling_params, "num_logprobs", None) if sampling_params is not None else None
        if lp is not None and 0 <= row < len(lp) and int(lp[row]) != -2:  # (vllm_tt_plugin LOGPROBS_NONE_SENTINEL)
            return 0
        return spw

    def _spec_leave(self) -> None:
        if self._pend is not None:  # (GLM53_SPEC_ASYNC) the pre-enqueued step's tokens are held, then it leaves
            self._spec_resolve()
            return
        self.generator.spec_leave(self.kv_cache)
        self._spec = None
        self.counters["spec_leaves"] = self.counters.get("spec_leaves", 0) + 1

    def _other_state_holders(self) -> list[int]:
        """generator slots other than 0 that hold live KDA state: the adapter's slot mirror (a request whose prefill
        has started, chunked or not, or that decodes) and the generator binding (active, not finished). A speculative
        step would overwrite rows 1..K of that state (GLM53_SPEC_SOLE_STATE)."""
        held = {int(g) for g in self.slot_table.slots if int(g) != 0}
        b = getattr(self.generator, "_binding", None)
        if b is not None:
            bd = b(self.kv_cache)
            held |= {s for s in range(1, bd.max_batch) if bd.slots[s].active and not bd.slots[s].finished}
        return sorted(held)

    # ------------------------------------------------------------- GLM53_SPEC_ASYNC: the pre-enqueued step
    def _held_patch(self, out, held_row, S: int, *, tokens_out: bool):
        """a decode step's host output with slot 0's row = its held answer (tokens [S], or logits [S, V] whose row
        has the answer as its only finite maximum); out None (no replay ran): built from the answer alone"""
        r0, ans = held_row
        if tokens_out:
            t = torch.zeros(S, dtype=torch.int64) if out is None else torch.as_tensor(out).reshape(S, -1)[:, 0].clone().to(torch.int64)
            t[r0] = ans
            return t
        V = int(self.generator.backend.vocab_size)
        lg = torch.full((S, V), -1e30, dtype=torch.float32) if out is None else torch.as_tensor(out).reshape(S, -1).clone()
        lg[r0] = -1e30
        lg[r0, ans] = 0.0
        return lg

    def _spec_submit_next(self, page_row) -> None:
        """After a lone step's record: enqueue the next step now when slot 0's pages already hold its verify positions
        (the runner's row of this block, written first) and the context / the drafter's cache allow it."""
        if not SPEC_ASYNC or self._spec is None or self._pend is not None:
            return
        g = self.generator
        g.spec_page_row(self.kv_cache, page_row)
        if not (g.spec_room(self.kv_cache) and g.spec_step_covered(self.kv_cache)):
            self.counters["spec_presubmit_skips"] = self.counters.get("spec_presubmit_skips", 0) + 1
            return
        g.spec_step_submit(self.kv_cache)
        self._pend = dict(pos=int(self._spec["pos"]), block=[int(self._spec["bonus"])] + [int(v) for v in self._spec["drafts"]])
        self.counters["spec_presubmits"] = self.counters.get("spec_presubmits", 0) + 1

    def _spec_resolve(self) -> None:
        """A call other than the pre-enqueued step's own verify block: read its record (the device committed n + 1
        tokens of slot 0 the runner has not seen -- a step cannot be undone: its verify overwrote the KDA candidate
        rows, its ring move is one-way), leave speculation, and HOLD those tokens: the runner's next inputs for slot 0
        at positions pos .. pos + n are answered from them (_held_out) until it reaches the device's position."""
        if self._pend is None:
            return
        rec = self.generator.spec_step_collect(self.kv_cache)
        pend, self._pend = self._pend, None
        n, a = int(rec["n"]), [int(v) for v in rec["a"]]
        self._held = dict(base=pend["pos"], inputs=[pend["block"][0]] + a[:n], outs=a[: n + 1])
        self.generator.spec_leave(self.kv_cache)
        self._spec = None
        for k, v in (("spec_leaves", 1), ("spec_held", 1), ("spec_held_tokens", n + 1)):
            self.counters[k] = self.counters.get(k, 0) + v

    def _held_out(self, p: int, x: int):
        """slot 0's input x (-1: the resident one) at position p -> the output the device already committed there, or
        None once the runner has reached the device's position (slot 0 then decodes on device again)"""
        h = self._held
        if h is None:
            return None
        j = int(p) - h["base"]
        if j >= len(h["outs"]):
            self._held = None
            return None
        if j < 0 or (int(x) >= 0 and int(x) != h["inputs"][j]):
            raise RuntimeError(f"slot 0's input {int(x)} at position {int(p)}: the device committed {h['inputs']} "
                               f"from position {h['base']} (the held tokens)")  # fmt: skip
        if j == len(h["outs"]) - 1:
            self._held = None  # (the next input is the device's own position)
        self.counters["spec_held_answers"] = self.counters.get("spec_held_answers", 0) + 1
        return h["outs"][j]

    def _verify(self, tokens, start_pos, page_table, kv_cache, *, num_valid_drafts, accepted_counts, spec_mode,
                enable_trace, sampling_params, slot_remap):  # fmt: skip
        """A verify block (the runner's [S, 1+K] tokens / positions): the lone speculating request (generator slot 0,
        its drafts = the ones propose_draft_tokens returned) runs ONE speculative-step replay (verify on K+1 rows,
        accept, draft) -> its row of greedy ids = the step's a_0..a_K (the runner's accept_greedy_drafts reproduces
        the device's n). A block without drafts is an ordinary step on column 0 (speculation left first)."""
        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        if spec_mode != "argmax_ids":
            raise NotImplementedError(f"GLM-5.3-Flash verifies greedy drafts only (spec_mode {spec_mode!r})")
        self._check_cache(kv_cache)
        tokens = torch.as_tensor(tokens)
        start_pos = torch.as_tensor(start_pos)
        S, W = int(tokens.shape[0]), int(tokens.shape[1])
        check_spec_side_tensors(num_valid_drafts, accepted_counts, rows=S, num_drafts=W - 1, call="verify")
        if slot_remap is not None and _as_list(slot_remap) != list(range(S)):
            if self._spec is not None:
                self._spec_leave()
        p0 = start_pos[:, 0]
        live = [r for r in range(S) if int(p0[r]) >= 0]
        drafted = [r for r in live if int(num_valid_drafts[r]) > 0]
        ids = torch.full((S, W), -1, dtype=torch.int32)
        if drafted:
            r = drafted[0]
            sp = self._spec
            expect = None if sp is None else [sp["bonus"]] + list(sp["drafts"])
            ok = (len(drafted) == 1 and live == [r] and sp is not None and self.gen_slot[r] == 0
                  and int(num_valid_drafts[r]) == W - 1 == self.generator.spec_k()
                  and [int(v) for v in tokens[r].tolist()] == expect and int(p0[r]) == sp["pos"]
                  and [int(v) for v in start_pos[r].tolist()] == list(range(sp["pos"], sp["pos"] + W))
                  and not (SPEC_SOLE_STATE and self._other_state_holders()))  # fmt: skip
            if not ok:
                # any other block (a second request arrived after the lone proposal, a slot remap, several drafted
                # rows, plain rows mixed in): every live row binds its OWN drafts, verified by ordinary decode
                # sub-steps (_verify_plain: lossless by construction, any mix of plain and drafted rows); speculation
                # is left first
                if self._spec is not None:
                    self._spec_leave()
                nv = {r_: int(num_valid_drafts[r_]) for r_ in live}
                ids = self._verify_plain(tokens, start_pos, page_table, kv_cache, nv, live, enable_trace=enable_trace,
                                         sampling_params=sampling_params, slot_remap=slot_remap)  # fmt: skip
                return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        if drafted:
            rows = vllm_page_rows(page_table)
            self.slot_table.check_decode_inputs([0], [int(p0[r])], [int(page_table[r, 0])])
            t_c = time.perf_counter()
            if self._pend is not None:  # (GLM53_SPEC_ASYNC) this block's step was enqueued at the last verify
                self._pend = None
                rec = self.generator.spec_step_collect(kv_cache)
            else:
                rec = self.generator.spec_step(kv_cache, page_row=rows[r])
            t_r = time.perf_counter()
            n = int(rec["n"])
            ids[r] = torch.tensor(rec["a"], dtype=torch.int32)
            for _ in range(n + 1):
                self.slot_table.advance([0])
            self._resident = {0}
            self._spec = dict(bonus=int(rec["a"][n]), drafts=list(rec["drafts"]), pos=int(p0[r]) + n + 1)
            self.counters["spec_steps"] = self.counters.get("spec_steps", 0) + 1
            self.counters["spec_tokens"] = self.counters.get("spec_tokens", 0) + n + 1
            self._spec_submit_next(rows[r])  # (GLM53_SPEC_ASYNC) the next step runs during the runner's host turn
            self._spec_tl = (t_c, t_r)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)
        # no drafts in the block: an ordinary step on column 0
        if self._spec is not None:
            self._spec_leave()
        out = self.decode_forward(tokens[:, :1].contiguous(), p0.contiguous(), page_table, kv_cache,
                                  enable_trace=enable_trace, read_from_device=True, sampling_params=sampling_params,
                                  slot_remap=slot_remap, reload_inputs=True)  # fmt: skip
        if not isinstance(out, torch.Tensor):
            out = self.process_decode_output_host(self.read_decode_output(out), is_tokens=sampling_params is not None)
        tok = out.reshape(S, -1)
        if tok.shape[1] != 1:  # (the adapter does no sampling math: a verify step samples on device)
            raise ValueError("a speculative verify step needs device sampling (sampling_params)")
        col = tok[:, 0]
        for r in live:
            ids[r, 0] = int(col[r])
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=None)

    def _spec_timeline(self, t_in: float, t_c: float, t_r: float, prev_exit) -> None:
        """the lone speculative step's host timeline per verify call (counters spec_t_<part>_s_total / _n): pre = entry
        to the record wait (input checks), collect = the record wait (spec_step_collect / spec_step: the device step
        the runner's turn did not hide, plus the read), post = bookkeeping + the next step's page row and enqueue
        (GLM53_SPEC_ASYNC), turn = the runner's time since the previous call returned, when that call was a lone
        speculative verify too"""
        t_out = time.perf_counter()
        c = self.counters
        parts = [("pre", t_c - t_in), ("collect", t_r - t_c), ("post", t_out - t_r)]
        if prev_exit is not None:
            parts.append(("turn", t_in - prev_exit))
        for k, v in parts:
            c[f"spec_t_{k}_s_total"] = c.get(f"spec_t_{k}_s_total", 0.0) + v
            c[f"spec_t_{k}_n"] = c.get(f"spec_t_{k}_n", 0) + 1
        c["spec_t_collect_s_max"] = max(c.get("spec_t_collect_s_max", 0.0), t_r - t_c)
        self._spec_t_exit = t_out

    def _verify_plain(self, tokens, start_pos, page_table, kv_cache, nv, live, *, enable_trace, sampling_params,
                      slot_remap):  # fmt: skip
        """A verify block computed by ordinary decode steps: sub-step j decodes every row still accepting, with its
        column-j token (the committed input, then its drafts) at its column-j position -> ids[r, j]; a row stops at
        its first rejection (ids[r, j] != tokens[r, j + 1]) or at its num_valid. Each sub-step is the plain decode, so
        the block is lossless by construction; 1 + max num_valid replays (the transitions' path; 1 without drafts)."""
        S, W = int(tokens.shape[0]), int(tokens.shape[1])
        ids = torch.full((S, W), -1, dtype=torch.int32)
        going = set(live)
        moves = None  # the step's slot moves (applied by sub-step 0; the runner confirms them after the step)
        # every replay draws from every row's device RNG: a seeded sampled row keeps the state after its LAST token of
        # the block (a plain step's one draw), restored after the sub-steps it sat out
        seeded, held = None, {}
        for j in range(W):
            rows = sorted(going)
            if not rows:
                break
            tok = torch.zeros((S, 1), dtype=torch.int64)
            pos = torch.full((S,), -1, dtype=torch.int64)
            for r in rows:
                tok[r, 0], pos[r] = int(tokens[r, j]), int(start_pos[r, j])
            out = self.decode_forward(tok, pos, page_table, kv_cache, enable_trace=enable_trace, read_from_device=True,
                                      sampling_params=sampling_params, slot_remap=slot_remap if j == 0 else None,
                                      reload_inputs=True)  # fmt: skip
            if not isinstance(out, torch.Tensor):
                out = self.process_decode_output_host(self.read_decode_output(out), is_tokens=sampling_params is not None)
            if j == 0:
                moves = self._last_moves
            col = out.reshape(S, -1)
            if col.shape[1] != 1:
                raise ValueError("a speculative verify step needs device sampling (sampling_params)")
            for r in rows:
                ids[r, j] = int(col[r, 0])
                if j >= nv.get(r, 0) or int(col[r, 0]) != int(tokens[r, j + 1]):
                    going.discard(r)
            self.counters["spec_plain_substeps"] = self.counters.get("spec_plain_substeps", 0) + 1
            if going:  # another sub-step follows
                if seeded is None:  # (after sub-step 0's slot moves)
                    seeded = [g for g in self.slot_table.slots if _device_seeded(self._params.get(g))]
                on = {self.gen_slot[r_] for r_ in going}
                done = [g for g in seeded if g not in held and g not in on]
                if done:
                    held.update(self.generator.sampler_rng(done) or {})
        self.generator.restore_sampler_rng(held)
        self._last_moves = moves
        self.counters["spec_plain_verifies"] = self.counters.get("spec_plain_verifies", 0) + 1
        return ids

    def propose_draft_tokens(self, num_drafts, committed, positions, counts, hidden=None):
        """Drafts for the next verify (runner contract): the lone live greedy request in generator slot 0 gets the K
        drafts the device already made (the last speculative step's, or a bootstrap replay after a plain step);
        every other configuration drafts nothing (num_valid 0: the plugin then sends ordinary steps)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        committed, positions, counts = (torch.as_tensor(v) for v in (committed, positions, counts))
        S, K = int(committed.shape[0]), int(num_drafts)
        ids = torch.zeros((S, K), dtype=torch.int32)
        nv = torch.zeros(S, dtype=torch.int32)
        live = [r for r in range(S) if int(positions[r].reshape(-1)[0]) >= 0]
        params = self._params.get(0)
        greedy = params is None or bool(getattr(params, "is_greedy", False))
        if len(live) != 1 or K != self.generator.spec_k() or self.gen_slot[live[0]] != 0 or not greedy:
            return DraftOutput(draft_token_ids=ids, num_valid=nv)
        if SPEC_SOLE_STATE:
            others = self._other_state_holders()
            if others:  # (a mid-prefill / held request: the step would write its rows) no drafts; leave a running one
                if self._spec is not None:
                    self._spec_leave()
                self.counters["spec_declines_state"] = self.counters.get("spec_declines_state", 0) + 1
                self._event("spec_decline", reason=f"other generator slots hold live state: {others}")
                return DraftOutput(draft_token_ids=ids, num_valid=nv)
        r = live[0]
        last = int(committed[r, int(counts[r]) - 1])
        if self._held is not None:  # (GLM53_SPEC_ASYNC) the runner first catches up with slot 0's held tokens
            return DraftOutput(draft_token_ids=ids, num_valid=nv)
        if self._spec is None:
            # (async scheduling) the plugin may already have submitted the next plain step for slot 0 (an overlapped
            # step) when it asks for drafts after the previous one: the mirror is then one position past the runner's
            # last committed token, and entering here would start the speculative state one step stale. A plain step
            # commits one token, so its committed position is positions[r][0]; no drafts until no step is in flight.
            st = self.slot_table.slots.get(0)
            if int(counts[r]) == 1 and st is not None and st.next_pos != int(positions[r].reshape(-1)[0]):
                self.counters["spec_declines_inflight"] = self.counters.get("spec_declines_inflight", 0) + 1
                return DraftOutput(draft_token_ids=ids, num_valid=nv)
            try:
                drafts = self.generator.spec_enter(self.kv_cache, 0)
            except ValueError as exc:  # no room for K+1 positions / not the only live slot: no drafts this step
                self._event("spec_decline", reason=str(exc))
                return DraftOutput(draft_token_ids=ids, num_valid=nv)
            st = self.slot_table.slots.get(0)
            self._spec = dict(bonus=last, drafts=list(drafts), pos=st.next_pos if st is not None else -1)
            self.counters["spec_enters"] = self.counters.get("spec_enters", 0) + 1
        elif self._pend is None and not self.generator.spec_room(self.kv_cache):
            # the next step would pass the context or the drafter's cache (GLM53_DF2_CACHE_CAP): leave now, the
            # request continues with plain steps (spec_enter declines from here on)
            self._spec_leave()
            self._event("spec_decline", reason="the next speculative step passes the context / the drafter's cache")
            return DraftOutput(draft_token_ids=ids, num_valid=nv)
        if self._spec["bonus"] != last:
            raise RuntimeError(f"propose: the runner committed {last} last, the device's next input is {self._spec}")
        ids[r] = torch.tensor(self._spec["drafts"], dtype=torch.int32)
        nv[r] = K
        return DraftOutput(draft_token_ids=ids, num_valid=nv)

    def read_decode_output(self, tt_out, async_read=False):
        """async_read=True (plugin async split): the generator enqueues the token copy behind the step and returns
        (pending tokens, events); the plugin waits on the events and calls process_decode_output_host, possibly after
        submitting the next step."""
        if isinstance(tt_out, torch.Tensor):  # logits / tokens already read back by the generator
            return (tt_out, []) if async_read else tt_out
        host = self.generator.read_decode_output(tt_out, async_read=async_read)
        if isinstance(host, tuple):
            return host
        return (host, []) if async_read else host

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if not isinstance(tt_out, torch.Tensor):
            from models.autoports.zai_org_glm_5_3_flash.tt.generator import PendingTokens

            if not isinstance(tt_out, PendingTokens):  # a DecodeOutput nobody read yet: read it now (blocking)
                tt_out = self.generator.read_decode_output(tt_out)
            tt_out = self.generator.process_decode_output_host(tt_out, is_tokens=is_tokens)
        if is_tokens:
            return tt_out.reshape(-1, 1).to(torch.int64)
        return tt_out.reshape(self.max_batch_size, 1, -1)

    # ------------------------------------------------------------------------------------------------- warmup
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device=False, **kwargs):
        """Plugin warm-up. Phase 1 (enable_trace=False): the generator prepares every prefill bucket without capturing.
        Phase 2 (enable_trace=True, trace_mode all; the plugin calls it after both phase-1 calls and before the decode
        capture): the generator captures its serving traces -- the per-bucket prefill traces and the decode + sampler
        traces (idempotent)."""
        self._check_cache(kv_cache)
        with _phase_profile("capture_prefill" if enable_trace else "warmup_prefill"):
            if enable_trace:
                self.generator.warmup_serving(kv_cache, phase="capture", prefill=True, decode=False)
                self._traces_captured = True
            else:
                self.generator.warmup_serving(kv_cache, phase="prepare", prefill=True, decode=False)
        self._reset_slots()
        self.startup.mark("capture_prefill_done" if enable_trace else "warmup_prefill_done")

    def warmup_model_decode(
        self, kv_cache, enable_trace, max_batch_size=None, num_blocks=None, can_sample_on_device=False, **kwargs
    ):
        """Phase 2 (enable_trace=True): capture the decode trace and both sampler traces (once; idempotent).
        `num_blocks` is the page-table width the runner will send; the generator accepts any width that covers a
        request's blocks."""
        self._check_cache(kv_cache)
        if max_batch_size is not None and int(max_batch_size) != self.max_batch_size:
            raise ValueError(f"decode warmup for {max_batch_size} rows, model built for {self.max_batch_size}")
        with _phase_profile("capture_decode" if enable_trace else "warmup_decode"):
            self.generator.warmup_serving(
                kv_cache, phase="capture" if enable_trace else "prepare", prefill=False, decode=True
            )
        self._traces_captured = self._traces_captured or bool(enable_trace)
        self._reset_slots()
        self._event("warmup_decode", wait=True, enable_trace=bool(enable_trace), report=self.trace_report())
        # the plugin's order: prefill prepare, decode prepare, prefill capture, decode capture (the last warm-up call)
        self.startup.mark("model_ready" if enable_trace else "warmup_done")
        if enable_trace and not self.startup.emitted:
            self._event("startup", wait=True, timeline=self.startup.emit())

    # --------------------------------------------------------------------------------------------- lifecycle
    def note_state_slots_moved(self, moves):
        """The runner reports an accepted gather ({old vLLM slot: new vLLM slot}); decode_forward already re-pointed
        the indirection for the same permutation, so this only checks that both sides agree."""
        moves = {int(k): int(v) for k, v in dict(moves).items()}
        if moves != (self._last_moves or {}):
            raise RuntimeError(f"plugin reports slot moves {moves}, the model applied {self._last_moves}")
        self._last_moves = None

    def release_request(self, slot, token_ids=None, **kwargs):
        """A request finished or was preempted: its generator slot stops holding live state. `token_ids` (the plugin's
        prompt + output ids of the request) keys the slot's trailing generation snapshot (prefix cache, class G); a
        preempted request (no ids) drops it."""
        gslot = self.gen_slot[int(slot)]
        if self._spec is not None and gslot == 0:  # the speculating request ends: its candidate rows are dropped
            self.generator.spec_abort(self.kv_cache)  # (a pre-enqueued step runs to completion, its output unread)
            self._spec = None
            self._pend = None
        if gslot == 0:
            self._held = None
        # The G commit below reads no device state: it keys the trailing snapshot that decode_forward copied at the
        # slot's last boundary crossing in PLAIN decode (decode_forward and prefill_forward leave speculation before
        # they run; the generator refuses a snapshot or export while slot 0 speculates). Spec steps take no trailing
        # snapshot, so after speculation the G entry stays at the last plain-decode boundary (valid, shorter).
        if self.prefix is not None:
            if token_ids is not None and self.prefix.trailing_position(gslot) > 0:
                b = self.generator._binding(self.kv_cache)
                pages = [int(p) for p in b.pt_shadow[gslot].tolist()]
                ent = self.prefix.commit_trailing(gslot, [int(t) for t in token_ids], pages)
                if ent is not None:
                    self.counters["prefix_generation_entries"] = self.counters.get("prefix_generation_entries", 0) + 1
            self.prefix.release_slot(gslot)
        self._slot_tokens.pop(gslot, None)
        self.slot_table.release(gslot)
        self._resident.discard(gslot)
        self._params.pop(gslot, None)
        self._prefix_states.pop(int(slot), None)  # a hit whose continuation never ran
        self.generator.release_slot(self.kv_cache, gslot)
        self.counters["slot_releases"] += 1

    def release_persistent_capture(self):
        """Plugin shutdown hook (mesh still open): release the traces and the model's device resources (socket pairs,
        drained queues) before the plugin closes the submeshes and the parent mesh."""
        self._event("shutdown", wait=True, report=self.trace_report())
        dma, self._hit_dma = getattr(self, "_hit_dma", None), None
        if dma is not None:  # the page-DMA kernels stop while the mesh is open
            logger.info(f"LMCache hit page-in stats: {dma.stats}")
            dma.close()
        close = getattr(self.generator, "close", None)
        if callable(close):
            close()
        else:
            self.generator.teardown()

    def trace_report(self) -> dict:
        """The generator's trace / host-work counters plus the adapter's own (Stage 9 mixed-shape evidence)."""
        rep = {
            "generator": self.generator.trace_report(),
            "adapter": dict(self.counters),
            "host_step": self.host_step_report(),
        }
        if self.prefix is not None:
            rep["prefix_cache"] = self.prefix.report()
        return rep


def _pooled_store_plan(model, pairs, pages=None):
    """Glm5NextForCausalLM.kv_pooled_store_plan (a module function: kv_read_pooled_pages calls it on host-test stand-ins
    of the model too)"""
    import numpy as np  # noqa: PLC0415

    from models.autoports.zai_org_glm_5_3_flash.tt.page_pool import half_of_block_page, pooled_halves  # noqa: PLC0415

    backend = model.generator.backend
    if getattr(backend, "pooled_paged", lambda kv: False)(model.kv_cache):
        if pages is None or len(pages) != len(pairs):
            raise ValueError("GLM53_DSA_PAGED_POOLS: kv_read_pooled_pages needs the pages' logical blocks")
        obm = model.generator.dsa_shard_map(model.kv_cache)
        phys = [obm.physical(int(b)) if hasattr(obm, "physical") else obm.phys.get(int(b)) for b in pages]
        if any(g is None for g in phys):
            raise ValueError(f"pooled keys of blocks without device content: {[b for b, g in zip(pages, phys) if g is None]}")

        def finish_paged(data):
            model.counters["pooled_page_reads"] = model.counters.get("pooled_page_reads", 0) + len(phys)
            return [half_of_block_page(d) for d in data]

        return backend.pooled_block_pages(model.kv_cache, phys), finish_paged
    g = [(model.gen_slot[int(v)], int(j)) for v, j in pairs]
    units = list(dict.fromkeys((gs, j // 2) for gs, j in g))
    at = {k: n for n, k in enumerate(units)}
    targets = [backend.pooled_unit_pages(model.kv_cache, gs, [u])[0] for gs, u in units]

    def finish(data):
        out = []
        for d in data:
            h = pooled_halves(d)  # [U, 2, 4,096]
            out.append(np.stack([h[at[(gs, j // 2)], j % 2] for gs, j in g]) if g else np.zeros((0, 4096), np.uint8))
        model.counters["pooled_page_reads"] = model.counters.get("pooled_page_reads", 0) + len(g)
        return out

    return targets, finish
