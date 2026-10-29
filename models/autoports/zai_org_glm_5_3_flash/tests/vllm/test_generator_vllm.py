# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-level tests of tt/generator_vllm.py over 06a's Glm53Generator + CPU MockModel (09a, host-only).

Decode-side tests drive the adapter through the plugin's own `TTAsyncDecodeController.submit_decode` (the
version-1 reload-command planner and submission path the server uses), feeding deliberately stale host tokens
and positions on steps where the contract says they are stale. Generated tokens are compared with 06a's independent
unpaged oracle (`MockModel.reference_greedy`), so a stale input, a wrong position, a wrong page or a wrong slot shows
up as a token mismatch; the generator's host-work counters show which refreshes happened.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tests.mock_model import VOCAB
from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import (
    build_mock_generator,
    prompt_ids,
    reference_greedy,
)
from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv
from models.autoports.zai_org_glm_5_3_flash.tt.generator import DecodeOutput
from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import (
    BLOCK_SIZE,
    Glm5NextForCausalLM,
    StateSlotTable,
    build_serving_cache_spec,
    generator_sampling_params,
    layer_kinds,
    load_context_contract,
    vllm_page_rows,
)

MLA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]


def _factory(dsa_mode, full_width=None, per_width=None, fused_sampling=None):
    return lambda *a, **kw: build_mock_generator(
        *a, dsa_mode=dsa_mode, full_width=full_width, per_width=per_width, fused_sampling=fused_sampling, **kw
    )


def _adapter(
    hf_config,
    *,
    max_batch=4,
    num_blocks=256,
    max_seq_len=8192,
    dsa_mode="unbounded",
    warm=True,
    all_layers=True,
    full_width=None,
    per_width=None,
    fused_sampling=None,
):
    model = Glm5NextForCausalLM.initialize_vllm_model(
        hf_config,
        None,
        max_batch,
        max_seq_len=max_seq_len,
        generator_factory=_factory(dsa_mode, full_width, per_width, fused_sampling),
    )
    n = 45 if all_layers else 11
    model.allocate_kv_cache_per_layer([((num_blocks, 1, BLOCK_SIZE, 512), torch.bfloat16, i) for i in range(n)])
    if warm:  # the plugin's TTModelRunner.warmup_model order
        model.warmup_model_prefill(kv_cache=model.kv_cache, enable_trace=False, can_sample_on_device=True)
        model.warmup_model_decode(
            kv_cache=model.kv_cache,
            enable_trace=False,
            max_batch_size=max_batch,
            num_blocks=64,
            can_sample_on_device=True,
        )
        model.warmup_model_prefill(kv_cache=model.kv_cache, enable_trace=True, can_sample_on_device=True)
        model.warmup_model_decode(
            kv_cache=model.kv_cache,
            enable_trace=True,
            max_batch_size=max_batch,
            num_blocks=64,
            can_sample_on_device=True,
        )
    return model


def _events(model, kind):
    return [e for e in model.generator.backend.events if e["event"] == kind]


class Pages:
    """vLLM-like block allocator: each request owns distinct physical blocks, handed out as it grows."""

    def __init__(self, num_blocks: int, width: int, start: int = 5):
        self.free = list(range(start, num_blocks))[::-1]
        self.width = width
        self.rows: dict[str, list[int]] = {}

    def ensure(self, req: str, tokens: int) -> list[int]:
        row = self.rows.setdefault(req, [])
        while len(row) * BLOCK_SIZE < tokens:
            row.append(self.free.pop())
        return row

    def release(self, req: str) -> None:
        self.free.extend(reversed(self.rows.pop(req)))

    def table(self, reqs) -> torch.Tensor:
        t = torch.zeros(len(reqs), self.width, dtype=torch.int32)
        for i, r in enumerate(reqs):
            if r is not None:
                row = self.rows[r]
                t[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
        return t


def _sampling(rows: int, temperature=0.0, seed=-1, **over):
    """The runner's TTSamplingParams for `rows` rows; `temperature` / `seed` are scalars or per-row tensors."""
    from vllm_tt_plugin.model_input import TTSamplingParams

    col = lambda v, dtype: torch.as_tensor(v, dtype=dtype).expand(rows).clone()
    fields = dict(
        temperature=col(temperature, torch.float32),
        top_k=torch.full((rows,), 32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=col(seed, torch.int64),
        num_logprobs=torch.full((rows,), -2),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )
    fields.update(over)
    return TTSamplingParams(**fields)


class MiniServe:
    """Drives the adapter the way the plugin runner does: front-packed prefill rows placed by `empty_slots`, then
    decode steps through the plugin's TTAsyncDecodeController (version-1 commands, async split). `slot_req[i]` is the
    request vLLM keeps in state slot i (= decode row i)."""

    def __init__(self, model: Glm5NextForCausalLM, pages: Pages, *, device_sampling: bool = True):
        from vllm_tt_plugin.async_decode import TTAsyncDecodeController

        self.model = model
        self.backend = model.generator.backend
        self.pages = pages
        self.S = model.max_batch_size
        self.device_sampling = device_sampling
        self.slot_req: list = [None] * self.S
        self.seq: dict[str, list[int]] = {}
        self.prompt: dict[str, list[int]] = {}
        self.out: dict[str, list[int]] = {}
        self.params: dict[str, dict] = {}  # request -> temperature / top_k / seed (default: greedy, unseeded)
        self.layout_changed = True
        self.runner = SimpleNamespace(
            model=model,
            trace_mode="all",
            kv_caches=model.kv_cache,
            request_specific_rope=False,
            note_decode_layout_consumed=lambda: None,
            note_decode_state_slots_settled=lambda: None,
        )
        self.controller = TTAsyncDecodeController(self.runner)
        self.plans = []

    def sampling(self, reqs) -> object:
        """The runner's per-row TTSamplingParams for `reqs` (None rows: greedy padding)."""
        rows = [self.params.get(r, {}) if r is not None else {} for r in reqs]
        return _sampling(
            len(rows),
            temperature=torch.tensor([float(q.get("temperature", 0.0)) for q in rows]),
            top_k=torch.tensor([int(q.get("top_k", 32)) for q in rows]),
            seed=torch.tensor([int(q.get("seed", -1)) for q in rows]),
        )

    def ref(self, req: str, n: int | None = None) -> list[int]:
        return reference_greedy(self.backend, self.prompt[req], len(self.out[req]) if n is None else n)

    def prefill(self, reqs: dict, slots: list, chunks: dict | None = None):
        names = list(reqs)
        width = max(len(p) for p in reqs.values())
        tokens = torch.zeros(len(names), width, dtype=torch.int32)
        starts, ends = [], []
        for i, r in enumerate(names):
            self.prompt.setdefault(r, list(reqs[r]))
            self.seq.setdefault(r, list(reqs[r]))
            s, e = (chunks or {}).get(r, (0, len(reqs[r])))
            tokens[i, : len(reqs[r])] = torch.tensor(reqs[r], dtype=torch.int32)
            self.pages.ensure(r, e)
            starts.append(s)
            ends.append(e)
        final = all(e == len(reqs[r]) for r, e in zip(names, ends))
        out = self.model.prefill_forward(
            tokens=tokens,
            page_table=self.pages.table(names),
            kv_cache=self.model.kv_cache,
            enable_trace=False,
            prompt_lens=torch.tensor(ends),
            start_pos=torch.tensor(starts),
            empty_slots=slots,
            sampling_params=self.sampling(names) if (final and self.device_sampling) else None,
        )
        for i, (r, slot) in enumerate(zip(names, slots)):
            self.slot_req[slot] = r
            if ends[i] == len(reqs[r]):  # final chunk: the first output token
                tok = int(out[i]) if out.dim() == 1 else int(out[i, -1].argmax())
                self.out.setdefault(r, []).append(tok)
                self.seq[r].append(tok)
        self.layout_changed = True
        return out

    def finish(self, req: str) -> None:
        """vLLM finished a request: release its slot (plugin release_request) and its blocks."""
        slot = self.slot_req.index(req)
        self.model.release_request(slot)
        self.slot_req[slot] = None
        self.pages.release(req)
        self.layout_changed = True

    def decode(self, *, stale_hosts: bool = True, slot_remap=None, sampling=None):
        reqs = list(self.slot_req)
        sub = self.submit(stale_hosts=stale_hosts, slot_remap=slot_remap, sampling=sampling)
        self.complete(sub, reqs)
        return sub

    def submit(self, *, stale_hosts: bool = True, slot_remap=None, sampling=None, in_flight: int = 0):
        """Plan and submit one decode step (no host read yet). `in_flight` = steps submitted but not completed: their
        tokens are not in `seq` yet (vLLM async scheduling), so positions and page growth count them."""
        S = self.S
        reqs = self.slot_req
        tokens = torch.zeros(S, 1, dtype=torch.int32)
        pos = torch.full((S,), -1, dtype=torch.int32)
        for i, r in enumerate(reqs):
            if r is not None:
                tokens[i, 0] = self.seq[r][-1]
                pos[i] = len(self.seq[r]) - 1 + in_flight
                self.pages.ensure(r, len(self.seq[r]) + in_flight)
        model_input = SimpleNamespace(
            input_tokens=tokens,
            input_positions=pos,
            block_tables=self.pages.table(reqs),
            block_tables_per_group=[self.pages.table(reqs)],
            block_tables_per_layer=None,
            unpadded_batch_size=sum(r is not None for r in reqs),
            tt_sampling_params=sampling or self.sampling(reqs),
            perform_device_sampling=self.device_sampling,
            prompt_tokens=None,
            output_tokens=None,
            decode_layout_changed=self.layout_changed,
            num_valid_drafts=None,
            accepted_counts=None,
            spec_mode=None,
            slot_remap=slot_remap,
        )
        plan = self.controller.plan_decode_reload(model_input)
        if stale_hosts and not plan.reload_inputs:
            # The contract: these are one step behind and must not be read. Make them wrong on purpose.
            model_input.input_tokens = torch.full_like(tokens, 4242)
            model_input.input_positions = torch.where(pos >= 0, pos - 1, pos)
        sub = self.controller.submit_decode(model_input, read_from_device=False, async_read=True)
        self.plans.append(sub.reload_plan)
        self.layout_changed = False
        return sub

    def complete(self, sub, reqs):
        """The runner's finalize: wait on the read events, format on host, append the step's tokens."""
        final = self.controller.finalize_decode(sub)
        if self.device_sampling:
            toks = final.tt_out.reshape(-1)
        else:
            toks = final.tt_out[:, -1, :].argmax(-1)
        for i, r in enumerate(reqs):
            if r is not None:
                t = int(toks[i])
                self.out[r].append(t)
                self.seq[r].append(t)


# ------------------------------------------------------------------------------------------------ cache spec
def test_cache_spec_from_layer_types(text_config, evidence):
    kinds = layer_kinds(text_config)
    assert [i for i, k in enumerate(kinds) if k == "mla"] == MLA_LAYERS
    assert kinds.count("kda") == 34
    spec = build_serving_cache_spec(text_config, num_blocks=16416, max_slots=32)
    mla = [s for s in spec.layers if s.kind == "mla"]
    kda = [s for s in spec.layers if s.kind == "kda"]
    assert all(s.shape == (16416, 1, 64, 512) and s.dtype == torch.bfloat16 for s in mla)
    assert all(s.aux_shape == (16416, 1, 16, 128) for s in mla)
    assert all(s.shape == (32, 64, 128, 128) and s.dtype == torch.float32 for s in kda)
    assert all(s.aux_shape == (32, 3, 24576) and s.aux_dtype == torch.bfloat16 for s in kda)
    summary = spec.summary()
    assert summary["latent_bytes_per_token_per_layer"] == 1024
    assert summary["pooled_key_bytes_per_token_per_layer"] == 64
    per_slot = 34 * (64 * 128 * 128 * 4 + 3 * 24576 * 2)
    assert summary["kda_bytes_per_slot"] == per_slot
    evidence(summary=summary, kda_bytes_per_slot_mib=per_slot / 2**20)


def test_layer_type_spellings_and_unknown_kinds(text_config, expect_error):
    legacy = list(text_config.layer_types)
    for spelling in ("deepseek_sparse_attention", "indexed_attention"):
        cfg = SimpleNamespace(
            num_hidden_layers=45, layer_types=[t if t == "linear_attention" else spelling for t in legacy]
        )
        assert [i for i, k in enumerate(layer_kinds(cfg)) if k == "mla"] == MLA_LAYERS
    with expect_error(ValueError, "unsupported layer type 'sliding_attention'"):
        layer_kinds(SimpleNamespace(num_hidden_layers=2, layer_types=["linear_attention", "sliding_attention"]))
    with expect_error(ValueError, "layer_types has 1 entries for 2"):
        layer_kinds(SimpleNamespace(num_hidden_layers=2, layer_types=["linear_attention"]))


def test_vllm_kv_cache_spec_is_one_mla_spec_per_dsa_layer(hf_config, evidence, expect_error, monkeypatch):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from models.autoports.zai_org_glm_5_3_flash.tt import prefix_cache as PC

    # get_kv_cache_spec records vLLM's prefix-caching switch on the class (the adapter's PrefixStore follows it):
    # restore it afterwards, the other adapter tests here build the adapter without the spec query (store off)
    monkeypatch.setattr(Glm5NextForCausalLM, "_vllm_prefix_caching", False, raising=False)
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False, block_size=64, cache_dtype="auto"),
        model_config=SimpleNamespace(hf_config=hf_config),
    )
    spec = Glm5NextForCausalLM.get_kv_cache_spec(cfg)
    assert list(spec) == [f"model.layers.{i}.self_attn" for i in MLA_LAYERS]
    assert all(isinstance(s, MLAAttentionSpec) for s in spec.values())
    page = next(iter(spec.values())).page_size_bytes
    assert page == 64 * 1024
    evidence(layers=len(spec), page_size_bytes=page, bytes_per_token_per_layer=page // 64)
    # prefix caching: accepted with the slot-state snapshots on (srv/prefixcache, the default), refused with them off
    apc = SimpleNamespace(**{**vars(cfg.cache_config), "enable_prefix_caching": True})
    assert PC.enabled()
    assert list(
        Glm5NextForCausalLM.get_kv_cache_spec(SimpleNamespace(cache_config=apc, model_config=cfg.model_config))
    ) == list(spec)
    assert Glm5NextForCausalLM._vllm_prefix_caching is True
    monkeypatch.setenv(PC.ENABLED_ENV, "0")
    for field, value, match in (
        ("block_size", 128, "--block-size 64"),
        ("enable_prefix_caching", True, "prefix caching"),
        ("cache_dtype", "fp8", "bf16"),
    ):
        bad = SimpleNamespace(**{**vars(cfg.cache_config), field: value})
        with expect_error(ValueError, match):
            Glm5NextForCausalLM.get_kv_cache_spec(SimpleNamespace(cache_config=bad, model_config=cfg.model_config))


def test_allocate_kv_cache_accepts_vllm_layer_counts_and_rejects_bad_geometry(hf_config, expect_error):
    for all_layers in (True, False):
        model = _adapter(hf_config, all_layers=all_layers, warm=False)
        assert model.cache_spec.mla_layers == MLA_LAYERS and model.cache_spec.num_blocks == 256
        assert model.generator._binding(model.kv_cache).max_batch == 4  # the generator registered vLLM's cache
    model = Glm5NextForCausalLM.initialize_vllm_model(
        hf_config, None, 4, max_seq_len=8192, generator_factory=_factory("unbounded")
    )
    with expect_error(ValueError, r"expected \[N, 1, 64, 512\] bf16"):
        model.allocate_kv_cache_per_layer([((256, 1, 32, 512), torch.bfloat16, i) for i in range(11)])
    with expect_error(ValueError, "expected 45 .* or 11"):
        model.allocate_kv_cache_per_layer([((256, 1, 64, 512), torch.bfloat16, i) for i in range(12)])
    legacy = model.allocate_kv_cache((256, 1, 64, 512), torch.bfloat16, 45)
    assert legacy is model.kv_cache


def test_generator_pool_is_sized_from_vllm_num_blocks(hf_config, evidence):
    """vLLM owns the page pool: the adapter passes vLLM's num_blocks to `allocate_kv_cache`, so the dense-exact
    generator (2051 positions per request) takes block ids anywhere in vLLM's 1M-token pool (16416 blocks), here
    from its top end."""
    model = _adapter(hf_config, num_blocks=16416, max_seq_len=1048576, dsa_mode="dense_exact")
    b = model.generator._binding(model.kv_cache)
    assert b.num_blocks == 16416 and model.generator.supported_context == 2051
    serve = MiniServe(model, Pages(16416, 16384, start=16400))  # rows as wide as the runner's (1M / 64 entries)
    serve.prefill({"r": prompt_ids(21, 300)}, slots=[1])
    for _ in range(4):
        serve.decode()
    assert serve.pages.rows["r"] == list(range(16400, 16405))
    assert serve.out["r"] == serve.ref("r")
    evidence(vllm_num_blocks=16416, generator_pool_blocks=b.num_blocks, request_blocks=serve.pages.rows["r"])


def test_initialize_requires_the_generator_serving_api(hf_config, expect_error):
    """A generator without the serving API (06a contract.md section 2.1: release_slot, allocate_kv_cache(...,
    num_blocks=), warmup_serving, ...) is refused at startup, not at the first finished request."""

    def proxy_factory(hide=(), allocate=None):
        def factory(*a, **kw):
            gen = build_mock_generator(*a, dsa_mode="unbounded", **kw)

            class Proxy:
                def __getattr__(self, name):
                    if name in hide:
                        raise AttributeError(name)
                    if name == "allocate_kv_cache" and allocate is not None:
                        return allocate
                    return getattr(gen, name)

            return Proxy()

        return factory

    init = lambda f: Glm5NextForCausalLM.initialize_vllm_model(hf_config, None, 4, generator_factory=f)
    with expect_error(TypeError, r"missing \['release_slot'\]"):
        init(proxy_factory(hide=("release_slot",)))
    with expect_error(TypeError, r"missing \['warmup_serving', 'remap_slots'\]"):
        init(proxy_factory(hide=("warmup_serving", "remap_slots")))
    with expect_error(TypeError, r"allocate_kv_cache\(\.\.\., num_blocks=\)"):
        init(proxy_factory(allocate=lambda max_batch=None, max_seq_len=None: None))


def test_kv_pool_budget(monkeypatch, evidence, expect_error):
    """Default pool: max_num_seqs x min(max_model_len, 32768) tokens, floored at the generator's page-table width
    (blocks_per_seq pages: one max_model_len request + the padded tail of its last prefill chunk), so the 1-seq
    profile allocates without GLM53_VLLM_KV_POOL_TOKENS (2051 -> 40 pages = 2560 tokens, not 2051 -> 34 pages)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import blocks_per_seq

    f = Glm5NextForCausalLM.get_max_tokens_all_users
    assert f(max_model_len=2051, max_num_seqs=1) == blocks_per_seq(2051) * 64 == 2560
    assert f(max_model_len=2051, max_num_seqs=32) == 32 * 2051
    assert f(max_model_len=1048576, max_num_seqs=32) == blocks_per_seq(1048576) * 64 == 1049088
    assert f(max_model_len=1048576, max_num_seqs=1) == 1049088
    assert f(max_model_len=4096, max_num_seqs=32) == 32 * 4096
    assert f(max_model_len=65536, max_num_seqs=32) == 32 * 32768
    monkeypatch.setenv(gv.KV_POOL_TOKENS_ENV, str(2 * 1048576))
    assert f(max_model_len=1048576, max_num_seqs=32) == 2 * 1048576
    monkeypatch.setenv(gv.KV_POOL_TOKENS_ENV, "4096")
    with expect_error(ValueError, "cannot hold one max_model_len=1048576"):
        f(max_model_len=1048576, max_num_seqs=32)
    evidence(default_pool_tokens_1M_32seqs=1049088, default_pool_tokens_2051_1seq=2560)


def test_initialize_validates_the_serving_configuration(hf_config, expect_error, monkeypatch):
    def init(b=4, **kw):
        kw.setdefault("generator_factory", _factory("unbounded"))
        return Glm5NextForCausalLM.initialize_vllm_model(hf_config, None, b, **kw)

    model = init()
    assert model.max_seq_len == 1048576 == load_context_contract().served_max_model_len
    assert model.generator.max_seq_len == 1048576
    with expect_error(ValueError, "DP 1"):
        init(tt_data_parallel=2)
    with expect_error(ValueError, r"outside \[1, 32\]"):
        init(b=33)
    with expect_error(ValueError, "advertised context"):
        init(max_seq_len=1048577)
    with expect_error(ValueError, "optimizations"):
        init(optimizations="fastest")
    wrong = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=46, hidden_size=4096, vocab_size=154880, kv_lora_rank=512)
    )
    with expect_error(ValueError, "not the zai-org/GLM-5.3-Flash"):
        Glm5NextForCausalLM.initialize_vllm_model(wrong, None, 4, generator_factory=_factory("unbounded"))
    # the class capability was computed at import: it must still be what the contract (GLM53_CONTEXT_CONTRACT) says
    monkeypatch.setitem(Glm5NextForCausalLM.model_capabilities, "tt_request_context_limit", 0)
    with expect_error(ValueError, "imported against another contract"):
        init()


def test_initialize_refuses_a_contract_beyond_the_generator(hf_config, expect_error, lifted_contract):
    """Once the contract lifts the request limit (sparse DSA), a generator still in the dense-exact regime would
    be asked for positions it cannot compute: refused at startup."""
    # the message names the contract file by its path: GLM53_CONTEXT_CONTRACT (e.g. a ladder rung's
    # context_contract_<rung>.json) or the model dir's doc/context_contract.json
    with expect_error(ValueError, r"update \S*context_contract[^/\s]*\.json current_supported_context"):
        Glm5NextForCausalLM.initialize_vllm_model(hf_config, None, 4, generator_factory=_factory("dense_exact"))
    model = Glm5NextForCausalLM.initialize_vllm_model(hf_config, None, 4, generator_factory=_factory("unbounded"))
    assert model.max_seq_len == 1048576 and Glm5NextForCausalLM.model_capabilities["tt_request_context_limit"] == 0


def test_context_contract_and_request_limit(evidence, tmp_path, expect_error):
    contract = load_context_contract()
    assert contract.advertised == contract.target == 1048576
    assert contract.supported == 2051 and contract.supported_by_kind["sparse_dsa"] == 2051
    assert contract.request_context_limit == 2051
    assert Glm5NextForCausalLM.model_capabilities["tt_request_context_limit"] == 2051
    data = {"hf_advertised_context": 1048576, "target_context": 1048576, "current_supported_context": 1048576}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    assert gv.load_context_contract(p).request_context_limit == 0  # lifted (sparse DSA): no request limit
    # a hard physical limit lowers the served max_model_len (vllm_serving.served_max_model_len <= supported) instead
    # of a request limit; without that value, or above the supported context, the contract does not load
    data.update(current_supported_context=65536, capability_reduction={"hard_physical_limit": True})
    p.write_text(json.dumps(data))
    with expect_error(ValueError, "needs vllm_serving.served_max_model_len"):
        gv.load_context_contract(p)
    data["vllm_serving"] = {"served_max_model_len": 65537}
    p.write_text(json.dumps(data))
    with expect_error(ValueError, "needs vllm_serving.served_max_model_len"):
        gv.load_context_contract(p)
    data["vllm_serving"] = {"served_max_model_len": 65536}
    p.write_text(json.dumps(data))
    hard = gv.load_context_contract(p)
    assert hard.served_max_model_len == 65536 and hard.request_context_limit == 0
    evidence(
        served_max_model_len=contract.served_max_model_len,
        request_context_limit=contract.request_context_limit,
        reduction_reason=contract.reduction.get("reduction_reason"),
    )


def test_served_context_contract_is_the_dram_bound(evidence):
    """doc/context_contract.json = the served line's contract: the served value (65,536 since int27c: the largest value
    validated on device for the served configuration) below the advertised 1,048,576 for a device-DRAM reason with
    capacity evidence (what check_context_contract.py requires). Since the round-2 review ruling (program
    2026-10-22T22:58:11Z) it is labelled the IMPLEMENTATION limit of the replicated latent pool, NOT a hard physical limit:
    the adapter keeps the advertised value as its max_model_len bound and serves the stage-scope request limit
    tt_request_context_limit = the served value (every Stage 9 launch passes --max-model-len 65536)."""
    path = gv.MODEL_DIR / "doc" / "context_contract.json"
    data = json.loads(path.read_text())
    c = gv.load_context_contract(path)
    served = data["vllm_serving"]["served_max_model_len"]
    assert c.advertised == c.target == 1048576 and c.supported == served >= 32768
    assert not c.reduction["hard_physical_limit"] and c.reduction["limit_kind"].startswith("implementation")
    assert c.served_max_model_len == c.target and c.request_context_limit == served
    assert data["reduction_reason"] == c.reduction["reduction_reason"] == "device_dram"
    assert data["capacity_evidence"] and c.reduction["capacity_evidence"]
    assert c.supported_by_kind["sparse_dsa"] == served
    assert data["tested_context_by_layer_kind"]["sparse_dsa"]["decode_context"] == served
    evidence(served_max_model_len=c.served_max_model_len, request_context_limit=c.request_context_limit,
             reduction_reason=data["reduction_reason"], limit_kind=c.reduction["limit_kind"],
             evidence_keys=sorted(data["capacity_evidence"]))


def test_served_context_contract_initializes_at_its_bound(hf_config, expect_error, monkeypatch, tmp_path):
    """Under the served contract (implementation limit, not a hard physical limit) the adapter accepts a launch up to
    the advertised context and declares the served value as its request limit: the plugin refuses a request whose
    prompt + max_tokens exceeds it; the Stage 9 launch passes --max-model-len = the served value."""
    path = gv.MODEL_DIR / "doc" / "context_contract.json"
    load = gv.load_context_contract
    monkeypatch.setattr(gv, "load_context_contract", lambda p=path: load(p))
    served = json.loads(path.read_text())["vllm_serving"]["served_max_model_len"]
    assert load(path).request_context_limit == served
    monkeypatch.setitem(Glm5NextForCausalLM.model_capabilities, "tt_request_context_limit", load(path).request_context_limit)

    def init(**kw):
        return Glm5NextForCausalLM.initialize_vllm_model(hf_config, None, 4, generator_factory=_factory("unbounded"), **kw)

    model = init(max_seq_len=served)
    assert model.max_seq_len == served == model.generator.max_seq_len
    assert init(max_seq_len=16384).max_seq_len == 16384
    assert Glm5NextForCausalLM.model_capabilities["tt_request_context_limit"] == served


# --------------------------------------------------------------------------------------------------- prefill
@pytest.mark.parametrize("length", [1, 31, 33, 63, 65, 129, 130, 131, 2047, 2049, 2051])
def test_adapter_nonaligned_prompt_padding_positions(hf_config, length, evidence):
    model = _adapter(hf_config)
    prompt = prompt_ids(length, length)
    serve = MiniServe(model, Pages(256, 64))
    first = serve.prefill({"r": prompt}, slots=[2])
    assert tuple(first.shape) == (1,)  # device-sampled first token
    chunks = _events(model, "prefill_chunk")
    assert [c["n"] for c in chunks] == [min(2048, length - o) for o in range(0, length, 2048)]
    assert [c["start"] for c in chunks] == list(range(0, length, 2048))
    assert all(c["bucket"] == math.ceil(c["n"] / 64) * 64 for c in chunks)
    assert model.gen_slot[2] == 0  # the generator declares decode buckets: the fresh request takes generator slot 0
    assert all(c["slot"] == 0 for c in chunks)
    for _ in range(3):
        serve.decode()
    assert serve.out["r"] == serve.ref("r")
    evidence(length=length, chunks=[(c["start"], c["n"], c["bucket"]) for c in chunks], matches_oracle=True)


def test_adapter_nonaligned_prompt_chunked_continuation(hf_config, evidence, expect_error):
    """vLLM chunked prefill: [0, 512) then [512, 1100) then the decode; the adapter slices each chunk out of the
    token history and passes logical lengths and absolute starts."""
    model = _adapter(hf_config)
    prompt = prompt_ids(5, 1100)
    serve = MiniServe(model, Pages(256, 64))
    logits = serve.prefill({"r": prompt}, slots=[1], chunks={"r": (0, 512)})
    assert tuple(logits.shape) == (1, 1, VOCAB)  # intermediate chunk: host logits path, never sampled
    serve.prefill({"r": prompt}, slots=[1], chunks={"r": (512, 1100)})
    for _ in range(2):
        serve.decode()
    assert serve.out["r"] == serve.ref("r")
    chunks = [(c["start"], c["n"]) for c in _events(model, "prefill_chunk")]
    assert chunks == [(0, 512), (512, 588)]
    evidence(chunks=chunks)
    with expect_error(ValueError, "not a multiple of the 64-token latent page"):
        serve.prefill({"q": prompt_ids(6, 300)}, slots=[2], chunks={"q": (100, 300)})


def test_prefill_continuation_in_the_wrong_slot_fails_loudly(hf_config, expect_error):
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    prompt = prompt_ids(9, 700)
    serve.prefill({"r": prompt}, slots=[3], chunks={"r": (0, 512)})
    with expect_error(RuntimeError, rf"request's state is at \[{model.gen_slot[3]}\]"):
        serve.prefill({"r": prompt}, slots=[1], chunks={"r": (512, 700)})


def test_dense_exact_regime_stops_at_2051(hf_config, expect_error):
    """The current generator (dense-exact DSA) refuses position 2051; the plugin's request limit keeps vLLM short of
    it, this shows the generator-side guard is a ValueError, not a silent wrong answer."""
    model = _adapter(hf_config, num_blocks=128, dsa_mode="dense_exact")
    serve = MiniServe(model, Pages(128, 64))
    serve.prefill({"r": prompt_ids(3, 2040)}, slots=[0])
    for _ in range(10):  # positions 2040..2049
        serve.decode()
    assert serve.out["r"] == serve.ref("r")
    serve.decode()  # consumes position 2050 (kv_len 2051)
    with expect_error(ValueError, "decode position 2051 >= supported context 2051"):
        serve.decode()


# --------------------------------------------------------------------------------------------- slot lifecycle
def test_slot_lifecycle_empty_slots_remap_release(hf_config, evidence):
    model = _adapter(hf_config, max_batch=4)
    serve = MiniServe(model, Pages(256, 64))
    a, b = prompt_ids(1, 70), prompt_ids(2, 200)
    serve.prefill({"a": a, "b": b}, slots=[2, 0])  # the scheduler put the new requests in slots 2 and 0
    serve.decode()
    assert sorted(model.slot_table.slots) == [0, 1]  # packed into the lowest generator slots
    # rows move: a (slot 2) to row 0, b (slot 0) to row 1 -- remap[i] = the slot row i reads
    serve.slot_req = ["a", "b", None, None]
    serve.layout_changed = True
    before = list(model.gen_slot)
    serve.decode(slot_remap=torch.tensor([2, 0, 1, 3], dtype=torch.int32))
    model.note_state_slots_moved({2: 0, 0: 1, 1: 2})
    assert model.gen_slot == [before[j] for j in (2, 0, 1, 3)]  # an indirection: generator state did not move
    for _ in range(3):
        serve.decode()
    assert serve.out["a"] == serve.ref("a") and serve.out["b"] == serve.ref("b")
    assert _events(model, "copy_slot") == []
    serve.finish("b")
    assert 0 not in model.slot_table.slots and model.counters["slot_releases"] == 1
    evidence(gen_slot=model.gen_slot, remaps=model.counters["remaps"], copy_slot_calls=0)


def test_slot_remap_cycle_is_an_indirection(hf_config, expect_error, evidence):
    """Two live requests swap rows (a cyclic remap). Glm53Generator.remap_slots refuses cycles without a scratch slot;
    the adapter never calls it -- the remap only re-points vLLM slot -> generator slot, so all 4 slots can be live."""
    model = _adapter(hf_config, max_batch=4)
    serve = MiniServe(model, Pages(256, 64))
    reqs = {f"r{i}": prompt_ids(10 + i, 40 + 30 * i) for i in range(4)}
    serve.prefill(reqs, slots=[0, 1, 2, 3])
    serve.decode()
    serve.slot_req = ["r1", "r0", "r3", "r2"]
    serve.layout_changed = True
    serve.decode(slot_remap=torch.tensor([1, 0, 3, 2], dtype=torch.int32))
    for _ in range(3):
        serve.decode()
    for r in reqs:
        assert serve.out[r] == serve.ref(r), r
    assert _events(model, "copy_slot") == []
    with expect_error(ValueError, "cyclic slot_remap needs a free scratch slot"):
        model.generator.remap_slots(model.kv_cache, {0: 1, 1: 0, 2: 3, 3: 2})  # control: the generator's own path
    evidence(remap=[1, 0, 3, 2], live_slots=4, copy_slot_calls=0)


def test_release_request_frees_the_generator_slot_for_block_reuse(hf_config, expect_error, evidence):
    """After a request finishes vLLM reuses its blocks for a new request in another slot. release_request ->
    Glm53Generator.release_slot; without it the generator still counts the old slot as live and its stale
    page-table row collides with the new one (control)."""
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(1, 60), "keep": prompt_ids(2, 90)}, slots=[0, 1])
    serve.decode()
    serve.finish("a")  # releases slot 0 and a's blocks
    serve.prefill({"b": prompt_ids(3, 50)}, slots=[2])  # b reuses a's block
    assert serve.pages.rows["b"][0] in (5, 6)
    serve.decode()
    serve.decode()
    assert serve.out["b"] == serve.ref("b") and serve.out["keep"] == serve.ref("keep")
    # control: the same block reuse without the release is refused by the generator's overlap check
    model2 = _adapter(hf_config)
    model2.pack_slots = False  # keep b in its own generator slot (2): packing would put it into a's unreleased slot 0
    serve2 = MiniServe(model2, Pages(256, 64))
    serve2.prefill({"a": prompt_ids(1, 60)}, slots=[0])
    serve2.decode()
    serve2.slot_req[0] = None
    serve2.pages.release("a")  # vLLM freed the blocks, but nobody released the slot
    with expect_error(RuntimeError, "reuses page 5, still recorded as live at slot 0"):  # the adapter's mirror
        serve2.prefill({"b": prompt_ids(3, 50)}, slots=[2])
    model2.slot_table.release(0)  # adapter-side release only: the generator still holds slot 0 live
    with expect_error(ValueError, "mapped by live slots 0 and 2"):  # the generator's own overlap check
        serve2.prefill({"b": prompt_ids(3, 50)}, slots=[2])
    evidence(reused_block=serve.pages.rows["b"][0], released_slots=model.counters["slot_releases"])


def test_state_slot_table_rejects_bad_reloads(expect_error):
    t = StateSlotTable(4)
    t.begin_prefill(0, key=11, start=0, end=64)
    with expect_error(RuntimeError, "decode row 0 consumes position 63"):
        t.check_decode_inputs([0, 1, 2, 3], [63, -1, -1, -1], [11, 0, 0, 0])
    t.check_decode_inputs([0, 1, 2, 3], [64, -1, -1, -1], [11, 0, 0, 0])
    with expect_error(RuntimeError, "still recorded as live at slot 0"):
        t.begin_prefill(1, key=11, start=0, end=10)
    with expect_error(RuntimeError, "fresh prefill into slot 0, which still holds a live request"):
        t.check_prefill(0, key=12, start=0, end=10)
    assert t.resident_positions([3, 0], {0}) == [-1, 64]
    t.move(0, 2)
    assert t.slots[2].key == 11 and 0 not in t.slots


def test_prefill_mirror_commits_only_after_the_generator_accepts(hf_config, expect_error):
    """A prefill the generator refuses (here: a device error) leaves the slot mirror untouched, so the request can be
    retried in the same slot; committing first would leave a phantom live request there."""
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    prompt = prompt_ids(31, 90)
    original = model.generator.prefill_forward

    def failing(tokens, **kw):
        raise RuntimeError("device error during prefill")

    model.generator.prefill_forward = failing
    with expect_error(RuntimeError, "device error during prefill"):
        serve.prefill({"a": prompt}, slots=[0])
    assert model.slot_table.slots == {}
    model.generator.prefill_forward = original
    serve.prefill({"a": prompt}, slots=[0])
    for _ in range(3):
        serve.decode()
    assert serve.out["a"] == serve.ref("a")


def test_fresh_prefill_into_a_live_slot_is_refused(hf_config, expect_error):
    """A new request admitted into a vLLM slot whose request was never released (a runner bookkeeping error) would
    silently replace that request's KDA state: refused; after release_request the same admission works."""
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(32, 70)}, slots=[1])
    serve.decode()
    with expect_error(RuntimeError, f"fresh prefill into slot {model.gen_slot[1]}, which still holds a live request"):
        serve.prefill({"b": prompt_ids(33, 40)}, slots=[1])
    serve.finish("a")
    serve.prefill({"b": prompt_ids(33, 40)}, slots=[1])
    serve.decode()
    assert serve.out["b"] == serve.ref("b")


# --------------------------------------------------------------------------------------- decode input contract
def test_stale_token_steady_decode_uses_resident_token(hf_config, evidence):
    """Steady device-sampled decode: the plugin sends reload_inputs=False and stale host tokens (here deliberately
    wrong); the adapter passes token -1 (the token the device sampled) and the resident positions, so the generator
    writes nothing and the sampled token feeds the next replay on device."""
    model = _adapter(hf_config)
    gen = model.generator
    serve = MiniServe(model, Pages(256, 64))
    p = {"a": prompt_ids(3, 40), "b": prompt_ids(4, 90)}
    serve.prefill(p, slots=[0, 1])
    before = dict(gen.counters)
    for _ in range(8):
        serve.decode(stale_hosts=True)
    steady = [pl for pl in serve.plans if not pl.reload_inputs]
    assert len(steady) == 7 and serve.plans[0].reload_inputs
    for r in p:
        assert serve.out[r] == serve.ref(r)
    delta = {k: gen.counters[k] - before[k] for k in gen.counters}
    sampler_replays = 0 if gen.fused_sampling else 8  # a decode trace that samples replays no sampler trace
    assert delta["decode_trace_replays"] == 8 and delta["sampling_trace_replays"] == sampler_replays
    assert delta["token_refresh"] == 0 and delta["pos_refresh"] == 1 and delta["page_table_refresh"] == 0
    evidence(decode_steps=8, steady_steps=len(steady), counter_delta=delta)


def test_stale_token_negative_control_detects_a_reloaded_stale_token(hf_config):
    """Control for the test above: if a stale host token WERE loaded on a steady step (the right positions, the wrong
    token), the output diverges from the oracle, so the positive test can see that bug class."""
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(3, 40)}, slots=[0])
    serve.decode()
    original = model.generator.decode_forward

    def forced(tokens, start_pos, **kw):
        if bool((tokens < 0).all()):  # a steady step: load a wrong token instead of the device's
            tokens = torch.full_like(tokens, 4242)
        return original(tokens, start_pos, **kw)

    model.generator.decode_forward = forced
    serve.decode()
    serve.decode()
    assert serve.out["a"] != serve.ref("a")


def test_current_position_advances_once_per_step_on_device(hf_config, evidence):
    model = _adapter(hf_config)
    gen = model.generator
    io = gen._binding(model.kv_cache).io
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(8, 62)}, slots=[0])
    positions = []
    before = gen.counters["pos_refresh"]
    for _ in range(5):
        serve.decode()
        positions.append(int(gen.backend.read(io.cur_pos)[0]))
    # after the step consuming position k the device position is k + 1 (one plus_one per emitted token)
    assert positions == [63, 64, 65, 66, 67]
    assert gen.counters["pos_refresh"] - before == 1
    assert serve.out["a"] == serve.ref("a")
    assert model.slot_table.slots[0].next_pos == 67
    evidence(device_positions=positions, position_refreshes=gen.counters["pos_refresh"] - before)


def test_current_position_after_layout_change_reloads_authoritative_inputs(hf_config):
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(1, 30)}, slots=[0])
    serve.decode()
    serve.decode()
    serve.prefill({"b": prompt_ids(2, 50)}, slots=[1])  # prefill -> decode transition: reload
    sub = serve.decode()
    assert sub.reload_plan.reload_inputs and sub.reload_plan.reset_sampling_state
    serve.decode()
    assert serve.out["a"] == serve.ref("a") and serve.out["b"] == serve.ref("b")


def test_page_table_refresh_only_when_allocation_changes(hf_config, evidence):
    """Unchanged page table: no copy. A request crossing a 64-token page gets a new block: page-table-only refresh
    (tokens / positions stay resident) and the next token reads the new page."""
    model = _adapter(hf_config)
    gen = model.generator
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(12, 60)}, slots=[0])
    before = dict(gen.counters)
    for _ in range(10):
        serve.decode()
    plans = [(p.reload_inputs, p.reload_page_table) for p in serve.plans]
    page_only = [i for i, (ri, rp) in enumerate(plans) if rp]
    assert plans[0] == (True, False)
    assert len(page_only) == 1, plans  # the block for positions 64..127 appears once
    delta = {k: gen.counters[k] - before[k] for k in gen.counters}
    assert delta["page_table_refresh"] == 1 and delta["token_refresh"] == 0
    assert serve.out["a"] == serve.ref("a")
    evidence(plans=plans, page_table_only_step=page_only[0], counter_delta=delta)


def test_page_table_stale_copy_negative_control(hf_config, expect_error):
    """Control: drop the page-table-only refresh. The generator's row for the request then covers only the blocks
    of the last full reload (vLLM's null-block padding reaches the generator as -1: vllm_page_rows), so the step at
    position 64 is refused instead of writing into a block the request does not own."""
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(12, 60)}, slots=[0])
    original = model.generator.decode_forward

    def drop_page_refresh(tokens, start_pos, *, page_table=None, **kw):
        stale = page_table if bool((tokens >= 0).any()) else None  # only the reload step's table reaches the device
        return original(tokens, start_pos, page_table=stale, **kw)

    model.generator.decode_forward = drop_page_refresh
    with expect_error(ValueError, "position 64 is outside its page-table coverage"):
        for _ in range(10):
            serve.decode()
    assert len(serve.out["a"]) == 5 and serve.out["a"] == serve.ref("a")  # positions 60..63 were still right


def test_vllm_null_block_padding_becomes_generator_padding():
    """vLLM's BlockPool reserves block 0 as its null block and never hands it to a request; the runner pads block
    tables with it. The adapter passes that padding as -1 (the generator's row terminator)."""
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=16, enable_caching=False, hash_block_size=64)
    assert pool.null_block.block_id == 0 and 0 not in [blk.block_id for blk in pool.get_new_blocks(15)]
    table = torch.tensor([[7, 9, 0, 0], [0, 0, 0, 0]], dtype=torch.int32)
    assert vllm_page_rows(table).tolist() == [[7, 9, -1, -1], [-1, -1, -1, -1]]
    assert table.tolist() == [[7, 9, 0, 0], [0, 0, 0, 0]]  # the runner's tensor is not modified


# ------------------------------------------------------------------------------------------------- async split
def test_async_decode_split_submit_read_process(hf_config, evidence):
    assert Glm5NextForCausalLM.model_capabilities["supports_async_decode"] is True
    assert Glm5NextForCausalLM.decode_input_update_contract == 1
    model = _adapter(hf_config)
    gen = model.generator
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(4, 33)}, slots=[0])
    S = model.max_batch_size
    tokens = torch.zeros(S, 1, dtype=torch.int32)
    tokens[0, 0] = serve.seq["a"][-1]
    pos = torch.tensor([33] + [-1] * (S - 1), dtype=torch.int32)
    reads = gen.counters["token_readback"]
    handle = model.decode_forward(
        tokens=tokens,
        start_pos=pos,
        page_table=serve.pages.table(["a", None, None, None]),
        kv_cache=model.kv_cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=_sampling(S),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    assert isinstance(handle, DecodeOutput)  # device-resident, not read
    assert gen.counters["token_readback"] == reads
    host, events = model.read_decode_output(handle, async_read=True)
    assert events == [] and gen.counters["token_readback"] == reads + 1
    io = gen._binding(model.kv_cache).io
    resident = (gen.backend.read(io.tokens).clone(), gen.backend.read(io.cur_pos).clone())
    model.read_decode_output(handle, async_read=True)  # readback must not mutate decode state
    assert torch.equal(resident[0], gen.backend.read(io.tokens))
    assert torch.equal(resident[1], gen.backend.read(io.cur_pos))
    toks = model.process_decode_output_host(host, is_tokens=True)
    assert tuple(toks.shape) == (S, 1) and toks.dtype == torch.int64
    assert int(toks[0, 0]) == reference_greedy(gen.backend, prompt_ids(4, 33), 2)[1]
    evidence(handle=type(handle).__name__, token_readbacks=gen.counters["token_readback"] - reads)


def test_async_decode_contract_rejects_legacy_and_incoherent_commands(hf_config, expect_error):
    model = _adapter(hf_config)
    S = model.max_batch_size
    common = dict(
        tokens=torch.zeros(S, 1, dtype=torch.int32),
        start_pos=torch.full((S,), -1),
        page_table=torch.zeros(S, 64, dtype=torch.int32),
        kv_cache=model.kv_cache,
    )
    with expect_error(TypeError, "reset_batch"):
        model.decode_forward(**common, reset_batch=True)
    with expect_error(ValueError, "reload_page_table must be False"):
        model.decode_forward(**common, reload_inputs=True, reload_page_table=True)
    with expect_error(ValueError, "reset_sampling_state requires reload_inputs"):
        model.decode_forward(**common, reload_inputs=False, reset_sampling_state=True)
    with expect_error(ValueError, "trace_mode all"):
        model.decode_forward(**common, enable_trace=False)
    with expect_error(ValueError, "non-identity slot_remap needs reload_inputs"):
        model.decode_forward(**common, reload_inputs=False, slot_remap=torch.tensor([1, 0, 2, 3]))
    with expect_error(TypeError, "num_valid_drafts"):
        model.decode_forward(**common, num_valid_drafts=torch.zeros(S))
    with expect_error(ValueError, "penalties are host-sampled"):
        model.decode_forward(**common, prompt_tokens=torch.zeros(S, 4), output_tokens=torch.zeros(S, 4))


def test_async_decode_host_sampling_mode_returns_logits(hf_config):
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64), device_sampling=False)
    serve.prefill({"a": prompt_ids(6, 45)}, slots=[1])
    for _ in range(4):
        serve.decode()
    assert all(pl.reload_inputs for pl in serve.plans)  # host sampling reloads every step
    assert serve.out["a"] == serve.ref("a")


# ---------------------------------------------------------------------------------------------- warm-up / trace
def test_warmup_two_phase_prepares_then_captures_once(hf_config, evidence, expect_error):
    model = _adapter(hf_config, warm=False)
    gen = model.generator
    kw = dict(kv_cache=model.kv_cache, can_sample_on_device=True)
    model.warmup_model_prefill(enable_trace=False, **kw)
    model.warmup_model_decode(enable_trace=False, max_batch_size=4, num_blocks=64, **kw)
    assert gen.counters["decode_trace_captures"] == gen.counters["sampling_trace_captures"] == 0
    with expect_error(RuntimeError, "none were captured: launch with tt.trace_mode all"):  # a trace_mode none launch
        MiniServe(model, Pages(256, 64)).prefill({"early": prompt_ids(1, 10)}, slots=[0])
    model.warmup_model_prefill(enable_trace=True, **kw)  # trace_mode all: the serving traces are captured here
    widths = len(gen.decode_buckets)  # one decode trace per bucket width: (1, 2, 4) at 4 slots
    assert widths == 3 and gen.counters["decode_trace_captures"] == widths
    assert gen.counters["sampling_trace_captures"] == (0 if gen.fused_sampling else 2)
    model.warmup_model_decode(enable_trace=True, max_batch_size=4, num_blocks=64, **kw)
    assert gen.counters["decode_trace_captures"] == widths, "a repeated warm-up must reuse the captured traces"
    compiled = len(gen.backend.device.compile_log)
    serve = MiniServe(model, Pages(256, 64))
    for n in (1, 2047, 2048, 700):  # every prefill bucket was prepared: nothing compiles behind the live traces
        serve.prefill({f"r{n}": prompt_ids(n, n)}, slots=[0])
        model.release_request(0)
        serve.pages.release(f"r{n}")
    assert len(gen.backend.device.compile_log) == compiled
    evidence(captures={"decode": widths, "sampling": 2}, programs_compiled_in_warmup=compiled)


def test_adapter_has_no_sampling_or_model_math():
    """Delegation contract: no sampler, no logits math, no ttnn ops in the adapter."""
    src = Path(inspect.getsourcefile(gv)).read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for forbidden in (r"\bargmax\b", r"softmax", r"\btopk\b", r"multinomial", r"import ttnn", r"ttnn\."):
        assert not re.search(forbidden, code), forbidden


def test_sampling_params_translation(expect_error):
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams

    p = generator_sampling_params(
        _sampling(3, temperature=0.7, seed=5, top_k=torch.tensor([32, -1, 1]), top_p=torch.tensor([0.9, 1.0, 0.5])), 4
    )
    approx = lambda sp: (pytest.approx(sp.temperature), sp.top_k, pytest.approx(sp.top_p), sp.seed)
    as_tuple = lambda sp: (sp.temperature, sp.top_k, sp.top_p, sp.seed)  # vLLM's tensors are float32
    assert as_tuple(p[0]) == approx(SamplingParams(temperature=0.7, top_k=32, top_p=0.9, seed=5))
    assert as_tuple(p[1]) == approx(SamplingParams(temperature=0.7, top_k=None, top_p=1.0, seed=5))
    assert p[2].is_greedy  # top_k 1
    assert p[3] == SamplingParams()  # padded row beyond the list: greedy, inactive anyway
    assert generator_sampling_params(_sampling(2, seed=-1), 2)[0].seed is None
    with expect_error(ValueError, "repetition_penalty"):
        generator_sampling_params(_sampling(2, repetition_penalty=torch.tensor([1.0, 1.1])), 2)


def test_prefill_passes_device_sampling_params_through(hf_config):
    model = _adapter(hf_config)
    seen = {}
    original = model.generator.prefill_forward

    def spy(tokens, **kw):
        seen.update(kw)
        return original(tokens, **kw)

    model.generator.prefill_forward = spy
    pages = Pages(256, 64)
    pages.ensure("a", 77)
    out = model.prefill_forward(
        tokens=torch.tensor([prompt_ids(5, 77)]),
        page_table=pages.table(["a"]),
        kv_cache=model.kv_cache,
        prompt_lens=torch.tensor([77]),
        start_pos=torch.tensor([0]),
        empty_slots=[0],
        sampling_params=_sampling(1),
    )
    assert len(seen["sampling_params"]) == 1 and seen["sampling_params"][0].is_greedy
    assert seen["empty_slots"] == [0] and seen["start_pos"] == [0] and seen["prompt_lens"] == [77]
    assert int(out[0]) == reference_greedy(model.generator.backend, prompt_ids(5, 77), 1)[0]


# --------------------------------------------------------------------------------- sampling-state commands
def test_sampling_params_upload_only_when_commanded(hf_config, evidence):
    """reload_sampling_params is a command: steady steps keep the parameters uploaded at the last commanded reload
    (here the runner's per-step values are deliberately different and must not reach the device); the next
    commanded reload uploads them. reset_sampling_state does not re-seed continuing requests (seed writes happen
    only at admission)."""
    model = _adapter(hf_config)
    gen = model.generator
    b = gen._binding(model.kv_cache)
    serve = MiniServe(model, Pages(256, 64))
    serve.params["a"] = dict(temperature=0.9, top_k=20, seed=11)
    serve.prefill({"a": prompt_ids(41, 50)}, slots=[2])
    seeds_after_admission = gen.counters["seed_writes"]
    first = serve.decode()
    assert first.reload_plan.reload_sampling_params and first.reload_plan.reset_sampling_state
    g = model.gen_slot[2]
    assert b.slots[g].params.temperature == pytest.approx(0.9) and b.slots[g].params.seed == 11
    refresh = gen.counters["sampling_param_refresh"]
    stale = _sampling(4, temperature=0.3, seed=99)
    for _ in range(3):
        sub = serve.decode(sampling=stale)
        assert not sub.reload_plan.reload_inputs and not sub.reload_plan.reload_sampling_params
    assert gen.counters["sampling_param_refresh"] == refresh
    assert b.slots[g].params.temperature == pytest.approx(0.9)
    serve.layout_changed = True
    sub = serve.decode(sampling=stale)
    assert sub.reload_plan.reload_sampling_params and sub.reload_plan.reset_sampling_state
    assert gen.counters["sampling_param_refresh"] == refresh + 1
    assert b.slots[g].params.temperature == pytest.approx(0.3)
    assert gen.counters["seed_writes"] == seeds_after_admission, "a continuing request was re-seeded"
    evidence(
        steady_steps_without_upload=3,
        commanded_uploads=gen.counters["sampling_param_refresh"] - refresh,
        seed_writes_after_admission=gen.counters["seed_writes"] - seeds_after_admission,
    )


# ------------------------------------------------------------------------------------- pow2 decode buckets
def _pow2(n: int) -> int:
    w = 1
    while w < n:
        w *= 2
    return w


def test_pack_slots_lowest_free_admission_and_compaction(hf_config, evidence, expect_error):
    """With bucketed decode widths (DECISIONS 2026-10-17 ~22:15Z; the generator declares `decode_buckets`) live state
    must sit in the lowest generator slots. Fresh requests take the lowest free generator slot (a relabelling, nothing
    moves) wherever vLLM put them; when requests finish, survivors above the covering bucket width move down through
    Glm53Generator.remap_slots before the next decode step, and the generator replays the narrower trace. Greedy
    outputs stay equal to the oracle throughout."""
    model = _adapter(hf_config, max_batch=8)
    assert model.pack_slots and model.generator.decode_buckets == (1, 2, 4, 8)
    serve = MiniServe(model, Pages(512, 64))
    vslots = [7, 6, 5, 4, 3, 2]
    reqs = {f"r{i}": prompt_ids(60 + i, 30 + 23 * i) for i in range(6)}
    serve.prefill(reqs, slots=vslots)  # the scheduler picked high slots
    assert sorted(model.gen_slot[v] for v in vslots) == [0, 1, 2, 3, 4, 5]
    assert {c["slot"] for c in _events(model, "prefill_chunk")} == {0, 1, 2, 3, 4, 5}
    widths = []
    original = model.generator.decode_forward

    def spy(tokens, start_pos, **kw):
        live = sorted(model.slot_table.slots)
        widths.append((len(live), max(live) + 1))
        return original(tokens, start_pos, **kw)

    model.generator.decode_forward = spy
    for _ in range(2):
        serve.decode()
    owner = lambda g: serve.slot_req[model.gen_slot.index(g)]
    for r in [owner(0), owner(1), owner(2)]:  # the requests in the three lowest generator slots finish
        serve.finish(r)
    serve.decode()  # reloading step: generator slots 4, 5 move to 0, 1 -> width 4 covers the 3 live requests
    moves = [(c["src"], c["dst"]) for c in _events(model, "copy_slot")]
    assert moves == [(4, 0), (5, 1)] and model.counters["compaction_moves"] == 2
    for _ in range(3):
        serve.decode()
    serve.prefill({"r6": prompt_ids(70, 45)}, slots=[7])  # the lowest free generator slot is 2
    assert model.gen_slot[7] == 2
    for _ in range(3):
        serve.decode()
    for r in serve.out:
        assert serve.out[r] == serve.ref(r), r
    assert all(top <= _pow2(n) for n, top in widths), widths
    replays = dict(model.generator._binding(model.kv_cache).width_replays)
    assert replays[4] > 0 and replays[8] > 0  # 6 live -> width 8; 3-4 live after compaction -> the width-4 trace
    assert [ev["width"] for ev in _events(model, "decode_step")] == [8, 8] + [4] * 7
    with expect_error(RuntimeError, "fresh prefill into slot 2, which still holds a live request"):
        serve.prefill({"x": prompt_ids(71, 20)}, slots=[7])  # packing never hides an unreleased slot
    evidence(
        compaction_moves=moves,
        live_vs_highest_slot=widths,
        width_replays=replays,
        slot_swaps=model.counters["slot_swaps"],
        gen_slot=model.gen_slot,
    )


def test_per_width_backend_compacts_but_never_moves_a_seeded_request(hf_config, evidence, monkeypatch):
    """tt/per_width.py: only widths 1 / 2 / 4 have their own traces (the generator's buckets are those + max_batch, not
    every pow2) and the backend's copy_slot re-seeds the destination's sampler row (per-core PRNG state cannot be
    copied). Compaction still packs greedy requests into the lowest slots, but an explicitly seeded sampled request is
    never moved, so it samples the same stream as alone. Control: allowing the move restarts its stream."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import decode_widths

    assert decode_widths(32, (1, 2, 4)) == (1, 2, 4, 32) and decode_widths(8, (4, 1, 8, 16)) == (1, 4, 8)
    seeded = {"temperature": 1.0, "top_k": 8, "seed": 1234}

    def run(churn: bool):
        model = _adapter(hf_config, max_batch=8, per_width=(1, 2, 4))
        assert model.pack_slots and model.generator.decode_buckets == (1, 2, 4, 8)
        serve = MiniServe(model, Pages(512, 64))
        serve.params["s"] = seeded
        if not churn:
            serve.prefill({"s": prompt_ids(90, 41)}, slots=[0])
            for _ in range(24):
                serve.decode()
            return serve, model
        reqs = {"g0": prompt_ids(91, 33), "s": prompt_ids(90, 41), "g2": prompt_ids(92, 70)}
        reqs.update({"g3": prompt_ids(93, 12), "g4": prompt_ids(94, 50)})
        serve.prefill(reqs, slots=[0, 1, 2, 3, 4])  # generator slots 0..4, the seeded request in slot 1
        for _ in range(3):
            serve.decode()
        for r in ("g0", "g2", "g3"):
            serve.finish(r)
        for _ in range(5):  # live: s (slot 1), g4 (slot 4) -> width 2: g4 moves to slot 0, s stays
            serve.decode()
        serve.finish("g4")
        for _ in range(16):  # s alone in slot 1: width 2 (a move to slot 0 would give width 1 and a fresh RNG)
            serve.decode()
        return serve, model

    alone, _ = run(churn=False)
    serve, model = run(churn=True)
    assert serve.out["s"] == alone.out["s"] and len(serve.out["s"]) == 25
    for r in ("g0", "g2", "g3", "g4"):
        assert serve.out[r] == serve.ref(r), r
    moves = [(c["src"], c["dst"]) for c in _events(model, "copy_slot")]
    assert moves == [(4, 0)]
    assert [ev["width"] for ev in _events(model, "decode_step")] == [8] * 3 + [2] * 21
    with monkeypatch.context() as m:  # control: the seeded request is allowed to move
        m.setattr(gv, "_device_seeded", lambda params: False)
        control, cmodel = run(churn=True)
    cmoves = [(c["src"], c["dst"]) for c in _events(cmodel, "copy_slot")]
    assert cmoves == [(4, 0), (1, 0)] and control.out["s"] != alone.out["s"]
    evidence(
        decode_buckets=list(model.generator.decode_buckets),
        moves=moves,
        widths=[ev["width"] for ev in _events(model, "decode_step")],
        seeded_tokens=serve.out["s"],
        control_moves=cmoves,
        control_first_divergence=next(i for i, (a, b) in enumerate(zip(control.out["s"], alone.out["s"])) if a != b),
    )


@pytest.mark.parametrize("per_width", [(8, 16), (1, 2, 4, 8, 16)], ids=["int8-w8-16", "w1-2-4-8-16"])
def test_native_width_buckets_serve_a_lone_user_at_the_smallest_width(hf_config, evidence, per_width):
    """The Stage 14b stack's native width buckets (int8: GLM53_STACK_NATIVE_WIDTHS=8,16 on the native 32-row step:
    width_bucketed_decode (8, 16); the next int adds w <= 4 (program 2026-10-21 01:47:14Z); the decode trace
    samples, copy_slot moves the RNG row) through the unchanged adapter (program 2026-10-21 00:06:09Z): a lone user
    decodes at the smallest width, ten live users at 16; when requests finish, compaction moves a straggler from a high
    slot into the low slots and the width drops back to the smallest; greedy tokens equal the reference throughout.
    Device: resident1 (int8) replayed the width-8 trace for its lone users. The (1, 2, 4, 8, 16) case models the next
    int's stack (this tree's stack_model keeps only 4 < w < 32): the unchanged adapter puts a lone user on width 1 (the
    intermediate widths 2 / 4 / 8 are not exercised here; per-width compaction: test_per_width_backend_compacts_*)."""
    model = _adapter(hf_config, max_batch=32, per_width=per_width, fused_sampling=True)
    model.generator.backend.copy_slot_keeps_sampler_rng = True  # the stack: the RNG row moves with the slot
    assert model.pack_slots and model.generator.decode_buckets == (*per_width, 32)
    w0 = per_width[0]
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(201, 20)}, slots=[0])
    for _ in range(3):
        serve.decode()
    more = {f"u{i}": prompt_ids(210 + i, 12 + i) for i in range(1, 10)}
    serve.prefill(more, slots=list(range(1, 10)))  # ten live users in generator slots 0..9
    for _ in range(3):
        serve.decode()
    for r in ["a", *(f"u{i}" for i in range(1, 9))]:
        serve.finish(r)
    for _ in range(4):  # u9 alone in slot 9: compaction moves it to slot 0 -> the smallest width
        serve.decode()
    widths = [ev["width"] for ev in _events(model, "decode_step")]
    moves = [(c["src"], c["dst"]) for c in _events(model, "copy_slot")]
    assert widths == [w0] * 3 + [16] * 3 + [w0] * 4, widths
    assert moves == [(9, 0)], moves
    for r in ["a", *more]:
        assert serve.out[r] == serve.ref(r), r
    evidence(decode_buckets=list(model.generator.decode_buckets), widths=widths, moves=moves)


# ------------------------------------------------------------------------------------------- Stage 9 (device wiring)
def test_full_width_backend_keeps_one_trace_width_and_never_moves_state(hf_config, evidence):
    """The 8-chip pipeline replays one full-width decode trace for every width (tt/model.py width_bucketed_decode
    False). The generator then keeps a single width, the adapter does not pack slots, and request churn never copies
    KDA state (each move would copy ~141 MiB on device and re-seed the slot's RNG for no saving). Same churn as the
    packing test, plus a fresh request in a freed slot; greedy outputs equal the oracle."""
    model = _adapter(hf_config, max_batch=8, full_width=True)
    gen = model.generator
    assert gen.decode_buckets == (8,) and not model.pack_slots
    serve = MiniServe(model, Pages(512, 64))
    vslots = [7, 6, 5, 4, 3, 2]
    reqs = {f"r{i}": prompt_ids(60 + i, 30 + 23 * i) for i in range(6)}
    serve.prefill(reqs, slots=vslots)
    assert [model.gen_slot[v] for v in vslots] == vslots  # identity indirection: nothing relabelled
    for _ in range(2):
        serve.decode()
    for r in ("r0", "r1", "r2"):
        serve.finish(r)
    for _ in range(4):
        serve.decode()
    serve.prefill({"r6": prompt_ids(70, 45)}, slots=[6])  # a freed slot, its blocks reused
    for _ in range(3):
        serve.decode()
    for r in serve.out:
        assert serve.out[r] == serve.ref(r), r
    assert _events(model, "copy_slot") == [] and model.counters["compaction_moves"] == 0
    assert model.counters["slot_swaps"] == 0
    b = gen._binding(model.kv_cache)
    assert sorted(b.decode_traces) == [8] and gen.counters["decode_trace_captures"] == 1
    assert {ev["width"] for ev in _events(model, "decode_step")} == {8}
    evidence(
        decode_buckets=list(gen.decode_buckets),
        decode_trace_captures=gen.counters["decode_trace_captures"],
        width_replays=dict(b.width_replays),
        copy_slot_calls=0,
        adapter=dict(model.counters),
    )


def test_async_split_read_survives_the_next_submission(hf_config, evidence):
    """vLLM async scheduling submits step N+1 before it finalizes step N. The async read is enqueued behind step N,
    so it yields step N's tokens; completing it after step N+1 was submitted must neither return N+1's tokens nor
    overwrite the host copy of the slots' (device-resident) next inputs. Three requests, 70 overlapped steady steps
    across their 64-token page boundaries, against the oracle; the overlapped steps refresh no token and no position."""
    model = _adapter(hf_config, max_batch=4, full_width=True)
    gen = model.generator
    serve = MiniServe(model, Pages(256, 64))
    reqs = {"a": prompt_ids(80, 61), "b": prompt_ids(81, 127), "c": prompt_ids(82, 5)}
    serve.prefill(reqs, slots=[0, 1, 2])
    serve.decode()  # the reloading step after admission (the runner drains before a layout change)
    before = dict(gen.counters)
    order = list(serve.slot_req)
    pending = serve.submit()
    overlapped = 0
    for _ in range(70):
        nxt = serve.submit(in_flight=1)  # step N+1 goes in before step N's tokens reach the host
        serve.complete(pending, order)
        pending = nxt
        overlapped += 1
    serve.complete(pending, order)
    for r in reqs:
        assert serve.out[r] == serve.ref(r), r
    plans = serve.plans[-71:]
    assert not any(p.reload_inputs for p in plans), "steady overlapped steps must not reload host inputs"
    assert gen.counters["token_refresh"] == before["token_refresh"]
    assert gen.counters["pos_refresh"] == before["pos_refresh"]
    reads = [e for e in gen.backend.events if e["event"] in ("read_async", "finish_read")]
    assert reads and all(e["buffer"] == "sampled" for e in reads)
    assert sum(e["event"] == "read_async" for e in reads) == sum(e["event"] == "finish_read" for e in reads)
    evidence(
        overlapped_submissions=overlapped,
        page_table_refreshes=gen.counters["page_table_refresh"] - before["page_table_refresh"],
        token_refresh_delta=0,
        pos_refresh_delta=0,
        tokens_per_request={r: len(serve.out[r]) for r in reqs},
    )


def test_shutdown_releases_traces_then_closes_the_backend(hf_config, evidence):
    """The plugin's shutdown hook (mesh still open) releases every trace and then the backend's device resources
    (on the pipeline: drained queues, socket pairs, quiesced parent) before the plugin closes the mesh."""
    model = _adapter(hf_config)
    gen = model.generator
    b = gen._binding(model.kv_cache)
    traces = [*b.decode_traces.values(), *b.sampling_traces.values()]
    assert traces and not any(t.released for t in traces)
    model.release_persistent_capture()
    assert all(t.released for t in traces) and not b.decode_traces and not b.sampling_traces
    assert [e["event"] for e in gen.backend.events if e["event"] == "backend_close"] == ["backend_close"]
    evidence(released_traces=len(traces), backend_close=1)


def test_device_factory_needs_the_2x4_mesh_and_its_graph_descriptor(hf_config, expect_error, monkeypatch):
    """Without a factory override the adapter builds the device pipeline on the plugin's mesh: it refuses any mesh
    but the 2x4 parent and a launch without the 8-chip graph descriptor, before the backend loads anything."""
    monkeypatch.delenv(gv.GENERATOR_FACTORY_ENV, raising=False)
    mesh = lambda rows, cols: SimpleNamespace(shape=(rows, cols))
    with expect_error(ValueError, r"mesh \(1, 8\).*2x4"):
        Glm5NextForCausalLM.initialize_vllm_model(hf_config, mesh(1, 8), 32, max_seq_len=2051)
    monkeypatch.delenv(gv.MESH_GRAPH_DESC_ENV, raising=False)
    with expect_error(ValueError, "TT_MESH_GRAPH_DESC_PATH=''"):
        Glm5NextForCausalLM.initialize_vllm_model(hf_config, mesh(2, 4), 32, max_seq_len=2051)
    line = gv.MODEL_DIR / "mesh" / "glm8_1x8_line_mesh_graph_descriptor.textproto"
    monkeypatch.setenv(gv.MESH_GRAPH_DESC_ENV, str(line))
    with expect_error(ValueError, "glm8_2x4_mesh_graph_descriptor"):
        Glm5NextForCausalLM.initialize_vllm_model(hf_config, mesh(2, 4), 32, max_seq_len=2051)
    with expect_error(ValueError, "one precision policy"):
        Glm5NextForCausalLM.initialize_vllm_model(
            hf_config, None, 4, max_seq_len=2051, optimizations="accuracy", generator_factory=_factory("dense_exact")
        )


def test_startup_timeline_is_printed_written_and_exported(hf_config, evidence, monkeypatch, tmp_path, capsys):
    """Program 05:48:56Z (a): the serving startup timeline (EngineCore start -> model init -> generator built (load)
    -> KV cache -> warm-up preparation -> trace captures = model ready) is printed once at model ready, written to
    $GLM53_STARTUP_TIMELINE with Stage 12's fast-start timeline merged in, logged as the adapter event `startup`, and
    set on the Prometheus gauge glm53_startup_seconds{phase} when multiprocess metrics are on."""
    events, out, fast = tmp_path / "events.jsonl", tmp_path / "startup.json", tmp_path / "fast_start.json"
    fast.write_text(json.dumps({"weight_source": "tensors", "load_s": 1.5}))
    (tmp_path / "prom").mkdir()
    monkeypatch.setenv(gv.EVENTS_ENV, str(events))
    monkeypatch.setenv(gv.STARTUP_TIMELINE_ENV, str(out))
    monkeypatch.setenv(gv.FAST_START_TIMELINE_ENV, str(fast))
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path / "prom"))
    model = _adapter(hf_config, max_batch=4)
    rep = json.loads(out.read_text())
    phases = rep["phases_s"]
    # + Stage 12 fast start 2's halves (a6634422691, restart_time.md): prefill / decode prepare, prefill / decode capture
    halves = ["warmup_prefill", "warmup_decode", "capture_prefill", "capture_decode"]
    assert list(phases) == ["engine_start", "load", "kv_cache", "warmup", "capture", "total"] + halves, phases
    assert all(v >= 0 for v in phases.values())
    assert phases["load"] + phases["kv_cache"] + phases["warmup"] + phases["capture"] <= phases["total"] + 1e-3
    assert abs(phases["warmup_prefill"] + phases["warmup_decode"] - phases["warmup"]) <= 2e-3, phases
    assert abs(phases["capture_prefill"] + phases["capture_decode"] - phases["capture"]) <= 2e-3, phases
    assert rep["fast_start"] == {"weight_source": "tensors", "load_s": 1.5} and rep["prometheus"] is True
    assert rep["max_num_seqs"] == 4 and rep["num_blocks"] == 256 and "generator_factory" in rep
    assert [e["event"] for e in map(json.loads, events.read_text().splitlines())].count("startup") == 1
    assert capsys.readouterr().out.count("[glm53-serving] startup ") == 1
    assert gv._STARTUP_GAUGE.labels(phase="total")._value.get() == phases["total"]
    # a second capture call (the plugin re-warming) does not emit again
    model.warmup_model_decode(kv_cache=model.kv_cache, enable_trace=True, max_batch_size=4, num_blocks=64)
    assert [e["event"] for e in map(json.loads, events.read_text().splitlines())].count("startup") == 1
    evidence(phases_s=phases, info={k: v for k, v in rep.items() if k not in ("phases_s", "marks_utc", "fast_start")})


def test_context_by_users_is_advertised(hf_config, monkeypatch, tmp_path):
    """Program 2026-10-20 16:46:18Z: the endpoint advertises the max ISL + OSL per concurrent-user count — vLLM's
    page pool (minus the null block) split N ways in whole 64-token pages, capped by max_model_len and the request
    context limit; in the startup timeline and on the Prometheus gauge glm53_max_context_tokens{concurrent_users}."""
    # a 576K-token pool (9,000 usable pages) under a 576K max_model_len: pool-limited per user
    assert gv.context_by_users(9001, 576_000, 0, 32) == {
        1: 576_000, 2: 288_000, 4: 144_000, 8: 72_000, 16: 562 * 64, 32: 281 * 64
    }  # fmt: skip
    # the request context limit (dense-exact DSA 2051) caps every count; max_num_seqs itself is always listed
    assert gv.context_by_users(9001, 576_000, 2051, 12) == {1: 2051, 2: 2051, 4: 2051, 8: 2051, 12: 2051}
    assert gv.context_by_users(9001, 4096, 0, 1) == {1: 4096}
    (tmp_path / "prom").mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path / "prom"))
    model = _adapter(hf_config, max_batch=4)
    limit = Glm5NextForCausalLM.model_capabilities["tt_request_context_limit"]
    want = gv.context_by_users(256, model.max_seq_len, limit, 4)
    assert model.startup.info["context_by_users"] == want and set(want) == {1, 2, 4}
    assert gv._CONTEXT_GAUGE.labels(concurrent_users="4")._value.get() == want[4]


def test_kv_connector_views_expose_the_latent_pages(hf_config, monkeypatch, expect_error):
    """LMCache A1 (vllm-tt-plugin glm53-lmcache-a1 `resolve_connector_kv_views`): the adapter hands a KV connector
    {`model.layers.{i}.self_attn`: layer i's paged latent} for exactly vLLM's MLA spec names, from the backend
    (ModelBackend.latent_pages); a backend whose layers differ from the spec, or a call before the cache exists, is
    refused; a connector hit's continuation without the request's KDA state (start > 0, no slot holds it) is refused
    by the slot table instead of serving a recurrence that never saw the prefix."""
    model = _adapter(hf_config, max_batch=4)
    backend = model.generator.backend
    with expect_error(ValueError, "vLLM's KV spec"):  # the mock's own 3-layer stack (L1 = its sparse_dsa)
        model.kv_cache_connector_views()
    mla = [i for i, k in enumerate(gv.layer_kinds(gv.text_config_of(hf_config))) if k == "mla"]
    assert len(mla) == 11
    lat = backend.latent_pages(model.kv_cache)[1]
    assert lat.shape[0] == model.kv_cache.num_blocks == 256
    monkeypatch.setattr(backend, "latent_pages", lambda kv: {i: lat for i in mla})
    views = model.kv_cache_connector_views()
    assert list(views) == [f"model.layers.{i}.self_attn" for i in mla] and all(v is lat for v in views.values())
    monkeypatch.setattr(model, "kv_cache", None)
    with expect_error(RuntimeError, "before allocate_kv_cache_per_layer"):
        model.kv_cache_connector_views()
    with expect_error(RuntimeError, "must run in the slot that holds the request's KDA state"):
        gv.StateSlotTable(4).check_prefill(0, key=5, start=192, end=200)


def test_kda_checkpoint_prefix_hit_continues_like_a_full_prefill(hf_config, expect_error, evidence):
    """LMCache join (vllm-tt-plugin KDACheckpointStore over the adapter's export_kda_state / import_kda_state):
    request A prefills to the page boundary 128 and the plugin exports its slot's KDA state there. Request B with the
    same prompt is a connector hit of 128 tokens: its blocks receive A's latent pages (the connector's page load) and
    the plugin imports A's checkpoint for B's vLLM slot before B's continuation [128, n). B is placed and restored as a
    fresh request and decodes exactly the reference greedy tokens of its whole prompt, as does A. The same hit
    without the checkpoint is refused; a connector on a backend without KDA checkpoints is refused at startup."""
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    prompt = prompt_ids(41, 150)
    serve.prefill({"a": prompt}, slots=[0], chunks={"a": (0, 128)})
    states = model.export_kda_state(model.kv_cache, 0)  # finish() of the step whose chunk ended at 128
    serve.prefill({"a": prompt}, slots=[0], chunks={"a": (128, 150)})
    serve.decode()
    # a hit without the checkpoint: no slot holds a state that saw the prefix
    serve.pages.ensure("c", 150)
    with expect_error(RuntimeError, "must run in the slot that holds the request's KDA state"):
        serve.prefill({"c": prompt}, slots=[2], chunks={"c": (128, 150)})
    serve.pages.release("c")
    # B: the connector loads A's prefix pages into B's own blocks, the plugin imports A's checkpoint
    serve.pages.ensure("b", 150)
    for lat in model.generator.backend.latent_pages(model.kv_cache).values():
        for pa, pb in zip(serve.pages.rows["a"][:2], serve.pages.rows["b"][:2]):
            lat[pb].copy_(lat[pa])
    model.import_kda_state(model.kv_cache, 1, states)
    serve.prefill({"b": prompt}, slots=[1], chunks={"b": (128, 150)})
    for _ in range(4):
        serve.decode()
    assert serve.out["b"] == serve.ref("b") and serve.out["a"] == serve.ref("a")
    assert model.counters["kda_exports"] == 1 and model.counters["kda_imports"] == 1
    assert model.generator.counters["prefix_state_imports"] == 1 and not model._prefix_states
    # control: the same pages with a zeroed checkpoint (a recurrence that never saw the prefix) serve other tokens
    serve.pages.ensure("d", 150)
    for lat in model.generator.backend.latent_pages(model.kv_cache).values():
        for pa, pd in zip(serve.pages.rows["a"][:2], serve.pages.rows["d"][:2]):
            lat[pd].copy_(lat[pa])
    assert states["next_pos"] == 128
    zeroed = [{**d, "S": d["S"] * 0, "C": d["C"] * 0} for d in states["states"]]
    # a checkpoint whose length is not the hit's is refused, and stays stashed for a retry
    model.import_kda_state(model.kv_cache, 2, {"next_pos": 64, "states": zeroed})
    with expect_error(ValueError, "KDA checkpoint after 64 tokens for a hit continuing at 128"):
        serve.prefill({"d": prompt}, slots=[2], chunks={"d": (128, 150)})
    assert model._prefix_states[2]["next_pos"] == 64
    model.import_kda_state(model.kv_cache, 2, {"next_pos": 128, "states": zeroed})
    serve.prefill({"d": prompt}, slots=[2], chunks={"d": (128, 150)})
    for _ in range(4):
        serve.decode()
    assert serve.out["d"] != serve.ref("d")
    # an abandoned hit: the vLLM slot's next request is prefilled from 0, and the stale checkpoint is dropped
    model.import_kda_state(model.kv_cache, 3, {"next_pos": 128, "states": zeroed})
    serve.prefill({"e": prompt_ids(43, 70)}, slots=[3])
    assert 3 not in model._prefix_states and model.counters["kda_imports"] == 2
    serve.decode()
    assert serve.out["e"] == serve.ref("e")
    # a backend without KDA checkpoints serves no connector
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import ModelBackend

    backend = type(model.generator.backend)
    real = backend.export_kda_state
    try:
        backend.export_kda_state = ModelBackend.export_kda_state
        with expect_error(NotImplementedError, "has no KDA checkpoint"):
            model.kv_cache_connector_views()
    finally:
        backend.export_kda_state = real
    evidence(b_tokens=serve.out["b"], hit_tokens=128, prompt_tokens=len(prompt))


def test_weight_source_selects_the_fast_start_factory(hf_config, monkeypatch):
    """GLM53_WEIGHT_SOURCE (Stage 12: image | tensors) makes the device path build the generator with
    tt/fast_start.py:build_generator (same contract), after the same 2x4 mesh checks; a tree without that module serves
    with the tensor loader (tt/generator.py) and says so in the startup timeline."""
    import sys
    import types

    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator

    calls = []

    def fast_build(*a, **kw):
        calls.append(kw)
        return build_mock_generator(*a, dsa_mode="dense_exact", **kw)

    fake = types.ModuleType("fast_start")
    fake.build_generator = fast_build
    monkeypatch.setitem(sys.modules, gv.FAST_START_FACTORY.split(":")[0], fake)
    monkeypatch.delenv(gv.GENERATOR_FACTORY_ENV, raising=False)
    monkeypatch.setenv(gv.WEIGHT_SOURCE_ENV, "image")
    monkeypatch.setenv(gv.MESH_GRAPH_DESC_ENV, str(gv.MODEL_DIR / "mesh" / "glm8_2x4_mesh_graph_descriptor.textproto"))
    model = Glm5NextForCausalLM.initialize_vllm_model(hf_config, SimpleNamespace(shape=(2, 4)), 4, max_seq_len=2051)
    assert calls == [{"max_batch": 4, "max_seq_len": 2051}]
    assert model.startup.info["generator_factory"] == gv.FAST_START_FACTORY
    assert model.startup.info["weight_source_requested"] == "image"
    monkeypatch.setitem(sys.modules, gv.FAST_START_FACTORY.split(":")[0], None)  # module absent -> ImportError
    info = {}
    assert gv._weight_source_factory(info) is build_generator
    assert "fast_start unavailable" in info["generator_factory"]


def test_weight_source_image_restores_the_tp_stack(hf_config, monkeypatch, expect_error):
    """GLM53_WEIGHT_SOURCE=image with a TP stack backend module (tt/generator.py TP_IMAGE_MODULES; the Stage 14b stack)
    builds through tt/generator.py + tt/tp_image.py (never tt/fast_start.py, which builds the layer pipeline): the
    backend construction goes to tp_image.build_or_restore (a JSON-safe signature: the image manifest), and the startup
    timeline records the weight source. GLM53_IMAGE_REQUIRED=1 (SPEED RULES 2026-10-21: no tensor loads in a mesh
    window): images are tried newest first (GLM53_IMAGE_PIN: only that one); a failed replay that already built layers
    leaves the shared weight cache empty for the next image; no usable image raises before any tensor load, and so does
    every construction path that would not restore an image. Another non-default module with a weight source raises;
    GLM53_WEIGHT_SOURCE=tensors builds directly."""
    import json
    import sys
    import types

    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import make_backend
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI

    builds, seen = [], {}

    def build_model_backend(model_dir, mesh_device, **kw):
        wc = kw.get("weight_cache")
        builds.append({**kw, "weight_cache": None if wc is None else len(wc)})  # the cache size at the build's start
        if wc is not None:
            wc[("dsa", len(builds))] = object()  # the layers a construction leaves in the shared cache
        return make_backend(4, "dense_exact", full_width=True, fused_sampling=True)

    def build_or_restore(build, mesh, *, signature, teardown, timeline, on_fail):
        seen.update(signature=json.loads(json.dumps(signature)))  # the manifest takes it as JSON
        timeline.update(source="image", image="/img/new", restore_s=1.0)
        return build(), timeline

    def restore_backend(build, mesh, d, *, teardown, timeline):
        be = build()  # the replay constructs (and fills the cache) before a mismatch shows
        if d.endswith("stale"):
            raise TI.ImageMismatch("upload #201 at another address")
        timeline.update(restore_s=2.0)
        return be

    fake = types.ModuleType("fake_tp8_stack")
    fake.build_model_backend = build_model_backend
    fake.serving_fabric_config = lambda: {"config": "FABRIC_1D_FAKE", "router_config": None}
    monkeypatch.setitem(sys.modules, "fake_tp8_stack", fake)
    monkeypatch.setattr(G, "TP_IMAGE_MODULES", (*G.TP_IMAGE_MODULES, "fake_tp8_stack"))
    monkeypatch.setattr(TI, "build_or_restore", build_or_restore)
    monkeypatch.setattr(TI, "restore_backend", restore_backend)
    monkeypatch.setattr(TI, "weights_identity", lambda: {"id": "w", "sha256": "w"})
    monkeypatch.setenv(G.BACKEND_MODULE_ENV, "fake_tp8_stack")
    monkeypatch.setenv(G.BACKEND_KW_ENV, '{"native_t": true, "weight_cache": {}}')
    monkeypatch.delenv(gv.GENERATOR_FACTORY_ENV, raising=False)
    monkeypatch.delenv(G.IMAGE_REQUIRED_ENV, raising=False)
    monkeypatch.delenv(G.IMAGE_PIN_ENV, raising=False)
    monkeypatch.setenv(gv.WEIGHT_SOURCE_ENV, "image")
    monkeypatch.setenv(gv.MESH_GRAPH_DESC_ENV, str(gv.MODEL_DIR / "mesh" / "glm8_2x4_mesh_graph_descriptor.textproto"))
    init = lambda: Glm5NextForCausalLM.initialize_vllm_model(
        hf_config, SimpleNamespace(shape=(2, 4)), 4, max_seq_len=2051
    )
    model = init()
    assert builds == [{"native_t": True, "weight_cache": 0}]
    assert seen["signature"] == {"B": 4, "module": "fake_tp8_stack", "kw": {"native_t": True, "weight_cache": True}}
    assert "tp_image.py restore" in model.startup.info["generator_factory"]
    assert model.startup.info["weight_source"] == {"source": "image", "image": "/img/new", "restore_s": 1.0}
    # required: the newest image whose replay holds; a stale one is skipped (recorded) and its layers dropped
    monkeypatch.setenv(G.IMAGE_REQUIRED_ENV, "1")
    monkeypatch.setattr(TI, "candidates", lambda root, w, mesh: ["/img/stale", "/img/good"])
    n = len(builds)
    ws = init().startup.info["weight_source"]
    assert ws["source"] == "image" and ws["image"] == "/img/good" and ws["restore_s"] == 2.0
    assert [t[0] for t in ws["tried"]] == ["stale"] and "ImageMismatch" in ws["tried"][0][1]
    assert [b["weight_cache"] for b in builds[n:]] == [0, 0]  # the second image's replay starts from an empty cache
    monkeypatch.setenv(G.IMAGE_PIN_ENV, "good")  # pinned: only that image
    n = len(builds)
    ws = init().startup.info["weight_source"]
    assert ws["image"] == "/img/good" and ws["tried"] == [] and len(builds) == n + 1
    monkeypatch.setenv(G.IMAGE_PIN_ENV, "/img/other")
    with expect_error(RuntimeError, "pinned /img/other"):
        init()
    monkeypatch.delenv(G.IMAGE_PIN_ENV)
    # required, no usable image: raises, and the tensor loader never ran (only the replays built)
    for cands, replays in (([], 0), (["/img/stale"], 1)):
        monkeypatch.setattr(TI, "candidates", lambda root, w, mesh, c=cands: c)
        n = len(builds)
        with expect_error(RuntimeError, "no usable TP DRAM image"):
            init()
        assert len(builds) == n + replays
    # required, but a path that would not restore an image: raises before any build
    n = len(builds)
    monkeypatch.setenv(gv.WEIGHT_SOURCE_ENV, "tensors")
    with expect_error(ValueError, "does not restore a TP DRAM image"):
        init()
    monkeypatch.setenv(gv.WEIGHT_SOURCE_ENV, "image")
    with expect_error(ValueError, "does not restore a TP DRAM image"):  # a factory override
        Glm5NextForCausalLM.initialize_vllm_model(
            hf_config, SimpleNamespace(shape=(2, 4)), 4, max_seq_len=2051,
            generator_factory=lambda *a, **k: build_mock_generator(*a, dsa_mode="dense_exact", **k),
        )  # fmt: skip
    monkeypatch.delenv(G.BACKEND_MODULE_ENV)  # the layer pipeline: tt/fast_start.py has no required mode
    with expect_error(ValueError, "does not restore a TP DRAM image"):
        init()
    monkeypatch.setenv(G.BACKEND_MODULE_ENV, "fake_tp8_stack")
    assert len(builds) == n
    monkeypatch.delenv(G.IMAGE_REQUIRED_ENV)
    # tensors: the plain construction (no tp_image)
    monkeypatch.setenv(gv.WEIGHT_SOURCE_ENV, "tensors")
    monkeypatch.setattr(TI, "build_or_restore", None)
    model = init()
    assert model.startup.info["weight_source"] is None and len(builds) == n + 1
    assert "tensor loader" in model.startup.info["generator_factory"]
    # a non-default module without an image path: no silent fall to tt/fast_start.py (the layer pipeline)
    monkeypatch.setattr(G, "TP_IMAGE_MODULES", ())
    with expect_error(ValueError, "no DRAM-image path for backend module fake_tp8_stack"):
        init()


def test_context_contract_env_selects_a_ladder_rung(tmp_path, evidence):
    """GLM53_CONTEXT_CONTRACT (the context ladder, program 2026-10-21 02:57:57Z): a rung's contract = the program
    contract with the rung's current_supported_context; the class capability (read at import, in a fresh process the
    way the vLLM worker imports the adapter) and the contract loader both take it; unset = doc/context_contract.json."""
    import subprocess
    import sys

    data = json.loads(Path(gv.CONTEXT_CONTRACT_PATH).read_text())  # the dense-exact program contract (conftest)
    probe = (
        "from models.autoports.zai_org_glm_5_3_flash.tt import generator_vllm as gv; "
        "print(gv.CONTEXT_CONTRACT_PATH, gv.Glm5NextForCausalLM.model_capabilities['tt_request_context_limit'], "
        "gv.load_context_contract().request_context_limit)"
    )
    got = {}
    # "served" = no GLM53_CONTEXT_CONTRACT: the model dir's doc/context_contract.json (the served value, an implementation
    # limit since the round-2 ruling: a stage-scope request limit = the served value); the rungs derive from the
    # dense-exact program contract
    for rung, supported in (("served", None), ("dense_exact", 2051), ("r2_32k", 32768), ("lifted", data["target_context"])):
        env = {k: v for k, v in os.environ.items() if k != gv.CONTEXT_CONTRACT_ENV}
        if supported is not None:
            path = tmp_path / f"context_contract_{rung}.json"
            path.write_text(json.dumps({**data, "current_supported_context": supported}))
            env[gv.CONTEXT_CONTRACT_ENV] = str(path)
        out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        path_s, cls_limit, loader_limit = out.stdout.split()[-3:]
        got[rung] = (int(cls_limit), int(loader_limit))
        assert path_s == (str(path) if supported is not None else str(gv.MODEL_DIR / "doc" / "context_contract.json"))
    served = json.loads((gv.MODEL_DIR / "doc" / "context_contract.json").read_text())["vllm_serving"]["served_max_model_len"]
    assert got == {"served": (served, served), "dense_exact": (2051, 2051), "r2_32k": (32768, 32768), "lifted": (0, 0)}, got
    evidence(limits=got)


def test_decoder_flag_reaches_the_backend_or_falls_back(hf_config, monkeypatch):
    """GLM53_DECODER (program 05:48:56Z item b) is forwarded as the factory's `decoder=` (tt/model.py
    build_model_backend) when tt/model.py lists it in SERVING_DECODERS (Stage 7a adds "persistent"); a decoder the tree
    does not provide serves as "optimized" (the K=1 op-chain decoder), recorded in the startup timeline; unset leaves
    the factory call unchanged."""
    from models.autoports.zai_org_glm_5_3_flash.tt import model as tt_model

    calls = []

    def factory(*a, **kw):
        calls.append(kw.get("decoder"))
        return build_mock_generator(*a, dsa_mode="dense_exact", **kw)

    init = lambda: Glm5NextForCausalLM.initialize_vllm_model(
        hf_config, None, 4, max_seq_len=2051, generator_factory=factory
    )
    monkeypatch.delenv(gv.DECODER_ENV, raising=False)
    assert "decoder_requested" not in init().startup.info and calls == [None]
    monkeypatch.setenv(gv.DECODER_ENV, "persistent")
    assert tt_model.SERVING_DECODERS == ("optimized",)
    info = init().startup.info
    assert calls[-1] == "optimized" and info["decoder_requested"] == "persistent"
    assert info["decoder_active"] == "optimized"
    monkeypatch.setattr(tt_model, "SERVING_DECODERS", ("optimized", "persistent"))
    assert init().startup.info["decoder_active"] == "persistent" and calls[-1] == "persistent"


def test_fused_sampling_backend_replays_one_trace_per_step(hf_config, evidence):
    """PRODUCTION_PATH.md M5 / M6: a backend whose decode trace also samples (ModelBackend.samples_in_decode_trace, the
    Stage 14 persistent program) serves through the unchanged adapter with ONE trace replay per decode step: no sampler
    trace is captured or replayed, the async split overlaps as before, and greedy + explicitly seeded sampled rows give
    the same tokens as the split decode + sampler traces on the same schedule."""

    def run(fused):
        model = _adapter(hf_config, max_batch=4, full_width=True, fused_sampling=fused)
        serve = MiniServe(model, Pages(256, 64))
        serve.params["s"] = {"temperature": 1.0, "top_k": 8, "seed": 77}
        reqs = {"a": prompt_ids(120, 61), "s": prompt_ids(121, 70), "c": prompt_ids(122, 9)}
        serve.prefill(reqs, slots=[0, 1, 2])
        serve.decode()  # the reloading step after admission
        order = list(serve.slot_req)
        pending = serve.submit()
        for _ in range(40):  # overlapped steady steps across page boundaries
            nxt = serve.submit(in_flight=1)
            serve.complete(pending, order)
            pending = nxt
        serve.complete(pending, order)
        serve.finish("c")
        for _ in range(6):
            serve.decode()
        return serve, model

    split, smodel = run(False)
    fused, fmodel = run(True)
    g = fmodel.generator
    assert g.fused_sampling and not smodel.generator.fused_sampling
    for r in ("a", "c"):
        assert fused.out[r] == fused.ref(r), r
    assert fused.out == split.out  # greedy rows and the seeded sampled row alike
    c = g.counters
    assert c["sampling_trace_captures"] == 0 and c["sampling_trace_replays"] == 0
    assert c["fused_sampling_steps"] == c["decode_trace_replays"] > 40 and c["eager_decode_steps"] == 0
    assert smodel.generator.counters["sampling_trace_replays"] == smodel.generator.counters["decode_trace_replays"]
    assert g._binding(fmodel.kv_cache).sampling_traces == {}
    evidence(
        fused_counters={k: c[k] for k in ("decode_trace_replays", "fused_sampling_steps", "sampling_trace_replays")},
        tokens_equal_split=True,
    )


@pytest.mark.parametrize("trace", [True, False], ids=["traced", "eager"])
def test_fused_backend_logits_route_steps_leave_no_stale_token_rows(trace):
    """A fused backend's decode step samples on every route (traced, and eagerly on the Stage 14b stack, which the
    fused mock now mirrors), so a logits-route (host-route) step overwrites every token row too. The generator treats
    it like a sampler step: (1) two logits-route steps feeding the SAME token give the split generator's logits (the
    second step must write its input, not trust a row the first step overwrote); (2) a slot whose sampled next token
    exists only on device and that sits out a logits-route step of another slot -- which consumes ITS OWN device token,
    so nothing is read back before the step -- still consumes that token afterwards."""
    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import make_backend
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import GREEDY, build_generator

    def gen(fused):
        be = make_backend(4, "dense_exact", full_width=True, fused_sampling=fused)
        g = build_generator(None, None, backend=be, tokenizer=object(), max_batch=4, max_seq_len=512,
                            stop_token_ids=())  # fmt: skip
        kv = g.allocate_kv_cache(4, 512, num_blocks=64)
        g.warmup_serving(kv, phase="prepare", prefill=True, decode=True)
        g.warmup_serving(kv, phase="capture", prefill=True, decode=True)
        tables = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
        prompts = torch.tensor([prompt_ids(3, 40), prompt_ids(4, 40)])
        g.prefill_forward(prompts, page_table=tables, kv_cache=kv, prompt_lens=[40, 40], empty_slots=[0, 1],
                          start_pos=[0, 0])  # fmt: skip
        return g, kv, tables

    def same_token_twice(fused):
        g, kv, tables = gen(fused)
        step = lambda i: g.decode_forward(  # noqa: E731
            torch.tensor([[100]]), torch.tensor([40 + i]), page_table=tables[:1], kv_cache=kv, slots=[0],
            enable_trace=trace, sampling_params=None,
        )  # fmt: skip
        return [step(i)[0].float() for i in range(2)]

    split, fused = same_token_twice(False), same_token_twice(True)
    assert torch.equal(split[1], fused[1])

    def offstep(fused):
        g, kv, tables = gen(fused)
        # both slots sample on device; their next tokens stay on device (async split, not read)
        g.decode_forward(torch.tensor([[100], [200]]), torch.tensor([40, 40]), page_table=tables, kv_cache=kv,
                         slots=[0, 1], enable_trace=trace, sampling_params=[GREEDY, GREEDY], read_from_device=False)  # fmt: skip
        # slot 1 alone on the logits route, consuming its own device token: the fused step overwrites slot 0's row
        g.decode_forward(torch.tensor([[-1]]), torch.tensor([41]), page_table=tables[1:], kv_cache=kv, slots=[1],
                         enable_trace=trace, sampling_params=None)  # fmt: skip
        # slot 0 consumes its device-sampled token (-1 = the slot's current next input)
        return g.decode_forward(torch.tensor([[-1]]), torch.tensor([41]), page_table=tables[:1], kv_cache=kv,
                                slots=[0], enable_trace=trace, sampling_params=None)[0].float()  # fmt: skip

    assert torch.equal(offstep(False), offstep(True))


def test_backend_module_supplies_builder_and_fabric(hf_config, monkeypatch, expect_error):
    """GLM53_BACKEND_MODULE (tt/generator.py) names the device backend's module - Stage 14's TP8 megakernel model plugs
    in there (PRODUCTION_PATH.md M6): the unchanged adapter builds its generator through that module's
    build_model_backend (after the same 2x4 mesh checks) and declares that module's fabric keywords to the plugin."""
    import sys
    import types

    from models.autoports.zai_org_glm_5_3_flash.tests.vllm.glm53_mock_stack import make_backend
    from models.autoports.zai_org_glm_5_3_flash.tt import generator as G

    calls = []

    def build_model_backend(model_dir, mesh_device, **kw):
        calls.append((mesh_device.shape, kw))
        return make_backend(4, "dense_exact", full_width=True, fused_sampling=True)

    fake = types.ModuleType("fake_tp8_backend")
    fake.build_model_backend = build_model_backend
    fake.serving_fabric_config = lambda: {"config": "FABRIC_2D_FAKE", "router_config": None}
    monkeypatch.setitem(sys.modules, "fake_tp8_backend", fake)
    monkeypatch.setenv(G.BACKEND_MODULE_ENV, "fake_tp8_backend")
    monkeypatch.delenv(gv.GENERATOR_FACTORY_ENV, raising=False)
    monkeypatch.delenv(gv.WEIGHT_SOURCE_ENV, raising=False)
    monkeypatch.setenv(gv.MESH_GRAPH_DESC_ENV, str(gv.MODEL_DIR / "mesh" / "glm8_2x4_mesh_graph_descriptor.textproto"))
    # router_config None = the binding's default: dropped (ttnn.set_fabric_config takes no None there)
    assert dict(Glm5NextForCausalLM.model_capabilities["fabric_config"]) == {"config": "FABRIC_2D_FAKE"}
    model = Glm5NextForCausalLM.initialize_vllm_model(hf_config, SimpleNamespace(shape=(2, 4)), 4, max_seq_len=2051)
    assert calls == [((2, 4), {})] and model.generator.fused_sampling
    # GLM53_BACKEND_KW: a serving profile's extra keywords for that builder (the 14b stack's native 32-row step)
    monkeypatch.setenv(G.BACKEND_KW_ENV, '{"native_t": true}')
    Glm5NextForCausalLM.initialize_vllm_model(hf_config, SimpleNamespace(shape=(2, 4)), 4, max_seq_len=2051)
    assert calls[-1] == ((2, 4), {"native_t": True})
    with expect_error(ValueError, "which the caller passes too"):  # a profile never overrides the caller
        G.build_generator(None, SimpleNamespace(shape=(2, 4)), native_t=False, tokenizer=object())
    monkeypatch.delenv(G.BACKEND_KW_ENV)
    monkeypatch.delenv(G.BACKEND_MODULE_ENV)
    assert dict(Glm5NextForCausalLM.model_capabilities["fabric_config"])["router_config"] is not None  # the pipeline


def test_fabric_capability_is_the_pipeline_fabric(evidence, monkeypatch):
    """model_capabilities["fabric_config"] reaches the plugin's set_fabric as the keywords Stage 4 opened the mesh
    with (FABRIC_1D, router payload FABRIC_PAYLOAD_BYTES), over the plugin's hardware default and reliability mode."""
    import vllm_tt_plugin.worker as worker

    import ttnn
    from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import FABRIC_PAYLOAD_BYTES

    fabric = Glm5NextForCausalLM.model_capabilities["fabric_config"]
    assert sorted(fabric) == ["config", "router_config"]
    seen = {}
    monkeypatch.setattr(worker, "get_fabric_config", lambda tt_config, n: ttnn.FabricConfig.FABRIC_2D)
    monkeypatch.setattr(worker.ttnn, "set_fabric_config", lambda **kw: seen.update(kw))
    worker.set_fabric({}, 8, fabric)
    assert seen["config"] == ttnn.FabricConfig.FABRIC_1D
    assert seen["router_config"].max_packet_payload_size_bytes == FABRIC_PAYLOAD_BYTES
    assert seen["reliability_mode"] == ttnn.FabricReliabilityMode.STRICT_INIT
    evidence(config=str(seen["config"]), payload=FABRIC_PAYLOAD_BYTES, reliability=str(seen["reliability_mode"]))


def test_host_step_overhead_probe_and_off_hot_path_events(hf_config, monkeypatch, tmp_path, evidence):
    """Program 2026-10-22 10:05:13Z: TPOT = the device step + <= 1 ms of host. (a) Every steady decode step (decode
    after decode, resident inputs) records its wall interval per live-row count; GLM53_DECODE_PROBE_EVERY=N drains the
    device before every N-th steady step and synchronizes after its submit (the step alone), so trace_report()
    ["host_step"] gives wall p50 - probe device p50 = host overhead per step; the step after a probe and the step after
    a prefill are not steady; probes change no token. (b) The serving hot path (prefill / decode events) never calls
    trace_report() (device queries): its rows carry counter copies and a writer thread appends them."""
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv(gv.EVENTS_ENV, str(events))
    monkeypatch.setenv(gv.DECODE_PROBE_ENV, "3")
    model = _adapter(hf_config)
    gen = model.generator
    syncs = []
    gen.backend.synchronize = lambda: syncs.append(model.counters["decode_calls"])

    def no_device_query():
        raise AssertionError("trace_report() on the serving hot path")

    monkeypatch.setattr(gen, "trace_report", no_device_query)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(3, 40)}, slots=[0])
    for _ in range(10):
        serve.decode()
    assert serve.out["a"] == serve.ref("a")  # the probes change no token
    # call 1 follows the prefill (reload); steady #1..#3 = calls 2-4, #3 probes; call 5 follows the probe; #4..#6 =
    # calls 6-8, #6 probes; call 9 follows it; #7 = call 10
    rep = model.host_step_report()
    row = rep["by_live_rows"]["1"]
    assert rep["probe_every"] == 3 and model.counters["decode_steady_steps"] == 7
    assert row["steady_steps"] == 7 and row["probes"] == 2 and model.counters["decode_probes"] == 2
    assert syncs == [3, 3, 7, 7]  # drained before the probe step's submit, synchronized after it
    assert row["wall_ms_p50"] > 0 and row["probe_device_ms_p50"] >= 0 and "host_overhead_ms_p50" in row
    model._event_q.join()  # the writer thread has appended every queued row
    rows = [json.loads(x) for x in events.read_text().splitlines()]
    hot = [r for r in rows if r["event"] in ("prefill", "decode")]
    assert [r["event"] for r in hot][:2] == ["prefill", "decode"] and any(r.get("probe") for r in hot) is False
    for r in hot:
        assert set(r["report"]) == {"generator", "adapter"} and "counters" in r["report"]["generator"]
        assert all("coverage" not in b for b in r["report"]["generator"]["bindings"])
    # a second request makes the live-row count 2: its steady steps land in their own row
    serve.prefill({"b": prompt_ids(4, 50)}, slots=[1])
    for _ in range(4):
        serve.decode()
    assert model.host_step_report()["by_live_rows"]["2"]["steady_steps"] >= 2
    evidence(host_step=model.host_step_report(), syncs=syncs, hot_rows=len(hot))


def test_host_step_probe_off_by_default_and_without_a_device(hf_config, monkeypatch):
    """No GLM53_DECODE_PROBE_EVERY: no drain / sync ever; steady intervals are still recorded. A backend with neither
    `synchronize` nor `mesh` (no device) turns probes into plain steps."""
    monkeypatch.delenv(gv.DECODE_PROBE_ENV, raising=False)
    model = _adapter(hf_config)
    syncs = []
    model.generator.backend.synchronize = lambda: syncs.append(1)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(5, 30)}, slots=[0])
    for _ in range(6):
        serve.decode()
    assert syncs == [] and model.counters["decode_probes"] == 0 and model.counters["decode_steady_steps"] == 5
    monkeypatch.setenv(gv.DECODE_PROBE_ENV, "1")
    model2 = _adapter(hf_config)
    assert not hasattr(model2.generator.backend, "synchronize") and getattr(model2.generator.backend, "mesh", None) is None
    serve2 = MiniServe(model2, Pages(256, 64))
    serve2.prefill({"a": prompt_ids(5, 30)}, slots=[0])
    for _ in range(4):
        serve2.decode()
    assert model2.counters["decode_probes"] == 0 and model2.counters["decode_steady_steps"] == 3
    assert serve2.out["a"] == serve2.ref("a")


def test_release_resets_the_slot_to_a_canonical_idle_row(hf_config, monkeypatch, evidence):
    """Program 2026-10-22 10:51:21Z (row-isolation leak + after-burst TPOT): a released slot goes back to the
    fresh-server idle state. The generator calls the backend's optional release_slot hook with the binding's cache
    (StackBackend: zero KDA state / conv rows, GLM53_IDLE_CANON=0 = off) and the next whole-buffer token write puts the
    idle row's token at 0; the live request's tokens are unchanged. A backend without the hook is untouched."""
    from models.autoports.zai_org_glm_5_3_flash.tt import stack_model

    model = _adapter(hf_config)
    gen = model.generator
    b = gen._binding(model.kv_cache)
    calls = []
    hook = gen.backend.release_slot  # the mock's own hook (the lifecycle SlotLedger) still runs: recorded, then called

    def recording_hook(kv, slot):
        calls.append((kv is b.kv_cache, slot))
        hook(kv, slot)

    gen.backend.release_slot = recording_hook
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(3, 40), "b": prompt_ids(4, 50)}, slots=[0, 1])
    for _ in range(3):
        serve.decode()
    ga = model.gen_slot[serve.slot_req.index("a")]
    serve.finish("a")
    assert calls == [(True, ga)]
    serve.prefill({"c": prompt_ids(5, 30)}, slots=[serve.slot_req.index(None)])
    for _ in range(3):
        serve.decode()
    live = {model.gen_slot[i] for i, r in enumerate(serve.slot_req) if r is not None}
    toks = gen.backend.read(b.io.tokens).reshape(-1)
    assert serve.out["b"] == serve.ref("b") and serve.out["c"] == serve.ref("c")
    # StackBackend's hook: reset_slot (zero state) + the idle_resets counter; GLM53_IDLE_CANON=0 skips the device work
    resets = []
    fake = SimpleNamespace(reset_slot=lambda kv, s: resets.append(s), counters={"idle_resets": 0})
    stack_model.StackBackend.release_slot(fake, "kv", 7)
    monkeypatch.setenv("GLM53_IDLE_CANON", "0")
    stack_model.StackBackend.release_slot(fake, "kv", 8)
    assert resets == [7] and fake.counters["idle_resets"] == 1
    evidence(released=ga, live_slots=sorted(live), token_rows=[int(t) for t in toks[: model.max_batch_size]])


def test_engine_gap_and_gc_pause_probes(hf_config, monkeypatch, tmp_path):
    """Program 2026-10-22 10:51:21Z (c): a model call that starts >= ENGINE_GAP_EVENT_S after the previous one ended
    while a request is live is an `engine_gap` event (previous / next call kinds, the gen-2 GC time so far); gen-2
    collections are timed into counters. No gap is counted while nothing is live."""
    import gc
    import time as _time

    events = tmp_path / "events.jsonl"
    monkeypatch.setenv(gv.EVENTS_ENV, str(events))
    monkeypatch.setattr(gv, "ENGINE_GAP_EVENT_S", 0.05)
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(3, 40)}, slots=[0])
    _time.sleep(0.08)  # the engine stalls between the prefill and the first decode
    gc.collect(2)
    serve.decode()
    serve.decode()
    assert model.counters["engine_gaps"] == 1 and model.counters["engine_gap_s_max"] >= 0.08
    assert model.counters["gc_gen2"] >= 1 and model.counters["gc_pause_s_total"] > 0
    serve.finish("a")
    _time.sleep(0.08)  # idle server: no live request, no gap
    serve.prefill({"b": prompt_ids(4, 30)}, slots=[0])
    assert model.counters["engine_gaps"] == 1
    model._event_q.join()
    gaps = [json.loads(x) for x in events.read_text().splitlines() if '"engine_gap"' in x]
    assert len(gaps) == 1 and gaps[0]["after"] == "prefill" and gaps[0]["before"] == "decode" and gaps[0]["live"] == 1
    assert serve.out["a"] == serve.ref("a")


def test_engine_stall_sampler_records_the_engine_stack(hf_config, monkeypatch, tmp_path):
    """int26 11:36:47Z (a 1.29-s gap between two decode calls of a lone request, outside the adapter): with the events
    sink on, a daemon thread samples the engine thread's stack when no model call has run for STALL_SAMPLE_S while a
    request is live -> `engine_stall_stack` events (at most STALL_SAMPLES per stall) naming where the engine waits; no
    sample while nothing is live or a call is running."""
    import time as _time

    events = tmp_path / "events.jsonl"
    monkeypatch.setenv(gv.EVENTS_ENV, str(events))
    monkeypatch.setattr(gv, "STALL_SAMPLE_S", 0.1)
    model = _adapter(hf_config)
    serve = MiniServe(model, Pages(256, 64))
    serve.prefill({"a": prompt_ids(3, 40)}, slots=[0])

    def stuck_in_the_scheduler():
        _time.sleep(0.55)

    stuck_in_the_scheduler()
    serve.decode()
    serve.finish("a")
    _time.sleep(0.4)  # idle server: nothing live, no sample
    model._event_q.join()
    rows = [json.loads(x) for x in events.read_text().splitlines() if '"engine_stall_stack"' in x]
    assert 1 <= len(rows) <= gv.STALL_SAMPLES, rows
    assert rows[0]["after"] == "prefill" and rows[0]["live"] == 1 and rows[0]["idle_s"] >= 0.1
    assert any("stuck_in_the_scheduler" in ln for ln in rows[0]["stack"]), rows[0]["stack"]
    assert serve.out["a"] == serve.ref("a")
