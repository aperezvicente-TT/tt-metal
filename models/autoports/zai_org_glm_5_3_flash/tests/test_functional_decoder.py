# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight tests of the GLM-5.3-Flash functional decoder against the fp32 CPU goldens.

Covers every layer kind with the real config shapes and real weights (pre-packed bfp4/bfp8 tile cache):
  * test_golden_prompts           — every captured golden layer (L0-2 dense_kda, L4/L44 sparse_kda, L3/L7/L43
                                    sparse_dsa) x golden prompts: chunked paged prefill PCC (+ per-token statistics),
                                    KDA final state PCC, expert agreement, first decode step through a captured trace
                                    (repeated replay and eager == trace bitwise), multi-chunk prefill determinism.
  * test_prefill_lengths          — one loaded instance, increasing non-aligned logical lengths around tile (32), page
                                    (64), physical-chunk (128 multiples), chunk (2048) and DSA (2051) boundaries, a short
                                    request reusing the slot, explicit prefix continuation, traced follow-on decode.
  * test_batched_paged_decode     — up to 32 users in shuffled slots with a permuted page table, different prompt
                                    lengths/positions, multi-step traced decode with refreshed inputs/positions.
  * test_prefill_under_live_trace — all prefill programs prepared first; a decode trace is captured and replayed; new
                                    requests of never-seen lengths, tails and continuation starts are prefilled into
                                    another slot while the trace is live (no new program, no allocation alive at replay);
                                    decode then continues for every user.
Evidence rows are appended to $GLM53_STAGE_EVIDENCE (default: the stage evidence dir) as JSON lines.
"""

from __future__ import annotations

import os
import random

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import (
    LAYERS,
    PCC_BAR,
    RecordingRouter,
    check,
    compare,
    delta_pcc,
    golden,
    hf_config,
    int_tensor,
    load_decoder,
    open_mesh,
    pcc,
    record,
    row_stats,
    streams4,
    to_dev,
    to_host,
)
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import plan_prefill_chunks

KIND_OF = {layer: kind for kind, layers in LAYERS.items() for layer in layers}


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh()
    yield m
    ttnn.close_mesh_device(m)


@pytest.fixture(scope="module")
def cfg():
    return hf_config()


def _prompts(layer: int) -> list[int]:
    """Golden prompt lengths captured for this layer (MLA/DSA: only kv_len <= 2051, the dense-exact DSA regime)."""
    d = os.path.join(compare.GOLDENS_DIR, "layer_io", f"L{layer:02d}")
    have = sorted(int(t) for t in os.listdir(d) if t.isdigit())
    only = os.environ.get("GLM53_GOLDEN_PROMPTS")  # e.g. "128,512" for the watcher run
    if only:
        have = [t for t in have if t in {int(x) for x in only.split(",")}]
    return [t for t in have if KIND_OF[layer] != "sparse_dsa" or t < DSA_DENSE_EXACT_MAX_KV]


def _max_seq(dec, n):
    return n + 64 if dec.is_kda else min(n + 64, DSA_DENSE_EXACT_MAX_KV)


class TracedDecode:
    """Warm -> restore state -> capture -> replay over persistent device inputs (streams, cur_pos, page table)."""

    def __init__(self, dec, cache, page_table, batch: int):
        self.dec, self.cache, self.page_table = dec, cache, page_table
        self.streams = to_dev(torch.zeros(1, batch, 4, 4096), dec.device)
        self.cur_pos = int_tensor([0] * batch, dec.device)
        self.trace = None

    def set_inputs(self, streams_rows: torch.Tensor, positions: list[int]):
        """streams_rows [1, B, 4, 4096] host; refreshed into the persistent trace inputs (outside capture)."""
        self.dec.check_decode_positions(positions, self.cache)
        host_s = ttnn.from_torch(
            streams_rows.reshape(1, 1, -1, 4 * 4096).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host_s, self.streams)
        host_p = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32)
        ttnn.copy_host_to_device_tensor(host_p, self.cur_pos)

    def capture(self):
        snap = self.dec.snapshot_state(self.cache)
        eager = self.dec.decode_forward(
            self.streams, cache=self.cache, page_table=self.page_table, cur_pos=self.cur_pos
        )  # compile + allocate (program cache warm-up)
        eager_host = to_host(eager)
        del eager
        self.dec.restore_state(self.cache, snap)
        ttnn.synchronize_device(self.dec.device)
        self.trace = self.dec.capture_decode_trace(
            self.streams, cache=self.cache, page_table=self.page_table, cur_pos=self.cur_pos
        )
        self.snap = snap
        return eager_host

    def step(self) -> torch.Tensor:
        return to_host(self.trace.replay(blocking=False))

    def release(self):
        if self.trace is not None:
            self.trace.release()
            self.trace = None


def _routing_rows(rec: RecordingRouter, seq_len: int, n_experts: int, align: int) -> torch.Tensor:
    rows = []
    for (off, n, phys), t in zip(plan_prefill_chunks(seq_len, align), rec.log):
        rows.append(ttnn.to_torch(t).float().reshape(phys, n_experts)[:n])
    return torch.cat(rows)


# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer", [0, 1, 2, 4, 44, 3, 7, 43])
def test_golden_prompts(mesh, cfg, layer):
    kind = KIND_OF[layer]
    dec = load_decoder(cfg, layer, mesh)
    assert dec.spec.kind == kind
    rec = None
    if dec.spec.mlp == "moe":
        rec = RecordingRouter(dec.layer.mlp.router)
        dec.layer.mlp.router = rec
    failures = []
    prompts = _prompts(layer)
    for T in prompts:
        g = golden(layer, T)
        cache = dec.allocate_cache(max_batch=1, max_seq_len=_max_seq(dec, T))
        pt = None if dec.is_kda else dec.make_page_table(cache)
        if rec is not None:
            rec.log, rec.recording = [], True
        y = streams4(dec.prefill_forward(g["streams_in"], cache=cache, user_slot=0))
        if rec is not None:
            rec.recording = False
        row = dict(
            layer_kind=kind,
            mode="prefill",
            layer=layer,
            seq_len=T,
            prompt=T,
            pcc=pcc(y, g["streams_out"]),
            delta_pcc=delta_pcc(y, g["streams_out"], g["streams_in"]),
            weights="real",
            trace=False,
            **row_stats(y, g["streams_out"]),
        )
        if dec.is_kda:
            st = compare.load_kda_state(layer, T)
            row["kda_state_pcc"] = pcc(ttnn.to_torch(cache.kda.recurrent[0]).float(), st["recurrent"])
            conv_tt = ttnn.to_torch(cache.kda.conv[0]).float()[0]  # [3, 24576]
            row["kda_conv_pcc"] = pcc(conv_tt, st["conv"][0, :, -3:].t())
        if rec is not None:
            routing = _routing_rows(rec, T, cfg.n_routed_experts, dec.prefill_align)
            router = compare.load_router(layer, T)
            row["expert_agreement"] = compare.expert_agreement(routing.topk(8, dim=-1).indices, router["indices"])[
                "set"
            ]
            rec.log = []
        record(row)
        if row["pcc"] < PCC_BAR:
            failures.append(row)

        # determinism: the same request again (fresh slot state): single chunk at 128, multi-chunk at the largest
        # prompt <= 4096 (KDA: 2 chunks of 2048; MLA: 2048 tokens)
        if T == 128 or T == max(t for t in prompts if t <= 4096):
            y2 = streams4(dec.prefill_forward(g["streams_in"], cache=cache, user_slot=0))
            same = bool(torch.equal(y, y2))
            record(
                dict(
                    layer_kind=kind,
                    mode="prefill_repeat",
                    layer=layer,
                    seq_len=T,
                    prompt=T,
                    chunks=len(plan_prefill_chunks(T, dec.prefill_align)),
                    bitwise_equal=same,
                ),
                name="determinism.jsonl",
            )
            check([] if same else [dict(layer=layer, prompt=T)], "prefill not deterministic")

        # first greedy decode step at position T through a captured trace (+ repeated replay). The 32K golden set
        # has no decode0 files: decode position T-1 after a (T-1)-token prefill (golden = prefill row T-1).
        if "decode0_streams_in" in g:
            dec_in, dec_ref, dec_pos = g["decode0_streams_in"], g["decode0_streams_out"], T
        else:
            dec_in, dec_ref, dec_pos = g["streams_in"][:, T - 1 : T], g["streams_out"][:, T - 1 : T], T - 1
            dec.prefill_forward(g["streams_in"][:, : T - 1], cache=cache, user_slot=0)
        td = TracedDecode(dec, cache, pt, batch=1)
        td.set_inputs(dec_in, [dec_pos])
        eager = td.capture()
        out1 = td.step()
        dec.restore_state(cache, td.snap)
        out2 = td.step()
        drow = dict(
            layer_kind=kind,
            mode="decode",
            layer=layer,
            seq_len=dec_pos + 1,
            prompt=T,
            position=dec_pos,
            pcc=pcc(out1, dec_ref),
            delta_pcc=delta_pcc(out1, dec_ref, dec_in),
            weights="real",
            trace=True,
            replay_bitwise_equal=bool(torch.equal(out1, out2)),
            eager_vs_trace_bitwise_equal=bool(torch.equal(eager, out1)),
        )
        record(drow)
        td.release()
        if drow["pcc"] < PCC_BAR or not drow["replay_bitwise_equal"]:
            failures.append(drow)
        del cache
    check(failures)


# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer", [0, 4, 3])
def test_prefill_lengths(mesh, cfg, layer):
    """Increasing logical lengths on ONE loaded instance; every prefix n is checked against golden streams_out[:n]."""
    kind = KIND_OF[layer]
    dec = load_decoder(cfg, layer, mesh)
    src_prompt = 4096
    g = golden(layer, src_prompt)
    if dec.is_kda:
        lengths = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 2051, 2052, 4095, 4096]
    else:
        lengths = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 2047, 2048, 2049, 2050, DSA_DENSE_EXACT_MAX_KV]
    cache = dec.allocate_cache(max_batch=1, max_seq_len=max(lengths) + 64 if dec.is_kda else DSA_DENSE_EXACT_MAX_KV)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    failures = []
    for n in lengths + [33]:  # ... then reuse a short request: stale state from the long request must not leak
        y = streams4(dec.prefill_forward(g["streams_in"][:, :n], cache=cache, user_slot=0))
        row = dict(
            layer_kind=kind,
            mode="prefill",
            layer=layer,
            seq_len=n,
            prompt=f"{src_prompt}[:{n}]",
            pcc=pcc(y, g["streams_out"][:, :n]),
            weights="real",
            test="lengths",
        )
        record(row)
        if row["pcc"] < PCC_BAR:
            failures.append(row)

    # explicit prefix continuation: [0, P) then [P, n) with start_pos = P (KDA: incl. a 2-token prefix, shorter than
    # the 3-row conv history; MLA: page multiples)
    cont_cases = (
        [(2, 40), (37, 300), (1000, 4095), (2048, 4096)] if dec.is_kda else [(64, 130), (1024, 2051), (2048, 2050)]
    )
    for p, n in cont_cases:
        dec.prefill_forward(g["streams_in"][:, :p], cache=cache, user_slot=0, start_pos=0)
        y = streams4(dec.prefill_forward(g["streams_in"][:, p:n], cache=cache, user_slot=0, start_pos=p))
        row = dict(
            layer_kind=kind,
            mode="prefill_continuation",
            layer=layer,
            seq_len=n - p,
            start_pos=p,
            prompt=f"{src_prompt}[{p}:{n}]",
            pcc=pcc(y, g["streams_out"][:, p:n]),
            weights="real",
            test="continuation",
        )
        record(row)
        if row["pcc"] < PCC_BAR:
            failures.append(row)

    # traced follow-on decode consuming the filled cache: prefix n0, then rows n0, n0+1, ... through the trace
    n0 = 2045 if not dec.is_kda else 3001
    dec.prefill_forward(g["streams_in"][:, :n0], cache=cache, user_slot=0, start_pos=0)
    td = TracedDecode(dec, cache, pt, batch=1)
    td.set_inputs(g["streams_in"][:, n0 : n0 + 1], [n0])
    td.capture()
    steps = 6 if not dec.is_kda else 8  # MLA: positions 2045..2050 -> kv_len up to 2051
    outs = []
    for s in range(steps):
        p = n0 + s
        td.set_inputs(g["streams_in"][:, p : p + 1], [p])
        out = td.step()
        outs.append(out)
        row = dict(
            layer_kind=kind,
            mode="decode",
            layer=layer,
            seq_len=p + 1,
            position=p,
            prompt=f"{src_prompt}[{p}]",
            pcc=pcc(out, g["streams_out"][:, p : p + 1]),
            weights="real",
            trace=True,
            test="follow_on_decode",
        )
        record(row)
        if row["pcc"] < PCC_BAR:
            failures.append(row)
    check([dict(layer=layer)] if torch.equal(outs[0], outs[1]) else [], "trace replay ignored refreshed inputs")
    td.release()
    check(failures)


# ---------------------------------------------------------------------------------------------------------------------
def _batched_users(dec, batch, steps, seed):
    rng = random.Random(seed)
    prompts = [128, 512, 2048] + ([4096] if dec.is_kda else [])
    slots = list(range(batch))
    rng.shuffle(slots)
    users = []
    for u, slot in enumerate(slots):
        T = prompts[u % len(prompts)]
        hi = min(T, DSA_DENSE_EXACT_MAX_KV) - steps if not dec.is_kda else T - steps
        users.append((slot, T, rng.randint(1, hi)))
    return users


@pytest.mark.parametrize("layer,batch", [(0, 32), (4, 32), (3, 32), (3, 4)])
def test_batched_paged_decode(mesh, cfg, layer, batch):
    """B users with different prompts/lengths in shuffled slots, permuted page table, multi-step traced decode.

    Gates: (1) per step, the decode PCC over all B users' rows vs the fp32 goldens >= 0.995; (2) per user and step,
    decode PCC >= min(0.995, PCC of the same row from the prompt-gated prefill path of a single-user cache - 0.002): a
    slot / page-table / position / trace bug would put a user far below both. Single MoE-dominated tokens can sit below
    0.995 in *both* paths because of the bfloat4_b routed experts (CPU control: bfp4_control.jsonl), and the recorded
    decode-vs-prefill-path PCC dips where the two numerically different paths fall on opposite sides of a near-tied
    top-8 routing decision."""
    kind = KIND_OF[layer]
    dec = load_decoder(cfg, layer, mesh)
    max_seq = DSA_DENSE_EXACT_MAX_KV if not dec.is_kda else 4096 + 64
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=max_seq)
    pt = None
    if not dec.is_kda:
        perm = torch.randperm(batch * cache.blocks_per_seq, generator=torch.Generator().manual_seed(7))
        pt = dec.make_page_table(cache, permutation=perm)
    steps = 4
    users = _batched_users(dec, batch, steps, 1234 + layer + batch)
    # prefill-path reference rows n..n+steps-1 per user (single-user cache), then the real batched prefill per slot
    ref_cache = dec.allocate_cache(max_batch=1, max_seq_len=max_seq)
    if not dec.is_kda:
        dec.make_page_table(ref_cache)
    prefill_rows = {}
    for slot, T, n in users:
        y = streams4(dec.prefill_forward(golden(layer, T)["streams_in"][:, : n + steps], cache=ref_cache, user_slot=0))
        prefill_rows[slot] = y[:, n : n + steps]
    del ref_cache
    for slot, T, n in users:  # prefill order != slot order
        dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=slot, start_pos=0)
    by_slot = sorted(users)
    td = TracedDecode(dec, cache, pt, batch=batch)

    def inputs(step):
        rows = torch.cat([golden(layer, T)["streams_in"][:, n + step : n + step + 1] for _, T, n in by_slot], dim=1)
        return rows, [n + step for _, _, n in by_slot]

    rows, pos = inputs(0)
    td.set_inputs(rows, pos)
    td.capture()
    failures = []
    for step in range(steps):
        rows, pos = inputs(step)
        td.set_inputs(rows, pos)
        out = td.step()
        ref = torch.cat([golden(layer, T)["streams_out"][:, n + step : n + step + 1] for _, T, n in by_slot], dim=1)
        users_rows = []
        for i, (slot, T, n) in enumerate(by_slot):
            p_dec = pcc(out[:, i : i + 1], ref[:, i : i + 1])
            p_pre = pcc(prefill_rows[slot][:, step : step + 1], ref[:, i : i + 1])
            p_cons = pcc(out[:, i : i + 1], prefill_rows[slot][:, step : step + 1])
            users_rows.append(
                dict(
                    slot=slot,
                    prompt=T,
                    position=n + step,
                    decode_pcc=p_dec,
                    prefill_path_pcc=p_pre,
                    decode_vs_prefill_path_pcc=p_cons,
                )
            )
            if p_dec < min(PCC_BAR, p_pre - 0.002):
                failures.append(dict(layer=layer, step=step, **users_rows[-1]))
        row = dict(
            layer_kind=kind,
            mode="decode",
            layer=layer,
            batch=batch,
            step=step,
            pcc=pcc(out, ref),
            min_user_pcc=min(u["decode_pcc"] for u in users_rows),
            min_decode_vs_prefill_path_pcc=min(u["decode_vs_prefill_path_pcc"] for u in users_rows),
            seq_len=max(pos) + 1,
            prompt="mixed 128/512/2048/4096 prefixes",
            positions=pos,
            weights="real",
            trace=True,
            test="batched_paged_decode",
        )
        record(row)
        record(
            dict(layer_kind=kind, layer=layer, batch=batch, step=step, users=users_rows),
            name="batched_decode_users.jsonl",
        )
        if row["pcc"] < PCC_BAR:
            failures.append(row)
    td.release()
    check(failures)


# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layer", [0, 4, 3])
def test_prefill_under_live_trace(mesh, cfg, layer):
    """Trace-safe interleaving: prepare every prefill program, capture a B=4 decode trace, then prefill new requests
    (never-seen logical lengths, padded tails, a continuation start) into slot 3 while the trace stays live, and keep
    decoding every user from the same trace. Asserts: no program-cache growth during those prefills, decode PCC vs
    golden for all users before/after (slot 3 decodes its new request), and — under TT_METAL_TRACE_ALLOC_TRACKING=1 —
    no unsafe allocation alive at replay (prefill outputs are read back and freed before the next replay)."""
    kind = KIND_OF[layer]
    dec = load_decoder(cfg, layer, mesh)
    batch, steps = 4, 2
    max_seq = DSA_DENSE_EXACT_MAX_KV if not dec.is_kda else 4096 + 64
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=max_seq)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    dec.prepare_prefill(cache)  # every bucket + per-slot programs, before any capture
    users = {0: (512, 300), 1: (2048, 1500), 2: (128, 77), 3: (2048, 700)}  # slot -> (golden prompt, prefix)
    for slot, (T, n) in users.items():
        dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=slot)
    td = TracedDecode(dec, cache, pt, batch=batch)

    def run_steps(tag):
        bad = []
        for _ in range(steps):
            rows = torch.cat([golden(layer, T)["streams_in"][:, n : n + 1] for T, n in users.values()], dim=1)
            ref = torch.cat([golden(layer, T)["streams_out"][:, n : n + 1] for T, n in users.values()], dim=1)
            td.set_inputs(rows, [n for _, n in users.values()])
            out = td.step()
            p = pcc(out, ref)
            per_user = [pcc(out[:, i : i + 1], ref[:, i : i + 1]) for i in range(batch)]
            record(
                dict(
                    layer_kind=kind,
                    mode="decode",
                    layer=layer,
                    batch=batch,
                    pcc=p,
                    per_user_pcc=per_user,
                    seq_len=max(n for _, n in users.values()) + 1,
                    prompt=f"live-trace {tag}",
                    positions=[n for _, n in users.values()],
                    weights="real",
                    trace=True,
                    test="prefill_under_live_trace",
                )
            )
            if p < PCC_BAR:
                bad.append((tag, p))
            for slot, (T, n) in list(users.items()):
                users[slot] = (T, n + 1)
        return bad

    td.set_inputs(
        torch.cat([golden(layer, T)["streams_in"][:, n : n + 1] for T, n in users.values()], dim=1),
        [n for _, n in users.values()],
    )
    td.capture()
    failures = run_steps("before")
    entries = mesh.num_program_cache_entries()
    # new requests in slot 3 while the decode trace is live: never-seen lengths / tails / continuation
    if dec.is_kda:
        new_requests = [(4096, 1), (4096, 333), (4096, 2050), (4096, 3000)]
    else:
        new_requests = [(2048, 1), (2048, 333), (2048, 2046)]
    for T, n in new_requests:
        y = streams4(dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=3))
        p = pcc(y, golden(layer, T)["streams_out"][:, :n])
        record(
            dict(
                layer_kind=kind,
                mode="prefill",
                layer=layer,
                seq_len=n,
                prompt=f"{T}[:{n}] (slot 3, live trace)",
                pcc=p,
                weights="real",
                test="prefill_under_live_trace",
            )
        )
        if p < PCC_BAR:
            failures.append(("prefill", n, p))
    if dec.is_kda:  # continuation of the last request at a non-page offset
        T, n = 4096, 3000
        y = streams4(
            dec.prefill_forward(golden(layer, T)["streams_in"][:, n:3100], cache=cache, user_slot=3, start_pos=n)
        )
        p = pcc(y, golden(layer, T)["streams_out"][:, n:3100])
        record(
            dict(
                layer_kind=kind,
                mode="prefill_continuation",
                layer=layer,
                seq_len=100,
                start_pos=n,
                prompt=f"{T}[{n}:3100] (slot 3, live trace)",
                pcc=p,
                weights="real",
                test="prefill_under_live_trace",
            )
        )
        if p < PCC_BAR:
            failures.append(("continuation", n, p))
        users[3] = (4096, 3100)
    else:
        users[3] = (2048, 2046)
    new_entries = mesh.num_program_cache_entries() - entries
    record(
        dict(layer_kind=kind, layer=layer, program_cache_entries_added_during_live_trace_prefill=new_entries),
        name="prefill_live_trace.jsonl",
    )
    if new_entries:
        failures.append(("program cache grew while a decode trace was live", new_entries))
    failures += run_steps("after")
    td.release()
    check(failures, "prefill under a live decode trace")


# ---------------------------------------------------------------------------------------------------------------------
def _slot_state(dec, cache, slot: int, n_tokens: int) -> list[torch.Tensor]:
    """Host copy of one slot's carried state: KDA S + conv rows; MLA latent rows 0..n_tokens-1 of the slot, gathered
    through the host page table (so a remapped table is compared by logical position)."""
    if dec.is_kda:
        return [ttnn.to_torch(cache.kda.recurrent[slot]), ttnn.to_torch(cache.kda.conv[slot])]
    lat = ttnn.to_torch(cache.latent)  # [blocks, 1, 64, 512]
    nb = -(-n_tokens // dec.block_size)
    blocks = cache.page_table_host[slot][:nb].long()
    return [lat[blocks].reshape(nb * dec.block_size, -1)[:n_tokens]]


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_decode_between_prefill_chunks(mesh, cfg, layer):
    """Interleaving under a live decode trace: a multi-chunk request is prefilled chunk by chunk into slot 3 while a B=4
    decode trace keeps decoding users 0-2 between its chunks, with slot 3 inactive (cur_pos -1). Asserts: every replay
    leaves slot 3's carried state bitwise unchanged; the interleaved prefill output and final state are bitwise equal to
    an uninterrupted prefill of the same request; users 0-2 decode correctly throughout; slot 3 then decodes its next
    token; no program-cache growth; (MLA) slot 3 runs on a page-table mapping rewritten in place under the live trace.
    """
    kind = KIND_OF[layer]
    dec = load_decoder(cfg, layer, mesh)
    batch = 4
    max_seq = DSA_DENSE_EXACT_MAX_KV if not dec.is_kda else 4096 + 64
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=max_seq)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    dec.prepare_prefill(cache)
    users = {0: (512, 300), 1: (2048, 1500), 2: (128, 77)}  # slot -> (golden prompt, position of the next token)
    for slot, (T, n) in users.items():
        dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=slot)
    req_T, req_n = 4096, (4000 if dec.is_kda else 2050)  # 2 chunks, non-aligned tail; MLA decode then at kv 2051
    req = golden(layer, req_T)
    y_ref = dec.prefill_forward(req["streams_in"][:, :req_n], cache=cache, user_slot=3)  # uninterrupted reference
    ref_state = _slot_state(dec, cache, 3, req_n)

    failures, flags = [], dict(slot3_unchanged_by_every_replay=True)
    td = TracedDecode(dec, cache, pt, batch=batch)

    def inputs(slot3_pos):
        rows = [golden(layer, T)["streams_in"][:, n : n + 1] for T, n in users.values()]
        rows.append(req["streams_in"][:, slot3_pos : slot3_pos + 1] if slot3_pos >= 0 else torch.zeros_like(rows[0]))
        td.set_inputs(torch.cat(rows, dim=1), [n for _, n in users.values()] + [slot3_pos])

    def replay(tag, slot3_pos=-1):
        before = _slot_state(dec, cache, 3, req_n) if slot3_pos < 0 else None
        inputs(slot3_pos)
        out = td.step()
        ref = torch.cat([golden(layer, T)["streams_out"][:, n : n + 1] for T, n in users.values()], dim=1)
        p = pcc(out[:, :3], ref)
        record(
            dict(
                layer_kind=kind,
                mode="decode",
                layer=layer,
                batch=batch,
                pcc=p,
                per_user_pcc=[pcc(out[:, i : i + 1], ref[:, i : i + 1]) for i in range(3)],
                seq_len=max(n for _, n in users.values()) + 1,
                prompt=f"interleaved {tag} (slot 3 {'inactive' if slot3_pos < 0 else f'at {slot3_pos}'})",
                positions=[n for _, n in users.values()] + [slot3_pos],
                weights="real",
                trace=True,
                test="decode_between_prefill_chunks",
            )
        )
        if p < PCC_BAR:
            failures.append((tag, p))
        if before is not None:
            same = all(torch.equal(a, b) for a, b in zip(before, _slot_state(dec, cache, 3, req_n)))
            flags["slot3_unchanged_by_every_replay"] &= same
            if not same:
                failures.append((tag, "inactive slot 3 state changed by a decode replay"))
        for slot, (T, n) in list(users.items()):
            users[slot] = (T, n + 1)
        return out

    inputs(-1)
    td.capture()
    entries = mesh.num_program_cache_entries()
    replay("idle")  # slot 3 holds the reference request's state; must stay untouched
    if not dec.is_kda:  # new request in slot 3: fresh block assignment (its blocks reversed), rewritten in place
        perm = cache.page_table_host.clone()
        perm[3] = perm[3].flip(0)
        dec.update_page_table(cache, pt, perm.reshape(-1))
        flags["page_table_updated_in_place_under_live_trace"] = True
    ys = []
    for i, (off, n, _) in enumerate(plan_prefill_chunks(req_n, dec.prefill_align)):
        ys.append(dec.prefill_forward(req["streams_in"][:, off : off + n], cache=cache, user_slot=3, start_pos=off))
        replay(f"after chunk {i}")
    y = torch.cat(ys, dim=1)
    flags["interleaved_prefill_output_bitwise_equal"] = bool(torch.equal(y, y_ref))
    flags["interleaved_final_state_bitwise_equal"] = all(
        torch.equal(a, b) for a, b in zip(_slot_state(dec, cache, 3, req_n), ref_state)
    )
    p_req = pcc(streams4(y), req["streams_out"][:, :req_n])
    out = replay("slot 3 decodes", slot3_pos=req_n)
    p_dec3 = pcc(out[:, 3:4], req["streams_out"][:, req_n : req_n + 1])
    new_entries = mesh.num_program_cache_entries() - entries
    td.release()
    record(
        dict(
            layer_kind=kind,
            mode="prefill",
            layer=layer,
            seq_len=req_n,
            prompt=f"{req_T}[:{req_n}] (slot 3, chunks interleaved with decode replays)",
            pcc=p_req,
            weights="real",
            test="decode_between_prefill_chunks",
        )
    )
    record(
        dict(
            layer_kind=kind,
            mode="decode",
            layer=layer,
            seq_len=req_n + 1,
            prompt=f"{req_T} slot 3 first decode at {req_n} after interleaved prefill",
            pcc=p_dec3,
            position=req_n,
            weights="real",
            trace=True,
            test="decode_between_prefill_chunks",
        )
    )
    record(
        dict(layer_kind=kind, layer=layer, request_tokens=req_n, program_cache_entries_added=new_entries, **flags),
        name="interleaved_decode.jsonl",
    )
    if p_req < PCC_BAR or p_dec3 < PCC_BAR:
        failures.append(("slot 3", p_req, p_dec3))
    if new_entries:
        failures.append(("program cache grew while a decode trace was live", new_entries))
    if not (flags["interleaved_prefill_output_bitwise_equal"] and flags["interleaved_final_state_bitwise_equal"]):
        failures.append(("interleaved prefill differs from the uninterrupted one", flags))
    check(failures, "decode between prefill chunks")


@pytest.mark.parametrize("layer", [0, 4, 3])
def test_inactive_slot_nonfinite_input(mesh, cfg, layer):
    """An inactive slot's input row is don't-care: in a full model it can hold anything a previous layer produced for
    that row. With slot 3 inactive (cur_pos -1) and its streams row set to NaN, +Inf and 1e30 in turn, one traced
    replay each from the same starting state must leave slot 3's carried state bitwise unchanged and give users 0-2
    outputs and carried state bitwise equal to a replay with a zero row in slot 3."""
    kind = KIND_OF[layer]
    dec = load_decoder(cfg, layer, mesh)
    batch = 4
    cache = dec.allocate_cache(max_batch=batch, max_seq_len=DSA_DENSE_EXACT_MAX_KV if not dec.is_kda else 2048 + 64)
    pt = None if dec.is_kda else dec.make_page_table(cache)
    users = {0: (512, 300), 1: (2048, 1500), 2: (128, 77), 3: (2048, 1000)}
    for slot, (T, n) in users.items():
        dec.prefill_forward(golden(layer, T)["streams_in"][:, :n], cache=cache, user_slot=slot)
    rows = torch.cat([golden(layer, T)["streams_in"][:, n : n + 1] for T, n in users.values()], dim=1)
    positions = [n for _, n in list(users.values())[:3]] + [-1]
    td = TracedDecode(dec, cache, pt, batch=batch)

    def states():  # each slot's carried state incl. the position its decode writes (MLA latent row n)
        return [_slot_state(dec, cache, s, users[s][1] + 1) for s in range(batch)]

    def run(fill):
        r = rows.clone()
        r[:, 3] = fill
        dec.restore_state(cache, snap)
        td.set_inputs(r, positions)
        return td.step(), states()

    rows0 = rows.clone()
    rows0[:, 3] = 0.0
    td.set_inputs(rows0, positions)
    td.capture()
    snap = td.snap  # taken before the capture: no device allocation may happen while the trace is live
    before = states()
    out0, st0 = run(0.0)
    flags, failures = {}, []
    for name, fill in (("nan", float("nan")), ("inf", float("inf")), ("huge", 1e30)):
        out, st = run(fill)
        flags[f"{name}_slot3_state_unchanged"] = all(torch.equal(a, b) for a, b in zip(st[3], before[3]))
        flags[f"{name}_users_output_bitwise_equal"] = bool(torch.equal(out[:, :3], out0[:, :3]))
        flags[f"{name}_users_state_bitwise_equal"] = all(
            torch.equal(a, b) for s in range(3) for a, b in zip(st[s], st0[s])
        )
        if not all(v for k, v in flags.items() if k.startswith(name)):
            failures.append((name, {k: v for k, v in flags.items() if k.startswith(name)}))
    ref = torch.cat([golden(layer, T)["streams_out"][:, n : n + 1] for T, n in list(users.values())[:3]], dim=1)
    p = pcc(out0[:, :3], ref)
    td.release()
    record(
        dict(
            layer_kind=kind,
            mode="decode",
            layer=layer,
            batch=batch,
            pcc=p,
            seq_len=max(positions) + 1,
            prompt="users 0-2 with slot 3 inactive (zero row)",
            positions=positions,
            weights="real",
            trace=True,
            test="inactive_slot_nonfinite_input",
        )
    )
    record(dict(layer_kind=kind, layer=layer, **flags), name="inactive_nonfinite.jsonl")
    if p < PCC_BAR:
        failures.append(("users 0-2 decode", p))
    check(failures, "inactive slot with a non-finite input row")
