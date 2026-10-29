# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests (no device) of W1's generator wiring (GLM53_PF_GROUP_MAX, design/w1_grpmax_design.md §8): which chunks
of a request go to StackBackend.prefill_group (full chunks, sizes from GLM53_PF_GROUP_SIZES, greedy), where the head /
the prefix-cache state source and snapshot land, the store-A cut chunk (snap_at) running alone, and the per-chunk
fallback when a group is not served."""
import collections
from types import SimpleNamespace

import torch

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G

C = 2048


class FakeBackend:
    prefill_state_io = True
    continuation_align = 128
    supported_context = 1 << 21
    prefill_align = 32
    block_size = 32
    prefill_chunk_tokens = C
    vocab_size = 8

    def __init__(self, serve=lambda n, start, head: True):
        self.calls, self.serve = [], serve

    def prefill_chunk(self, tokens, **kw):
        self.calls.append(("chunk", kw["chunk_start"], kw["n_valid"], kw["logits_rows"], kw.get("state_src"),
                           kw.get("state_snap"), int(tokens.reshape(-1)[0]), kw.get("snap_at")))  # fmt: skip
        return torch.zeros(1, 8) if kw["logits_rows"] == "last" else None

    def prefill_group(self, tokens, **kw):
        n, head = len(kw["n_valid"]), kw["logits_rows"] != "none"
        if not self.serve(n, kw["chunk_start"], head):
            return NotImplemented
        self.calls.append(("group", kw["chunk_start"], tuple(kw["n_valid"]), kw["logits_rows"], kw.get("state_src"),
                           kw.get("state_snap"), [int(v) for v in tokens.reshape(-1)[::C]]))  # fmt: skip
        return torch.zeros(1, 8) if head else None


def run(backend, n_tok, *, start_pos=0, state_src=None, state_snap=None, snap_at=None):
    g = object.__new__(G.Glm53Generator)
    g.backend, g.host_sampling = backend, True
    g.counters = collections.defaultdict(int)
    b = SimpleNamespace(slots={0: SimpleNamespace(seq_len=start_pos)}, max_seq_len=1 << 21, coverage={0: 1 << 21},
                        kv_cache=None, io=None, tok_val={}, dev_ok={})  # fmt: skip
    ids = torch.arange(start_pos, start_pos + n_tok, dtype=torch.int64)
    out = g._prefill_user(b, 0, ids, start_pos=start_pos, state_src=state_src, state_snap=state_snap, snap_at=snap_at)
    return backend.calls, out, g.counters


def test_off_by_default_every_chunk_alone(monkeypatch):
    monkeypatch.delenv("GLM53_PF_GROUP_MAX", raising=False)
    calls, out, _ = run(FakeBackend(), 4 * C)
    assert [c[0] for c in calls] == ["chunk"] * 4 and out.shape == (1, 8)


def test_a_32k_prompt_is_one_group_of_16_with_the_head(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, out, cnt = run(FakeBackend(), 16 * C)
    assert calls == [("group", 0, (C,) * 16, "last", None, None, [k * C for k in range(16)])]
    assert out.shape == (1, 8) and cnt["prefill_groups"] == 1 and cnt["prefill_chunks"] == 16


def test_a_partial_tail_runs_alone_and_carries_the_head(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, _, _ = run(FakeBackend(), 16 * C + 100)
    assert calls[0][:4] == ("group", 0, (C,) * 16, "none")
    assert calls[1][:4] == ("chunk", 16 * C, 100, "last") and len(calls) == 2


def test_runs_decompose_greedily_into_the_allowed_sizes(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, _, _ = run(FakeBackend(), 15 * C)  # 8 + 4 + 2 + 1
    assert [(c[0], len(c[2]) if c[0] == "group" else 1, c[3]) for c in calls] == [
        ("group", 8, "none"), ("group", 4, "none"), ("group", 2, "none"), ("chunk", 1, "last")]
    assert [c[1] for c in calls] == [0, 8 * C, 12 * C, 14 * C]


def test_the_cap_and_the_size_set_bound_the_groups(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "4")
    calls, _, _ = run(FakeBackend(), 9 * C)  # sizes {4, 2}: 4 + 4 + 1
    assert [len(c[2]) if c[0] == "group" else 1 for c in calls] == [4, 4, 1]
    monkeypatch.setenv("GLM53_PF_GROUP_SIZES", "4")
    calls, _, _ = run(FakeBackend(), 7 * C)  # only 4: 4 + 1 + 1 + 1
    assert [c[0] for c in calls] == ["group", "chunk", "chunk", "chunk"]


def test_an_unserved_group_falls_back_to_every_chunk(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, out, cnt = run(FakeBackend(serve=lambda n, s, h: False), 4 * C)
    assert [c[:4] for c in calls] == [("chunk", k * C, C, "last" if k == 3 else "none") for k in range(4)]
    assert out.shape == (1, 8) and cnt["prefill_groups"] == 0


def test_a_cold_later_group_falls_back_from_there(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, _, _ = run(FakeBackend(serve=lambda n, s, h: s == 0), 12 * C)  # 8 served, then 4 cold
    assert calls[0][:3] == ("group", 0, (C,) * 8)
    assert [c[:2] for c in calls[1:]] == [("chunk", k * C) for k in range(8, 12)]


def test_a_cold_first_group_runs_alone_and_the_next_groups_are_served(monkeypatch):
    # a request's first group (chunk 0: window 0 / FRESH) has no prewarmed key; it runs chunk by chunk and the later
    # groups are still tried (program 2026-10-25 20:58Z: the old loop broke at it and served no group at all)
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "4")
    monkeypatch.setenv("GLM53_PF_GROUP_SIZES", "4")
    calls, out, cnt = run(FakeBackend(serve=lambda n, s, h: s != 0), 12 * C)
    assert [c[:2] for c in calls] == [("chunk", 0), ("chunk", C), ("chunk", 2 * C), ("chunk", 3 * C), ("group", 4 * C), ("group", 8 * C)]
    assert calls[-1][3] == "last" and out.shape == (1, 8)
    assert cnt["prefill_groups"] == 2 and cnt["prefill_chunks"] == 12


def test_the_state_source_and_snapshot_go_to_the_first_and_the_last_run(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, _, _ = run(FakeBackend(), 6 * C, start_pos=4 * C, state_src=3, state_snap=5)  # 4 + 2
    assert [(c[0], c[1], c[4], c[5]) for c in calls] == [("group", 4 * C, 3, None), ("group", 8 * C, None, 5)]
    calls, _, _ = run(FakeBackend(), 4 * C + 64, start_pos=4 * C, state_src=3, state_snap=5)  # 4 + the tail
    assert [(c[0], c[4], c[5]) for c in calls] == [("group", 3, None), ("chunk", None, 5)]


def test_prewarm_keys_parse_both_heads_and_need_the_switch(monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM

    be = object.__new__(SM.StackBackend)
    seen = []

    class G:
        c = C

        def __init__(self, n):
            self.n = n

        def windows(self, start, nv):
            seen.append((self.n, start, len(nv)))
            return tuple([0 if start == 0 and k == 0 else 65536 for k in range(len(nv))])

    be.prefill_group_ok = lambda n: n >= 2
    be.prefill_group_obj = lambda n: G(n)
    monkeypatch.setenv("GLM53_PF_GROUP_PREWARM", "16@0,4@32768")
    monkeypatch.delenv("GLM53_PF_GROUP_MAX", raising=False)
    assert be._group_prewarm_keys() == []  # the prewarm needs the group switch
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    keys = be._group_prewarm_keys()
    assert [(n, s, h) for n, s, _, h in keys] == [(16, 0, True), (16, 0, False), (4, 32768, True), (4, 32768, False)]
    assert keys[0][2][0] == 0 and set(keys[2][2]) == {65536}


def test_the_store_a_cut_chunk_runs_alone(monkeypatch):
    monkeypatch.setenv("GLM53_PF_GROUP_MAX", "16")
    calls, _, _ = run(FakeBackend(), 16 * C, snap_at=16 * C)  # the cut ends the last chunk: 8 + 4 + 2 + 1, then the cut
    assert [(c[0], c[1]) for c in calls] == [("group", 0), ("group", 8 * C), ("group", 12 * C), ("chunk", 14 * C),
                                             ("chunk", 15 * C)]  # fmt: skip
    assert calls[-1][3] == "last" and calls[-1][7] == 16 * C and calls[-2][7] is None
    calls, _, _ = run(FakeBackend(), 16 * C, snap_at=5 * C + 128)  # a cut inside chunk 5: groups only before it
    assert [c[0] for c in calls] == ["group"] + ["chunk"] * 12 and len(calls[0][2]) == 4
    assert [c[7] for c in calls[1:3]] == [None, 5 * C + 128] and all(c[7] is None for c in calls[3:])
