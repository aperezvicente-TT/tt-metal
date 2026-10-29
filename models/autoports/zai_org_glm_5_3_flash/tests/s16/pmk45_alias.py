# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""14b stack45: the 45-layer composite's WEIGHT ALIAS per kind and the owner-aware page table of its units.

alias: layer L reads the tile-cache entries (TileCacheSource._entry) and the permuted experts (the fused-experts cache
dirs L<L> symlinked to L<first>) of its kind's FIRST layer. Every layer still allocates its own device tensors (the
DRAM, the programs, the census and the launch count are the 45-layer composite's); the host reads 3 layers' files
instead of 45 and writes ~5 GB of permuted experts instead of ~114 GB (the 2026-10-20 root-disk fill). An A/B of two
paths on the SAME aliased weights (the PMK chain vs the op graph) stays a valid comparison; the real-weight numerics
come from the served construction (its TP image).

page table (program 2026-10-23 14:07Z, binding for device jobs): an explicit physical page table is published
before any prefill and the slot's page row is asserted to be a real row (never the null row / block 0)."""
import os

import torch


def kind_reps(cfg) -> dict:
    """{L: the first layer of L's kind}"""
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind

    first, rep = {}, {}
    for L in range(int(cfg.num_hidden_layers)):
        k = layer_kind(cfg, L)
        first.setdefault(k, L)
        rep[L] = first[k]
    return rep


def install(cfg, *, cache_root: str = "/opt/glm53/work-s14b/mock/") -> dict:
    """the alias (idempotent within a process): needs GLM53_FUSED_EXPERTS_CACHE under cache_root and
    GLM53_EP_KEEP_PERMUTED=1 (the first layer's permuted experts stay; the others' dirs are symlinks to them)"""
    from models.autoports.zai_org_glm_5_3_flash.tt import weights as WT

    cache_root = os.environ.get("GLM53_PMK45_CACHE_ROOT", cache_root)  # host-p300: its own work dir
    rep = kind_reps(cfg)
    orig = WT.TileCacheSource._entry
    if not getattr(orig, "_pmk45_alias", False):

        def _entry(self, layer_idx, name, _orig=orig):
            return _orig(self, rep.get(int(layer_idx), int(layer_idx)), name)

        _entry._pmk45_alias = True
        WT.TileCacheSource._entry = _entry
    fcd = os.environ.get("GLM53_FUSED_EXPERTS_CACHE", "")
    assert fcd and fcd.startswith(cache_root), f"alias: GLM53_FUSED_EXPERTS_CACHE must be under {cache_root} ({fcd!r})"
    assert os.environ.get("GLM53_EP_KEEP_PERMUTED") == "1", "alias: GLM53_EP_KEEP_PERMUTED=1"
    links = 0
    for L, r in rep.items():
        os.makedirs(os.path.join(fcd, f"L{r:02d}"), exist_ok=True)
        d = os.path.join(fcd, f"L{L:02d}")
        if r != L and not os.path.lexists(d):
            os.symlink(f"L{r:02d}", d)
            links += 1
    return dict(on=True, rep_by_kind=sorted(set(rep.values())), cache=fcd, links=links)


def publish_page_table(backend, kv, io, pf, rows: int = 1) -> dict:
    """an owner-aware PHYSICAL page table for slots 0 .. rows - 1, published through backend.write(io.page_table)
    before any prefill: the DSA shard pool's owner map when the backend has one (16's s16common.publish_owner_table:
    column j's block on chip (j // bpu) % C, local blocks from 1), else distinct blocks from 1 on the replicated pool
    (block 0 = the null block is never handed out). Asserts every slot's TPPrefill.page_row is that row and never the
    null row."""
    bps, nblk = int(kv.blocks_per_seq), int(kv.num_blocks)
    spec = backend.dsa_shard_owners(kv) if hasattr(backend, "dsa_shard_owners") else None
    pt = torch.zeros(rows, bps, dtype=torch.int64)
    if spec is not None:
        C_, nb_local, bpu = (int(v) for v in spec)
        nxt = [1] * C_
        for s_ in range(rows):
            for j in range(bps):
                r = (j // bpu) % C_
                assert nxt[r] < nb_local, f"owner table: chip {r} out of blocks ({nb_local}) at slot {s_} column {j}"
                pt[s_, j] = nxt[r] * C_ + r
                nxt[r] += 1
        form = "dsa-shard owner map"
    else:  # the replicated pool (no shard null block): slot s takes slot (s + 1) % rows's identity blocks -- a real,
        # NON-identity physical table (block 0 never in slot 0's row), so a chain that ignored the table would read the
        # wrong pages; needs rows >= 2 (the unit builds B = 2)
        assert rows >= 2 and rows * bps <= nblk, f"page table: rows {rows} (>= 2) x {bps} blocks within {nblk}"
        pt = torch.stack([torch.arange(((s_ + 1) % rows) * bps, ((s_ + 1) % rows + 1) * bps, dtype=torch.int64) for s_ in range(rows)])
        form = "replicated pool: slot s -> slot (s + 1) % B's blocks (non-identity, block 0 not in slot 0)"
    backend.write(io.page_table, pt.to(torch.int32))
    null = pf.sm.null_row(bps).to(torch.int64) if getattr(pf, "sm", None) is not None and hasattr(pf.sm, "null_row") else torch.zeros(bps, dtype=torch.int64)
    for s_ in range(rows):
        row = pf.page_row(s_).to(torch.int64)
        assert torch.equal(row, pt[s_]), f"slot {s_}: page_row {row[:4].tolist()}.. != the published {pt[s_, :4].tolist()}.."
        assert not torch.equal(row, null), f"slot {s_}: the NULL page row"
    row0 = pf.page_row(0).to(torch.int64)
    assert int(row0.min()) > 0, f"slot 0: block 0 (the null block) in its row {row0[:4].tolist()}.."
    return dict(form=form, rows=rows, blocks_per_seq=bps, num_blocks=nblk, first=pt[0, :4].tolist(), asserted=True)


def trace_mib(mesh):
    """the trace region in use per chip (MiB): ttnn.get_memory_view(TRACE) (tests/test_multichip_decoder.trace_bytes)"""
    import ttnn

    try:
        v = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return round(int(v.total_bytes_allocated_per_bank) * int(v.num_banks) / 2**20, 2)
    except Exception:  # noqa: BLE001
        return None


class AliasCache(dict):
    """a weight_cache (stack_model.StackBackend._build_tp: wc) whose per-layer WEIGHT entries resolve to the kind's first
    layer once that one is cached: ("dense" | "sparse" | "moe" | "moe_rs", L) -> (same, rep[L]). The layers then SHARE
    the device tensors (attention weights, the XSL expert slices in w.xsl_experts, router / shared expert); the layer
    OBJECTS ("dsa", L) are never aliased (their L is baked in)."""

    KINDS = ("dense", "sparse", "moe", "moe_rs")

    def __init__(self, rep: dict, base=None):
        super().__init__(base or {})
        self.rep = dict(rep)
        self.hits = 0

    def _k(self, key):
        if isinstance(key, tuple) and len(key) == 2 and key[0] in self.KINDS:
            r = (key[0], self.rep.get(int(key[1]), int(key[1])))
            if dict.__contains__(self, r):
                return r
        return key

    def __contains__(self, key):
        return dict.__contains__(self, self._k(key))

    def __getitem__(self, key):
        k = self._k(key)
        if k is not key:
            self.hits += 1
        return dict.__getitem__(self, k)

    def get(self, key, default=None):
        k = self._k(key)
        if dict.__contains__(self, k):
            if k is not key:
                self.hits += 1
            return dict.__getitem__(self, k)
        return default

    def __setitem__(self, key, value):
        if self._k(key) != key and dict.__contains__(self, self._k(key)):
            return  # the alias already resolves to the representative's entry: keep one copy
        dict.__setitem__(self, key, value)


def install_device_share(cfg, ctx: dict) -> dict:
    """the DEVICE-level alias (the pair, where 45 layers of distinct experts cannot fit: ~45 GB of experts per chip at
    TP2): ctx["weight_cache"] -> an AliasCache (sparse_kda / dense_kda layers share the first layer's weights), and the
    sparse_dsa layers' MoE (EP experts, router, shared) built once per kind (ep_experts.build_moe_ep /
    tp8_sparse.build_moe wrapped with a cache keyed (first layer, routed)). Each layer keeps its own layer objects,
    KV / state pages, PMK program copies and the mhc weights."""
    from models.autoports.zai_org_glm_5_3_flash.tt import ep_experts as EPX
    from models.autoports.zai_org_glm_5_3_flash.tt import tp8_sparse as TPSP

    rep = kind_reps(cfg)
    wc = ctx.get("weight_cache")
    if not isinstance(wc, AliasCache):
        ctx["weight_cache"] = AliasCache(rep, wc)
    built = {}
    for mod, name in ((EPX, "build_moe_ep"), (TPSP, "build_moe")):
        fn = getattr(mod, name)
        if getattr(fn, "_pmk45_share", False):
            continue

        def wrapped(mesh, cfg_, source, layer, ocfg, *a, _fn=fn, _name=name, **kw):
            key = (_name, rep.get(int(layer), int(layer)), tuple(a), tuple(sorted(kw.items())))
            if key not in built:
                built[key] = _fn(mesh, cfg_, source, layer, ocfg, *a, **kw)
            return built[key]

        wrapped._pmk45_share = True
        setattr(mod, name, wrapped)
    return dict(device_share=True, rep_by_kind=sorted(set(rep.values())))
