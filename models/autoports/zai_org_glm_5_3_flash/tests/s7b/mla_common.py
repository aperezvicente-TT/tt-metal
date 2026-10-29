# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7b shared pieces of the sparse_dsa (MLA) device checks: evidence rows, the L3 layer setup (decoder, latent
cache filled with fixed random latents, page table, per-row positions inside the dense-exact range) and the op chain of
the attention half it replaces (MhcPre -> OptimizedMLA.decode (30 ops) -> MhcPost, stacked residual)."""

from __future__ import annotations

import json
import os
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, int_tensor
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_mk_a import mhc_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_moe_parts import ffn_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M
from models.autoports.zai_org_glm_5_3_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
from models.autoports.zai_org_glm_5_3_flash.tt.mhc_post import OUT_STACKED, X_STACKED, MhcPost
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

PERF_KEYS = ("time_utc", "run", "test", "t", "execution", "timing_us", "watcher", "worker_l1", "variant",
             "steady_per_layer_us", "speedup", "code", "rev", "mk_code", "tree", "rings_one", "kb_one", "mk_readers",
             "placement", "config", "pos_last", "pos_step", "knob_env")  # fmt: skip


def mk_code() -> str:
    """the code that runs: MLA_REV + hashes of tt/mk_mla.py and of every kernel source under tt/kernels/{mla,mhc_pre}
    (GLM53_CODE_SHA is the tree's HEAD when the server started: a long-lived server runs later working-tree texts)"""
    import glob
    import hashlib

    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla

    tt = os.path.dirname(mk_mla.__file__)
    h = hashlib.sha1()
    py = [os.path.join(tt, n) for n in ("mk_mla.py", "tp_mla.py", "fabric_ar.py", "mhc_pre.py", "mhc_post.py",
                                        "mk_readers.py")]  # fmt: skip
    ks = [g for d in ("mla", "mhc_pre", "mhc_post", "mk", "fabric_ar") for g in glob.glob(os.path.join(tt, "kernels", d, "*"))]
    for f in [p_ for p_ in py if os.path.exists(p_)] + sorted(ks):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return f"{mk_mla.MLA_REV}:{h.hexdigest()[:12]}"


class Rows:
    """Evidence rows of one job: pcc.jsonl (every row), tests.jsonl (status rows), perf.jsonl (rows with timing)."""

    def __init__(self, layer: int):
        self.out = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
        self.run = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
        self.layer = layer
        self.rows, self.flushed = [], 0
        self.mk_code = mk_code()
        # the variant knobs of this job (read by the programs at build / call time)
        self.knob_env = {k: v for k, v in sorted(os.environ.items())
                         if k.startswith(("GLM53_MLA_", "GLM53_MHC_", "GLM53_MK_W_DELAY", "GLM53_TP_", "GLM53_DSA_"))}
        self.tree = os.environ.get("GLM53_TREE") or os.path.dirname(os.path.abspath(__file__)).split("/models/")[0]

    def __call__(self, name, **kw):
        r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=self.run, layer=self.layer,
                 test=os.environ.get("GLM53_ROW_PREFIX", "") + name, watcher=bool(os.environ.get("TT_METAL_WATCHER")),
                 code=os.environ.get("GLM53_CODE_SHA"), mk_code=self.mk_code, tree=self.tree, knob_env=self.knob_env,
                 **kw)  # fmt: skip
        self.rows.append(r)
        print("ROW", json.dumps(r, default=str), flush=True)
        return r

    def flush(self):
        new, self.flushed = self.rows[self.flushed :], len(self.rows)
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "pcc.jsonl"), "a") as f:
            for r in new:
                f.write(json.dumps(r, default=str) + "\n")
        with open(os.path.join(self.out, "tests.jsonl"), "a") as f:
            for r in new:
                if "status" in r:
                    f.write(json.dumps({k: r[k] for k in ("time_utc", "run", "test", "status", "t", "rev", "mk_code",
                                                          "watcher") if k in r}) + "\n")  # fmt: skip
        with open(os.path.join(self.out, "perf.jsonl"), "a") as f:
            for r in new:
                if r.get("timing_us"):
                    f.write(json.dumps({k: r[k] for k in PERF_KEYS if k in r}, default=str) + "\n")

    def failed(self):
        return [r for r in self.rows if r.get("status") == "fail"]


def aiclk_mhz():
    """the chip's AICLK now (kmd sysfs, no device access): the probes convert cycles at 1350 MHz (the busy clock)"""
    chip = os.environ.get("GLM53_CHIP")
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{int(chip)}/tt_aiclk") as f:
            return int(f.read().strip())
    except (TypeError, ValueError, OSError):
        return None


def host(t):
    return ttnn.to_torch(t).float()


def timed(tr, replays: int = 30):
    """median wall-clock us of `replays` blocking replays of the current trace"""
    if not tr.enabled:
        return None
    d = []
    for _ in range(replays):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return round(sorted(d)[len(d) // 2], 1)


def compare(r: dict, got: dict, want: dict):
    """bitwise / pcc / maxabs of every key of `got` vs `want` into row dict r"""
    from models.autoports.zai_org_glm_5_3_flash.tests.common import pcc

    for k in got:
        a, b = got[k].reshape(-1), want[k].reshape(-1)[: got[k].numel()]
        r[f"{k}_bitwise"] = bool(torch.equal(a, b))
        r[f"{k}_pcc"] = pcc(a, b)
        r[f"{k}_maxabs"] = float((a - b).abs().max())
        # PCC is scale-blind (a deferred-norm bug scaled latent by 0.04 at PCC 0.99999): ||a - b|| / ||b|| too
        r[f"{k}_rel_l2"] = float((a.double() - b.double()).norm() / max(b.double().norm(), 1e-30))


def code_version() -> str:
    """the MLA programs' code version: MLA_REV + a hash of tt/mk_mla.py (a resident object of an older text is rebuilt)"""
    import hashlib

    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla

    with open(mk_mla.__file__, "rb") as f:
        # + the class identity: a module reload makes new class objects, and a resident object of the old ones fails
        # every isinstance check of the reloaded code
        return f"{mk_mla.MLA_REV}:{hashlib.sha1(f.read()).hexdigest()[:12]}:{id(mk_mla.GroupedWeight):x}"


def resident(ctx, key: tuple, build):
    """CTX-resident object of this device process (program JOB PIPELINE RULE 2026-10-19T23:52:53Z): built once per
    (key, code version) and reused by every later job (weights uploaded once, programs' objects built once)"""
    if ctx is None:
        return build()
    k = ("s7b",) + tuple(key) + (code_version(),)
    store = ctx.setdefault("s7b_resident", {})
    if k not in store:
        store[k] = build()
    return store[k]


def layer_of(mesh, ctx, layer: int):
    """the resident MlaLayer (decoder, mHC args, golden streams, per-T cases) of this process"""
    if ctx is None:
        return MlaLayer(mesh, None, layer)
    store = ctx.setdefault("s7b_layers", {})
    if layer not in store:
        store[layer] = MlaLayer(mesh, ctx, layer)
    return store[layer]


class MeshMla:
    """The MLA attention of `layer` on a multi-chip mesh (weights replicated from the tile cache) with the decoder's
    cache helpers, without the full decoder (whose setup reads mesh tensors back to host): FusedMLA as
    FusedDecoderLayer builds it -> OptimizedMLA (DS linears); no FFN (mlp None)."""

    is_kda = False

    def __init__(self, cfg, layer: int, mesh, source):
        from types import SimpleNamespace

        from models.autoports.zai_org_glm_5_3_flash.tests.common_optimized import ocfg_from_env
        from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import BLOCK_SIZE, PREFILL_ALIGN
        from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import FusedMLA, _row
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedMLA, _upgrade

        DRAM = ttnn.DRAM_MEMORY_CONFIG
        dev = lambda n: source.device(layer, n, mesh)  # noqa: E731
        ocfg = ocfg_from_env()
        attn = FusedMLA(
            mesh,
            w_qa_kva=ttnn.concat([dev("self_attn.q_a_proj.weight"), dev("self_attn.kv_a_proj_with_mqa.weight")], dim=-1,
                                 memory_config=DRAM),
            qa_norm=_row(dev("self_attn.q_a_layernorm.weight")), kva_norm=_row(dev("self_attn.kv_a_layernorm.weight")),
            w_qb=dev("self_attn.q_b_proj.weight"), w_uk=dev("self_attn.kv_b_proj.W_uk"),
            w_uv=dev("self_attn.kv_b_proj.W_uv"), w_o=dev("self_attn.o_proj.weight"),
            num_heads=cfg.num_attention_heads, q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
            qk_head_dim=cfg.qk_head_dim, v_head_dim=cfg.v_head_dim, eps=cfg.rms_norm_eps, block_size=BLOCK_SIZE,
        )  # fmt: skip
        attn.fcfg = ocfg
        _upgrade(attn, OptimizedMLA, ocfg)
        self.device, self.block_size, self.prefill_align = mesh, BLOCK_SIZE, PREFILL_ALIGN["mla"]
        self.layer = SimpleNamespace(attn=attn, mlp=None)

    def allocate_cache(self, *, max_batch: int, max_seq_len: int):
        from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import DecoderCache

        cache = DecoderCache(max_batch=max_batch, max_seq_len=max_seq_len)
        blocks = -(-(max_seq_len + self.prefill_align - 1) // self.block_size)
        cache.blocks_per_seq = -(-blocks // 8) * 8
        cache.latent = self.layer.attn.allocate_cache(max_batch * cache.blocks_per_seq)
        return cache

    def make_page_table(self, cache, permutation=None):
        n = cache.max_batch * cache.blocks_per_seq
        perm = torch.arange(n) if permutation is None else permutation
        cache.page_table_host = perm.reshape(cache.max_batch, cache.blocks_per_seq).to(torch.int32)
        return ttnn.from_torch(cache.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


class MlaLayer:
    """L3 (sparse_dsa): decoder (cached in ctx; on a multi-chip mesh the attention only, MeshMla), mHC args, per-T
    caches and inputs."""

    def __init__(self, mesh, ctx, layer: int):
        self.mesh, self.layer = mesh, layer
        self.cfg = hf_config()
        self.source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        if int(mesh.get_num_devices()) > 1:
            key = ("s7b_mesh_mla", layer)
            store = ctx if ctx is not None else {}
            if key not in store:
                store[key] = MeshMla(self.cfg, layer, mesh, self.source)
            self.dec = store[key]
        else:
            self.dec = cached_decoder(self.cfg, layer, mesh, ctx)
        assert not self.dec.is_kda, f"layer {layer} is not an MLA layer"
        self.attn, self.mlp = self.dec.layer.attn, self.dec.layer.mlp
        # the chain's flash-MLA K chunk: the production 128 (rows <= 2) does not fit L1 next to the kernel-config ring
        # of the megakernel servers (worker_l1_size 1264512): every 7b job pins it (GLM53_MLA_KCHUNK, default 32)
        kc = int(os.environ.get("GLM53_MLA_KCHUNK", "32"))
        fc = self.attn.fcfg
        fc.mla_decode_k_chunk, fc.mla_decode_k_chunk_small = kc, kc
        self.kchunk = kc
        # the flash-MLA op's core grid: on the full grid its reduction staging CB (legacy (cores per head - 1) blocks)
        # does not fit next to the megakernel servers' kernel-config ring; GLM53_MLA_GRID="XxY" (default 8x4)
        gx, gy = (int(v) for v in os.environ.get("GLM53_MLA_GRID", "8x4").split("x"))
        self.flash_grid = ttnn.CoreCoord(gx, gy)
        self.a_args, self.f_args = mhc_args(self.source, self.cfg, layer), ffn_args(self.source, self.cfg, layer)
        self.ref_a = M.MhcPre(mesh, **self.a_args)
        self.mpost = MhcPost(mesh)
        self.g = golden(layer, 2048)["streams_in"].float()
        self.cases = {}

    def case(self, t: int, *, tag: str = "", last: int = DSA_DENSE_EXACT_MAX_KV - 2, step: int = 13, seed: int = 7):
        """per-T state: cache (max_batch t) filled with fixed random bf16 latents, page table (a fixed block
        permutation), positions last - step * i (row i), the layer input x [1,1,t,16384] fp32 DRAM (golden streams).
        Cases with different tags are independent copies of the same state (a megakernel vs its op chain)."""
        if (t, tag) in self.cases:
            return self.cases[(t, tag)]
        dec, mesh = self.dec, self.mesh
        cache = dec.allocate_cache(max_batch=t, max_seq_len=DSA_DENSE_EXACT_MAX_KV)
        n = t * cache.blocks_per_seq
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + t))
        pt = dec.make_page_table(cache, permutation=perm)
        shape = tuple(cache.latent.shape)
        lat0 = torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
        src = ttnn.from_torch(lat0, dtype=cache.latent.dtype, layout=ttnn.TILE_LAYOUT, device=mesh,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        ttnn.copy(src, cache.latent)
        pos = [last - step * i for i in range(t)]
        cur = int_tensor(pos, mesh)
        x = ttnn.from_torch(self.g[0, 300 : 300 + 37 * t : 37].reshape(1, 1, t, 16384), dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        c = dict(t=t, cache=cache, pt=pt, pos=pos, cur=cur, x=x, lat0=src, page_table_host=cache.page_table_host)
        self.cases[(t, tag)] = c
        return c

    # ---------------------------------------------------------------------------------------------- the op chain
    def mhc_pre(self, c):
        return self.ref_a(c["x"], hn_dtype=ttnn.bfloat16, t=c["t"], x_mode=M.X_FLAT, xout_mode=M.XOUT_STACKED,
                          pc_mode=M.PC_STACKED)  # fmt: skip

    def mla(self, c, hn, *, grid=None, heads=None, kc=None):
        """OptimizedMLA.decode (the 30 ops) with the flash-MLA op on `grid` (default self.flash_grid), `heads` per call
        (default: the decoder's rule) and K chunk `kc` (default self.kchunk) -- otherwise the same op sequence."""
        attn = self.attn
        bsz = hn.shape[-2]
        q_lat, latent = attn._project_decode(hn)
        q_lat = ttnn.permute(q_lat, (0, 2, 1, 3))  # [1,B,64,512]
        g = attn.device.compute_with_storage_grid_size()
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, attn.kv_lora),
            core_grid=ttnn.num_cores_to_corerangeset(bsz, g, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        lat = ttnn.to_memory_config(ttnn.reshape(latent, (1, bsz, 1, attn.kv_lora)), shard)
        ttnn.experimental.paged_update_cache(c["cache"].latent, lat, update_idxs_tensor=c["cur"], page_table=c["pt"])
        if heads is None:
            heads = attn.h if bsz >= attn.fcfg.mla_single_call_min_batch else attn.decode_heads
        o_lat = self.flash(c, q_lat, heads=heads, kc=kc, grid=grid)
        mem = dict(memory_config=ttnn.L1_MEMORY_CONFIG) if attn.ocfg.mla_uv_in0_l1 else {}
        return attn._output_decode(ttnn.permute(o_lat, (0, 2, 1, 3), **mem))

    def flash(self, c, q_lat, *, heads, kc=None, grid=None):
        """the flash-MLA op on q_lat [1,t,64,512] (heads per call, K chunk, core grid) -> o_lat [1,t,64,512]"""
        attn, t = self.attn, c["t"]
        program = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid or self.flash_grid, q_chunk_size=0,
                                         k_chunk_size=kc or self.kchunk, exp_approx_mode=False)  # fmt: skip
        groups = []
        for h0 in range(0, attn.h, heads):
            q_g = q_lat if heads == attn.h else ttnn.slice(q_lat, (0, 0, h0, 0), (1, t, h0 + heads, attn.kv_lora))
            o_g = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                q_g, c["cache"].latent, head_dim_v=attn.kv_lora, page_table_tensor=c["pt"], cur_pos_tensor=c["cur"],
                scale=attn.scale, program_config=program, compute_kernel_config=attn.ckc_sdpa, memory_config=attn.mem,
            )  # fmt: skip
            groups.append(ttnn.slice(o_g, (0, 0, 0, 0), (1, t, heads, attn.kv_lora)))
        return groups[0] if len(groups) == 1 else ttnn.concat(groups, dim=2)

    def mhc_post(self, c, a, r):
        return self.mpost(a, r.x_out, r.coef, t=c["t"], x_mode=X_STACKED, out_mode=OUT_STACKED)

    def chain_att(self, c):
        """the attention half: MhcPre -> MLA decode -> MhcPost -> s1 stacked [t,1,4,4096] fp32"""
        r = self.mhc_pre(c)
        return self.mhc_post(c, self.mla(c, r.hn), r)
