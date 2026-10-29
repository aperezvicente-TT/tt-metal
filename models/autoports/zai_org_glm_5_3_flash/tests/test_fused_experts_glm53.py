# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.deepseek.moe.fused_experts at GLM-5.3-Flash shapes on one Blackhole chip.

E=288 routed experts, I=2048, H=4096, top-8, routed_scaling_factor 2.5, SwiGLU clamp 10, bfloat4_b gate_up/down
loaded per expert from the shared tile cache (`$GLM53_CACHE_DIR`), routing from `ttnn.topk` over the fp32 sigmoid
router scores + e_score_correction_bias (selection), with the unbiased scores handed to the op.

Inputs come from `doc/prework/p2b_fused_experts/scripts/prep_moe_inputs.py` (CPU, run first), which turns the golden
layer_io streams of MoE layer `GLM53_P2B_LAYER` (prompt 128) into the MoE input `hf` with the fp32 reference layer and
stores the per-expert outputs Y[e, t] = down_e(swiglu(gate_up_e(hf_t))) for the checkpoint fp8-dequant weights and for
the tile-cache bfp4 weights, plus the gate_up files tile-permuted into per-core [gate_c | up_c] shards. Each batch of B
consecutive golden tokens is scored against
  ref_fp8   : the reference routed MoE (fp8-dequant weights, reference fp32 routing)       -- reference of record
  ref_bfp4  : the same routed MoE with the bfp4 cache weights the op multiplies              -- acceptance (>= 0.995)
  ref_dev   : bfp4 weights, the device's own top-k ids and bf16 scores, bf16-rounded input  -- op isolation (>= 0.995)
`num_experts` is the union of the batch's device selections (measured, asserted to cover it), or the static bound
min(E, B * top_k) for the trace-friendly variant.

Env: GLM53_P2B_PREP (prep output root), GLM53_P2B_LAYER (default 4), GLM53_P2B_EVIDENCE (append jsonl rows there),
GLM53_P2B_BLOCK (experts_block_size, default 1), GLM53_RECORD_ONLY=1 (record without asserting).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi4_fp32

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_args, **_kwargs):
        pass


pytestmark = pytest.mark.use_module_device

GLM53_WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
CACHE_DIR = os.environ.get("GLM53_CACHE_DIR", "/opt/glm53/tt-cache/glm53-flash")
PREP = os.environ.get("GLM53_P2B_PREP", "/opt/glm53/work-p2b")
LAYER = int(os.environ.get("GLM53_P2B_LAYER", "4"))
PROMPT = 128
EVIDENCE = os.environ.get("GLM53_P2B_EVIDENCE")
RECORD_ONLY = os.environ.get("GLM53_RECORD_ONLY") == "1"
BLOCK = int(os.environ.get("GLM53_P2B_BLOCK", "1"))  # fastest measured on BH (perf.md); blocks 1-5 are bit-identical

E, H, I, TOP_K = 288, 4096, 2048, 8
SCALE, LIMIT, EPS = 2.5, 10.0, 1e-20
PADDED_EXPERTS = 320  # top-k width (tile multiple); padded experts get bias -1e9 and are never selected
NUM_CORES, DRAM_BANKS = 64, 8
PCC_BAR = 0.995

_STATE: dict = {}


def _prep_dir():
    return os.path.join(PREP, f"L{LAYER:02d}_T{PROMPT}")


def _require_prep():
    need = [os.path.join(_prep_dir(), f) for f in ("inputs.pt", "Y_fp8.npy", "Y_bfp4.npy", "Y_bfp4_xbf16.npy")]
    need.append(os.path.join(PREP, "perm", f"L{LAYER:02d}", f"{E - 1:03d}.gate_up_proj.tensorbin"))
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        pytest.skip(f"run prep_moe_inputs.py {LAYER} {PROMPT} {PREP} first (missing {missing[0]})")


def pcc(a, b) -> float:
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


def _nd_dram(rows: int, shard_cols: int) -> ttnn.MemoryConfig:
    """One [rows, shard_cols] shard per compute core, round-robin over the DRAM banks (the op's weight contract)."""
    banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(DRAM_BANKS)])
    spec = ttnn.NdShardSpec(
        shard_shape=[rows, shard_cols],
        grid=banks,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)


def _state(device) -> dict:
    """Experts, router and references for LAYER, loaded once per module (module-scoped device)."""
    if _STATE.get("device") is device:
        return _STATE
    _require_prep()
    man = json.load(open(os.path.join(GLM53_WS, "weights", "manifest.json")))["tensors"]
    t0 = time.time()
    gu_cfg = _nd_dram(H, 2 * I // NUM_CORES)
    dn_cfg = _nd_dram(I, H // NUM_CORES)
    perm_dir = os.path.join(PREP, "perm", f"L{LAYER:02d}")
    gate_up, down = [], []
    for e in range(E):
        t = ttnn.load_tensor(os.path.join(perm_dir, f"{e:03d}.gate_up_proj.tensorbin"))
        gate_up.append(ttnn.to_device(t, device, memory_config=gu_cfg))
        t = ttnn.load_tensor(os.path.join(CACHE_DIR, man[f"layers.{LAYER}.mlp.experts.{e}.down_proj"]["path"]))
        down.append(ttnn.to_device(t, device, memory_config=dn_cfg))
    load_s = time.time() - t0
    router_w = ttnn.load_tensor(os.path.join(CACHE_DIR, man[f"layers.{LAYER}.mlp.gate.weight"]["path"]), device=device)
    router_w = ttnn.pad(router_w, ((0, 0), (0, PADDED_EXPERTS - E)), 0.0)
    inp = torch.load(os.path.join(_prep_dir(), "inputs.pt"))
    bias = torch.cat([inp["bias"].float(), torch.full((PADDED_EXPERTS - E,), -1e9)]).reshape(1, 1, 1, PADDED_EXPERTS)
    bias_tt = ttnn.from_torch(
        bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ys = {k: np.load(os.path.join(_prep_dir(), f"{k}.npy"), mmap_mode="r") for k in ("Y_fp8", "Y_bfp4", "Y_bfp4_xbf16")}
    _STATE.clear()
    _STATE.update(
        device=device,
        gate_up=gate_up,
        down=down,
        router_w=router_w,
        bias=bias_tt,
        inp=inp,
        ys=ys,
        ckc=hifi4_fp32(device),
        load_s=load_s,
    )
    return _STATE


def _route(st, x_tt, batch, routing="fp32_topk"):
    """fp32 router logits -> sigmoid -> + bias -> top-8 ids as uint16; unbiased scores cast to bf16 for the op.

    fp32_topk: top-k ranks the fp32 choice (as the reference does); on this main an fp32 top-k emits UINT32 ids, which
               are typecast to the op's UINT16.
    bf16_topk: the choice is cast to bf16 first, so ttnn.topk emits UINT16 ids natively (coarser ranking).
    """
    logits = ttnn.linear(x_tt, st["router_w"], compute_kernel_config=st["ckc"], dtype=ttnn.float32)
    scores = ttnn.sigmoid(logits)
    choice = ttnn.add(scores, st["bias"])
    if routing == "bf16_topk":
        choice_bf = ttnn.typecast(choice, ttnn.bfloat16)
        _, idx = ttnn.topk(choice_bf, k=TOP_K, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(choice_bf)
    else:
        _, idx32 = ttnn.topk(choice, k=TOP_K, dim=-1, largest=True, sorted=True)
        idx = idx32 if idx32.dtype == ttnn.uint16 else ttnn.typecast(idx32, ttnn.uint16)
        if idx is not idx32:
            same = torch.equal(ttnn.to_torch(idx32).to(torch.int64), ttnn.to_torch(idx).to(torch.int64))
            assert same, "uint32 -> uint16 id typecast changed the ids"
            ttnn.deallocate(idx32)
    assert idx.dtype == ttnn.uint16, idx.dtype
    scores_bf = ttnn.typecast(ttnn.slice(scores, (0, 0, 0, 0), (1, 1, batch, E)), ttnn.bfloat16)
    for t in (logits, scores, choice):
        ttnn.deallocate(t)
    return idx, scores_bf


def _fused(st, x_tt, idx, scores_bf, num_experts, block):
    return ttnn.experimental.deepseek.moe.fused_experts(
        x_tt,
        routing_indices=idx,
        routing_scores=scores_bf,
        gate_up_weights=st["gate_up"],
        down_weights=st["down"],
        num_experts=num_experts,
        intermediate_size=I,
        swiglu_limit=LIMIT,
        top_k=TOP_K,
        routed_scaling_factor=SCALE,
        routing_eps=EPS,
        experts_block_size=block,
    )


def _x(st, rows):
    hf = st["inp"]["hf"][rows]
    return ttnn.from_torch(
        hf.reshape(1, 1, len(rows), H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=st["device"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _combine(y, idx, w, rows):
    """sum_j w[b, j] * Y[idx[b, j], row_b] for a [B, k] routing."""
    out = torch.zeros(len(rows), H, dtype=torch.float64)
    for b, r in enumerate(rows):
        for j in range(idx.shape[1]):
            out[b] += float(w[b, j]) * torch.from_numpy(np.asarray(y[int(idx[b, j]), r], dtype=np.float64))
    return out


def _record(row: dict):
    print("P2B " + json.dumps(row), flush=True)
    if EVIDENCE:
        os.makedirs(EVIDENCE, exist_ok=True)
        with open(os.path.join(EVIDENCE, "pcc.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")


def _windows(batch: int):
    return [list(range(s, s + batch)) for s in range(0, PROMPT, batch)]


def test_fused_experts_glm53_weights_layout(device):
    """The permuted gate_up shard of one expert on device equals the cache tensor with per-core [gate_c | up_c] tiles."""
    st = _state(device)
    man = json.load(open(os.path.join(GLM53_WS, "weights", "manifest.json")))["tensors"]
    import sys

    sys.path.insert(0, os.path.join(GLM53_WS, "weights"))
    import tile_codec

    e = 17
    ent = man[f"layers.{LAYER}.mlp.experts.{e}.gate_up_proj"]
    ref = torch.from_numpy(tile_codec.decode(os.path.join(CACHE_DIR, ent["path"]), ent["shape"], ent["dtype"], "TILE"))
    ref = ref.reshape(H, 2, I // 32, 32).permute(0, 2, 1, 3).reshape(H, 2 * I)
    got = ttnn.to_torch(st["gate_up"][e]).float()
    ent = man[f"layers.{LAYER}.mlp.experts.{e}.down_proj"]
    ref_dn = torch.from_numpy(
        tile_codec.decode(os.path.join(CACHE_DIR, ent["path"]), ent["shape"], ent["dtype"], "TILE")
    )
    got_dn = ttnn.to_torch(st["down"][e]).float()
    row = dict(
        kind="weights_layout",
        layer=LAYER,
        expert=e,
        gate_up_exact=bool(torch.equal(got, ref)),
        down_exact=bool(torch.equal(got_dn, ref_dn)),
        load_all_experts_s=round(st["load_s"], 1),
    )
    _record(row)
    if not RECORD_ONLY:
        assert row["gate_up_exact"] and row["down_exact"]


@pytest.mark.parametrize("batch", (8, 32), ids=lambda b: f"B{b}")
def test_fused_experts_glm53_pcc(device, batch):
    st = _state(device)
    inp, ys = st["inp"], st["ys"]
    failures = []
    for wi, rows in enumerate(_windows(batch)):
        x_tt = _x(st, rows)
        for routing in ("fp32_topk", "bf16_topk"):
            idx_tt, scores_tt = _route(st, x_tt, batch, routing)
            idx_dev = ttnn.to_torch(idx_tt).to(torch.int64).reshape(batch, TOP_K)
            scores_dev = ttnn.to_torch(scores_tt).float().reshape(batch, E)
            union = sorted(set(idx_dev.flatten().tolist()))
            modes = [("union", len(union))]
            if wi == 0 and routing == "fp32_topk":
                modes.append(("static_bound", min(E, batch * TOP_K)))
            for mode, num_experts in modes:
                assert num_experts >= len(union), (num_experts, len(union))
                block = min(BLOCK, num_experts)
                out = _fused(st, x_tt, idx_tt, scores_tt, num_experts, block)
                got = ttnn.to_torch(out).float().reshape(batch, H).double()
                ttnn.deallocate(out)

                idx_ref, w_ref = inp["idx"][rows], inp["w"][rows]
                ref_fp8 = _combine(ys["Y_fp8"], idx_ref, w_ref, rows)
                ref_bfp4 = _combine(ys["Y_bfp4"], idx_ref, w_ref, rows)
                sel = torch.gather(scores_dev, 1, idx_dev)
                w_dev = (SCALE * sel / (sel.sum(1, keepdim=True) + EPS)).bfloat16().float()
                ref_dev = _combine(ys["Y_bfp4_xbf16"], idx_dev, w_dev, rows)
                routed_ref = inp["routed"][rows].double()
                shared = inp["shared"][rows].double()
                agree = np.mean(
                    [len(set(idx_dev[b].tolist()) & set(idx_ref[b].tolist())) / TOP_K for b in range(batch)]
                )
                row = dict(
                    kind="glm_pcc",
                    layer=LAYER,
                    prompt=PROMPT,
                    batch=batch,
                    rows=[rows[0], rows[-1]],
                    routing=routing,
                    mode=mode,
                    union=len(union),
                    num_experts=num_experts,
                    experts_block_size=block,
                    pcc_vs_ref_fp8=pcc(got, ref_fp8),
                    pcc_vs_ref_bfp4=pcc(got, ref_bfp4),
                    pcc_vs_bfp4_device_routing=pcc(got, ref_dev),
                    min_row_pcc_vs_ref_bfp4=min(pcc(got[b], ref_bfp4[b]) for b in range(batch)),
                    min_row_pcc_vs_bfp4_device_routing=min(pcc(got[b], ref_dev[b]) for b in range(batch)),
                    moe_pcc_vs_ref_fp8_with_ref_shared=pcc(got + shared, routed_ref + shared),
                    ref_recon_vs_reference_routed=pcc(ref_fp8, routed_ref),
                    bfp4_weight_ceiling_pcc=pcc(ref_bfp4, ref_fp8),
                    expert_set_agreement_vs_ref=float(agree),
                    max_abs_err_vs_bfp4_device_routing=float((got - ref_dev).abs().max()),
                )
                _record(row)
                # bf16_topk ranks a bf16-rounded choice and flips near-ties (diagnostic only); the op itself is gated
                # on its own selection in both routing forms.
                gated = ("pcc_vs_bfp4_device_routing",) + (("pcc_vs_ref_bfp4",) if routing == "fp32_topk" else ())
                for key in gated:
                    if row[key] < PCC_BAR:
                        failures.append((rows[0], routing, mode, key, row[key]))
            for t in (idx_tt, scores_tt):
                ttnn.deallocate(t)
        ttnn.deallocate(x_tt)
    if not RECORD_ONLY:
        assert not failures, failures


@pytest.mark.parametrize("batch", (8, 32), ids=lambda b: f"B{b}")
def test_fused_experts_glm53_block_sweep(device, batch):
    """L1 footprint vs experts_block_size: run the first window at increasing block sizes until the CBs no longer fit."""
    st = _state(device)
    rows = _windows(batch)[0]
    x_tt = _x(st, rows)
    idx_tt, scores_tt = _route(st, x_tt, batch)
    idx_dev = ttnn.to_torch(idx_tt).to(torch.int64).reshape(batch, TOP_K)
    union = len(set(idx_dev.flatten().tolist()))
    base = None
    for block in list(range(1, 13)) + [0]:
        if block > union:
            continue
        row = dict(
            kind="block_sweep", layer=LAYER, batch=batch, union=union, num_experts=union, experts_block_size=block
        )
        try:
            out = _fused(st, x_tt, idx_tt, scores_tt, union, block)
            got = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            base = got if base is None else base
            row.update(fits=True, pcc_vs_block1=pcc(got, base))
        except RuntimeError as err:
            msg = str(err).splitlines()
            row.update(fits=False, error=next((m for m in msg if "L1" in m or "circular" in m.lower()), msg[0])[:400])
        _record(row)
    for t in (x_tt, idx_tt, scores_tt):
        ttnn.deallocate(t)


@pytest.mark.parametrize("batch", (1, 8, 32), ids=lambda b: f"B{b}")
def test_fused_experts_glm53_perf(device, batch):
    """Signposted warm runs for the device profiler: FE_B<batch>_<mode>_k<block> .. _END, PERF_REPS op calls each.

    GLM53_P2B_PERF_BLOCKS (comma list, default GLM53_P2B_BLOCK) lists the experts_block_size values to time."""
    st = _state(device)
    reps = int(os.environ.get("GLM53_P2B_PERF_REPS", "5"))
    blocks = [int(b) for b in os.environ.get("GLM53_P2B_PERF_BLOCKS", str(BLOCK)).split(",")]
    rows = _windows(batch)[0]
    x_tt = _x(st, rows)
    idx_tt, scores_tt = _route(st, x_tt, batch)
    idx_dev = ttnn.to_torch(idx_tt).to(torch.int64).reshape(batch, TOP_K)
    union = len(set(idx_dev.flatten().tolist()))
    for block in blocks:
        modes = [("union", union)] + ([("static_bound", min(E, batch * TOP_K))] if batch * TOP_K > union else [])
        for mode, num_experts in modes:
            blk = min(block, num_experts)
            for _ in range(2):
                ttnn.deallocate(_fused(st, x_tt, idx_tt, scores_tt, num_experts, blk))
            ttnn.synchronize_device(device)
            tag = f"FE_B{batch}_{mode}_k{blk}"
            signpost(tag)
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.deallocate(_fused(st, x_tt, idx_tt, scores_tt, num_experts, blk))
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) / reps
            signpost(f"{tag}_END")
            _record(
                dict(
                    kind="perf_wall",
                    layer=LAYER,
                    batch=batch,
                    mode=mode,
                    union=union,
                    num_experts=num_experts,
                    experts_block_size=blk,
                    reps=reps,
                    host_wall_us_per_call=round(dt * 1e6, 1),
                    weight_bytes_per_call=num_experts * (H * 2 * I + I * H) * 576 // 1024,
                    signpost=tag,
                )
            )
    for t in (x_tt, idx_tt, scores_tt):
        ttnn.deallocate(t)
