# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of `tt/sampling.py` (stage 06a): host reference vs transformers' warpers, greedy tie-break,
the device candidate algorithm (split top-32 -> k/p/temp) vs full-vocab sampling, seeded sampling, and the
checkpoint-default (temperature 1.0, top_p 0.95, no top_k) nucleus measured on the fp32 golden logits.

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_sampling.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from host_evidence import expect_error, record  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import sampling as S  # noqa: E402

WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
PARAMS = [
    S.SamplingParams(temperature=0.7, top_k=None, top_p=1.0),
    S.SamplingParams(temperature=1.0, top_k=20, top_p=1.0),
    S.SamplingParams(temperature=1.0, top_k=None, top_p=0.95),
    S.SamplingParams(temperature=0.6, top_k=32, top_p=0.9),
    S.SamplingParams(temperature=1.3, top_k=5, top_p=0.5),
]


def test_warp_matches_transformers():
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 154880, generator=g) * 4
    for p in PARAMS:
        ref = x.clone()
        ids = torch.zeros(4, 1, dtype=torch.long)
        if p.temperature not in (0.0, 1.0):
            ref = TemperatureLogitsWarper(p.temperature)(ids, ref)
        if p.top_k:
            ref = TopKLogitsWarper(p.top_k)(ids, ref)
        if p.top_p < 1.0:
            ref = TopPLogitsWarper(p.top_p)(ids, ref)
        ours = S.warp(x, p)
        assert torch.equal(torch.isfinite(ours), torch.isfinite(ref)), p
        assert torch.allclose(ours.softmax(-1), ref.softmax(-1), atol=1e-7), p
    record("sampling_warp_matches_transformers", params=[vars(p) for p in PARAMS])


def test_greedy_tie_break_lowest_id():
    x = torch.zeros(3, 154880)
    x[0, [5, 77, 154000]] = 3.0
    x[1, [154879, 40000]] = 1.0
    x[2] -= 5
    x[2, 9] = -1.0
    assert S.greedy(x).tolist() == [5, 40000, 9]
    for n_shards in (4, 8):
        for r in range(3):
            tok, _, _ = S.candidate_sample_reference(x[r], S.GREEDY, n_shards=n_shards)
            assert int(tok) == int(S.greedy(x[r]))
    record("sampling_greedy_tie_break", rule="lowest token id among tied maxima (models/common/sampling tie-break)")


def test_candidate_path_equals_full_vocab_for_small_k():
    """For k <= 32 the union of per-split top-32 candidates contains the global top-k, so the device algorithm's
    distribution equals the host reference exactly (any temperature / top-p); greedy is exact."""
    g = torch.Generator().manual_seed(1)
    rows = 0
    for n_shards in (S.vocab_splits_single_device(154880), 8):
        for _ in range(4):
            x = torch.randn(154880, generator=g) * 3
            for p in PARAMS:
                if p.top_k is None or p.top_k > 32:
                    continue
                _, ci, cp = S.candidate_sample_reference(
                    x, p, n_shards=n_shards, generator=torch.Generator().manual_seed(0)
                )
                full = S.probabilities(x, p)
                assert torch.allclose(full[ci], cp.float(), atol=1e-6), (n_shards, p)
                assert abs(float(full.sum() - cp.sum())) < 1e-5
                rows += 1
    record("sampling_candidate_path_exact_k_le_32", shard_counts=[4, 8], cases=rows)


def test_host_sampler_seeded_distribution():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(1, 1000, generator=g) * 2
    p = S.SamplingParams(temperature=0.9, top_k=10, top_p=1.0, seed=5)
    hs = S.HostSampler(1)
    hs.reset_slot(0, 5)
    draws = torch.stack([hs.sample(logits, [p], [0]) for _ in range(20000)]).reshape(-1)
    probs = S.probabilities(logits[0], p)
    support = torch.nonzero(probs).reshape(-1)
    assert set(draws.tolist()) <= set(support.tolist())
    counts = torch.bincount(draws, minlength=1000)[support].double()
    expected = probs[support].double() * draws.numel()
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 40.0, chi2  # 9 dof: p ~ 5e-6 threshold
    hs2 = S.HostSampler(1)
    hs2.reset_slot(0, 5)
    assert torch.equal(torch.stack([hs2.sample(logits, [p], [0]) for _ in range(200)]).reshape(-1), draws[:200])
    record("sampling_host_seeded_distribution", draws=20000, chi2_9dof=chi2, support=len(support))


def test_params_defaults_and_layout_facts():
    gen = json.load(
        open(
            os.path.join(
                os.environ.get(
                    "GLM53_HF_SNAPSHOT",
                    "/opt/glm53/hf/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
                ),
                "generation_config.json",
            )
        )
    )
    d = S.SamplingParams.from_generation_config(gen)
    assert d == S.CHECKPOINT_DEFAULT and not d.is_greedy and d.device_values() == (32, 0.95, 1.0)
    assert S.GREEDY.device_values() == (1, 0.0, 1.0) and S.GREEDY.device_mode == "greedy"
    assert S.SamplingParams(temperature=0.5, top_k=1).is_greedy
    with expect_error(ValueError):
        S.SamplingParams(top_p=0.0)
    assert S.vocab_splits_single_device(154880) == 4 and 154880 // 4 == 38720
    assert S.tp_padded_vocab(154880, 8) == 154880 and 154880 // 8 == 19360
    assert S.SAMPLER_PLAN["chosen"].startswith("models/common/sampling")
    x = torch.zeros(155648)
    x[154900] = 99.0
    assert int(S.greedy(S.mask_padded_vocab(x, 154880))) != 154900
    record(
        "sampling_defaults_and_layout",
        checkpoint_default=vars(d),
        single_chip_splits=4,
        tp8_cols_per_chip=19360,
        plan=S.SAMPLER_PLAN,
    )


def test_checkpoint_default_nucleus_on_golden_logits():
    """Measurement for the sampler plan: the checkpoint default (T 1.0, top_p 0.95, no top_k) on the fp32 golden
    continuation logits. The device path samples the top-p nucleus of the top-32 candidates; report how often the
    full nucleus is larger and the total-variation distance between the two distributions."""
    root = os.path.join(WS, "goldens_gpu_fp32")
    man = json.load(open(os.path.join(root, "manifest.json")))
    sizes, tvs, outside = [], [], []
    files = sorted(k for k in man["files"] if k.startswith("logits/") and k.endswith(".pt"))
    extra = os.path.join(WS, "goldens_gpu_fp32_aime24", "logits", "161.pt")
    paths = [os.path.join(root, k) for k in files] + ([extra] if os.path.exists(extra) else [])
    for path in paths:
        cont = torch.load(path)["continuation_logits"].float()
        for row in cont:
            full = S.probabilities(row, S.CHECKPOINT_DEFAULT)
            k, p, t = S.CHECKPOINT_DEFAULT.device_values()
            _, ci, cp = S.candidate_sample_reference(
                row, S.CHECKPOINT_DEFAULT, n_shards=4, generator=torch.Generator().manual_seed(0)
            )
            dev = torch.zeros_like(full)
            dev[ci] = cp.float()
            n = int((full > 0).sum())
            sizes.append(n)
            tvs.append(0.5 * float((full - dev).abs().sum()))
            outside.append(float(full.sum() - full[ci].sum()))
    sizes_t = torch.tensor(sizes, dtype=torch.float64)
    tv = torch.tensor(tvs, dtype=torch.float64)
    res = {
        "steps": len(sizes),
        "files": [os.path.relpath(p, WS) for p in paths],
        "nucleus_size_median": float(sizes_t.median()),
        "nucleus_size_p90": float(sizes_t.quantile(0.9)),
        "nucleus_size_max": int(sizes_t.max()),
        "frac_nucleus_gt_32": float((sizes_t > 32).double().mean()),
        "frac_nucleus_gt_128": float((sizes_t > 128).double().mean()),
        "tv_device_vs_full_mean": float(tv.mean()),
        "tv_device_vs_full_max": float(tv.max()),
        "mass_outside_top32_mean": float(torch.tensor(outside).mean()),
    }
    assert res["steps"] > 1000 and 0.0 <= res["tv_device_vs_full_mean"] <= 1.0
    record("sampling_checkpoint_default_vs_device_candidates", **res)
