# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 06a harness controls (host only; no device). Every harness runs end to end against
  * the golden-replay backend (fp32 goldens): positive control = top-1 1.0 / identical completions; a perturbed
    replay (every 3rd continuation row's argmax moved outside the golden top-100) is the negative control and must
    fail the gate / diverge;
  * the MockModel (Stage-1-contract MockDecoders) against its independent unpaged reference: positive control, and a
    padding-leak decoder bug as negative control.
Artifacts go to $GLM53_06A_EVIDENCE/logs/harness/ (temporary directory when unset).

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_harnesses.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness_common as H  # noqa: E402
import run_autoregressive as AR  # noqa: E402
import run_prefill_check as PC  # noqa: E402
import run_qualitative_suite as QS  # noqa: E402
import run_teacher_forcing as TF  # noqa: E402
from host_evidence import record  # noqa: E402

EVID = Path(os.environ.get("GLM53_06A_EVIDENCE") or tempfile.mkdtemp(prefix="glm53_06a_")) / "logs" / "harness"
TF_PROMPTS = (128, 512, 2048, 4096, 32768)
PREFILL_PROMPTS = (128, 512, 2048, 4096)  # 32768: [1, 33023, 154880] fp32 logits = 20 GB, outside the host budget


def _short(res):
    return {
        "aggregate": res["aggregate"],
        "gate": res["gate"],
        "backend": res["backend"],
        "elapsed_s": res["elapsed_s"],
        "per_reference": [
            {
                "reference": Path(f["reference"]).name,
                "top1": [round(r["top1"], 4) for r in f["per_entry"]],
                "total": [r["total"] for r in f["per_entry"]],
            }
            for f in res["references"]
        ],
    }


# ------------------------------------------------------------------------------------------------ teacher forcing
def test_teacher_forcing_positive_control():
    res = TF.golden_run(TF_PROMPTS, aime24=True, workdir=EVID / "refs")
    H.write_json(EVID / "teacher_forcing_positive_control.json", res)
    a = res["aggregate"]
    assert a["top1"] == 1.0 and a["top5"] == 1.0 and a["top100"] == 1.0 and res["gate"]["pass"]
    assert res["backend"]["mismatched_prefixes"] == 0
    record("teacher_forcing_positive_control", expected="top-1 = 1.0", **_short(res))


def test_teacher_forcing_negative_control():
    res = TF.golden_run((128, 512), aime24=True, perturb_every=3, workdir=EVID / "refs")
    H.write_json(EVID / "teacher_forcing_negative_control.json", res)
    assert not res["gate"]["pass"] and res["aggregate"]["top1"] < 0.9 and res["aggregate"]["top100"] < 1.0
    record(
        "teacher_forcing_negative_control",
        expected="gate FAILS (perturbed replay)",
        control_behaved=True,
        **_short(res),
    )


def test_teacher_forcing_mock():
    pos = TF.mock_run([1, 31, 33, 129, 2052, 4097], workdir=EVID / "refs")
    neg = TF.mock_run([33, 129], bug="pad_leak", workdir=EVID / "refs")
    H.write_json(EVID / "teacher_forcing_mock.json", {"positive": pos, "negative_pad_leak": neg})
    assert pos["aggregate"]["top1"] == 1.0 and pos["gate"]["pass"]
    assert neg["aggregate"]["top1"] < 1.0  # mock controls are exact: any miss is a detection
    record(
        "teacher_forcing_mock_controls",
        criterion="positive top-1 == 1.0, negative top-1 < 1.0 (gate verdict reported)",
        positive=_short(pos),
        negative_pad_leak=_short(neg),
    )


# ------------------------------------------------------------------------------------------------ prefill
def test_prefill_check_positive_control():
    res = PC.golden_run(PREFILL_PROMPTS, aime24=True, workdir=EVID / "refs")
    H.write_json(EVID / "prefill_check_positive_control.json", res)
    a = res["aggregate"]
    assert (
        a["top1"] == 1.0 and a["top100"] == 1.0 and res["gate"]["pass"] and res["backend"]["mismatched_prefixes"] == 0
    )
    record(
        "prefill_check_positive_control",
        expected="top-1 = 1.0",
        skipped_prompts={"32768": "20 GB of fp32 logits"},
        **_short(res),
    )


def test_prefill_check_negative_control():
    res = PC.golden_run((128,), aime24=True, perturb_every=3, workdir=EVID / "refs")
    H.write_json(EVID / "prefill_check_negative_control.json", res)
    assert not res["gate"]["pass"]
    record(
        "prefill_check_negative_control", expected="gate FAILS (perturbed replay)", control_behaved=True, **_short(res)
    )


def test_prefill_check_mock():
    """pad_leak cannot show in prefill logits (padding follows the valid rows of the last chunk; it corrupts only the
    carried state, which the teacher-forcing control catches), so the prefill negative control is a causal leak."""
    pos = PC.mock_run([1, 31, 33, 129, 2052, 4097], workdir=EVID / "refs")
    neg = PC.mock_run([33, 129], bug="causal_leak", workdir=EVID / "refs")
    H.write_json(EVID / "prefill_check_mock.json", {"positive": pos, "negative_causal_leak": neg})
    assert pos["aggregate"]["top1"] == 1.0 and neg["aggregate"]["top1"] < 1.0
    record(
        "prefill_check_mock_controls",
        criterion="positive top-1 == 1.0, negative top-1 < 1.0 (gate verdict reported)",
        positive=_short(pos),
        negative_causal_leak=_short(neg),
    )


# ------------------------------------------------------------------------------------------------ autoregressive
def test_autoregressive_controls():
    rows = {}
    for T, n in ((128, 256), ("aime24", 100)):
        r = AR.golden_run(T, EVID / "autoregressive" / f"golden_{T}", max_new_tokens=n)
        assert r["exact_match"] and r["degenerate_check"]["exit_code"] == 0, (T, r)
        rows[f"positive_{T}"] = {k: r[k] for k in ("exact_match", "first_divergence", "hf_tokens", "tt_tokens")} | {
            "degenerate_exit": r["degenerate_check"]["exit_code"],
            "tt_head": r["tt_head"][:160],
        }
    neg = AR.golden_run(128, EVID / "autoregressive" / "golden_128_perturbed", max_new_tokens=64, perturb_every=3)
    assert not neg["exact_match"] and neg["first_divergence"] == 2
    rows["negative_perturbed_128"] = {k: neg[k] for k in ("exact_match", "first_divergence")}
    mock = AR.mock_run([1, 33, 2052], EVID / "autoregressive" / "mock", max_new_tokens=24)
    assert mock["all_match"]
    rows["mock"] = mock
    record("autoregressive_hf_vs_tt_controls", **rows)


def test_golden_replay_multislot_permuted_pages():
    """Golden replay at batch 4: four fp32 golden prompts (128, 512, 2048, AIME24 161) in shuffled slots with every
    physical block permuted across slots, decoded together. The replay checks the exact token prefix of every row at
    every position through its own paged token cache, so identical completions and 0 mismatched prefixes prove slot
    routing, page-table indirection, positions and token feedback at batch > 1."""
    import torch

    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    be = H.golden_replay_backend([128, 512, 2048], include_aime24=True, max_batch=4)
    gen = Glm53Generator(be, max_batch=4, max_seq_len=2048 + 128 + 64, stop_token_ids=[154820, 154827, 154829])
    entries = be.entries
    prompts = [e.prompt_ids.tolist() for e in entries]
    n = 64
    slots = [2, 0, 3, 1]
    b = gen._own_binding()
    perm = torch.randperm(4 * b.blocks_per_seq, generator=torch.Generator().manual_seed(0)).reshape(4, b.blocks_per_seq)
    outs = gen.generate_batch(prompts, n, slots=slots, page_table=perm)
    exp = [H.truncate_at_stop(e.targets.tolist(), {154820, 154827, 154829}, n) for e in entries]
    assert outs == exp and be.mismatches == 0
    assert torch.equal(be.read(b.io.page_table).long()[slots], perm[slots])
    record(
        "golden_replay_multislot_permuted_pages",
        prompts=[len(p) for p in prompts],
        slots=slots,
        tokens_each=n,
        mismatched_prefixes=be.mismatches,
        lookups=be.lookups,
        counters=gen.counters,
    )


# ------------------------------------------------------------------------------------------------ qualitative suite
def test_qualitative_suite_driver_mock():
    res = QS.mock_run(EVID / "qualitative" / "mock")
    assert res["prompts"] == 6 and res["exact_matches"] == 6 and res["control_token_leaks"] == 0
    assert res["degenerate_check_exit"] in (0, 1)  # mock text is not language; no critical mechanical finding
    record("qualitative_suite_driver_mock", **res)


def test_qualitative_suite_driver_golden_replay():
    if not (QS.QUALSUITE_GOLDENS / "manifest.json").exists():
        if os.environ.get("GLM53_06A_REQUIRE_QUALSUITE") == "1":
            pytest.fail("goldens_gpu_fp32_qualsuite/manifest.json missing (GPU suite controls not captured)")
        pytest.skip("suite HF controls not captured yet")
    pos = QS.golden_run(EVID / "qualitative" / "golden_replay", max_new_tokens=128)
    assert pos["exact_matches"] == pos["prompts"] == 6 and pos["backend"]["mismatched_prefixes"] == 0
    assert pos["degenerate_check_exit"] == 0
    neg = QS.golden_run(EVID / "qualitative" / "golden_replay_perturbed", max_new_tokens=32, perturb_every=3)
    assert neg["exact_matches"] == 0
    record("qualitative_suite_driver_golden_replay", positive=pos, negative_perturbed=neg)
