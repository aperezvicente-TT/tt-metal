# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness prefill check for GLM-5.3-Flash: top-1 / top-5 / top-100 of `generator.prefill_logits` at every
generated position of a readiness_v1 reference, through the plugin's own runner
(`readiness_check.run_prefill_check.run_prefill_check`, which imports `tt/generator.py::build_generator`).

Backends: `golden_replay` (fp32 goldens; positive control, `--perturb-every k` = negative control), `mock`
(MockModel vs its independent reference). Stage 6 runs the same runner on device:
    python -m readiness_check.run_prefill_check --model-dir models/autoports/zai_org_glm_5_3_flash \
        --reference <refpt> --mesh-device <mesh>

    no-tt-devices python models/autoports/zai_org_glm_5_3_flash/tests/run_prefill_check.py \
        --backend golden_replay --prompts 128,512,2048,4096 --aime24 --out <json>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness_common as H  # noqa: E402


def run(reference_paths: list[Path], build_kwargs: dict, model_dir: Path = H.MODEL_DIR) -> dict:
    from readiness_check.run_prefill_check import run_prefill_check

    per, files = [], []
    t0 = time.time()
    for ref in reference_paths:
        rows = run_prefill_check(
            model_dir=model_dir, reference_path=Path(ref), mesh_device=None, build_kwargs=build_kwargs
        )
        per.extend(rows)
        files.append({"reference": str(ref), "per_entry": rows})
    agg = H.aggregate(per)
    return {
        "runner": "readiness_check.run_prefill_check",
        "references": files,
        "aggregate": agg,
        "gate": H.gate(agg),
        "elapsed_s": round(time.time() - t0, 2),
    }


def golden_run(prompts, *, aime24=False, perturb_every=0, workdir: Path | None = None) -> dict:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="glm53_06a_prefill_"))
    refs = [H.golden_refpt(T, workdir / f"golden_{T}.refpt") for T in prompts]
    if aime24:
        refs.append(H.registered_aime24())
    be = H.golden_replay_backend(prompts, perturb_every=perturb_every, include_aime24=aime24, max_batch=1)
    max_len = max([T + 256 for T in prompts] + ([161 + 100] if aime24 else []))
    res = run(refs, H.build_kwargs(be, max_batch=1, max_seq_len=max_len + 64))
    res["backend"] = {
        "kind": "golden_replay",
        "prompts": list(prompts),
        "aime24": aime24,
        "perturb_every": perturb_every,
        "mismatched_prefixes": be.mismatches,
        "beyond_reference_rows": be.beyond,
        "perturbed_positions": len(be.perturbed_positions),
    }
    return res


def mock_run(lengths, gen_len=12, *, bug=None, dsa_mode="unbounded", workdir: Path | None = None) -> dict:
    import mock_model as M
    import torch

    workdir = workdir or Path(tempfile.mkdtemp(prefix="glm53_06a_prefill_mock_"))
    ref_model = M.MockModel(max_batch=1, dsa_mode=dsa_mode)
    prompts = [
        torch.randint(0, M.TOKENIZER_VOCAB, (n,), generator=torch.Generator().manual_seed(n)).tolist() for n in lengths
    ]
    ref = H.mock_refpt(ref_model, prompts, gen_len, workdir / "mock.refpt")
    be = M.MockModel(max_batch=1, dsa_mode=dsa_mode, bug=bug)
    res = run([ref], H.build_kwargs(be, max_batch=1, max_seq_len=max(lengths) + gen_len + 64, stop_token_ids=()))
    res["backend"] = {"kind": "mock", "lengths": list(lengths), "gen_len": gen_len, "bug": bug, "dsa_mode": dsa_mode}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("golden_replay", "mock"), required=True)
    ap.add_argument("--prompts", default="128,512,2048,4096")
    ap.add_argument("--aime24", action="store_true")
    ap.add_argument("--perturb-every", type=int, default=0)
    ap.add_argument("--mock-lengths", default="1,31,33,129,2052")
    ap.add_argument("--bug", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.backend == "golden_replay":
        res = golden_run([int(x) for x in a.prompts.split(",") if x], aime24=a.aime24, perturb_every=a.perturb_every)
    else:
        res = mock_run([int(x) for x in a.mock_lengths.split(",")], bug=a.bug)
    H.write_json(a.out, res)
    print({"aggregate": res["aggregate"], "gate": res["gate"]["pass"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
