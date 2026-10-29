# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Free-running (autoregressive) HF-vs-TT comparison for GLM-5.3-Flash: the TT generator's greedy
`generate(next_input=None)` (stops at the checkpoint stop set) side by side with the HF reference completion of the
same chat-template prompt. Writes the plugin's artifact format (`hf_completion.txt`, `tt_completion.txt`,
`autoregressive_meta.json` with hf/tt token ids) plus the first divergence index, and runs the plugin's
`check_degenerate_output.py` on it.

HF side: the fp32 reference-of-record greedy continuation (goldens_gpu_fp32 teacher_forcing/<T>.pt targets, the
registered AIME24 reference, or the suite controls) - the 321B model cannot run `AutoModelForCausalLM.generate` on
this host; the upstream `readiness_check.run_autoregressive` (which loads the HF model) is the Stage 6 command when an
HF-capable host is available. Backends: golden_replay (positive control: identical completions; `--perturb-every k`:
negative control, must diverge), mock (MockModel vs its unpaged reference).

Evidence goes under the stage evidence directory, never under models/autoports/<model>/ (Stage 6's gate scans that
tree for its own device artifacts; a mock/replay artifact there would be a stale pass).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness_common as H  # noqa: E402


def run(
    generator,
    tokenizer,
    prompt_ids,
    hf_tokens,
    out_dir: Path,
    *,
    max_new_tokens: int,
    hf_source: str,
    hf_model_id: str = "zai-org/GLM-5.3-Flash",
    prompt_label: str = "",
    run_degenerate_check: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tt_tokens = generator.generate(
        prompt_token_ids=list(prompt_ids), max_new_tokens=max_new_tokens, next_input=None, enable_trace=True
    )
    elapsed = time.time() - t0
    hf_text = H.safe_decode(tokenizer, hf_tokens)
    tt_text = H.safe_decode(tokenizer, tt_tokens)
    (out_dir / "hf_completion.txt").write_text(hf_text, encoding="utf-8")
    (out_dir / "tt_completion.txt").write_text(tt_text, encoding="utf-8")
    div = H.first_divergence(hf_tokens, tt_tokens)
    meta = {
        "hf_model_id": hf_model_id,
        "prompt_file": prompt_label,
        "prompt_text": H.safe_decode(tokenizer, prompt_ids),
        "prompt_token_ids": list(map(int, prompt_ids)),
        "max_new_tokens": max_new_tokens,
        "hf": {"token_ids": list(map(int, hf_tokens)), "num_tokens": len(hf_tokens), "source": hf_source},
        "tt": {"token_ids": list(map(int, tt_tokens)), "num_tokens": len(tt_tokens)},
        "first_divergence": div,
        "exact_match": div is None,
        "prompt_format": {"mode": "chat", "renderer": "HF chat template (tt/tokenizer_chat parity-tested)"},
        "stop_token_ids": sorted(getattr(generator, "stop_token_ids", ())),
        "elapsed_s": round(elapsed, 3),
        "host_work_counters": generator.host_work_report(),
    }
    H.write_json(out_dir / "autoregressive_meta.json", meta)
    res = {k: meta[k] for k in ("first_divergence", "exact_match", "max_new_tokens")}
    res.update(
        {
            "hf_tokens": len(hf_tokens),
            "tt_tokens": len(tt_tokens),
            "out_dir": str(out_dir),
            "tt_head": tt_text[:300],
            "hf_head": hf_text[:300],
        }
    )
    if run_degenerate_check:
        res["degenerate_check"] = H.degenerate_check([out_dir], out_dir / "degenerate_check.json")
    return res


def golden_run(T, out_dir: Path, *, max_new_tokens=256, perturb_every=0) -> dict:
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    tok = load_tokenizer()
    if T == "aime24":
        from readiness_check.schema import load_reference

        ref = load_reference(H.registered_aime24()).entries[0]
        prompt, targets = ref.prompt_tokens[0].tolist(), ref.generated_tokens[0].tolist()
        be = H.golden_replay_backend([], include_aime24=True, perturb_every=perturb_every, max_batch=1)
        label = str(H.registered_aime24())
    else:
        tf = torch.load(H.GOLDENS_FP32 / "teacher_forcing" / f"{T}.pt")
        prompt, targets = tf["prompt_ids"].reshape(-1).tolist(), tf["targets"].reshape(-1).tolist()
        be = H.golden_replay_backend([T], perturb_every=perturb_every, max_batch=1)
        label = f"goldens_gpu_fp32/teacher_forcing/{T}.pt"
    max_new_tokens = min(max_new_tokens, len(targets))
    gen = build_generator(
        H.MODEL_DIR, None, backend=be, tokenizer=tok, max_batch=1, max_seq_len=len(prompt) + max_new_tokens + 64
    )
    hf = H.truncate_at_stop(targets, gen.stop_token_ids, max_new_tokens)
    res = run(
        gen,
        tok,
        prompt,
        hf,
        out_dir,
        max_new_tokens=max_new_tokens,
        prompt_label=label,
        hf_source="fp32 layer-streamed GPU reference greedy continuation (reference of record)",
    )
    gen.teardown()
    res["backend"] = {
        "kind": "golden_replay",
        "prompt": T,
        "perturb_every": perturb_every,
        "mismatched_prefixes": be.mismatches,
        "perturbed_positions": len(be.perturbed_positions),
    }
    return res


def mock_run(lengths, out_root: Path, *, max_new_tokens=48, bug=None) -> dict:
    import mock_model as M

    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    tok = load_tokenizer()
    be = M.MockModel(max_batch=1, dsa_mode="unbounded", bug=bug)
    gen = build_generator(
        H.MODEL_DIR, None, backend=be, tokenizer=tok, max_batch=1, max_seq_len=max(lengths) + max_new_tokens + 64
    )
    rows = []
    for n in lengths:
        p = torch.randint(0, M.TOKENIZER_VOCAB, (n,), generator=torch.Generator().manual_seed(n)).tolist()
        hf = be.reference_greedy(p, max_new_tokens, stop_ids=gen.stop_token_ids)
        r = run(
            gen,
            tok,
            p,
            hf,
            out_root / f"mock_{n}",
            max_new_tokens=max_new_tokens,
            prompt_label=f"mock {n}",
            hf_source="MockModel.reference_greedy (unpaged full-sequence reference)",
            hf_model_id="mock:MockModel",
            run_degenerate_check=False,
        )
        rows.append({"len": n, **{k: r[k] for k in ("first_divergence", "exact_match", "tt_tokens")}})
        gen.reset()
    gen.teardown()
    return {"backend": {"kind": "mock", "bug": bug}, "rows": rows, "all_match": all(r["exact_match"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("golden_replay", "mock"), required=True)
    ap.add_argument("--prompt", default="128", help="golden prompt length or 'aime24'")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--perturb-every", type=int, default=0)
    ap.add_argument("--mock-lengths", default="1,33,2052")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.backend == "golden_replay":
        T = a.prompt if a.prompt == "aime24" else int(a.prompt)
        res = golden_run(T, a.out_dir, max_new_tokens=a.max_new_tokens, perturb_every=a.perturb_every)
    else:
        res = mock_run([int(x) for x in a.mock_lengths.split(",")], a.out_dir)
    H.write_json(a.out_dir / "summary.json", res)
    print({k: v for k, v in res.items() if k not in ("tt_head", "hf_head")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
