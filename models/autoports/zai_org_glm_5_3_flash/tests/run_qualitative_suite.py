# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""$qualitative-check shared-suite driver for GLM-5.3-Flash (stage 06a; Stage 6/7/9 rerun it on device).

Prompt format (decided from the checkpoint, recorded in `qualitative_prompt_format.json`): the tokenizer has a
non-empty `chat_template.jinja`, so the model is judged as chat/instruct: each suite prompt is one user message
rendered by `tt/tokenizer_chat.Glm53Tokenizer.render_chat` (token-for-token equal to transformers
`apply_chat_template(..., add_generation_prompt=True)`, template defaults = `Reasoning Effort: Max`, generation prompt
`<|assistant|><think>`). Raw completion prompts are never used for a verdict.

Suite: the plugin's shared prompts (`runtime/readiness_check/vllm_prompts.txt`, blank-line separated exactly as
`run_vllm_server` reads them). Controls: HF reference greedy continuations of the same rendered prompts - the fp32
layer-streamed GPU reference (`goldens_gpu_fp32_qualsuite`, 128 tokens, produced by stage 06a) - or the MockModel's
unpaged reference for host-only wiring runs.

Artifacts per run (under --out-dir): qualitative_prompt_format.json, rendered_prompts.jsonl (id, text, rendered
text, token ids), outputs.jsonl (TT and control token ids/text, parsed reasoning/content, first divergence,
duplication / loop metrics, control-token leakage), verdict.md (table + snippets for the human verdict),
degenerate_check.json (the plugin's check_degenerate_output.py over the TT completions).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness_common as H  # noqa: E402

SUITE_FILE = H.RUNTIME / "readiness_check" / "vllm_prompts.txt"
QUALSUITE_GOLDENS = H.WS / "goldens_gpu_fp32_qualsuite"
_WORD = re.compile(r"\w+", re.UNICODE)


def load_suite(path: Path = SUITE_FILE) -> list[dict]:
    prompts = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    return [{"id": f"q{i:02d}", "prompt": p} for i, p in enumerate(prompts)]


def word_metrics(text: str) -> dict:
    w = [x.lower() for x in _WORD.findall(text)]
    dup = sum(1 for a, b in zip(w, w[1:]) if a == b) / max(len(w) - 1, 1)
    grams = {}
    for i in range(len(w) - 2):
        grams[tuple(w[i : i + 3])] = grams.get(tuple(w[i : i + 3]), 0) + 1
    loop = (max(grams.values()) * 3 / len(w)) if grams and len(w) >= 3 else 0.0
    return {"words": len(w), "adjacent_duplication": round(dup, 4), "trigram_loop_fraction": round(loop, 4)}


def prompt_format(tok, suite_path: Path, gen: dict) -> dict:
    d = tok.describe()
    return {
        "hf_model": d["hf_model"],
        "hf_revision": d["hf_revision"],
        "tokenizer_class": d["tokenizer_class_hf"],
        "tokenizer_impl": d["tokenizer_impl"],
        "tokenizer_files_sha256": d["files_sha256"],
        "chat_template_present": d["chat_template_present"],
        "prompt_mode": "chat",
        "rendering": "Glm53Tokenizer.render_chat(messages=[{'role': 'user', 'content': prompt}], add_generation_prompt=True) "
        "== transformers apply_chat_template (tests/test_tokenizer_chat.py parity)",
        "template_kwargs": {"reasoning_effort": "template default (Max)", "clear_thinking": "template default (False)"},
        "generation_prompt_suffix": "<|assistant|><think>",
        "stop_token_ids": d["stop_token_ids"],
        "generation": gen,
        "prompt_source": str(suite_path),
        "prompt_source_sha256": H.sha256(suite_path),
    }


def run(
    generator,
    tok,
    suite: list[dict],
    out_dir: Path,
    *,
    max_new_tokens: int,
    controls: dict | None,
    control_source: str,
    label: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_params = {"sampling": "greedy", "max_new_tokens": max_new_tokens, "stop_on_eos": True, "enable_trace": True}
    H.write_json(
        out_dir / "qualitative_prompt_format.json",
        {**prompt_format(tok, SUITE_FILE, gen_params), "suite_label": label, "control_source": control_source},
    )
    rendered, outputs = [], []
    t0 = time.time()
    for item in suite:
        messages = [{"role": "user", "content": item["prompt"]}]
        text = tok.render_chat(messages)
        ids = tok.encode(text)
        rendered.append({"id": item["id"], "prompt": item["prompt"], "rendered": text, "token_ids": ids})
        tt = generator.generate(prompt_token_ids=ids, max_new_tokens=max_new_tokens, enable_trace=True)
        generator.reset()
        tt_text = H.safe_decode(tok, tt)
        parsed = tok.parse_assistant(tt) if all(0 <= t < tok.vocab_size for t in tt) else None
        ctrl = controls.get(item["id"]) if controls else None
        row = {
            "id": item["id"],
            "prompt": item["prompt"],
            "tt_token_ids": tt,
            "tt_text": tt_text,
            "tt_stop": "stop_token" if tt and tt[-1] in generator.stop_token_ids else "max_new_tokens",
            "tt_reasoning_head": (parsed.reasoning_content or "")[:400] if parsed else None,
            "tt_content_head": (parsed.content or "")[:400] if parsed else None,
            "control_token_ids": ctrl,
            "control_text": H.safe_decode(tok, ctrl) if ctrl is not None else None,
            "first_divergence": H.first_divergence(ctrl, tt) if ctrl is not None else None,
            "exact_match": (ctrl == tt) if ctrl is not None else None,
            "control_token_leak": [t for t in tt[:-1] if t in tok.special_token_ids],
            **word_metrics(tt_text),
        }
        outputs.append(row)
    with open(out_dir / "rendered_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in rendered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "outputs.jsonl", "w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [
        f"# Qualitative suite: {label}",
        "",
        f"control: {control_source}; greedy, max_new_tokens {max_new_tokens}",
        "",
        "| id | prompt | TT tokens | stop | exact match vs control | first divergence | adj dup | trigram loop | TT head |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ]
    for r in outputs:
        head = r["tt_text"][:80].replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {r['id']} | {r['prompt'][:40]} | {len(r['tt_token_ids'])} | {r['tt_stop']} | {r['exact_match']} | "
            f"{r['first_divergence']} | {r['adjacent_duplication']} | {r['trigram_loop_fraction']} | {head} |"
        )
    (out_dir / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    deg_in = H.write_json(
        out_dir / "degenerate_input.json", [{"prompt": r["prompt"], "greedy_completion": r["tt_text"]} for r in outputs]
    )
    deg = H.degenerate_check([deg_in], out_dir / "degenerate_check.json")
    n_ctrl = sum(1 for r in outputs if r["exact_match"] is not None)
    return {
        "label": label,
        "prompts": len(outputs),
        "controls": n_ctrl,
        "exact_matches": sum(1 for r in outputs if r["exact_match"]),
        "first_divergences": {r["id"]: r["first_divergence"] for r in outputs},
        "max_adjacent_duplication": max(r["adjacent_duplication"] for r in outputs),
        "control_token_leaks": sum(len(r["control_token_leak"]) for r in outputs),
        "degenerate_check_exit": deg["exit_code"],
        "elapsed_s": round(time.time() - t0, 2),
        "out_dir": str(out_dir),
    }


def suite_controls(tok, max_new_tokens: int) -> tuple[dict, list]:
    """HF (fp32 GPU reference) greedy controls of the suite, truncated at the first stop token, keyed by suite id."""
    index = json.load(open(QUALSUITE_GOLDENS / "suite_index.json"))
    ctrl, lens = {}, []
    for e in index:
        tf = torch.load(QUALSUITE_GOLDENS / "teacher_forcing" / f"{e['T']}.pt")
        ctrl[e["suite_id"]] = H.truncate_at_stop(
            tf["targets"].reshape(-1).tolist(), set(tok.stop_token_ids), max_new_tokens
        )
        lens.append(e["T"])
    return ctrl, lens


def golden_run(out_dir: Path, *, max_new_tokens=128, perturb_every=0) -> dict:
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    tok = load_tokenizer()
    ctrl, lens = suite_controls(tok, max_new_tokens)
    be = H.golden_replay_backend([], perturb_every=perturb_every, max_batch=1, extra_dirs=[(QUALSUITE_GOLDENS, lens)])
    gen = build_generator(
        H.MODEL_DIR, None, backend=be, tokenizer=tok, max_batch=1, max_seq_len=max(lens) + max_new_tokens + 64
    )
    res = run(
        gen,
        tok,
        load_suite(),
        out_dir,
        max_new_tokens=max_new_tokens,
        controls=ctrl,
        control_source="HF reference: fp32 layer-streamed GPU greedy (goldens_gpu_fp32_qualsuite)",
        label=f"golden_replay perturb_every={perturb_every}",
    )
    gen.teardown()
    res["backend"] = {
        "kind": "golden_replay",
        "perturb_every": perturb_every,
        "mismatched_prefixes": be.mismatches,
        "perturbed_positions": len(be.perturbed_positions),
    }
    return res


def mock_run(out_dir: Path, *, max_new_tokens=32) -> dict:
    import mock_model as M

    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    tok = load_tokenizer()
    be = M.MockModel(max_batch=1)
    gen = build_generator(H.MODEL_DIR, None, backend=be, tokenizer=tok, max_batch=1, max_seq_len=2051)
    suite = load_suite()
    ctrl = {}
    for item in suite:
        ids = tok.encode_chat([{"role": "user", "content": item["prompt"]}])
        ctrl[item["id"]] = be.reference_greedy(ids, max_new_tokens, stop_ids=gen.stop_token_ids)
    res = run(
        gen,
        tok,
        suite,
        out_dir,
        max_new_tokens=max_new_tokens,
        controls=ctrl,
        control_source="MockModel.reference_greedy (wiring control; mock text is not language)",
        label="mock",
    )
    gen.teardown()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("golden_replay", "mock"), required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--perturb-every", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    if a.backend == "golden_replay":
        res = golden_run(a.out_dir, max_new_tokens=a.max_new_tokens, perturb_every=a.perturb_every)
    else:
        res = mock_run(a.out_dir, max_new_tokens=min(a.max_new_tokens, 32))
    H.write_json(a.out_dir / "summary.json", res)
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
