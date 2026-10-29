# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the stage-06a readiness harnesses (`run_prefill_check`, `run_teacher_forcing`,
`run_autoregressive`, `run_qualitative_suite`): reference files, backends, gates.

References (readiness_v1 `.refpt`, plugin `readiness_check.schema`):
  * `golden_refpt`: built from `$GLM53_WS/goldens_gpu_fp32/{teacher_forcing,logits}/<T>.pt` per its manifest
    (sha256 verified): prompt = the chat-template golden prompt, generated = the fp32 greedy continuation,
    topk = top-100 of the fp32 logits that predicted each continuation token.
  * the registered AIME24 chat reference (`goldens_gpu_fp32/manifest.json` -> registered_references).
  * `mock_refpt`: the same format from `MockModel.reference_logits` (independent unpaged full-sequence forward).

Gates: readiness bar (upstream Stage 6 goal) top-5 >= 0.98 and top-100 == 1.0; program bar (doc/DECISIONS.md
2026-10-16) top-1 >= 0.90 and top-5 >= 0.98. Positive controls must reach top-1 == 1.0; negative controls must fail.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
sys.path.insert(0, str(HERE))
WS = Path(os.environ.get("GLM53_WS", "/opt/glm53/kit"))
RUNTIME = WS / "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

GOLDENS_FP32 = WS / "goldens_gpu_fp32"
GOLDEN_TF_PROMPTS = (128, 512, 2048, 4096, 32768)
PROGRAM_BAR = {"top1": 0.90, "top5": 0.98}
READINESS_BAR = {"top5": 0.98, "top100": 1.0}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def golden_refpt(
    T: int, out_path: Path, *, goldens_dir: Path = GOLDENS_FP32, gen_len: int | None = None, k: int = 100
) -> Path:
    from readiness_check.schema import Reference, ReferenceEntry, save_reference

    man = json.load(open(goldens_dir / "manifest.json"))
    tf_rel, lg_rel = f"teacher_forcing/{T}.pt", f"logits/{T}.pt"
    for rel in (tf_rel, lg_rel):
        if sha256(goldens_dir / rel) != man["files"][rel]["sha256"]:
            raise ValueError(f"sha256 mismatch: {goldens_dir / rel}")
    tf = torch.load(goldens_dir / tf_rel)
    cont = torch.load(goldens_dir / lg_rel)["continuation_logits"].float()
    targets = tf["targets"].reshape(-1).long()
    G = gen_len or targets.numel()
    ref = Reference(
        k=k,
        hf_model_id="zai-org/GLM-5.3-Flash",
        entries=[
            ReferenceEntry(
                prompt_text=f"goldens_gpu_fp32 prompt {T} (chat template, see goldens/prompts/{T}.json)",
                prompt_tokens=tf["prompt_ids"].reshape(1, -1).long(),
                generated_tokens=targets[:G].reshape(1, -1),
                topk_tokens=torch.topk(cont[:G], k, dim=-1).indices.to(torch.int32),
                tf_prompt_len=int(tf["prompt_ids"].numel()),
            )
        ],
        token_ids_meta={"bos_id": None, "eos_id": 154820, "pad_id": 154820},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_reference(ref, out_path)


def registered_aime24() -> Path:
    """The AIME24 chat reference registered in goldens_gpu_fp32/manifest.json (sha256 verified)."""
    man = json.load(open(GOLDENS_FP32 / "manifest.json"))
    reg = man["registered_references"]["aime24_chat_g100_k100"]
    p = WS / reg["path"]
    if sha256(p) != reg["sha256"]:
        raise ValueError(f"sha256 mismatch: {p}")
    return p


def mock_refpt(model, prompts: Sequence[Sequence[int]], gen_len: int, out_path: Path, k: int = 100) -> Path:
    """readiness_v1 reference from the MockModel's independent full-sequence forward (greedy continuation +
    top-k of the logits that predicted each continuation token)."""
    from readiness_check.schema import Reference, ReferenceEntry, save_reference

    entries = []
    for p in prompts:
        gen = model.reference_greedy(p, gen_len)
        logits = model.reference_logits(list(p) + gen)
        pred = logits[len(p) - 1 : len(p) - 1 + len(gen)]
        entries.append(
            ReferenceEntry(
                prompt_text=f"mock prompt ({len(p)} tokens)",
                prompt_tokens=torch.tensor([list(p)], dtype=torch.long),
                generated_tokens=torch.tensor([gen], dtype=torch.long),
                topk_tokens=torch.topk(pred, k, dim=-1).indices.to(torch.int32),
                tf_prompt_len=len(p),
            )
        )
    ref = Reference(
        k=k,
        hf_model_id="mock:MockModel",
        entries=entries,
        token_ids_meta={"bos_id": None, "eos_id": 154820, "pad_id": 154820},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return save_reference(ref, out_path)


def aggregate(per_entry: list[dict]) -> dict:
    total = sum(s["total"] for s in per_entry)
    agg = {k: sum(s[f"matches_{k}"] for s in per_entry) / total for k in ("top1", "top5", "top100")}
    agg.update({f"matches_{k}": sum(s[f"matches_{k}"] for s in per_entry) for k in ("top1", "top5", "top100")})
    agg["total"] = total
    return agg


def gate(agg: dict) -> dict:
    program = agg["top1"] >= PROGRAM_BAR["top1"] and agg["top5"] >= PROGRAM_BAR["top5"]
    readiness = agg["top5"] >= READINESS_BAR["top5"] and agg["top100"] >= READINESS_BAR["top100"]
    return {
        "program_bar": program,
        "readiness_bar": readiness,
        "pass": program and readiness,
        "bars": {"program": PROGRAM_BAR, "readiness": READINESS_BAR},
    }


def golden_replay_backend(
    prompts: Sequence[int | str],
    *,
    perturb_every: int = 0,
    max_batch: int = 4,
    include_aime24: bool = False,
    extra_dirs: Sequence[tuple[Path, Sequence]] = (),
):
    import mock_model as M

    entries = M.load_golden_entries(str(GOLDENS_FP32), prompts) if prompts else []
    if include_aime24:
        entries += M.load_golden_entries(str(WS / "goldens_gpu_fp32_aime24"), [161])
    for d, ps in extra_dirs:
        entries += M.load_golden_entries(str(d), ps)
    return M.GoldenReplayModel(entries, max_batch=max_batch, perturb_every=perturb_every)


def build_kwargs(backend, *, max_batch: int | None = None, max_seq_len: int | None = None, **kw) -> dict[str, Any]:
    out = {"backend": backend, "max_batch": max_batch or backend.max_batch}
    if max_seq_len is not None:
        out["max_seq_len"] = max_seq_len
    out.update(kw)
    return out


def safe_decode(tokenizer, ids) -> str:
    """Decode for evidence text: ids outside the tokenizer vocabulary (lm_head padding rows a broken model may emit)
    are shown as <|invalid:ID|> instead of raising."""
    out, run = [], []
    for t in [int(i) for i in ids]:
        if 0 <= t < tokenizer.vocab_size:
            run.append(t)
        else:
            if run:
                out.append(tokenizer.decode(run))
                run = []
            out.append(f"<|invalid:{t}|>")
    if run:
        out.append(tokenizer.decode(run))
    return "".join(out)


def first_divergence(a, b) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def truncate_at_stop(tokens, stop_ids, n) -> list[int]:
    out = []
    for t in list(tokens)[:n]:
        out.append(int(t))
        if int(t) in stop_ids:
            break
    return out


def degenerate_check(paths, json_out: Path) -> dict:
    """Run the plugin's check_degenerate_output.py (by file path) on explicit artifact paths."""
    import subprocess

    script = RUNTIME / "readiness_check" / "check_degenerate_output.py"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            *[str(p) for p in paths],
            "--scope",
            "autoregressive",
            "--missing-artifacts",
            "critical",
            "--json",
            str(json_out),
        ],
        capture_output=True,
        text=True,
    )
    return {"exit_code": r.returncode, "stdout_tail": r.stdout[-1500:], "report": str(json_out)}


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            obj, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
        )
    os.replace(tmp, path)
    return path
