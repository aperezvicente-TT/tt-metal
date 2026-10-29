# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dump transformers' tokenizer / chat-template outputs for every parity case (golden prompts + random conversations).

Run in the reference venv (transformers >= 5.16, which knows `glm5_next`), host only:

    .agents/scripts/no-tt-devices $GLM53_REF_VENV/bin/python \
        models/autoports/zai_org_glm_5_3_flash/tests/tokenizer_reference_hf.py --out <json>

`tests/test_tokenizer_chat.py` compares `tt/tokenizer_chat.py` against this file token for token (in the TT env, i.e.
across tokenizers/jinja2 versions) and, when transformers is importable in-process, against transformers directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tokenizer_cases as C  # noqa: E402


def hf_case_outputs(tok, messages, tools, kwargs):
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, **kwargs)
    ids = tok.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False, **kwargs)
    return {
        "text": text,
        "ids": [int(i) for i in ids],
        "decode": tok.decode(ids, skip_special_tokens=False),
        "decode_skip_special": tok.decode(ids, skip_special_tokens=True),
    }


def build(snapshot: str, ws: str, n_random: int, seed: int) -> dict:
    import tokenizers
    import torch
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot)
    out = {
        "meta": {
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "tokenizer_class": type(tok).__name__,
            "snapshot": snapshot,
            "n_random": n_random,
            "seed": seed,
            "special_tokens_map": tok.special_tokens_map,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "bos_token_id": tok.bos_token_id,
            "vocab_size_with_added": len(tok),
        },
        "random": [],
        "golden_prompts": [],
        "teacher_forcing": [],
    }
    for case in C.random_conversations(n_random, seed):
        row = {"id": case["id"]}
        row.update(hf_case_outputs(tok, case["messages"], case["tools"], case["kwargs"]))
        out["random"].append(row)
    files = C.golden_prompt_files(ws)
    for jpath in files["prompt_json"]:
        meta = C.load_json(jpath)
        text = meta.get("text") or meta.get("rendered_text")
        messages = meta.get("messages") or [{"role": "user", "content": C.user_content_of(text)}]
        kwargs = dict(meta.get("chat_template_kwargs") or {"add_generation_prompt": True})
        pt = jpath[: -len(".json")] + ".pt"
        stored = torch.load(pt).reshape(-1).tolist() if os.path.exists(pt) else None
        row = {
            "path": os.path.relpath(jpath, ws),
            "messages": messages,
            "kwargs": kwargs,
            "stored_text": text,
            "stored_ids": stored,
        }
        row.update(hf_case_outputs(tok, messages, None, kwargs))
        row["ids_from_stored_text"] = tok(text, add_special_tokens=False)["input_ids"]
        out["golden_prompts"].append(row)
    for tpath in files["teacher_forcing"]:
        d = torch.load(tpath)
        pid = d["prompt_ids"].reshape(-1).tolist()
        tgt = d["targets"].reshape(-1).tolist()
        out["teacher_forcing"].append(
            {
                "path": os.path.relpath(tpath, ws),
                "prompt_ids": pid,
                "targets": tgt,
                "decode_prompt": tok.decode(pid, skip_special_tokens=False),
                "decode_targets": tok.decode(tgt, skip_special_tokens=False),
                "decode_targets_skip_special": tok.decode(tgt, skip_special_tokens=True),
                "reencode_decoded_prompt": tok(tok.decode(pid), add_special_tokens=False)["input_ids"],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("GLM53_HF_SNAPSHOT"))
    ap.add_argument("--ws", default=C.default_workspace())
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = build(args.snapshot, args.ws, args.n_random, args.seed)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "random": len(data["random"]),
                "golden_prompts": len(data["golden_prompts"]),
                "teacher_forcing": len(data["teacher_forcing"]),
                "meta": data["meta"]["transformers"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
