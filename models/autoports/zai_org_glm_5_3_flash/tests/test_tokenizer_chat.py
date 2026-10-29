# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only parity tests of `tt/tokenizer_chat.py` against transformers (no device; run under no-tt-devices).

Reference: the JSON written by `tests/tokenizer_reference_hf.py` with transformers 5.19 ($GLM53_TOKENIZER_HF_REF,
default `$GLM53_WS/doc/ttnn/06a-full-model-prep/logs/tokenizer/hf_reference.json`). When transformers >= 5.16 is
importable in this interpreter (the reference venv), every case is additionally re-rendered by transformers
in-process, so the comparison holds in both environments (TT env: tokenizers 0.22 / transformers 5.12; reference
venv: tokenizers 0.23 / transformers 5.19).

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_tokenizer_chat.py
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tokenizer_cases as C  # noqa: E402
from host_evidence import expect_error, record  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import (  # noqa: E402
    EXPECTED_STOP_TOKENS,
    MODEL_VOCAB_SIZE,
    TEMPLATE_MODES,
    load_tokenizer,
)

WS = C.default_workspace()
REF_PATH = os.environ.get(
    "GLM53_TOKENIZER_HF_REF", os.path.join(WS, "doc/ttnn/06a-full-model-prep/logs/tokenizer/hf_reference.json")
)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


@pytest.fixture(scope="module")
def ref():
    if not os.path.exists(REF_PATH):
        pytest.fail(f"transformers reference {REF_PATH} missing: run tests/tokenizer_reference_hf.py in the ref venv")
    return C.load_json(REF_PATH)


@pytest.fixture(scope="module")
def hf_tok():
    """transformers tokenizer when this interpreter has a glm5_next-capable transformers, else None."""
    try:
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.16"):
            return None
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(load_tokenizer().snapshot))
    except Exception:  # noqa: BLE001
        return None


def _env():
    import jinja2
    import tokenizers

    try:
        import transformers

        tv = transformers.__version__
    except ImportError:
        tv = None
    return {"tokenizers": tokenizers.__version__, "jinja2": jinja2.__version__, "transformers_in_env": tv}


def test_random_conversation_parity(tok, ref, hf_tok):
    cases = C.random_conversations(ref["meta"]["n_random"], ref["meta"]["seed"])
    assert len(cases) == len(ref["random"]) == 50
    cov = C.coverage(cases)
    for key in ("multi_turn", "system_prompt", "assistant_tool_call", "tool_response", "reasoning_content"):
        assert cov[key] >= 5, (key, cov)
    n_tokens = 0
    in_process = 0
    for case, want in zip(cases, ref["random"]):
        assert case["id"] == want["id"]
        text = tok.render_chat(case["messages"], tools=case["tools"], **case["kwargs"])
        ids = tok.encode_chat(case["messages"], tools=case["tools"], **case["kwargs"])
        assert text == want["text"], case["id"]
        assert ids == want["ids"], case["id"]
        assert tok.decode(ids) == want["decode"], case["id"]
        assert tok.decode(ids, skip_special_tokens=True) == want["decode_skip_special"], case["id"]
        if hf_tok is not None:
            hf_ids = hf_tok.apply_chat_template(
                case["messages"], tools=case["tools"], tokenize=True, return_dict=False, **case["kwargs"]
            )
            assert ids == list(hf_ids), case["id"]
            in_process += 1
        n_tokens += len(ids)
    record(
        "tokenizer_parity_random_conversations",
        cases=len(cases),
        tokens_compared=n_tokens,
        coverage=cov,
        in_process_transformers_cases=in_process,
        reference=REF_PATH,
        reference_transformers=ref["meta"]["transformers"],
        reference_tokenizers=ref["meta"]["tokenizers"],
        env=_env(),
    )


def test_golden_prompt_parity(tok, ref, hf_tok):
    files = C.golden_prompt_files(WS)
    assert len(ref["golden_prompts"]) == len(files["prompt_json"]) >= 5
    rows = []
    for want in ref["golden_prompts"]:
        text = tok.render_chat(want["messages"], **want["kwargs"])
        ids = tok.encode_chat(want["messages"], **want["kwargs"])
        assert text == want["text"] == want["stored_text"], want["path"]
        assert ids == want["ids"] == want["stored_ids"] == want["ids_from_stored_text"], want["path"]
        assert tok.encode(want["stored_text"]) == want["stored_ids"], want["path"]
        assert tok.encode(want["stored_text"], add_special_tokens=True) == want["stored_ids"], want["path"]
        if hf_tok is not None:
            assert hf_tok(want["stored_text"], add_special_tokens=True)["input_ids"] == want["stored_ids"]
        assert tok.decode(ids) == want["decode"] == want["stored_text"], want["path"]
        if hf_tok is not None:
            assert ids == list(
                hf_tok.apply_chat_template(want["messages"], tokenize=True, return_dict=False, **want["kwargs"])
            )
        rows.append({"path": want["path"], "tokens": len(ids)})
    record("tokenizer_parity_golden_prompts", prompts=rows, env=_env(), in_process_transformers=hf_tok is not None)


def test_teacher_forcing_prompt_ids(tok, ref):
    """Every goldens*/teacher_forcing/<T>.pt: prompt ids decode identically and equal the rendered golden prompt;
    the greedy targets decode identically (HF vs ours, both skip modes)."""
    files = C.golden_prompt_files(WS)
    assert len(ref["teacher_forcing"]) == len(files["teacher_forcing"]) >= 5
    prompt_by_len = {len(g["stored_ids"]): g["stored_ids"] for g in ref["golden_prompts"]}
    rows = []
    for want in ref["teacher_forcing"]:
        pid, tgt = want["prompt_ids"], want["targets"]
        assert tok.decode(pid) == want["decode_prompt"], want["path"]
        assert tok.encode(want["decode_prompt"]) == want["reencode_decoded_prompt"] == pid, want["path"]
        assert tok.decode(tgt) == want["decode_targets"], want["path"]
        assert tok.decode(tgt, skip_special_tokens=True) == want["decode_targets_skip_special"], want["path"]
        if len(pid) in prompt_by_len:
            assert pid == prompt_by_len[len(pid)], want["path"]
        rows.append(
            {
                "path": want["path"],
                "prompt_tokens": len(pid),
                "targets": len(tgt),
                "stop_in_targets": [i for i, t in enumerate(tgt) if tok.is_stop_token(t)][:3],
            }
        )
    record("tokenizer_parity_teacher_forcing_files", files=rows)


def test_stop_tokens_and_template_modes(tok, ref):
    assert dict(zip(tok.stop_token_ids, tok.convert_ids_to_tokens(list(tok.stop_token_ids)))) == EXPECTED_STOP_TOKENS
    assert tok.eos_token_id == ref["meta"]["eos_token_id"] == 154820
    assert tok.pad_token_id == ref["meta"]["pad_token_id"] == 154820
    assert ref["meta"]["bos_token_id"] is None and tok.bos_token_id is None
    assert tok.vocab_size == ref["meta"]["vocab_size_with_added"] == 154856 and MODEL_VOCAB_SIZE == 154880
    msgs = [{"role": "user", "content": "hi"}]
    headers = {}
    for eff in (None, "low", "high", "max", "medium"):
        text = tok.render_chat(msgs, reasoning_effort=eff)
        headers[str(eff)] = text.split("<|user|>")[0]
        assert text.endswith("<|assistant|><think>")
    assert headers == {
        "None": "[gMASK]<sop><|system|>Reasoning Effort: Max",
        "low": "[gMASK]<sop><|system|>Reasoning Effort: Low",
        "high": "[gMASK]<sop><|system|>Reasoning Effort: High",
        "max": "[gMASK]<sop><|system|>Reasoning Effort: Max",
        "medium": "[gMASK]<sop><|system|>Reasoning Effort: Max",
    }
    hist = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
    ]
    kept = tok.render_chat(hist, clear_thinking=False)
    cleared = tok.render_chat(hist, clear_thinking=True)
    assert "<think>r1</think>a1" in kept and "<think></think>a1" in cleared and "r1" not in cleared
    assert tok.render_chat(msgs, add_generation_prompt=False).endswith("<|user|>hi")
    record(
        "tokenizer_stop_tokens_and_modes",
        stop_token_ids=list(tok.stop_token_ids),
        stop_tokens={str(k): v for k, v in EXPECTED_STOP_TOKENS.items()},
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        generation_defaults=tok.generation_defaults,
        reasoning_effort_headers=headers,
        template_modes=TEMPLATE_MODES,
    )


def test_parse_assistant_roundtrip(tok):
    """Template-rendered assistant turns (reasoning + content + tool calls) parse back to the same fields."""
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Zürich", "days": 3, "opts": {"u": "c"}}},
    }
    msgs = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "Checking.", "reasoning_content": "need the tool", "tool_calls": [call]},
    ]
    text = tok.render_chat(msgs, add_generation_prompt=False)
    gen = text.split("<|assistant|>", 1)[1]
    parsed = tok.parse_assistant(gen)
    assert parsed.reasoning_content == "need the tool"
    assert parsed.content == "Checking."
    assert parsed.tool_calls == [
        {"name": "get_weather", "arguments": {"location": "Zürich", "days": 3, "opts": {"u": "c"}}}
    ]
    ids = tok.encode(gen[len("<think>") :]) + [154829]  # generated ids after the <think> prompt, ending in a stop token
    assert tok.parse_assistant(ids).tool_calls == parsed.tool_calls
    record("tokenizer_parse_assistant_roundtrip", tool_calls=parsed.tool_calls)


def test_decode_rejects_lm_head_padding_ids(tok):
    with expect_error(ValueError):
        tok.decode([154856])
    with expect_error(ValueError):
        tok.decode([MODEL_VOCAB_SIZE - 1])
    record("tokenizer_rejects_padding_ids", tokenizer_vocab=tok.vocab_size, model_vocab=MODEL_VOCAB_SIZE)
