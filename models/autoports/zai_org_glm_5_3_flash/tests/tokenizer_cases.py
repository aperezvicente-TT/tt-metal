# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer / chat-template parity cases for GLM-5.3-Flash (stdlib only, importable from any venv).

* `golden_prompt_cases(ws)`: every golden prompt under `$GLM53_WS/goldens*/` - `prompts/<T>.json` (rendered text +
  token ids in `<T>.pt`) and the `prompt_ids` of every `teacher_forcing/<T>.pt` (plus their greedy `targets`, used for
  decode parity).
* `random_conversations(n=50, seed=...)`: seeded random chat conversations covering multi-turn dialogues, system
  prompts, tool definitions, assistant tool calls and tool responses, reasoning content (field and inline `</think>`),
  content-part lists, multilingual / code / whitespace edge cases, and the template modes (`reasoning_effort`,
  `clear_thinking`, `add_generation_prompt`, `continue_final_message`).
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. Zürich"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search Tenstorrent documentation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1}},
                "required": ["query"],
            },
        },
    },
    {
        "name": "run_python",
        "description": "Execute a short Python snippet and return stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
]

SNIPPETS = [
    "Hello! Can you help me?",
    "Explain the difference between DRAM and SRAM in two sentences.",
    "What is 17 * 23?",
    "Write a haiku about Blackhole chips.",
    "  leading and trailing spaces  ",
    "Line one\nLine two\n\nLine four after a blank line",
    "Tabs\tand\tmore\ttabs",
    "Résumé, naïve café, Zürich, São Paulo.",
    "中文测试：请用一句话介绍你自己。",
    "日本語のテキストも大丈夫ですか？",
    "مرحبا بالعالم",
    "Emoji check 🚀🔥✨ and a flag 🇨🇭",
    "```python\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n```",
    "LaTeX: $\\int_0^1 x^2\\,dx = \\frac{1}{3}$ and \\boxed{42}",
    'JSON: {"a": [1, 2, 3], "b": {"c": null}}',
    "Numbers 1234567890 and 3.14159e-10",
    "A literal <think> tag and </think> inside user text.",
    "Markup <b>bold</b> &amp; entities &lt;tag&gt;",
    "Trailing newline\n",
    "\n\nLeading newlines",
    "Mixed\r\nline\r\nendings",
    "Multiple    internal     spaces",
    "Quote: 'single' and \"double\" and `back`",
    "",
]

REASONING = [
    "The user wants a short answer. Let me compute it.",
    "Step 1: parse the question.\nStep 2: answer concisely.",
    "",
    "思考：先理解问题。",
]

TOOL_ARG_VALUES = ["Zürich", "celsius", 3, 2.5, True, None, ["a", "b"], {"nested": {"k": [1, 2]}}, "print(1+1)"]


def _snippet(rng: random.Random) -> str:
    k = rng.randint(1, 3)
    return " ".join(rng.choice(SNIPPETS) for _ in range(k)) if rng.random() < 0.3 else rng.choice(SNIPPETS)


def _content(rng: random.Random) -> Any:
    if rng.random() < 0.12:  # content-part list (text parts only; media parts would render placeholders)
        return [{"type": "text", "text": _snippet(rng)} for _ in range(rng.randint(1, 3))]
    return _snippet(rng)


def _assistant(rng: random.Random, call_idx: list[int]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant"}
    r = rng.random()
    if r < 0.35:
        msg["reasoning_content"] = rng.choice(REASONING)
        msg["content"] = _snippet(rng)
    elif r < 0.55:
        msg["content"] = f"<think>{rng.choice(REASONING)}</think>{_snippet(rng)}"
    else:
        msg["content"] = _snippet(rng)
    return msg


def _tool_round(rng: random.Random, call_idx: list[int]) -> list[dict[str, Any]]:
    n_calls = rng.randint(1, 2)
    calls = []
    for _ in range(n_calls):
        call_idx[0] += 1
        tool = rng.choice(TOOLS)
        name = tool.get("function", tool)["name"]
        args = {f"arg{j}": rng.choice(TOOL_ARG_VALUES) for j in range(rng.randint(1, 3))}
        calls.append({"id": f"call_{call_idx[0]}", "type": "function", "function": {"name": name, "arguments": args}})
    asst = {
        "role": "assistant",
        "content": rng.choice(["", "Let me check.", "Calling the tool now."]),
        "tool_calls": calls,
    }
    if rng.random() < 0.4:
        asst["reasoning_content"] = rng.choice(REASONING)
    msgs = [asst]
    order = list(calls)
    if rng.random() < 0.4:
        rng.shuffle(order)  # out-of-order tool results exercise the template's sort-by-tool_call_id branch
    for c in order:
        tool_msg = {"role": "tool", "content": rng.choice(['{"temp_c": 21}', "No results.", "stdout: 2\n", "ok"])}
        if rng.random() < 0.85:
            tool_msg["tool_call_id"] = c["id"]
        msgs.append(tool_msg)
    return msgs


def random_conversations(n: int = 50, seed: int = 20261017) -> list[dict[str, Any]]:
    """Seeded random conversations. Every 5th case is forced to contain a tool call + tool response, every 3rd a
    system prompt, and every case after the first 5 has >= 2 user turns, so coverage does not depend on the seed."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        call_idx = [100 * i]
        msgs: list[dict[str, Any]] = []
        force_tools = i % 5 == 0
        if i % 3 == 0 or rng.random() < 0.3:
            msgs.append({"role": "system", "content": _content(rng)})
        n_turns = rng.randint(2, 4) if i >= 5 else rng.randint(1, 2)
        use_tools = force_tools or rng.random() < 0.35
        for t in range(n_turns):
            msgs.append({"role": "user", "content": _content(rng)})
            if t < n_turns - 1:
                if use_tools and (t == 0 or rng.random() < 0.3):
                    msgs.extend(_tool_round(rng, call_idx))
                    if rng.random() < 0.7:
                        msgs.append(_assistant(rng, call_idx))
                else:
                    msgs.append(_assistant(rng, call_idx))
        if force_tools and not any(m.get("tool_calls") for m in msgs):
            msgs.extend(_tool_round(rng, call_idx))
        kwargs: dict[str, Any] = {}
        end = rng.random()
        if end < 0.75:
            kwargs["add_generation_prompt"] = True
        elif end < 0.9:
            msgs.append(_assistant(rng, call_idx))
            kwargs["add_generation_prompt"] = False
        else:
            msgs.append({"role": "assistant", "content": "Partial answer that continues"})
            kwargs["add_generation_prompt"] = False
            kwargs["continue_final_message"] = True
        eff = rng.choice([None, None, "low", "high", "max", "medium"])
        if eff is not None:
            kwargs["reasoning_effort"] = eff
        ct = rng.choice([None, None, True, False])
        if ct is not None:
            kwargs["clear_thinking"] = ct
        tools = TOOLS if (use_tools or rng.random() < 0.15) else None
        cases.append({"id": f"random_{i:02d}", "messages": msgs, "tools": tools, "kwargs": kwargs})
    return cases


def coverage(cases: list[dict[str, Any]]) -> dict[str, int]:
    def has(c, pred):
        return any(pred(m) for m in c["messages"])

    return {
        "cases": len(cases),
        "multi_turn": sum(1 for c in cases if sum(m["role"] == "user" for m in c["messages"]) >= 2),
        "system_prompt": sum(1 for c in cases if has(c, lambda m: m["role"] == "system")),
        "tool_definitions": sum(1 for c in cases if c["tools"]),
        "assistant_tool_call": sum(1 for c in cases if has(c, lambda m: bool(m.get("tool_calls")))),
        "tool_response": sum(1 for c in cases if has(c, lambda m: m["role"] == "tool")),
        "reasoning_content": sum(1 for c in cases if has(c, lambda m: "reasoning_content" in m)),
        "inline_think": sum(
            1
            for c in cases
            if has(
                c,
                lambda m: isinstance(m.get("content"), str) and "</think>" in m["content"] and m["role"] == "assistant",
            )
        ),
        "content_parts": sum(1 for c in cases if has(c, lambda m: isinstance(m.get("content"), list))),
        "generation_prompt": sum(1 for c in cases if c["kwargs"].get("add_generation_prompt")),
        "continue_final_message": sum(1 for c in cases if c["kwargs"].get("continue_final_message")),
        "reasoning_effort_set": sum(1 for c in cases if "reasoning_effort" in c["kwargs"]),
        "clear_thinking_set": sum(1 for c in cases if "clear_thinking" in c["kwargs"]),
    }


def default_workspace() -> str:
    return os.environ.get("GLM53_WS", "/opt/glm53/kit")


def golden_prompt_files(ws: str | None = None) -> dict[str, list[str]]:
    """Paths of every golden prompt artifact under <ws>/goldens*/ (prompt json/pt pairs and teacher-forcing files)."""
    ws = ws or default_workspace()
    return {
        "prompt_json": sorted(glob.glob(os.path.join(ws, "goldens*", "prompts", "*.json"))),
        "teacher_forcing": sorted(glob.glob(os.path.join(ws, "goldens*", "teacher_forcing", "*.pt"))),
    }


def user_content_of(rendered: str) -> str:
    """User content of a single-user-turn rendered golden prompt (`...<|user|>{content}<|assistant|><think>`)."""
    return rendered.split("<|user|>", 1)[1].rsplit("<|assistant|>", 1)[0]


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
