"""LLM backends for the agent.

``LlamaCppClient`` talks to a local ``llama-server`` (llama.cpp) through its
OpenAI-compatible ``/v1/chat/completions`` endpoint, so the model stays on the
Pi. Start the server with tool calling enabled, for example:

    llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --jinja -c 2048 --port 8080

``ScriptedLLM`` replays canned replies for tests and offline demos.

NOTE: LlamaCppClient has not been run against a real server yet; verify it on
the Pi.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMReply: ...


class ScriptedLLM:
    """Returns the given replies in order; useful for tests and demos."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages, tools) -> LLMReply:
        self.calls.append([dict(m) for m in messages])
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


def parse_reply(body: dict[str, Any]) -> LLMReply:
    """Turn an OpenAI-style chat completion body into an ``LLMReply``."""
    message = body["choices"][0]["message"]
    calls = []
    for item in message.get("tool_calls") or []:
        fn = item["function"]
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except ValueError:
            args = {"_unparseable": raw}  # tool layer will reject it
        calls.append(ToolCall(item.get("id", fn["name"]), fn["name"], args))
    return LLMReply(message.get("content") or "", calls)


class LlamaCppClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_s: float = 120.0,
                 temperature: float = 0.2) -> None:
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._timeout = timeout_s
        self._temperature = temperature

    def complete(self, messages, tools) -> LLMReply:
        payload = {"messages": messages, "temperature": self._temperature}
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
            return parse_reply(json.load(response))
