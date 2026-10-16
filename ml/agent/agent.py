"""The agent loop: goal -> pick tool -> use it -> check result -> repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ml.agent.llm import LLMClient
from ml.agent.tools import ToolError, ToolRegistry

SYSTEM_PROMPT = (
    "You are the monitoring agent for a Raspberry Pi sensor node with a distance "
    "sensor, a temperature sensor and a camera. Use the provided tools to read "
    "data; never guess numbers. All counts and statistics come from tool results. "
    "Only change an LED if the user's goal asks for it. Answer in short sentences."
)


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = True  # False if the step limit was hit


def run_agent(llm: LLMClient, registry: ToolRegistry, goal: str, max_steps: int = 5) -> AgentResult:
    """Run the loop for at most ``max_steps`` model calls."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        reply = llm.complete(messages, registry.specs())
        if not reply.tool_calls:
            return AgentResult(reply.content, step, tool_log)

        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in reply.tool_calls
                ],
            }
        )
        for call in reply.tool_calls:
            try:
                result = registry.call(call.name, call.arguments)
                ok = True
            except ToolError as exc:  # report the problem back so the model can retry
                result, ok = json.dumps({"error": str(exc)}), False
            tool_log.append({"tool": call.name, "arguments": call.arguments, "ok": ok})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return AgentResult("Stopped: step limit reached before a final answer.", max_steps,
                       tool_log, finished=False)
