import json

import pytest

from ml.agent.agent import run_agent
from ml.agent.llm import LLMReply, ScriptedLLM, ToolCall, parse_reply
from ml.agent.stats import label_counts, numeric_summary
from ml.agent.tools import DryRunActuator, ToolError, build_registry


class ListStore:
    def __init__(self, records):
        self._records = records

    def latest(self, n=10):
        return self._records[-n:]


@pytest.fixture
def stores():
    return {
        "distance": ListStore([{"distance_cm": v} for v in (6.6, 7.7, 7.0)]),
        "temperature": ListStore([{"temperature_c": v} for v in (28.1, 30.9)]),
        "vision": ListStore(
            [{"object": o} for o in ("person", "bird", "person", "bear", "person")]
        ),
    }


# --- stats: Python does the math --------------------------------------
def test_numeric_summary():
    s = numeric_summary([{"v": 1}, {"v": 2}, {"v": 6}], "v")
    assert s == {"count": 3, "min": 1, "max": 6, "mean": 3.0, "latest": 6}


def test_numeric_summary_ignores_bad_values_and_handles_empty():
    assert numeric_summary([{"v": "x"}, {}], "v") is None
    assert numeric_summary([{"v": 2}, {"v": None}], "v")["count"] == 1


def test_label_counts_are_exact_and_sorted():
    recs = [{"object": o} for o in ["a", "b", "a", "a", "b", "c"]]
    assert label_counts(recs) == {"a": 3, "b": 2, "c": 1}
    assert list(label_counts(recs)) == ["a", "b", "c"]


# --- tools -------------------------------------------------------------
def test_stats_tool_matches_python_math(stores):
    out = json.loads(build_registry(stores).call("get_sensor_stats", {"sensor": "distance"}))
    assert out["distance_cm"] == {"count": 3, "min": 6.6, "max": 7.7, "mean": 7.1, "latest": 7.0}


def test_vision_stats_counts_objects(stores):
    out = json.loads(build_registry(stores).call("get_sensor_stats", {"sensor": "vision"}))
    assert out["object_counts"] == {"person": 3, "bird": 1, "bear": 1}


def test_system_summary_covers_all_sensors(stores):
    out = json.loads(build_registry(stores).call("get_system_summary", {}))
    assert set(out) == {"distance_cm", "temperature_c", "object_counts"}
    assert out["temperature_c"]["max"] == 30.9


def test_unknown_tool_is_refused(stores):
    with pytest.raises(ToolError):
        build_registry(stores).call("delete_database", {})


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_latest_readings", {"sensor": "lidar"}),
        ("get_latest_readings", {"sensor": "distance", "n": 0}),
        ("get_latest_readings", {"sensor": "distance", "n": 999}),
        ("get_latest_readings", {"sensor": "distance", "n": True}),
        ("get_latest_readings", {"wrong": 1}),
        ("set_led", {"led": "blue", "on": True}),
        ("set_led", {"led": "red", "on": "yes"}),
    ],
)
def test_bad_arguments_are_rejected(stores, name, args):
    with pytest.raises(ToolError):
        build_registry(stores).call(name, args)


def test_set_led_is_dry_run_by_default(stores):
    actuator = DryRunActuator()
    out = json.loads(build_registry(stores, actuator).call("set_led", {"led": "red", "on": True}))
    assert out["dry_run"] is True and actuator.requests == [("red", True)]


def test_set_led_reports_live_when_real_actuator_injected(stores):
    class Live:
        def __init__(self):
            self.calls = []

        def set(self, led, on):
            self.calls.append((led, on))

    live = Live()
    out = json.loads(build_registry(stores, live).call("set_led", {"led": "green", "on": False}))
    assert out["dry_run"] is False and live.calls == [("green", False)]


def test_specs_are_openai_style(stores):
    for spec in build_registry(stores).specs():
        assert spec["type"] == "function" and spec["function"]["parameters"]["type"] == "object"


# --- agent loop --------------------------------------------------------
def test_agent_uses_tool_then_answers(stores):
    llm = ScriptedLLM(
        [
            LLMReply(tool_calls=[ToolCall("c1", "get_system_summary", {})]),
            LLMReply("Three people were seen."),
        ]
    )
    result = run_agent(llm, build_registry(stores), "What did the camera see?")
    assert result.answer == "Three people were seen." and result.finished and result.steps == 2
    assert result.tool_log == [{"tool": "get_system_summary", "arguments": {}, "ok": True}]
    # the tool result (exact Python-computed counts) was fed back to the model
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and '"person": 3' in tool_msg["content"]


def test_agent_survives_bad_tool_call_and_recovers(stores):
    llm = ScriptedLLM(
        [
            LLMReply(tool_calls=[ToolCall("c1", "format_disk", {})]),
            LLMReply("Sorry, I can't do that."),
        ]
    )
    result = run_agent(llm, build_registry(stores), "wipe it")
    assert result.tool_log[0]["ok"] is False
    assert "unknown tool" in llm.calls[1][-1]["content"]
    assert result.answer == "Sorry, I can't do that."


def test_agent_stops_at_step_limit(stores):
    call = LLMReply(tool_calls=[ToolCall("c", "get_system_summary", {})])
    result = run_agent(ScriptedLLM([call] * 3), build_registry(stores), "loop", max_steps=3)
    assert result.finished is False and result.steps == 3


def test_parse_reply_handles_tool_calls_and_bad_json():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "a", "function": {"name": "set_led", "arguments": '{"led": "red", "on": true}'}},
                        {"id": "b", "function": {"name": "set_led", "arguments": "{oops"}},
                    ],
                }
            }
        ]
    }
    reply = parse_reply(body)
    assert reply.tool_calls[0].arguments == {"led": "red", "on": True}
    assert "_unparseable" in reply.tool_calls[1].arguments
