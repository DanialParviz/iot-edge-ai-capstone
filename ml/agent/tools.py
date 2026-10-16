"""The agent's approved tools.

Course rule: the agent never touches hardware directly, only through approved
tools. Every tool is registered here with a description and a JSON schema, and
the registry refuses anything not on the list. Read tools only read stored
data; the one action tool (``set_led``) goes through an injected actuator port,
validates its arguments, and defaults to a dry run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from ml.agent.stats import label_counts, numeric_summary
from ml.summary import collect_stats

SENSORS = ("distance", "temperature", "vision")
LEDS = ("red", "yellow", "green")
MAX_RECORDS = 50


class ToolError(Exception):
    """A tool was called with bad arguments or an unknown name."""


class ActuatorPort(Protocol):
    def set(self, led: str, on: bool) -> None: ...


class DryRunActuator:
    """Default actuator: records requests but touches no hardware."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, bool]] = []

    def set(self, led: str, on: bool) -> None:
        self.requests.append((led, on))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    func: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function spec (also used by llama.cpp's server)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run an approved tool and return its result as a JSON string."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; approved tools: {self.names()}")
        try:
            result = tool.func(**arguments)
        except ToolError:
            raise
        except TypeError as exc:  # wrong / missing argument names
            raise ToolError(f"bad arguments for {name}: {exc}") from exc
        return json.dumps(result, sort_keys=True)


def _check_sensor(sensor: str) -> str:
    if sensor not in SENSORS:
        raise ToolError(f"sensor must be one of {list(SENSORS)}, got {sensor!r}")
    return sensor


def _check_n(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_RECORDS:
        raise ToolError(f"n must be an integer from 1 to {MAX_RECORDS}, got {n!r}")
    return n


def build_registry(stores: dict[str, Any], actuator: Optional[ActuatorPort] = None) -> ToolRegistry:
    """Create the registry. ``stores`` maps sensor name -> object with ``latest(n)``."""
    actuator = actuator or DryRunActuator()
    dry_run = isinstance(actuator, DryRunActuator)

    def get_latest_readings(sensor: str, n: int = 5):
        records = stores[_check_sensor(sensor)].latest(_check_n(n))
        return {"sensor": sensor, "readings": records}

    def get_sensor_stats(sensor: str, n: int = 10):
        records = stores[_check_sensor(sensor)].latest(_check_n(n))
        if sensor == "distance":
            return {"sensor": sensor, "distance_cm": numeric_summary(records, "distance_cm")}
        if sensor == "temperature":
            return {"sensor": sensor, "temperature_c": numeric_summary(records, "temperature_c")}
        return {
            "sensor": sensor,
            "object_counts": label_counts(records, "object"),
            "face_counts": label_counts(records, "identity"),
        }

    def get_system_summary(n: int = 10):
        return collect_stats(stores, _check_n(n))

    def set_led(led: str, on: bool):
        if led not in LEDS:
            raise ToolError(f"led must be one of {list(LEDS)}, got {led!r}")
        if not isinstance(on, bool):
            raise ToolError(f"on must be true or false, got {on!r}")
        actuator.set(led, on)
        return {"led": led, "on": on, "dry_run": dry_run}

    n_schema = {"type": "integer", "minimum": 1, "maximum": MAX_RECORDS}
    sensor_schema = {"type": "string", "enum": list(SENSORS)}
    return ToolRegistry(
        [
            Tool(
                "get_latest_readings",
                "Return the newest raw readings stored for one sensor.",
                {
                    "type": "object",
                    "properties": {"sensor": sensor_schema, "n": n_schema},
                    "required": ["sensor"],
                },
                get_latest_readings,
            ),
            Tool(
                "get_sensor_stats",
                "Return min, max, mean and latest (or object counts for vision) "
                "computed in Python over the newest n readings of one sensor.",
                {
                    "type": "object",
                    "properties": {"sensor": sensor_schema, "n": n_schema},
                    "required": ["sensor"],
                },
                get_sensor_stats,
            ),
            Tool(
                "get_system_summary",
                "Return statistics for distance, temperature and vision together.",
                {"type": "object", "properties": {"n": n_schema}},
                get_system_summary,
            ),
            Tool(
                "set_led",
                "Turn one status LED on or off. The only tool that changes hardware.",
                {
                    "type": "object",
                    "properties": {
                        "led": {"type": "string", "enum": list(LEDS)},
                        "on": {"type": "boolean"},
                    },
                    "required": ["led", "on"],
                },
                set_led,
            ),
        ]
    )
