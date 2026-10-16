"""Ask the agent something from the command line.

    python -m ml.agent.cli "How close is the nearest object and is it warm?"
    python -m ml.agent.cli "Turn the red LED on" --live

By default LED requests are a dry run. With ``--live`` the agent's ``set_led``
tool publishes ``command/led`` to the local broker and the actuator controller
(the only process that owns the GPIO pins) carries it out.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from edge.actuator_controller import COMMAND_TOPIC
from edge.config import load_config
from edge.storage import store_for
from ml.agent.agent import run_agent
from ml.agent.llm import LlamaCppClient
from ml.agent.tools import build_registry


class MqttLedActuator:
    """Actuator port that sends an LED command through the local broker."""

    def __init__(self, broker) -> None:
        self._broker = broker

    def set(self, led: str, on: bool) -> None:
        self._broker.publish(COMMAND_TOPIC, {"led": led, "on": on})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CE 4201 sensor agent")
    parser.add_argument("goal", help="what to ask the agent")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server URL")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="really send LED commands")
    args = parser.parse_args(argv)

    config = load_config()
    stores = {name: store_for(name, config.data_dir) for name in ("distance", "temperature", "vision")}
    broker: Any = None
    actuator = None
    if args.live:
        from edge.messaging import LocalBroker

        broker = LocalBroker(config.mqtt_host, config.mqtt_port, "pi5-agent")
        broker.connect()
        actuator = MqttLedActuator(broker)

    try:
        result = run_agent(LlamaCppClient(args.server), build_registry(stores, actuator),
                           args.goal, args.max_steps)
    except OSError as exc:  # server not running / unreachable
        print(f"Could not reach the model server at {args.server}: {exc}\n"
              "Start it with: llama-server -m <model>.gguf --jinja -c 2048 --port 8080")
        return 2
    finally:
        if broker is not None:
            broker.close()
    print(result.answer)
    print(json.dumps({"steps": result.steps, "finished": result.finished,
                      "tools": result.tool_log}, indent=2))
    return 0 if result.finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
