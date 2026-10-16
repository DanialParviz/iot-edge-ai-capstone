"""The single owner of the LEDs.

Only one process may drive the GPIO pins, so sensor programs just publish and
this controller subscribes, applies the rules, switches LEDs, and raises alerts.
Run standalone with ``python -m edge.actuator_controller`` (subscribes to the
local Mosquitto broker); ``EdgeNode`` also uses it in-process for simulation.
"""

from __future__ import annotations

import argparse
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from edge.alerts import notifier_from_env
from edge.config import Config, load_config
from edge.rules import LEDS, Alert, AlertTracker, evaluate

COMMAND_TOPIC = "command/led"

log = logging.getLogger(__name__)


class ActuatorController:
    def __init__(self, config: Config, leds, notifier=None, publisher=None) -> None:
        self.config = config
        self.leds = leds
        self.notifier = notifier  # object with notify(Alert)
        self.publisher = publisher  # object with publish(topic, payload)
        self.tracker = AlertTracker()

    def handle(self, topic: str, payload: dict[str, Any]) -> list[Alert]:
        """Apply the rule for one message; returns any new alerts."""
        decision = evaluate(topic, payload, self.config.thresholds, self.config.topics)
        if decision is None:
            return []
        self.leds.set(decision.led, decision.on)
        if not self.tracker.update(decision):
            return []
        alert = Alert(decision.led, decision.reason)
        log.warning("ALERT %s LED: %s", alert.led, alert.reason)
        self._announce(alert)
        return [alert]

    def handle_command(self, _topic: str, payload: dict[str, Any]) -> bool:
        """Run an LED command from an approved sender (e.g. the AI agent).

        The agent never touches GPIO; it publishes ``{"led": ..., "on": ...}``
        to ``command/led`` and this controller, the only GPIO owner, validates
        and carries it out. The next sensor message re-applies the normal
        rules. Returns True if the command was valid and applied.
        """
        led, on = payload.get("led"), payload.get("on")
        if led not in LEDS or not isinstance(on, bool):
            log.warning("rejected LED command: %r", payload)
            return False
        self.leds.set(led, on)
        log.info("LED command: %s -> %s", led, "ON" if on else "OFF")
        return True

    def _announce(self, alert: Alert) -> None:
        # A failure to announce must never stop the LEDs or the sensors.
        if self.notifier is not None:
            try:
                self.notifier.notify(alert)
            except Exception:
                log.exception("notifier failed")
        if self.publisher is not None:
            try:
                self.publisher.publish(
                    self.config.topics.alerts,
                    {
                        "led": alert.led,
                        "reason": alert.reason,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception:
                log.exception("alert publish failed")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LED + alert controller")
    parser.add_argument("--simulate", action="store_true", help="fake LEDs")
    parser.add_argument("--aws", action="store_true", help="also publish alerts to AWS IoT")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from edge.actuators import make_leds
    from edge.messaging import LocalBroker

    config = load_config()
    leds = make_leds(config.leds, args.simulate)
    publisher = None
    if args.aws:
        from edge.cloud import build_cloud_publisher

        publisher = build_cloud_publisher(config, "pi5-actuator")

    controller = ActuatorController(config, leds, notifier_from_env(), publisher)
    broker = LocalBroker(config.mqtt_host, config.mqtt_port, "pi5-actuator-local")
    broker.connect()
    broker.subscribe("sensor/#", controller.handle)
    broker.subscribe(COMMAND_TOPIC, controller.handle_command)
    log.info("actuator controller running; Ctrl+C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        broker.close()
        leds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
