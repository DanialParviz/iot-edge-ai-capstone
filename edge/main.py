"""Edge node entry point.

    python -m edge.main --simulate              # laptop, no hardware, no broker
    python -m edge.main --simulate --cycles 5   # run 5 read cycles then stop
    python -m edge.main                         # on the Pi: real sensors + GPIO
    python -m edge.main --mqtt                  # also publish to Mosquitto

Each cycle: read every sensor, store the reading, publish it (optional), then
apply the LED rules. Alerts fire only on an OFF -> ON change.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from edge.actuators import make_leds
from edge.config import Config, load_config
from edge.rules import AlertTracker, evaluate
from edge.storage import SensorStore, store_for

log = logging.getLogger("edge")


@dataclass
class Alert:
    led: str
    reason: str


class EdgeNode:
    """Wires sensors, storage, rules, LEDs and (optionally) MQTT together."""

    def __init__(self, config: Config, distance, temperature, leds, broker=None,
                 stores: Optional[dict[str, SensorStore]] = None) -> None:
        self.config = config
        self.sensors = {"distance": distance, "temperature": temperature}
        self.leds = leds
        self.broker = broker
        self.stores = stores or {
            name: store_for(name, config.data_dir) for name in self.sensors
        }
        self.tracker = AlertTracker()
        self._topic = {
            "distance": config.topics.distance,
            "temperature": config.topics.temperature,
        }

    def step(self) -> list[Alert]:
        """Run one read cycle. Returns the new alerts raised this cycle."""
        alerts: list[Alert] = []
        for name, sensor in self.sensors.items():
            record = sensor.read()
            if record is None:  # invalid / out-of-range reading
                continue
            self.stores[name].insert(record)
            topic = self._topic[name]
            if self.broker is not None:
                self.broker.publish(topic, record)
            alerts.extend(self.apply(topic, record))
        return alerts

    def apply(self, topic: str, payload: dict[str, Any]) -> list[Alert]:
        """Apply the LED rule for one message (also used for vision messages)."""
        decision = evaluate(topic, payload, self.config.thresholds, self.config.topics)
        if decision is None:
            return []
        self.leds.set(decision.led, decision.on)
        if self.tracker.update(decision):
            alert = Alert(decision.led, decision.reason)
            log.warning("ALERT %s LED: %s", alert.led, alert.reason)
            if self.broker is not None:
                self.broker.publish(
                    self.config.topics.alerts, {"led": alert.led, "reason": alert.reason}
                )
            return [alert]
        return []

    def close(self) -> None:
        self.leds.close()
        for store in self.stores.values():
            store.close()
        if self.broker is not None:
            self.broker.close()


def build_node(config: Config, simulate: bool, use_mqtt: bool) -> EdgeNode:
    if simulate:
        from edge.sensors.distance import SimulatedDistanceSensor
        from edge.sensors.temperature import SimulatedTemperatureSensor

        distance, temperature = SimulatedDistanceSensor(), SimulatedTemperatureSensor()
    else:
        from edge.sensors.distance import Vl53l0xSensor
        from edge.sensors.temperature import Bme680Sensor

        distance, temperature = Vl53l0xSensor(), Bme680Sensor(config.temperature_i2c_addr)

    broker = None
    if use_mqtt:
        from edge.messaging import LocalBroker

        broker = LocalBroker(config.mqtt_host, config.mqtt_port, "pi5-edge-node")
        broker.connect()

    return EdgeNode(config, distance, temperature, make_leds(config.leds, simulate), broker)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CE 4201 edge node")
    parser.add_argument("--simulate", action="store_true", help="fake sensors and LEDs")
    parser.add_argument("--mqtt", action="store_true", help="publish to the local broker")
    parser.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = forever)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    node = build_node(config, args.simulate, args.mqtt)
    try:
        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            node.step()
            cycle += 1
            if args.cycles == 0 or cycle < args.cycles:
                time.sleep(config.read_interval_s)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        node.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
