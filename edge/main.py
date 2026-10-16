"""Edge node entry point for the distance and temperature sensors.

    python -m edge.main --simulate              # laptop, no hardware, no broker
    python -m edge.main --simulate --cycles 5   # run 5 read cycles then stop
    python -m edge.main                         # on the Pi: real sensors + GPIO
    python -m edge.main --mqtt --no-leds        # publish only (another process owns the LEDs)
    python -m edge.main --mqtt --aws            # also publish to AWS IoT with an offline queue

Each cycle: read every sensor, store the reading, publish it, then (unless
``--no-leds``) apply the LED rules. Alerts fire only on an OFF -> ON change.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Optional

from edge.actuator_controller import ActuatorController
from edge.alerts import notifier_from_env
from edge.actuators import make_leds
from edge.config import Config, load_config
from edge.rules import Alert
from edge.storage import SensorStore, store_for

log = logging.getLogger("edge")


class EdgeNode:
    """Wires sensors, storage, publishing and (optionally) the LED rules."""

    def __init__(self, config: Config, distance, temperature, leds=None, broker=None,
                 stores: Optional[dict[str, SensorStore]] = None, notifier=None) -> None:
        self.config = config
        self.sensors = {"distance": distance, "temperature": temperature}
        self.leds = leds
        self.broker = broker  # anything with publish(topic, payload) and close()
        self.stores = stores or {
            name: store_for(name, config.data_dir) for name in self.sensors
        }
        self.controller = (
            ActuatorController(config, leds, notifier, broker) if leds is not None else None
        )
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
                self._publish(topic, record)
            alerts.extend(self.apply(topic, record))
        return alerts

    def _publish(self, topic: str, record: dict[str, Any]) -> None:
        try:
            self.broker.publish(topic, record)
        except Exception:  # a broken link must never stop sensing
            log.exception("publish failed for %s", topic)

    def apply(self, topic: str, payload: dict[str, Any]) -> list[Alert]:
        """Apply the LED rule for one message (no-op when LEDs are disabled)."""
        return self.controller.handle(topic, payload) if self.controller else []

    def close(self) -> None:
        if self.leds is not None:
            self.leds.close()
        for store in self.stores.values():
            store.close()
        if self.broker is not None:
            self.broker.close()


def build_node(config: Config, simulate: bool, use_mqtt: bool, use_aws: bool,
               use_leds: bool) -> EdgeNode:
    if simulate:
        from edge.sensors.distance import SimulatedDistanceSensor
        from edge.sensors.temperature import SimulatedTemperatureSensor

        distance, temperature = SimulatedDistanceSensor(), SimulatedTemperatureSensor()
    else:
        from edge.sensors.distance import Vl53l0xSensor
        from edge.sensors.temperature import Bme680Sensor

        distance, temperature = Vl53l0xSensor(), Bme680Sensor(config.temperature_i2c_addr)

    broker = None
    if use_mqtt or use_aws:
        from edge.cloud import build_publisher

        broker = build_publisher(config, "pi5-edge-node", use_mqtt, use_aws)

    leds = make_leds(config.leds, simulate) if use_leds else None
    return EdgeNode(config, distance, temperature, leds, broker, notifier=notifier_from_env())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CE 4201 edge node")
    parser.add_argument("--simulate", action="store_true", help="fake sensors and LEDs")
    parser.add_argument("--mqtt", action="store_true", help="publish to the local broker")
    parser.add_argument("--aws", action="store_true", help="publish to AWS IoT (offline queue)")
    parser.add_argument("--no-leds", action="store_true", help="do not drive LEDs here")
    parser.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = forever)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    node = build_node(config, args.simulate, args.mqtt, args.aws, not args.no_leds)
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
