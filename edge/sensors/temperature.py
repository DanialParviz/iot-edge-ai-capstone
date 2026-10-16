"""BME680 environmental sensor (I2C address 0x77), temperature only."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any, Optional


def make_record(temperature_c: float) -> dict[str, Any]:
    return {
        "sensor": "temperature",
        "temperature_c": round(temperature_c, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class Bme680Sensor:
    """Real sensor. Needs adafruit-circuitpython-bme680 on the Pi."""

    def __init__(self, address: int = 0x77) -> None:
        import board
        import busio
        import adafruit_bme680

        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=address)

    def read(self) -> Optional[dict[str, Any]]:
        return make_record(float(self._sensor.temperature))


class SimulatedTemperatureSensor:
    """Fake temperature that drifts around 29 C so the 30 C limit gets crossed."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._t = 0

    def read(self) -> Optional[dict[str, Any]]:
        self._t += 1
        value = 29 + 2.5 * math.sin(self._t / 9) + self._rng.uniform(-0.2, 0.2)
        return make_record(value)
