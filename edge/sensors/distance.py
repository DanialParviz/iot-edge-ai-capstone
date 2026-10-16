"""VL53L0X time-of-flight distance sensor (I2C address 0x29)."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any, Optional

INVALID_MM = 8191  # the sensor reports this when nothing is in range


def mm_to_cm(range_mm: int, invalid_mm: int = INVALID_MM) -> Optional[float]:
    """Convert a raw millimetre reading to centimetres.

    Returns None for the sensor's invalid/out-of-range value or any reading
    that is not a positive number, so bad data never reaches storage.
    """
    if range_mm is None or range_mm == invalid_mm or range_mm <= 0:
        return None
    return round(range_mm / 10.0, 1)


def make_record(distance_cm: float) -> dict[str, Any]:
    return {
        "sensor": "distance",
        "distance_cm": distance_cm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class Vl53l0xSensor:
    """Real sensor. Needs adafruit-circuitpython-vl53l0x on the Pi."""

    def __init__(self) -> None:
        import board
        import busio
        import adafruit_vl53l0x

        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_vl53l0x.VL53L0X(i2c)

    def read(self) -> Optional[dict[str, Any]]:
        cm = mm_to_cm(self._sensor.range)
        return None if cm is None else make_record(cm)


class SimulatedDistanceSensor:
    """Smoothly varying fake distance that dips below 5 cm now and then."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._t = 0

    def read(self) -> Optional[dict[str, Any]]:
        self._t += 1
        base = 15 + 12 * math.sin(self._t / 6)  # roughly 3..27 cm
        cm = mm_to_cm(int(max(base + self._rng.uniform(-1, 1), 1) * 10))
        return None if cm is None else make_record(cm)
