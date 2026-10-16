"""Central configuration for the edge node.

Every value can be overridden with an environment variable so the same code
runs on the Pi and on a laptop. Secrets (AWS certificates, keys) are never
stored here; they stay in files that are git-ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Thresholds:
    """Limits that turn an LED (and an alert) on."""

    distance_cm: float = field(default_factory=lambda: _env_float("DISTANCE_ALERT_CM", 5.0))
    temperature_c: float = field(default_factory=lambda: _env_float("TEMP_ALERT_C", 30.0))
    alert_object: str = field(default_factory=lambda: os.environ.get("ALERT_OBJECT", "person"))


@dataclass(frozen=True)
class LedPins:
    """BCM GPIO numbers for the three indicator LEDs."""

    red: int = field(default_factory=lambda: _env_int("LED_RED_BCM", 17))
    yellow: int = field(default_factory=lambda: _env_int("LED_YELLOW_BCM", 27))
    green: int = field(default_factory=lambda: _env_int("LED_GREEN_BCM", 22))


@dataclass(frozen=True)
class Topics:
    """MQTT topic names (the same names are used locally and on AWS IoT)."""

    distance: str = "sensor/distance"
    temperature: str = "sensor/temperature"
    vision: str = "sensor/vision"
    alerts: str = "sensor/alerts"
    combined_summary: str = "sensor/combined/slm"


@dataclass(frozen=True)
class Config:
    mqtt_host: str = field(default_factory=lambda: os.environ.get("MQTT_HOST", "localhost"))
    mqtt_port: int = field(default_factory=lambda: _env_int("MQTT_PORT", 1883))
    read_interval_s: float = field(default_factory=lambda: _env_float("READ_INTERVAL_S", 5.0))
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR", "data/local-db"))
    )
    thresholds: Thresholds = field(default_factory=Thresholds)
    leds: LedPins = field(default_factory=LedPins)
    topics: Topics = field(default_factory=Topics)

    # Hardware facts from the build log.
    distance_i2c_addr: int = 0x29  # VL53L0X
    temperature_i2c_addr: int = 0x77  # BME680
    distance_invalid_mm: int = 8191  # sensor's "out of range" value


def load_config() -> Config:
    """Build a Config from the current environment."""
    return Config()
