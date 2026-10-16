"""LED actuators.

``GpioLeds`` drives the real pins through gpiozero (Raspberry Pi only).
``FakeLeds`` just records state, so the rest of the code runs on a laptop.
Both satisfy the same small interface: ``set(led, on)`` and ``get(led)``.
"""

from __future__ import annotations

import logging

from edge.config import LedPins
from edge.rules import LEDS

log = logging.getLogger(__name__)


class FakeLeds:
    """In-memory LEDs for tests and for running without hardware."""

    def __init__(self) -> None:
        self._state = {led: False for led in LEDS}

    def set(self, led: str, on: bool) -> None:
        if led not in self._state:
            raise ValueError(f"unknown LED {led!r}; expected one of {LEDS}")
        self._state[led] = on
        log.info("LED %s -> %s", led, "ON" if on else "OFF")

    def get(self, led: str) -> bool:
        return self._state[led]

    def close(self) -> None:
        for led in self._state:
            self._state[led] = False


class GpioLeds:
    """Real LEDs on the Pi's GPIO header. Needs the ``gpiozero`` package."""

    def __init__(self, pins: LedPins) -> None:
        from gpiozero import LED  # imported here so laptops don't need it

        self._leds = {
            "red": LED(pins.red),
            "yellow": LED(pins.yellow),
            "green": LED(pins.green),
        }

    def set(self, led: str, on: bool) -> None:
        if led not in self._leds:
            raise ValueError(f"unknown LED {led!r}; expected one of {LEDS}")
        self._leds[led].on() if on else self._leds[led].off()

    def get(self, led: str) -> bool:
        return bool(self._leds[led].is_lit)

    def close(self) -> None:
        for device in self._leds.values():
            device.off()
            device.close()


def make_leds(pins: LedPins, simulate: bool):
    """Return fake LEDs when simulating, real GPIO LEDs otherwise."""
    return FakeLeds() if simulate else GpioLeds(pins)
