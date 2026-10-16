"""Decision rules: which LED should be on for a given sensor message.

These are pure functions with no hardware or network access, so they are easy
to test and they keep working with no internet.

Alerts are edge-triggered: ``AlertTracker`` reports a change only when an LED
goes from OFF to ON, so a chat or cloud alert is sent once, not on every
reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from edge.config import Thresholds, Topics

LED_RED = "red"
LED_YELLOW = "yellow"
LED_GREEN = "green"
LEDS = (LED_RED, LED_YELLOW, LED_GREEN)


@dataclass(frozen=True)
class LedDecision:
    led: str
    on: bool
    reason: str


def evaluate(
    topic: str, payload: dict[str, Any], thresholds: Thresholds, topics: Topics
) -> Optional[LedDecision]:
    """Return the LED decision for one message, or None if it is irrelevant
    or the payload is missing the field the rule needs."""
    if topic == topics.distance:
        value = payload.get("distance_cm")
        if not isinstance(value, (int, float)):
            return None
        on = value < thresholds.distance_cm
        return LedDecision(LED_RED, on, f"distance {value:.1f} cm (limit {thresholds.distance_cm} cm)")

    if topic == topics.temperature:
        value = payload.get("temperature_c")
        if not isinstance(value, (int, float)):
            return None
        on = value > thresholds.temperature_c
        return LedDecision(
            LED_YELLOW, on, f"temperature {value:.1f} C (limit {thresholds.temperature_c} C)"
        )

    if topic == topics.vision:
        label = payload.get("object")
        if not isinstance(label, str):
            return None
        on = label.lower() == thresholds.alert_object.lower()
        return LedDecision(LED_GREEN, on, f"saw {label!r}")

    return None


class AlertTracker:
    """Remembers each LED's state and reports only OFF -> ON changes."""

    def __init__(self) -> None:
        self._state: dict[str, bool] = {led: False for led in LEDS}

    def state(self, led: str) -> bool:
        return self._state[led]

    def update(self, decision: LedDecision) -> bool:
        """Record ``decision``. Returns True if this is a new OFF -> ON alert."""
        was_on = self._state[decision.led]
        self._state[decision.led] = decision.on
        return decision.on and not was_on


@dataclass(frozen=True)
class Alert:
    """A new OFF -> ON event that should be announced (Discord, AWS)."""

    led: str
    reason: str
