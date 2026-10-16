"""Discord alert notifier (needs internet; the LEDs do not).

The webhook URL is a secret. It is read from the DISCORD_WEBHOOK_URL
environment variable and is never logged or committed.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional

from edge.rules import Alert

log = logging.getLogger(__name__)


def format_alert(alert: Alert, node_name: str = "SA-pi5") -> str:
    return f"[{node_name}] {alert.led.upper()} LED ON: {alert.reason}"


class DiscordNotifier:
    def __init__(self, webhook_url: str, timeout_s: float = 5.0) -> None:
        self._url = webhook_url
        self._timeout = timeout_s

    def notify(self, alert: Alert) -> bool:
        """Send one alert. Returns False (never raises) if offline or rejected."""
        body = json.dumps({"content": format_alert(alert)}).encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                # Discord's CDN rejects urllib's default User-Agent.
                "User-Agent": "ce4201-edge-node/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):  # noqa: S310
                return True
        except Exception as exc:  # offline, DNS failure, HTTP error ...
            log.warning("Discord alert not sent: %s", type(exc).__name__)
            return False


def notifier_from_env() -> Optional[DiscordNotifier]:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    return DiscordNotifier(url) if url else None
