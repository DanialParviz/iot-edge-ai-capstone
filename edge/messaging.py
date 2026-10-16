"""MQTT helpers for the local Mosquitto broker.

Payloads are JSON. ``encode_payload`` / ``decode_payload`` are pure so they can
be tested without a broker. AWS IoT Core publishing (mutual TLS with device
certificates) is a separate adapter that will use the same topic names.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


def encode_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def decode_payload(raw: bytes | str) -> Optional[dict[str, Any]]:
    """Parse a message body; returns None if it is not a JSON object."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class LocalBroker:
    """Thin wrapper over paho-mqtt (>= 2.0) for publish and subscribe."""

    def __init__(self, host: str, port: int, client_id: str) -> None:
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self._host, self._port = host, port

    def connect(self) -> None:
        self._client.connect(self._host, self._port, keepalive=30)
        self._client.loop_start()

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._client.publish(topic, encode_payload(payload), qos=1)

    def subscribe(self, topic: str, handler: Callable[[str, dict[str, Any]], None]) -> None:
        def on_message(_client, _userdata, msg):
            data = decode_payload(msg.payload)
            if data is None:
                log.warning("ignoring non-JSON message on %s", msg.topic)
                return
            handler(msg.topic, data)

        self._client.on_message = on_message
        self._client.subscribe(topic, qos=1)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
