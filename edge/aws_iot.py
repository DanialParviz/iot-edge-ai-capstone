"""AWS IoT Core transport (mutual TLS with the device certificate).

Certificates and the private key are read from paths in environment
variables; they live outside git (see .gitignore). Client IDs must start with
``pi5-`` to match the IoT policy.

    AWS_IOT_ENDPOINT   a1xxxxxxxx-ats.iot.us-east-2.amazonaws.com
    AWS_IOT_CERT       path to the device certificate (.pem.crt)
    AWS_IOT_KEY        path to the private key
    AWS_IOT_CA         path to the Amazon Root CA

NOTE: not yet run against the real AWS account from this repo; verify on the Pi.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from edge.messaging import encode_payload

log = logging.getLogger(__name__)

_ENV = {
    "endpoint": "AWS_IOT_ENDPOINT",
    "cert": "AWS_IOT_CERT",
    "key": "AWS_IOT_KEY",
    "ca": "AWS_IOT_CA",
}


def settings_from_env() -> dict[str, str]:
    missing = [name for name in _ENV.values() if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing environment variable(s): " + ", ".join(missing))
    return {key: os.environ[name] for key, name in _ENV.items()}


class AwsIotTransport:
    def __init__(self, endpoint: str, cert: str, key: str, ca: str, client_id: str,
                 timeout_s: float = 5.0) -> None:
        if not client_id.startswith("pi5-"):
            raise ValueError("client_id must start with 'pi5-' to match the IoT policy")
        self._args = dict(endpoint=endpoint, cert=cert, key=key, ca=ca)
        self._client_id = client_id
        self._timeout = timeout_s
        self._conn = None

    def _connect(self) -> None:
        from awsiot import mqtt_connection_builder

        conn = mqtt_connection_builder.mtls_from_path(
            endpoint=self._args["endpoint"],
            port=8883,
            cert_filepath=self._args["cert"],
            pri_key_filepath=self._args["key"],
            ca_filepath=self._args["ca"],
            client_id=self._client_id,
            clean_session=False,
            keep_alive_secs=30,
        )
        conn.connect().result(timeout=self._timeout)
        self._conn = conn
        log.info("connected to AWS IoT as %s", self._client_id)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish with QoS 1 and wait for the broker's acknowledgement.
        Raises on any failure so the caller can queue the message."""
        from awscrt import mqtt

        if self._conn is None:
            self._connect()
        try:
            future, _ = self._conn.publish(
                topic=topic, payload=encode_payload(payload), qos=mqtt.QoS.AT_LEAST_ONCE
            )
            future.result(timeout=self._timeout)
        except Exception:
            self._drop()  # reconnect cleanly on the next attempt
            raise

    def _drop(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.disconnect().result(timeout=2)
            except Exception:
                pass

    def close(self) -> None:
        self._drop()
