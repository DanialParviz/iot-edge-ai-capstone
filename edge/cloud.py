"""Builders that assemble the publishers used by the edge programs."""

from __future__ import annotations

from edge.config import Config
from edge.messaging import FanoutPublisher, LocalBroker


def build_cloud_publisher(config: Config, client_id: str):
    """AWS IoT publisher backed by a durable on-disk outbox."""
    from edge.aws_iot import AwsIotTransport, settings_from_env
    from edge.offline_queue import OfflineQueue, ReliablePublisher

    transport = AwsIotTransport(client_id=client_id, **settings_from_env())
    queue = OfflineQueue(config.data_dir / f"outbox_{client_id}.sqlite")
    return ReliablePublisher(transport, queue)


def build_publisher(config: Config, client_id: str, use_mqtt: bool, use_aws: bool):
    """Local Mosquitto, AWS IoT, or both (local keeps working with no internet)."""
    targets = []
    if use_mqtt:
        broker = LocalBroker(config.mqtt_host, config.mqtt_port, client_id)
        broker.connect()
        targets.append(broker)
    if use_aws:
        targets.append(build_cloud_publisher(config, client_id))
    if not targets:
        raise ValueError("choose at least one of use_mqtt / use_aws")
    return targets[0] if len(targets) == 1 else FanoutPublisher(targets)
