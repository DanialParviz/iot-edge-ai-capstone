"""The offline-sync guarantee: nothing taken offline is lost, order is kept,
and a lost acknowledgement can be de-duplicated by msg_id."""

import pytest

from edge.messaging import FanoutPublisher
from edge.offline_queue import OfflineQueue, ReliablePublisher


class Transport:
    def __init__(self):
        self.online = True
        self.sent = []

    def publish(self, topic, payload):
        if not self.online:
            raise ConnectionError("offline")
        self.sent.append((topic, payload))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def parts(tmp_path):
    transport, clock = Transport(), Clock()
    queue = OfflineQueue(tmp_path / "outbox.sqlite")
    return transport, queue, ReliablePublisher(transport, queue, retry_after_s=10, clock=clock), clock


def test_online_messages_go_straight_through(parts):
    transport, queue, pub, _ = parts
    pub.publish("sensor/distance", {"distance_cm": 7.0})
    assert len(transport.sent) == 1 and queue.pending() == 0


def test_offline_messages_are_kept_then_delivered_in_order(parts):
    transport, queue, pub, clock = parts
    transport.online = False
    for i in range(5):
        pub.publish("sensor/distance", {"seq": i})
    assert transport.sent == [] and queue.pending() == 5

    transport.online = True
    clock.now = 11  # past the retry delay
    assert pub.flush() == 5
    assert [p["seq"] for _, p in transport.sent] == [0, 1, 2, 3, 4]
    assert queue.pending() == 0


def test_survives_a_restart(tmp_path):
    transport = Transport()
    transport.online = False
    path = tmp_path / "outbox.sqlite"
    first = ReliablePublisher(transport, OfflineQueue(path))
    first.publish("sensor/temperature", {"temperature_c": 29.0})
    first.queue.close()  # simulate power loss / process exit

    transport.online = True
    second = ReliablePublisher(transport, OfflineQueue(path), retry_after_s=0)
    assert second.flush() == 1
    assert transport.sent[0][1]["temperature_c"] == 29.0


def test_failed_send_blocks_retries_until_backoff_expires(parts):
    transport, queue, pub, clock = parts
    transport.online = False
    pub.publish("t", {"a": 1})
    attempts_before = len(transport.sent)
    transport.online = True
    pub.publish("t", {"a": 2})  # still inside the backoff window
    assert len(transport.sent) == attempts_before and queue.pending() == 2
    clock.now = 10
    pub.flush()
    assert queue.pending() == 0


def test_partial_failure_keeps_unsent_messages(parts):
    transport, queue, pub, clock = parts
    ok = transport.publish
    calls = {"n": 0}

    def flaky(topic, payload):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ConnectionError("dropped")
        ok(topic, payload)

    transport.publish = flaky
    transport.online = False
    for i in range(4):
        queue.enqueue("t", {"i": i})
    transport.online = True
    clock.now = 100
    assert pub.flush() == 2  # third send fails, stop there
    assert queue.pending() == 2
    clock.now = 200
    assert pub.flush() == 2
    assert [p["i"] for _, p in transport.sent] == [0, 1, 2, 3]


def test_every_message_gets_a_unique_msg_id_for_dedup(parts):
    _, queue, _, _ = parts
    queue.enqueue("t", {"x": 1})
    queue.enqueue("t", {"x": 1})
    ids = [p["msg_id"] for _, _, p in queue.peek()]
    assert len(set(ids)) == 2


def test_existing_msg_id_is_kept(parts):
    _, queue, _, _ = parts
    queue.enqueue("t", {"msg_id": "abc"})
    assert queue.peek()[0][2]["msg_id"] == "abc"


def test_resend_after_lost_ack_carries_same_msg_id(parts):
    """At-least-once: a message re-sent after a lost ack has the same msg_id."""
    transport, queue, pub, _ = parts
    queue.enqueue("t", {"v": 1})
    _, _, first = queue.peek()[0]
    transport.publish("t", first)  # sent, but the ack was "lost": row not deleted
    pub.flush()  # re-sent
    assert transport.sent[0][1]["msg_id"] == transport.sent[1][1]["msg_id"]


def test_fanout_isolates_a_failing_target():
    good, bad = Transport(), Transport()
    bad.online = False
    FanoutPublisher([bad, good]).publish("t", {"a": 1})
    assert len(good.sent) == 1  # local still works when the cloud is down
