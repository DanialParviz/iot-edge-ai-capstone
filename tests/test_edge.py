from edge.actuators import FakeLeds
from edge.config import Config, Thresholds, Topics
from edge.main import EdgeNode
from edge.messaging import decode_payload, encode_payload
from edge.rules import AlertTracker, LedDecision, evaluate
from edge.sensors.distance import SimulatedDistanceSensor, mm_to_cm
from edge.sensors.temperature import SimulatedTemperatureSensor
from edge.storage import store_for

T, TOPICS = Thresholds(), Topics()


# --- sensor conversion -------------------------------------------------
def test_mm_to_cm_converts():
    assert mm_to_cm(123) == 12.3


def test_mm_to_cm_rejects_invalid_value():
    assert mm_to_cm(8191) is None


def test_mm_to_cm_rejects_zero_negative_and_none():
    assert mm_to_cm(0) is None
    assert mm_to_cm(-5) is None
    assert mm_to_cm(None) is None


# --- rules -------------------------------------------------------------
def test_distance_rule_boundary():
    below = evaluate(TOPICS.distance, {"distance_cm": 4.9}, T, TOPICS)
    at = evaluate(TOPICS.distance, {"distance_cm": 5.0}, T, TOPICS)
    assert below.led == "red" and below.on
    assert not at.on  # limit is strictly "< 5 cm"


def test_temperature_rule_boundary():
    at = evaluate(TOPICS.temperature, {"temperature_c": 30.0}, T, TOPICS)
    above = evaluate(TOPICS.temperature, {"temperature_c": 30.1}, T, TOPICS)
    assert not at.on  # limit is strictly "> 30 C"
    assert above.led == "yellow" and above.on


def test_vision_rule_matches_person_case_insensitively():
    assert evaluate(TOPICS.vision, {"object": "Person"}, T, TOPICS).on
    assert not evaluate(TOPICS.vision, {"object": "bird"}, T, TOPICS).on


def test_rule_ignores_unknown_topic_and_missing_fields():
    assert evaluate("sensor/other", {"distance_cm": 1}, T, TOPICS) is None
    assert evaluate(TOPICS.distance, {}, T, TOPICS) is None
    assert evaluate(TOPICS.distance, {"distance_cm": "close"}, T, TOPICS) is None


def test_alert_fires_only_on_off_to_on():
    tracker = AlertTracker()
    on = LedDecision("red", True, "x")
    off = LedDecision("red", False, "x")
    assert tracker.update(on) is True  # OFF -> ON
    assert tracker.update(on) is False  # still ON
    assert tracker.update(off) is False  # ON -> OFF
    assert tracker.update(on) is True  # OFF -> ON again


# --- storage -----------------------------------------------------------
def test_store_latest_returns_newest_in_order(tmp_path):
    store = store_for("distance", tmp_path)
    for i in range(5):
        store.insert({"distance_cm": float(i)})
    assert [r["distance_cm"] for r in store.latest(3)] == [2.0, 3.0, 4.0]
    assert store.latest(0) == []
    assert store.count() == 5


def test_sources_use_separate_files(tmp_path):
    a, b = store_for("distance", tmp_path), store_for("temperature", tmp_path)
    a.insert({"x": 1})
    b.insert({"x": 2})  # would collide if they shared one file
    assert a.path != b.path and a.count() == 1 and b.count() == 1


def test_store_rejects_unknown_source(tmp_path):
    try:
        store_for("lidar", tmp_path)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- messaging ---------------------------------------------------------
def test_payload_round_trip():
    payload = {"b": 2, "a": 1.5}
    assert decode_payload(encode_payload(payload)) == payload


def test_decode_rejects_garbage_and_non_objects():
    assert decode_payload(b"not json") is None
    assert decode_payload("[1, 2]") is None


# --- node wiring -------------------------------------------------------
class FakeBroker:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload):
        self.published.append((topic, payload))

    def close(self):
        pass


class StubSensor:
    def __init__(self, records):
        self._records = list(records)

    def read(self):
        return self._records.pop(0)


def _node(tmp_path, distance, temperature, broker=None):
    config = Config(data_dir=tmp_path)
    return EdgeNode(config, distance, temperature, FakeLeds(), broker)


def test_step_stores_publishes_and_alerts_once(tmp_path):
    close = {"distance_cm": 3.0}
    temp = {"temperature_c": 25.0}
    broker = FakeBroker()
    node = _node(tmp_path, StubSensor([close, close]), StubSensor([temp, temp]), broker)

    first = node.step()
    second = node.step()

    assert [a.led for a in first] == ["red"]
    assert second == []  # same condition: no repeat alert
    assert node.leds.get("red") is True
    assert node.stores["distance"].count() == 2
    topics = [t for t, _ in broker.published]
    assert topics.count("sensor/distance") == 2
    assert topics.count("sensor/alerts") == 1
    node.close()


def test_step_skips_invalid_reading(tmp_path):
    node = _node(tmp_path, StubSensor([None]), StubSensor([{"temperature_c": 20.0}]))
    node.step()
    assert node.stores["distance"].count() == 0
    assert node.stores["temperature"].count() == 1
    node.close()


def test_led_turns_off_when_condition_clears(tmp_path):
    node = _node(
        tmp_path,
        StubSensor([{"distance_cm": 3.0}, {"distance_cm": 20.0}]),
        StubSensor([{"temperature_c": 20.0}] * 2),
    )
    node.step()
    assert node.leds.get("red") is True
    node.step()
    assert node.leds.get("red") is False
    node.close()


def test_vision_message_lights_green(tmp_path):
    node = _node(tmp_path, StubSensor([]), StubSensor([]))
    alerts = node.apply("sensor/vision", {"object": "person", "confidence": 0.8})
    assert [a.led for a in alerts] == ["green"] and node.leds.get("green")
    node.close()


def test_simulated_sensors_produce_valid_records():
    d, t = SimulatedDistanceSensor(seed=1), SimulatedTemperatureSensor(seed=1)
    readings = [d.read() for _ in range(40)]
    assert all(r is None or r["distance_cm"] > 0 for r in readings)
    assert any(r and r["distance_cm"] < 5 for r in readings)  # crosses the alert limit
    assert all("temperature_c" in t.read() for _ in range(10))
