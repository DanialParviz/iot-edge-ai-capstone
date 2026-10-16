import json
import sys
import time

from edge.actuator_controller import ActuatorController
from edge.actuators import FakeLeds
from edge.alerts import DiscordNotifier, format_alert
from edge.config import Config
from edge.launcher import Launcher, Program
from edge.rules import Alert


class Recorder:
    def __init__(self):
        self.alerts, self.published = [], []

    def notify(self, alert):
        self.alerts.append(alert)

    def publish(self, topic, payload):
        self.published.append((topic, payload))


def make(notifier=None, publisher=None):
    leds = FakeLeds()
    return ActuatorController(Config(), leds, notifier, publisher), leds


def test_controller_alerts_once_and_announces_everywhere():
    rec = Recorder()
    ctl, leds = make(rec, rec)
    assert len(ctl.handle("sensor/temperature", {"temperature_c": 31.0})) == 1
    assert ctl.handle("sensor/temperature", {"temperature_c": 32.0}) == []
    assert leds.get("yellow") is True
    assert len(rec.alerts) == 1
    topic, payload = rec.published[0]
    assert topic == "sensor/alerts" and payload["led"] == "yellow" and "timestamp" in payload


def test_controller_ignores_its_own_alert_topic():
    ctl, _ = make(Recorder(), Recorder())
    assert ctl.handle("sensor/alerts", {"led": "red", "reason": "x"}) == []


def test_failing_notifier_never_stops_leds():
    class Broken:
        def notify(self, alert):
            raise RuntimeError("discord down")

        def publish(self, topic, payload):
            raise RuntimeError("cloud down")

    ctl, leds = make(Broken(), Broken())
    ctl.handle("sensor/distance", {"distance_cm": 2.0})
    assert leds.get("red") is True


def test_handle_command_accepts_valid_and_rejects_invalid():
    ctl, leds = make()
    assert ctl.handle_command("command/led", {"led": "green", "on": True}) is True
    assert leds.get("green") is True
    for bad in ({"led": "blue", "on": True}, {"led": "red", "on": "yes"}, {}):
        assert ctl.handle_command("command/led", bad) is False
    assert leds.get("red") is False


def test_format_alert_has_no_secrets_and_is_readable():
    text = format_alert(Alert("red", "distance 3.0 cm (limit 5.0 cm)"))
    assert "RED LED ON" in text and "3.0 cm" in text


def test_discord_notifier_sends_json_with_user_agent(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        seen["ua"] = request.get_header("User-agent")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert DiscordNotifier("https://example.invalid/hook").notify(Alert("red", "r")) is True
    assert "content" in seen["body"] and seen["ua"].startswith("ce4201")


def test_discord_notifier_returns_false_when_offline(monkeypatch):
    def boom(request, timeout):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert DiscordNotifier("https://example.invalid/hook").notify(Alert("red", "r")) is False


def test_launcher_starts_and_stops_everything_including_children():
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    # a program that spawns its own child, to prove the whole group is stopped
    parent = [sys.executable, "-c",
              "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
              "'import time; time.sleep(60)']); time.sleep(60)"]
    launcher = Launcher([Program("a", sleeper), Program("b", parent)], grace_s=3)
    launcher.start()
    time.sleep(0.5)
    assert sorted(launcher.running()) == ["a", "b"]
    started = time.monotonic()
    launcher.stop()
    assert launcher.running() == [] and time.monotonic() - started < 5


def test_launcher_kills_a_program_that_ignores_sigterm():
    stubborn = [sys.executable, "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]
    launcher = Launcher([Program("stubborn", stubborn)], grace_s=1)
    launcher.start()
    time.sleep(0.5)
    launcher.stop()
    assert launcher.running() == []
