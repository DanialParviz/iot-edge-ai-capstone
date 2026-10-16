# IoT Edge AI Capstone

End-to-end IoT/AIoT prototype built for UT Dallas CE 4201 (Internet of Things Laboratory). The system integrates edge sensing, networking, cloud ingestion, and on-device AI into a working pipeline, presented via live demo and technical poster at UT Dallas.

## Team

- Subhi Alhalak (Lead)
- Geon Kang
- Danial Parviz

## System Overview

A Raspberry Pi 5 edge node collects data from a heterogeneous sensor/actuator array, runs on-device inference, and streams telemetry to the cloud for monitoring and control.

```
Sensors/Actuators → Raspberry Pi 5 (edge) → MQTT → AWS IoT Core → Dashboard
                          │
                          ├─ TinyML inference (on-device)
                          └─ SLM agentic workflow (tool-calling over device/system state)
```

**Pipeline stages:**

1. **Sensing** — heterogeneous sensors/actuators wired into the Raspberry Pi 5
2. **Edge compute** — TinyML models run locally for low-latency inference
3. **Networking** — MQTT publishes sensor telemetry from the edge node
4. **Cloud ingestion** — AWS IoT Core receives and routes telemetry
5. **Monitoring/control** — real-time dashboard for telemetry visualization and automated actuator control
6. **AI agent layer** — a small language model (SLM) interprets system state and executes device actions via tool calling

## Repository Structure

```
├── edge/           # Raspberry Pi edge device code (sensor I/O, actuator control, MQTT client)
├── cloud/          # AWS IoT Core config, ingestion, telemetry/dashboard backend
├── ml/             # TinyML models and SLM agentic workflow/tool-calling code
├── docs/           # Design docs, architecture notes, poster/presentation materials
└── tests/          # Test scripts and validation data
```

## Tech Stack

- **Hardware:** Raspberry Pi 5, sensors/actuators
- **Edge AI:** TinyML
- **Messaging:** MQTT
- **Cloud:** AWS IoT Core
- **AI Agent:** Small Language Model (SLM) with tool calling

## Quick Start (no hardware needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # run the tests
python -m edge.main --simulate --cycles 20   # fake sensors + LEDs, stores to data/local-db/
```

On the Raspberry Pi, install `requirements-pi.txt` instead and run `python -m edge.main`
(add `--mqtt` to publish to the local Mosquitto broker). Settings such as thresholds,
LED pins and the broker address are environment variables; see `edge/config.py`.

## What's Implemented

| Area | Code | Status |
|---|---|---|
| Config, thresholds, topics | `edge/config.py` | tested |
| Per-sensor storage (TinyDB) | `edge/storage.py` | tested |
| LED rules, edge-triggered alerts | `edge/rules.py` | tested |
| LED controller (sole GPIO owner) + `command/led` | `edge/actuator_controller.py` | tested |
| LEDs (GPIO or fake) | `edge/actuators.py` | fake tested; GPIO needs the Pi |
| Discord alerts | `edge/alerts.py` | tested with a mocked network |
| Distance + temperature sensors | `edge/sensors/`, `edge/main.py` | logic tested; drivers need the Pi |
| Local MQTT + fan-out publishing | `edge/messaging.py` | payload logic tested; broker needs Mosquitto |
| AWS IoT Core transport | `edge/aws_iot.py` | needs verifying on the Pi |
| **Offline outbox** (no data lost, ordered, de-duplicable) | `edge/offline_queue.py` | tested |
| Camera: objects (EfficientDet-Lite0) + faces (YuNet/EdgeFace) | `edge/vision/` | matching/records tested; models need the Pi |
| One-command launcher, clean Ctrl+C | `edge/launcher.py` | tested |
| Deterministic stats | `ml/agent/stats.py` | tested |
| Checked 4-line summary (Distance/Temperature/Vision/Faces) | `ml/summary.py`, `ml/combined.py` | tested |
| Tool-calling agent (approved tools only) | `ml/agent/` | loop tested with a scripted model |
| Model benchmark (Qwen vs IBM Granite) | `ml/benchmark_slm.py` | harness tested; run it on the Pi |

Run `pytest` for the unit tests. Anything marked "needs the Pi" was written to the project's
documented setup but has not been run on the real hardware from this repo yet. The checklist
for doing that, and the measurements for the poster, is in [`docs/test-plan.md`](docs/test-plan.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md): data flow, design rules, topics, environment variables
- [`docs/test-plan.md`](docs/test-plan.md): on-Pi verification and measurements
- [`docs/demo.md`](docs/demo.md): live demo script
- [`docs/poster-outline.md`](docs/poster-outline.md): poster layout

## Status

Course: CE 4201, Fall 2026 — 15-week capstone timeline, in progress.

## License

MIT — see [LICENSE](LICENSE).
