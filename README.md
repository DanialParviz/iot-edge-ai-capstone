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

| Area | Code | Notes |
|---|---|---|
| Config, thresholds, topics | `edge/config.py` | Env-overridable, no secrets |
| Per-sensor storage | `edge/storage.py` | One TinyDB file per source, avoids write collisions |
| LED decision rules | `edge/rules.py` | Pure functions; alerts fire only on OFF→ON |
| LEDs | `edge/actuators.py` | GPIO (`gpiozero`) or fake for laptops |
| Sensors | `edge/sensors/` | VL53L0X, BME680, plus simulated versions |
| Local MQTT | `edge/messaging.py` | paho-mqtt wrapper for Mosquitto |
| Edge loop | `edge/main.py` | read → store → publish → rules → LEDs |
| Agent tools | `ml/agent/tools.py` | Approved-tool registry; `set_led` is dry-run by default |
| Deterministic stats | `ml/agent/stats.py` | Python does all counting/math; the model only writes text |
| Agent loop + LLM client | `ml/agent/agent.py`, `llm.py` | llama.cpp server client (not yet run on the Pi) |

**Not yet done:** camera/TinyML detection, face recognition, AWS IoT Core publishing,
offline sync queue, and testing the agent against the Pi's local model.
The VL53L0X/BME680/GPIO drivers and the llama.cpp client are written but have not been run
on real hardware yet.

## Status

Course: CE 4201, Fall 2026 — 15-week capstone timeline, in progress.

## License

MIT — see [LICENSE](LICENSE).
