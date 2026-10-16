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

## Status

Course: CE 4201, Fall 2026 — 15-week capstone timeline, in progress.

## License

MIT — see [LICENSE](LICENSE).
