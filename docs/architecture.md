# Architecture

Raspberry Pi 5 (8 GB) edge node: read sensors, decide locally, send results to AWS.

```
VL53L0X (I2C 0x29) ─┐
BME680  (I2C 0x77) ─┼─► edge.main ──┬─► TinyDB (data/local-db/*_data.json)
IMX708 camera ──────┘               ├─► Mosquitto (local MQTT) ─► edge.actuator_controller ─► 3 LEDs
   edge.vision.run_vision ──────────┤                                   └─► Discord alert + AWS sensor/alerts
   (EfficientDet-Lite0 + YuNet/EdgeFace)
                                    └─► AWS IoT Core (mTLS) ─► IoT Rule ─► DynamoDB
                                             ▲
                         durable outbox (SQLite) ── holds messages while offline

ml.combined: last 10 records of each database ─► Python stats ─► local SLM (llama.cpp) ─► 4 checked lines
             ─► sensor/combined/slm
ml.agent:    goal ─► SLM picks an approved tool ─► result ─► ... ─► answer
```

## Design rules

1. **Python does the math; the model writes sentences.** Small models miscounted, so
   counts, ranges and averages come from `ml/agent/stats.py`. The summary is then checked
   (`ml/summary.py`): four lines in order, every number exactly as in the facts. Otherwise
   the plain facts are sent instead.
2. **One process owns the LEDs.** GPIO pins cannot be shared. `edge/actuator_controller.py`
   subscribes to `sensor/#` and `command/led`; everything else only publishes.
3. **The agent never drives hardware directly.** It can only call the tools in
   `ml/agent/tools.py`. `set_led` is a dry run unless `--live`, and then only publishes
   `command/led`, which the controller validates.
4. **Local control works offline.** LEDs depend only on localhost MQTT. The cloud path
   (Discord, AWS) is best-effort and never blocks sensing.
5. **Nothing is lost offline.** Every cloud message is written to the SQLite outbox first
   (`edge/offline_queue.py`), sent oldest-first, deleted only after the broker acknowledges
   it, and tagged with a `msg_id` so a re-sent duplicate can be recognised.
6. **Separate database per source.** Two programs writing one TinyDB file caused ID errors.

## Topics

| Topic | Payload |
|---|---|
| `sensor/distance` | `distance_cm`, `timestamp` |
| `sensor/temperature` | `temperature_c`, `timestamp` |
| `sensor/vision` | `object`, `confidence`, `timestamp` |
| `sensor/face-recognition` | `identity` (`Person1`/`Unknown`), `similarity`, `timestamp` |
| `sensor/alerts` | `led`, `reason`, `timestamp` |
| `sensor/combined/slm` | `summary`, `lines`, `source` (`slm`/`fallback`), `timestamp` |
| `command/led` (local only) | `led`, `on` |

AWS client IDs must start with `pi5-` (IoT policy). Secrets (certificates, private key,
Discord webhook) come from environment variables and files outside git.

## Environment variables

| Variable | Purpose |
|---|---|
| `AWS_IOT_ENDPOINT`, `AWS_IOT_CERT`, `AWS_IOT_KEY`, `AWS_IOT_CA` | AWS IoT connection kit |
| `DISCORD_WEBHOOK_URL` | Discord alerts (optional) |
| `MQTT_HOST`, `MQTT_PORT` | Local broker (default `localhost:1883`) |
| `DATA_DIR` | Database/outbox folder (default `data/local-db`) |
| `DISTANCE_ALERT_CM`, `TEMP_ALERT_C`, `ALERT_OBJECT` | Alert thresholds |
| `LED_RED_BCM`, `LED_YELLOW_BCM`, `LED_GREEN_BCM` | LED pins (17, 27, 22) |
| `READ_INTERVAL_S` | Sensor period (default 5) |

## Known limits

- Face recognition knows one enrolled person versus "Unknown".
- Screen sharing shows one display; the OpenCV window needs the Pi desktop (use headless
  mode over a remote shell).
- Turning off Wi-Fi during a Raspberry Pi Connect session ends the session.
