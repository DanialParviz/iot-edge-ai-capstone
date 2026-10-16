# Poster outline

**Title:** An edge-AI sensing node that keeps deciding when the internet goes away

**One-line claim:** A Raspberry Pi 5 senses, recognises, alerts and summarises locally, and
syncs to AWS whenever it can, with nothing lost in between.

## Layout (left to right)

1. **Problem and goal** (3 lines). Why decisions must stay on the device.
2. **System diagram.** Reuse the diagram in `architecture.md`: sensors → Pi → local path / cloud path / AI path.
3. **Hardware table.** VL53L0X, BME680, IMX708, 3 LEDs, with addresses and pins.
4. **Three paths, one reading each.**
   - Local: Mosquitto → LED + Discord (works offline).
   - Cloud: AWS IoT Core → DynamoDB (mTLS, offline outbox).
   - AI: local SLM four-line summary, checked against the facts.
5. **Design decisions that mattered** (pick 3):
   - Python counts, the model only writes (small models miscounted).
   - One process owns the LEDs; the agent only acts through approved tools.
   - Write-ahead outbox: no data lost offline.
6. **Results.** The numbers from `docs/test-plan.md`: offline sync (sent / delivered / drain time),
   summary valid rate and latency per model, face accept/reject rates, RAM use.
7. **Problems we solved.** OOM at 6.8 GB (capped context), database collisions (file per sensor),
   AWS policy (`pi5-*` client IDs), remote-shell camera window (headless mode).
8. **Limits and next steps.** One enrolled face, one display, Wi-Fi dependency for cloud.
9. **QR code** to the GitHub repo and a short live demo (see `docs/demo.md`).

## Figures to prepare

- Architecture diagram (vector).
- Photo of the wired Pi with labels.
- Screenshot: `sensor/#` arriving in the AWS MQTT Test Client.
- Chart: summary latency and valid rate, Qwen vs Granite.
- Chart or table: offline window, messages produced vs delivered.
