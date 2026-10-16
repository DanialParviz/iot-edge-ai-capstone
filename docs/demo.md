# Live demo script (about 3 minutes)

Run on the Pi desktop terminal (the camera window cannot open over a remote shell, or use the
headless vision mode). Have the AWS IoT MQTT Test Client open on a laptop, subscribed to `sensor/#`.

```bash
cd ~/iot-edge-ai-capstone
source .venv/bin/activate
# terminal 1: the model server
llama-server -m <model>.gguf -c 2048 --port 8080
# terminal 2: everything else, one command (Ctrl+C stops all of it)
python -m edge.launcher --aws --faces
# terminal 3: watch local messages
mosquitto_sub -h localhost -t "sensor/#" -v
```

| Step | Do | Expect |
|---|---|---|
| 1 | Hand close to the distance sensor | Red LED on, one Discord alert, `sensor/alerts` in AWS |
| 2 | Warm the BME680 (hold your finger on it) | Yellow LED when above 30 °C |
| 3 | Step in front of the camera | Green LED, `person` on `sensor/vision`, `Person1` on `sensor/face-recognition` |
| 4 | Wait about 10 s | Four-line summary on `sensor/combined/slm` |
| 5 | Ask the agent: `python -m ml.agent.cli "What is the temperature and what did the camera see?"` | Tool calls, then an answer that matches the data |
| 6 | Unplug the internet for 30 s, then plug in | LEDs keep working; queued messages arrive after reconnecting |
| 7 | Ctrl+C once | Everything stops |

Do not turn Wi-Fi off from a Raspberry Pi Connect session during the demo.
