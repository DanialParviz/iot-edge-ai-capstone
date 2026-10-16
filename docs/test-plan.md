# Test plan and measurements

Unit tests (`pytest`) cover the logic that does not need hardware. This plan covers
what must be checked **on the Pi**, and produces the numbers for the poster.
Record results in `docs/results.md`.

## 1. Offline data reaches DynamoDB (open task)

Goal: prove readings taken with no internet arrive after reconnecting, in order, with no loss.

1. Start the system with cloud publishing: `python -m edge.launcher --aws ...`
2. Note the DynamoDB item count (or watch `sensor/#` in the MQTT Test Client).
3. **Keep local access** (monitor or SSH over Ethernet). Do **not** disable Wi-Fi from a
   Raspberry Pi Connect session, it will drop the session. Instead block only the internet
   path (unplug the uplink or block the AWS endpoint with a firewall rule).
4. Wait 2 minutes. Confirm the LEDs and `mosquitto_sub -t "sensor/#" -v` still work.
5. Check pending messages: `sqlite3 data/local-db/outbox_pi5-edge-node.sqlite "select count(*) from outbox"`
   (it should be about 24 per sensor for 2 minutes at 5 s).
6. Restore the connection. Within ~10 s the outbox should drain to 0.
7. In DynamoDB, count the items for the offline window and compare with the local TinyDB
   count for the same timestamps. Duplicates share a `msg_id`.

Record: messages produced offline, messages delivered, duplicates, time to drain.

## 2. Face-aware four-line summary (open task)

1. Run the vision node with `--faces`, step in front of the camera, then walk away.
2. Run `python -m ml.combined --once`. Expect four lines: Distance, Temperature, Vision, Faces.
3. Check the `source` field in `sensor/combined/slm`: `slm` means the model's text passed the
   number check; `fallback` means plain facts were sent (see `reason` in the terminal).
4. Compare the Faces line with the real count in `data/local-db/vision_data.json`.

## 3. Model comparison, Qwen vs IBM Granite (open task)

The professor asked for US-made models. For each model:

```
llama-server -m <model>.gguf -c 2048 --port 8080          # add --jinja for tool calling
python -m ml.benchmark_slm --label "<name>" --runs 10 --out docs/results.jsonl
python -m ml.agent.cli "How warm is it and what did the camera see?"   # tool-calling check
```

Watch memory while it runs: `ps -o rss= -p $(pgrep llama-server)`. Fill in:

| Model | Median latency | p95 latency | Valid-summary rate | Peak RAM | Tool calling works? |
|---|---|---|---|---|---|
| Qwen2.5-3B Q4_K_M | | | | | |
| IBM Granite (state exact file) | | | | | |

Choose the model by valid rate first, latency second. Whether a given Granite file calls
tools correctly through llama.cpp depends on its chat template; verify rather than assume.

## 4. Face recognition accuracy

Capture N frames of Person1 and N frames of other people under the same lighting. Report true
accept rate for Person1, false accept rate for others, and the threshold used (default 0.5).

## 5. Agent on the Pi (graded Experiment 4)

Ask 5 fixed questions (distance, temperature, camera objects, system summary, a refused
request such as "delete the database") and record: correct tools chosen, answer matches the
Python stats, and refused-request behaviour. `ml.agent.cli` prints the tool log.
