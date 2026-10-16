"""Benchmark a model behind a llama.cpp server on the four-line summary task.

Use it to compare models on the Pi (for example the current Qwen2.5-3B against
an IBM Granite model) and to get numbers for the poster:

    # terminal 1: serve one model
    llama-server -m <model>.gguf -c 2048 --port 8080
    # terminal 2:
    python -m ml.benchmark_slm --label "Qwen2.5-3B Q4_K_M" --runs 10 --out docs/results.jsonl

Reported per model: median / p95 / max latency, and the *valid rate*: the
share of runs whose text passed the number-and-format check in ``ml.summary``
(the rest would have fallen back to plain facts). Memory is not measured here;
watch it with ``top`` or ``ps -o rss= -p <llama-server pid>`` while it runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Optional

from ml.agent.llm import LlamaCppClient
from ml.summary import summarize

FIXTURE_STATS = {
    "distance_cm": {"count": 10, "min": 6.6, "max": 7.7, "mean": 7.1, "latest": 7.0},
    "temperature_c": {"count": 10, "min": 28.1, "max": 30.9, "mean": 29.5, "latest": 30.2},
    "object_counts": {"person": 6, "bird": 2, "bear": 1},
    "face_counts": {"Person1": 4, "Unknown": 2},
}


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    index = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def run_benchmark(llm, runs: int, stats: Optional[dict[str, Any]] = None,
                  clock=time.perf_counter) -> dict[str, Any]:
    stats = stats or FIXTURE_STATS
    latencies: list[float] = []
    valid = 0
    reasons: dict[str, int] = {}
    for _ in range(runs):
        start = clock()
        result = summarize(llm, stats)
        latencies.append(clock() - start)
        if result.source == "slm":
            valid += 1
        else:
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
    ordered = sorted(latencies)
    return {
        "runs": runs,
        "valid_rate": round(valid / runs, 3) if runs else 0.0,
        "median_s": round(statistics.median(ordered), 2) if ordered else None,
        "p95_s": round(percentile(ordered, 0.95), 2) if ordered else None,
        "max_s": round(ordered[-1], 2) if ordered else None,
        "fallback_reasons": reasons,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SLM benchmark")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--label", required=True, help="model name for the results table")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", help="append the result as a JSON line to this file")
    args = parser.parse_args(argv)

    result = {"model": args.label, **run_benchmark(LlamaCppClient(args.server), args.runs)}
    print(json.dumps(result, indent=2))
    print(f"| {args.label} | {result['median_s']} s | {result['p95_s']} s | "
          f"{result['valid_rate'] * 100:.0f}% |")
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
