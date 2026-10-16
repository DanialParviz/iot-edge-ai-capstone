"""Combined local summary: read the three databases, write four lines, publish.

    python -m ml.combined --once --server http://127.0.0.1:8080
    python -m ml.combined --interval 30 --mqtt --aws

The model runs on the Pi (llama.cpp server). If it is unreachable or writes
anything that does not match the facts, the plain facts are published instead.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from edge.config import Config, load_config
from edge.storage import store_for
from ml.agent.llm import LlamaCppClient
from ml.summary import SummaryResult, collect_stats, summarize

log = logging.getLogger("combined")


def run_once(stores: dict[str, Any], llm, publisher, config: Config, n: int = 10) -> SummaryResult:
    print("AI Agent: Analyzing sensor data...", flush=True)
    result = summarize(llm, collect_stats(stores, n))
    for line in result.lines:
        print(line, flush=True)
    if result.source == "fallback":
        print(f"(plain facts used: {result.reason})", flush=True)
    if publisher is not None:
        payload = {
            "summary": "\n".join(result.lines),
            "lines": result.lines,
            "source": result.source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            publisher.publish(config.topics.combined_summary, payload)
        except Exception:
            log.exception("could not publish summary")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Combined SLM summary")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server URL")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--n", type=int, default=10, help="records per source")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--mqtt", action="store_true")
    parser.add_argument("--aws", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config()
    stores = {name: store_for(name, config.data_dir) for name in ("distance", "temperature", "vision")}
    publisher = None
    if args.mqtt or args.aws:
        from edge.cloud import build_publisher

        publisher = build_publisher(config, "pi5-combined-slm", args.mqtt, args.aws)
    llm = LlamaCppClient(args.server)
    try:
        while True:
            run_once(stores, llm, publisher, config, args.n)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if publisher is not None:
            publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
