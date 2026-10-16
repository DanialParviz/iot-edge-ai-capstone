"""Local storage: one TinyDB JSON file per data source.

A single shared file caused duplicate document-ID errors when two sensor
programs wrote at once, so each source (distance, temperature, vision) gets
its own file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tinydb import TinyDB

SOURCES = ("distance", "temperature", "vision")


class SensorStore:
    """Append-only store for the readings of a single sensor."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = TinyDB(self.path)

    def insert(self, record: dict[str, Any]) -> int:
        return self._db.insert(record)

    def latest(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the newest ``n`` records, oldest first."""
        if n <= 0:
            return []
        for attempt in range(3):
            try:
                records = self._db.all()
                break
            except json.JSONDecodeError:  # another process was mid-write
                if attempt == 2:
                    raise
                time.sleep(0.05)
        return [dict(r) for r in records[-n:]]

    def count(self) -> int:
        return len(self._db)

    def close(self) -> None:
        self._db.close()


def store_for(source: str, data_dir: Path | str) -> SensorStore:
    """Open the store for ``source`` under ``data_dir``."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return SensorStore(Path(data_dir) / f"{source}_data.json")
