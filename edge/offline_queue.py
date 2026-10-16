"""Durable outbox so readings taken offline reach the cloud later.

Design (write-ahead, at-least-once):

1. Every message is written to a SQLite outbox *before* any send is tried, so a
   crash, power loss or lost Wi-Fi cannot lose it.
2. ``flush`` sends oldest-first and deletes a row only after the transport
   confirms the send. If a send fails, it stops (order is preserved) and waits
   ``retry_after_s`` before trying again, so an offline Pi never blocks the
   sensor loop on connection timeouts.
3. If a send succeeds but its acknowledgement is lost, the message is sent
   again. Each message carries a unique ``msg_id`` so the cloud side can drop
   duplicates.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class OfflineQueue:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS outbox ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._db.commit()

    def enqueue(self, topic: str, payload: dict[str, Any]) -> int:
        body = dict(payload)
        body.setdefault("msg_id", uuid.uuid4().hex)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO outbox (topic, payload) VALUES (?, ?)", (topic, json.dumps(body))
            )
            self._db.commit()
            return int(cur.lastrowid)

    def peek(self, limit: int = 100) -> list[tuple[int, str, dict[str, Any]]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, topic, payload FROM outbox ORDER BY id LIMIT ?", (limit,)
            ).fetchall()
        return [(row_id, topic, json.loads(payload)) for row_id, topic, payload in rows]

    def delete(self, row_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM outbox WHERE id = ?", (row_id,))
            self._db.commit()

    def pending(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()


class ReliablePublisher:
    """Publishes through ``transport`` with an on-disk queue behind it.

    ``transport`` is any object with ``publish(topic, payload)`` that raises on
    failure (for example ``AwsIotTransport``).
    """

    def __init__(self, transport, queue: OfflineQueue, retry_after_s: float = 10.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.transport = transport
        self.queue = queue
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._next_try = 0.0

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.queue.enqueue(topic, payload)
        self.flush()

    def flush(self, limit: int = 200) -> int:
        """Send queued messages oldest-first. Returns how many were sent."""
        if self._clock() < self._next_try:
            return 0
        sent = 0
        for row_id, topic, payload in self.queue.peek(limit):
            try:
                self.transport.publish(topic, payload)
            except Exception as exc:
                self._next_try = self._clock() + self._retry_after_s
                log.warning("cloud unreachable (%s); %d message(s) queued",
                            type(exc).__name__, self.queue.pending())
                return sent
            self.queue.delete(row_id)
            sent += 1
        if sent:
            log.info("sent %d queued message(s) to the cloud", sent)
        return sent

    def close(self) -> None:
        self._next_try = 0.0
        try:
            self.flush()
        finally:
            closer: Optional[Callable[[], None]] = getattr(self.transport, "close", None)
            if closer:
                closer()
            self.queue.close()
