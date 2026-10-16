"""Record formats for vision results (pure, no camera or model needed).

Object detections and face matches are stored in the vision database as
separate record types and publish to separate MQTT topics. Structured
results are published, never images.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

FACE_TOPIC = "sensor/face-recognition"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detection_records(
    detections: Iterable[tuple[str, float]], min_score: float = 0.5
) -> list[dict[str, Any]]:
    """Turn (label, score) pairs into storable records, dropping weak ones."""
    return [
        {"sensor": "vision", "object": label, "confidence": round(float(score), 3),
         "timestamp": _now()}
        for label, score in detections
        if score >= min_score
    ]


def face_record(identity: str, similarity: float) -> dict[str, Any]:
    return {"sensor": "face", "identity": identity, "similarity": round(float(similarity), 3),
            "timestamp": _now()}
