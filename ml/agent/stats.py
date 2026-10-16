"""Deterministic statistics.

Design rule from the project: Python does all counting and math; the language
model only writes sentences. Small models miscounted, so nothing numeric is
left for the model to compute.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional


def numeric_summary(records: Iterable[dict[str, Any]], field: str) -> Optional[dict[str, float]]:
    """min / max / mean / latest / count of ``field``; None if no usable values."""
    values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    if not values:
        return None
    return {
        "count": len(values),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "mean": round(sum(values) / len(values), 1),
        "latest": round(values[-1], 1),
    }


def label_counts(records: Iterable[dict[str, Any]], field: str = "object") -> dict[str, int]:
    """How many times each label appears, most common first."""
    counts = Counter(r[field] for r in records if isinstance(r.get(field), str))
    return dict(counts.most_common())
