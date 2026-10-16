"""The combined four-line system summary: Distance, Temperature, Vision, Faces.

Python builds the facts (all counts and ranges). The model only rewrites them
as sentences. The model's text is then *checked*: it must be exactly four lines
in order, and every number in it must appear in the facts. If the check fails,
or the model is unreachable, the plain facts are used instead, so the cloud
never receives a summary with wrong numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ml.agent.stats import label_counts, numeric_summary

LABELS = ("Distance", "Temperature", "Vision", "Faces")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

SYSTEM_PROMPT = (
    "You rewrite sensor facts as short plain-English lines. Use only the numbers "
    "and names given. Never add, remove or change a number."
)


def collect_stats(stores: dict[str, Any], n: int = 10) -> dict[str, Any]:
    """Statistics over the newest ``n`` readings of each source (Python only).

    The vision database holds both object and face records; they are told apart
    by their fields (``object`` vs ``identity``).
    """
    vision = stores["vision"].latest(n * 4)
    objects = [r for r in vision if "object" in r][-n:]
    faces = [r for r in vision if "identity" in r][-n:]
    return {
        "distance_cm": numeric_summary(stores["distance"].latest(n), "distance_cm"),
        "temperature_c": numeric_summary(stores["temperature"].latest(n), "temperature_c"),
        "object_counts": label_counts(objects, "object"),
        "face_counts": label_counts(faces, "identity"),
    }


def _range_line(label: str, s: Optional[dict[str, float]], unit: str) -> str:
    if not s:
        return f"{label}: no data"
    return f"{label}: {s['min']:.1f} to {s['max']:.1f} {unit} (latest {s['latest']:.1f})"


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} x{count}" for name, count in counts.items())


def fact_lines(stats: dict[str, Any]) -> list[str]:
    """The four deterministic lines (also the fallback summary)."""
    return [
        _range_line("Distance", stats["distance_cm"], "cm"),
        _range_line("Temperature", stats["temperature_c"], "C"),
        "Vision: " + (_counts(stats["object_counts"]) or "nothing detected"),
        "Faces: " + (_counts(stats["face_counts"]) or "none seen"),
    ]


def numbers_in(text: str) -> set[str]:
    """Numbers exactly as written ('7.0' and '7' are different on purpose).

    Comparing as floats would let a wrong count of 7 hide behind a reading of
    7.0. Decimals only match decimals, whole numbers only match counts.
    """
    return set(_NUMBER.findall(text))


_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _clean(line: str) -> str:
    """Drop a leading bullet or list number the model may add."""
    return _BULLET.sub("", line).strip()


def validate_lines(lines: list[str], facts: list[str]) -> tuple[bool, str]:
    """Check a model-written summary against the facts. Returns (ok, reason)."""
    if len(lines) != 4:
        return False, f"expected 4 lines, got {len(lines)}"
    for line, label in zip(lines, LABELS):
        if not line.lower().startswith(label.lower() + ":"):
            return False, f"line does not start with '{label}:'"
    extra = numbers_in("\n".join(lines)) - numbers_in("\n".join(facts))
    if extra:
        return False, f"numbers not in the facts: {sorted(extra)}"
    # (the model must also keep every number, or it could drop the data)
    missing = numbers_in("\n".join(facts)) - numbers_in("\n".join(lines))
    if missing:
        return False, f"numbers left out: {sorted(missing)}"
    return True, "ok"


@dataclass(frozen=True)
class SummaryResult:
    lines: list[str]
    source: str  # "slm" or "fallback"
    reason: str = "ok"


def build_messages(facts: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Rewrite these four facts as four short plain-English lines, one per "
                "fact, in the same order. Keep the label ('Distance:', 'Temperature:', "
                "'Vision:', 'Faces:') at the start of each line. Output only the four "
                "lines.\n\n" + "\n".join(facts)
            ),
        },
    ]


def summarize(llm, stats: dict[str, Any]) -> SummaryResult:
    facts = fact_lines(stats)
    try:
        reply = llm.complete(build_messages(facts), [])
    except Exception as exc:  # server down, timeout, ...
        return SummaryResult(facts, "fallback", f"model unavailable: {type(exc).__name__}")
    lines = [_clean(line) for line in reply.content.splitlines() if line.strip()]
    ok, reason = validate_lines(lines, facts)
    return SummaryResult(lines, "slm", "ok") if ok else SummaryResult(facts, "fallback", reason)
