from ml.agent.llm import LLMReply, ScriptedLLM
from ml.benchmark_slm import FIXTURE_STATS, percentile, run_benchmark
from ml.summary import (
    LABELS,
    collect_stats,
    fact_lines,
    numbers_in,
    summarize,
    validate_lines,
)

STATS = FIXTURE_STATS
FACTS = fact_lines(STATS)


class ListStore:
    def __init__(self, records):
        self._r = records

    def latest(self, n=10):
        return self._r[-n:]


GOOD = "\n".join(
    [
        "Distance: between 6.6 to 7.7 cm, latest 7.0.",
        "Temperature: 28.1 to 30.9 C, latest 30.2.",
        "Vision: a person x6, bird x2, bear x1.",
        "Faces: Person1 x4 and Unknown x2.",
    ]
)


def test_fact_lines_are_four_and_correct():
    assert len(FACTS) == 4
    assert FACTS[0] == "Distance: 6.6 to 7.7 cm (latest 7.0)"
    assert FACTS[2] == "Vision: person x6, bird x2, bear x1"
    assert FACTS[3] == "Faces: Person1 x4, Unknown x2"


def test_fact_lines_handle_no_data():
    empty = {"distance_cm": None, "temperature_c": None, "object_counts": {}, "face_counts": {}}
    assert fact_lines(empty) == [
        "Distance: no data",
        "Temperature: no data",
        "Vision: nothing detected",
        "Faces: none seen",
    ]


def test_collect_stats_separates_objects_and_faces():
    vision = [
        {"object": "person"}, {"identity": "Person1"}, {"object": "person"},
        {"identity": "Unknown"}, {"identity": "Person1"}, {"object": "bird"},
    ]
    stores = {
        "distance": ListStore([{"distance_cm": 5.0}, {"distance_cm": 7.0}]),
        "temperature": ListStore([]),
        "vision": ListStore(vision),
    }
    stats = collect_stats(stores, n=10)
    assert stats["object_counts"] == {"person": 2, "bird": 1}
    assert stats["face_counts"] == {"Person1": 2, "Unknown": 1}
    assert stats["distance_cm"]["mean"] == 6.0 and stats["temperature_c"] is None


def test_validate_accepts_good_summary():
    assert validate_lines(GOOD.splitlines(), FACTS) == (True, "ok")


def test_validate_rejects_wrong_line_count_and_labels():
    assert not validate_lines(GOOD.splitlines()[:3], FACTS)[0]
    swapped = GOOD.splitlines()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert not validate_lines(swapped, FACTS)[0]


def test_validate_rejects_invented_numbers():
    bad = GOOD.replace("x6", "x7").splitlines()  # model miscounted
    ok, reason = validate_lines(bad, FACTS)
    # a count of 7 must not hide behind the reading 7.0
    assert not ok and "'7'" in reason


def test_validate_rejects_dropped_numbers():
    short = GOOD.replace(", latest 7.0", "").splitlines()
    ok, reason = validate_lines(short, FACTS)
    assert not ok and "left out" in reason


def test_numbers_in_keeps_decimals_and_counts_distinct():
    assert numbers_in("latest 7.0, x3 and 30") == {"7.0", "3", "30"}


def test_summarize_uses_valid_model_output():
    result = summarize(ScriptedLLM([LLMReply(GOOD)]), STATS)
    assert result.source == "slm" and len(result.lines) == 4


def test_summarize_strips_bullets_the_model_adds():
    bulleted = "\n".join(f"- {line}" for line in GOOD.splitlines())
    assert summarize(ScriptedLLM([LLMReply(bulleted)]), STATS).source == "slm"


def test_summarize_falls_back_on_wrong_numbers():
    result = summarize(ScriptedLLM([LLMReply(GOOD.replace("x6", "x9"))]), STATS)
    assert result.source == "fallback" and result.lines == FACTS


def test_summarize_falls_back_when_model_unreachable():
    class Down:
        def complete(self, messages, tools):
            raise ConnectionError("no server")

    result = summarize(Down(), STATS)
    assert result.source == "fallback" and result.lines == FACTS
    assert "ConnectionError" in result.reason


def test_labels_constant():
    assert LABELS == ("Distance", "Temperature", "Vision", "Faces")


# --- benchmark ---------------------------------------------------------
def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0 and percentile(values, 1.0) == 5.0
    assert percentile(values, 0.5) == 3.0


def test_benchmark_counts_valid_and_fallback_runs():
    llm = ScriptedLLM([LLMReply(GOOD), LLMReply("nonsense"), LLMReply(GOOD), LLMReply(GOOD)])
    ticks = iter(range(0, 100, 2))  # each run takes 2 "seconds"
    report = run_benchmark(llm, runs=4, clock=lambda: next(ticks))
    assert report["runs"] == 4 and report["valid_rate"] == 0.75
    assert report["median_s"] == 2.0 and report["max_s"] == 2.0
    assert sum(report["fallback_reasons"].values()) == 1
