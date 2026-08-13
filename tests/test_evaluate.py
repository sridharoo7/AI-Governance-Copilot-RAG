"""Verifies the benchmark contract used by local and CI evaluation commands."""

from pathlib import Path

from rag_copilot.evaluate import validate_benchmark


def test_benchmark_contains_exactly_200_cases() -> None:
    """The golden benchmark cannot silently shrink without failing tests."""

    result = validate_benchmark(Path("data/evaluation/benchmark_200.jsonl"))
    assert len(result) == 200
    assert len({case["question"] for case in result}) == 200
