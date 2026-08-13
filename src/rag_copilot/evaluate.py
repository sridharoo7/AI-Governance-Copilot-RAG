"""Portable benchmark runner that emits CI-friendly metrics and root-cause diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from .settings import rag_config, settings


class JudgeQuotaExhausted(RuntimeError):
    """Raised when a daily judge quota is exhausted and waiting cannot make progress today."""


def validate_benchmark(dataset_path: Path) -> list[dict]:
    """Validates frozen-source benchmark integrity before an expensive model evaluation run."""

    cases = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line]
    if len(cases) != 200 or len({case["question"] for case in cases}) != 200:
        raise ValueError(f"Expected exactly 200 evaluation cases, got {len(cases)}.")
    categories = {case["category"] for case in cases}
    expected = {"direct_control", "multi_document_synthesis", "scope_boundary", "adversarial_abstention"}
    if categories != expected:
        raise ValueError(f"Benchmark categories must be {expected}.")
    if sum(bool(case.get("evidence_quotes") or case.get("evidence_quote")) for case in cases) != 180:
        raise ValueError("Exactly 180 answerable cases must contain frozen-source evidence quotes.")
    for case in cases:
        if case["expected_abstention"]:
            continue
        titles = case.get("source_titles", [])
        if not titles or any(title not in case["question"] for title in titles):
            raise ValueError(
                f"Answerable case {case['id']} must name each canonical source title in its question."
            )
    return cases


def classify_failures(records: list[dict]) -> dict[str, int]:
    """Groups supplied failures by the RAG stage so a failed CI run is actionable."""

    # Each online evaluator record may attach the stage recorded by tracing. Unknown failures
    # remain visible rather than being silently assigned to a convenient category.
    known = {"ingestion", "chunking", "retrieval", "reranking", "citation_validation", "generation", "provider", "judge_config"}
    counts = {stage: 0 for stage in sorted(known)}
    for record in records:
        stage = record.get("failure_stage", "judge_config")
        counts[stage if stage in known else "judge_config"] += bool(record.get("failed", False))
    return counts


async def _score_with_backoff(scorer, **score_inputs: object) -> float:
    """Retries temporary quota failures while respecting a provider-supplied retry delay."""

    for attempt in range(8):
        try:
            result = await scorer.ascore(**score_inputs)
            return float(result.value)
        except Exception as error:
            if "GenerateRequestsPerDay" in str(error):
                raise JudgeQuotaExhausted(
                    "The evaluator judge's daily request quota is exhausted; resume after quota reset or switch judges."
                ) from error
            quota_error = (
                getattr(error, "code", None) == 429
                or "RESOURCE_EXHAUSTED" in str(error)
                or "429" in str(error)
            )
            if not quota_error or attempt == 7:
                raise
            # Gemini includes a retry-in-N-seconds hint on quota failures. Honor it rather
            # than repeatedly consuming the same exhausted free-tier quota window.
            await asyncio.sleep(_retry_delay_seconds(error, attempt))
    raise AssertionError("Unreachable retry state.")


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Extracts a quota retry delay from provider text, with a safe exponential fallback."""

    match = re.search(r"retry(?:delay| in)?[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", str(error), re.IGNORECASE)
    return max(float(match.group(1)) + 3 if match else 0, float(2**attempt))


def _load_checkpoint(path: Path | None) -> dict[str, dict[str, float]]:
    """Loads only compatible per-case judge scores so an interrupted evaluation can resume."""

    empty = {"faithfulness": {}, "answer_correctness": {}}
    if path is None or not path.exists():
        return empty
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or any(not isinstance(stored.get(key), dict) for key in empty):
        raise ValueError("Ragas checkpoint has an unsupported format.")
    return {key: {str(case_id): float(score) for case_id, score in stored[key].items()} for key in empty}


def _write_checkpoint(path: Path | None, checkpoint: dict[str, dict[str, float]]) -> None:
    """Persists completed individual judge scores immediately after each successful call."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def _score_records(
    scorer, records: list[dict], score_name: str, checkpoint: dict[str, dict[str, float]],
    checkpoint_path: Path | None, score_delay_seconds: float, input_factory
) -> list[float]:
    """Scores one metric sequentially with quota pacing and durable per-case recovery points."""

    scores = checkpoint[score_name]
    for position, row in enumerate(records, start=1):
        if row["id"] in scores:
            continue
        scores[row["id"]] = await _score_with_backoff(scorer, **input_factory(row))
        _write_checkpoint(checkpoint_path, checkpoint)
        print(json.dumps({"metric": score_name, "completed": position, "total": len(records)}), flush=True)
        if score_delay_seconds and position < len(records):
            # Faithfulness itself can make more than one judge call. This outer pacing keeps
            # the aggregate call rate safely below Gemini's 15-request/minute free-tier cap.
            await asyncio.sleep(score_delay_seconds)
    return [scores[row["id"]] for row in records]


async def ragas_faithfulness(
    responses_path: Path, cases: list[dict], checkpoint_path: Path | None = None, score_delay_seconds: float = 12.0
) -> dict:
    """Scores real answer/context records with Ragas; no placeholder quality numbers are produced."""

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerCorrectness, Faithfulness

    records = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line]
    expected_questions = {case["question"] for case in cases}
    observed_questions = {record["question"] for record in records}
    if len(records) != 200 or observed_questions != expected_questions:
        raise ValueError("Responses must contain exactly one real result for every frozen benchmark question.")
    runtime = settings()
    if runtime.ragas_judge_provider != "gemini_openai_compatible" or not runtime.google_api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY and RAGAS_JUDGE_PROVIDER=gemini_openai_compatible "
            "for the configured Gemini release judge."
        )
    # Telemetry is unrelated to evaluation and can block an air-gapped or restricted CI
    # runner; disable it before Ragas initializes its LLM factory.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    # Gemini exposes an official OpenAI-compatible endpoint. It retains Gemini as the
    # evaluator while avoiding a native google-genai HTTP-client issue reproduced on macOS.
    client = AsyncOpenAI(
        api_key=runtime.google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    scorer = Faithfulness(
        llm=llm_factory(
            runtime.ragas_judge_model,
            provider="openai",
            client=client,
            adapter="instructor",
            temperature=0,
            max_tokens=rag_config()["evaluation"]["judge_max_tokens"],
        )
    )
    correctness_scorer = AnswerCorrectness(llm=scorer.llm, weights=[1.0, 0.0])
    if score_delay_seconds < 0:
        raise ValueError("score_delay_seconds must be zero or greater.")
    checkpoint = _load_checkpoint(checkpoint_path)
    # Faithfulness judges only answer-bearing responses. Expected abstentions deliberately
    # contain no factual claims, so scoring them as 0 would make a 0.95 faithfulness gate
    # mathematically unreachable for a benchmark that correctly includes refusal tests.
    answered_records = [row for row in records if not row["actual_abstention"]]
    faithfulness_scores = await _score_records(
        scorer, answered_records, "faithfulness", checkpoint, checkpoint_path, score_delay_seconds,
        lambda row: {"user_input": row["question"], "response": row["answer"], "retrieved_contexts": row["retrieved_contexts"]},
    )
    # Correctness is measured on answerable questions. False abstentions receive a
    # deterministic zero; expected abstentions are evaluated by abstention F1 instead.
    answerable_records = [row for row in records if not row["expected_abstention"]]
    answerable_answered_records = [row for row in answerable_records if not row["actual_abstention"]]
    await _score_records(
        correctness_scorer, answerable_answered_records, "answer_correctness", checkpoint, checkpoint_path, score_delay_seconds,
        lambda row: {"user_input": row["question"], "response": row["answer"], "reference": row["reference_answer"]},
    )
    faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
    citation_validity = sum(float(row["citation_valid"]) for row in records) / len(records)
    correctness = sum(checkpoint["answer_correctness"].get(row["id"], 0.0) for row in answerable_records) / len(answerable_records)
    true_positive = sum(row["actual_abstention"] and row["expected_abstention"] for row in records)
    false_positive = sum(row["actual_abstention"] and not row["expected_abstention"] for row in records)
    false_negative = sum(not row["actual_abstention"] and row["expected_abstention"] for row in records)
    abstention_f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if true_positive else 0.0
    return {"case_count": len(records), "faithfulness": faithfulness, "citation_validity": citation_validity,
            "answer_correctness": correctness, "abstention_f1": abstention_f1,
            "diagnostics": {"status": "ragas", "responses": str(responses_path),
                            "failure_stages": classify_failures(records),
                            "faithfulness_case_count": len(answered_records),
                            "answer_correctness_case_count": len(answerable_records),
                            "judge": {"provider": runtime.ragas_judge_provider, "model": runtime.ragas_judge_model}}}


def quality_gate(metrics: dict, baseline: dict | None = None) -> None:
    """Fails the process when any governed metric is below the reviewed threshold."""

    thresholds = rag_config()["quality_gate"]
    for metric, threshold_key in (("faithfulness", "faithfulness_min"), ("citation_validity", "citation_validity_min"),
                                  ("answer_correctness", "answer_correctness_min"), ("abstention_f1", "abstention_f1_min")):
        if metrics[metric] < thresholds[threshold_key]:
            raise SystemExit(f"Quality gate failed: {metric}={metrics[metric]} < {thresholds[threshold_key]}")
        # A passing absolute score can still conceal a damaging regression from the approved run.
        if baseline and metric in baseline and baseline[metric] - metrics[metric] > thresholds["max_regression"]:
            raise SystemExit(f"Quality gate failed: {metric} regressed by more than {thresholds['max_regression']}.")


def main() -> None:
    """Runs the portable evaluator and prints a JSON result suitable for CI artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--responses", type=Path, help="Real query outputs with retrieved contexts for Ragas scoring.")
    parser.add_argument("--baseline", type=Path, help="Approved metrics JSON used to detect quality regressions.")
    parser.add_argument("--write-baseline", type=Path, help="Writes a baseline only after a successful real evaluation.")
    parser.add_argument("--report", type=Path, help="Always writes real metrics before applying the quality gate.")
    parser.add_argument("--checkpoint", type=Path, help="Per-case Ragas score checkpoint used to resume after quota waits.")
    parser.add_argument(
        "--score-delay-seconds",
        type=float,
        default=rag_config()["evaluation"]["free_tier_score_delay_seconds"],
        help="Pace judge metric calls for free-tier quotas.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = validate_benchmark(args.dataset)
    if args.validate_only:
        print(json.dumps({"case_count": len(cases), "status": "benchmark-valid"}, indent=2)); return
    if not args.responses:
        raise SystemExit("A real --responses file is required for a quality gate; fixture scores are prohibited.")
    baseline = json.loads(args.baseline.read_text()) if args.baseline and args.baseline.exists() else None
    metrics = asyncio.run(ragas_faithfulness(args.responses, cases, args.checkpoint, args.score_delay_seconds))
    if args.report:
        # Persist evidence-backed metrics even when the gate intentionally fails, so the
        # engineering review has actionable diagnostics instead of only a process exit code.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics, indent=2) + "\n")
    quality_gate(metrics, baseline)
    if args.write_baseline:
        # This explicit operation ensures a baseline is never created from fixtures or failures.
        args.write_baseline.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
