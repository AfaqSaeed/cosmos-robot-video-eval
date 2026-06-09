from __future__ import annotations

from evaluation.aggregate import aggregate_metrics, verdict_from_score, weighted_overall_score


def test_weighted_overall_score() -> None:
    metrics = {
        "temporal_score": 1.0,
        "semantic_score": 0.8,
        "physical_plausibility_score": 0.6,
        "task_usefulness_score": 0.4,
    }

    score = weighted_overall_score(metrics)

    assert round(score, 3) == 0.760


def test_aggregate_metrics_keeps_failure_modes_and_verdict() -> None:
    record = aggregate_metrics(
        "video_a",
        {"prompt": "robot picks object"},
        {"temporal_score": 0.9},
        {"semantic_score": 0.8},
        {"physical_plausibility_score": 0.7, "failure_modes": ["minor_flicker"]},
        {"task_usefulness_score": 0.6, "verdict": "partially_usable"},
    )

    assert record["overall_score"] > 0.7
    assert record["failure_modes"] == ["minor_flicker"]
    assert record["verdict"] == "partially_usable"


def test_verdict_from_score() -> None:
    assert verdict_from_score(0.8) == "usable"
    assert verdict_from_score(0.5) == "partially_usable"
    assert verdict_from_score(0.2) == "not_usable"

