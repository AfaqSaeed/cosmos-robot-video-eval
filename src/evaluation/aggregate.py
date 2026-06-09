"""Metric aggregation and persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


WEIGHTS = {
    "temporal_score": 0.35,
    "semantic_score": 0.25,
    "physical_plausibility_score": 0.25,
    "task_usefulness_score": 0.15,
}


def weighted_overall_score(metrics: dict[str, Any]) -> float:
    """Compute the weighted 0-1 overall score."""
    return float(
        sum(float(metrics.get(metric_name, 0.0)) * weight for metric_name, weight in WEIGHTS.items())
    )


def aggregate_metrics(
    video_id: str,
    metadata: dict[str, Any],
    temporal: dict[str, Any],
    semantic: dict[str, Any],
    physical: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Combine metric groups into one serializable record."""
    flat_scores = {
        "temporal_score": float(temporal.get("temporal_score", 0.0)),
        "semantic_score": float(semantic.get("semantic_score", 0.0)),
        "physical_plausibility_score": float(physical.get("physical_plausibility_score", 0.0)),
        "task_usefulness_score": float(task.get("task_usefulness_score", 0.0)),
    }
    overall_score = weighted_overall_score(flat_scores)
    return {
        "video_id": video_id,
        "prompt": metadata.get("prompt", ""),
        "metadata": metadata,
        "temporal": temporal,
        "semantic": semantic,
        "physical": physical,
        "task_usefulness": task,
        **flat_scores,
        "overall_score": overall_score,
        "failure_modes": physical.get("failure_modes", []),
        "verdict": task.get("verdict", verdict_from_score(overall_score)),
    }


def verdict_from_score(score: float) -> str:
    """Map a score to a coarse final verdict."""
    if score >= 0.75:
        return "usable"
    if score >= 0.45:
        return "partially_usable"
    return "not_usable"


def save_metrics(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """Save one metrics JSON file under output_dir."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{metrics['video_id']}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_summary_csv(metrics_records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Save a CSV summary across videos."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(record) for record in metrics_records]
    if rows:
        pd.DataFrame(rows).sort_values("overall_score", ascending=False).to_csv(
            destination, index=False, quoting=csv.QUOTE_MINIMAL
        )
    else:
        pd.DataFrame(
            columns=[
                "video_id",
                "overall_score",
                "temporal_score",
                "semantic_score",
                "physical_plausibility_score",
                "task_usefulness_score",
                "verdict",
                "failure_modes",
                "prompt",
            ]
        ).to_csv(destination, index=False)
    return destination


def summary_row(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one metrics record for table display."""
    failure_modes = record.get("failure_modes", [])
    return {
        "video_id": record.get("video_id", ""),
        "overall_score": record.get("overall_score", 0.0),
        "temporal_score": record.get("temporal_score", 0.0),
        "semantic_score": record.get("semantic_score", 0.0),
        "physical_plausibility_score": record.get("physical_plausibility_score", 0.0),
        "task_usefulness_score": record.get("task_usefulness_score", 0.0),
        "verdict": record.get("verdict", ""),
        "failure_modes": ";".join(failure_modes) if isinstance(failure_modes, list) else str(failure_modes),
        "prompt": record.get("prompt", ""),
    }

