"""Markdown report generation for evaluation runs."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import PROJECT_ROOT


RESEARCH_QUESTION = (
    "Are world-model-generated robot videos temporally stable, semantically consistent, "
    "physically plausible, and useful enough for downstream robotics data generation?"
)


def load_metrics(metrics_dir: str | Path = PROJECT_ROOT / "data" / "metrics") -> list[dict[str, Any]]:
    """Load metric JSON records."""
    directory = Path(metrics_dir)
    if not directory.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def generate_report(
    metrics_dir: str | Path = PROJECT_ROOT / "data" / "metrics",
    output_dir: str | Path = PROJECT_ROOT / "data" / "reports",
) -> Path:
    """Generate a Markdown research report and return its path."""
    records = load_metrics(metrics_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output_dir) / f"cosmos_robot_video_eval_{timestamp}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(records, key=lambda item: float(item.get("overall_score", 0.0)), reverse=True)
    rows = [
        {
            "video_id": record.get("video_id", ""),
            "overall_score": round(float(record.get("overall_score", 0.0)), 3),
            "verdict": record.get("verdict", ""),
            "failure_modes": ", ".join(record.get("failure_modes", [])) or "None",
        }
        for record in ranked
    ]
    table = pd.DataFrame(rows).to_markdown(index=False) if rows else "No videos were evaluated."
    failure_counter = Counter(
        failure
        for record in records
        for failure in record.get("failure_modes", [])
    )
    common_failures = (
        "\n".join(f"- {failure}: {count}" for failure, count in failure_counter.most_common())
        if failure_counter
        else "- No recurring failure modes detected."
    )
    best = ranked[:3]
    worst = list(reversed(ranked[-3:])) if ranked else []

    report = f"""# Cosmos Robot Video Evaluation

## Research Question

{RESEARCH_QUESTION}

## Run Summary

- Videos evaluated: {len(records)}
- Report timestamp: {timestamp}

## Ranking Table

{table}

## Best Examples

{_example_list(best)}

## Worst Examples

{_example_list(worst)}

## Common Failure Modes

{common_failures}

## Conclusion

This MVP evaluates generated robot videos across temporal stability, semantic alignment,
physical plausibility, and downstream task usefulness. The overall ranking should be read
as a research triage signal: high-scoring clips are better candidates for robotics data
generation, while low-scoring clips expose failure modes that need model, prompt, or
post-processing improvements.
"""
    destination.write_text(report, encoding="utf-8")
    return destination


def _example_list(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No examples available."
    lines = []
    for record in records:
        lines.append(
            f"- {record.get('video_id', '')}: overall={float(record.get('overall_score', 0.0)):.3f}, "
            f"verdict={record.get('verdict', '')}"
        )
    return "\n".join(lines)

