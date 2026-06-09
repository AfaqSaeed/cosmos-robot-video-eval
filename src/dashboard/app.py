"""Streamlit dashboard for Cosmos robot video evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.config import PROJECT_ROOT  # noqa: E402


METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"


def load_metric_records(metrics_dir: Path = METRICS_DIR) -> list[dict[str, Any]]:
    """Load per-video metric JSON files."""
    if not metrics_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            st.warning(f"Skipping invalid metrics JSON: {path.name}")
    return records


def summary_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the dashboard summary table."""
    rows = []
    for record in records:
        rows.append(
            {
                "video_id": record.get("video_id"),
                "overall_score": record.get("overall_score"),
                "temporal_score": record.get("temporal_score"),
                "semantic_score": record.get("semantic_score"),
                "physical_plausibility_score": record.get("physical_plausibility_score"),
                "task_usefulness_score": record.get("task_usefulness_score"),
                "verdict": record.get("verdict"),
                "failure_modes": ", ".join(record.get("failure_modes", [])),
            }
        )
    return pd.DataFrame(rows).sort_values("overall_score", ascending=False) if rows else pd.DataFrame()


def video_path_for(record: dict[str, Any]) -> Path:
    """Resolve video path from metrics metadata or generated directory."""
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        nested = metadata.get("path") or metadata.get("output_path")
        if nested:
            path = Path(str(nested))
            if path.exists():
                return path
    sidecar = GENERATED_DIR / f"{record.get('video_id')}.mp4"
    return sidecar


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Cosmos Robot Video Evaluation", layout="wide")
    st.title("Cosmos Robot Video Evaluation")

    records = load_metric_records()
    if not records:
        st.info(
            "No metrics found. Add MP4 files to data/generated/ and run "
            "python scripts/03_evaluate_videos.py --video_dir data/generated."
        )
        videos = sorted(GENERATED_DIR.glob("*.mp4"))
        if videos:
            st.subheader("Local videos without metrics")
            for video in videos:
                st.write(video.name)
        return

    df = summary_dataframe(records)
    st.subheader("Evaluation Summary")
    st.dataframe(df, use_container_width=True, hide_index=True)

    best = max(records, key=lambda item: float(item.get("overall_score", 0.0)))
    worst = min(records, key=lambda item: float(item.get("overall_score", 0.0)))
    col_best, col_worst = st.columns(2)
    col_best.metric("Best Video", best.get("video_id", ""), f"{best.get('overall_score', 0.0):.3f}")
    col_worst.metric("Worst Video", worst.get("video_id", ""), f"{worst.get('overall_score', 0.0):.3f}")

    selected_id = st.selectbox("Select video", [record["video_id"] for record in records])
    selected = next(record for record in records if record["video_id"] == selected_id)

    left, right = st.columns([1.15, 1.0])
    with left:
        path = video_path_for(selected)
        if path.exists():
            st.video(str(path))
        else:
            st.error(f"Video file not found: {path}")
        st.caption(selected.get("prompt", ""))

    with right:
        score_cols = st.columns(2)
        score_cols[0].metric("Temporal", f"{selected.get('temporal_score', 0.0):.3f}")
        score_cols[1].metric("Semantic", f"{selected.get('semantic_score', 0.0):.3f}")
        score_cols[0].metric(
            "Physical", f"{selected.get('physical_plausibility_score', 0.0):.3f}"
        )
        score_cols[1].metric("Task Usefulness", f"{selected.get('task_usefulness_score', 0.0):.3f}")
        st.metric("Overall Score", f"{selected.get('overall_score', 0.0):.3f}")
        st.write(f"Final verdict: `{selected.get('verdict', 'unknown')}`")
        failures = selected.get("failure_modes", [])
        st.write("Failure modes:", ", ".join(failures) if failures else "None")

    with st.expander("Metadata", expanded=False):
        st.json(selected.get("metadata", {}))

    st.subheader("Temporal Metric Curves")
    curves = selected.get("temporal", {}).get("curves", {})
    if curves:
        curve_df = pd.DataFrame({key: pd.Series(value) for key, value in curves.items()})
        st.line_chart(curve_df)
    else:
        st.info("No temporal curves available for this video.")


if __name__ == "__main__":
    main()

