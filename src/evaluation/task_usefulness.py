"""Proxy task-usefulness scoring for robotics data generation."""

from __future__ import annotations

from typing import Any


class TaskUsefulnessEvaluator:
    """Estimate whether a video is useful enough for downstream robotics workflows."""

    def __init__(
        self,
        min_frames: int = 16,
        min_semantic_consistency: float = 0.55,
        min_temporal_score: float = 0.45,
    ) -> None:
        self.min_frames = min_frames
        self.min_semantic_consistency = min_semantic_consistency
        self.min_temporal_score = min_temporal_score

    def evaluate(
        self,
        video_metadata: dict[str, Any],
        temporal_metrics: dict[str, Any],
        semantic_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Return proxy usefulness score and verdict."""
        checks = {
            "video_readable": bool(video_metadata.get("readable", False)),
            "enough_frames": int(video_metadata.get("frame_count", 0)) >= self.min_frames,
            "motion_not_too_unstable": float(temporal_metrics.get("temporal_score", 0.0))
            >= self.min_temporal_score,
        }
        metric_status = str(semantic_metrics.get("metric_status", "unknown"))
        if metric_status == "available":
            checks["semantic_consistency_above_threshold"] = (
                float(semantic_metrics.get("semantic_score", 0.0))
                >= self.min_semantic_consistency
            )
        else:
            checks["semantic_consistency_above_threshold"] = True

        score = sum(1.0 for value in checks.values() if value) / len(checks)
        if score >= 0.85:
            verdict = "usable"
        elif score >= 0.5:
            verdict = "partially_usable"
        else:
            verdict = "not_usable"
        return {
            "task_usefulness_score": float(score),
            "verdict": verdict,
            "checks": checks,
        }

