"""Evaluate generated or local sample videos."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.aggregate import aggregate_metrics, save_metrics, save_summary_csv  # noqa: E402
from evaluation.physical_reasoning import PhysicalReasoningEvaluator  # noqa: E402
from evaluation.semantic import SemanticEvaluator  # noqa: E402
from evaluation.task_usefulness import TaskUsefulnessEvaluator  # noqa: E402
from evaluation.temporal import evaluate_video_temporal  # noqa: E402
from preprocessing.video_io import list_videos, load_sidecar_metadata, probe_video  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", default="data/generated")
    parser.add_argument("--metrics_dir", default="data/metrics")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    videos = list_videos(args.video_dir)
    if not videos:
        logging.warning("No videos found in %s", args.video_dir)
        save_summary_csv([], Path(args.metrics_dir) / "summary.csv")
        return 0

    temporal_cfg: dict[str, Any] = config.get("temporal", {})
    task_cfg: dict[str, Any] = config.get("task_usefulness", {})
    semantic = SemanticEvaluator()
    physical = PhysicalReasoningEvaluator(
        jump_threshold=float(temporal_cfg.get("jump_threshold", 0.35)),
        flicker_threshold=float(temporal_cfg.get("flicker_threshold", 0.18)),
        flow_instability_threshold=float(temporal_cfg.get("flow_instability_threshold", 0.30)),
    )
    task = TaskUsefulnessEvaluator(
        min_frames=int(task_cfg.get("min_frames", 16)),
        min_semantic_consistency=float(config.get("semantic", {}).get("min_consistency", 0.55)),
    )

    records: list[dict[str, Any]] = []
    failures = 0
    for video in videos:
        try:
            video_metadata = probe_video(video).to_dict()
            sidecar = load_sidecar_metadata(video)
            prompt = str(sidecar.get("prompt", ""))
            metadata = {**sidecar, **video_metadata}
            temporal_metrics = evaluate_video_temporal(video)
            semantic_metrics = semantic.evaluate(video, prompt)
            physical_metrics = physical.evaluate(temporal_metrics)
            task_metrics = task.evaluate(video_metadata, temporal_metrics, semantic_metrics)
            record = aggregate_metrics(
                video.stem,
                metadata,
                temporal_metrics,
                semantic_metrics,
                physical_metrics,
                task_metrics,
            )
            save_metrics(record, args.metrics_dir)
            records.append(record)
            logging.info("Evaluated %s overall_score=%.3f", video.name, record["overall_score"])
        except Exception as exc:
            failures += 1
            logging.error("Failed to evaluate %s: %s", video, exc)

    save_summary_csv(records, Path(args.metrics_dir) / "summary.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

