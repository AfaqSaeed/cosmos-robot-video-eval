"""Temporal stability metrics for generated videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_video_frames(video_path: str | Path, max_frames: int | None = None) -> list[np.ndarray]:
    """Load frames from a video as RGB uint8 arrays."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video is corrupted or unreadable: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    if not frames:
        raise ValueError(f"Video has no readable frames: {video_path}")
    return frames


def mean_absolute_frame_differences(frames: list[np.ndarray]) -> list[float]:
    """Return normalized mean absolute RGB differences for adjacent frames."""
    if len(frames) < 2:
        return []
    differences: list[float] = []
    for current, following in zip(frames, frames[1:]):
        diff = np.mean(np.abs(current.astype(np.float32) - following.astype(np.float32))) / 255.0
        differences.append(float(diff))
    return differences


def flicker_scores(frame_differences: list[float]) -> list[float]:
    """Return sudden-change scores based on acceleration in frame differences."""
    if len(frame_differences) < 2:
        return []
    return [
        float(abs(frame_differences[index] - frame_differences[index - 1]))
        for index in range(1, len(frame_differences))
    ]


def optical_flow_magnitudes(frames: list[np.ndarray]) -> list[float]:
    """Compute mean Farneback optical-flow magnitude for adjacent frames."""
    if len(frames) < 2:
        return []
    magnitudes: list[float] = []
    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(magnitude)))
        previous_gray = gray
    return magnitudes


def temporal_smoothness_score(
    frame_differences: list[float],
    flicker: list[float],
    flow_magnitudes: list[float],
) -> float:
    """Combine temporal curves into a normalized 0-1 smoothness score."""
    if not frame_differences:
        return 0.0
    mean_diff = float(np.mean(frame_differences))
    mean_flicker = float(np.mean(flicker)) if flicker else 0.0
    flow_cv = _coefficient_of_variation(flow_magnitudes)
    penalty = 1.8 * mean_diff + 2.5 * mean_flicker + 0.4 * flow_cv
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


def evaluate_frames(frames: list[np.ndarray]) -> dict[str, Any]:
    """Evaluate temporal stability for in-memory frames."""
    frame_diffs = mean_absolute_frame_differences(frames)
    flicker = flicker_scores(frame_diffs)
    flow = optical_flow_magnitudes(frames)
    score = temporal_smoothness_score(frame_diffs, flicker, flow)
    return {
        "temporal_score": score,
        "mean_frame_difference": float(np.mean(frame_diffs)) if frame_diffs else 0.0,
        "max_frame_difference": float(np.max(frame_diffs)) if frame_diffs else 0.0,
        "mean_flicker": float(np.mean(flicker)) if flicker else 0.0,
        "max_flicker": float(np.max(flicker)) if flicker else 0.0,
        "mean_optical_flow_magnitude": float(np.mean(flow)) if flow else 0.0,
        "optical_flow_stability": float(1.0 - np.clip(_coefficient_of_variation(flow), 0.0, 1.0)),
        "curves": {
            "frame_difference": frame_diffs,
            "flicker": flicker,
            "optical_flow_magnitude": flow,
        },
    }


def evaluate_video_temporal(
    video_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a video and optionally save temporal curves and summary JSON."""
    frames = load_video_frames(video_path)
    metrics = evaluate_frames(frames)
    metrics["video_id"] = Path(video_path).stem
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        curve_path = destination / f"{Path(video_path).stem}_temporal.json"
        curve_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metrics["temporal_metrics_path"] = str(curve_path)
    return metrics


def _coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = float(np.mean(values))
    if mean_value <= 1e-8:
        return 0.0
    return float(np.std(values) / mean_value)

