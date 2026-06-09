from __future__ import annotations

import numpy as np

from evaluation.temporal import evaluate_frames, mean_absolute_frame_differences


def test_mean_absolute_frame_difference_on_synthetic_frames() -> None:
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 25, dtype=np.uint8),
        np.full((8, 8, 3), 50, dtype=np.uint8),
    ]

    differences = mean_absolute_frame_differences(frames)

    assert len(differences) == 2
    assert differences[0] == differences[1]
    assert 0.09 < differences[0] < 0.10


def test_temporal_score_is_high_for_static_frames() -> None:
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]

    metrics = evaluate_frames(frames)

    assert metrics["temporal_score"] == 1.0
    assert metrics["mean_frame_difference"] == 0.0

