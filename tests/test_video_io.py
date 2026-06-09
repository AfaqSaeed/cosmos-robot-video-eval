from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from preprocessing.video_io import probe_video


def _write_dummy_video(path: Path, frame_count: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (32, 24),
    )
    for index in range(frame_count):
        frame = np.full((24, 32, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_probe_video_reads_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "dummy.mp4"
    _write_dummy_video(video_path)

    metadata = probe_video(video_path)

    assert metadata.video_id == "dummy"
    assert metadata.readable is True
    assert metadata.frame_count == 8
    assert metadata.width == 32
    assert metadata.height == 24
    assert metadata.fps > 0

