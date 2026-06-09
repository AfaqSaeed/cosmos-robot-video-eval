"""Frame extraction utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from preprocessing.video_io import VideoMetadata, probe_video

LOGGER = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    output_root: str | Path,
    every_n_frames: int = 1,
) -> VideoMetadata:
    """Extract frames from a video under output_root/<video_id>/."""
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")

    metadata = probe_video(video_path)
    frame_dir = Path(output_root) / metadata.video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video is corrupted or unreadable: {video_path}")

    saved = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every_n_frames == 0:
            frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved += 1
        frame_index += 1
    capture.release()

    metadata_path = frame_dir / "metadata.json"
    payload = metadata.to_dict()
    payload["saved_frames"] = saved
    payload["every_n_frames"] = every_n_frames
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Extracted %s frames from %s", saved, video_path)
    return metadata

