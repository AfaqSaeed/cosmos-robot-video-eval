"""Video reading and metadata helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata for a readable video file."""

    video_id: str
    path: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    readable: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metadata."""
        return asdict(self)


def list_videos(video_dir: str | Path) -> list[Path]:
    """List supported video files in a directory."""
    directory = Path(video_dir)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
    )


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Validate a video and return metadata, raising ValueError if unreadable."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Video is corrupted or unreadable: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, _ = capture.read()
    capture.release()

    if not ok or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Video has no readable frames: {path}")

    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        video_id=path.stem,
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        width=width,
        height=height,
        readable=True,
    )


def load_sidecar_metadata(video_path: str | Path) -> dict[str, object]:
    """Load generation metadata next to a video if present."""
    sidecar = Path(video_path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))

