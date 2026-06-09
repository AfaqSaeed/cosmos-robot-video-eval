"""Extract frames from local MP4 files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.frame_extractor import extract_frames  # noqa: E402
from preprocessing.video_io import list_videos  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", default="data/generated")
    parser.add_argument("--frames_dir", default="data/frames")
    parser.add_argument("--every_n_frames", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    videos = list_videos(args.video_dir)
    if not videos:
        logging.warning("No videos found in %s", args.video_dir)
        return 0
    failures = 0
    for video in videos:
        try:
            extract_frames(video, args.frames_dir, args.every_n_frames)
        except Exception as exc:
            failures += 1
            logging.error("Failed to extract frames from %s: %s", video, exc)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

