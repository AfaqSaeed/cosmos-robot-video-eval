"""Generate a Markdown report from saved metrics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reporting.make_report import generate_report  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_dir", default="data/metrics")
    parser.add_argument("--output_dir", default="data/reports")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    report_path = generate_report(args.metrics_dir, args.output_dir)
    logging.info("Saved report to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

