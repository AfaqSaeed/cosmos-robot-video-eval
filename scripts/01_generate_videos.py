"""Generate robotics videos through NVIDIA NIM/Cosmos APIs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generation.cosmos_generator import CosmosGenerator  # noqa: E402
from generation.nim_client import NIMConfigurationError  # noqa: E402
from utils.logging import configure_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prompts_robotics.yaml")
    parser.add_argument("--output_dir", default="data/generated")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    generator = CosmosGenerator(args.config, args.output_dir)
    try:
        generator.generate_all(dry_run=args.dry_run)
    except NIMConfigurationError as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

