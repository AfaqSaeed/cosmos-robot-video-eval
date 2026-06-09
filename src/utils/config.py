"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment(env_path: str | Path | None = None) -> None:
    """Load local .env values if python-dotenv is installed.

    Existing process environment variables take precedence, so CI secrets or
    shell exports are not overwritten by a local file.
    """
    path = Path(env_path) if env_path is not None else PROJECT_ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_fallback(path)
        return
    load_dotenv(path, override=False)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty mapping when it has no content."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return loaded


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_env_file_fallback(path: Path) -> None:
    """Minimal KEY=VALUE loader used only when python-dotenv is unavailable."""
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
