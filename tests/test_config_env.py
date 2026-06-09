from __future__ import annotations

import os
from pathlib import Path

from utils.config import load_environment


def test_load_environment_reads_dotenv_without_overriding_existing_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "NVIDIA_API_KEY=file-key\nNVIDIA_BASE_URL=https://example.test\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "existing-key")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)

    load_environment(env_path)

    assert os.environ["NVIDIA_API_KEY"] == "existing-key"
    assert os.environ["NVIDIA_BASE_URL"] == "https://example.test"
