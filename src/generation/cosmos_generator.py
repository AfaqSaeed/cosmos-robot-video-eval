"""Cosmos video generation orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generation.nim_client import NIMClient
from utils.config import PROJECT_ROOT, ensure_dir, load_yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptSpec:
    """Prompt entry loaded from the robotics prompt config."""

    id: str
    prompt: str
    seed: int | None = None


class CosmosGenerator:
    """Generate videos from prompt YAML using an NVIDIA NIM client."""

    def __init__(
        self,
        config_path: str | Path,
        output_dir: str | Path = PROJECT_ROOT / "data" / "generated",
        client: NIMClient | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)
        self.output_dir = ensure_dir(output_dir)
        self.client = client

    def load_prompts(self) -> list[PromptSpec]:
        """Return prompt specs from the configured YAML file."""
        prompts = self.config.get("prompts", [])
        if not isinstance(prompts, list):
            raise ValueError("prompts_robotics.yaml must contain a list under 'prompts'")
        specs: list[PromptSpec] = []
        for item in prompts:
            if not isinstance(item, dict) or not item.get("id") or not item.get("prompt"):
                raise ValueError(f"Invalid prompt entry: {item!r}")
            specs.append(
                PromptSpec(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    seed=item.get("seed"),
                )
            )
        return specs

    def build_payload(self, prompt: PromptSpec) -> dict[str, Any]:
        """Build the request payload for one prompt."""
        defaults = self.config.get("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}
        payload: dict[str, Any] = dict(defaults)
        payload["prompt"] = prompt.prompt
        if prompt.seed is not None:
            payload["seed"] = prompt.seed
        return payload

    def generate_all(self, dry_run: bool = False) -> list[dict[str, Any]]:
        """Generate one video per prompt, or print payloads in dry-run mode."""
        endpoint = str(self.config.get("endpoint", "/v1/video/generations"))
        model_name = str(self.config.get("model", "cosmos-video"))
        results: list[dict[str, Any]] = []
        for prompt in self.load_prompts():
            payload = self.build_payload(prompt)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            video_name = f"{timestamp}_{prompt.id}.mp4"
            video_path = self.output_dir / video_name
            metadata_path = video_path.with_suffix(".json")
            common_metadata: dict[str, Any] = {
                "video_id": video_path.stem,
                "prompt_id": prompt.id,
                "prompt": prompt.prompt,
                "model_name": model_name,
                "timestamp": timestamp,
                "endpoint_name": endpoint,
                "seed": payload.get("seed"),
                "output_path": str(video_path),
            }
            if dry_run:
                dry_run_record = {**common_metadata, "payload": payload, "dry_run": True}
                print(json.dumps(dry_run_record, indent=2))
                results.append(dry_run_record)
                continue
            if self.client is None:
                self.client = NIMClient()
            response_metadata = self.client.generate_video(endpoint, payload, video_path)
            metadata = {
                **common_metadata,
                "payload": payload,
                "api_response_metadata": response_metadata,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            LOGGER.info("Saved generated video metadata to %s", metadata_path)
            results.append(metadata)
        return results

