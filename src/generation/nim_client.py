"""NVIDIA NIM/Cosmos API client for video generation."""

from __future__ import annotations

import base64
import copy
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from utils.config import load_environment

LOGGER = logging.getLogger(__name__)


class NIMClientError(RuntimeError):
    """Base exception raised by the NIM client."""


class NIMConfigurationError(NIMClientError):
    """Raised when required NIM configuration is missing."""


class NIMResponseError(NIMClientError):
    """Raised when an API response is invalid or unsuccessful."""


class NIMClient:
    """Small JSON POST client for NVIDIA NIM/Cosmos endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        load_environment()
        self.api_key = api_key if api_key is not None else os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url if base_url is not None else os.getenv("NVIDIA_BASE_URL")
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise NIMConfigurationError(
                "NVIDIA_API_KEY is not set. Video generation requires an API key, "
                "but local MP4 files in data/generated/ can still be extracted, "
                "evaluated, reported, and viewed in the dashboard."
            )
        if not self.base_url:
            raise NIMConfigurationError(
                "NVIDIA_BASE_URL is not set. Set it to your NVIDIA NIM/Cosmos base URL."
            )

    def post_json(self, endpoint_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request and return response metadata plus parsed JSON."""
        url = self._url_for(endpoint_path)
        started = time.perf_counter()
        LOGGER.info("Calling NVIDIA endpoint %s", endpoint_path)
        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise NIMClientError(f"NVIDIA request failed: {exc}") from exc

        elapsed_seconds = time.perf_counter() - started
        if response.status_code >= 400:
            message = response.text[:1000]
            raise NIMResponseError(
                f"NVIDIA endpoint returned HTTP {response.status_code}: {message}"
            )

        try:
            response_json = response.json()
        except ValueError as exc:
            raise NIMResponseError("NVIDIA endpoint did not return valid JSON") from exc

        return {
            "request": {
                "url": url,
                "endpoint_path": endpoint_path,
                "payload": payload,
            },
            "response": {
                "status_code": response.status_code,
                "elapsed_seconds": elapsed_seconds,
                "headers": {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() in {"date", "request-id", "x-request-id", "content-type"}
                },
                "json": self._redact_video_payload(response_json),
            },
            "raw_response_json": response_json,
        }

    def generate_video(
        self,
        endpoint_path: str,
        payload: dict[str, Any],
        output_path: str | Path,
    ) -> dict[str, Any]:
        """Call a video endpoint, decode the returned base64 MP4, and save it."""
        metadata = self.post_json(endpoint_path, payload)
        video_b64 = self._find_base64_video(metadata["raw_response_json"])
        if not video_b64:
            raise NIMResponseError(
                "Could not find a base64 MP4 field in the response. Expected one of "
                "video, video_base64, mp4, data.video, data.video_base64, "
                "or artifacts[0].base64."
            )

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.write_bytes(base64.b64decode(video_b64))
        except (ValueError, OSError) as exc:
            raise NIMResponseError(f"Failed to decode or save returned MP4: {exc}") from exc

        metadata.pop("raw_response_json", None)
        metadata["output_path"] = str(destination)
        metadata["output_bytes"] = destination.stat().st_size
        return metadata

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url_for(self, endpoint_path: str) -> str:
        base = str(self.base_url).rstrip("/")
        path = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        return f"{base}{path}"

    @staticmethod
    def _find_base64_video(payload: dict[str, Any]) -> str | None:
        candidates: list[Any] = [
            payload.get("video"),
            payload.get("video_base64"),
            payload.get("b64_video"),
            payload.get("mp4"),
        ]
        data = payload.get("data")
        if isinstance(data, dict):
            candidates.extend(
                [
                    data.get("video"),
                    data.get("video_base64"),
                    data.get("b64_video"),
                    data.get("mp4"),
                ]
            )
        artifacts = payload.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if isinstance(artifact, dict):
                    candidates.extend(
                        [
                            artifact.get("base64"),
                            artifact.get("video"),
                            artifact.get("video_base64"),
                            artifact.get("b64_video"),
                        ]
                    )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                if "," in candidate and candidate.lstrip().startswith("data:"):
                    return candidate.split(",", 1)[1]
                return candidate
        return None

    @classmethod
    def _redact_video_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        redacted = copy.deepcopy(payload)

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key in list(value.keys()):
                    if key in {"video", "video_base64", "b64_video", "mp4", "base64"} and isinstance(
                        value[key], str
                    ):
                        value[key] = f"<base64 redacted: {len(value[key])} chars>"
                    else:
                        walk(value[key])
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(redacted)
        return redacted
