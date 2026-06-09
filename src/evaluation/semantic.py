"""Semantic consistency metrics with an optional OpenCLIP backend."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class SemanticEvaluator:
    """Evaluate prompt-frame alignment and frame embedding drift."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.backend_available = False
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore

            self._torch = torch
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self._model.eval()
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self.backend_available = True
        except Exception as exc:
            LOGGER.info("open_clip unavailable; semantic metrics will use deterministic mocks: %s", exc)

    def evaluate(
        self,
        video_path: str | Path,
        prompt: str,
        frames: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Return semantic consistency metrics for a video."""
        if not self.backend_available:
            return self._mock_metrics(video_path, prompt)
        if frames is None:
            frames = self._sample_frames(video_path)
        return self._open_clip_metrics(frames, prompt)

    def _open_clip_metrics(self, frames: list[np.ndarray], prompt: str) -> dict[str, Any]:
        from PIL import Image

        torch = self._torch
        if not frames:
            return {
                "semantic_score": 0.0,
                "metric_status": "available",
                "prompt_to_frame_similarity": 0.0,
                "embedding_drift": 1.0,
                "frame_similarities": [],
            }

        with torch.no_grad():
            text_tokens = self._tokenizer([prompt])
            text_features = self._model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_tensors = [
                self._preprocess(Image.fromarray(frame)).unsqueeze(0) for frame in frames
            ]
            images = torch.cat(image_tensors, dim=0)
            image_features = self._model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(1).cpu().numpy()
            first = image_features[0:1]
            drifts = (1.0 - (image_features @ first.T).squeeze(1)).cpu().numpy()

        prompt_similarity = float(np.mean(similarities))
        embedding_drift = float(np.mean(drifts))
        semantic_score = float(np.clip(0.5 + 0.5 * prompt_similarity - 0.25 * embedding_drift, 0.0, 1.0))
        return {
            "semantic_score": semantic_score,
            "metric_status": "available",
            "prompt_to_frame_similarity": prompt_similarity,
            "embedding_drift": embedding_drift,
            "frame_similarities": [float(value) for value in similarities],
        }

    @staticmethod
    def _sample_frames(video_path: str | Path, max_frames: int = 8) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Video is corrupted or unreadable: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = set(np.linspace(0, max(frame_count - 1, 0), num=min(max_frames, max(frame_count, 1)), dtype=int))
        frames: list[np.ndarray] = []
        index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if index in indices:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            index += 1
        capture.release()
        return frames

    @staticmethod
    def _mock_metrics(video_path: str | Path, prompt: str) -> dict[str, Any]:
        digest = hashlib.sha256(f"{Path(video_path).stem}:{prompt}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        semantic_score = 0.58 + 0.18 * value
        drift = 0.18 + 0.12 * (1.0 - value)
        return {
            "semantic_score": float(semantic_score),
            "metric_status": "unavailable_mock",
            "prompt_to_frame_similarity": float(semantic_score),
            "embedding_drift": float(drift),
            "frame_similarities": [],
        }

