"""Rule-based physical plausibility checks for generated robot videos."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class PhysicalReasoningEvaluator:
    """Evaluate physical plausibility from temporal metric signals."""

    def __init__(
        self,
        jump_threshold: float = 0.35,
        flicker_threshold: float = 0.18,
        flow_instability_threshold: float = 0.30,
    ) -> None:
        self.jump_threshold = jump_threshold
        self.flicker_threshold = flicker_threshold
        self.flow_instability_threshold = flow_instability_threshold

    def evaluate(
        self,
        temporal_metrics: dict[str, Any],
        use_external_reasoner: bool = False,
    ) -> dict[str, Any]:
        """Return rule-based physical plausibility metrics.

        use_external_reasoner is a placeholder for future NVIDIA Cosmos
        Reasoner/VLM calls. It is intentionally not required for the MVP.
        """
        if use_external_reasoner:
            LOGGER.info("External Cosmos Reasoner/VLM evaluator is not wired in this MVP.")

        failure_modes: list[str] = []
        max_diff = float(temporal_metrics.get("max_frame_difference", 0.0))
        max_flicker = float(temporal_metrics.get("max_flicker", 0.0))
        flow_stability = float(temporal_metrics.get("optical_flow_stability", 1.0))
        mean_flow = float(temporal_metrics.get("mean_optical_flow_magnitude", 0.0))

        if max_diff > self.jump_threshold:
            failure_modes.append("excessive_frame_jumps")
        if max_flicker > self.flicker_threshold:
            failure_modes.append("extreme_flicker")
        if flow_stability < self.flow_instability_threshold:
            failure_modes.append("unstable_optical_flow")
        if max_diff > self.jump_threshold and mean_flow < 0.5:
            failure_modes.append("likely_camera_discontinuity")

        penalty = 0.18 * len(failure_modes)
        penalty += max(0.0, max_diff - self.jump_threshold)
        penalty += max(0.0, max_flicker - self.flicker_threshold)
        penalty += max(0.0, self.flow_instability_threshold - flow_stability)
        score = max(0.0, min(1.0, 1.0 - penalty))

        if failure_modes:
            explanation = "Rule-based checks found temporal artifacts that reduce physical plausibility."
        else:
            explanation = "Rule-based checks did not find major temporal or camera discontinuity artifacts."

        return {
            "physical_plausibility_score": float(score),
            "failure_modes": failure_modes,
            "explanation": explanation,
            "external_reasoner_status": "placeholder_not_used",
        }

