"""Empirical outbound cruise correction for the verified CMU campaign subset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

import yaml

from .m100_contract import ExperimentConfig, m100_v2_energy_assessment


MODEL_ID = "M100_STABLE_CRUISE_SPEED_CMU_V1"
MODEL_STATUS = "B_M100_STABLE_CRUISE_SPEED_CMU_SAME_CAMPAIGN"
MODEL_WARNING = "STABLE_CRUISE_SPEED_EVALUATED_ON_SEPARATE_FLIGHTS_FROM_SAME_CMU_CAMPAIGN"


@dataclass(frozen=True)
class StableCruiseSpeedFit:
    reference_speed_mps: float
    offset_w: float
    slope_w_per_mps: float
    eligible_commanded_speeds_mps: tuple[float, ...]
    min_nominal_outbound_duration_s: float
    evidence_artifact: str
    evidence_sha256: str
    dataset_doi: str

    @classmethod
    def load(cls, path: Path) -> StableCruiseSpeedFit:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema") != "M100_STABLE_CRUISE_SPEED_CALIBRATION_V1"
            or data.get("model_id") != MODEL_ID
        ):
            raise ValueError("unsupported M100 stable-cruise calibration")
        result = cls(
            reference_speed_mps=float(data["reference_speed_mps"]),
            offset_w=float(data["offset_W"]),
            slope_w_per_mps=float(data["slope_W_per_mps"]),
            eligible_commanded_speeds_mps=tuple(
                float(speed) for speed in data["eligible_commanded_speeds_mps"]
            ),
            min_nominal_outbound_duration_s=float(data["min_nominal_outbound_duration_s"]),
            evidence_artifact=str(data["evidence_artifact"]),
            evidence_sha256=str(data["evidence_sha256"]),
            dataset_doi=str(data["dataset_doi"]),
        )
        if not (
            all(math.isfinite(value) for value in (result.offset_w, result.slope_w_per_mps))
            and result.eligible_commanded_speeds_mps == (4.0, 6.0, 8.0)
            and result.reference_speed_mps == 6.0
            and result.min_nominal_outbound_duration_s == 20.0
            and re.fullmatch(r"[a-f0-9]{64}", result.evidence_sha256)
            and result.evidence_artifact
            and result.dataset_doi
        ):
            raise ValueError("invalid M100 stable-cruise calibration or evidence reference")
        return result

    def adjustment_w(self, config: ExperimentConfig | None) -> float | None:
        if config is None or not m100_v2_energy_assessment(
            config, "CRUISE", measurement_active=True
        ).inside_reported_envelope:
            return None
        speed = config.cruise_speed_mps
        if (
            speed not in self.eligible_commanded_speeds_mps
            or config.route_length_m / speed < self.min_nominal_outbound_duration_s
        ):
            return None
        return self.offset_w + self.slope_w_per_mps * (speed - self.reference_speed_mps)

    def provenance(self, adjustment_w: float) -> dict[str, object]:
        return {
            **asdict(self),
            "model_id": MODEL_ID,
            "status": MODEL_STATUS,
            "planned_outbound_adjustment_w": adjustment_w,
            "field_evaluation_from_independent_campaign": False,
            "applies_to_outbound_measurement_only": True,
        }
