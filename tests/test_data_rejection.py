from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from data_processing.extract_stable_hover import DataQualityError, validate_timestamps
from energy_model.hover_energy_v1 import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "hover_energy_v1.yaml"


def test_duplicate_timestamps_rejected() -> None:
    with pytest.raises(DataQualityError, match="duplicate"):
        validate_timestamps(np.array([1, 2, 2, 3]), "test", 2.1)


def test_large_timestamp_gap_rejected() -> None:
    with pytest.raises(DataQualityError, match="gap"):
        validate_timestamps(np.array([0, 10, 20, 100]), "test", 2.1)


def test_float_actuator_outputs_is_required(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["propulsion"]["motor_command_message"] = "esc_status"
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataQualityError, match="forbidden"):
        load_config(bad_config)


def test_rotor_diameter_requires_sensitivity_label(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["propulsion"]["rotor_diameter_sensitivity_cases"][0]["label"] = "diameter"
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataQualityError, match="sensitivity label"):
        load_config(bad_config)

