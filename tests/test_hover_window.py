from __future__ import annotations

from pathlib import Path

import pytest

from data_processing.extract_stable_hover import extract_stable_hover
from energy_model.hover_energy_v1 import analyze_hover, load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "hover_energy_v1.yaml"
ULOG = ROOT / "data/restored/hover_energy_v1/00_55_32.ulg"


@pytest.mark.skipif(not ULOG.is_file(), reason="Approved local integration ULog is unavailable")
def test_approved_ulog_hover_window() -> None:
    data = extract_stable_hover(ULOG, load_config(CONFIG))
    assert data.duration_s == pytest.approx(14.956)
    assert (data.posctl_start_us - data.log_start_us) / 1e6 == pytest.approx(47.060)
    assert (data.auto_land_start_us - data.log_start_us) / 1e6 == pytest.approx(82.016)
    assert data.quality["accepted"] is True


@pytest.mark.skipif(not ULOG.is_file(), reason="Approved local integration ULog is unavailable")
def test_approved_ulog_expected_physics_results() -> None:
    report = analyze_hover(ULOG, CONFIG)
    assert report["thrust"]["required_hover_thrust_N"] == pytest.approx(20.2302154, abs=1e-6)
    assert report["thrust"]["mean_command_based_thrust_N"] == pytest.approx(20.2276, abs=0.01)
    assert report["thrust"]["thrust_balance_error_pct"] == pytest.approx(-0.013, abs=0.06)
    assert len(report["sensitivity_results"]) == 2
    for result in report["sensitivity_results"]:
        assert result["fidelity"] == "ideal_induced_power_energy_lower_bound"
        assert 110.0 < result["mean_ideal_induced_power_lower_bound_W"] < 120.0
        assert 0.45 < result["ideal_induced_energy_lower_bound_Wh"] < 0.50
    serialized = str(report)
    assert "esc_status.esc_rpm" not in serialized


@pytest.mark.skipif(not ULOG.is_file(), reason="Approved local integration ULog is unavailable")
def test_electrical_quantities_are_not_generated() -> None:
    report = analyze_hover(ULOG, CONFIG)
    assert "electrical_power_W" not in report
    assert "battery_energy_Wh" not in report
    assert "measured_electrical_energy_Wh" not in report
