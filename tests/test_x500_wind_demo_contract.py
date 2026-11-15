from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "config" / "x500_first_flight_v1.yaml"
WIND_CONFIG = ROOT / "config" / "x500_wind_demo_v1.yaml"
COMPARISON_DOCS = (
    ROOT / "docs" / "x500_wind_demo_v1.md",
    ROOT / "docs" / "x500_first_flight_v1.md",
    ROOT / "docs" / "x500_scenario_inputs_v1.md",
)


def _leaf_values(value: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        leaves: dict[tuple[str, ...], Any] = {}
        for key, child in value.items():
            leaves.update(_leaf_values(child, (*path, str(key))))
        return leaves
    return {path: value}


def test_wind_demo_comparison_claim_matches_declared_inputs() -> None:
    baseline = yaml.safe_load(BASELINE_CONFIG.read_text(encoding="utf-8"))
    wind_demo = yaml.safe_load(WIND_CONFIG.read_text(encoding="utf-8"))
    baseline_leaves = _leaf_values(baseline)
    wind_leaves = _leaf_values(wind_demo)

    identity_paths = {
        ("scenario", "mission_name"),
        ("scenario", "run_id"),
    }
    wind_path = ("operating_conditions", "wind", "velocity_enu_m_s")
    experimental_differences = {
        path: (baseline_leaves.get(path), wind_leaves.get(path))
        for path in baseline_leaves.keys() | wind_leaves.keys()
        if path not in identity_paths and baseline_leaves.get(path) != wind_leaves.get(path)
    }
    non_wind_differences = {
        path: values for path, values in experimental_differences.items() if path != wind_path
    }

    assert experimental_differences[wind_path] == ([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    assert non_wind_differences == {("mission", "hover_duration_s"): (5.0, 4.0)}

    contract_text = "\n".join(
        [WIND_CONFIG.read_text(encoding="utf-8")]
        + [path.read_text(encoding="utf-8") for path in COMPARISON_DOCS]
    ).casefold()
    forbidden_isolation_claims = (
        "wind is the only changed experimental factor",
        "wind is the only changed factor",
        "wind is the only changed variable",
        "this recorded run is a single-factor wind experiment",
        "this is a single-factor wind experiment",
    )

    if non_wind_differences:
        assert "not a single-factor" in contract_text
        assert "not formally isolated" in contract_text or "non-isolated" in contract_text
        assert not any(claim in contract_text for claim in forbidden_isolation_claims)
