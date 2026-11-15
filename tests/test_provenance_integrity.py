from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from data_processing.build_x500_flight_csv import (
    PROJECT_ROOT,
    FlightDataError,
    _scenario_provenance,
    _source_tree_revision,
)
from simulation.scenario import build_gazebo_scenario, load_scenario_config


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_legacy_hover_result_is_preserved_and_exactly_reproducible() -> None:
    provenance = _load(ROOT / "results/logs/hover_energy_v1_provenance.json")
    artifact = provenance["artifact"]
    report_path = ROOT / artifact["path"]
    report = _load(report_path)

    assert _sha256(report_path) == artifact["sha256"]
    assert provenance["status"] == "legacy_result_source_restored_and_verified"
    assert provenance["exact_reproduction"] == "verified_byte_identical_from_restored_source"
    assert provenance["scientific_result_recomputed"] is True
    assert provenance["replacement_data_used"] is False
    assert provenance["source_tree_revision"] == {
        "status": "unavailable_empty_git_directory",
        "git_metadata_path": ".git",
        "commit": None,
    }
    assert report["input"]["ulog_sha256"] == provenance["source_input"]["expected_sha256"]
    assert report["input"]["ulog_path"] == provenance["source_input"]["recorded_path"]
    current_source = provenance["current_reproduction_source"]
    source_path = ROOT / current_source["path"]
    assert source_path.is_file()
    assert source_path.stat().st_size == current_source["size_bytes"]
    assert _sha256(source_path) == current_source["sha256"]
    assert current_source["sha256"] == provenance["source_input"][
        "expected_sha256"
    ]
    recorded_path = Path(provenance["source_input"]["recorded_path"])
    if recorded_path.is_file():
        assert _sha256(recorded_path) == current_source["sha256"]
    assert provenance["source_input"]["recorded_path_is_volatile"] is True
    restore_script = provenance["restoration"]["restore_script"]
    assert _sha256(ROOT / restore_script) == provenance["restoration"][
        "restore_script_sha256"
    ]
    verification = provenance["reproduction_verification"]
    assert verification["byte_identical"] is True
    assert verification["structured_field_differences"] == 0
    assert verification["temporary_output_sha256"] == artifact["sha256"]
    assert verification["preserved_output_sha256"] == artifact["sha256"]

    config = provenance["configuration_reference"]
    assert _sha256(ROOT / config["current_path"]) == config["current_sha256"]
    assert "did not record a config hash" in config["relationship_to_legacy_run"]
    assert "byte-for-byte" in config["relationship_to_legacy_run"]


@pytest.mark.parametrize(
    "run_id",
    ["x500_first_flight_v1_baseline", "x500_wind_demo_v1_3ms_east"],
)
def test_legacy_run_supplemental_manifest_covers_previously_unhashed_assets(
    run_id: str,
) -> None:
    integrity = _load(ROOT / "results/runs" / run_id / "integrity_manifest.json")
    assert integrity["scientific_result_recomputed"] is False
    assert integrity["historical_manifests_rewritten"] is False
    expected_artifacts = {
        "run_manifest",
        "validation",
        "scenario_manifest",
        "scenario_model_config",
        "scenario_model_sdf",
        "scenario_world_sdf",
    }
    if run_id == "x500_wind_demo_v1_3ms_east":
        expected_artifacts.update(
            {
                "visualization_gazebo_wind_demo",
                "visualization_gazebo_wind_indicator",
            }
        )
    assert set(integrity["artifacts"]) == expected_artifacts
    for artifact in integrity["artifacts"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]

    revision = integrity["source_tree_revision"]
    assert revision["status"] == "unavailable_empty_git_directory"
    assert revision["commit"] is None
    git_metadata = ROOT / revision["git_metadata_path"]
    assert git_metadata.is_dir()
    assert not any(git_metadata.iterdir())

    historical = integrity["historical_config"]
    assert _sha256(ROOT / historical["path"]) == historical["current_sha256"]
    run_manifest = _load(ROOT / "results/runs" / run_id / "run_manifest.json")
    assert run_manifest["sources"]["config"]["sha256"] == historical["recorded_sha256"]


def test_every_legacy_wind_png_is_covered_by_integrity_manifest() -> None:
    run = ROOT / "results/runs/x500_wind_demo_v1_3ms_east"
    integrity = _load(run / "integrity_manifest.json")
    recorded_pngs = {
        ROOT / artifact["path"]
        for artifact in integrity["artifacts"].values()
        if artifact["path"].endswith(".png")
    }
    assert recorded_pngs == set(run.glob("*.png"))


def test_energy_plugin_provenance_path_does_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = ROOT / "build/gazebo_energy_plugin/libMotorBatteryLoadSystem.so"
    monkeypatch.chdir(tmp_path)
    assert PROJECT_ROOT == ROOT
    assert (PROJECT_ROOT / "build/gazebo_energy_plugin/libMotorBatteryLoadSystem.so") == expected
    assert expected.is_file()


def test_empty_git_metadata_is_reported_without_inventing_revision() -> None:
    assert _source_tree_revision() == {
        "status": "unavailable_empty_git_directory",
        "git_metadata_path": str(ROOT / ".git"),
        "commit": None,
    }


def test_current_wind_config_keeps_historical_scenario_semantics() -> None:
    run_id = "x500_wind_demo_v1_3ms_east"
    inputs = load_scenario_config(ROOT / "config/x500_wind_demo_v1.yaml")
    scenario = _load(ROOT / "build/scenarios" / run_id / "scenario_manifest.json")

    assert inputs.run_id == scenario["run_id"]
    assert inputs.fidelity == scenario["fidelity"]
    assert inputs.total_vehicle_mass_kg == pytest.approx(scenario["total_vehicle_mass_kg"])
    assert asdict(inputs.mission) == scenario["mission"]
    assert json.loads(json.dumps(asdict(inputs.payload))) == scenario["payload"]
    assert list(inputs.wind.velocity_enu_m_s) == pytest.approx(
        scenario["wind"]["velocity_enu_m_s"]
    )
    assert json.loads(json.dumps(asdict(inputs.logging))) == scenario["logging"]
    assert inputs.provenance == scenario["provenance"]


def test_new_scenario_manifests_hash_every_generated_asset(tmp_path: Path) -> None:
    config_path = ROOT / "config/x500_first_flight_v1.yaml"
    inputs = load_scenario_config(config_path)
    output = tmp_path / "scenario"
    manifest = build_gazebo_scenario(inputs, output, config_path=config_path)

    assert set(manifest["generated_files"]) == {"model_config", "model_sdf", "world_sdf"}
    assert set(manifest["generated_file_sha256"]) == set(manifest["generated_files"])
    for label, raw_path in manifest["generated_files"].items():
        assert _sha256(Path(raw_path)) == manifest["generated_file_sha256"][label]

    provenance = _scenario_provenance(
        output / "scenario_manifest.json",
        config_path=config_path,
        expected_run_id=inputs.run_id,
    )
    assert set(provenance["scenario_generated_files"]) == {
        "model_config",
        "model_sdf",
        "world_sdf",
    }

    world = Path(manifest["generated_files"]["world_sdf"])
    world.write_text(world.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FlightDataError, match="hash mismatch: world_sdf"):
        _scenario_provenance(
            output / "scenario_manifest.json",
            config_path=config_path,
            expected_run_id=inputs.run_id,
        )
