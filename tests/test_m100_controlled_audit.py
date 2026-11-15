from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data_processing.audit_m100_controlled import (
    Sample,
    _select_ground_samples,
    _select_ground_segments,
    _summary,
    run_audit,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "m100_controlled_audit_v2.json"
ARCHIVE = ROOT / "data" / "external" / "m100" / "flights.zip"
PUBLISHED_RESULT = ROOT / "results" / "logs" / "m100_controlled_audit_v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="Verified M100 archive is unavailable")
def test_source_and_sign_are_audited() -> None:
    report = run_audit(ROOT, CONFIG)
    assert all(item["verified"] for item in report["source_audit"].values())
    assert report["source_provenance"] == {
        "article_doi": "10.1038/s41597-021-00930-x",
        "dataset_doi": "10.1184/R1/12683453",
    }
    assert report["schema"] == "m100_controlled_data_audit_v2"
    assert report["report_metadata"]["supersedes_sha256"] == (
        "5c68050e0a365bc1a3b0963cfd25b76fb23fcba2e3a4ab4e9425e5b55a60fa34"
    )
    assert report["reproducibility"]["config"]["path"] == (
        "config/m100_controlled_audit_v2.json"
    )
    assert report["reproducibility"]["processor"]["path"] == (
        "src/data_processing/audit_m100_controlled.py"
    )
    for artifact in report["reproducibility"].values():
        assert _sha256(ROOT / artifact["path"]) == artifact["sha256"]
    assert report["aggregate"]["a1_zero_current_offset_A"] == pytest.approx(-0.15, abs=0.03)
    assert "signed values retained" in report["protocol"]["current_sign"]


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="Verified M100 archive is unavailable")
def test_controlled_ground_states_are_separated() -> None:
    report = run_audit(ROOT, CONFIG)
    means = {
        route: sum(flight["mean_signed_power_W"] for flight in flights) / len(flights)
        for route, flights in report["ground_flights"].items()
    }
    assert means["A1"] < 0.0
    assert 9.0 < means["A2"] < 11.0
    assert 35.0 < means["A3"] < 38.0


def test_ground_energy_does_not_bridge_filtered_samples() -> None:
    samples = [
        Sample(float(second), 10.0, current, 0.0, 0.0, 0.0, 0.0)
        for second, current in enumerate((0.5, 0.5, 5.0, 0.5))
    ]
    segments = _select_ground_segments(samples, (0.2, 1.0))
    selected = _select_ground_samples(samples, (0.2, 1.0))

    summary = _summary(selected, integration_segments=segments)

    assert [len(segment) for segment in segments] == [2, 1]
    assert summary["mean_signed_power_W"] == pytest.approx(5.0)
    assert summary["integrated_signed_energy_Wh"] == pytest.approx(5.0 / 3600.0)
    assert summary["duration_s"] == pytest.approx(3.0)
    assert summary["integrated_duration_s"] == pytest.approx(1.0)
    assert summary["excluded_gap_duration_s"] == pytest.approx(2.0)
    assert summary["integration_segment_count"] == 2


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="Verified M100 archive is unavailable")
def test_ground_energy_regression_for_discontinuous_flights() -> None:
    report = run_audit(ROOT, CONFIG)
    by_flight = {
        result["flight"]: result
        for results in report["ground_flights"].values()
        for result in results
    }

    assert by_flight[214]["mean_signed_power_W"] == pytest.approx(10.192421399610)
    assert by_flight[216]["mean_signed_power_W"] == pytest.approx(10.086061745318)
    assert by_flight[214]["integrated_signed_energy_Wh"] == pytest.approx(
        0.343769571621
    )
    assert by_flight[216]["integrated_signed_energy_Wh"] == pytest.approx(
        0.138604242250
    )
    assert by_flight[214]["integrated_duration_s"] == pytest.approx(121.32)
    assert by_flight[214]["excluded_gap_duration_s"] == pytest.approx(1.89)
    assert by_flight[214]["integration_segment_count"] == 4
    assert by_flight[216]["integrated_duration_s"] == pytest.approx(49.40)
    assert by_flight[216]["excluded_gap_duration_s"] == pytest.approx(2.09)
    assert by_flight[216]["integration_segment_count"] == 3


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="Verified M100 archive is unavailable")
def test_hover_result_is_exploratory_and_flight_level() -> None:
    report = run_audit(ROOT, CONFIG)
    assert len(report["hover_flights"]) == 4
    assert [
        flight["integrated_signed_energy_Wh"] for flight in report["hover_flights"]
    ] == pytest.approx(
        [
            2.5567982959674502,
            2.713504984335412,
            2.6983648184572653,
            2.665629004639882,
        ]
    )
    assert report["aggregate"]["strict_motion_pass_count"] == 0
    assert 470.0 < report["aggregate"]["hover_raw_mean_of_flight_means_W"] < 490.0
    assert report["aggregate"]["hover_raw_between_flight_coefficient_of_variation_pct"] < 5.0
    assert "not_X500_ground_truth" in report["fidelity"]


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="Verified M100 archive is unavailable")
def test_published_v2_result_matches_the_processor() -> None:
    published = json.loads(PUBLISHED_RESULT.read_text(encoding="utf-8"))
    regenerated = run_audit(ROOT, CONFIG)
    assert published == regenerated

    legacy = ROOT / regenerated["report_metadata"]["supersedes_artifact"]
    assert _sha256(legacy) == regenerated["report_metadata"]["supersedes_sha256"]
