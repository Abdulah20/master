"""Audit controlled Matrice 100 ground and hover recordings.

The processed Figshare CSV files use positive current during motor operation,
despite the raw ROS BatteryState convention documented in the accompanying
README.  This module therefore preserves the processed sign and never applies
an absolute value silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FIELDS = (
    "time",
    "battery_voltage",
    "battery_current",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
)


class M100AuditError(ValueError):
    """Raised when the controlled data cannot support a reproducible audit."""


@dataclass(frozen=True)
class Sample:
    time_s: float
    voltage_V: float
    current_A: float
    altitude_m: float
    velocity_x_m_s: float
    velocity_y_m_s: float
    velocity_z_m_s: float

    @property
    def horizontal_speed_m_s(self) -> float:
        return math.hypot(self.velocity_x_m_s, self.velocity_y_m_s)

    @property
    def signed_power_W(self) -> float:
        return self.voltage_V * self.current_A


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise M100AuditError("Cannot calculate a percentile from no samples")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parameters(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return {int(row["flight"]): row for row in rows}


def _load_flight(archive: zipfile.ZipFile, flight_id: int) -> list[Sample]:
    member = f"flights/{flight_id}.csv"
    try:
        raw_stream = archive.open(member)
    except KeyError as exc:
        raise M100AuditError(f"Archive member is missing: {member}") from exc

    with raw_stream, io.TextIOWrapper(raw_stream, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise M100AuditError(f"Flight {flight_id} is missing fields: {sorted(missing)}")
        samples: list[Sample] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[field]) for field in REQUIRED_FIELDS]
            except (TypeError, ValueError) as exc:
                raise M100AuditError(
                    f"Flight {flight_id} has non-numeric data at row {row_number}"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise M100AuditError(
                    f"Flight {flight_id} has non-finite data at row {row_number}"
                )
            samples.append(Sample(*values))

    if len(samples) < 3:
        raise M100AuditError(f"Flight {flight_id} has fewer than three valid samples")
    gaps = [right.time_s - left.time_s for left, right in zip(samples, samples[1:])]
    if any(gap <= 0 for gap in gaps):
        raise M100AuditError(f"Flight {flight_id} timestamps are duplicate or non-monotonic")
    if any(sample.voltage_V <= 0 for sample in samples):
        raise M100AuditError(f"Flight {flight_id} contains non-positive voltage")
    return samples


def _integrate_energy_Wh(samples: Sequence[Sample], current_offset_A: float = 0.0) -> float:
    energy_J = 0.0
    for left, right in zip(samples, samples[1:]):
        delta_s = right.time_s - left.time_s
        left_power = left.voltage_V * (left.current_A - current_offset_A)
        right_power = right.voltage_V * (right.current_A - current_offset_A)
        energy_J += 0.5 * (left_power + right_power) * delta_s
    return energy_J / 3600.0


def _summary(
    samples: Sequence[Sample],
    current_offset_A: float = 0.0,
    integration_segments: Sequence[Sequence[Sample]] | None = None,
) -> dict[str, float | int]:
    """Summarize samples without integrating across excluded source rows.

    ``integration_segments`` is optional so continuous callers, including the
    hover-window audit, retain their existing behavior.  Ground-state callers
    pass the contiguous runs produced before flattening their selected samples.
    """
    powers = [
        sample.voltage_V * (sample.current_A - current_offset_A) for sample in samples
    ]
    energy_segments = tuple(
        integration_segments if integration_segments is not None else (samples,)
    )
    integrated_duration_s = math.fsum(
        segment[-1].time_s - segment[0].time_s
        for segment in energy_segments
        if len(segment) >= 2
    )
    selected_span_s = samples[-1].time_s - samples[0].time_s
    return {
        "sample_count": len(samples),
        "start_s": samples[0].time_s,
        "end_s": samples[-1].time_s,
        "duration_s": selected_span_s,
        "integrated_duration_s": integrated_duration_s,
        "excluded_gap_duration_s": selected_span_s - integrated_duration_s,
        "integration_segment_count": len(energy_segments),
        "mean_voltage_V": statistics.fmean(sample.voltage_V for sample in samples),
        "mean_current_A": statistics.fmean(sample.current_A for sample in samples),
        "mean_signed_power_W": statistics.fmean(powers),
        "median_signed_power_W": statistics.median(powers),
        "power_standard_deviation_W": statistics.stdev(powers),
        "integrated_signed_energy_Wh": math.fsum(
            _integrate_energy_Wh(segment, current_offset_A) for segment in energy_segments
        ),
    }


def _select_ground_segments(
    samples: Sequence[Sample], current_band_A: Sequence[float]
) -> list[list[Sample]]:
    """Return in-band samples grouped by contiguous runs in the source flight."""
    lower, upper = map(float, current_band_A)
    segments: list[list[Sample]] = []
    current_segment: list[Sample] = []
    for sample in samples:
        if lower <= sample.current_A <= upper:
            current_segment.append(sample)
        elif current_segment:
            segments.append(current_segment)
            current_segment = []
    if current_segment:
        segments.append(current_segment)
    if not segments:
        raise M100AuditError("Ground current band selected no samples")
    return segments


def _select_ground_samples(
    samples: Sequence[Sample], current_band_A: Sequence[float]
) -> list[Sample]:
    return [
        sample
        for segment in _select_ground_segments(samples, current_band_A)
        for sample in segment
    ]


def _window_motion(window: Sequence[Sample], limits: Mapping[str, float]) -> dict[str, Any]:
    median_altitude = statistics.median(sample.altitude_m for sample in window)
    metrics = {
        "horizontal_speed_95_m_s": _percentile(
            (sample.horizontal_speed_m_s for sample in window), 0.95
        ),
        "absolute_vertical_speed_95_m_s": _percentile(
            (abs(sample.velocity_z_m_s) for sample in window), 0.95
        ),
        "altitude_deviation_95_m": _percentile(
            (abs(sample.altitude_m - median_altitude) for sample in window), 0.95
        ),
    }
    ratios = [metrics[name] / float(limit) for name, limit in limits.items()]
    metrics["strict_pass"] = all(ratio <= 1.0 for ratio in ratios)
    metrics["selection_score"] = max(ratios)
    return metrics


def _least_motion_hover_window(
    samples: Sequence[Sample],
    window_config: Mapping[str, float],
    motion_limits: Mapping[str, float],
) -> tuple[list[Sample], dict[str, Any]]:
    duration_s = float(window_config["duration_s"])
    minimum_span_s = float(window_config["minimum_span_s"])
    minimum_current_A = float(window_config["minimum_current_A"])
    maximum_gap_s = float(window_config["maximum_timestamp_gap_s"])
    candidates: list[tuple[float, float, list[Sample], dict[str, Any]]] = []

    for start_index, first in enumerate(samples):
        if first.current_A <= minimum_current_A:
            continue
        window: list[Sample] = []
        for sample in samples[start_index:]:
            if sample.time_s > first.time_s + duration_s:
                break
            if sample.current_A <= minimum_current_A:
                window = []
                break
            window.append(sample)
        if not window or window[-1].time_s - window[0].time_s < minimum_span_s:
            continue
        if any(
            right.time_s - left.time_s > maximum_gap_s
            for left, right in zip(window, window[1:])
        ):
            continue
        motion = _window_motion(window, motion_limits)
        candidates.append((float(motion["selection_score"]), first.time_s, window, motion))

    if not candidates:
        raise M100AuditError("No continuous powered hover window satisfies the duration rule")
    _, _, selected, diagnostics = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, diagnostics


def run_audit(project_root: Path | str, config_path: Path | str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = json.loads(config_file.read_text(encoding="utf-8"))

    dataset = config["dataset"]
    source_provenance_raw = dataset.get("source_provenance")
    source_provenance = None
    if source_provenance_raw is not None:
        source_provenance = {
            "article_doi": str(source_provenance_raw["article_doi"]),
            "dataset_doi": str(source_provenance_raw["dataset_doi"]),
        }
    source_paths = {
        "README.txt": root / dataset["readme"],
        "parameters.csv": root / dataset["parameters"],
        "flights.zip": root / dataset["archive"],
    }
    source_audit: dict[str, Any] = {}
    for name, path in source_paths.items():
        if not path.is_file():
            raise M100AuditError(f"Required source file is missing: {path}")
        observed = _md5(path)
        expected = dataset["expected_md5"][name]
        if observed != expected:
            raise M100AuditError(f"MD5 mismatch for {name}: {observed} != {expected}")
        source_audit[name] = {"bytes": path.stat().st_size, "md5": observed, "verified": True}

    parameters = _load_parameters(source_paths["parameters.csv"])
    recordings = config["recordings"]
    ground_results: dict[str, list[dict[str, Any]]] = {route: [] for route in ("A1", "A2", "A3")}
    hover_loaded: dict[int, list[Sample]] = {}

    with zipfile.ZipFile(source_paths["flights.zip"]) as archive:
        for route in ("A1", "A2", "A3"):
            for flight_id in recordings[route]:
                if parameters.get(flight_id, {}).get("route") != route:
                    raise M100AuditError(f"Flight {flight_id} does not match expected route {route}")
                samples = _load_flight(archive, flight_id)
                segments = _select_ground_segments(
                    samples, config["ground_current_bands_A"][route]
                )
                selected = [sample for segment in segments for sample in segment]
                result = {
                    "flight": flight_id,
                    "route": route,
                    **_summary(selected, integration_segments=segments),
                }
                result["selected_fraction"] = len(selected) / len(samples)
                ground_results[route].append(result)
        for flight_id in recordings["H"]:
            if parameters.get(flight_id, {}).get("route") != "H":
                raise M100AuditError(f"Flight {flight_id} does not match expected route H")
            hover_loaded[flight_id] = _load_flight(archive, flight_id)

    a1_currents = [
        sample.current_A
        for flight_id in recordings["A1"]
        for sample in _select_ground_samples(
            _reload_from_results_source(source_paths["flights.zip"], flight_id),
            config["ground_current_bands_A"]["A1"],
        )
    ]
    zero_current_offset_A = statistics.fmean(a1_currents)

    hover_results: list[dict[str, Any]] = []
    for flight_id, samples in hover_loaded.items():
        selected, motion = _least_motion_hover_window(
            samples,
            config["hover_window"],
            config["strict_motion_diagnostics"],
        )
        raw = _summary(selected)
        corrected = _summary(selected, current_offset_A=zero_current_offset_A)
        hover_results.append(
            {
                "flight": flight_id,
                "route": "H",
                **raw,
                "zero_offset_corrected_mean_power_W": corrected["mean_signed_power_W"],
                "zero_offset_corrected_energy_Wh": corrected["integrated_signed_energy_Wh"],
                "motion_diagnostics": motion,
            }
        )

    raw_hover_powers = [float(result["mean_signed_power_W"]) for result in hover_results]
    corrected_hover_powers = [
        float(result["zero_offset_corrected_mean_power_W"]) for result in hover_results
    ]
    aggregate = {
        "a1_zero_current_offset_A": zero_current_offset_A,
        "a1_interpretation": "sensor-zero diagnostic; not identifiable auxiliary power",
        "hover_flight_count": len(hover_results),
        "hover_raw_mean_of_flight_means_W": statistics.fmean(raw_hover_powers),
        "hover_raw_between_flight_standard_deviation_W": statistics.stdev(raw_hover_powers),
        "hover_raw_between_flight_coefficient_of_variation_pct": (
            100.0 * statistics.stdev(raw_hover_powers) / statistics.fmean(raw_hover_powers)
        ),
        "hover_zero_offset_corrected_mean_of_flight_means_W": statistics.fmean(
            corrected_hover_powers
        ),
        "strict_motion_pass_count": sum(
            bool(result["motion_diagnostics"]["strict_pass"]) for result in hover_results
        ),
    }

    report = {
        "schema": str(config.get("report_schema", "m100_controlled_data_audit_v1")),
        "fidelity": "measured_Matrice_100_DC_battery_side_only_not_X500_ground_truth",
        "source_audit": source_audit,
        "protocol": {
            "power_equation": "P_W = battery_voltage_V * battery_current_A",
            "current_sign": "processed CSV positive during motor operation; signed values retained",
            "window_selection": (
                "20 s least-motion powered window selected without using electrical power"
            ),
            "strict_motion_thresholds_are_diagnostic_not_silently_relaxed": True,
            "ground_energy_integration": (
                "trapezoidal integration within contiguous in-band source runs only; "
                "excluded rows are never bridged"
            ),
            "duration_s_semantics": (
                "selected sample span; integrated_duration_s is the summed duration "
                "of integration segments"
            ),
        },
        "ground_flights": ground_results,
        "hover_flights": hover_results,
        "aggregate": aggregate,
        "claim_limits": [
            "A1 cannot identify auxiliary power because its measured current has a negative zero offset",
            "All strict 20 s motion diagnostics fail; hover numbers are exploratory",
            "Four zero-payload hover flights cannot validate payload dependence",
            "Matrice 100 measurements are not X500 electrical ground truth",
        ],
    }
    if source_provenance is not None:
        report["source_provenance"] = source_provenance
    if "report_metadata" in config:
        report["report_metadata"] = dict(config["report_metadata"])
    if report["schema"] == "m100_controlled_data_audit_v2":
        processor_file = Path(__file__).resolve()
        try:
            config_display_path = str(config_file.resolve().relative_to(root))
            processor_display_path = str(processor_file.relative_to(root))
        except ValueError:
            config_display_path = str(config_file.resolve())
            processor_display_path = str(processor_file)
        report["reproducibility"] = {
            "config": {
                "path": config_display_path,
                "sha256": _sha256(config_file),
            },
            "processor": {
                "path": processor_display_path,
                "sha256": _sha256(processor_file),
            },
        }
    return report


def _reload_from_results_source(archive_path: Path, flight_id: int) -> list[Sample]:
    with zipfile.ZipFile(archive_path) as archive:
        return _load_flight(archive, flight_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("config/m100_controlled_audit_v2.json")
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_audit(arguments.project_root, arguments.config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        output = arguments.output
        if not output.is_absolute():
            output = arguments.project_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
