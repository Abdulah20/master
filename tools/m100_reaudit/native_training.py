"""Package labels written by the M100 recorder without recomputing flight energy."""
import csv
import hashlib
import io
import math
from pathlib import Path
import tempfile


PLANNED_FEATURES = (
    "payload_kg", "commanded_speed_mps", "commanded_altitude_agl_m",
    "planned_outbound_measurement_m", "planned_hover_duration_s", "commanded_wind_mps",
    "commanded_wind_direction_deg", "ambient_reference_C",
)
TARGET = "estimated_airborne_energy_Wh"
TRAINING_COLUMNS = ("case_id", "run_id", "label_source", *PLANNED_FEATURES, TARGET)


def _write_atomic(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(output.name)
    try:
        with output:
            output.write(contents)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def training_example(contents, receipt, case_id, run_id, outcome):
    if hashlib.sha256(contents).hexdigest() != receipt["sha256"]:
        raise ValueError(f"{run_id}: airborne training row differs from recording receipt")
    rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8"), newline="")))
    if len(rows) != 1 or rows[0]["run_id"] != run_id:
        raise ValueError(f"{run_id}: expected one matching airborne training row")
    row = rows[0]
    eligible = receipt["training_eligible"]
    if (type(eligible) is not bool or row["training_eligible"].lower() != str(eligible).lower()
            or row["label_source"] != receipt["label_source"] or row["training_status"] != receipt["status"]):
        raise ValueError(f"{run_id}: inconsistent training status or source")
    energy = float(row[TARGET]) if row[TARGET] else None
    official = receipt[TARGET]
    if ((energy is None) != (official is None) or
            energy is not None and (not math.isfinite(energy) or energy <= 0 or
                                    not math.isclose(energy, official, rel_tol=1e-12))):
        raise ValueError(f"{run_id}: invalid or inconsistent airborne energy")
    if not eligible or outcome != "COMPLETE":
        return None
    if energy is None or row["mission_outcome"] != "COMPLETE":
        raise ValueError(f"{run_id}: eligible training row lacks a completed flight and target")
    return {"case_id": case_id, **{name: row[name] for name in TRAINING_COLUMNS[1:]}}


def archive_row(directory, out, case_id, run_id, provenance, outcome):
    receipt = provenance.get("airborne_training")
    if receipt is None:  # Legacy recordings predate direct airborne labels.
        return {}, None
    if receipt["relative_path"] != f"training/{run_id}.csv":
        raise ValueError(f"{run_id}: unexpected airborne training row filename")
    contents = (directory / receipt["relative_path"]).read_bytes()
    sample = training_example(contents, receipt, case_id, run_id, outcome)
    archive = out / "airborne_rows" / f"{case_id}-{run_id}.csv"
    _write_atomic(archive, contents)
    return {"airborne_row_path": str(archive.relative_to(out)), "airborne_row_sha256": receipt["sha256"]}, sample


def aggregate_bytes(examples):
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=TRAINING_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(examples, key=lambda row: (row["case_id"], row["run_id"])))
    return table.getvalue().encode("utf-8")


def publish_aggregate(out, examples, has_receipts):
    path = out / "training_airborne.csv"
    if has_receipts or path.exists():
        _write_atomic(path, aggregate_bytes(examples))
