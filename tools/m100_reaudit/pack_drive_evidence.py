"""Archive the three native flights and check cruise mode against the PX4 ULogs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "tmp/m100_reaudit_pack_three"
OUTPUT = ROOT / "results/m100_reaudit/pack_drive_three"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def time_fraction(data, field, expected, start, end):
    """PX4 categorical states apply until the next message on each topic."""
    time = data["timestamp"] * 1e-6
    if not time[0] <= start < end <= time[-1]:
        raise ValueError(f"The ULog does not bracket the cruise interval for {field}")
    edges = np.r_[start, time[(time > start) & (time < end)], end]
    positions = np.searchsorted(time, edges[:-1], side="right") - 1
    return float(np.diff(edges)[data[field][positions] == expected].sum() / (end - start))


def flight_control(ulog, trace, cid):
    with gzip.open(trace, "rt", newline="") as stream:
        boot_times = [float(row["px4_time_s"]) for row in csv.DictReader(stream)
                      if row["phase"] == "CRUISE" and row["measuring"] == "1" and row["px4_time_s"]]
    if not boot_times:
        return {"case_id": cid, "cruise_observed": False}
    start, end = min(boot_times), max(boot_times)
    log = ULog(str(ulog), message_name_filter_list=["vehicle_status", "vehicle_attitude", "vehicle_land_detected"])
    status = log.get_dataset("vehicle_status").data
    state = log.get_dataset("vehicle_land_detected").data
    attitude = log.get_dataset("vehicle_attitude").data
    time = attitude["timestamp"] * 1e-6
    select = (time >= start) & (time <= end)
    q = np.array([attitude[f"q[{i}]"] for i in range(4)], dtype=float)
    q /= np.linalg.norm(q, axis=0)
    tilt_deg = np.degrees(np.arccos(np.clip(1 - 2 * (q[1]**2 + q[2]**2), -1, 1)))
    # Align the attitude, arming and land-detection streams on PX4 boot time.
    # Keep only the common bracket where both categorical states are known.
    armed_t = status["timestamp"] * 1e-6
    landed_t = state["timestamp"] * 1e-6
    edges = np.unique(np.concatenate((time, armed_t, landed_t)))
    edges = edges[(edges >= max(time[0], armed_t[0], landed_t[0])) &
                  (edges <= min(time[-1], armed_t[-1], landed_t[-1]))]
    left = edges[:-1]
    airborne = ((status["arming_state"][np.searchsorted(armed_t, left, side="right")-1] == 2) &
                (state["landed"][np.searchsorted(landed_t, left, side="right")-1] == 0))
    airborne_tilt = tilt_deg[np.searchsorted(time, left[airborne], side="right")-1]
    maximum = int(np.argmax(airborne_tilt))
    return {
        "case_id": cid, "cruise_observed": True, "px4_cruise_start_s": start, "px4_cruise_end_s": end,
        "offboard_time_fraction": time_fraction(status, "nav_state", 14, start, end),
        "armed_time_fraction": time_fraction(status, "arming_state", 2, start, end),
        "failsafe_time_fraction": time_fraction(status, "failsafe", 1, start, end),
        "airborne_time_fraction": time_fraction(state, "landed", 0, start, end),
        "cruise_median_tilt_deg": float(np.median(tilt_deg[select])), "cruise_max_tilt_deg": float(tilt_deg[select].max()),
        "airborne_peak_tilt_deg": float(airborne_tilt[maximum]), "airborne_peak_tilt_px4_s": float(left[airborne][maximum]),
        "airborne_time_tilt_over_dji_35deg_s": float(np.diff(edges)[airborne][airborne_tilt > 35].sum()),
        "ulog_initial_mpc_tiltmax_air_deg": log.initial_parameters.get("MPC_TILTMAX_AIR"),
        "ulog_dropout_events": len(log.dropouts), "ulog_final_landed": bool(state["landed"][-1]),
    }


def collect(work, out):
    protocol = json.loads((out / "protocol.json").read_text())
    expected_hash = protocol["input_sha256"]
    manifest, control = [], []
    for case in protocol["cases"]:
        cid = case["case_id"]
        raw = work / cid / "attempt_1"
        if not (raw / "stopped_utc").is_file():
            raise RuntimeError(f"{cid} is still open")
        receipt_path = raw / f"{cid}.json"
        receipt = json.loads(receipt_path.read_text())
        run = receipt["run_id"]
        provenance_path = raw / f"{run}.provenance.json"
        source = json.loads(provenance_path.read_text())
        native = source["native_pack_drive"]
        recorded = native["trace"]
        actual = source["source_files"]
        for key, label in (("m100_experiment_manager", "manager"), ("m100_canonical_base_sdf", "base_sdf"),
                           ("gazebo_m100_pack_drive_plugin", "plugin"), ("m100_energy_v2_config", "energy_config")):
            if actual[key]["sha256"] != expected_hash[label]:
                raise ValueError(f"{cid}: different acquisition input {key}")
        if (raw / "launcher_sha256").read_text().strip() != expected_hash["launcher"]:
            raise ValueError(f"{cid}: different native launcher")
        if native["cmu_development_fit_sha256"] != expected_hash["fit"]:
            raise ValueError(f"{cid}: different voltage calibration")
        model_key = "base_sdf" if case["payload_kg"] == 0 else f"payload{round(case['payload_kg'] * 100):03d}"
        if source["temperature_application"]["runtime_model_sdf_sha256"] != expected_hash[model_key]:
            raise ValueError(f"{cid}: payload SDF differs from protocol")
        pack = raw / recorded["filename"]
        telemetry = raw / f"{run}.csv"
        if recorded["io_error"] or digest(pack) != recorded["sha256"] or digest(telemetry) != source["csv"]["sha256"]:
            raise ValueError(f"Acquisition integrity error for {cid}")
        folder = out / "evidence" / cid
        folder.mkdir(exist_ok=True, parents=True)
        with (folder / "pack_drive.jsonl.gz").open("wb") as target, pack.open("rb") as infile:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                shutil.copyfileobj(infile, compressed)
        with gzip.open(folder / "pack_drive.jsonl.gz", "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != recorded["sha256"]:
                raise ValueError(f"Pack trace archive mismatch: {cid}")
        trace = raw / f"{run}.trace.csv.gz"
        for path, name in ((receipt_path, "receipt.json"), (provenance_path, "provenance.json"),
                           (telemetry, "manager_1hz.csv"), (trace, "flight_trace.csv.gz")):
            shutil.copyfile(path, folder / name)
        (ulog,) = raw.glob("state/rootfs/log/*/*.ulg")
        control.append(flight_control(ulog, trace, cid))
        manifest.append({
            "case_id": cid, "run_id": run, "outcome": receipt["outcome"], "recorded_rows": recorded["rows"],
            "started_utc": (raw / "started_utc").read_text().strip(), "stopped_utc": (raw / "stopped_utc").read_text().strip(),
            "ulog_relative_to_work": str(ulog.relative_to(work)), "ulog_in_download": f"ulog/{cid}.ulg",
            "ulog_sha256": digest(ulog),
            "files_sha256": {p.name: digest(p) for p in sorted(folder.iterdir())},
        })
    (out / "evidence/index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out / "px4_control.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for item in control for k in item)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(control)


def bundle(work, out, destination):
    manifest = json.loads((out / "evidence/index.json").read_text())
    files = [out / name for name in ("protocol.json", "summary.csv", "motion.csv", "px4_control.csv",
             "electrical.csv", "phases.csv", "cmu_context.csv", "cmu_context_flights.csv",
             "flight_profiles.png", "report_ar.md", "evidence/index.json", "data/cmu_windows.csv.gz")]
    for flight in manifest:
        cid = flight["case_id"]
        files.extend(out / "data" / f"{cid}.{kind}.csv.gz" for kind in ("trajectory", "pack"))
        for name, sha in flight["files_sha256"].items():
            path = out / "evidence" / cid / name
            if digest(path) != sha:
                raise ValueError(f"Changed recording: {path}")
            files.append(path)
    sums = {str(path.relative_to(out)): digest(path) for path in sorted(files)}
    checksums = out / "SHA256SUMS"
    checksums.write_text("".join(f"{sha}  {name}\n" for name, sha in sums.items()))
    # Include directly openable CSV and original PX4 logs in the local download.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".pack-drive-") as stage:
        temp = Path(stage) / destination.name
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
            for path in [*files, checksums]:
                method = ZIP_STORED if path.suffix in {".gz", ".png"} else ZIP_DEFLATED
                archive.write(path, str(path.relative_to(out)), compress_type=method)
                if path.parent == out / "data":
                    with gzip.open(path, "rb") as src, archive.open(f"csv/{path.stem}", "w") as dst:
                        shutil.copyfileobj(src, dst)
            for flight in manifest:
                archive.write(work / flight["ulog_relative_to_work"], flight["ulog_in_download"])
        with ZipFile(temp) as archive:
            if (bad := archive.testzip()) is not None:
                raise ValueError(f"Bad archive member: {bad}")
            for flight in manifest:
                with archive.open(flight["ulog_in_download"]) as src:
                    if hashlib.file_digest(src, "sha256").hexdigest() != flight["ulog_sha256"]:
                        raise ValueError(f"{flight['case_id']}: archived ULog digest differs")
        # Keep a verified archive if another packaging attempt is interrupted.
        temp.replace(destination)
    print(f"{destination}  {destination.stat().st_size} bytes  sha256 {digest(destination)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--bundle", type=Path, metavar="ZIP", help="Package and verify the finished release and original ULogs")
    args = parser.parse_args()
    if args.bundle:
        bundle(args.work, args.out, args.bundle)
    else:
        collect(args.work, args.out)
