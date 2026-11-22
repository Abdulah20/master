"""Assemble the three verified, recorder-written labels and plot their field comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from .native_training import publish_aggregate, training_example


DEFAULT = Path(__file__).resolve().parents[2] / "results/m100_reaudit/direct_airborne_three"
BUNDLE = "m100_direct_airborne_full.zip"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_attempts(out):
    evidence = out / "evidence"
    index = json.loads((evidence / "index.json").read_text())
    selection = json.loads((out / "selection.json").read_text())["cases"]
    selected = {c["case_id"]: (c["run_id"], c["attempt"]) for c in selection}
    rows = []
    for item in sorted(index, key=lambda x: (x["case_id"], x["attempt"])):
        folder = evidence / item["path"]
        audit = json.loads((folder / "audit.json").read_text())
        status = (item["run_id"], f"attempt_{item['attempt']}") == selected[item["case_id"]]
        if status != item["training_eligible"]:
            raise ValueError(f"Unexpected training selection for {item['run_id']}")
        rows.append({
            "case_id": item["case_id"], "attempt": item["attempt"], "run_id": item["run_id"],
            "mission_outcome": audit["mission_outcome"], "training_status": item["training_status"],
            "selected_for_training": status, "airborne_duration_s": audit["duration_s"],
            "covered_s": audit["native_covered_s"], "uncovered_s": audit["native_uncovered_s"],
            "known_subtotal_Wh": audit["native_known_Wh"],
            "estimated_airborne_energy_Wh": audit["estimated_airborne_energy_Wh"],
            "original_training_file": (folder / "training.csv").relative_to(out).as_posix(),
            "training_file_sha256": audit["direct_row_sha256"],
        })
    if len(selected) != sum(r["selected_for_training"] for r in rows):
        raise ValueError("A chosen payload is missing a verified original row")
    with (out / "attempts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def direct_labels(out, comparison):
    by_case = {r["case_id"]: r for r in comparison}
    cases = json.loads((out / "protocol.json").read_text())["cases"]
    if len(cases) != len(comparison) or {c["case_id"] for c in cases} != by_case.keys():
        raise ValueError("Comparison differs from the planned cases")
    examples = []
    for case in cases:
        cid = case["case_id"]
        directory = out / "evidence" / cid
        provenance = json.loads((directory / "provenance.json").read_text())
        receipt = json.loads((directory / "receipt.json").read_text())
        audit = json.loads((directory / "audit.json").read_text())
        if receipt["case_id"] != cid or receipt["run_id"] != by_case[cid]["direct_run_id"]:
            raise ValueError(f"{cid}: mismatched recording identity")
        original = (directory / "training.csv").read_bytes()
        example = training_example(original, provenance["airborne_training"], cid, receipt["run_id"], receipt["outcome"])
        if (example is None or audit["native_uncovered_s"] != 0
                or not audit["ulog_and_native_edges_match"] or not audit["training_eligible"]):
            raise ValueError(f"{cid}: incomplete airborne evidence for this flight")
        energy = float(example["estimated_airborne_energy_Wh"])
        if (abs(energy - audit["estimated_airborne_energy_Wh"]) > 1e-9
                or abs(energy - float(by_case[cid]["direct_energy_Wh"])) > 1e-8):
            raise ValueError(f"{cid}: reported energy differs from the recorder")
        examples.append(example)
    publish_aggregate(out, examples, has_receipts=True)
    return examples


def draw(out, rows, field):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharey=True)
    series = [
        ("Flight energy (Wh)", "direct_energy_Wh", "previous_energy_Wh", "measured_airborne_energy_Wh"),
        ("Mean airborne power (W)", "direct_mean_power_W", "previous_mean_power_W", "measured_airborne_mean_power_W"),
    ]
    for ax, (title, direct, previous, measured) in zip(axes, series):
        for j, row in enumerate(rows):
            points = [float(f[measured]) for f in field if float(f["payload_kg"]) == float(row["payload_kg"])]
            ax.hlines(j, min(points), max(points), color="#878b94", lw=2.1, zorder=1)
            ax.scatter(points, [j] * len(points), s=28, color="#626772", zorder=2)
            ax.scatter(float(row[previous]), j + .18, marker="D", s=44, facecolor="white", edgecolor="#b2581b", lw=1.6, zorder=3)
            if row[direct]:
                ax.scatter(float(row[direct]), j - .18, marker="o", s=55, color="#006a79", zorder=4)
        ax.set_title(title, fontsize=12, loc="left", pad=10)
        ax.grid(axis="x", color="#dce0e3", linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#8b9095")
        ax.tick_params(length=0, pad=6)
        ax.margins(x=.10)
    axes[0].set_yticks(range(len(rows)), [f"{float(r['payload_kg']):g} kg" for r in rows])
    axes[0].set_ylabel("Payload")
    axes[0].set_ylim(len(rows) - .60, -.4)
    legend = [
        Line2D([], [], ls="", marker="o", color="#006a79", label="New flight, recorded in Gazebo"),
        Line2D([], [], ls="", marker="D", markerfacecolor="white", markeredgecolor="#b2581b", label="Previous flight, offline estimate"),
        Line2D([], [], marker="o", color="#626772", label="Individual CMU flights"),
    ]
    fig.legend(handles=legend, loc="lower center", frameon=False, ncol=3, fontsize=8.1)
    fig.suptitle("Matrice 100 from takeoff to landing", fontsize=15, x=.11, ha="left", fontweight="bold")
    fig.subplots_adjust(left=.12, right=.97, top=.75, bottom=.21, wspace=.15)
    fig.savefig(out / "flight_comparison.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def manifest(out):
    files = [path for path in sorted(out.rglob("*")) if path.is_file()
             and path.name not in ("SHA256SUMS", ".gitattributes", ".gitignore", BUNDLE)]
    entries = []
    for path in files:
        with path.open("rb") as stream:
            entries.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path.relative_to(out)}\n")
    (out / "SHA256SUMS").write_text("".join(entries))


def bundle(out, capture):
    """Include original ULogs and high-rate manager telemetry in the local download."""
    evidence = out / "evidence"
    attempts = json.loads((evidence / "index.json").read_text())
    public = out / "SHA256SUMS"
    published = [(out / name, name, sha) for sha, name in
                 (line.split("  ", 1) for line in public.read_text().splitlines())]
    members = list(published)
    raw_checks = []
    for item in attempts:
        attempt = f"attempt_{item['attempt']}"
        local = capture / item["case_id"] / attempt
        proof = json.loads((evidence / item["path"] / "manifest.json").read_text())
        rid = item["run_id"]
        ulog = local / proof["source_ulog_relative_path"]
        telemetry = local / f"{rid}.trace.csv.gz"
        with telemetry.open("rb") as stream:
            telemetry_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        for path, suffix, sha in ((ulog, ".ulg", item["ulog_sha256"]), (telemetry, ".trace.csv.gz", telemetry_sha)):
            name = f"native_logs/{item['case_id']}/{attempt}/{rid}{suffix}"
            members.append((path, name, sha))
            raw_checks.append(f"{sha}  {name}\n")
    with TemporaryDirectory(dir=out.parent) as temporary:
        target = Path(temporary) / BUNDLE
        with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=4) as z:
            z.write(public, public.name)
            z.writestr("raw_log_SHA256SUMS", "".join(raw_checks))
            for path, name, _ in members:
                z.write(path, name)
        with ZipFile(target) as z:
            for _, name, sha in members:
                with z.open(name) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                        raise ValueError(f"Download differs from source: {name}")
        target.replace(out / BUNDLE)


def main(out):
    comparison = sorted(read_rows(out / "comparison.csv"), key=lambda row: float(row["payload_kg"]))
    write_attempts(out)
    examples = direct_labels(out, comparison)
    draw(out, comparison, read_rows(out / "cmu_matched_measurements.csv"))
    manifest(out)
    print(f"{len(examples)} directly recorded rows packaged from {len(comparison)} flights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--bundle-from", type=Path, help="Include full original logs from this acquisition directory in a local ZIP")
    args = parser.parse_args()
    main(args.out)
    if args.bundle_from:
        bundle(args.out, args.bundle_from)
