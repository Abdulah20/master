"""Run exactly the named native flights once, retaining each attempt and its training CSV."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def run(protocol_file, work, attempt, case_ids):
    protocol = json.loads(protocol_file.read_text())
    case_map = {c["case_id"]: c for c in protocol["cases"]}
    if not case_ids or len(set(case_ids)) != len(case_ids) or any(c not in case_map for c in case_ids):
        raise ValueError("Select unique cases from the acquisition protocol")
    if attempt != protocol.get("attempt", 1):
        raise ValueError("Attempt number differs from the acquisition protocol")
    if any(c not in protocol.get("planned_replacement_cases", case_map) for c in case_ids):
        raise ValueError("Selected case is outside the registered acquisition")
    project = Path(os.environ.get("M100_RUNTIME", str(Path.home() / "Master/Drone/project")))
    px4 = Path(os.environ.get("M100_PX4", str(Path.home() / "PX4-Autopilot")))
    launcher = Path(__file__).with_name("isolated_native.sh").resolve()
    files = {
        "plugin": project / "ros2_ws/install/m100_gazebo_gui/lib/m100_gazebo_gui/libM100PackDrive.so",
        "manager": project / "ros2_ws/src/drone_nav/drone_nav/m100_experiment_manager.py",
        "training_writer": project / "ros2_ws/src/drone_nav/drone_nav/m100_airborne_training.py",
        "controller": px4 / "build/px4_sitl_default/bin/px4",
        "initial_parameters": px4 / "build/px4_sitl_default/rootfs/parameters.bson",
        "launcher": launcher, "bridge": launcher.with_name("run_native.sh"), "native": launcher.with_name("native.py"),
    }
    for cid in case_ids:
        for key, path in files.items():
            with path.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != protocol["source_sha256"][key]:
                    raise ValueError(f"{cid}: {key} differs from the registered build")
        case = case_map[cid]
        folder = work.resolve() / cid
        folder.mkdir(parents=True, exist_ok=True)
        plan = folder / "plan.json"
        if not plan.exists():
            with plan.open("x") as f:
                json.dump([case], f, indent=2)
                f.write("\n")
        if json.loads(plan.read_text()) != [case]:
            raise ValueError(f"{cid}: stored flight plan differs")
        directory = folder / f"attempt_{attempt}"
        directory.mkdir()
        print(f"Launching {cid} in {directory.name}", flush=True)
        with (directory / "acquisition.log").open("x") as log:
            result = subprocess.run([str(launcher), str(plan), str(directory)],
                                    env=dict(os.environ, M100_GAZEBO_SEED=str(case["seed"])),
                                    stdout=log, stderr=subprocess.STDOUT)
        receipt_file = directory / f"{cid}.json"
        if not receipt_file.is_file():
            raise RuntimeError(f"{cid}: acquisition stopped before receipt, exit {result.returncode}")
        receipt = json.loads(receipt_file.read_text())
        print(cid, receipt["outcome"], receipt.get("run_id"), receipt.get("reason", ""), flush=True)
        if receipt["outcome"] != "COMPLETE" or not receipt["landed"] or receipt["armed"]:
            raise RuntimeError(f"{cid}: flight did not complete with a disarmed landing")
        rid = receipt["run_id"]
        doc = json.loads((directory / f"{rid}.provenance.json").read_text())
        recorded = doc["airborne_training"]
        print("Recorder", recorded["status"], "airborne Wh", recorded["estimated_airborne_energy_Wh"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--work", type=Path, default=ROOT / "tmp/m100_direct_airborne_three")
    p.add_argument("--attempt", required=True, type=int)
    p.add_argument("cases", nargs="+")
    a = p.parse_args()
    if a.attempt < 1:
        p.error("attempt must be positive")
    run(a.protocol, a.work, a.attempt, a.cases)
