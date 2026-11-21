"""Apply the identified PX4 flight-clock correction to the linked SITL manager."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess

PATCH = Path(__file__).with_name("braking.patch")
MANAGER = "ros2_ws/src/drone_nav/drone_nav/m100_experiment_manager.py"
MANAGER_BEFORE = "b5de6ec30656a9e9013c5546943f19161878c8ce4b13c60c2f357000083dfcb6"
MANAGER_AFTER = "96e630ba7edaf4a017a601fcce4faddcc943963de047913209695798d88efebc"
GUIDANCE_TEST = "ros2_ws/src/drone_nav/test/test_m100_return_guidance.py"
FIXTURE_BEFORE = "2e2cfc5f46f74362f99d803b73545221317e75ad0acabcd17878a6119b509e7b"
FIXTURE_AFTER = "8ed3c9582e8613ef521c6e67e50c4fb49e44316567992a04697c9cbfb4d86d0e"


def ensure_braking_patch(runtime_root=None):
    """Apply to the density-corrected source, preserving an absent optional test tree."""
    root = Path(runtime_root or os.environ.get("M100_RUNTIME_ROOT") or Path.home() / "Master/Drone/project").resolve()
    checked = [(MANAGER, MANAGER_BEFORE, MANAGER_AFTER)]
    if (root / GUIDANCE_TEST).is_file():
        checked.append((GUIDANCE_TEST, FIXTURE_BEFORE, FIXTURE_AFTER))

    def hashes():
        return [hashlib.sha256((root / path).read_bytes()).hexdigest() for path, _, _ in checked]

    before = [old for _, old, _ in checked]
    after = [new for _, _, new in checked]
    actual = hashes()
    if actual == after:
        return "ALREADY_APPLIED"
    if actual != before:
        raise ValueError(f"Unexpected source revision in {root}; verify the original density correction first")
    command = ["git", "apply", *(f"--include={path}" for path, _, _ in checked)]
    subprocess.run([*command, "--check", str(PATCH)], cwd=root, check=True)
    subprocess.run([*command, str(PATCH)], cwd=root, check=True)
    if hashes() != after:
        raise OSError("Unexpected manager or test fixture after the flight-clock correction")
    return "APPLIED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-runtime", action="store_true", required=True)
    parser.add_argument("--runtime-root", type=Path)
    print(ensure_braking_patch(parser.parse_args().runtime_root))
