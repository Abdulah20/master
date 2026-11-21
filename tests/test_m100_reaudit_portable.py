"""Clone-side reproduction checks for the bundled M100 model and original protocol."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from tools.m100_reaudit import physics
from tools.m100_reaudit.historical import ORIGINAL_SHA256, read_study


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_pinned_snapshot_and_original_study_view():
    root = physics.PORTABLE_RUNTIME
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert digest(root / name) == expected, name
    provenance = json.loads((root / "provenance.json").read_text())
    for name, expected in provenance["public_field_evaluator"]["dependency_paths_sha256"].items():
        assert digest(physics.ROOT / name) == expected

    study = physics.PORTABLE_STUDY
    protocol, records, export = read_study(study)
    assert len(protocol["cases"]) == len(records) == len(export["sources"]) == 69
    assert export["csv_sha256"] == ORIGINAL_SHA256
    original = provenance["original_study"]["source_sha256"]
    assert export["protocol_sha256"] == original["protocol.json"]
    assert export["original_results_sha256"] == original["results.json"]
    assert export["original_export_sha256"] == original["M100_69_experiments_summary.csv.export.json"]
    assert export["results_sha256"] != export["original_results_sha256"]
    with (study / "M100_69_experiments_summary.csv").open() as handle:
        assert len(csv.DictReader(handle).fieldnames) == 18
    assert all(set(r) == set(export["portable_results_fields"]) for r in records.values())


def test_runtime_selection_requires_explicit_roots_to_exist(monkeypatch, tmp_path):
    monkeypatch.delenv("M100_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr(physics, "DEFAULT_RUNTIME", tmp_path)
    assert physics.runtime_path() == physics.PORTABLE_RUNTIME

    # Explicit choices must never acquire an unintended model by fallback.
    absent = tmp_path / "missing"
    monkeypatch.setenv("M100_RUNTIME_ROOT", str(absent))
    assert physics.runtime_path() == absent
    with pytest.raises(FileNotFoundError, match="M100 runtime missing"):
        physics.Model.load()
    assert physics.runtime_path(physics.PORTABLE_RUNTIME) == physics.PORTABLE_RUNTIME


def test_physics_without_installed_drone_nav_or_project_checkout(tmp_path):
    program = """
import math
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.m100_reaudit import physics
assert physics.runtime_path() == physics.PORTABLE_RUNTIME
model = physics.Model.load()
air = model.atmosphere(15, 100)
assert abs(model.phase_power('CRUISE', 0, air.density_kg_m3) - 464.69522402455533) < 1e-8
assert math.isclose(model.assess(.25, 6, 0, 100, 15).physical_mass_kg, 3.93, abs_tol=1e-12)
import drone_nav.m100_transient_power, drone_nav.m100_wind_simulation
assert all(Path(module.__file__).resolve().is_relative_to(physics.PORTABLE_RUNTIME)
           for name, module in sys.modules.items() if name.split('.')[0] == 'drone_nav')
"""
    environment = dict(os.environ, HOME=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    environment.pop("M100_RUNTIME_ROOT", None)
    subprocess.run([sys.executable, "-I", "-c", program, str(physics.ROOT)],
                   env=environment, cwd=tmp_path, check=True)


@pytest.mark.parametrize("payload,suffix", [(0, ""), (.25, "_payload_025"), (.5, "_payload_050"),
                                          (.75, "_payload_075"), (1., "_payload_100"), (1.25, "_payload_125")])
def test_sdf_mass_and_physics_are_traceable_without_external_meshes(payload, suffix):
    root = physics.PORTABLE_RUNTIME
    px4 = physics.PORTABLE_PX4
    assets = Path("Tools/simulation/gz/models")
    provenance = json.loads((root / "provenance.json").read_text())
    expected = provenance["native_installation"]["current_px4_artifact_sha256"]
    base = assets / "m100_cmu_v2/model.sdf"
    assert digest(px4 / base) == expected[str(base)]
    model = ET.parse(px4 / base).getroot().find("model")
    links = list(model.findall("link"))
    motors = model.findall("plugin[@name='gz::sim::systems::MulticopterMotorModel']")
    assert len(motors) == 4
    assert {m.findtext("motorNumber") for m in motors} == {"0", "1", "2", "3"}
    if suffix:
        wrapper = assets / ("m100_cmu_v2" + suffix) / "model.sdf"
        assert digest(px4 / wrapper) == expected[str(wrapper)]
        extra = ET.parse(px4 / wrapper).getroot().find("model")
        assert extra.findtext("include/uri") == "model://m100_cmu_v2"
        links.extend(extra.findall("link"))
    assert sum(float(link.findtext("inertial/mass")) for link in links) == pytest.approx(3.680 + payload)
    assert any(link.findall("collision") for link in links)
    assert not any(file.suffix.lower() in {".stl", ".obj", ".dae", ".step"} for file in px4.rglob("*"))
