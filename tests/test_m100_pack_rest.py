"""Native two-RC rest, transition and missing-observation regression."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_confirmed_rest_with_repeated_disarmed_messages(tmp_path):
    compiler = shutil.which("c++")
    pkg_config = shutil.which("pkg-config")
    if compiler is None or pkg_config is None:
        pytest.skip("Native Gazebo development tools are unavailable")
    env = dict(os.environ)
    # Use the same Gazebo distribution as the active ROS environment.
    prefixes = [Path(p) / "lib/pkgconfig" for p in env.get("CMAKE_PREFIX_PATH", "").split(os.pathsep) if p]
    env["PKG_CONFIG_PATH"] = os.pathsep.join([*(str(p) for p in prefixes if p.is_dir()),
                                             env.get("PKG_CONFIG_PATH", "")])
    packages = ["gz-sim8", "gz-transport13", "gz-plugin2-register", "Qt5Core"]
    if subprocess.run([pkg_config, "--exists", *packages], env=env).returncode:
        pytest.skip("Gazebo Harmonic and Qt5 development files are unavailable")
    flags = subprocess.check_output([pkg_config, "--cflags", "--libs", *packages], env=env, text=True)
    source = Path(__file__).resolve().parents[1] / "tools/m100_reaudit/pack_drive/test/pack_rest.cc"
    program = tmp_path / "pack_rest"
    subprocess.run([compiler, "-std=c++17", "-fPIC", "-O0", str(source), "-o", str(program),
                    *shlex.split(flags), "-UNDEBUG"], check=True, env=env)
    subprocess.run([program], check=True, env=env)
