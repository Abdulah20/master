# master

UAV energy thesis research: PX4/Gazebo X500 simulation, hover-energy modelling,
and DJI Matrice 100 flight-data analysis.

## Project files

- `src/`: energy models, flight-data processing, simulation tools, and Gazebo plugin.
- `config/`: experiment and analysis configurations.
- `models/` and `worlds/`: Gazebo simulation assets.
- `data/`: external datasets and restored research inputs.
- `results/`: recorded experiment results and provenance.
- `tests/`: existing Python and C++ checks.
- `docs/`: experiment methods, validation, limitations, and reproduction commands.

Simulation energy estimates are not calibrated physical measurements. See the
individual experiment documents for scientific scope and validation status.

## Third-party dependency

NREL Thevenin is included as a Git submodule. Clone with `git clone --recurse-submodules`,
or run `git submodule update --init --recursive` after cloning. Its version and license
are documented in `external/third_party/THEVENIN_PROVENANCE.md`.

Python environments, build products, and recovery snapshots are excluded from Git.
Research inputs and recorded results remain included.
# master
