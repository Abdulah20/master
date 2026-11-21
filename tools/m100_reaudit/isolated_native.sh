#!/usr/bin/env bash
# Private loopback, process table, IPC and PX4 working files permit independent SITL workers.
set -eo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
plan=$(realpath "${1:-$root/config/m100_reaudit/native_pilots.json}")
run_dir=$(realpath -m "${2:-$root/tmp/m100_reaudit_live/isolated}")
mkdir -p "$run_dir/state/rootfs" "$run_dir/state/px4_text_logs" "$run_dir/state/temperature_models"
px4=${M100_PX4:-/home/abdulah/PX4-Autopilot}/build/px4_sitl_default/rootfs
# Only copy initial bootstrap files. Preserve the initial parameters and the logs of this worker on retry.
if [[ ! -f "$run_dir/state/rootfs/gz_env.sh" ]]; then
  cp -a "$px4/gz_env.sh" "$px4/dataman" "$px4/eeprom" "$px4"/parameters*.bson "$run_dir/state/rootfs/"
fi
models=$(realpath "${M100_PX4:-/home/abdulah/PX4-Autopilot}/Tools/simulation/gz/models")
# A non-reference temperature generates a Gazebo model and registers a model-directory alias.
# Keep this alias inside the worker, even when two workers generate the same model.
warm=$(python3 -c 'import json,sys; print(int(any(c["temperature_c"] != 15 for c in json.load(open(sys.argv[1])))))' "$plan")
sha256sum "$root/tools/m100_reaudit/isolated_native.sh" | cut -d' ' -f1 > "$run_dir/launcher_sha256"
exec unshare --user --map-root-user --net --ipc --mount --pid --fork --mount-proc bash -c '
  set -eo pipefail
  mount --make-rprivate /
  ip link set lo multicast on up
  ip route add 224.0.0.0/4 dev lo
  mount -t tmpfs tmpfs /tmp
  mount -t tmpfs tmpfs /dev/shm
  mount --bind "$3/state/rootfs" "$4"
  if [[ $6 == 1 ]]; then
    # Keep original SDFs read-only while allowing generated aliases at the registry root.
    # Real directory mount points preserve the canonical validator PX4 / ROMFS ancestry.
    sources="$3/state/shared_models"
    registry="$3/state/model_registry"
    mkdir -p "$sources" "$registry"
    mount --bind "$5" "$sources"
    mount -o remount,bind,ro "$sources"
    mount --bind "$registry" "$5"
    for model in "$sources"/*; do
      name=${model##*/}
      if [[ $name == *_temp_* ]]; then continue; fi
      if [[ -d "$model" ]]; then
        mkdir -p "$5/$name"
      else
        touch "$5/$name"
      fi
      mount --bind "$model" "$5/$name"
      mount -o remount,bind,ro "$5/$name"
    done
  fi
  if [[ -n ${M100_GAZEBO_SEED:-} ]]; then
    if ! [[ $M100_GAZEBO_SEED =~ ^[0-9]+$ ]]; then exit 2; fi
    source_rc=$(realpath "$4/../etc/init.d-posix/px4-rc.gzsim")
    grep -q -- "--verbose=\${GZ_VERBOSE:=1} -r -s" "$source_rc"
    sed "s/--verbose=\${GZ_VERBOSE:=1} -r -s/--seed $M100_GAZEBO_SEED --verbose=\${GZ_VERBOSE:=1} -r -s/" "$source_rc" > "$3/state/px4-rc.gzsim"
    chmod +x "$3/state/px4-rc.gzsim"
    mount --bind "$3/state/px4-rc.gzsim" "$source_rc"
  fi
  export M100_AUDIT_ISOLATED=1
  exec "$1/tools/m100_reaudit/run_native.sh" "$2" "$3"
' bash "$root" "$plan" "$run_dir" "$px4" "$models" "$warm"
