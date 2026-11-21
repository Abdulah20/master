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
exec unshare --user --map-root-user --net --ipc --mount --pid --fork --mount-proc bash -c '
  set -eo pipefail
  mount --make-rprivate /
  ip link set lo multicast on up
  ip route add 224.0.0.0/4 dev lo
  mount -t tmpfs tmpfs /tmp
  mount -t tmpfs tmpfs /dev/shm
  mount --bind "$3/state/rootfs" "$4"
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
' bash "$root" "$plan" "$run_dir" "$px4"
