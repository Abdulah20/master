#!/usr/bin/env bash
set -euo pipefail

archive="/home/abdulah/UAV_Energy_Thesis_Backup_2026-08-23/FULL_03_PX4_Autopilot_COMPLETE_2026-08-23.tar.zst"
archive_sha256="8691708fc06fdf9e157c0cc2ffe75a2898a04e64bcd28531fe693120ddf5b782"
member="PX4-Autopilot/build/px4_sitl_default/rootfs/log/2026-08-17/00_55_32.ulg"
ulog_sha256="430113e447b75526e10ad7742eef3bf586c85852cd0ca2911cf6055479517f19"
project_root="/home/abdulah/uav-energy-thesis"
durable_copy="${project_root}/data/restored/hover_energy_v1/00_55_32.ulg"
recorded_path="/home/abdulah/${member}"

if [[ ! -f "${archive}" ]]; then
  echo "missing verified archive: ${archive}" >&2
  exit 1
fi

printf '%s  %s\n' "${archive_sha256}" "${archive}" | sha256sum -c -

restore_stage="$(mktemp -d /tmp/hover-energy-v1-restore.XXXXXX)"
trap 'rm -rf -- "${restore_stage}"' EXIT

tar --zstd -xf "${archive}" -C "${restore_stage}" "${member}"
staged_ulog="${restore_stage}/${member}"
printf '%s  %s\n' "${ulog_sha256}" "${staged_ulog}" | sha256sum -c -

install -d -m 0755 "$(dirname "${durable_copy}")"
install -m 0644 "${staged_ulog}" "${durable_copy}"

# Retain the historical recorded path for compatibility. PX4 may recreate its
# build/rootfs tree during later simulations; the durable thesis copy remains
# the canonical current reproduction input if that volatile path disappears.
install -d -m 0755 "$(dirname "${recorded_path}")"
install -m 0644 "${staged_ulog}" "${recorded_path}"

printf '%s  %s\n' "${ulog_sha256}" "${durable_copy}" | sha256sum -c -
printf '%s  %s\n' "${ulog_sha256}" "${recorded_path}" | sha256sum -c -

echo "durable_copy=${durable_copy}"
echo "recorded_path=${recorded_path}"
