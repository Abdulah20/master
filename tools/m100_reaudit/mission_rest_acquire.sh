#!/usr/bin/env bash
# Run one fresh native case, retaining the original pack stream beyond disarm.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
prepare_only=0
args=()
for arg in "$@"; do
  case "$arg" in
    --prepare-only) prepare_only=1 ;;
    -h|--help)
      echo "Usage: $0 [--prepare-only] [--step 0.004|0.002] [--seed 3771]"
      exit 0 ;;
    *) args+=("$arg") ;;
  esac
done
base="$root/tmp/m100_closure/rest_checks"
mkdir -p "$base"
raw=$(mktemp -d "$base/run-XXXXXXXX")
echo "Raw acquisition directory: $raw"
tool=(/usr/bin/python3 -m tools.m100_reaudit.mission_rest_check)
"${tool[@]}" prepare "$raw" "${args[@]}"
bash -n "$raw/launchers/isolated_native.sh"
bash -n "$raw/launchers/run_native.sh"
if ((prepare_only)); then exit 0; fi
"${tool[@]}" check-install "$raw"
export M100_REST_ROOT="$root" M100_REST_INNER_LAUNCHER="$raw/launchers/run_native.sh"
export M100_INITIAL_SOC_PERCENT=100 M100_MISSION_ROUTE=direct M100_PACK_DYNAMICS=cmu_2rc
unset M100_BATTERY_RESEARCH_PROFILE
M100_GAZEBO_SEED=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["seed"])' "$raw/request.json")
export M100_GAZEBO_SEED
code=0
"$raw/launchers/isolated_native.sh" "$raw/request.json" "$raw" >"$raw/launcher.log" 2>&1 || code=$?
echo "Native launcher exit code: $code"
"${tool[@]}" analyze "$raw"
exit "$code"
