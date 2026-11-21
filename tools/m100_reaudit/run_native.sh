#!/usr/bin/env bash
# Start just the campaign's headless ROS nodes. The manager owns its SITL child.
set -eo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
project=${M100_RUNTIME:-/home/abdulah/Master/Drone/project}
px4=${M100_PX4:-/home/abdulah/PX4-Autopilot}
run_dir=${2:-"$root/tmp/m100_reaudit_live/after"}
plan=${1:-"$root/config/m100_reaudit/native_pilots.json"}
mkdir -p "$run_dir"
source /opt/ros/jazzy/setup.bash
source "$project/ros2_ws/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-84} ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION=${GZ_PARTITION:-m100_reaudit}
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix --share m100_gazebo_assets)/models:$px4/Tools/simulation/gz/models:$px4/Tools/simulation/gz/worlds"
export GZ_GUI_PLUGIN_PATH="$(ros2 pkg prefix m100_gazebo_gui)/lib/m100_gazebo_gui"
cd "$root"
"$(ros2 pkg prefix ros_gz_bridge)/lib/ros_gz_bridge/parameter_bridge" '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock' '/m100/rotor_joint_states@sensor_msgs/msg/JointState[gz.msgs.Model' >"$run_dir/bridge.log" 2>&1 &
bridge_pid=$!
/usr/bin/python3 -m tools.m100_reaudit.native manager "$run_dir" >"$run_dir/manager.log" 2>&1 &
manager_pid=$!
cleanup() {
  kill -INT "$manager_pid" "$bridge_pid" 2>/dev/null || true
  for _ in {1..8}; do
    if ! kill -0 "$manager_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if [[ ${M100_AUDIT_ISOLATED:-0} == 1 ]]; then
    kill -KILL "$manager_pid" "$bridge_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
/usr/bin/python3 -m tools.m100_reaudit.native collect "$run_dir" --plan "$plan"
