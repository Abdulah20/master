"""Run the short X500 takeoff-hover-translate-land mission through MAVLink."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from pymavlink import mavutil

from simulation.scenario import ScenarioInputs, load_scenario_config


class FlightError(RuntimeError):
    """Raised when PX4 does not safely complete a commanded phase."""


class OffboardMission:
    def __init__(self, connection: str, inputs: ScenarioInputs) -> None:
        self.inputs = inputs
        self.master = mavutil.mavlink_connection(
            connection,
            source_system=245,
            source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
        )
        self.position: Any | None = None
        self.heartbeat: Any | None = None
        self.landed_state: int | None = None
        self.last_gcs_heartbeat_wall_s = 0.0
        self.markers: list[dict[str, Any]] = []

    def connect(self, timeout_s: float = 20.0) -> None:
        heartbeat = self.master.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            raise FlightError("Timed out waiting for the PX4 heartbeat")
        self.heartbeat = heartbeat
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            20,
            1,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.position is None:
            self._service_messages(0.1)
        if self.position is None:
            raise FlightError("PX4 did not provide LOCAL_POSITION_NED")

    def _send_gcs_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self.last_gcs_heartbeat_wall_s >= 0.8:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
            self.last_gcs_heartbeat_wall_s = now

    def _service_messages(self, duration_s: float = 0.0) -> None:
        deadline = time.monotonic() + duration_s
        while True:
            message = self.master.recv_match(blocking=False)
            if message is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.005)
                continue
            message_type = message.get_type()
            if message_type == "LOCAL_POSITION_NED":
                self.position = message
            elif message_type == "HEARTBEAT" and message.get_srcSystem() == self.master.target_system:
                self.heartbeat = message
            elif message_type == "EXTENDED_SYS_STATE":
                self.landed_state = int(message.landed_state)
        self._send_gcs_heartbeat()

    def _sim_time_s(self) -> float:
        if self.position is None:
            raise FlightError("Local position is unavailable")
        return float(self.position.time_boot_ms) / 1000.0

    def mark(self, phase: str) -> None:
        self._service_messages()
        self.markers.append(
            {
                "phase": phase,
                "px4_time_s": self._sim_time_s(),
                "position_ned_m": [
                    float(self.position.x),
                    float(self.position.y),
                    float(self.position.z),
                ],
            }
        )

    def _send_position_target(self, x_m: float, y_m: float, z_m: float, yaw_rad: float) -> None:
        ignore = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self.master.mav.set_position_target_local_ned_send(
            int(self._sim_time_s() * 1000.0),
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            ignore,
            x_m,
            y_m,
            z_m,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            yaw_rad,
            0.0,
        )

    def _stream_target(
        self,
        target: tuple[float, float, float, float],
        *,
        duration_s: float | None = None,
        reached: Callable[[], bool] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        start = time.monotonic()
        while True:
            self._service_messages()
            self._send_position_target(*target)
            elapsed = time.monotonic() - start
            if duration_s is not None and elapsed >= duration_s:
                return
            if reached is not None and reached():
                return
            if timeout_s is not None and elapsed >= timeout_s:
                raise FlightError("A flight phase did not reach its target before timeout")
            time.sleep(0.08)

    def _command(self, command: int, parameters: tuple[float, ...], timeout_s: float = 8.0) -> None:
        if len(parameters) != 7:
            raise ValueError("MAVLink COMMAND_LONG requires seven parameters")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            command,
            0,
            *parameters,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._send_gcs_heartbeat()
            acknowledgement = self.master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.2)
            if acknowledgement is None or int(acknowledgement.command) != command:
                continue
            if int(acknowledgement.result) in (
                mavutil.mavlink.MAV_RESULT_ACCEPTED,
                mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
            ):
                return
            raise FlightError(
                f"PX4 rejected command {command} with MAV_RESULT {acknowledgement.result}"
            )
        raise FlightError(f"No PX4 acknowledgement for command {command}")

    def run(self) -> list[dict[str, Any]]:
        self.connect()
        self._service_messages(1.0)
        start_x = float(self.position.x)
        start_y = float(self.position.y)
        start_z = float(self.position.z)
        target_z = start_z - self.inputs.mission.target_relative_altitude_m
        yaw = float(getattr(self.master.messages.get("ATTITUDE"), "yaw", 0.0))
        takeoff_target = (start_x, start_y, target_z, yaw)

        self.mark("prearm")
        self._stream_target((start_x, start_y, start_z, yaw), duration_s=2.0)
        self._command(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            (
                float(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                6.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        self._command(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

        self.mark("takeoff")
        self._stream_target(
            takeoff_target,
            reached=lambda: float(self.position.z) <= target_z + 0.25,
            timeout_s=15.0,
        )
        self.mark("hover")
        self._stream_target(takeoff_target, duration_s=self.inputs.mission.hover_duration_s)

        target_x = start_x + self.inputs.mission.translation_north_m
        target_y = start_y + self.inputs.mission.translation_east_m
        move_target = (target_x, target_y, target_z, yaw)
        distance = lambda: math.hypot(float(self.position.x) - target_x, float(self.position.y) - target_y)
        move_distance = math.hypot(
            self.inputs.mission.translation_north_m,
            self.inputs.mission.translation_east_m,
        )
        move_timeout = max(8.0, 3.0 * move_distance / self.inputs.mission.translation_speed_m_s)
        self.mark("move")
        self._stream_target(move_target, reached=lambda: distance() <= 0.25, timeout_s=move_timeout)
        self.mark("post_move_hover")
        self._stream_target(move_target, duration_s=3.0)

        self.mark("landing")
        self._command(
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            (0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 0.0),
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            self._service_messages(0.1)
            if self.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND:
                break
        else:
            raise FlightError("PX4 did not report landing before timeout")
        self.mark("landed")
        return self.markers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--connection", default="udpin:127.0.0.1:14540")
    args = parser.parse_args()
    mission = OffboardMission(args.connection, load_scenario_config(args.config))
    markers = mission.run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"result": "pass", "phases": markers}, indent=2) + "\n")
    print(json.dumps({"result": "pass", "output": str(args.output.resolve()), "phases": markers}))


if __name__ == "__main__":
    main()
