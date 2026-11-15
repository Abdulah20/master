"""Record timestamped Gazebo rotor, energy, battery, and clock messages as JSONL."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

from gz.msgs10.battery_state_pb2 import BatteryState
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.double_pb2 import Double
from gz.msgs10.float_v_pb2 import Float_V
from gz.transport13 import Node


class GazeboEnergyRecorder:
    """Thread-safe JSONL recorder using simulation time rather than wall time."""

    def __init__(self, output_path: Path, model_name: str, world_name: str) -> None:
        self.output_path = output_path
        self.model_name = model_name
        self.world_name = world_name
        self.node = Node()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_clock_s: float | None = None
        self.record_count = 0
        self.stream = None

    @staticmethod
    def _header_time_s(message: Any) -> float | None:
        if not message.HasField("header") or not message.header.HasField("stamp"):
            return None
        stamp = message.header.stamp
        return float(stamp.sec) + float(stamp.nsec) * 1e-9

    def _write(self, source: str, message: Any, values: dict[str, Any]) -> None:
        timestamp = self._header_time_s(message)
        with self.lock:
            if timestamp is None:
                timestamp = self.latest_clock_s
            if timestamp is None or self.stream is None:
                return
            record = {"source": source, "sim_time_s": timestamp, **values}
            self.stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.stream.flush()
            self.record_count += 1

    def _clock_callback(self, message: Clock) -> None:
        with self.lock:
            self.latest_clock_s = float(message.sim.sec) + float(message.sim.nsec) * 1e-9

    def _scalar_callback(self, source: str) -> Callable[[Double], None]:
        return lambda message: self._write(source, message, {"value": float(message.data)})

    def _rotor_callback(self, message: Float_V) -> None:
        self._write("rotor_speed_rad_s", message, {"values": [float(value) for value in message.data]})

    def _battery_callback(self, message: BatteryState) -> None:
        soc_percent = float(message.percentage)
        self._write(
            "battery_state",
            message,
            {
                "voltage_v": float(message.voltage),
                "current_a": float(message.current),
                "charge_ah": float(message.charge),
                "capacity_ah": float(message.capacity),
                "soc_fraction": soc_percent / 100.0,
                "soc_percent": soc_percent,
            },
        )

    def subscribe(self) -> None:
        root = f"/model/{self.model_name}/energy"
        self.node.subscribe(Clock, f"/world/{self.world_name}/clock", self._clock_callback)
        self.node.subscribe(Float_V, f"{root}/rotor_speed_rad_s", self._rotor_callback)
        self.node.subscribe(Double, f"{root}/shaft_power_w", self._scalar_callback("shaft_power_w"))
        self.node.subscribe(
            Double, f"{root}/electrical_load_w", self._scalar_callback("electrical_load_w")
        )
        self.node.subscribe(
            Double,
            f"{root}/cumulative_energy_j",
            self._scalar_callback("cumulative_energy_j"),
        )
        battery_topic = f"/model/{self.model_name}/battery/x500_energy_battery/state"
        self.node.subscribe(BatteryState, battery_topic, self._battery_callback)

    def run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as stream:
            self.stream = stream
            self.subscribe()
            while not self.stop_event.wait(0.1):
                pass
            self.stream = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="x500_energy")
    parser.add_argument("--world", default="x500_energy_first_flight")
    args = parser.parse_args()
    recorder = GazeboEnergyRecorder(args.output, args.model, args.world)

    def stop(_signum: int, _frame: Any) -> None:
        recorder.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    recorder.run()
    print(json.dumps({"output": str(args.output.resolve()), "records": recorder.record_count}))


if __name__ == "__main__":
    main()
