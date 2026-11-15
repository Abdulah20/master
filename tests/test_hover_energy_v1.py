from __future__ import annotations

import numpy as np
import pytest

from energy_model.hover_energy_v1 import (
    command_thrust_N,
    ideal_induced_power_lower_bound_W,
    integrate_energy_J,
    nominal_vehicle_mass_kg,
    required_hover_thrust_N,
    total_disk_area_m2,
    total_mass_kg,
)


def test_nominal_mass_sum_si() -> None:
    assert nominal_vehicle_mass_kg(2.0, 4, 0.016076923076923075) == pytest.approx(2.0643076923)


def test_total_mass_includes_payload_once() -> None:
    assert total_mass_kg(2.0, 0.25) == pytest.approx(2.25)
    with pytest.raises(ValueError):
        total_mass_kg(2.0, -0.01)


def test_disk_area_units() -> None:
    diameter = 0.28
    assert total_disk_area_m2(4, diameter) == pytest.approx(4 * np.pi * (diameter / 2) ** 2)


def test_command_thrust_quadratic_scaling() -> None:
    omega = np.full((3, 4), 400.0)
    base = command_thrust_N(omega, 8.54858e-6)
    doubled = command_thrust_N(2 * omega, 8.54858e-6)
    assert doubled == pytest.approx(4 * base)


def test_hover_thrust_equals_weight() -> None:
    assert required_hover_thrust_N(2.0, 9.8) == pytest.approx(19.6)


def test_ned_vertical_acceleration_sign() -> None:
    mass = 2.0
    gravity = 9.8
    upward_acceleration_ned = -1.0
    assert mass * (gravity - upward_acceleration_ned) > mass * gravity


def test_induced_power_mass_and_density_scaling() -> None:
    area = 0.25
    p1 = ideal_induced_power_lower_bound_W(20.0, 1.2, area)
    p_mass = ideal_induced_power_lower_bound_W(40.0, 1.2, area)
    p_density = ideal_induced_power_lower_bound_W(20.0, 2.4, area)
    assert p_mass / p1 == pytest.approx(2.0**1.5)
    assert p_density / p1 == pytest.approx(2.0**-0.5)


def test_energy_constant_power_and_joule_to_wh() -> None:
    timestamps = np.array([0, 1_000_000, 3_000_000], dtype=np.int64)
    energy_j = integrate_energy_J(timestamps, np.full(3, 1200.0))
    assert energy_j == pytest.approx(3600.0)
    assert energy_j / 3600.0 == pytest.approx(1.0)


def test_trapezoidal_integration_irregular_timestamps() -> None:
    timestamps = np.array([0, 500_000, 2_000_000], dtype=np.int64)
    power = np.array([0.0, 2.0, 2.0])
    assert integrate_energy_J(timestamps, power) == pytest.approx(3.5)

