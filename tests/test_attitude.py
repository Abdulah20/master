from __future__ import annotations

import numpy as np
import pytest

from data_processing.attitude import attitude_csv_columns, quaternion_wxyz_to_euler_rad


def test_identity_quaternion_is_level_and_zero_yaw() -> None:
    result = quaternion_wxyz_to_euler_rad(np.array([1.0, 0.0, 0.0, 0.0]))
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_quaternion_converts_to_roll_pitch_yaw_radians() -> None:
    half = np.sqrt(0.5)
    yaw = quaternion_wxyz_to_euler_rad(np.array([half, 0.0, 0.0, half]))
    roll = quaternion_wxyz_to_euler_rad(np.array([half, half, 0.0, 0.0]))
    assert yaw == pytest.approx([0.0, 0.0, np.pi / 2.0])
    assert roll == pytest.approx([np.pi / 2.0, 0.0, 0.0])


def test_non_unit_quaternion_is_normalized() -> None:
    assert quaternion_wxyz_to_euler_rad(np.array([2.0, 0.0, 0.0, 0.0])) == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_attitude_csv_columns_preserve_quaternion_and_add_euler() -> None:
    source = np.array([[1.0, 0.0, 0.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]])
    columns = attitude_csv_columns(source)
    assert tuple(columns) == ("q_w", "q_x", "q_y", "q_z", "roll_rad", "pitch_rad", "yaw_rad")
    assert columns["yaw_rad"] == pytest.approx([0.0, np.pi / 2.0])


def test_invalid_quaternion_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        quaternion_wxyz_to_euler_rad(np.zeros(4))
    with pytest.raises(ValueError, match="four values"):
        quaternion_wxyz_to_euler_rad(np.ones(3))
