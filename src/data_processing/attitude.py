"""PX4 quaternion conversion utilities for the unified research CSV."""

from __future__ import annotations

import numpy as np


def quaternion_wxyz_to_euler_rad(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert PX4 ``[w, x, y, z]`` quaternions to roll, pitch, yaw radians."""

    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.ndim == 0 or quaternion.shape[-1] != 4:
        raise ValueError("Quaternion input must end with four values ordered w, x, y, z")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("Quaternion input must be finite")

    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("Quaternion norm must be non-zero")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(pitch_argument)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def attitude_csv_columns(quaternion_wxyz: np.ndarray) -> dict[str, np.ndarray]:
    """Return explicitly unit-labelled quaternion and Euler CSV columns."""

    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    euler = quaternion_wxyz_to_euler_rad(quaternion)
    return {
        "q_w": quaternion[..., 0],
        "q_x": quaternion[..., 1],
        "q_y": quaternion[..., 2],
        "q_z": quaternion[..., 3],
        "roll_rad": euler[..., 0],
        "pitch_rad": euler[..., 1],
        "yaw_rad": euler[..., 2],
    }
