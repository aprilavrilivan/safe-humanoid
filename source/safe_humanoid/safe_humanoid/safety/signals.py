"""Raw, unit-bearing evaluation signals without guessed G1 hardware limits."""

from __future__ import annotations

import math
from collections.abc import Sequence


def velocity_sample(
    command_b: Sequence[float],
    linear_velocity_b: Sequence[float],
    angular_velocity_b: Sequence[float],
    projected_gravity_b: Sequence[float],
    joint_torque_nm: Sequence[float],
    joint_velocity_rad_s: Sequence[float],
) -> dict[str, float]:
    """Compute one robot's tracking and raw loading signals at a policy step.

    These are observables, not physical safety-limit violations. Torques are
    *applied* rather than commanded torque and sampling is at policy rate.
    """

    if not all(len(vector) >= 3 for vector in (
        command_b, linear_velocity_b, angular_velocity_b, projected_gravity_b
    )):
        raise ValueError("command and base-state vectors must have at least three components")
    if not joint_torque_nm or len(joint_torque_nm) != len(joint_velocity_rad_s):
        raise ValueError("joint torque and velocity vectors must be nonempty and aligned")
    values = (*command_b[:3], *linear_velocity_b[:3], *angular_velocity_b[:3],
              *projected_gravity_b[:3], *joint_torque_nm, *joint_velocity_rad_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("raw simulator state contains NaN or Inf")
    return {
        "tracking_error_xy_m_s": math.hypot(
            command_b[0] - linear_velocity_b[0], command_b[1] - linear_velocity_b[1]
        ),
        "tracking_error_yaw_rad_s": abs(command_b[2] - angular_velocity_b[2]),
        "upright_alignment": max(0.0, min(1.0, -projected_gravity_b[2])),
        "joint_torque_rms_nm": math.sqrt(
            sum(torque * torque for torque in joint_torque_nm) / len(joint_torque_nm)
        ),
        "joint_torque_peak_nm": max(abs(torque) for torque in joint_torque_nm),
        "joint_speed_peak_rad_s": max(abs(speed) for speed in joint_velocity_rad_s),
        "absolute_mechanical_power_w": sum(
            abs(torque * speed) for torque, speed in zip(joint_torque_nm, joint_velocity_rad_s)
        ),
    }
