"""Four task-only terms for the clean G1 velocity-tracking baseline."""

from __future__ import annotations

import torch


def track_velocity_axis_exp(env, command_name: str, axis: int, std: float) -> torch.Tensor:
    """Unitless exp(-squared_error / std**2) for x, y, or yaw velocity."""

    if axis not in (0, 1, 2) or std <= 0:
        raise ValueError("axis must be 0, 1, or 2 and std must be positive")
    command = env.command_manager.get_command(command_name)[:, axis]
    robot = env.scene["robot"]
    actual = robot.data.root_lin_vel_b[:, axis] if axis < 2 else robot.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(command - actual) / (std * std))


def upright_alignment(env) -> torch.Tensor:
    """One when torso z is upright; zero at horizontal or inverted attitude."""

    gravity_b = env.scene["robot"].data.projected_gravity_b
    return torch.clamp(-gravity_b[:, 2], min=0.0, max=1.0)


def rolling_torque_rms_cost(env) -> torch.Tensor:
    """Nonnegative, normalized sustained-loading proxy at physics rate."""

    return env.safe_humanoid_safety_buffer.time_cost()


def high_band_motion_cost(env) -> torch.Tensor:
    """Nonnegative, normalized 25–100 Hz torque/speed-band proxy."""

    return env.safe_humanoid_safety_buffer.frequency_cost()
