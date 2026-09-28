"""Simple full-joint and floating-root imitation rewards."""

import torch
from isaaclab.utils.math import quat_error_magnitude


def _reference(env, command_name: str):
    return env.command_manager.get_term(command_name)


def track_joint_pos_exp(env, command_name: str, std: float) -> torch.Tensor:
    robot = env.scene["robot"]
    reference = _reference(env, command_name)
    mse = torch.mean(torch.square(robot.data.joint_pos - reference.joint_pos), dim=1)
    return torch.exp(-mse / (std * std))


def track_joint_vel_exp(env, command_name: str, std: float) -> torch.Tensor:
    robot = env.scene["robot"]
    reference = _reference(env, command_name)
    mse = torch.mean(torch.square(robot.data.joint_vel - reference.joint_vel), dim=1)
    return torch.exp(-mse / (std * std))


def track_root_height_exp(env, command_name: str, std: float) -> torch.Tensor:
    robot = env.scene["robot"]
    reference = _reference(env, command_name)
    height_error = robot.data.root_pos_w[:, 2] - reference.root_pos_w[:, 2]
    return torch.exp(-torch.square(height_error) / (std * std))


def track_root_orientation_exp(env, command_name: str, std: float) -> torch.Tensor:
    robot = env.scene["robot"]
    reference = _reference(env, command_name)
    angle = quat_error_magnitude(robot.data.root_quat_w, reference.root_quat_w)
    return torch.exp(-torch.square(angle) / (std * std))
