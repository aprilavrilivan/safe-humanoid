"""A one-shot, episode-relative velocity kick for push-recovery evaluation.

This follows Isaac Lab's ``push_by_setting_velocity`` semantics. It is not a
calibrated contact force and must not be reported in newtons.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase


class ScheduledVelocityKick(ManagerTermBase):
    """Apply the configured world-frame velocity increment once per episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self._fired[env_ids if env_ids is not None else slice(None)] = False

    def __call__(self, env, env_ids, delta_velocity_world_m_s: tuple[float, float, float]):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        pending = env_ids[~self._fired[env_ids]]
        if pending.numel() == 0:
            return
        robot = env.scene["robot"]
        velocity = robot.data.root_vel_w[pending].clone()
        velocity[:, :3] += torch.tensor(delta_velocity_world_m_s, device=env.device)
        robot.write_root_velocity_to_sim(velocity, env_ids=pending)
        self._fired[pending] = True


class RandomPlanarVelocityKick(ScheduledVelocityKick):
    """One random-direction velocity increment per episode, at a sampled time.

    Isaac Lab's per-environment interval timer samples the time on each reset.
    The class-level fired flag suppresses later interval triggers in that episode.
    This is an impulse surrogate in m/s, not a calibrated force in newtons.
    """

    def __call__(self, env, env_ids, planar_delta_speed_range_m_s: tuple[float, float]):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        pending = env_ids[~self._fired[env_ids]]
        if pending.numel() == 0:
            return
        lower, upper = planar_delta_speed_range_m_s
        magnitudes = lower + (upper - lower) * torch.rand(pending.numel(), device=env.device)
        angles = 2 * math.pi * torch.rand(pending.numel(), device=env.device)
        velocity = env.scene["robot"].data.root_vel_w[pending].clone()
        velocity[:, 0] += magnitudes * torch.cos(angles)
        velocity[:, 1] += magnitudes * torch.sin(angles)
        env.scene["robot"].write_root_velocity_to_sim(velocity, env_ids=pending)
        self._fired[pending] = True
