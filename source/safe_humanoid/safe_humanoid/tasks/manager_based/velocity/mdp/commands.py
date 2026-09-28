"""Episode-relative scripted velocity commands for deterministic evaluation."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class ScheduledVelocityCommand(UniformVelocityCommand):
    """Use the same three-dimensional command contract as Isaac Lab's velocity task."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._schedule = torch.tensor(cfg.schedule, device=self.device, dtype=self.vel_command_b.dtype)

    def _resample_command(self, env_ids):
        # ManagerBasedRLEnv clears episode_length_buf after command reset. Start at t=0.
        self.vel_command_b[env_ids] = torch.tensor(
            self.cfg.schedule[0][1:], device=self.device, dtype=self.vel_command_b.dtype
        )
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False

    def _update_command(self):
        elapsed = self._env.episode_length_buf * self._env.step_dt
        index = torch.searchsorted(self._schedule[:, 0].contiguous(), elapsed.contiguous(), right=True) - 1
        self.vel_command_b[:] = self._schedule[index.clamp(min=0), 1:]


@configclass
class ScheduledVelocityCommandCfg(UniformVelocityCommandCfg):
    """Absolute episode times (s) followed by body-frame vx, vy (m/s), wz (rad/s)."""

    class_type: type = ScheduledVelocityCommand
    schedule: tuple[tuple[float, float, float, float], ...] = ((0.0, 0.0, 0.0, 0.0),)
