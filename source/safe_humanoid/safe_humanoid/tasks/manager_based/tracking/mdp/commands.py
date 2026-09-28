"""Deterministic per-episode reference trajectory command."""

from __future__ import annotations

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from safe_humanoid.motion import load_motion


class ReferenceMotionCommand(CommandTerm):
    cfg: ReferenceMotionCommandCfg

    def __init__(self, cfg: ReferenceMotionCommandCfg, env):
        if not cfg.motion_file:
            raise ValueError("tracking task needs a retargeted G1 motion_file")
        super().__init__(cfg, env)
        robot = env.scene[cfg.asset_name]
        clip = load_motion(cfg.motion_file).in_robot_order(robot.data.joint_names)
        self.fps = clip.fps
        self.frame_count = clip.frame_count
        self._joint_pos = torch.tensor(clip.joint_pos, dtype=torch.float32, device=self.device)
        self._joint_vel = torch.tensor(clip.joint_vel, dtype=torch.float32, device=self.device)
        self._root_pos = torch.tensor(clip.root_pos_w, dtype=torch.float32, device=self.device)
        self._root_quat = torch.tensor(clip.root_quat_w, dtype=torch.float32, device=self.device)
        limits = robot.data.soft_joint_pos_limits[0]
        outside = ((self._joint_pos < limits[:, 0]) | (self._joint_pos > limits[:, 1])).any(dim=0)
        if outside.any().item():
            names = [name for index, name in enumerate(clip.joint_names) if outside[index].item()]
            raise ValueError(f"reference motion exceeds the G1 soft joint limits: {names}")

    @property
    def time_steps(self) -> torch.Tensor:
        # Isaac Lab computes rewards *before* command_manager.compute() each step.
        # Derive phase from the episode buffer so rewards and observations see
        # the same reference frame without a one-step command lag.
        elapsed = self._env.episode_length_buf * (self._env.cfg.decimation * self._env.cfg.sim.dt)
        return torch.clamp((elapsed * self.fps).long(), max=self.frame_count - 1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[self.time_steps]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._root_pos[self.time_steps]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._root_quat[self.time_steps]

    @property
    def command(self) -> torch.Tensor:
        # Joint positions and velocities are in the exact simulator order.
        return torch.cat(
            (self.joint_pos, self.joint_vel, self.root_pos_w[:, 2:3], self.root_quat_w), dim=1
        )

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids) -> None:
        pass

    def _update_command(self) -> None:
        pass


@configclass
class ReferenceMotionCommandCfg(CommandTermCfg):
    class_type: type = ReferenceMotionCommand
    asset_name: str = "robot"
    motion_file: str = ""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
