"""GPU-resident 200 Hz signal windows for online reward engineering.

The pinned Isaac Lab recorder callback runs after every physics step. This term
does not export a dataset or transfer rollout samples to the CPU. Its torque
signal is the implicit-PD *estimate* exposed by Isaac Lab, not motor telemetry.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass


class OnlineSafetyBuffer(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isclose(1.0 / env.cfg.sim.dt, cfg.sample_rate_hz, rel_tol=0, abs_tol=1e-6):
            raise ValueError("online safety reward requires the configured physics sample rate")
        if cfg.profile not in ("time", "frequency") or cfg.window_samples < 2:
            raise ValueError("online safety buffer needs a time or frequency profile")
        self._robot = env.scene["robot"]
        self._cursor = 0
        self._physics_ticks = 0
        self._last_fft_tick = 0
        self._counts = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        shape = (env.num_envs, len(self._robot.data.joint_names), cfg.window_samples)
        self._torques = torch.zeros(shape, device=env.device)
        self._speeds = torch.zeros(shape, device=env.device) if cfg.profile == "frequency" else None
        self._frequency_cost = torch.zeros(env.num_envs, device=env.device)
        if cfg.profile == "frequency":
            self._hann = torch.hann_window(cfg.window_samples, periodic=False, device=env.device)
            frequencies = torch.fft.rfftfreq(
                cfg.window_samples, d=1.0 / cfg.sample_rate_hz
            ).to(env.device)
            lower, upper = cfg.band_hz
            self._band_mask = (frequencies >= lower) & (
                (frequencies < upper) | ((frequencies == upper) & (upper == cfg.sample_rate_hz / 2))
            )
            self._one_sided = torch.full_like(frequencies, 2.0)
            self._one_sided[0] = 1.0
            self._one_sided[-1] = 1.0
        # RecorderManager is constructed before RewardManager in Isaac Lab 2.3.2.
        env.safe_humanoid_safety_buffer = self

    def reset(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self._counts[indices] = 0
        self._torques[indices] = 0
        if self._speeds is not None:
            self._speeds[indices] = 0
        self._frequency_cost[indices] = 0

    def record_post_physics_decimation_step(self):
        self._torques[:, :, self._cursor] = self._robot.data.applied_torque
        if self._speeds is not None:
            # The cached robot.data.joint_vel is updated *after* this callback.
            self._speeds[:, :, self._cursor] = self._robot.root_physx_view.get_dof_velocities()
        self._cursor = (self._cursor + 1) % self.cfg.window_samples
        self._physics_ticks += 1
        self._counts.add_(1).clamp_(max=self.cfg.window_samples)
        return None, None

    def time_cost(self) -> torch.Tensor:
        """Mean-joint squared rolling torque RMS, normalized by a soft reference."""

        denominator = self._counts.clamp(min=1).to(self._torques.dtype).unsqueeze(1)
        mean_square = self._torques.square().sum(dim=2) / denominator
        return mean_square.mean(dim=1) / self.cfg.torque_reference_nm**2

    def _band_power(self, history: torch.Tensor, reference: float) -> torch.Tensor:
        ordered = torch.roll(history, shifts=-self._cursor, dims=-1)
        centered = ordered - ordered.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered * self._hann, dim=-1)
        power = spectrum.abs().square() * self._one_sided
        power = power[..., self._band_mask].sum(dim=-1)
        return power.mean(dim=1) / (
            self.cfg.window_samples * self._hann.square().sum() * reference**2
        )

    def frequency_cost(self) -> torch.Tensor:
        """Held high-band Welch-like power; refreshed every configured hop."""

        if self._physics_ticks - self._last_fft_tick >= self.cfg.hop_samples:
            self._last_fft_tick = self._physics_ticks
            valid = self._counts >= self.cfg.window_samples
            self._frequency_cost[~valid] = 0
            if torch.any(valid):
                torque_power = self._band_power(
                    self._torques[valid], self.cfg.torque_reference_nm
                )
                speed_power = self._band_power(
                    self._speeds[valid], self.cfg.joint_speed_reference_rad_s
                )
                self._frequency_cost[valid] = 0.5 * (torque_power + speed_power)
        return self._frequency_cost


@configclass
class OnlineSafetyBufferCfg(RecorderTermCfg):
    class_type: type = OnlineSafetyBuffer
    profile: str = "time"
    sample_rate_hz: float = 200.0
    window_samples: int = 100
    hop_samples: int = 0
    band_hz: tuple[float, float] = (25.0, 100.0)
    torque_reference_nm: float = 50.0
    joint_speed_reference_rad_s: float = 10.0


@configclass
class OnlineSafetyRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False
    safety_buffer: OnlineSafetyBufferCfg = OnlineSafetyBufferCfg()
