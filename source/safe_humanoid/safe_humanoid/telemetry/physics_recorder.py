"""Isaac Lab 2.3.2 recorder term for a deliberately small evaluation cohort.

This module is imported only after AppLauncher starts Isaac Sim. It does not
change the task's observations, rewards or policy action frequency.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.managers import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from safe_humanoid.telemetry.physics_trace import PhysicsSample, PhysicsTraceWriter


class PhysicsTraceTerm(RecorderTerm):
    """Capture post-step PhysX speed and pre-step implicit-PD torque estimate."""

    def __init__(self, cfg: PhysicsTraceTermCfg, env):
        super().__init__(cfg, env)
        if not 0 < env.num_envs <= cfg.max_envs or cfg.max_episodes_per_env <= 0:
            raise ValueError("physics recorder needs a small positive cohort and episode cap")
        if not cfg.output_dir:
            raise ValueError("physics recorder needs an output directory")
        self._robot = env.scene["robot"]
        self._writer = PhysicsTraceWriter(
            Path(cfg.output_dir), self._robot.data.joint_names, env.cfg.sim.dt, cfg.asset
        )
        self._episode_ids = [0] * env.num_envs
        self._steps = [0] * env.num_envs

    def record_post_physics_decimation_step(self):
        # Isaac Lab calls this after sim.step() but BEFORE scene.update(). The
        # cached robot.data.joint_vel would be one physics tick old here.
        speeds = self._robot.root_physx_view.get_dof_velocities().detach().cpu().tolist()
        torques = self._robot.data.applied_torque.detach().cpu().tolist()
        for env_id in range(self._env.num_envs):
            if self._episode_ids[env_id] >= self.cfg.max_episodes_per_env:
                continue
            self._steps[env_id] += 1
            step = self._steps[env_id]
            self._writer.record(PhysicsSample(
                env_id=env_id,
                episode_id=self._episode_ids[env_id],
                physics_step=step,
                time_s=step * self._writer.dt_s,
                estimated_actuator_torque_nm=tuple(torques[env_id]),
                joint_velocity_rad_s=tuple(speeds[env_id]),
            ))
        return None, None

    def record_pre_reset(self, env_ids):
        indices = range(self._env.num_envs) if env_ids is None else env_ids
        for env_id in indices:
            env_id = int(env_id)
            if self._steps[env_id] > 0:
                self._episode_ids[env_id] += 1
                self._steps[env_id] = 0
        return None, None

    def close(self, file_path: str):
        self._writer.close()


@configclass
class PhysicsTraceTermCfg(RecorderTermCfg):
    class_type: type = PhysicsTraceTerm
    output_dir: str = ""
    max_envs: int = 8
    max_episodes_per_env: int = 3
    asset: str = "Isaac Lab v2.3.2 G1_MINIMAL_CFG"


@configclass
class PhysicsRecorderManagerCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False
    physics_trace: PhysicsTraceTermCfg = PhysicsTraceTermCfg()
