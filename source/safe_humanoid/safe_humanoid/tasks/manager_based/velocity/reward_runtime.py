"""Apply one online safety-reward profile without changing policy I/O."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg

from safe_humanoid.reward_profiles import RewardProfile
from safe_humanoid.tasks.manager_based.velocity.mdp import rewards
from safe_humanoid.telemetry.online_safety import OnlineSafetyRecorderCfg


def apply_reward_profile(env_cfg, profile: RewardProfile) -> None:
    if profile.name == "clean":
        return
    env_cfg.recorders = OnlineSafetyRecorderCfg()
    buffer_cfg = env_cfg.recorders.safety_buffer
    buffer_cfg.profile = profile.name
    buffer_cfg.sample_rate_hz = profile.sample_rate_hz
    buffer_cfg.window_samples = profile.window_samples
    buffer_cfg.hop_samples = profile.hop_samples
    buffer_cfg.torque_reference_nm = profile.torque_reference_nm
    if profile.band_hz is not None:
        buffer_cfg.band_hz = profile.band_hz
        buffer_cfg.joint_speed_reference_rad_s = profile.joint_speed_reference_rad_s
    env_cfg.rewards.safety_cost = RewardTermCfg(
        func=(
            rewards.rolling_torque_rms_cost
            if profile.name == "time" else rewards.high_band_motion_cost
        ),
        weight=profile.weight,
    )
