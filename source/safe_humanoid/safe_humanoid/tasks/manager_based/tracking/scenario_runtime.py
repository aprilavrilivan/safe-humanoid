"""Apply a validated tracking study to a fresh Isaac Lab environment config."""

from safe_humanoid.motion import MotionClip
from safe_humanoid.tracking_scenarios import TrackingScenario


def apply_tracking_scenario(env_cfg, scenario: TrackingScenario, clip: MotionClip) -> None:
    scenario.check_motion(clip)
    env_cfg.commands.motion.motion_file = str(clip.source)
    env_cfg.episode_length_s = clip.duration_s
    env_cfg.observations.policy.enable_corruption = False
    rewards = env_cfg.rewards
    rewards.joint_pos.params["std"] = scenario.joint_pos_std_rad
    rewards.joint_vel.params["std"] = scenario.joint_vel_std_rad_s
    rewards.root_height.params["std"] = scenario.root_height_std_m
    rewards.root_orientation.params["std"] = scenario.root_orientation_std_rad
