"""Preserved upstream baseline and a separate clean-reward G1 configuration."""

from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from safe_humanoid.tasks.manager_based.velocity.mdp import rewards as task_rewards


@configclass
class CleanVelocityRewardsCfg:
    """Only vx, vy, yaw-rate, and uprightness; no hidden safety penalty."""

    r_vx = RewardTermCfg(
        func=task_rewards.track_velocity_axis_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "axis": 0, "std": 0.5},
    )
    r_vy = RewardTermCfg(
        func=task_rewards.track_velocity_axis_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "axis": 1, "std": 0.5},
    )
    r_wz = RewardTermCfg(
        func=task_rewards.track_velocity_axis_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "axis": 2, "std": 0.5},
    )
    r_upright = RewardTermCfg(func=task_rewards.upright_alignment, weight=1.0)
    safety_cost: RewardTermCfg | None = None


@configclass
class CleanVelocityEventsCfg(EventCfg):
    """Declare the optional scripted perturbation as a proper manager field."""

    scripted_push: EventTermCfg | None = None


@configclass
class SafeHumanoidG1FlatEnvCfg(G1FlatEnvCfg):
    """Unmodified G1 flat-terrain baseline used to validate the extension boundary.

    Scenario overlays will be added only after this class is shown to have the same
    observation, action, reward, termination, and timing contract as the upstream task.
    """

    pass


@configclass
class SafeHumanoidG1CleanEnvCfg(G1FlatEnvCfg):
    """G1 flat locomotion with a task-only reward and no surprise pushes."""

    events: CleanVelocityEventsCfg = CleanVelocityEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # G1FlatEnvCfg.__post_init__ edits inherited reward fields, so replace
        # them only after the upstream G1 configuration has completed.
        self.rewards = CleanVelocityRewardsCfg()
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.debug_vis = False
