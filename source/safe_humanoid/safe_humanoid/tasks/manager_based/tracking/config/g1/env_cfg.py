"""Reference-motion task over the Isaac Lab 2.3.2 flat G1 asset.

This is a modest joint/root tracking baseline, not a port of BeyondMimic's
body-state rewards, adaptive start-state curriculum, or custom G1 cylinder USD.
"""

from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp

from safe_humanoid.tasks.manager_based.tracking.mdp.commands import ReferenceMotionCommandCfg
from safe_humanoid.tasks.manager_based.tracking.mdp.events import reset_to_reference_motion
from safe_humanoid.tasks.manager_based.tracking.mdp import rewards as task_rewards


@configclass
class TrackingPolicyCfg(ObservationsCfg.PolicyCfg):
    velocity_commands: ObservationTermCfg | None = None
    motion_reference: ObservationTermCfg = ObservationTermCfg(
        func=base_mdp.generated_commands, params={"command_name": "motion"}
    )


@configclass
class TrackingObservationsCfg(ObservationsCfg):
    policy: TrackingPolicyCfg = TrackingPolicyCfg()


@configclass
class TrackingCommandsCfg:
    motion: ReferenceMotionCommandCfg = ReferenceMotionCommandCfg()


@configclass
class TrackingRewardsCfg:
    joint_pos = RewardTermCfg(
        func=task_rewards.track_joint_pos_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.35},
    )
    joint_vel = RewardTermCfg(
        func=task_rewards.track_joint_vel_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 2.0},
    )
    root_height = RewardTermCfg(
        func=task_rewards.track_root_height_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.12},
    )
    root_orientation = RewardTermCfg(
        func=task_rewards.track_root_orientation_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.5},
    )


@configclass
class SafeHumanoidG1TrackingEnvCfg(G1FlatEnvCfg):
    """Official flat scene/action/termination, with project tracking MDP terms."""

    def __post_init__(self):
        super().__post_init__()
        # A caller sets motion_file after parsing this config, before gym.make.
        self.commands = TrackingCommandsCfg()
        self.observations = TrackingObservationsCfg()
        self.observations.policy.height_scan = None
        self.rewards = TrackingRewardsCfg()
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_base = EventTermCfg(
            func=reset_to_reference_motion,
            mode="reset",
            params={"command_name": "motion"},
        )
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
