"""Translate validated, simulator-independent scenarios into Isaac Lab configs."""

from __future__ import annotations

from copy import deepcopy

from isaaclab.managers import EventTermCfg

from safe_humanoid.scenarios import ScenarioSpec
from safe_humanoid.tasks.manager_based.velocity.mdp.commands import ScheduledVelocityCommandCfg
from safe_humanoid.tasks.manager_based.velocity.mdp.events import (
    RandomPlanarVelocityKick,
    ScheduledVelocityKick,
)


def apply_scenario(env_cfg, scenario: ScenarioSpec, *, evaluation: bool) -> None:
    """Modify a fresh clean G1 config before the Isaac environment is created."""

    env_cfg.episode_length_s = scenario.episode_length_s
    command_cfg = env_cfg.commands.base_velocity
    if evaluation:
        schedule = tuple((point.time_s, *point.command) for point in scenario.evaluation)
        never_resample_s = scenario.episode_length_s + 1.0
        env_cfg.commands.base_velocity = ScheduledVelocityCommandCfg(
            asset_name="robot",
            ranges=deepcopy(command_cfg.ranges),
            resampling_time_range=(never_resample_s, never_resample_s),
            heading_command=False,
            rel_heading_envs=0.0,
            rel_standing_envs=0.0,
            debug_vis=False,
            schedule=schedule,
        )
        env_cfg.observations.policy.enable_corruption = False
        # With a fixed yaw, a world-frame push direction has the same meaning
        # for every episode. The official G1 reset already sets velocity to zero.
        for axis in ("x", "y", "yaw"):
            env_cfg.events.reset_base.params["pose_range"][axis] = (0.0, 0.0)
        if scenario.push is not None:
            env_cfg.events.scripted_push = EventTermCfg(
                func=ScheduledVelocityKick,
                mode="interval",
                interval_range_s=(scenario.push.time_s, scenario.push.time_s),
                is_global_time=False,
                params={"delta_velocity_world_m_s": scenario.push.delta_velocity_world_m_s},
            )
        else:
            env_cfg.events.scripted_push = None
    else:
        ranges = scenario.training
        command_cfg.ranges.lin_vel_x = ranges.lin_vel_x
        command_cfg.ranges.lin_vel_y = ranges.lin_vel_y
        command_cfg.ranges.ang_vel_z = ranges.ang_vel_z
        command_cfg.resampling_time_range = ranges.resampling_time_range_s
        command_cfg.heading_command = False
        command_cfg.rel_heading_envs = 0.0
        command_cfg.rel_standing_envs = 0.0
        command_cfg.ranges.heading = None
        if scenario.training_push is not None:
            push = scenario.training_push
            env_cfg.events.scripted_push = EventTermCfg(
                func=RandomPlanarVelocityKick,
                mode="interval",
                interval_range_s=push.time_range_s,
                is_global_time=False,
                params={"planar_delta_speed_range_m_s": push.planar_delta_speed_range_m_s},
            )
        else:
            env_cfg.events.scripted_push = None
