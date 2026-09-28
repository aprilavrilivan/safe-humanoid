#!/usr/bin/env python3
"""Create and step a G1 motion-tracking task on a Linux GPU host."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", required=True, choices=("squat_stand", "fast_leg_swing"))
parser.add_argument("--motion", required=True, type=Path)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.steps <= 0:
    parser.error("--num-envs and --steps must be positive")
launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import safe_humanoid  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from safe_humanoid.motion import load_motion
    from safe_humanoid.tracking_scenarios import load_tracking_scenario
    from safe_humanoid.tasks.manager_based.tracking.scenario_runtime import (
        apply_tracking_scenario,
    )

    scenario = load_tracking_scenario(args.scenario, ROOT / "configs" / "task" / "tracking")
    clip = load_motion(args.motion)
    scenario.check_motion(clip)
    env_cfg = parse_env_cfg(scenario.task_id, device=args.device, num_envs=args.num_envs)
    apply_tracking_scenario(env_cfg, scenario, clip)
    env = gym.make(scenario.task_id, cfg=env_cfg)
    try:
        observations, _ = env.reset(seed=0)
        command = env.unwrapped.command_manager.get_term("motion")
        robot = env.unwrapped.scene["robot"]
        if not torch.allclose(
            robot.data.joint_pos,
            command._joint_pos[0].expand(args.num_envs, -1),
            atol=1.0e-3,
        ):
            raise AssertionError("reset joint positions differ from reference frame 0")
        if not torch.allclose(
            robot.data.root_pos_w[:, 2],
            command._root_pos[0, 2].expand(args.num_envs),
            atol=1.0e-3,
        ):
            raise AssertionError("reset root heights differ from reference frame 0")
        action_dim = env.unwrapped.action_manager.total_action_dim
        action = torch.zeros((args.num_envs, action_dim), device=env.unwrapped.device)
        for step in range(args.steps):
            policy = observations["policy"]
            if policy.shape[0] != args.num_envs or not torch.isfinite(policy).all().item():
                raise AssertionError(
                    f"non-finite or wrong-sized policy observations at step {step}"
                )
            observations, reward, terminated, truncated, _ = env.step(action)
            if not torch.isfinite(reward).all().item():
                raise AssertionError(f"non-finite rewards at step {step}")
            if terminated.shape != (args.num_envs,) or truncated.shape != (args.num_envs,):
                raise AssertionError("invalid termination tensor shape")
        print(f"[PASS] {scenario.task_id}: reset + {args.steps} zero-action steps")
        print("[NOTE] Smoke success checks mechanics, not tracking quality or training success.")
    finally:
        env.close()
finally:
    simulation_app.close()
