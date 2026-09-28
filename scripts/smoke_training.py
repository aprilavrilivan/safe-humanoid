#!/usr/bin/env python3
"""GPU smoke for training-time command, push and online-reward paths."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scenario", choices=("nominal", "aggressive", "abrupt", "push_recovery"),
    required=True,
)
parser.add_argument("--reward-profile", choices=("clean", "time", "frequency"), required=True)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.steps <= 0:
    parser.error("--num-envs and --steps must be positive")
args.headless = True
simulation_app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import safe_humanoid  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from safe_humanoid.reward_profiles import load_reward_profile
    from safe_humanoid.scenarios import load_scenario
    from safe_humanoid.tasks.manager_based.velocity.config.g1 import CLEAN_TASK_ID
    from safe_humanoid.tasks.manager_based.velocity.reward_runtime import apply_reward_profile
    from safe_humanoid.tasks.manager_based.velocity.scenario_runtime import apply_scenario

    scenario = load_scenario(args.scenario, ROOT / "configs" / "task" / "velocity")
    profile = load_reward_profile(
        args.reward_profile, ROOT / "configs" / "experiments" / "velocity_reward_ppo.yaml"
    )
    cfg = parse_env_cfg(CLEAN_TASK_ID, device=args.device, num_envs=args.num_envs)
    apply_scenario(cfg, scenario, evaluation=False)
    apply_reward_profile(cfg, profile)
    cfg.seed = args.seed
    # Zero actions cause an untrained G1 to fall. This diagnostic must reach
    # the sampled command/push times without claiming policy competence.
    cfg.terminations.base_contact = None
    env = gym.make(CLEAN_TASK_ID, cfg=cfg)
    try:
        env.reset(seed=args.seed)
        initial_physics_ticks = (
            env.unwrapped.safe_humanoid_safety_buffer._physics_ticks
            if profile.name != "clean" else 0
        )
        previous_command = env.unwrapped.command_manager.get_command("base_velocity").clone()
        command_changes = 0
        actions = torch.zeros(
            (args.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )
        for _ in range(args.steps):
            _, rewards, _, _, _ = env.step(actions)
            if not torch.isfinite(rewards).all():
                raise AssertionError("training reward contains NaN or Inf")
            command = env.unwrapped.command_manager.get_command("base_velocity")
            command_changes += int(torch.any(command != previous_command).item())
            previous_command = command.clone()
        if scenario.name == "abrupt" and command_changes < 2:
            raise AssertionError("abrupt training did not expose repeated command changes")
        if scenario.training_push is not None:
            push = env.unwrapped.event_manager.get_term_cfg("scripted_push").func
            if not torch.all(push._fired):
                raise AssertionError("random training push did not fire for every environment")
        if profile.name != "clean":
            buffer = env.unwrapped.safe_humanoid_safety_buffer
            if int(buffer._counts.min().item()) < profile.window_samples:
                raise AssertionError("physics-rate reward window did not fill")
            cost = buffer.time_cost() if profile.name == "time" else buffer.frequency_cost()
            if not torch.isfinite(cost).all() or torch.any(cost < 0):
                raise AssertionError("online safety cost is not finite and nonnegative")
            if not torch.any(cost > 0):
                raise AssertionError(
                    "online safety cost stayed identically zero after a full window"
                )
            if not math.isclose(
                buffer._physics_ticks - initial_physics_ticks, args.steps * cfg.decimation
            ):
                raise AssertionError("safety buffer did not capture every physics step")
        print(
            f"[PASS] training scenario={scenario.name} reward={profile.name} "
            f"steps={args.steps} command_changes={command_changes}"
        )
    finally:
        env.close()
finally:
    simulation_app.close()
