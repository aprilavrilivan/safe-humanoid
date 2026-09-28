#!/usr/bin/env python3
"""Capture and analyze a short G1 physics trace without a trained policy."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument(
    "--steps", type=int, default=64, help="policy steps; each contains four physics steps"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 < args.num_envs <= 8 or args.steps <= 0:
    parser.error("--num-envs must be 1..8 and --steps must be positive")
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import safe_humanoid  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from safe_humanoid.safety.analysis import analyze_run
    from safe_humanoid.tasks.manager_based.velocity.config.g1 import CLEAN_TASK_ID
    from safe_humanoid.telemetry.physics_recorder import PhysicsRecorderManagerCfg
    from safe_humanoid.telemetry.physics_trace import read_physics_trace

    output_dir = args.output_dir or ROOT / "outputs" / "smoke_safety" / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-%f"
    )
    env_cfg = parse_env_cfg(CLEAN_TASK_ID, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.recorders = PhysicsRecorderManagerCfg()
    env_cfg.recorders.physics_trace.output_dir = str(output_dir)
    # A zero-action robot may reset often; the smoke test must still record every step.
    env_cfg.recorders.physics_trace.max_episodes_per_env = args.steps + 1
    env = None
    try:
        env = gym.make(CLEAN_TASK_ID, cfg=env_cfg)
        env.reset(seed=args.seed)
        actions = torch.zeros(
            env.action_space.shape, dtype=torch.float32, device=env.unwrapped.device
        )
        with torch.inference_mode():
            for _ in range(args.steps):
                env.step(actions)
    finally:
        if env is not None:
            env.close()
    metadata, samples = read_physics_trace(output_dir)
    expected_samples = args.num_envs * args.steps * env_cfg.decimation
    if len(samples) != expected_samples:
        raise AssertionError(
            f"recorded {len(samples)} physics samples, expected {expected_samples}"
        )
    if metadata["sample_rate_hz"] != 200.0 or metadata["torque_is_measured"] is not False:
        raise AssertionError(f"unexpected trace metadata: {metadata}")
    summary = analyze_run(output_dir, ROOT / "configs" / "safety")
    if summary["hardware_limit_status"] != "uncalibrated":
        raise AssertionError("default hardware limits must remain uncalibrated")
    print(
        f"[PASS] {len(samples)} physics samples at {metadata['sample_rate_hz']:g} Hz; "
        f"offline summary: {output_dir / 'safety_summary.json'}"
    )
finally:
    simulation_app.close()
