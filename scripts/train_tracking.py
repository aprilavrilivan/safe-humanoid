#!/usr/bin/env python3
"""Train the G1 joint/root motion-tracking pilot with RSL-RL PPO on Linux GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", required=True, choices=("squat_stand", "fast_leg_swing"))
parser.add_argument("--motion", required=True, type=Path)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-iterations", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.max_iterations <= 0:
    parser.error("--num-envs and --max-iterations must be positive")
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    import safe_humanoid  # noqa: F401
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

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
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(scenario.task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.max_iterations = args.max_iterations

    run_dir = ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + f"_{scenario.name}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run_dir)
    dump_yaml(str(run_dir / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "agent.yaml"), agent_cfg)
    (run_dir / "metadata.json").write_text(
        json.dumps({
            "task": scenario.task_id,
            "scenario": scenario.name,
            "scenario_sha256": sha256_file(scenario.source),
            "motion_file": str(clip.source),
            "motion_sha256": sha256_file(clip.source),
            "motion_frames": clip.frame_count,
            "motion_fps": clip.fps,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "isaac_lab_ref": "v2.3.2",
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    env = gym.make(scenario.task_id, cfg=env_cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device
        )
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    finally:
        env.close()
    print(f"[PASS] Training artifacts: {run_dir}")
finally:
    simulation_app.close()
