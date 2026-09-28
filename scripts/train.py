#!/usr/bin/env python3
"""Train the clean G1 velocity task with Isaac Lab's RSL-RL PPO integration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", default="nominal", choices=(
    "nominal", "aggressive", "abrupt", "push_recovery"
))
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument(
    "--run-dir", type=Path, default=None,
    help="explicit new directory for reproducible experiment batches",
)
parser.add_argument("--reward-profile", choices=("clean", "time", "frequency"), default="clean")
parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
parser.add_argument("--wandb-entity", default="ivanxu-uc-berkeley")
parser.add_argument("--wandb-project", default="safe-humanoid")
parser.add_argument(
    "--wandb-offline", action="store_true", help="write a local W&B run for later sync"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or (args.max_iterations is not None and args.max_iterations <= 0):
    parser.error("--num-envs and --max-iterations must be positive")
if args.wandb_offline and args.logger != "wandb":
    parser.error("--wandb-offline requires --logger wandb")
if args.logger == "wandb" and importlib.util.find_spec("wandb") is None:
    parser.error("wandb is not installed; rerun scripts/setup/bootstrap_linux.sh")
if args.logger == "wandb" and not args.wandb_offline and not os.environ.get("WANDB_API_KEY"):
    parser.error("online W&B logging requires WANDB_API_KEY in the environment")
if args.logger == "wandb":
    # RSL-RL 3.0.1 reads WANDB_USERNAME as the W&B entity/team.
    os.environ["WANDB_USERNAME"] = args.wandb_entity
    os.environ["WANDB_MODE"] = "offline" if args.wandb_offline else "online"
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

    from safe_humanoid.scenarios import load_scenario
    from safe_humanoid.reward_profiles import load_reward_profile
    from safe_humanoid.tasks.manager_based.velocity.config.g1 import CLEAN_TASK_ID
    from safe_humanoid.tasks.manager_based.velocity.reward_runtime import apply_reward_profile
    from safe_humanoid.tasks.manager_based.velocity.scenario_runtime import apply_scenario

    scenario = load_scenario(args.scenario, ROOT / "configs" / "task" / "velocity")
    profile = load_reward_profile(
        args.reward_profile, ROOT / "configs" / "experiments" / "velocity_reward_ppo.yaml"
    )
    env_cfg = parse_env_cfg(CLEAN_TASK_ID, device=args.device, num_envs=args.num_envs)
    apply_scenario(env_cfg, scenario, evaluation=False)
    apply_reward_profile(env_cfg, profile)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(CLEAN_TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    agent_cfg.logger = args.logger
    if args.logger == "wandb":
        agent_cfg.wandb_project = args.wandb_project
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    run_dir = args.run_dir or ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        + f"_{scenario.name}_{profile.name}_seed{args.seed}"
    )
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run_dir)
    dump_yaml(str(run_dir / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "agent.yaml"), agent_cfg)
    metadata = {
        "task": CLEAN_TASK_ID,
        "scenario": scenario.name,
        "scenario_file": str(scenario.source),
        "scenario_sha256": hashlib.sha256(scenario.source.read_bytes()).hexdigest(),
        "reward_profile": profile.name,
        "reward_profile_sha256": hashlib.sha256(profile.source.read_bytes()).hexdigest(),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_iterations": agent_cfg.max_iterations,
        "logger": args.logger,
        "wandb_entity": args.wandb_entity if args.logger == "wandb" else None,
        "wandb_project": args.wandb_project if args.logger == "wandb" else None,
        "isaac_lab_ref": "v2.3.2",
    }
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.logger == "wandb":
        os.environ["WANDB_DIR"] = str(run_dir)

    env = gym.make(CLEAN_TASK_ID, cfg=env_cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device
        )
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        if args.logger == "wandb":
            import wandb

            if wandb.run is None:
                raise RuntimeError("RSL-RL did not initialize a W&B run")
            metadata["wandb_run_id"] = wandb.run.id
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            wandb.config.update(metadata, allow_val_change=True)
            wandb.run.tags = tuple(dict.fromkeys((
                *(wandb.run.tags or ()), "ppo", f"scenario:{scenario.name}",
                f"reward:{profile.name}",
            )))
            wandb.save(str(metadata_path), base_path=str(run_dir))
    finally:
        env.close()
        if args.logger == "wandb":
            import wandb

            if wandb.run is not None:
                wandb.finish()
    print(f"[PASS] Training artifacts: {run_dir}")
finally:
    simulation_app.close()
