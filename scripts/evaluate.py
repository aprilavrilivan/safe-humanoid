#!/usr/bin/env python3
"""Replay one RSL-RL checkpoint on a scripted G1 velocity scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--scenario", choices=("nominal", "aggressive", "abrupt", "push_recovery"), required=True
)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--episodes", type=int, default=3, help="completed episodes per environment")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--no-physics-trace", action="store_true", help="omit 200 Hz safety trace")
parser.add_argument(
    "--wandb", action="store_true",
    help="append evaluation scalars to the training W&B run",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.episodes <= 0:
    parser.error("--num-envs and --episodes must be positive")
if args.num_envs > 8 and not args.no_physics_trace:
    parser.error("physics tracing is capped at 8 environments; use --no-physics-trace")
checkpoint = args.checkpoint.expanduser().resolve()
if not checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {checkpoint}")
train_metadata_path = checkpoint.parent / "metadata.json"
train_metadata = (
    json.loads(train_metadata_path.read_text(encoding="utf-8"))
    if train_metadata_path.is_file() else {}
)
if args.wandb and not all(train_metadata.get(field) for field in (
    "wandb_run_id", "wandb_entity", "wandb_project"
)):
    parser.error("--wandb needs a checkpoint whose training metadata contains a W&B run ID")
args.headless = True
launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    import safe_humanoid  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    from safe_humanoid.safety.analysis import analyze_run
    from safe_humanoid.safety.signals import velocity_sample
    from safe_humanoid.scenarios import load_scenario
    from safe_humanoid.tasks.manager_based.velocity.config.g1 import CLEAN_TASK_ID
    from safe_humanoid.tasks.manager_based.velocity.scenario_runtime import apply_scenario
    from safe_humanoid.telemetry.episode_writer import EvaluationWriter
    from safe_humanoid.telemetry.physics_recorder import PhysicsRecorderManagerCfg

    scenario = load_scenario(args.scenario, ROOT / "configs" / "task" / "velocity")
    env_cfg = parse_env_cfg(CLEAN_TASK_ID, device=args.device, num_envs=args.num_envs)
    apply_scenario(env_cfg, scenario, evaluation=True)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(CLEAN_TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    output_dir = args.output_dir or ROOT / "outputs" / "evaluation" / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + f"_{scenario.name}"
    )
    if not args.no_physics_trace:
        env_cfg.recorders = PhysicsRecorderManagerCfg()
        env_cfg.recorders.physics_trace.output_dir = str(output_dir)
        env_cfg.recorders.physics_trace.max_episodes_per_env = args.episodes
    metadata = {
        "task": CLEAN_TASK_ID,
        "scenario": scenario.name,
        "scenario_sha256": hashlib.sha256(scenario.source.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "episodes_per_env": args.episodes,
        "sample_timing": "state and command before step; reward and done from resulting transition",
        "push_semantics": "world-frame velocity increment in m/s, not a calibrated force",
        "isaac_lab_ref": "v2.3.2",
        "physics_trace_enabled": not args.no_physics_trace,
        "train_scenario": train_metadata.get("scenario"),
        "reward_profile": train_metadata.get("reward_profile", "clean"),
    }
    writer = EvaluationWriter(output_dir, metadata)
    env = None
    try:
        env = gym.make(CLEAN_TASK_ID, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
        observations = wrapped.get_observations()
        push_term = (
            wrapped.unwrapped.event_manager.get_term_cfg("scripted_push").func
            if scenario.push is not None else None
        )
        completed = [0] * args.num_envs
        with torch.inference_mode():
            while simulation_app.is_running() and any(count < args.episodes for count in completed):
                robot = wrapped.unwrapped.scene["robot"]
                command = (
                    wrapped.unwrapped.command_manager.get_command("base_velocity").cpu().tolist()
                )
                linear = robot.data.root_lin_vel_b.cpu().tolist()
                angular = robot.data.root_ang_vel_b.cpu().tolist()
                gravity = robot.data.projected_gravity_b.cpu().tolist()
                torque = robot.data.applied_torque.cpu().tolist()
                joint_speed = robot.data.joint_vel.cpu().tolist()
                elapsed = (
                    wrapped.unwrapped.episode_length_buf * wrapped.unwrapped.step_dt
                ).cpu().tolist()
                push_before = push_term._fired.clone() if push_term is not None else None
                actions = policy(observations)
                observations, rewards, dones, _ = wrapped.step(actions)
                push_fired = (
                    (push_term._fired & ~push_before).cpu().tolist()
                    if push_term is not None else [False] * args.num_envs
                )
                reward_values = rewards.cpu().tolist()
                terminated = wrapped.unwrapped.reset_terminated.cpu().tolist()
                truncated = wrapped.unwrapped.reset_time_outs.cpu().tolist()
                for env_id in range(args.num_envs):
                    if completed[env_id] >= args.episodes:
                        continue
                    metrics = velocity_sample(
                        command[env_id], linear[env_id], angular[env_id], gravity[env_id],
                        torque[env_id], joint_speed[env_id],
                    )
                    writer.record({
                        "env_id": env_id,
                        "episode_id": completed[env_id],
                        "time_s": elapsed[env_id],
                        "command_vx_m_s": command[env_id][0],
                        "command_vy_m_s": command[env_id][1],
                        "command_wz_rad_s": command[env_id][2],
                        **metrics,
                        "transition_reward": reward_values[env_id],
                        "terminated": terminated[env_id],
                        "truncated": truncated[env_id],
                        "push_fired": push_fired[env_id],
                    })
                    if terminated[env_id] or truncated[env_id]:
                        completed[env_id] += 1
                runner.alg.policy.reset(dones)
        if any(count < args.episodes for count in completed):
            raise RuntimeError(f"evaluation stopped before all episodes completed: {completed}")
    finally:
        if env is not None:
            env.close()
        summary = writer.close()
    if not args.no_physics_trace:
        safety_summary = analyze_run(output_dir, ROOT / "configs" / "safety")
        print(
            "[INFO] Physics-rate safety analysis: "
            f"{safety_summary['temporal_aggregate']['physics_samples']} samples; "
            f"hardware limits {safety_summary['hardware_limit_status']}"
        )
    print(f"[PASS] Evaluation artifacts: {output_dir}; summary={summary}")
finally:
    simulation_app.close()

if args.wandb:
    import wandb

    with wandb.init(
        entity=train_metadata["wandb_entity"],
        project=train_metadata["wandb_project"],
        id=train_metadata["wandb_run_id"],
        resume="must",
        mode="online",
    ) as run:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                run.summary[f"eval/{scenario.name}/{key}"] = value
        if not args.no_physics_trace:
            for group in ("temporal_aggregate", "frequency_aggregate"):
                for key, value in safety_summary.get(group, {}).items():
                    if isinstance(value, (int, float)):
                        run.summary[f"eval/{scenario.name}/{group}/{key}"] = value
        for filename in ("metadata.json", "summary.json", "safety_summary.json"):
            artifact = output_dir / filename
            if artifact.is_file():
                wandb.save(str(artifact), base_path=str(output_dir))
