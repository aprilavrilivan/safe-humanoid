#!/usr/bin/env python3
"""Reset and step the project G1 environment while checking its tensor contract."""

from __future__ import annotations

import argparse
import math
import traceback
from collections.abc import Mapping
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_TASK_ID = "SafeHumanoid-Velocity-Flat-G1-v0"
DEFAULT_CLEAN_TASK_ID = "SafeHumanoid-Velocity-Clean-Flat-G1-v0"
DEFAULT_REFERENCE_TASK_ID = "Isaac-Velocity-Flat-G1-v0"


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


parser = argparse.ArgumentParser(
    description=(
        "Create the project-owned G1 environment, run deterministic zero actions, "
        "and validate its contract against the official Isaac Lab task."
    )
)
parser.add_argument("--task", default=None, help="project task ID; inferred from --scenario")
parser.add_argument(
    "--scenario",
    choices=("nominal", "aggressive", "abrupt", "push_recovery"),
    default=None,
    help="validate a clean-reward scenario instead of the untouched upstream clone",
)
parser.add_argument(
    "--reference-task",
    default=DEFAULT_REFERENCE_TASK_ID,
    help="official task whose configuration contract must be preserved",
)
parser.add_argument(
    "--num-envs",
    "--num_envs",
    dest="num_envs",
    type=positive_int,
    default=4,
    help="number of parallel environments (default: 4)",
)
parser.add_argument(
    "--steps",
    type=positive_int,
    default=32,
    help="number of zero-action environment steps (default: 32)",
)
parser.add_argument("--seed", type=int, default=0, help="deterministic reset seed")
parser.add_argument(
    "--exercise-timing",
    action="store_true",
    help=(
        "disable contact termination during a scenario smoke test so its full "
        "command schedule and optional push can be reached"
    ),
)
parser.add_argument(
    "--disable-fabric",
    "--disable_fabric",
    dest="disable_fabric",
    action="store_true",
    help="use USD I/O rather than Fabric",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.exercise_timing and args_cli.scenario is None:
    parser.error("--exercise-timing requires --scenario")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim must be running before the remaining simulator modules are imported.
try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import safe_humanoid  # noqa: F401
    from safe_humanoid.scenarios import load_scenario
    from safe_humanoid.tasks.manager_based.velocity.scenario_runtime import apply_scenario
    from isaaclab.managers import (
        ActionTermCfg,
        CommandTermCfg,
        CurriculumTermCfg,
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )
    from isaaclab_tasks.utils import parse_env_cfg
except Exception:
    simulation_app.close()
    raise


def pass_message(message: str) -> None:
    """Print one successful smoke-test checkpoint."""

    print(f"[PASS] {message}")


def active_terms(config: object, term_type: type) -> tuple[str, ...]:
    """Return ordered fields containing active terms of one manager type."""

    return tuple(
        name
        for name, value in vars(config).items()
        if not name.startswith("_") and isinstance(value, term_type)
    )


def observation_signature(config: object) -> dict[str, tuple[str, ...]]:
    """Describe observation groups and their ordered active terms."""

    return {
        group_name: active_terms(group_cfg, ObservationTermCfg)
        for group_name, group_cfg in vars(config.observations).items()
        if isinstance(group_cfg, ObservationGroupCfg)
    }


def manager_config_signature(config: object) -> dict[str, object]:
    """Capture the MDP surface that a scenario is not allowed to change yet."""

    return {
        "commands": active_terms(config.commands, CommandTermCfg),
        "actions": active_terms(config.actions, ActionTermCfg),
        "observations": observation_signature(config),
        "rewards": active_terms(config.rewards, RewardTermCfg),
        "terminations": active_terms(config.terminations, TerminationTermCfg),
        "events": active_terms(config.events, EventTermCfg),
        "curriculum": active_terms(config.curriculum, CurriculumTermCfg),
    }


def assert_same_baseline_contract(project_cfg: object, reference_cfg: object) -> None:
    """Check that the project config is still a pass-through official G1 config."""

    assert isinstance(project_cfg, type(reference_cfg)), (
        f"{type(project_cfg).__name__} must inherit {type(reference_cfg).__name__}"
    )

    scalar_fields = (
        ("simulation dt", project_cfg.sim.dt, reference_cfg.sim.dt),
        ("control decimation", project_cfg.decimation, reference_cfg.decimation),
        ("episode length", project_cfg.episode_length_s, reference_cfg.episode_length_s),
        ("number of environments", project_cfg.scene.num_envs, reference_cfg.scene.num_envs),
    )
    for name, project_value, reference_value in scalar_fields:
        if isinstance(project_value, float) or isinstance(reference_value, float):
            equal = math.isclose(float(project_value), float(reference_value))
        else:
            equal = project_value == reference_value
        assert equal, f"{name} differs: project={project_value}, official={reference_value}"

    project_signature = manager_config_signature(project_cfg)
    reference_signature = manager_config_signature(reference_cfg)
    assert project_signature == reference_signature, (
        "project manager-term contract differs from the official G1 baseline:\n"
        f"project={project_signature}\nofficial={reference_signature}"
    )


def assert_compatible_tensor_contract(project_cfg: object, reference_cfg: object) -> None:
    """Allow deliberate reward/event changes but preserve policy I/O and timing."""

    project = manager_config_signature(project_cfg)
    reference = manager_config_signature(reference_cfg)
    for name in ("commands", "actions", "observations", "terminations"):
        assert project[name] == reference[name], f"{name} changed relative to official G1"
    assert math.isclose(project_cfg.sim.dt, reference_cfg.sim.dt)
    assert project_cfg.decimation == reference_cfg.decimation
    assert math.isclose(project_cfg.episode_length_s, reference_cfg.episode_length_s)


def tensor_leaves(value: object, path: str = "value"):
    """Yield ``(path, tensor)`` leaves from nested observation structures."""

    if torch.is_tensor(value):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            yield from tensor_leaves(nested_value, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            yield from tensor_leaves(nested_value, f"{path}[{index}]")
        return
    raise AssertionError(f"{path} contains unsupported value type {type(value).__name__}")


def assert_tensor_batch(
    value: object,
    *,
    name: str,
    num_envs: int,
    device: torch.device,
) -> None:
    """Validate batch size, device, and finite values for all tensor leaves."""

    leaves = list(tensor_leaves(value, name))
    assert leaves, f"{name} contains no tensors"
    for path, tensor in leaves:
        assert tensor.ndim >= 1, f"{path} must include an environment batch dimension"
        assert tensor.shape[0] == num_envs, (
            f"{path} batch is {tensor.shape[0]}, expected {num_envs}"
        )
        assert tensor.device == device, f"{path} is on {tensor.device}, expected {device}"
        assert torch.isfinite(tensor).all().item(), f"{path} contains NaN or Inf"


def run_smoke_test() -> None:
    """Run registration, configuration, reset, and step checks."""

    task_id = args_cli.task or (DEFAULT_CLEAN_TASK_ID if args_cli.scenario else DEFAULT_TASK_ID)
    project_spec = gym.spec(task_id)
    reference_spec = gym.spec(args_cli.reference_task)
    assert "env_cfg_entry_point" in project_spec.kwargs
    assert "env_cfg_entry_point" in reference_spec.kwargs
    pass_message(f"registered task: {task_id}")

    project_cfg = parse_env_cfg(
        task_id,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    reference_cfg = parse_env_cfg(
        args_cli.reference_task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.scenario:
        assert task_id == DEFAULT_CLEAN_TASK_ID, (
            "--scenario requires the clean G1 task; omit --task or use "
            f"{DEFAULT_CLEAN_TASK_ID}"
        )
        scenario = load_scenario(
            args_cli.scenario, Path(__file__).resolve().parents[1] / "configs" / "task" / "velocity"
        )
        apply_scenario(project_cfg, scenario, evaluation=True)
        assert_compatible_tensor_contract(project_cfg, reference_cfg)
        assert active_terms(project_cfg.rewards, RewardTermCfg) == (
            "r_vx", "r_vy", "r_wz", "r_upright"
        )
        assert (project_cfg.events.scripted_push is not None) == (scenario.push is not None)
        pass_message(f"policy I/O contract preserved for scenario: {scenario.name}")
        if args_cli.exercise_timing:
            # Zero actions normally cause a fall and reset before the scripted
            # event times. This diagnostic keeps one episode alive to exercise
            # the clock; it is not a policy-performance evaluation.
            project_cfg.terminations.base_contact = None
            latest_time_s = max(point.time_s for point in scenario.evaluation)
            if scenario.push is not None:
                latest_time_s = max(latest_time_s, scenario.push.time_s)
            available_time_s = args_cli.steps * project_cfg.decimation * project_cfg.sim.dt
            assert available_time_s > latest_time_s, (
                f"{args_cli.steps} steps reach only {available_time_s:g}s; "
                f"scenario's last event is at {latest_time_s:g}s"
            )
    else:
        assert_same_baseline_contract(project_cfg, reference_cfg)
        pass_message(f"configuration contract matches: {args_cli.reference_task}")

    env = None
    try:
        env = gym.make(task_id, cfg=project_cfg)
        unwrapped = env.unwrapped
        expected_device = torch.device(unwrapped.device)
        expected_action_shape = (
            args_cli.num_envs,
            unwrapped.action_manager.total_action_dim,
        )
        assert env.action_space.shape == expected_action_shape, (
            f"action space is {env.action_space.shape}, expected {expected_action_shape}"
        )
        assert tuple(unwrapped.action_manager.active_terms) == active_terms(
            project_cfg.actions, ActionTermCfg
        )
        assert unwrapped.observation_manager.active_terms == {
            group: list(terms)
            for group, terms in observation_signature(project_cfg).items()
        }
        pass_message(
            "environment created with "
            f"observation_space={env.observation_space} action_space={env.action_space}"
        )

        observations, info = env.reset(seed=args_cli.seed)
        assert isinstance(info, Mapping), f"reset info must be a mapping, got {type(info).__name__}"
        assert_tensor_batch(
            observations,
            name="reset observations",
            num_envs=args_cli.num_envs,
            device=expected_device,
        )
        pass_message("reset observation tensors are finite and correctly batched")
        if args_cli.scenario:
            command = unwrapped.command_manager.get_command("base_velocity")
            expected = torch.tensor(scenario.command_at(0.0), device=expected_device)
            assert torch.allclose(command, expected.expand_as(command)), (
                f"initial command differs from {scenario.name} schedule"
            )

        actions = torch.zeros(
            env.action_space.shape,
            dtype=torch.float32,
            device=expected_device,
        )
        with torch.inference_mode():
            for step_index in range(1, args_cli.steps + 1):
                observations, rewards, terminated, truncated, info = env.step(actions)
                if args_cli.exercise_timing:
                    assert not torch.any(terminated | truncated).item(), (
                        f"step {step_index}: episode reset before the scripted timing check"
                    )
                assert isinstance(info, Mapping), (
                    f"step {step_index}: info must be a mapping, got {type(info).__name__}"
                )
                assert_tensor_batch(
                    observations,
                    name=f"step {step_index} observations",
                    num_envs=args_cli.num_envs,
                    device=expected_device,
                )
                for name, tensor in (
                    ("rewards", rewards),
                    ("terminated", terminated),
                    ("truncated", truncated),
                ):
                    assert torch.is_tensor(tensor), f"step {step_index}: {name} is not a tensor"
                    assert tensor.shape == (args_cli.num_envs,), (
                        f"step {step_index}: {name} has shape {tensor.shape}"
                    )
                    assert tensor.device == expected_device, (
                        f"step {step_index}: {name} is on {tensor.device}, "
                        f"expected {expected_device}"
                    )
                assert terminated.dtype == torch.bool
                assert truncated.dtype == torch.bool
                assert torch.isfinite(rewards).all().item(), (
                    f"step {step_index}: rewards contain NaN or Inf"
                )
                if args_cli.scenario:
                    command = unwrapped.command_manager.get_command("base_velocity")
                    elapsed = (unwrapped.episode_length_buf * unwrapped.step_dt).cpu().tolist()
                    expected = torch.tensor(
                        [scenario.command_at(time_s) for time_s in elapsed],
                        device=expected_device,
                        dtype=command.dtype,
                    )
                    assert torch.allclose(command, expected), (
                        f"step {step_index}: command differs from {scenario.name} schedule"
                    )

        if args_cli.exercise_timing:
            if scenario.push is not None:
                push_term = unwrapped.event_manager.get_term_cfg("scripted_push").func
                assert torch.all(push_term._fired).item(), "scheduled push never fired"
            pass_message(f"scripted timing reached {latest_time_s:g}s without an episode reset")

        pass_message(f"completed {args_cli.steps} zero-action steps with a valid tensor contract")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        run_smoke_test()
    except BaseException:
        # Kit shutdown may terminate the process before Python prints the
        # original traceback. Emit it first so a failed smoke test cannot
        # appear to have passed just because shutdown returned exit code 0.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
