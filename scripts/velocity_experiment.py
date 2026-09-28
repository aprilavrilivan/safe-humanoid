#!/usr/bin/env python3
"""Plan and audit a four-scenario G1 velocity pilot without Isaac Sim.

This checks whether the *data* are comparable and include scripted events. It
does not establish policy competence, statistical significance, or safety.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "experiments" / "velocity_baseline_pilot.yaml"
SCENARIOS = ("nominal", "aggressive", "abrupt", "push_recovery")
TASK_ID = "SafeHumanoid-Velocity-Clean-Flat-G1-v0"
TRACE_FIELDS = (
    "env_id", "episode_id", "time_s", "command_vx_m_s", "command_vy_m_s",
    "command_wz_rad_s", "tracking_error_xy_m_s", "tracking_error_yaw_rad_s",
    "upright_alignment", "joint_torque_rms_nm", "joint_torque_peak_nm",
    "joint_speed_peak_rad_s", "absolute_mechanical_power_w", "transition_reward",
    "terminated", "truncated", "push_fired",
)
PHYSICS_FIELDS = (
    "env_id", "episode_id", "physics_step", "time_s",
    "estimated_actuator_torque_nm", "joint_velocity_rad_s",
)


def _scenarios_module():
    """Load the pure-data module without importing Gym or Isaac Lab registration."""

    path = ROOT / "source" / "safe_humanoid" / "safe_humanoid" / "scenarios.py"
    spec = importlib.util.spec_from_file_location("velocity_pilot_scenarios", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scenario validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def load_manifest(path: Path) -> dict:
    """Validate the JSON-compatible YAML pilot before producing any commands."""

    data = _json_object(path)
    if set(data) != {"schema_version", "name", "purpose", "training", "evaluation", "readiness"}:
        raise ValueError("pilot manifest has missing or unexpected top-level fields")
    if data["schema_version"] != 1 or data["name"] != "velocity_baseline_pilot":
        raise ValueError("unsupported pilot manifest version or name")
    if not isinstance(data["purpose"], str) or not data["purpose"]:
        raise ValueError("purpose must describe the pilot")
    train, evaluate, readiness = data["training"], data["evaluation"], data["readiness"]
    if not isinstance(train, dict) or set(train) != {
        "scenario", "num_envs", "max_iterations", "seed"
    } or train["scenario"] != "nominal":
        raise ValueError("training must specify the nominal scenario and required fields")
    _positive_int(train["num_envs"], "training.num_envs")
    _positive_int(train["max_iterations"], "training.max_iterations")
    _positive_int(train["seed"], "training.seed", minimum=0)
    if not isinstance(evaluate, dict) or set(evaluate) != {
        "scenarios", "num_envs", "episodes_per_env", "seed", "physics_trace"
    } or evaluate["scenarios"] != list(SCENARIOS):
        raise ValueError("evaluation must list the four scenarios in evaluate_all.py order")
    if not 1 <= _positive_int(evaluate["num_envs"], "evaluation.num_envs") <= 8:
        raise ValueError("evaluation.num_envs must be <= 8 with physics tracing")
    _positive_int(evaluate["episodes_per_env"], "evaluation.episodes_per_env")
    _positive_int(evaluate["seed"], "evaluation.seed", minimum=0)
    if evaluate["physics_trace"] is not True:
        raise ValueError("the pilot requires physics_trace=true")
    if not isinstance(readiness, dict) or set(readiness) != {
        "minimum_exposed_episodes_per_scenario"
    }:
        raise ValueError("readiness requires an exposure minimum")
    minimum = _positive_int(
        readiness["minimum_exposed_episodes_per_scenario"], "readiness exposure minimum"
    )
    if minimum > evaluate["num_envs"] * evaluate["episodes_per_env"]:
        raise ValueError("exposure minimum exceeds the evaluation cohort")
    return data


def scenario_specs():
    module = _scenarios_module()
    directory = ROOT / "configs" / "task" / "velocity"
    return {name: module.load_scenario(name, directory) for name in SCENARIOS}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _flag(value: str, label: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"{label} must be True or False")
    return value == "True"


def _trace_status(path: Path, scenario, evaluate: dict) -> dict:
    expected = {
        (env, episode)
        for env in range(evaluate["num_envs"])
        for episode in range(evaluate["episodes_per_env"])
    }
    states = {}
    samples = terminations = timeouts = pushes = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
            raise ValueError("policy trace columns do not match evaluate.py")
        for row in reader:
            if None in row:
                raise ValueError("policy trace row has extra columns")
            key = (int(row["env_id"]), int(row["episode_id"]))
            if key not in expected:
                raise ValueError(f"unexpected policy episode {key}")
            state = states.setdefault(key, {
                "last_time": None, "done": False, "pushed": False, "seen_points": set(),
            })
            if state["done"]:
                raise ValueError(f"policy trace continues after episode {key} ended")
            time_s = _finite_float(row["time_s"], "policy time")
            if time_s < 0 or (state["last_time"] is not None and time_s <= state["last_time"]):
                raise ValueError(f"non-increasing policy time in episode {key}")
            state["last_time"] = time_s
            command = tuple(
                _finite_float(row[field], field)
                for field in ("command_vx_m_s", "command_vy_m_s", "command_wz_rad_s")
            )
            for field in TRACE_FIELDS[6:14]:
                _finite_float(row[field], field)
            terminated = _flag(row["terminated"], "terminated")
            truncated = _flag(row["truncated"], "truncated")
            pushed = _flag(row["push_fired"], "push_fired")
            if scenario.push is not None:
                state["pushed"] |= pushed
            else:
                for index, point in enumerate(scenario.evaluation):
                    end = (
                        scenario.evaluation[index + 1].time_s
                        if index + 1 < len(scenario.evaluation) else scenario.episode_length_s
                    )
                    if point.time_s - 1e-4 <= time_s < end - 1e-4 and all(
                        math.isclose(actual, target, rel_tol=0, abs_tol=1e-4)
                        for actual, target in zip(command, point.command)
                    ):
                        state["seen_points"].add(index)
            state["done"] = terminated or truncated
            samples += 1
            terminations += terminated
            timeouts += truncated
            pushes += pushed
    completed = sum(state["done"] for state in states.values())
    exposed = sum(
        state["done"] and (
            state["pushed"] if scenario.push is not None
            else len(state["seen_points"]) == len(scenario.evaluation)
        )
        for state in states.values()
    )
    return {
        "samples": samples, "terminations": terminations, "timeouts": timeouts,
        "pushes": pushes, "completed_episodes": completed, "exposed_episodes": exposed,
        "expected_episodes": len(expected), "seen_episodes": len(states),
    }


def _physics_status(run_dir: Path, expected_episodes: set[tuple[int, int]]) -> int:
    metadata = _json_object(run_dir / "physics_metadata.json")
    safety = _json_object(run_dir / "safety_summary.json")
    expected_rate = _json_object(ROOT / "configs" / "safety" / "time_domain.yaml")[
        "sample_rate_hz"
    ]
    for label, value in (
        ("physics metadata", metadata.get("sample_rate_hz")),
        ("safety summary", safety.get("sample_rate_hz")),
    ):
        if type(value) not in (float, int) or not math.isclose(value, expected_rate):
            raise ValueError(f"{label} sample rate disagrees with {expected_rate} Hz")
    if metadata.get("torque_is_measured") is not False:
        raise ValueError("physics metadata incorrectly claims measured torque")
    if safety.get("physical_torque_violation_claim_available") is not False:
        raise ValueError("safety summary incorrectly claims physical torque violations")
    fingerprints = safety.get("config_sha256")
    if not isinstance(fingerprints, dict) or any(
        fingerprints.get(filename) != _sha256(ROOT / "configs" / "safety" / filename)
        for filename in ("time_domain.yaml", "frequency_domain.yaml", "g1_limits.yaml")
    ):
        raise ValueError("safety analysis config hashes disagree with current files")
    count = 0
    episodes = set()
    with (run_dir / "physics_trace.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PHYSICS_FIELDS:
            raise ValueError("physics trace columns do not match the recorder")
        for row in reader:
            if None in row:
                raise ValueError("physics trace row has extra columns")
            episodes.add((int(row["env_id"]), int(row["episode_id"])))
            count += 1
    aggregate = safety.get("temporal_aggregate", {})
    if not isinstance(aggregate, dict) or aggregate.get("physics_samples") != count:
        raise ValueError("physics sample count disagrees with safety summary")
    if episodes != expected_episodes or aggregate.get("episode_count") != len(episodes):
        raise ValueError("physics trace does not cover every expected episode")
    return count


def _check_scenario(run_dir: Path, name: str, scenario, evaluate: dict) -> dict:
    metadata = _json_object(run_dir / "metadata.json")
    expected = {
        "task": TASK_ID, "scenario": name, "seed": evaluate["seed"],
        "num_envs": evaluate["num_envs"],
        "episodes_per_env": evaluate["episodes_per_env"],
        "physics_trace_enabled": True, "isaac_lab_ref": "v2.3.2",
        "scenario_sha256": _sha256(scenario.source),
    }
    for field, value in expected.items():
        actual = metadata.get(field)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"metadata {field} disagrees with pilot plan or current config")
    digest = metadata.get("checkpoint_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("metadata has no valid checkpoint SHA-256")
    trace = _trace_status(run_dir / "trace.csv", scenario, evaluate)
    summary = _json_object(run_dir / "summary.json")
    for field in ("samples", "terminations", "timeouts", "pushes"):
        if type(summary.get(field)) is not int or summary[field] != trace[field]:
            raise ValueError(f"summary {field} disagrees with policy trace")
    if trace["completed_episodes"] != trace["expected_episodes"]:
        raise ValueError(
            f"only {trace['completed_episodes']}/{trace['expected_episodes']} episodes completed"
        )
    expected_episodes = {
        (env, episode)
        for env in range(evaluate["num_envs"])
        for episode in range(evaluate["episodes_per_env"])
    }
    trace["physics_samples"] = _physics_status(run_dir, expected_episodes)
    trace["checkpoint_sha256"] = digest
    return trace


def audit(manifest: dict, run_root: Path) -> dict:
    specs = scenario_specs()
    evaluate = manifest["evaluation"]
    minimum = manifest["readiness"]["minimum_exposed_episodes_per_scenario"]
    result = {
        "status": "ready_for_comparison",
        "meaning": "data completeness only; not policy performance or physical safety",
        "manifest_sha256": None,
        "run_root": str(run_root),
        "scenarios": {},
        "issues": [],
    }
    digests = set()
    for name in SCENARIOS:
        try:
            details = _check_scenario(run_root / name, name, specs[name], evaluate)
            result["scenarios"][name] = details
            digests.add(details["checkpoint_sha256"])
            if details["exposed_episodes"] < minimum:
                event = "push" if specs[name].push is not None else "entire command schedule"
                result["issues"].append(
                    f"{name}: only {details['exposed_episodes']} completed episodes observed {event}; "
                    f"pilot minimum is {minimum}"
                )
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
            result["scenarios"][name] = {"error": str(exc)}
            result["issues"].append(f"{name}: {exc}")
    if len(digests) != 1:
        result["issues"].append("four scenarios do not share one verified checkpoint SHA-256")
    comparison = run_root / "comparison.csv"
    if not comparison.is_file():
        result["issues"].append("evaluate_all.py comparison.csv is missing")
    else:
        try:
            with comparison.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "scenario" not in (reader.fieldnames or []):
                    raise ValueError("comparison.csv has no scenario column")
                names = [row["scenario"] for row in reader]
                if names != list(SCENARIOS):
                    raise ValueError("comparison.csv does not contain the four scenarios in order")
        except (OSError, ValueError, KeyError, csv.Error) as exc:
            result["issues"].append(f"comparison.csv: {exc}")
    result["status"] = "incomplete" if result["issues"] else "ready_for_comparison"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="validate configs and print Linux commands")
    audit_parser = subparsers.add_parser("audit", help="check a completed four-scenario run offline")
    for subparser in (plan_parser, audit_parser):
        subparser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    audit_parser.add_argument("--run-root", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, help="optional JSON audit report")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        specs = scenario_specs()
        if args.command == "plan":
            train = manifest["training"]
            evaluate = manifest["evaluation"]
            print(f"Pilot only: {manifest['purpose']}")
            print(f"Manifest SHA-256: {_sha256(args.manifest)}")
            for name, scenario in specs.items():
                event = scenario.push.time_s if scenario.push else scenario.evaluation[-1].time_s
                print(f"  {name}: config SHA-256 {_sha256(scenario.source)}; final event at {event:g} s")
            commands = (
                ["python", "scripts/train.py", "--scenario", train["scenario"],
                 "--num-envs", str(train["num_envs"]), "--max-iterations",
                 str(train["max_iterations"]), "--seed", str(train["seed"])],
                ["python", "scripts/evaluate_all.py", "--checkpoint", "/PATH/TO/model_XXX.pt",
                 "--num-envs", str(evaluate["num_envs"]), "--episodes",
                 str(evaluate["episodes_per_env"]), "--seed", str(evaluate["seed"])],
                ["python3", "scripts/velocity_experiment.py", "audit", "--run-root",
                 "outputs/evaluation/YOUR_RUN"],
            )
            print("On Linux GPU, from the repository root with .venv activated:")
            for command in commands:
                print("  " + shlex.join(command))
            print("Replace placeholders after training. No simulator or cloud job was launched.")
            return 0
        report = audit(manifest, args.run_root)
        report["manifest_sha256"] = _sha256(args.manifest)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for name, details in report["scenarios"].items():
            if "error" not in details:
                print(
                    f"{name}: completed {details['completed_episodes']}/"
                    f"{details['expected_episodes']}; exposed {details['exposed_episodes']}; "
                    f"physics samples {details['physics_samples']}"
                )
        for issue in report["issues"]:
            print(f"[ISSUE] {issue}")
        print(f"[{report['status'].upper()}] Data completeness only, not a safety claim.")
        return 0 if report["status"] == "ready_for_comparison" else 1
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"[ERROR] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
