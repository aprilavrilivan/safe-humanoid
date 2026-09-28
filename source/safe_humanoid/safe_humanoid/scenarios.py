"""Simulator-independent, validated velocity-scenario specifications.

The ``.yaml`` files use the JSON subset of YAML 1.2 so they can be checked on a
Mac without installing Isaac Sim, PyYAML, or a CUDA-enabled PyTorch build.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

SCENARIO_NAMES = ("nominal", "aggressive", "abrupt", "push_recovery")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class VelocityRanges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    resampling_time_range_s: tuple[float, float]


@dataclass(frozen=True)
class TrainingPushSpec:
    time_range_s: tuple[float, float]
    planar_delta_speed_range_m_s: tuple[float, float]


@dataclass(frozen=True)
class SchedulePoint:
    time_s: float
    command: tuple[float, float, float]


@dataclass(frozen=True)
class PushSpec:
    time_s: float
    delta_velocity_world_m_s: tuple[float, float, float]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    training: VelocityRanges
    training_push: TrainingPushSpec | None
    evaluation: tuple[SchedulePoint, ...]
    push: PushSpec | None
    episode_length_s: float
    source: Path

    def command_at(self, time_s: float) -> tuple[float, float, float]:
        """Return the commanded body-frame velocity at an episode-relative time."""

        value = self.evaluation[0].command
        for point in self.evaluation[1:]:
            if time_s < point.time_s:
                break
            value = point.command
        return value


def load_scenario(name: str, config_dir: Path) -> ScenarioSpec:
    """Load one allow-listed scenario, rejecting malformed units or schedules."""

    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIO_NAMES}")
    path = Path(config_dir) / f"{name}.yaml"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "name", "episode_length_s", "training", "evaluation"
    }:
        raise ValueError(f"{path}: invalid top-level fields")
    if payload["schema_version"] != 1 or payload["name"] != name:
        raise ValueError(f"{path}: schema version or scenario name mismatch")
    duration = _number(payload["episode_length_s"], "episode_length_s")
    if duration <= 0:
        raise ValueError("episode_length_s must be positive")

    training = payload["training"]
    if not isinstance(training, dict) or set(training) != {
        "lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_z_rad_s",
        "resampling_time_range_s", "push"
    }:
        raise ValueError(f"{path}: invalid training fields")
    ranges = []
    for key in ("lin_vel_x_m_s", "lin_vel_y_m_s", "ang_vel_z_rad_s"):
        bounds = _vector(training[key], 2, key)
        if bounds[0] > bounds[1]:
            raise ValueError(f"{key}: lower bound exceeds upper bound")
        ranges.append(bounds)
    resampling = _vector(training["resampling_time_range_s"], 2, "resampling_time_range_s")
    if not 0 < resampling[0] <= resampling[1] < duration:
        raise ValueError("resampling_time_range_s must be positive, ordered and inside episode")
    training_push = None
    if training["push"] is not None:
        push_training = training["push"]
        if not isinstance(push_training, dict) or set(push_training) != {
            "time_range_s", "planar_delta_speed_range_m_s"
        }:
            raise ValueError("training.push requires time_range_s and planar_delta_speed_range_m_s")
        time_range = _vector(push_training["time_range_s"], 2, "training.push.time_range_s")
        speed_range = _vector(
            push_training["planar_delta_speed_range_m_s"], 2,
            "training.push.planar_delta_speed_range_m_s",
        )
        if not 0 < time_range[0] <= time_range[1] < duration:
            raise ValueError("training push time range must lie inside episode")
        if not 0 < speed_range[0] <= speed_range[1]:
            raise ValueError("training push delta-speed range must be positive and ordered")
        training_push = TrainingPushSpec(time_range, speed_range)
    if name == "abrupt" and resampling[1] > 1.0:
        raise ValueError("abrupt training must resample commands at most every second")
    if (training_push is not None) != (name == "push_recovery"):
        raise ValueError("only push_recovery must contain a training push")

    evaluation = payload["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {"command_schedule", "push"}:
        raise ValueError(f"{path}: invalid evaluation fields")
    schedule_data = evaluation["command_schedule"]
    if not isinstance(schedule_data, list) or not schedule_data:
        raise ValueError("command_schedule must be a nonempty list")
    schedule: list[SchedulePoint] = []
    for index, row in enumerate(schedule_data):
        if not isinstance(row, dict) or set(row) != {"time_s", "velocity_b"}:
            raise ValueError(f"command_schedule[{index}]: invalid fields")
        time_s = _number(row["time_s"], f"command_schedule[{index}].time_s")
        if time_s < 0 or time_s >= duration or (schedule and time_s <= schedule[-1].time_s):
            raise ValueError("command times must be strictly increasing and within the episode")
        schedule.append(SchedulePoint(time_s, _vector(row["velocity_b"], 3, "velocity_b")))
    if schedule[0].time_s != 0:
        raise ValueError("the first command must start at time 0")

    push_data = evaluation["push"]
    push = None
    if push_data is not None:
        if not isinstance(push_data, dict) or set(push_data) != {
            "time_s", "delta_velocity_world_m_s"
        }:
            raise ValueError("push must contain time_s and delta_velocity_world_m_s")
        push_time = _number(push_data["time_s"], "push.time_s")
        delta = _vector(push_data["delta_velocity_world_m_s"], 3, "push.delta_velocity")
        if not 0 < push_time < duration or not any(delta):
            raise ValueError("push must occur inside the episode and have nonzero magnitude")
        push = PushSpec(push_time, delta)

    return ScenarioSpec(
        name=name,
        training=VelocityRanges(*ranges, resampling),
        training_push=training_push,
        evaluation=tuple(schedule),
        push=push,
        episode_length_s=duration,
        source=path,
    )
