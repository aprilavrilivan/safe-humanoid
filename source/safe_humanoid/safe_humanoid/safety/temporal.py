"""Physics-rate time-domain loading metrics with explicit episode boundaries."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safe_humanoid.telemetry.physics_trace import PhysicsSample


@dataclass(frozen=True)
class TemporalConfig:
    sample_rate_hz: float
    torque_rms_window_s: float
    window_samples: int
    source: Path


def load_temporal_config(path: str | Path) -> TemporalConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "sample_rate_hz", "torque_rms_window_s", "notes"
    } or payload["schema_version"] != 1:
        raise ValueError("invalid time-domain config")
    fs, window_s = payload["sample_rate_hz"], payload["torque_rms_window_s"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (fs, window_s)
    ):
        raise ValueError("sample rate and RMS window must be numeric")
    if not math.isfinite(fs) or not math.isfinite(window_s) or fs <= 0 or window_s <= 0:
        raise ValueError("sample rate and RMS window must be positive and finite")
    window_samples = round(fs * window_s)
    if window_samples < 2 or not math.isclose(window_samples, fs * window_s, abs_tol=1.0e-8):
        raise ValueError("RMS window must contain an integer number of at least two samples")
    if not isinstance(payload["notes"], str) or not payload["notes"]:
        raise ValueError("time-domain notes must explain the analysis settings")
    return TemporalConfig(float(fs), float(window_s), window_samples, source)


def _episode_metrics(
    samples: list[PhysicsSample], config: TemporalConfig,
    hardware_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None,
) -> dict:
    joint_count = len(samples[0].estimated_actuator_torque_nm)
    if hardware_bounds is not None and any(
        len(bounds) != joint_count for bounds in hardware_bounds
    ):
        raise ValueError("hardware bounds do not match joint count")
    rolling = deque(maxlen=config.window_samples)
    previous_speed = None
    peak_torque = 0.0
    peak_speed = 0.0
    peak_power = 0.0
    peak_rms = None
    peak_accel = None
    power_sum = 0.0
    speed_exceed = 0
    dt = 1.0 / config.sample_rate_hz
    for index, sample in enumerate(samples, 1):
        if sample.physics_step != index:
            raise ValueError("physics steps must start at 1 and be contiguous within each episode")
        torque = sample.estimated_actuator_torque_nm
        speed = sample.joint_velocity_rad_s
        if len(torque) != joint_count or len(speed) != joint_count:
            raise ValueError("joint count changes inside an episode")
        if not all(math.isfinite(value) for value in (*torque, *speed)):
            raise ValueError("temporal signal contains NaN or Inf")
        peak_torque = max(peak_torque, *(abs(value) for value in torque))
        peak_speed = max(peak_speed, *(abs(value) for value in speed))
        power = sum(abs(t * v) for t, v in zip(torque, speed, strict=True))
        peak_power = max(peak_power, power)
        power_sum += power
        rolling.append(torque)
        if len(rolling) == config.window_samples:
            rms = max(
                math.sqrt(sum(frame[j] ** 2 for frame in rolling) / len(rolling))
                for j in range(joint_count)
            )
            peak_rms = rms if peak_rms is None else max(peak_rms, rms)
        if previous_speed is not None:
            acceleration = max(
                abs(a - b) / dt for a, b in zip(speed, previous_speed, strict=True)
            )
            peak_accel = acceleration if peak_accel is None else max(peak_accel, acceleration)
        previous_speed = speed
        if hardware_bounds is not None:
            _, speed_bounds = hardware_bounds
            speed_exceed += any(
                abs(value) > bound for value, bound in zip(speed, speed_bounds, strict=True)
            )
    count = len(samples)
    return {
        "env_id": samples[0].env_id,
        "episode_id": samples[0].episode_id,
        "samples": count,
        "duration_s": count * dt,
        "peak_estimated_joint_torque_nm": peak_torque,
        "peak_joint_speed_rad_s": peak_speed,
        "peak_estimated_absolute_mechanical_power_w": peak_power,
        "mean_estimated_absolute_mechanical_power_w": power_sum / count,
        "estimated_absolute_mechanical_energy_j": power_sum * dt,
        "peak_rolling_estimated_joint_torque_rms_nm": peak_rms,
        "peak_joint_acceleration_rad_s2": peak_accel,
        "hardware_torque_exceed_sample_fraction": None,
        "hardware_speed_exceed_sample_fraction": (
            speed_exceed / count if hardware_bounds is not None else None
        ),
    }


def analyze_temporal(
    samples: list[PhysicsSample], config: TemporalConfig,
    hardware_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> dict:
    if not samples:
        raise ValueError("physics trace has no samples")
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample.env_id, sample.episode_id)].append(sample)
    episodes = [
        _episode_metrics(group, config, hardware_bounds)
        for _, group in sorted(grouped.items())
    ]
    count = sum(item["samples"] for item in episodes)

    def maximum(field: str):
        values = [item[field] for item in episodes if item[field] is not None]
        return max(values) if values else None
    aggregate = {
        "episode_count": len(episodes),
        "physics_samples": count,
        "peak_estimated_joint_torque_nm": maximum("peak_estimated_joint_torque_nm"),
        "peak_joint_speed_rad_s": maximum("peak_joint_speed_rad_s"),
        "peak_estimated_absolute_mechanical_power_w": maximum(
            "peak_estimated_absolute_mechanical_power_w"
        ),
        "mean_estimated_absolute_mechanical_power_w": sum(
            item["mean_estimated_absolute_mechanical_power_w"] * item["samples"]
            for item in episodes
        ) / count,
        "total_estimated_absolute_mechanical_energy_j": sum(
            item["estimated_absolute_mechanical_energy_j"] for item in episodes
        ),
        "peak_rolling_estimated_joint_torque_rms_nm": maximum(
            "peak_rolling_estimated_joint_torque_rms_nm"
        ),
        "peak_joint_acceleration_rad_s2": maximum("peak_joint_acceleration_rad_s2"),
        "hardware_torque_exceed_sample_fraction": None,
        "hardware_speed_exceed_sample_fraction": None,
    }
    if hardware_bounds is not None:
        field = "hardware_speed_exceed_sample_fraction"
        aggregate[field] = sum(item[field] * item["samples"] for item in episodes) / count
    return {"aggregate": aggregate, "episodes": episodes}
