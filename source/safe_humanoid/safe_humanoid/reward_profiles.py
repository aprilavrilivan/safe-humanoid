"""Simulator-independent validation of online PPO reward-engineering profiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

REWARD_PROFILES = ("clean", "time", "frequency")


@dataclass(frozen=True)
class RewardProfile:
    name: str
    source: Path
    weight: float = 0.0
    sample_rate_hz: float = 0.0
    window_samples: int = 0
    hop_samples: int = 0
    band_hz: tuple[float, float] | None = None
    torque_reference_nm: float = 0.0
    joint_speed_reference_rad_s: float = 0.0


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _samples(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{label} must be an integer >= 2")
    return value


def load_reward_profile(name: str, path: str | Path) -> RewardProfile:
    """Load a fixed profile; coefficients are engineering choices, not limits."""

    if name not in REWARD_PROFILES:
        raise ValueError(f"unknown reward profile {name!r}")
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "notes", "profiles"}:
        raise ValueError("invalid reward profile manifest")
    if (
        payload["schema_version"] != 1
        or not isinstance(payload["notes"], str)
        or not payload["notes"]
    ):
        raise ValueError("reward profile manifest needs schema version 1 and explanatory notes")
    profiles = payload["profiles"]
    if (
        not isinstance(profiles, dict)
        or set(profiles) != set(REWARD_PROFILES)
        or profiles["clean"] is not None
    ):
        raise ValueError("manifest must define clean, time and frequency profiles")
    if name == "clean":
        return RewardProfile(name=name, source=source)
    row = profiles[name]
    required = {"weight", "sample_rate_hz", "window_samples", "torque_reference_nm"}
    if name == "frequency":
        required |= {"hop_samples", "band_hz", "joint_speed_reference_rad_s"}
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError(f"invalid {name} reward profile fields")
    weight = row["weight"]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (float, int))
        or not math.isfinite(weight)
        or weight >= 0
    ):
        raise ValueError("reward-engineering weight must be negative and finite")
    rate = _positive(row["sample_rate_hz"], "sample_rate_hz")
    window = _samples(row["window_samples"], "window_samples")
    torque_ref = _positive(row["torque_reference_nm"], "torque_reference_nm")
    if name == "time":
        return RewardProfile(
            name, source, float(weight), rate, window, torque_reference_nm=torque_ref
        )
    hop = _samples(row["hop_samples"], "hop_samples")
    if hop > window or window & (window - 1):
        raise ValueError("frequency window must be a power of two and hop <= window")
    band = row["band_hz"]
    if not isinstance(band, list) or len(band) != 2:
        raise ValueError("frequency band_hz must contain two endpoints")
    lower = _positive(band[0], "band_hz lower")
    upper = _positive(band[1], "band_hz upper")
    if not lower < upper <= rate / 2:
        raise ValueError("frequency band must lie below Nyquist")
    speed_ref = _positive(row["joint_speed_reference_rad_s"], "joint_speed_reference_rad_s")
    return RewardProfile(
        name, source, float(weight), rate, window, hop, (lower, upper), torque_ref, speed_ref
    )
