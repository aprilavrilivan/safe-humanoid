"""Validated, simulator-independent definitions for the two tracking studies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safe_humanoid.motion import MotionClip

EXPECTED_TASK_IDS = {
    "squat_stand": "SafeHumanoid-Tracking-SquatStand-G1-v0",
    "fast_leg_swing": "SafeHumanoid-Tracking-FastLegSwing-G1-v0",
}
SCENARIO_NAMES = tuple(EXPECTED_TASK_IDS)


def _positive(value: object, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {adjective}")
    return result


@dataclass(frozen=True)
class TrackingScenario:
    name: str
    task_id: str
    min_duration_s: float
    min_root_height_excursion_m: float
    min_leg_speed_rad_s: float
    joint_pos_std_rad: float
    joint_vel_std_rad_s: float
    root_height_std_m: float
    root_orientation_std_rad: float
    source: Path

    def check_motion(self, clip: MotionClip) -> None:
        """Reject a mislabeled or essentially static reference clip."""

        if clip.duration_s < self.min_duration_s:
            raise ValueError(
                f"{self.name} needs at least {self.min_duration_s}s; got {clip.duration_s:.3f}s"
            )
        heights = [point[2] for point in clip.root_pos_w]
        height_range = max(heights) - min(heights)
        if height_range < self.min_root_height_excursion_m:
            raise ValueError(
                f"{self.name} root height excursion {height_range:.3f}m is below "
                f"{self.min_root_height_excursion_m:.3f}m"
            )
        leg_indexes = [
            index for index, name in enumerate(clip.joint_names)
            if any(part in name for part in ("hip", "knee", "ankle"))
        ]
        if not leg_indexes:
            raise ValueError("reference motion has no named leg joints")
        peak_leg_speed = max(abs(frame[index]) for frame in clip.joint_vel for index in leg_indexes)
        if peak_leg_speed < self.min_leg_speed_rad_s:
            raise ValueError(
                f"{self.name} peak leg speed {peak_leg_speed:.3f}rad/s is below "
                f"{self.min_leg_speed_rad_s:.3f}rad/s"
            )


def load_tracking_scenario(name: str, config_dir: Path) -> TrackingScenario:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown tracking scenario {name!r}; choose from {SCENARIO_NAMES}")
    source = Path(config_dir) / f"{name}.yaml"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "name", "task_id", "reference_checks", "tracking_reward"
    }:
        raise ValueError(f"{source}: invalid top-level fields")
    if payload["schema_version"] != 1 or payload["name"] != name:
        raise ValueError(f"{source}: scenario name or schema version mismatch")
    checks = payload["reference_checks"]
    reward = payload["tracking_reward"]
    if not isinstance(checks, dict) or set(checks) != {
        "min_duration_s", "min_root_height_excursion_m", "min_leg_speed_rad_s"
    }:
        raise ValueError("invalid reference_checks")
    if not isinstance(reward, dict) or set(reward) != {
        "joint_pos_std_rad", "joint_vel_std_rad_s", "root_height_std_m",
        "root_orientation_std_rad"
    }:
        raise ValueError("invalid tracking_reward")
    task_id = payload["task_id"]
    if task_id != EXPECTED_TASK_IDS[name]:
        raise ValueError(f"{name} must use task ID {EXPECTED_TASK_IDS[name]}")
    return TrackingScenario(
        name=name,
        task_id=task_id,
        min_duration_s=_positive(checks["min_duration_s"], "min_duration_s"),
        min_root_height_excursion_m=_positive(
            checks["min_root_height_excursion_m"], "min_root_height_excursion_m", allow_zero=True
        ),
        min_leg_speed_rad_s=_positive(
            checks["min_leg_speed_rad_s"], "min_leg_speed_rad_s", allow_zero=True
        ),
        joint_pos_std_rad=_positive(reward["joint_pos_std_rad"], "joint_pos_std_rad"),
        joint_vel_std_rad_s=_positive(reward["joint_vel_std_rad_s"], "joint_vel_std_rad_s"),
        root_height_std_m=_positive(reward["root_height_std_m"], "root_height_std_m"),
        root_orientation_std_rad=_positive(
            reward["root_orientation_std_rad"], "root_orientation_std_rad"
        ),
        source=source,
    )
