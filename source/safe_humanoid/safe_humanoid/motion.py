"""GPU-free contract for retargeted G1 reference motions.

This is a deliberately smaller format than BeyondMimic's body-state NPZ: it
tracks all simulator joints plus the floating root. A producer must retarget
the source motion to the *same* G1 asset before writing this file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MOTION_FIELDS = {
    "schema_version", "fps", "joint_names", "joint_pos", "joint_vel",
    "root_pos_w", "root_quat_w",
}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _matrix(value: Any, rows: int, columns: int, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError(f"{label} must have {rows} frames")
    result = []
    for frame_index, frame in enumerate(value):
        if not isinstance(frame, (list, tuple)) or len(frame) != columns:
            raise ValueError(f"{label}[{frame_index}] must have {columns} values")
        result.append(tuple(_finite(x, f"{label}[{frame_index}]") for x in frame))
    return tuple(result)


@dataclass(frozen=True)
class MotionClip:
    fps: float
    joint_names: tuple[str, ...]
    joint_pos: tuple[tuple[float, ...], ...]
    joint_vel: tuple[tuple[float, ...], ...]
    root_pos_w: tuple[tuple[float, ...], ...]
    root_quat_w: tuple[tuple[float, ...], ...]
    source: Path

    @property
    def frame_count(self) -> int:
        return len(self.joint_pos)

    @property
    def duration_s(self) -> float:
        return (self.frame_count - 1) / self.fps

    def frame_at(self, time_s: float) -> int:
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("time_s must be finite and nonnegative")
        return min(int(time_s * self.fps), self.frame_count - 1)

    def in_robot_order(self, robot_joint_names: list[str] | tuple[str, ...]) -> MotionClip:
        """Reorder joint columns, rejecting even one missing or extra joint."""

        robot_names = tuple(robot_joint_names)
        if len(robot_names) != len(set(robot_names)):
            raise ValueError("robot joint names must be unique")
        missing = sorted(set(robot_names) - set(self.joint_names))
        extra = sorted(set(self.joint_names) - set(robot_names))
        if missing or extra:
            raise ValueError(f"motion/robot joint mismatch: missing={missing}, extra={extra}")
        indexes = [self.joint_names.index(name) for name in robot_names]
        return MotionClip(
            self.fps,
            robot_names,
            tuple(tuple(row[i] for i in indexes) for row in self.joint_pos),
            tuple(tuple(row[i] for i in indexes) for row in self.joint_vel),
            self.root_pos_w,
            self.root_quat_w,
            self.source,
        )


def validate_motion_data(payload: dict[str, Any], source: Path) -> MotionClip:
    if set(payload) != MOTION_FIELDS or payload["schema_version"] != 1:
        raise ValueError(
            f"{source}: expected motion schema version 1 with fields {sorted(MOTION_FIELDS)}"
        )
    fps = _finite(payload["fps"], "fps")
    if fps <= 0:
        raise ValueError("fps must be positive")
    names = payload["joint_names"]
    if (
        not isinstance(names, (list, tuple))
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("joint_names must be a nonempty list of unique strings")
    frames = payload["joint_pos"]
    if not isinstance(frames, (list, tuple)) or len(frames) < 2:
        raise ValueError("joint_pos must contain at least two frames")
    count = len(frames)
    joint_pos = _matrix(frames, count, len(names), "joint_pos")
    joint_vel = _matrix(payload["joint_vel"], count, len(names), "joint_vel")
    root_pos = _matrix(payload["root_pos_w"], count, 3, "root_pos_w")
    root_quat = _matrix(payload["root_quat_w"], count, 4, "root_quat_w")
    for index, (pos, quat) in enumerate(zip(root_pos, root_quat, strict=True)):
        if pos[2] <= 0:
            raise ValueError(f"root_pos_w[{index}] must be above the ground")
        norm = math.sqrt(sum(component * component for component in quat))
        if not 0.99 <= norm <= 1.01:
            raise ValueError(f"root_quat_w[{index}] must be a unit wxyz quaternion")
    return MotionClip(
        fps, tuple(names), joint_pos, joint_vel, root_pos, root_quat, Path(source)
    )


def load_motion(path: str | Path) -> MotionClip:
    """Read JSON (small/debug) or NPZ (research), never allowing pickle objects."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference motion not found: {source}")
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is needed to read NPZ motions") from exc
        with np.load(source, allow_pickle=False) as arrays:
            if set(arrays.files) != MOTION_FIELDS:
                raise ValueError(
                    f"{source}: NPZ fields must be {sorted(MOTION_FIELDS)}; "
                    f"found {sorted(arrays.files)}"
                )
            payload = {key: arrays[key].tolist() for key in arrays.files}
    else:
        raise ValueError("motion file must be .json or .npz")
    if not isinstance(payload, dict):
        raise ValueError("motion file must contain an object")
    return validate_motion_data(payload, source)
