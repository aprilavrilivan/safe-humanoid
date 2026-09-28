"""Portable physics-step trace contract, independent of Isaac Sim and Torch."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

TRACE_FIELDS = (
    "env_id", "episode_id", "physics_step", "time_s",
    "estimated_actuator_torque_nm", "joint_velocity_rad_s",
)
TRACE_FILENAME = "physics_trace.csv"
METADATA_FILENAME = "physics_metadata.json"


@dataclass(frozen=True)
class PhysicsSample:
    env_id: int
    episode_id: int
    physics_step: int
    time_s: float
    estimated_actuator_torque_nm: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]


def _finite_vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must have {size} entries")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{label} must be numeric")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def validate_sample(sample: PhysicsSample, joint_count: int, dt_s: float) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < minimum
        for value, minimum in (
            (sample.env_id, 0), (sample.episode_id, 0), (sample.physics_step, 1)
        )
    ):
        raise ValueError("env_id, episode_id and physics_step must be valid integers")
    if isinstance(sample.time_s, bool) or not isinstance(sample.time_s, (int, float)):
        raise ValueError("physics timestamp must be numeric")
    if not math.isfinite(sample.time_s) or not math.isclose(
        sample.time_s, sample.physics_step * dt_s, rel_tol=1.0e-6, abs_tol=1.0e-8
    ):
        raise ValueError("physics timestamp does not match physics_step * dt_s")
    _finite_vector(sample.estimated_actuator_torque_nm, joint_count, "estimated_actuator_torque_nm")
    _finite_vector(sample.joint_velocity_rad_s, joint_count, "joint_velocity_rad_s")


class PhysicsTraceWriter:
    """Write bounded-cohort traces without retaining complete episodes in RAM."""

    def __init__(self, output_dir: Path, joint_names: list[str], dt_s: float, asset: str):
        self.output_dir = Path(output_dir)
        names = tuple(joint_names)
        if not names or len(names) != len(set(names)) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("joint_names must be nonempty and unique")
        if (
            isinstance(dt_s, bool)
            or not isinstance(dt_s, (int, float))
            or not math.isfinite(dt_s)
            or dt_s <= 0
        ):
            raise ValueError("dt_s must be positive and finite")
        if not isinstance(asset, str) or not asset:
            raise ValueError("asset must be a nonempty string")
        self.joint_names = names
        self.dt_s = dt_s
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / TRACE_FILENAME
        metadata_path = self.output_dir / METADATA_FILENAME
        if trace_path.exists() or metadata_path.exists():
            raise FileExistsError("physics trace artifacts already exist")
        metadata = {
            "schema_version": 1,
            "joint_names": list(names),
            "dt_s": dt_s,
            "sample_rate_hz": 1.0 / dt_s,
            "asset": asset,
            "signal_semantics": (
                "one sample per physics step; joint velocity is read directly from the PhysX view "
                "after stepping; torque is Isaac Lab's pre-step approximate/clipped implicit "
                "PD actuator effort, not solver output or measured motor torque"
            ),
            "torque_is_measured": False,
        }
        self._handle = trace_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=TRACE_FIELDS)
        self._writer.writeheader()
        self._closed = False
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def record(self, sample: PhysicsSample) -> None:
        if self._closed:
            raise ValueError("physics trace is closed")
        validate_sample(sample, len(self.joint_names), self.dt_s)
        self._writer.writerow({
            "env_id": sample.env_id,
            "episode_id": sample.episode_id,
            "physics_step": sample.physics_step,
            "time_s": sample.time_s,
            "estimated_actuator_torque_nm": json.dumps(
                sample.estimated_actuator_torque_nm, separators=(",", ":")
            ),
            "joint_velocity_rad_s": json.dumps(
                sample.joint_velocity_rad_s, separators=(",", ":")
            ),
        })

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


def read_physics_trace(output_dir: Path) -> tuple[dict, list[PhysicsSample]]:
    directory = Path(output_dir)
    metadata = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version", "joint_names", "dt_s", "sample_rate_hz", "asset",
        "signal_semantics", "torque_is_measured"
    } or metadata["schema_version"] != 1:
        raise ValueError("invalid physics trace metadata")
    names = metadata["joint_names"]
    dt_s = metadata["dt_s"]
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name for name in names
    ) or len(names) != len(set(names)):
        raise ValueError("invalid physics joint names")
    if (
        isinstance(dt_s, bool)
        or not isinstance(dt_s, (float, int))
        or not math.isfinite(dt_s)
        or dt_s <= 0
    ):
        raise ValueError("invalid physics dt_s")
    sample_rate = metadata["sample_rate_hz"]
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, float))
        or not math.isfinite(sample_rate)
    ):
        raise ValueError("invalid physics sample rate")
    if not math.isclose(sample_rate, 1.0 / dt_s):
        raise ValueError("physics sample rate disagrees with dt_s")
    if metadata["torque_is_measured"] is not False:
        raise ValueError("this trace schema only supports estimated G1 implicit actuator torque")
    if any(
        not isinstance(metadata[field], str) or not metadata[field]
        for field in ("asset", "signal_semantics")
    ):
        raise ValueError("physics asset and signal semantics must be nonempty strings")
    samples = []
    with (directory / TRACE_FILENAME).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TRACE_FIELDS:
            raise ValueError("invalid physics trace columns")
        for row in reader:
            sample = PhysicsSample(
                env_id=int(row["env_id"]),
                episode_id=int(row["episode_id"]),
                physics_step=int(row["physics_step"]),
                time_s=float(row["time_s"]),
                estimated_actuator_torque_nm=tuple(json.loads(row["estimated_actuator_torque_nm"])),
                joint_velocity_rad_s=tuple(json.loads(row["joint_velocity_rad_s"])),
            )
            validate_sample(sample, len(names), dt_s)
            samples.append(sample)
    return metadata, samples
