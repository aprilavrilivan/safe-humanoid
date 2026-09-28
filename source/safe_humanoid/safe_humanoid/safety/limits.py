"""Provenance-checked G1 limit loading; uncalibrated means no violation claim."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class JointLimit:
    value: float
    unit: str
    source_url: str
    source_kind: str


@dataclass(frozen=True)
class G1Limits:
    robot_asset: str
    calibration_status: str
    joint_torque_abs_nm: dict[str, JointLimit]
    joint_speed_abs_rad_s: dict[str, JointLimit]
    notes: str
    source: Path

    def hardware_arrays(self, joint_names: list[str] | tuple[str, ...]):
        """Return fully aligned bounds, or None when not calibrated.

        A partial map must never be used as the denominator of a reported
        whole-robot violation rate.
        """

        names = tuple(joint_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("joint_names must be nonempty and unique")
        if self.calibration_status != "calibrated":
            return None
        if set(self.joint_torque_abs_nm) != set(names):
            raise ValueError("calibrated torque limits do not exactly cover simulator joints")
        if set(self.joint_speed_abs_rad_s) != set(names):
            raise ValueError("calibrated speed limits do not exactly cover simulator joints")
        for entry in (*self.joint_torque_abs_nm.values(), *self.joint_speed_abs_rad_s.values()):
            if entry.source_kind != "manufacturer_hardware":
                raise ValueError("physical violation rates require manufacturer_hardware sources")
        return (
            tuple(self.joint_torque_abs_nm[name].value for name in names),
            tuple(self.joint_speed_abs_rad_s[name].value for name in names),
        )


def _entries(value: object, unit: str, label: str) -> dict[str, JointLimit]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    result = {}
    for name, payload in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}: joint names must be nonempty strings")
        if not isinstance(payload, dict) or set(payload) != {
            "value", "unit", "source_url", "source_kind"
        }:
            raise ValueError(f"{label}.{name}: expected value, unit, source_url, source_kind")
        number = payload["value"]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"{label}.{name}: value must be a number")
        number = float(number)
        if not math.isfinite(number) or number <= 0 or payload["unit"] != unit:
            raise ValueError(f"{label}.{name}: expected positive finite value in {unit}")
        source_url = payload["source_url"]
        if not isinstance(source_url, str):
            raise ValueError(f"{label}.{name}: source_url must be HTTPS")
        parsed = urlparse(source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"{label}.{name}: source_url must be HTTPS")
        kind = payload["source_kind"]
        if kind not in ("manufacturer_hardware", "software_guard", "simulation_asset"):
            raise ValueError(f"{label}.{name}: unsupported source_kind")
        result[name] = JointLimit(number, unit, source_url, kind)
    return result


def load_g1_limits(path: str | Path) -> G1Limits:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version", "robot_asset", "calibration_status",
        "joint_torque_abs_nm", "joint_speed_abs_rad_s", "notes",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != 1:
        raise ValueError(f"{source}: invalid G1 limit schema")
    if not isinstance(payload["robot_asset"], str) or not payload["robot_asset"]:
        raise ValueError("robot_asset must be a nonempty string")
    status = payload["calibration_status"]
    if status not in ("uncalibrated", "calibrated"):
        raise ValueError("calibration_status must be uncalibrated or calibrated")
    if not isinstance(payload["notes"], str) or not payload["notes"]:
        raise ValueError("notes must explain provenance or uncertainty")
    limits = G1Limits(
        robot_asset=payload["robot_asset"],
        calibration_status=status,
        joint_torque_abs_nm=_entries(payload["joint_torque_abs_nm"], "Nm", "torque"),
        joint_speed_abs_rad_s=_entries(payload["joint_speed_abs_rad_s"], "rad/s", "speed"),
        notes=payload["notes"],
        source=source,
    )
    if status == "uncalibrated" and (
        limits.joint_torque_abs_nm or limits.joint_speed_abs_rad_s
    ):
        raise ValueError("uncalibrated limits must not contain numerical entries")
    if status == "calibrated" and (
        not limits.joint_torque_abs_nm or not limits.joint_speed_abs_rad_s
    ):
        raise ValueError("calibrated limits require torque and speed entries")
    return limits
