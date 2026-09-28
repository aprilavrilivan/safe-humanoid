"""Contract tests for safety-limit loading, units, joint mapping, and validation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.safety.limits import load_g1_limits  # noqa: E402


def _entry(value: float, unit: str, kind: str = "manufacturer_hardware") -> dict:
    return {
        "value": value,
        "unit": unit,
        "source_url": "https://example.test/g1-spec",
        "source_kind": kind,
    }


class TestG1Limits(unittest.TestCase):
    def test_default_is_explicitly_uncalibrated(self):
        limits = load_g1_limits(ROOT / "configs" / "safety" / "g1_limits.yaml")
        self.assertEqual(limits.calibration_status, "uncalibrated")
        self.assertIsNone(limits.hardware_arrays(["hip", "knee"]))

    def test_synthetic_full_coverage_aligns_by_joint_name(self):
        payload = {
            "schema_version": 1, "robot_asset": "synthetic G1", "calibration_status": "calibrated",
            "joint_torque_abs_nm": {"knee": _entry(20, "Nm"), "hip": _entry(10, "Nm")},
            "joint_speed_abs_rad_s": {"knee": _entry(4, "rad/s"), "hip": _entry(2, "rad/s")},
            "notes": "Synthetic test thresholds only; not actual G1 specifications.",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "limits.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            limits = load_g1_limits(path)
            self.assertEqual(limits.hardware_arrays(["hip", "knee"]), ((10, 20), (2, 4)))
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                limits.hardware_arrays(["hip"])
            payload["joint_torque_abs_nm"]["hip"]["unit"] = "N"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Nm"):
                load_g1_limits(path)

    def test_software_guard_cannot_become_hardware_limit(self):
        payload = {
            "schema_version": 1, "robot_asset": "synthetic G1", "calibration_status": "calibrated",
            "joint_torque_abs_nm": {"hip": _entry(10, "Nm", "software_guard")},
            "joint_speed_abs_rad_s": {"hip": _entry(2, "rad/s")},
            "notes": "Software guard is not a hardware specification.",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "limits.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manufacturer_hardware"):
                load_g1_limits(path).hardware_arrays(["hip"])


if __name__ == "__main__":
    unittest.main()
