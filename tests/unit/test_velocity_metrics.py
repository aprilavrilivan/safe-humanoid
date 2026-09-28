"""GPU-free arithmetic and artifact tests for the first evaluation signals."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_source(name: str, relative: str):
    path = ROOT / "source" / "safe_humanoid" / "safe_humanoid" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


signals = load_source("velocity_signals_for_tests", "safety/signals.py")
writer_module = load_source("velocity_writer_for_tests", "telemetry/episode_writer.py")


class TestVelocityMetrics(unittest.TestCase):
    def test_units_and_values(self):
        sample = signals.velocity_sample(
            command_b=[1, 0, 0.5],
            linear_velocity_b=[0, 0, 0],
            angular_velocity_b=[0, 0, 0],
            projected_gravity_b=[0, 0, -1],
            joint_torque_nm=[3, 4],
            joint_velocity_rad_s=[2, -1],
        )
        self.assertEqual(sample["tracking_error_xy_m_s"], 1)
        self.assertEqual(sample["tracking_error_yaw_rad_s"], 0.5)
        self.assertEqual(sample["upright_alignment"], 1)
        self.assertAlmostEqual(sample["joint_torque_rms_nm"], math.sqrt(12.5))
        self.assertEqual(sample["joint_torque_peak_nm"], 4)
        self.assertEqual(sample["joint_speed_peak_rad_s"], 2)
        self.assertEqual(sample["absolute_mechanical_power_w"], 10)

    def test_rejects_invalid_sensor_values(self):
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            signals.velocity_sample(
                [0, 0, 0], [math.nan, 0, 0], [0, 0, 0], [0, 0, -1], [1], [1]
            )

    def test_writer_summary_counts_falls_and_timeouts_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = writer_module.EvaluationWriter(
                Path(folder) / "run", {"scenario": "nominal", "checkpoint": "mock.pt", "seed": 0}
            )
            sample = signals.velocity_sample(
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, -1], [1], [2]
            )
            row = {
                "env_id": 0, "episode_id": 0, "time_s": 0.0,
                "command_vx_m_s": 0, "command_vy_m_s": 0, "command_wz_rad_s": 0,
                **sample, "transition_reward": 0.1, "terminated": True, "truncated": False,
                "push_fired": True,
            }
            writer.record(row)
            summary = writer.close()
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["terminations"], 1)
            self.assertEqual(summary["timeouts"], 0)
            self.assertEqual(summary["pushes"], 1)
            saved = json.loads((Path(folder) / "run" / "summary.json").read_text())
            self.assertEqual(saved, summary)
            report = Path(folder) / "comparison.csv"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "report.py"),
                str(Path(folder) / "run"), "--output", str(report),
            ], check=True, capture_output=True, text=True)
            self.assertIn("mean_tracking_error_xy_m_s", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
