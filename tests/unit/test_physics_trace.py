"""Portable physics-trace round trip and offline safety-report integration."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.safety.analysis import analyze_run  # noqa: E402
from safe_humanoid.telemetry.episode_writer import EvaluationWriter  # noqa: E402
from safe_humanoid.telemetry.physics_trace import (  # noqa: E402
    PhysicsSample, PhysicsTraceWriter, read_physics_trace,
)


class TestPhysicsTrace(unittest.TestCase):
    def test_round_trip_analysis_and_mixed_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "nominal"
            EvaluationWriter(run, {
                "scenario": "nominal", "checkpoint": "synthetic.pt", "seed": 0
            }).close()
            writer = PhysicsTraceWriter(
                run, ["synthetic_joint"], 0.005, "Isaac Lab v2.3.2 G1_MINIMAL_CFG"
            )
            for step in range(1, 257):
                writer.record(PhysicsSample(
                    0, 0, step, step * 0.005,
                    (math.sin(2 * math.pi * 40 * step / 200),),
                    (math.sin(2 * math.pi * 5 * step / 200),),
                ))
            writer.close()
            metadata, samples = read_physics_trace(run)
            self.assertEqual(metadata["sample_rate_hz"], 200)
            self.assertFalse(metadata["torque_is_measured"])
            self.assertEqual(len(samples), 256)
            self.assertEqual(samples[0].physics_step, 1)
            with self.assertRaises(FileExistsError):
                PhysicsTraceWriter(run, ["synthetic_joint"], 0.005, metadata["asset"])
            self.assertEqual(len(read_physics_trace(run)[1]), 256)
            safety = analyze_run(run, ROOT / "configs" / "safety")
            self.assertEqual(safety["temporal_aggregate"]["physics_samples"], 256)
            self.assertEqual(safety["hardware_limit_status"], "uncalibrated")
            self.assertIsNone(
                safety["temporal_aggregate"]["hardware_torque_exceed_sample_fraction"]
            )
            self.assertIsNone(safety["temporal_aggregate"]["hardware_speed_exceed_sample_fraction"])
            self.assertGreater(
                safety["frequency_aggregate"]["mean_estimated_torque_high_25_100_power_nm2"], 0.45
            )
            other = root / "legacy"
            EvaluationWriter(other, {
                "scenario": "abrupt", "checkpoint": "synthetic.pt", "seed": 1
            }).close()
            output = root / "comparison.csv"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "report.py"), str(run), str(other),
                "--output", str(output),
            ], check=True, capture_output=True, text=True)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["hardware_limit_status"], "uncalibrated")
            self.assertEqual(rows[1]["hardware_limit_status"], "")

    def test_rejects_wrong_vector_size_and_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = PhysicsTraceWriter(Path(folder), ["a", "b"], 0.005, "synthetic")
            with self.assertRaisesRegex(ValueError, "2 entries"):
                writer.record(PhysicsSample(0, 0, 1, 0.005, (1,), (2, 3)))
            with self.assertRaisesRegex(ValueError, "timestamp"):
                writer.record(PhysicsSample(0, 0, 1, 0.01, (1, 2), (3, 4)))
            writer.close()


if __name__ == "__main__":
    unittest.main()
