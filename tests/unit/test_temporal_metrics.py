"""Synthetic-signal tests for time-domain metrics and per-environment reset behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.safety.temporal import analyze_temporal, load_temporal_config  # noqa: E402
from safe_humanoid.telemetry.physics_trace import PhysicsSample  # noqa: E402


def _sample(episode: int, step: int, torque: float, speed: float) -> PhysicsSample:
    return PhysicsSample(0, episode, step, step / 200, (torque,), (speed,))


class TestTemporalMetrics(unittest.TestCase):
    def test_units_energy_and_reset_boundaries(self):
        config = load_temporal_config(ROOT / "configs" / "safety" / "time_domain.yaml")
        self.assertEqual(config.sample_rate_hz, 200)
        self.assertEqual(config.window_samples, 100)
        samples = [_sample(0, step, 2, 1) for step in range(1, 101)]
        samples += [_sample(1, 1, 4, 10), _sample(1, 2, 4, 11)]
        result = analyze_temporal(samples, config, ((5,), (2,)))
        first, second = result["episodes"]
        self.assertEqual(first["peak_rolling_estimated_joint_torque_rms_nm"], 2)
        self.assertIsNone(second["peak_rolling_estimated_joint_torque_rms_nm"])
        self.assertAlmostEqual(first["estimated_absolute_mechanical_energy_j"], 1.0)
        self.assertEqual(second["peak_joint_acceleration_rad_s2"], 200)
        self.assertEqual(result["aggregate"]["peak_joint_acceleration_rad_s2"], 200)
        self.assertIsNone(result["aggregate"]["hardware_torque_exceed_sample_fraction"])
        self.assertAlmostEqual(
            result["aggregate"]["hardware_speed_exceed_sample_fraction"], 2 / 102
        )

    def test_missing_physics_step_is_rejected(self):
        config = load_temporal_config(ROOT / "configs" / "safety" / "time_domain.yaml")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            analyze_temporal([_sample(0, 2, 1, 1)], config)
        with self.assertRaisesRegex(ValueError, "no samples"):
            analyze_temporal([], config)


if __name__ == "__main__":
    unittest.main()
