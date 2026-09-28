"""GPU-free validation of all four velocity experiment specifications."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "source" / "safe_humanoid" / "safe_humanoid" / "scenarios.py"
SPEC = importlib.util.spec_from_file_location("scenario_specs_for_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
scenario_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scenario_module
SPEC.loader.exec_module(scenario_module)
CONFIG_DIR = ROOT / "configs" / "task" / "velocity"


class TestVelocityScenarios(unittest.TestCase):
    def test_all_four_are_valid_and_episode_relative(self):
        for name in scenario_module.SCENARIO_NAMES:
            with self.subTest(name=name):
                scenario = scenario_module.load_scenario(name, CONFIG_DIR)
                self.assertEqual(scenario.evaluation[0].time_s, 0.0)
                self.assertEqual(scenario.command_at(0), scenario.evaluation[0].command)
                self.assertEqual(scenario.command_at(100), scenario.evaluation[-1].command)

    def test_abrupt_changes_at_exact_boundary(self):
        scenario = scenario_module.load_scenario("abrupt", CONFIG_DIR)
        self.assertEqual(scenario.command_at(3.99), (0.5, 0.0, 0.0))
        self.assertEqual(scenario.command_at(4.0), (1.0, 0.0, 0.0))
        self.assertEqual(scenario.command_at(4.5), (-0.5, 0.0, 0.0))

    def test_only_push_scenario_has_one_velocity_kick(self):
        for name in scenario_module.SCENARIO_NAMES:
            scenario = scenario_module.load_scenario(name, CONFIG_DIR)
            self.assertEqual(scenario.push is not None, name == "push_recovery")
        push = scenario_module.load_scenario("push_recovery", CONFIG_DIR).push
        self.assertEqual(push.time_s, 5.0)
        self.assertEqual(push.delta_velocity_world_m_s, (0.0, 0.6, 0.0))

    def test_training_distributions_expose_the_safety_challenges(self):
        nominal = scenario_module.load_scenario("nominal", CONFIG_DIR)
        aggressive = scenario_module.load_scenario("aggressive", CONFIG_DIR)
        abrupt = scenario_module.load_scenario("abrupt", CONFIG_DIR)
        recovery = scenario_module.load_scenario("push_recovery", CONFIG_DIR)
        self.assertEqual(nominal.training.resampling_time_range_s, (10.0, 10.0))
        self.assertGreater(aggressive.training.lin_vel_x[1], nominal.training.lin_vel_x[1])
        self.assertLessEqual(abrupt.training.resampling_time_range_s[1], 1.0)
        self.assertIsNone(nominal.training_push)
        self.assertIsNone(abrupt.training_push)
        self.assertEqual(recovery.training_push.time_range_s, (3.0, 7.0))
        self.assertEqual(recovery.training_push.planar_delta_speed_range_m_s, (0.3, 0.7))

    def test_rejects_missing_training_push(self):
        payload = json.loads((CONFIG_DIR / "push_recovery.yaml").read_text(encoding="utf-8"))
        payload["training"]["push"] = None
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "push_recovery.yaml").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training push"):
                scenario_module.load_scenario("push_recovery", Path(folder))

    def test_rejects_unsorted_command_times(self):
        payload = json.loads((CONFIG_DIR / "abrupt.yaml").read_text(encoding="utf-8"))
        payload["evaluation"]["command_schedule"][1]["time_s"] = 0
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "abrupt.yaml").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                scenario_module.load_scenario("abrupt", Path(folder))

    def test_rejects_nonfinite_and_reversed_ranges(self):
        payload = json.loads((CONFIG_DIR / "nominal.yaml").read_text(encoding="utf-8"))
        payload["training"]["lin_vel_x_m_s"] = [1.0, 0.0]
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "nominal.yaml"
            file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lower bound"):
                scenario_module.load_scenario("nominal", Path(folder))
            payload["training"]["lin_vel_x_m_s"] = [0.0, float("inf")]
            file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                scenario_module.load_scenario("nominal", Path(folder))


if __name__ == "__main__":
    unittest.main()
