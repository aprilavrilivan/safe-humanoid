"""Linux GPU regression checks for scenario training and online reward paths."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from simulator_subprocess import run_simulator
from test_env_contract import simulator_available

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(simulator_available(), "requires installed Linux Isaac Lab GPU stack")
class TestTrainingSafetyPaths(unittest.TestCase):
    def test_training_scenarios_and_reward_profiles(self):
        cases = (
            ("abrupt", "clean", 400),
            ("push_recovery", "clean", 400),
            ("nominal", "time", 150),
            ("nominal", "frequency", 150),
        )
        for scenario, profile, steps in cases:
            with self.subTest(scenario=scenario, profile=profile):
                result = run_simulator([
                    sys.executable, str(ROOT / "scripts" / "smoke_training.py"),
                    "--headless", "--scenario", scenario, "--reward-profile", profile,
                    "--num-envs", "2", "--steps", str(steps),
                ], cwd=ROOT)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    f"[PASS] training scenario={scenario} reward={profile}",
                    result.stdout, msg=result.stdout + result.stderr,
                )


if __name__ == "__main__":
    unittest.main()
