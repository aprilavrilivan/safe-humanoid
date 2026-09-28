"""GPU simulator smoke checks for every registered clean G1 scenario."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from test_env_contract import simulator_available
from simulator_subprocess import run_simulator

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(simulator_available(), "requires installed Linux Isaac Lab GPU stack")
class TestG1Scenarios(unittest.TestCase):
    def test_all_scenarios_create_reset_and_step(self):
        for scenario in ("nominal", "aggressive", "abrupt", "push_recovery"):
            with self.subTest(scenario=scenario):
                result = run_simulator(
                    [sys.executable, str(ROOT / "scripts" / "smoke_env.py"),
                     "--headless", "--scenario", scenario, "--num-envs", "2", "--steps", "4"],
                    cwd=ROOT,
                )
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                self.assertIn(
                    "[PASS] completed 4 zero-action steps with a valid tensor contract",
                    result.stdout,
                    msg=result.stdout + "\n" + result.stderr,
                )

    def test_scripted_command_and_push_timing(self):
        for scenario, steps, final_time_s in (
            ("abrupt", 405, 8),
            ("push_recovery", 255, 5),
        ):
            with self.subTest(scenario=scenario):
                result = run_simulator(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "smoke_env.py"),
                        "--headless",
                        "--scenario",
                        scenario,
                        "--exercise-timing",
                        "--num-envs",
                        "1",
                        "--steps",
                        str(steps),
                    ],
                    cwd=ROOT,
                )
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                self.assertIn(
                    f"[PASS] scripted timing reached {final_time_s}s without an episode reset",
                    result.stdout,
                    msg=result.stdout + "\n" + result.stderr,
                )


if __name__ == "__main__":
    unittest.main()
