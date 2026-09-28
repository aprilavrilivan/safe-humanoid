"""Planning must enumerate the matrix without starting Isaac Sim or writing files."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "reward_matrix.py"


class TestRewardMatrix(unittest.TestCase):
    def test_plan_has_all_twelve_distinct_train_and_evaluate_pairs(self):
        batch = ROOT / "logs" / "rsl_rl" / "reward_matrix" / "unit-plan-only"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "plan", "--batch-name", "unit-plan-only", "--seeds", "0"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.count("--reward-profile"), 12)
        self.assertEqual(result.stdout.count("evaluate_all.py"), 12)
        self.assertIn("--scenario abrupt --reward-profile frequency", result.stdout)
        self.assertIn("--scenario push_recovery --reward-profile time", result.stdout)
        self.assertFalse(batch.exists())

    def test_duplicate_seeds_rejected(self):
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "plan", "--batch-name", "unit-plan-only",
                "--seeds", "0", "0",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
