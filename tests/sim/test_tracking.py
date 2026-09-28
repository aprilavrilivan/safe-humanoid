"""Optional Linux GPU tracking smoke when a real retargeted motion is supplied."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
import unittest
from pathlib import Path

from simulator_subprocess import run_simulator

ROOT = Path(__file__).resolve().parents[2]


class TestTrackingEnvironment(unittest.TestCase):
    @unittest.skipUnless(
        platform.system() == "Linux"
        and shutil.which("nvidia-smi") is not None
        and importlib.util.find_spec("isaaclab") is not None,
        "requires installed Linux Isaac Lab GPU stack",
    )
    def test_available_reference_motions(self):
        configured = [
            ("squat_stand", os.environ.get("SAFE_HUMANOID_SQUAT_MOTION")),
            ("fast_leg_swing", os.environ.get("SAFE_HUMANOID_LEG_SWING_MOTION")),
        ]
        available = [(name, path) for name, path in configured if path]
        if not available:
            self.skipTest("set SAFE_HUMANOID_SQUAT_MOTION or SAFE_HUMANOID_LEG_SWING_MOTION")
        for name, path in available:
            with self.subTest(scenario=name):
                result = run_simulator(
                    [
                        sys.executable, str(ROOT / "scripts" / "smoke_tracking.py"),
                        "--headless", "--scenario", name, "--motion", path,
                        "--num-envs", "2", "--steps", "4",
                    ],
                    cwd=ROOT,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(
                    "reset + 4 zero-action steps",
                    result.stdout,
                    msg=result.stdout + result.stderr,
                )


if __name__ == "__main__":
    unittest.main()
