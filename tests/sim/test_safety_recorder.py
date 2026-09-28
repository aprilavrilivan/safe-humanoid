"""Linux GPU integration check for physics-rate recording and analysis."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from simulator_subprocess import run_simulator

ROOT = Path(__file__).resolve().parents[2]


def simulator_available() -> bool:
    return (
        platform.system() == "Linux"
        and shutil.which("nvidia-smi") is not None
        and importlib.util.find_spec("isaaclab") is not None
        and importlib.util.find_spec("safe_humanoid") is not None
    )


class TestSafetyRecorder(unittest.TestCase):
    @unittest.skipUnless(simulator_available(), "requires installed Linux Isaac Lab GPU stack")
    def test_physics_trace_and_offline_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_simulator([
                sys.executable, str(ROOT / "scripts" / "smoke_safety.py"), "--headless",
                "--num-envs", "2", "--steps", "32", "--output-dir", folder,
            ], cwd=ROOT)
        self.assertEqual(
            result.returncode, 0,
            msg=f"safety smoke failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn(
            "[PASS] 256 physics samples at 200 Hz",
            result.stdout,
            msg=f"safety smoke exited before analysis\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
