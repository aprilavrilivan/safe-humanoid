"""GPU simulator regression test for the project G1 tensor contract."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
import unittest
from pathlib import Path

from simulator_subprocess import run_simulator

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_env.py"


def simulator_available() -> bool:
    """Return whether this host can plausibly launch the pinned GPU simulator."""

    return (
        platform.system() == "Linux"
        and shutil.which("nvidia-smi") is not None
        and importlib.util.find_spec("isaaclab") is not None
        and importlib.util.find_spec("safe_humanoid") is not None
    )


class TestG1EnvironmentContract(unittest.TestCase):
    """Exercise the smoke test in an isolated Isaac Sim process."""

    @unittest.skipUnless(
        simulator_available(),
        "requires installed Linux Isaac Lab GPU stack",
    )
    def test_project_g1_matches_official_tensor_contract(self) -> None:
        command = [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--headless",
            "--num-envs",
            "2",
            "--steps",
            "4",
            "--seed",
            "0",
        ]
        result = run_simulator(command, cwd=REPO_ROOT)
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"smoke test failed with exit code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
        )
        self.assertIn(
            "[PASS] completed 4 zero-action steps with a valid tensor contract",
            result.stdout,
            msg=f"smoke test exited before completing steps\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
