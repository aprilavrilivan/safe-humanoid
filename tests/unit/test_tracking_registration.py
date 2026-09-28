"""Check tracking Gym registrations without Isaac Sim."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_FILE = (
    ROOT / "source" / "safe_humanoid" / "safe_humanoid" / "tasks"
    / "manager_based" / "tracking" / "config" / "g1" / "__init__.py"
)


class TestTrackingRegistrations(unittest.TestCase):
    def test_two_distinct_tasks_have_env_and_agent_configs(self):
        registrations = []
        fake_gym = types.ModuleType("gymnasium")
        fake_gym.register = lambda **kwargs: registrations.append(kwargs)
        spec = importlib.util.spec_from_file_location("test_tracking_registry", REGISTRATION_FILE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"gymnasium": fake_gym}):
            spec.loader.exec_module(module)
        self.assertEqual(len(registrations), 2)
        self.assertNotEqual(registrations[0]["id"], registrations[1]["id"])
        for registration in registrations:
            self.assertIn(
                "SafeHumanoidG1TrackingEnvCfg", registration["kwargs"]["env_cfg_entry_point"]
            )
            self.assertIn("rsl_rl_cfg_entry_point", registration["kwargs"])


if __name__ == "__main__":
    unittest.main()
