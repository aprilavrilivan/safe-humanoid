"""Offline validation for the three PPO reward variants."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "source" / "safe_humanoid" / "safe_humanoid" / "reward_profiles.py"
SPEC = importlib.util.spec_from_file_location("reward_profiles_for_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profiles = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profiles
SPEC.loader.exec_module(profiles)
MANIFEST = ROOT / "configs" / "experiments" / "velocity_reward_ppo.yaml"


class TestRewardProfiles(unittest.TestCase):
    def test_three_profiles_and_offline_window_alignment(self):
        clean, temporal, spectral = (
            profiles.load_reward_profile(name, MANIFEST)
            for name in profiles.REWARD_PROFILES
        )
        self.assertEqual(clean.weight, 0)
        self.assertEqual(temporal.window_samples, 100)
        self.assertEqual(spectral.window_samples, 128)
        self.assertEqual(spectral.hop_samples, 64)
        self.assertEqual(spectral.band_hz, (25.0, 100.0))
        self.assertLess(temporal.weight, 0)
        self.assertLess(spectral.weight, 0)
        time_analysis = json.loads((ROOT / "configs/safety/time_domain.yaml").read_text())
        frequency_analysis = json.loads((ROOT / "configs/safety/frequency_domain.yaml").read_text())
        self.assertEqual(temporal.sample_rate_hz, time_analysis["sample_rate_hz"])
        self.assertEqual(temporal.window_samples, round(
            time_analysis["sample_rate_hz"] * time_analysis["torque_rms_window_s"]
        ))
        self.assertEqual(spectral.sample_rate_hz, frequency_analysis["sample_rate_hz"])
        self.assertEqual(spectral.window_samples, frequency_analysis["window_samples"])
        self.assertEqual(spectral.hop_samples, frequency_analysis["hop_samples"])
        self.assertEqual(spectral.band_hz, tuple(frequency_analysis["bands_hz"]["high_25_100"]))

    def test_invalid_weight_and_frequency_band_are_rejected(self):
        payload = json.loads(MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reward.yaml"
            payload["profiles"]["time"]["weight"] = 0.0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "negative"):
                profiles.load_reward_profile("time", path)
            payload["profiles"]["frequency"]["band_hz"] = [25.0, 101.0]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Nyquist"):
                profiles.load_reward_profile("frequency", path)


if __name__ == "__main__":
    unittest.main()
