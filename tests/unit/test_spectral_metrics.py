"""Known-sinusoid and sampling-rate tests for spectral bands, windows, and normalization."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.safety.spectral import (  # noqa: E402
    analyze_spectral,
    load_spectral_config,
    welch_band_powers,
)
from safe_humanoid.telemetry.physics_trace import PhysicsSample  # noqa: E402


class TestSpectralMetrics(unittest.TestCase):
    def setUp(self):
        self.config = load_spectral_config(ROOT / "configs" / "safety" / "frequency_domain.yaml")

    def test_40_hz_sine_is_high_band_at_200_hz(self):
        signal = [math.sin(2 * math.pi * 40 * step / 200) for step in range(256)]
        result = welch_band_powers(signal, self.config)
        self.assertIsNotNone(result)
        powers, windows = result
        self.assertEqual(windows, 3)
        self.assertAlmostEqual(powers["high_25_100"], 0.5, delta=0.01)
        self.assertLess(powers["low_0_10"], 0.001)
        self.assertIsNone(welch_band_powers(signal[:127], self.config))
        constant, _ = welch_band_powers([7.0] * 128, self.config)
        self.assertTrue(all(abs(power) < 1e-12 for power in constant.values()))

    def test_episodes_are_not_fft_spliced(self):
        samples = [
            PhysicsSample(0, episode, step, step / 200, (1.0,), (1.0,))
            for episode in (0, 1) for step in range(1, 81)
        ]
        result = analyze_spectral(samples, self.config, ["joint"])
        self.assertEqual(result["aggregate"]["episodes_with_full_window"], 0)
        self.assertIsNone(result["aggregate"]["mean_estimated_torque_high_25_100_power_nm2"])

    def test_invalid_nyquist_band_rejected(self):
        payload = json.loads((ROOT / "configs" / "safety" / "frequency_domain.yaml").read_text())
        payload["sample_rate_hz"] = 50
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frequency.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Nyquist"):
                load_spectral_config(path)


if __name__ == "__main__":
    unittest.main()
