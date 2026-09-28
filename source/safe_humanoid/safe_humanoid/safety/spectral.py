"""Episode-separated, physics-rate Welch band powers (no simulator dependency)."""

from __future__ import annotations

import cmath
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safe_humanoid.telemetry.physics_trace import PhysicsSample


@dataclass(frozen=True)
class SpectralConfig:
    sample_rate_hz: float
    window_samples: int
    hop_samples: int
    bands_hz: dict[str, tuple[float, float]]
    source: Path


def load_spectral_config(path: str | Path) -> SpectralConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "sample_rate_hz", "window_samples", "hop_samples", "bands_hz", "notes"
    } or payload["schema_version"] != 1:
        raise ValueError("invalid frequency-domain config")
    fs = payload["sample_rate_hz"]
    size = payload["window_samples"]
    hop = payload["hop_samples"]
    if isinstance(fs, bool) or not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if isinstance(size, bool) or not isinstance(size, int) or size < 4 or size & (size - 1):
        raise ValueError("window_samples must be a power of two >= 4")
    if isinstance(hop, bool) or not isinstance(hop, int) or not 1 <= hop <= size:
        raise ValueError("hop_samples must lie within the FFT window")
    if not isinstance(payload["notes"], str) or not payload["notes"]:
        raise ValueError("frequency-domain notes must explain the analysis settings")
    raw_bands = payload["bands_hz"]
    if not isinstance(raw_bands, dict) or not raw_bands:
        raise ValueError("bands_hz must be a nonempty mapping")
    bands = {}
    last_upper = 0.0
    for name, edges in raw_bands.items():
        if not isinstance(name, str) or not name or not isinstance(edges, list) or len(edges) != 2:
            raise ValueError("each frequency band needs a name and [lower, upper] edges")
        if any(isinstance(edge, bool) or not isinstance(edge, (int, float)) for edge in edges):
            raise ValueError("frequency band edges must be numbers")
        lower, upper = map(float, edges)
        if not all(math.isfinite(edge) for edge in (lower, upper)) or not (
            0 <= last_upper <= lower < upper <= fs / 2
        ):
            raise ValueError("frequency bands must be ordered, nonoverlapping and below Nyquist")
        bands[name] = (lower, upper)
        last_upper = upper
    return SpectralConfig(float(fs), size, hop, bands, source)


def _fft(values: list[complex]) -> list[complex]:
    """Small radix-two FFT so synthetic tests also run on a bare Python install."""

    size = len(values)
    result = list(values)
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]
    length = 2
    while length <= size:
        root = cmath.exp(-2j * math.pi / length)
        half = length // 2
        for start in range(0, size, length):
            factor = 1.0 + 0j
            for offset in range(half):
                even = result[start + offset]
                odd = factor * result[start + offset + half]
                result[start + offset] = even + odd
                result[start + offset + half] = even - odd
                factor *= root
        length *= 2
    return result


def welch_band_powers(
    values: list[float], config: SpectralConfig
) -> tuple[dict[str, float], int] | None:
    """Return mean one-sided band power in signal-unit squared, or None if too short.

    Each segment is demeaned, Hann-windowed, and normalized by window energy.
    Bands are half-open except that the final band includes its upper Nyquist bin.
    """

    size = config.window_samples
    if len(values) < size:
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("spectral signal contains NaN or Inf")
    window = [0.5 - 0.5 * math.cos(2 * math.pi * i / (size - 1)) for i in range(size)]
    window_energy = sum(value * value for value in window)
    bins = {name: [] for name in config.bands_hz}
    last_name = next(reversed(config.bands_hz))
    for k in range(size // 2 + 1):
        frequency = k * config.sample_rate_hz / size
        for name, (lower, upper) in config.bands_hz.items():
            if lower <= frequency < upper or (name == last_name and frequency == upper):
                bins[name].append(k)
                break
    totals = {name: 0.0 for name in config.bands_hz}
    windows = 0
    for start in range(0, len(values) - size + 1, config.hop_samples):
        segment = values[start:start + size]
        mean = sum(segment) / size
        spectrum = _fft([
            complex((value - mean) * weight) for value, weight in zip(segment, window)
        ])
        for name, indices in bins.items():
            for k in indices:
                one_sided = 1 if k in (0, size // 2) else 2
                totals[name] += one_sided * abs(spectrum[k]) ** 2 / (size * window_energy)
        windows += 1
    return {name: power / windows for name, power in totals.items()}, windows


def analyze_spectral(
    samples: list[PhysicsSample], config: SpectralConfig, joint_names: list[str]
) -> dict:
    if not samples or not joint_names:
        raise ValueError("spectral analysis needs samples and joint names")
    grouped = defaultdict(list)
    for sample in samples:
        if (
            len(sample.estimated_actuator_torque_nm) != len(joint_names)
            or len(sample.joint_velocity_rad_s) != len(joint_names)
        ):
            raise ValueError("spectral sample joint count differs from metadata")
        grouped[(sample.env_id, sample.episode_id)].append(sample)
    episodes = []
    aggregate_sums = defaultdict(float)
    aggregate_max = defaultdict(float)
    aggregate_weights = defaultdict(int)
    for (env_id, episode_id), group in sorted(grouped.items()):
        result = {"env_id": env_id, "episode_id": episode_id, "samples": len(group), "windows": 0}
        if len(group) >= config.window_samples:
            for signal_name, field in (
                ("estimated_torque", "estimated_actuator_torque_nm"),
                ("joint_speed", "joint_velocity_rad_s"),
            ):
                for joint_index, _joint_name in enumerate(joint_names):
                    analysis = welch_band_powers(
                        [getattr(sample, field)[joint_index] for sample in group], config
                    )
                    assert analysis is not None
                    powers, windows = analysis
                    result["windows"] = windows
                    for band, power in powers.items():
                        key = f"{signal_name}_{band}_power"
                        aggregate_sums[key] += power * windows
                        aggregate_weights[key] += windows
                        aggregate_max[key] = max(aggregate_max[key], power)
        episodes.append(result)
    aggregate = {"episode_count": len(episodes), "episodes_with_full_window": sum(
        item["windows"] > 0 for item in episodes
    ), "window_samples": config.window_samples, "hop_samples": config.hop_samples}
    for signal_name, unit in (("estimated_torque", "nm2"), ("joint_speed", "rad2_s2")):
        for band in config.bands_hz:
            key = f"{signal_name}_{band}_power"
            aggregate[f"mean_{key}_{unit}"] = (
                aggregate_sums[key] / aggregate_weights[key] if aggregate_weights[key] else None
            )
            aggregate[f"peak_joint_{key}_{unit}"] = (
                aggregate_max[key] if aggregate_weights[key] else None
            )
    return {"aggregate": aggregate, "episodes": episodes}
