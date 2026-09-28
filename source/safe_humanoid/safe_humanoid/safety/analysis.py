"""Offline safety analysis of a versioned physics-step trace."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from safe_humanoid.safety.limits import load_g1_limits
from safe_humanoid.safety.spectral import analyze_spectral, load_spectral_config
from safe_humanoid.safety.temporal import analyze_temporal, load_temporal_config
from safe_humanoid.telemetry.physics_trace import read_physics_trace


def analyze_run(run_dir: str | Path, safety_config_dir: str | Path) -> dict:
    """Write safety_summary.json; fail closed on schema or sampling mismatches."""

    run_dir = Path(run_dir)
    config_dir = Path(safety_config_dir)
    metadata, samples = read_physics_trace(run_dir)
    temporal_path = config_dir / "time_domain.yaml"
    spectral_path = config_dir / "frequency_domain.yaml"
    limits_path = config_dir / "g1_limits.yaml"
    temporal_cfg = load_temporal_config(temporal_path)
    spectral_cfg = load_spectral_config(spectral_path)
    limits = load_g1_limits(limits_path)
    sample_rate = metadata["sample_rate_hz"]
    if not math.isclose(sample_rate, temporal_cfg.sample_rate_hz) or not math.isclose(
        sample_rate, spectral_cfg.sample_rate_hz
    ):
        raise ValueError("trace sample rate disagrees with time/frequency analysis config")
    if metadata["asset"] != limits.robot_asset:
        raise ValueError("G1 limit configuration targets a different asset")
    bounds = limits.hardware_arrays(metadata["joint_names"])
    temporal = analyze_temporal(samples, temporal_cfg, bounds)
    spectral = analyze_spectral(samples, spectral_cfg, metadata["joint_names"])
    if [(row["env_id"], row["episode_id"]) for row in temporal["episodes"]] != [
        (row["env_id"], row["episode_id"]) for row in spectral["episodes"]
    ]:
        raise ValueError("time and frequency analyses disagree on episode identities")
    warning = (
        "Isaac Lab G1_MINIMAL uses implicit actuators: torque and torque-derived power "
        "are approximate simulator-model signals, not measured motor or solver torque. "
        "No physical torque violation rate can be claimed from them."
    )
    if bounds is None:
        warning += " Uncalibrated hardware limits also prohibit physical speed-threshold claims."
    else:
        warning += " Speed comparisons use the fully sourced per-joint hardware map."
    result = {
        "schema_version": 1,
        "asset": metadata["asset"],
        "sample_rate_hz": sample_rate,
        "signal_semantics": metadata["signal_semantics"],
        "hardware_limit_status": limits.calibration_status,
        "physical_torque_violation_claim_available": False,
        "physical_speed_threshold_comparison_available": bounds is not None,
        "warning": warning,
        "config_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (temporal_path, spectral_path, limits_path)
        },
        "temporal_aggregate": temporal["aggregate"],
        "frequency_aggregate": spectral["aggregate"],
        "temporal_episodes": temporal["episodes"],
        "frequency_episodes": spectral["episodes"],
    }
    (run_dir / "safety_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
