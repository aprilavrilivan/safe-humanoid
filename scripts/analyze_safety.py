#!/usr/bin/env python3
"""Analyze an existing physics trace without Isaac Sim or a GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.safety.analysis import analyze_run  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
parser.add_argument("--safety-config-dir", type=Path, default=ROOT / "configs" / "safety")
args = parser.parse_args()
summary = analyze_run(args.run_dir, args.safety_config_dir)
print(
    f"[PASS] {args.run_dir / 'safety_summary.json'}: "
    f"{summary['temporal_aggregate']['physics_samples']} physics samples at "
    f"{summary['sample_rate_hz']:g} Hz; hardware limits {summary['hardware_limit_status']}"
)
