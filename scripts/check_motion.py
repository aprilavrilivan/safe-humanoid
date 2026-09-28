#!/usr/bin/env python3
"""Validate a retargeted G1 clip and its intended scenario without launching Isaac Sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.motion import load_motion  # noqa: E402
from safe_humanoid.tracking_scenarios import SCENARIO_NAMES, load_tracking_scenario  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=SCENARIO_NAMES)
    parser.add_argument("--motion", required=True, type=Path)
    args = parser.parse_args()
    try:
        clip = load_motion(args.motion)
        scenario = load_tracking_scenario(
            args.scenario, ROOT / "configs" / "task" / "tracking"
        )
        scenario.check_motion(clip)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"[FAIL] {exc}\n")
    print(
        f"[PASS] {scenario.name}: {clip.frame_count} frames, "
        f"{clip.fps:g} fps, {clip.duration_s:.3f} s, {len(clip.joint_names)} joints"
    )
    print("[NOTE] Exact joint match to the simulator is checked when creating the environment.")


if __name__ == "__main__":
    main()
