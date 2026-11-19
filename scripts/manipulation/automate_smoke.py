#!/usr/bin/env python3
"""Launch the upstream AutoMate assembly task with strict Isaac startup order."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Bounded AutoMate assembly inspection run")
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--assembly-id", default="00015")
parser.add_argument("--assembly-metadata", type=Path)
parser.add_argument("--automate-data-dir", type=Path)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0 or args.num_envs <= 0 or args.num_envs > 4:
    parser.error("steps must be positive and 1 <= num-envs <= 4")
if args.assembly_id != "00015" and not args.assembly_metadata:
    parser.error("non-default assembly IDs require inspected --assembly-metadata")
args.task = "automate_assembly"
args.factory_asset_dir = None
args.enable_cameras = args.video

launcher = AppLauncher(args)

# Isaac-facing and project imports must follow AppLauncher initialization.
from isaac_smoke import run_with_started_app  # noqa: E402


if __name__ == "__main__":
    run_with_started_app(args, launcher.app)
