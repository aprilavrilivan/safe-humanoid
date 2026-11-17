#!/usr/bin/env python3
"""Rank three geometrically different AutoMate assets after their OBJ files exist."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "safe_humanoid"))
from safe_humanoid.manipulation.automate_assets import write_selection  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asset_root", type=Path, help="directory containing 5-digit AutoMate asset folders")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for asset in write_selection(args.asset_root, args.output):
        print(asset)
    print("Geometry-only candidates: inspect rendered assets and metadata before using these IDs.")


if __name__ == "__main__":
    main()
