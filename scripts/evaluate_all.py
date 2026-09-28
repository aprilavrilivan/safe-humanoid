#!/usr/bin/env python3
"""Run the same checkpoint on all four scenarios in separate Isaac Sim processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-root", type=Path, default=None)
parser.add_argument("--wandb", action="store_true", help="append four summaries to the training W&B run")
args = parser.parse_args()
if not 0 < args.num_envs <= 8 or args.episodes <= 0:
    parser.error("--num-envs must be 1..8 and --episodes must be positive")
checkpoint = args.checkpoint.expanduser().resolve()
if not checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {checkpoint}")

output_root = args.output_root or ROOT / "outputs" / "evaluation" / datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S-%f"
)
if output_root.exists():
    parser.error(f"output root already exists: {output_root}")
output_root.mkdir(parents=True)
run_dirs = []
for scenario in ("nominal", "aggressive", "abrupt", "push_recovery"):
    run_dir = output_root / scenario
    command = [
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--checkpoint", str(checkpoint), "--scenario", scenario,
        "--num-envs", str(args.num_envs), "--episodes", str(args.episodes),
        "--seed", str(args.seed), "--output-dir", str(run_dir),
    ]
    if args.wandb:
        command.append("--wandb")
    subprocess.run(command, cwd=ROOT, check=True)
    run_dirs.append(run_dir)

subprocess.run([
    sys.executable, str(ROOT / "scripts" / "report.py"),
    *(str(run_dir) for run_dir in run_dirs),
    "--output", str(output_root / "comparison.csv"),
], cwd=ROOT, check=True)
print(f"[PASS] Four-scenario comparison: {output_root}")
