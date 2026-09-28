#!/usr/bin/env python3
"""Plan or explicitly launch the 4-scenario × 3-reward PPO training matrix.

``plan`` is read-only. ``run`` trains in waves of at most four on a Linux GPU.
``evaluate`` evaluates every completed checkpoint on all four held-out scripts.
Evaluation is deliberately separate so a failed/weak training run is not
silently treated as a valid safety comparison.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("nominal", "aggressive", "abrupt", "push_recovery")
PROFILES = ("clean", "time", "frequency")


def jobs(seeds: list[int]):
    for seed in seeds:
        for profile in PROFILES:
            for scenario in SCENARIOS:
                yield scenario, profile, seed


def command(scenario: str, profile: str, seed: int, run_dir: Path, args) -> list[str]:
    result = [
        str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "train.py"),
        "--scenario", scenario, "--reward-profile", profile,
        "--seed", str(seed), "--num-envs", str(args.num_envs),
        "--max-iterations", str(args.max_iterations), "--run-dir", str(run_dir),
        "--logger", "wandb", "--wandb-entity", args.wandb_entity,
        "--wandb-project", args.wandb_project,
    ]
    if args.offline:
        result.append("--wandb-offline")
    return result


def evaluation_command(checkpoint: Path, output_root: Path, args) -> list[str]:
    result = [
        str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "evaluate_all.py"),
        "--checkpoint", str(checkpoint), "--output-root", str(output_root),
        "--num-envs", str(args.eval_num_envs), "--episodes", str(args.eval_episodes),
        "--seed", str(args.eval_seed),
    ]
    if not args.offline:
        result.append("--wandb")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "evaluate"))
    parser.add_argument("--batch-name", required=True, help="unique experiment-batch name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=10000, help="held-out evaluation seed")
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--wandb-entity", default="ivanxu-uc-berkeley")
    parser.add_argument("--wandb-project", default="safe-humanoid")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if (
        not args.batch_name or not args.batch_name.replace("-", "").replace("_", "").isalnum()
        or any(seed < 0 for seed in args.seeds) or len(args.seeds) != len(set(args.seeds))
        or args.num_envs <= 0 or args.max_iterations <= 0 or not 1 <= args.parallel <= 4
        or args.eval_seed < 0 or not 1 <= args.eval_num_envs <= 8 or args.eval_episodes <= 0
    ):
        parser.error(
            "use a simple batch name, nonnegative seeds, positive budget and 1..4 parallel jobs"
        )
    batch_dir = ROOT / "logs" / "rsl_rl" / "reward_matrix" / args.batch_name
    planned = [
        (scenario, profile, seed, batch_dir / f"{scenario}_{profile}_seed{seed}")
        for scenario, profile, seed in jobs(args.seeds)
    ]
    if args.action == "plan":
        for scenario, profile, seed, run_dir in planned:
            print(shlex.join(command(scenario, profile, seed, run_dir, args)))
            checkpoint = run_dir / f"model_{args.max_iterations - 1}.pt"
            output = ROOT / "outputs" / "reward_matrix" / args.batch_name / run_dir.name
            print("  then:", shlex.join(evaluation_command(checkpoint, output, args)))
        print(
            f"[PLAN] {len(planned)} training runs; at most {args.parallel} concurrent; "
            "no files changed"
        )
        return 0

    if platform.system() != "Linux" or shutil.which("nvidia-smi") is None:
        parser.error(f"{args.action} requires a Linux NVIDIA GPU host")
    if shutil.which("script") is None or not (ROOT / ".venv" / "bin" / "python").is_file():
        parser.error(f"{args.action} requires util-linux script and the bootstrapped .venv")
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        parser.error("accept the NVIDIA EULA yourself before a noninteractive batch")
    if not args.offline and not os.environ.get("WANDB_API_KEY"):
        parser.error("set WANDB_API_KEY securely in the shell before an online W&B batch")
    if args.action == "evaluate":
        if not batch_dir.is_dir():
            parser.error(f"training batch does not exist: {batch_dir}")
        output_batch = ROOT / "outputs" / "reward_matrix" / args.batch_name
        if output_batch.exists():
            parser.error(f"evaluation batch already exists: {output_batch}")
        missing = [
            run_dir / f"model_{args.max_iterations - 1}.pt"
            for _, _, _, run_dir in planned
            if not (run_dir / f"model_{args.max_iterations - 1}.pt").is_file()
        ]
        if missing:
            print(
                f"[FAIL] Missing {len(missing)} checkpoints; first: {missing[0]}",
                file=sys.stderr,
            )
            return 1
        output_batch.mkdir(parents=True)
        for scenario, profile, seed, run_dir in planned:
            checkpoint = run_dir / f"model_{args.max_iterations - 1}.pt"
            output_root = output_batch / run_dir.name
            log_path = output_batch / f"{run_dir.name}.console.log"
            with log_path.open("w", encoding="utf-8") as handle:
                status = subprocess.run(
                    ["script", "-q", "-e", "-c", shlex.join(
                        evaluation_command(checkpoint, output_root, args)
                    ), "/dev/null"],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, check=False,
                ).returncode
            print(f"[{status}] {log_path}", flush=True)
            if status:
                print(
                    "[FAIL] Stopping evaluation; completed artifacts remain intact.",
                    file=sys.stderr,
                )
                return 1
        print(f"[PASS] Evaluation matrix complete: {output_batch}")
        return 0

    if batch_dir.exists():
        parser.error(f"batch directory already exists: {batch_dir}")
    batch_dir.mkdir(parents=True)
    for start in range(0, len(planned), args.parallel):
        wave = planned[start:start + args.parallel]
        running = []
        for scenario, profile, seed, run_dir in wave:
            log_path = batch_dir / f"{scenario}_{profile}_seed{seed}.console.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    "script", "-q", "-e", "-c",
                    shlex.join(command(scenario, profile, seed, run_dir, args)), "/dev/null",
                ],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            )
            running.append((process, handle, log_path))
        failures = []
        for process, handle, log_path in running:
            status = process.wait()
            handle.close()
            print(f"[{status}] {log_path}", flush=True)
            if status:
                failures.append(log_path)
        if failures:
            print(
                "[FAIL] Stopping before the next wave; completed runs and logs remain intact.",
                file=sys.stderr,
            )
            return 1
    print(f"[PASS] Training matrix complete: {batch_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
