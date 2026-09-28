#!/usr/bin/env python3
"""Generate report figures for the reward-engineered PPO experiment matrix.

The script reads the compact archive copied from the training server.  It does
not require simulator traces or Isaac Lab.  Matplotlib and NumPy are the only
non-standard dependencies.

Example:
    python scripts/plot_reward_ppo_results.py \
        --archive outputs/archives/reward-ppo-v1-20260922/data \
        --output figs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "safe-humanoid-matplotlib-cache")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCENARIOS = ("nominal", "aggressive", "abrupt", "push_recovery")
PROFILES = ("clean", "time", "frequency")
SCENARIO_LABELS = {
    "nominal": "Nominal",
    "aggressive": "Aggressive",
    "abrupt": "Abrupt",
    "push_recovery": "Push recovery",
}
PROFILE_LABELS = {
    "clean": "Clean PPO",
    "time": "Time-domain PPO",
    "frequency": "Frequency-domain PPO",
}
PROFILE_COLORS = {
    "clean": "#4C78A8",
    "time": "#E45756",
    "frequency": "#59A14F",
}
SCENARIO_COLORS = {
    "nominal": "#4C78A8",
    "aggressive": "#F28E2B",
    "abrupt": "#E15759",
    "push_recovery": "#59A14F",
}
SCENARIO_MARKERS = {
    "nominal": "o",
    "aggressive": "s",
    "abrupt": "^",
    "push_recovery": "D",
}

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RUN_RE = re.compile(
    r"^(nominal|aggressive|abrupt|push_recovery)_"
    r"(clean|time|frequency)_seed(\d+)$"
)
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")

TRAIN_METRICS = {
    "Mean reward": "mean_reward",
    "Mean episode length": "mean_episode_length",
    "Episode_Reward/safety_cost": "safety_cost",
    "Episode_Reward/r_vx": "r_vx",
    "Episode_Reward/r_vy": "r_vy",
    "Episode_Reward/r_wz": "r_wz",
    "Episode_Reward/r_upright": "r_upright",
    "Metrics/base_velocity/error_vel_xy": "train_error_xy",
    "Metrics/base_velocity/error_vel_yaw": "train_error_yaw",
    "Episode_Termination/time_out": "timeout_fraction",
    "Episode_Termination/base_contact": "base_contact_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("outputs/archives/reward-ppo-v1-20260922/data"),
        help="Root of the extracted compact experiment archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs"),
        help="Directory that receives PNG figures.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300)."
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_training_runs(training_root: Path) -> list[dict]:
    runs: list[dict] = []
    for metadata_path in sorted(training_root.glob("*/metadata.json")):
        run_dir = metadata_path.parent
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        console_path = training_root / f"{run_dir.name}.console.log"
        if not console_path.is_file():
            raise FileNotFoundError(f"Missing training log: {console_path}")

        records: list[dict] = []
        current: dict | None = None
        text = ANSI_RE.sub("", console_path.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            iteration_match = ITER_RE.search(line)
            if iteration_match:
                if current is not None:
                    records.append(current)
                current = {"iteration": int(iteration_match.group(1))}
                continue
            if current is None or ":" not in line:
                continue
            label, value = line.rsplit(":", 1)
            label = label.strip()
            metric_name = TRAIN_METRICS.get(label)
            if metric_name is None:
                continue
            try:
                current[metric_name] = float(value.strip())
            except ValueError:
                pass
        if current is not None:
            records.append(current)

        records_by_iteration = {record["iteration"]: record for record in records}
        if len(records_by_iteration) != 1500:
            raise ValueError(
                f"Expected 1500 unique iterations in {console_path}, "
                f"found {len(records_by_iteration)}"
            )

        runs.append(
            {
                "train_scenario": match.group(1),
                "profile": match.group(2),
                "seed": int(match.group(3)),
                "records": [records_by_iteration[index] for index in range(1500)],
            }
        )

    if len(runs) != 36:
        raise ValueError(f"Expected 36 training runs, found {len(runs)}")
    return runs


def load_evaluations(evaluation_root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(evaluation_root.glob("*/*/summary.json")):
        run_name = summary_path.parent.parent.name
        match = RUN_RE.match(run_name)
        if not match:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        safety_path = summary_path.parent / "safety_summary.json"
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        episodes = safety["temporal_episodes"]
        total_episodes = summary["terminations"] + summary["timeouts"]
        row = {
            "train_scenario": match.group(1),
            "profile": match.group(2),
            "seed": int(match.group(3)),
            "eval_scenario": summary_path.parent.name,
            "termination_rate": summary["terminations"] / total_episodes,
            "mean_episode_duration_s": float(
                np.mean([episode["duration_s"] for episode in episodes])
            ),
            **summary,
            **safety["temporal_aggregate"],
            **safety["frequency_aggregate"],
        }
        rows.append(row)

    if len(rows) != 144:
        raise ValueError(f"Expected 144 evaluation rows, found {len(rows)}")
    return rows


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    valid = np.convolve(values, kernel, mode="valid")
    prefix = np.array(
        [np.mean(values[: index + 1]) for index in range(window - 1)], dtype=float
    )
    return np.concatenate([prefix, valid])


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi)
    plt.close(fig)


def profile_legend() -> list[Line2D]:
    return [
        Line2D([0], [0], color=PROFILE_COLORS[profile], lw=2.5, label=PROFILE_LABELS[profile])
        for profile in PROFILES
    ]


def scenario_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=SCENARIO_MARKERS[scenario],
            color="#333333",
            markerfacecolor="white",
            markeredgecolor="#333333",
            lw=0,
            markersize=7,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in SCENARIOS
    ]


def within_task_evaluations(evaluations: list[dict]) -> list[dict]:
    """Keep evaluations whose training and evaluation scenarios match."""
    return [
        row
        for row in evaluations
        if row["train_scenario"] == row["eval_scenario"]
    ]


def figure_training_convergence(
    training_runs: list[dict], output_dir: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), sharex=True)
    x = np.arange(1500)
    grouped = defaultdict(list)
    for run in training_runs:
        grouped[(run["train_scenario"], run["profile"])].append(run)

    for ax, scenario in zip(axes.flat, SCENARIOS, strict=True):
        for profile in PROFILES:
            runs = sorted(grouped[(scenario, profile)], key=lambda item: item["seed"])
            curves = np.vstack(
                [
                    moving_average(
                        np.array(
                            [
                                record["r_vx"]
                                + record["r_vy"]
                                + record["r_wz"]
                                + record["r_upright"]
                                for record in run["records"]
                            ]
                        ),
                        25,
                    )
                    for run in runs
                ]
            )
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1)
            color = PROFILE_COLORS[profile]
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
            ax.plot(x, mean, color=color, lw=1.8, label=PROFILE_LABELS[profile])
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xlim(0, 1499)
        ax.set_xticks([0, 300, 600, 900, 1200, 1500])
        ax.set_ylabel("Basic task reward terms (sum)")
        ax.set_xlabel("Training iteration")

    fig.suptitle("Basic task reward during PPO training", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "r_vx + r_vy + r_wz + r_upright; safety_cost is excluded. Lines show the 3-seed mean and bands show ±1 SD.",
        ha="center",
        color="#555555",
    )
    fig.legend(handles=profile_legend(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.895))
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.86))
    save_figure(fig, output_dir, "fig01_training_convergence", dpi)


def figure_pareto(evaluations: list[dict], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    internal = within_task_evaluations(evaluations)
    for ax, eval_scenario in zip(axes.flat, SCENARIOS, strict=True):
        subset = [row for row in internal if row["eval_scenario"] == eval_scenario]
        for row in subset:
            size = 36 + 150 * np.sqrt(row["termination_rate"])
            ax.scatter(
                row["mean_joint_torque_rms_nm"],
                row["mean_tracking_error_xy_m_s"],
                s=size,
                marker="o",
                color=PROFILE_COLORS[row["profile"]],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.82,
            )
        ax.set_title(f"{SCENARIO_LABELS[eval_scenario]}: trained and evaluated")
        ax.set_xlabel("Mean joint torque RMS (N·m) ↓")
        ax.set_ylabel("XY velocity tracking error (m/s) ↓")
        ax.annotate(
            "preferred region",
            xy=(0.02, 0.04),
            xycoords="axes fraction",
            color="#666666",
            fontsize=8,
        )

    profile_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PROFILE_COLORS[profile],
            markeredgecolor="white",
            markersize=8,
            label=PROFILE_LABELS[profile],
        )
        for profile in PROFILES
    ]
    fig.suptitle("Within-task safety–performance trade-off", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.935,
        "Each panel contains 3 reward profiles × 3 seeds from the same task. Larger points indicate more terminations.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.legend(
        handles=profile_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.905),
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.84))
    save_figure(fig, output_dir, "fig02_safety_performance_pareto", dpi)


def paired_changes(evaluations: list[dict], profile: str, metric: str) -> list[dict]:
    index = {
        (row["train_scenario"], row["profile"], row["seed"], row["eval_scenario"]): row
        for row in evaluations
    }
    result = []
    for row in evaluations:
        if row["profile"] != profile or row["train_scenario"] != row["eval_scenario"]:
            continue
        clean = index[(row["train_scenario"], "clean", row["seed"], row["eval_scenario"])]
        result.append(
            {
                **row,
                "relative_change_pct": 100.0 * (row[metric] - clean[metric]) / clean[metric],
            }
        )
    return result


def figure_time_paired_changes(
    evaluations: list[dict], output_dir: Path, dpi: int
) -> None:
    metrics = (
        ("mean_joint_torque_rms_nm", "Joint torque RMS"),
        ("mean_absolute_mechanical_power_w", "Mechanical power"),
        ("mean_tracking_error_xy_m_s", "XY tracking error"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.35))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        rows = sorted(
            paired_changes(evaluations, "time", metric),
            key=lambda item: item["relative_change_pct"],
        )
        x = np.arange(1, len(rows) + 1)
        y = np.array([row["relative_change_pct"] for row in rows])
        ax.plot(x, y, color="#999999", lw=1.0, zorder=1)
        for scenario in SCENARIOS:
            positions = [index + 1 for index, row in enumerate(rows) if row["train_scenario"] == scenario]
            values = [row["relative_change_pct"] for row in rows if row["train_scenario"] == scenario]
            ax.scatter(
                positions,
                values,
                s=27,
                marker=SCENARIO_MARKERS[scenario],
                color=SCENARIO_COLORS[scenario],
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
        median = float(np.median(y))
        lower_count = int(np.sum(y < 0))
        ax.axhline(0, color="#222222", lw=0.9)
        ax.axhline(median, color=PROFILE_COLORS["time"], lw=1.2, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Matched evaluation pair (sorted)")
        ax.set_ylabel("Change from clean PPO (%)")
        ax.text(
            0.03,
            0.96,
            f"Median {median:+.1f}%\nLower in {lower_count}/{len(y)} pairs",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )

    fig.suptitle("Time-domain penalty: within-task changes from clean PPO", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.91,
        "Each task contributes 3 matched seeds (12 pairs total). Negative values indicate a reduction.",
        ha="center",
        color="#555555",
    )
    fig.legend(handles=scenario_legend(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.86))
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.80))
    save_figure(fig, output_dir, "fig03_time_penalty_paired_changes", dpi)


def final_reward_ratios(training_runs: list[dict]) -> dict[tuple[str, str], list[float]]:
    result: dict[tuple[str, str], list[float]] = defaultdict(list)
    for run in training_runs:
        if run["profile"] == "clean":
            continue
        ratios = []
        for record in run["records"][-100:]:
            required = ("safety_cost", "r_vx", "r_vy", "r_wz", "r_upright")
            if any(name not in record for name in required):
                continue
            task_reward = sum(record[name] for name in required[1:])
            if task_reward > 0:
                ratios.append(100.0 * abs(record["safety_cost"]) / task_reward)
        result[(run["train_scenario"], run["profile"])].append(float(np.mean(ratios)))
    return result


def figure_frequency_diagnostics(
    evaluations: list[dict], training_runs: list[dict], output_dir: Path, dpi: int
) -> None:
    metrics = (
        (
            "mean_estimated_torque_high_25_100_power_nm2",
            "Torque high-band power ratio",
        ),
        (
            "mean_joint_speed_high_25_100_power_rad2_s2",
            "Joint-speed high-band power ratio",
        ),
        ("mean_tracking_error_xy_m_s", "XY tracking-error ratio"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, (metric, title) in zip(axes.flat[:3], metrics, strict=True):
        rows = paired_changes(evaluations, "frequency", metric)
        for scenario_index, scenario in enumerate(SCENARIOS):
            values = [
                1.0 + row["relative_change_pct"] / 100.0
                for row in rows
                if row["train_scenario"] == scenario
            ]
            jitter = np.linspace(-0.16, 0.16, len(values))
            ax.scatter(
                scenario_index + jitter,
                values,
                s=32,
                marker=SCENARIO_MARKERS[scenario],
                color=SCENARIO_COLORS[scenario],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.plot(
                [scenario_index - 0.22, scenario_index + 0.22],
                [np.median(values), np.median(values)],
                color="#222222",
                lw=1.5,
            )
        ax.axhline(1.0, color="#222222", lw=0.9, ls="--")
        ax.set_yscale("log", base=2)
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xticks(range(4), [SCENARIO_LABELS[item] for item in SCENARIOS], rotation=18)
        ax.set_ylabel("Frequency PPO / clean PPO")
        ax.set_title(title)
        ax.text(
            0.02,
            0.96,
            "below 1 is better",
            transform=ax.transAxes,
            va="top",
            color="#666666",
            fontsize=8,
        )

    ax = axes.flat[3]
    ratios = final_reward_ratios(training_runs)
    x = np.arange(len(SCENARIOS))
    width = 0.34
    for offset, profile in ((-width / 2, "time"), (width / 2, "frequency")):
        means = [np.mean(ratios[(scenario, profile)]) for scenario in SCENARIOS]
        stds = [np.std(ratios[(scenario, profile)], ddof=1) for scenario in SCENARIOS]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            color=PROFILE_COLORS[profile],
            alpha=0.78,
            capsize=3,
            label=PROFILE_LABELS[profile],
        )
        for scenario_index, scenario in enumerate(SCENARIOS):
            seed_values = ratios[(scenario, profile)]
            jitter = np.linspace(-0.035, 0.035, len(seed_values))
            ax.scatter(
                np.full(len(seed_values), scenario_index + offset) + jitter,
                seed_values,
                color="#222222",
                s=15,
                zorder=3,
            )
    ax.set_xticks(x, [SCENARIO_LABELS[item] for item in SCENARIOS], rotation=18)
    ax.set_ylabel("|Safety penalty| / basic reward (%)")
    ax.set_title("Penalty scale near the end of training")
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.96,
        "Mean over last 100 iterations\nBars: 3-seed mean ± SD",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.suptitle("Frequency-domain penalty: within-task diagnostics", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.93,
        "Each task contains 3 seed-matched ratios against clean PPO. Horizontal segments show medians.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.89))
    save_figure(fig, output_dir, "fig04_frequency_penalty_diagnostics", dpi)


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    fmt: str,
    cmap: str,
    colorbar_label: str,
    row_labels: list[str],
    column_labels: list[str],
) -> None:
    image = ax.imshow(matrix, cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=22, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Reward profile")
    ax.set_ylabel("Task")
    ax.set_title(title)
    midpoint = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value > midpoint else "#222222"
            ax.text(col, row, format(value, fmt), ha="center", va="center", color=text_color, fontsize=9)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
    colorbar.set_label(colorbar_label)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure_within_task_summary(evaluations: list[dict], output_dir: Path, dpi: int) -> None:
    internal = within_task_evaluations(evaluations)
    metric_specs = (
        ("termination_rate", "Termination rate", ".0%", "Reds", "Fraction of episodes"),
        ("mean_tracking_error_xy_m_s", "XY tracking error", ".3f", "Oranges", "m/s"),
        ("mean_joint_torque_rms_nm", "Joint torque RMS", ".1f", "Blues", "N·m"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.5))
    for ax, (metric, title, fmt, cmap, colorbar_label) in zip(axes, metric_specs, strict=True):
        matrix = np.zeros((len(SCENARIOS), len(PROFILES)), dtype=float)
        for row_index, scenario in enumerate(SCENARIOS):
            for col_index, profile in enumerate(PROFILES):
                values = [
                    row[metric]
                    for row in internal
                    if row["train_scenario"] == scenario and row["profile"] == profile
                ]
                matrix[row_index, col_index] = np.mean(values)
        heatmap(
            ax,
            matrix,
            title,
            fmt,
            cmap,
            colorbar_label,
            [SCENARIO_LABELS[item] for item in SCENARIOS],
            [PROFILE_LABELS[item].replace(" PPO", "") for item in PROFILES],
        )

    fig.suptitle("Within-task held-out evaluation summary", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.91,
        "Training and evaluation use the same task. Each cell is the mean across 3 seeds × 20 episodes.",
        ha="center",
        color="#555555",
    )
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.85))
    save_figure(fig, output_dir, "fig05_within_task_evaluation_summary", dpi)


def figure_termination_duration(evaluations: list[dict], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.02, 1.35]})
    internal = within_task_evaluations(evaluations)
    ax = axes[0]
    for scenario in SCENARIOS:
        subset = [row for row in internal if row["eval_scenario"] == scenario]
        for profile in PROFILES:
            rows = [row for row in subset if row["profile"] == profile]
            ax.scatter(
                [100 * row["termination_rate"] for row in rows],
                [row["mean_episode_duration_s"] for row in rows],
                color=SCENARIO_COLORS[scenario],
                marker={"clean": "o", "time": "s", "frequency": "^"}[profile],
                s=31,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.45,
            )
    ax.set_xlabel("Terminated episodes (%)")
    ax.set_ylabel("Mean episode duration (s)")
    ax.set_title("All 36 within-task evaluations")
    ax.set_xlim(-3, 103)
    ax.set_ylim(0, 20.8)
    ax.axhline(20, color="#666666", lw=0.8, ls="--")
    worst = min(internal, key=lambda row: row["mean_episode_duration_s"])
    ax.annotate(
        f"{SCENARIO_LABELS[worst['train_scenario']]}–{PROFILE_LABELS[worst['profile']].split()[0]}\n"
        f"on {SCENARIO_LABELS[worst['eval_scenario']]}: {worst['mean_episode_duration_s']:.2f}s",
        xy=(100 * worst["termination_rate"], worst["mean_episode_duration_s"]),
        xytext=(57, 5.3),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        fontsize=8,
        color="#444444",
    )

    ax = axes[1]
    abrupt_rows = [
        row
        for row in internal
        if row["train_scenario"] == "abrupt" and row["eval_scenario"] == "abrupt"
    ]
    positions = []
    labels = []
    for profile_index, profile in enumerate(PROFILES):
        position = profile_index
        positions.append(position)
        labels.append(PROFILE_LABELS[profile])
        rows = sorted(
            [row for row in abrupt_rows if row["profile"] == profile],
            key=lambda row: row["seed"],
        )
        values = [row["mean_episode_duration_s"] for row in rows]
        jitter = np.array([-0.10, 0.0, 0.10])
        ax.scatter(
            np.full(3, position) + jitter,
            values,
            s=[35 + 85 * row["termination_rate"] for row in rows],
            color=PROFILE_COLORS[profile],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.85,
            zorder=3,
        )
        ax.plot(
            [position - 0.18, position + 0.18],
            [np.mean(values), np.mean(values)],
            color="#222222",
            lw=1.4,
            zorder=4,
        )
    ax.axhspan(4.0, 5.5, color="#F1CE63", alpha=0.24, label="First command-change window")
    ax.axhline(20, color="#666666", lw=0.8, ls="--")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Mean episode duration (s)")
    ax.set_title("Abrupt-command evaluation")
    ax.set_ylim(0, 20.8)
    ax.legend(loc="lower left", fontsize=8)
    ax.text(
        0.98,
        0.04,
        "Point size indicates termination rate\nBlack segment shows the 3-seed mean",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    eval_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SCENARIO_COLORS[scenario],
            markeredgecolor="white",
            markersize=7,
            label=SCENARIO_LABELS[scenario],
        )
        for scenario in SCENARIOS
    ]
    profile_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="#444444",
            markerfacecolor="#BBBBBB",
            lw=0,
            markersize=6,
            label=PROFILE_LABELS[profile],
        )
        for profile, marker in (("clean", "o"), ("time", "s"), ("frequency", "^"))
    ]
    fig.suptitle("Within-task termination and episode duration", fontsize=14, fontweight="bold")
    fig.legend(
        handles=eval_handles + profile_handles,
        loc="upper center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.91),
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.84))
    save_figure(fig, output_dir, "fig06_termination_duration_bias", dpi)


def _bar_panel(ax: plt.Axes, rows: list[dict], metric: str, title: str, scale: float = 1.0) -> None:
    x = np.arange(len(PROFILES))
    means = []
    stds = []
    seed_values_by_profile = []
    for profile in PROFILES:
        seed_rows = sorted(
            [row for row in rows if row["profile"] == profile],
            key=lambda row: row["seed"],
        )
        if len(seed_rows) != 3:
            raise ValueError(f"Expected 3 seeds for {profile}/{metric}, found {len(seed_rows)}")
        values = np.array([float(row[metric]) * scale for row in seed_rows])
        seed_values_by_profile.append(values)
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)))

    bars = ax.bar(
        x,
        means,
        width=0.66,
        color=[PROFILE_COLORS[profile] for profile in PROFILES],
        edgecolor=[PROFILE_COLORS[profile] for profile in PROFILES],
        alpha=0.78,
        yerr=stds,
        capsize=4,
        error_kw={"ecolor": "#222222", "elinewidth": 1.1, "capthick": 1.1},
        zorder=2,
    )
    for profile_index, values in enumerate(seed_values_by_profile):
        jitter = np.array([-0.10, 0.0, 0.10])
        ax.scatter(
            np.full(3, profile_index) + jitter,
            values,
            s=27,
            color=PROFILE_COLORS[PROFILES[profile_index]],
            edgecolor="#222222",
            linewidth=0.55,
            zorder=3,
            clip_on=False,
        )

    ax.set_title(title, pad=7)
    ax.set_xticks(x, ["Clean", "Time", "Frequency"], fontsize=8)
    ax.set_xlim(-0.55, 2.55)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if metric == "termination_rate":
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        if max(means) == 0:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    2.0,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#444444",
                )
    else:
        upper = max(
            max(values) for values in seed_values_by_profile
        )
        upper = max(upper, max(mean + std for mean, std in zip(means, stds, strict=True)))
        ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)


def figure_task_dashboard(
    evaluations: list[dict], scenario: str, output_dir: Path, dpi: int
) -> None:
    rows = [
        row
        for row in evaluations
        if row["train_scenario"] == scenario and row["eval_scenario"] == scenario
    ]
    if len(rows) != 9:
        raise ValueError(f"Expected 9 within-task rows for {scenario}, found {len(rows)}")

    fig = plt.figure(figsize=(15.2, 7.4))
    grid = fig.add_gridspec(2, 4)
    legend_ax = fig.add_subplot(grid[0, 0])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Patch(facecolor=PROFILE_COLORS[profile], edgecolor="none", label=PROFILE_LABELS[profile])
            for profile in PROFILES
        ],
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
    )
    legend_ax.text(
        0.02,
        0.47,
        "Bars: mean\nDots: 3 seeds\nError: ±1 SD\nLower is better\nHF: 25–100 Hz",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
        linespacing=1.45,
    )

    metrics = (
        ("termination_rate", "Termination rate\n(%)", 100.0),
        ("mean_tracking_error_xy_m_s", "XY tracking error\n(m/s)", 1.0),
        ("mean_tracking_error_yaw_rad_s", "Yaw tracking error\n(rad/s)", 1.0),
        ("mean_joint_torque_rms_nm", "Torque RMS\n(N·m)", 1.0),
        (
            "mean_estimated_absolute_mechanical_power_w",
            "Mean mechanical power\n(W)",
            1.0,
        ),
        (
            "mean_estimated_torque_high_25_100_power_nm2",
            "Torque HF power\n(N²·m²)",
            1.0,
        ),
        (
            "mean_joint_speed_high_25_100_power_rad2_s2",
            "Joint-speed HF power\n(rad²/s²)",
            1.0,
        ),
    )
    panel_positions = (
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
    )
    for (metric, title, scale), position in zip(metrics, panel_positions, strict=True):
        ax = fig.add_subplot(grid[position])
        _bar_panel(ax, rows, metric, title, scale)

    fig.suptitle(f"{SCENARIO_LABELS[scenario]} task", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.94,
        "Within-task evaluation · 20 episodes per seed",
        ha="center",
        color="#666666",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.05,
        right=0.985,
        bottom=0.09,
        top=0.86,
        wspace=0.34,
        hspace=0.52,
    )
    save_figure(fig, output_dir, f"{scenario}_metrics", dpi)


def write_readme(output_dir: Path) -> None:
    readme = """# Within-task PPO evaluation figures

Four task dashboards compare Clean, Time-domain, and Frequency-domain PPO using
the same seven metrics. Every bar is the mean across 3 training seeds, error bars
show ±1 seed SD, and dots show the individual seeds. Each seed was evaluated for
20 episodes in the same task used for training.

- `nominal_metrics.png`
- `aggressive_metrics.png`
- `abrupt_metrics.png`
- `push_recovery_metrics.png`

Metrics: termination rate, XY and yaw tracking error, policy-rate torque RMS,
200 Hz mean estimated mechanical power, and 200 Hz torque/joint-speed high-band
(25–100 Hz) power.

Important interpretation limits:

- Safety quantities are simulated actuator-loading proxies. Hardware torque and
  speed limits were not calibrated.
- There are 3 independent training seeds, but only one evaluation seed. The 20
  episodes per evaluation are not independent algorithm-level samples.
- Failed episodes are shorter and contribute fewer samples to mean torque and
  power metrics. Inspect termination and duration alongside those metrics.
- Raw 200 Hz traces were not retained locally, so event-aligned time histories,
  full PSD curves, recovery time, and overshoot cannot be reconstructed.

Regenerate from the repository root with:

```bash
python scripts/plot_reward_ppo_results.py \
  --archive outputs/archives/reward-ppo-v1-20260922/data \
  --output figs
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    evaluation_root = args.archive / "outputs/reward_matrix/reward-ppo-v1"
    evaluations = load_evaluations(evaluation_root)

    for scenario in SCENARIOS:
        figure_task_dashboard(evaluations, scenario, args.output, args.dpi)
    write_readme(args.output)
    print(f"[PASS] Wrote 4 PNG figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
