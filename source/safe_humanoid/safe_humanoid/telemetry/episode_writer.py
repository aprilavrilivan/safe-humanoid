"""Small-cohort evaluation artifacts, using only the Python standard library."""

from __future__ import annotations

import csv
import json
from pathlib import Path

METRIC_FIELDS = (
    "tracking_error_xy_m_s",
    "tracking_error_yaw_rad_s",
    "upright_alignment",
    "joint_torque_rms_nm",
    "joint_torque_peak_nm",
    "joint_speed_peak_rad_s",
    "absolute_mechanical_power_w",
)
TRACE_FIELDS = (
    "env_id", "episode_id", "time_s", "command_vx_m_s", "command_vy_m_s",
    "command_wz_rad_s", *METRIC_FIELDS, "transition_reward", "terminated", "truncated",
    "push_fired",
)


class EvaluationWriter:
    """Write policy-rate pre-step state plus resulting transition information."""

    def __init__(self, output_dir: Path, metadata: dict):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._handle = (self.output_dir / "trace.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=TRACE_FIELDS)
        self._writer.writeheader()
        self._sums = {field: 0.0 for field in METRIC_FIELDS}
        self._samples = 0
        self._terminations = 0
        self._timeouts = 0
        self._pushes = 0

    def record(self, row: dict) -> None:
        if set(row) != set(TRACE_FIELDS):
            raise ValueError("trace row has missing or unexpected fields")
        self._writer.writerow(row)
        self._samples += 1
        for field in METRIC_FIELDS:
            self._sums[field] += float(row[field])
        self._terminations += bool(row["terminated"])
        self._timeouts += bool(row["truncated"])
        self._pushes += bool(row["push_fired"])

    def close(self) -> dict:
        self._handle.close()
        summary = {
            "samples": self._samples,
            "terminations": self._terminations,
            "timeouts": self._timeouts,
            "pushes": self._pushes,
            **{
                f"mean_{field}": total / self._samples if self._samples else None
                for field, total in self._sums.items()
            },
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
