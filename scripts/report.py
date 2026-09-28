#!/usr/bin/env python3
"""Combine evaluation summaries into a CSV on any Python 3.11+ machine."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dirs", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

rows = []
for run_dir in args.run_dirs:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    row = {"run_dir": str(run_dir), "scenario": metadata["scenario"],
           "checkpoint": metadata["checkpoint"], "seed": metadata["seed"], **summary}
    safety_path = run_dir / "safety_summary.json"
    if safety_path.is_file():
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        row.update({
            "physics_sample_rate_hz": safety["sample_rate_hz"],
            "hardware_limit_status": safety["hardware_limit_status"],
            "physical_torque_violation_claim_available": safety[
                "physical_torque_violation_claim_available"
            ],
            "physical_speed_threshold_comparison_available": safety[
                "physical_speed_threshold_comparison_available"
            ],
            **{f"physics_{key}": value for key, value in safety["temporal_aggregate"].items()},
            **{f"frequency_{key}": value for key, value in safety["frequency_aggregate"].items()},
        })
    rows.append(row)

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="", encoding="utf-8") as handle:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[PASS] Wrote {len(rows)} evaluation summaries to {args.output}")
