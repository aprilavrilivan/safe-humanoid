"""GPU-free checks for pilot planning and evaluation-artifact auditing."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "velocity_experiment.py"
SPEC = importlib.util.spec_from_file_location("velocity_experiment_for_tests", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
MANIFEST = ROOT / "configs" / "experiments" / "velocity_baseline_pilot.yaml"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestVelocityExperiment(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = pilot.load_manifest(MANIFEST)

    def _make_complete_run(self) -> None:
        evaluate = self.manifest["evaluation"]
        specs = pilot.scenario_specs()
        digest = "a" * 64
        for name, scenario in specs.items():
            directory = self.root / name
            directory.mkdir()
            _write_json(directory / "metadata.json", {
                "task": pilot.TASK_ID,
                "scenario": name,
                "scenario_sha256": pilot._sha256(scenario.source),
                "checkpoint_sha256": digest,
                "seed": evaluate["seed"],
                "num_envs": evaluate["num_envs"],
                "episodes_per_env": evaluate["episodes_per_env"],
                "physics_trace_enabled": True,
                "isaac_lab_ref": "v2.3.2",
            })
            rows = []
            physics_rows = []
            for env in range(evaluate["num_envs"]):
                for episode in range(evaluate["episodes_per_env"]):
                    final = scenario.evaluation[-1]
                    observations = [
                        (point.time_s + (0.0 if index == 0 else 0.02), point.command, False)
                        for index, point in enumerate(scenario.evaluation)
                    ]
                    observations.append((
                        (scenario.push.time_s if scenario.push else final.time_s) + 0.04,
                        final.command, True,
                    ))
                    for time_s, command, done in observations:
                        row = {field: 0.0 for field in pilot.TRACE_FIELDS}
                        row.update({
                            "env_id": env, "episode_id": episode, "time_s": time_s,
                            "command_vx_m_s": command[0], "command_vy_m_s": command[1],
                            "command_wz_rad_s": command[2],
                            "terminated": False, "truncated": done,
                            "push_fired": bool(done and scenario.push is not None),
                        })
                        rows.append(row)
                    physics_rows.append({
                        "env_id": env, "episode_id": episode, "physics_step": 1,
                        "time_s": 0.005, "estimated_actuator_torque_nm": "[0]",
                        "joint_velocity_rad_s": "[0]",
                    })
            _write_csv(directory / "trace.csv", pilot.TRACE_FIELDS, rows)
            _write_json(directory / "summary.json", {
                "samples": len(rows), "terminations": 0,
                "timeouts": evaluate["num_envs"] * evaluate["episodes_per_env"],
                "pushes": len(physics_rows) if scenario.push else 0,
            })
            _write_json(directory / "physics_metadata.json", {
                "sample_rate_hz": 200.0, "torque_is_measured": False,
            })
            _write_csv(directory / "physics_trace.csv", pilot.PHYSICS_FIELDS, physics_rows)
            _write_json(directory / "safety_summary.json", {
                "sample_rate_hz": 200.0,
                "physical_torque_violation_claim_available": False,
                "config_sha256": {
                    filename: pilot._sha256(ROOT / "configs" / "safety" / filename)
                    for filename in ("time_domain.yaml", "frequency_domain.yaml", "g1_limits.yaml")
                },
                "temporal_aggregate": {
                    "physics_samples": len(physics_rows),
                    "episode_count": len(physics_rows),
                },
            })
        _write_csv(
            self.root / "comparison.csv", ("scenario",),
            [{"scenario": name} for name in pilot.SCENARIOS],
        )

    def test_plan_works_without_simulator(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "plan"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--scenario nominal --num-envs 64", result.stdout)
        self.assertIn("abrupt:", result.stdout)
        self.assertIn("final event at 14 s", result.stdout)

    def test_complete_run_is_comparable_only(self):
        self._make_complete_run()
        report = pilot.audit(self.manifest, self.root)
        self.assertEqual(report["status"], "ready_for_comparison")
        self.assertIn("not policy performance", report["meaning"])
        self.assertTrue(all(
            details["exposed_episodes"] == 6 for details in report["scenarios"].values()
        ))

    def test_different_checkpoint_fails(self):
        self._make_complete_run()
        path = self.root / "abrupt" / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["checkpoint_sha256"] = "b" * 64
        _write_json(path, metadata)
        report = pilot.audit(self.manifest, self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("one verified checkpoint", " ".join(report["issues"]))

    def test_no_scheduled_or_push_exposure_fails(self):
        self._make_complete_run()
        for name in ("nominal", "push_recovery"):
            path = self.root / name / "trace.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if name == "nominal":
                    if float(row["time_s"]) >= 14.0:
                        row["command_wz_rad_s"] = "0.0"
                else:
                    row["push_fired"] = "False"
            _write_csv(path, pilot.TRACE_FIELDS, rows)
            if name == "push_recovery":
                summary = self.root / name / "summary.json"
                data = json.loads(summary.read_text(encoding="utf-8"))
                data["pushes"] = 0
                _write_json(summary, data)
        report = pilot.audit(self.manifest, self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["scenarios"]["nominal"]["exposed_episodes"], 0)
        self.assertEqual(report["scenarios"]["push_recovery"]["exposed_episodes"], 0)

    def test_missing_physics_and_mismatched_scenario_hash_fail(self):
        self._make_complete_run()
        (self.root / "aggressive" / "safety_summary.json").unlink()
        metadata_path = self.root / "abrupt" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["scenario_sha256"] = "b" * 64
        _write_json(metadata_path, metadata)
        report = pilot.audit(self.manifest, self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("safety_summary.json", report["scenarios"]["aggressive"]["error"])
        self.assertIn("scenario_sha256", report["scenarios"]["abrupt"]["error"])

    def test_invalid_manifest_is_rejected(self):
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["evaluation"]["num_envs"] = 9
        path = self.root / "invalid.yaml"
        _write_json(path, data)
        with self.assertRaisesRegex(ValueError, "<= 8"):
            pilot.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
