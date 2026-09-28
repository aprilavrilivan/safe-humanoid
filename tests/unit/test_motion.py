"""Motion schema, joint mapping, and scenario checks without simulator deps."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))

from safe_humanoid.motion import load_motion, validate_motion_data  # noqa: E402
from safe_humanoid.tracking_scenarios import load_tracking_scenario  # noqa: E402

RUNTIME_FILE = (
    ROOT / "source" / "safe_humanoid" / "safe_humanoid" / "tasks"
    / "manager_based" / "tracking" / "scenario_runtime.py"
)
runtime_spec = importlib.util.spec_from_file_location(
    "tracking_scenario_runtime_test", RUNTIME_FILE
)
assert runtime_spec is not None and runtime_spec.loader is not None
runtime_module = importlib.util.module_from_spec(runtime_spec)
runtime_spec.loader.exec_module(runtime_module)
apply_tracking_scenario = runtime_module.apply_tracking_scenario


def sample_motion() -> dict:
    return {
        "schema_version": 1,
        "fps": 2,
        "joint_names": ["left_hip_pitch_joint", "right_knee_joint"],
        "joint_pos": [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4], [0.1, 0.2], [0.0, 0.0]],
        "joint_vel": [[0.0, 0.0], [0.2, 2.5], [0.2, 2.5], [-0.2, -2.5], [0.0, 0.0]],
        "root_pos_w": [[0.0, 0.0, z] for z in (0.75, 0.65, 0.50, 0.65, 0.75)],
        "root_quat_w": [[1.0, 0.0, 0.0, 0.0] for _ in range(5)],
    }


class TestMotionContract(unittest.TestCase):
    def setUp(self):
        self.payload = sample_motion()
        self.config_dir = ROOT / "configs" / "task" / "tracking"

    def test_validate_reorder_and_time_clamping(self):
        clip = validate_motion_data(self.payload, Path("example.json"))
        self.assertEqual(clip.duration_s, 2.0)
        self.assertEqual(clip.frame_at(1.25), 2)
        self.assertEqual(clip.frame_at(100.0), 4)
        reordered = clip.in_robot_order(["right_knee_joint", "left_hip_pitch_joint"])
        self.assertEqual(reordered.joint_pos[1], (0.2, 0.1))
        self.assertEqual(reordered.joint_vel[1], (2.5, 0.2))

    def test_reject_missing_joint(self):
        clip = validate_motion_data(self.payload, Path("example.json"))
        with self.assertRaisesRegex(ValueError, "missing=.*left_ankle"):
            clip.in_robot_order(["left_hip_pitch_joint", "left_ankle_pitch_joint"])

    def test_reject_duplicate_joint_bad_shape_and_nonfinite(self):
        bad = copy.deepcopy(self.payload)
        bad["joint_names"][1] = bad["joint_names"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_motion_data(bad, Path("bad.json"))
        bad = copy.deepcopy(self.payload)
        bad["joint_vel"][2].pop()
        with self.assertRaisesRegex(ValueError, "joint_vel"):
            validate_motion_data(bad, Path("bad.json"))
        bad = copy.deepcopy(self.payload)
        bad["joint_pos"][2][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_motion_data(bad, Path("bad.json"))

    def test_reject_nonunit_quaternion(self):
        bad = copy.deepcopy(self.payload)
        bad["root_quat_w"][2] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "unit wxyz quaternion"):
            validate_motion_data(bad, Path("bad.json"))

    def test_scenarios_accept_motion_and_reject_static_clip(self):
        clip = validate_motion_data(self.payload, Path("example.json"))
        for name in ("squat_stand", "fast_leg_swing"):
            scenario = load_tracking_scenario(name, self.config_dir)
            scenario.check_motion(clip)
        flat = copy.deepcopy(self.payload)
        flat["root_pos_w"] = [[0.0, 0.0, 0.75] for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "height excursion"):
            load_tracking_scenario("squat_stand", self.config_dir).check_motion(
                validate_motion_data(flat, Path("flat.json"))
            )
        slow = copy.deepcopy(self.payload)
        slow["joint_vel"] = [[0.0, 0.1] for _ in range(5)]
        with self.assertRaisesRegex(ValueError, "peak leg speed"):
            load_tracking_scenario("fast_leg_swing", self.config_dir).check_motion(
                validate_motion_data(slow, Path("slow.json"))
            )

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.json"
            path.write_text(json.dumps(self.payload), encoding="utf-8")
            self.assertEqual(load_motion(path).frame_count, 5)
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "scripts" / "check_motion.py"),
                    "--scenario", "squat_stand", "--motion", str(path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[PASS] squat_stand", result.stdout)

    def test_scenario_runtime_maps_all_parameters(self):
        clip = validate_motion_data(self.payload, Path("example.json"))
        scenario = load_tracking_scenario("squat_stand", self.config_dir)
        env_cfg = SimpleNamespace(
            commands=SimpleNamespace(motion=SimpleNamespace(motion_file="")),
            observations=SimpleNamespace(policy=SimpleNamespace(enable_corruption=True)),
            rewards=SimpleNamespace(**{
                name: SimpleNamespace(params={"std": -1})
                for name in ("joint_pos", "joint_vel", "root_height", "root_orientation")
            }),
            episode_length_s=0,
        )
        apply_tracking_scenario(env_cfg, scenario, clip)
        self.assertEqual(env_cfg.commands.motion.motion_file, "example.json")
        self.assertEqual(env_cfg.episode_length_s, 2.0)
        self.assertFalse(env_cfg.observations.policy.enable_corruption)
        self.assertEqual(env_cfg.rewards.joint_pos.params["std"], scenario.joint_pos_std_rad)
        self.assertEqual(env_cfg.rewards.joint_vel.params["std"], scenario.joint_vel_std_rad_s)
        self.assertEqual(env_cfg.rewards.root_height.params["std"], scenario.root_height_std_m)
        self.assertEqual(
            env_cfg.rewards.root_orientation.params["std"], scenario.root_orientation_std_rad
        )


if __name__ == "__main__":
    unittest.main()
