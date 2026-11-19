"""Portable checks for the new simulator-independent inspection contracts."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from safe_humanoid.manipulation.automate_assets import inspect_assembly, select_diverse
from safe_humanoid.manipulation.catalog import TASKS
from safe_humanoid.manipulation.trace import PhysicsTrace
from wipe_smoke import StepProxy
from isaac_smoke import attach_physics_trace, configure_automate_data_cache, configure_factory_asset_cache


class ManipulationContractTests(unittest.TestCase):
    def test_catalog_contains_exact_requested_upstream_tasks(self):
        self.assertEqual(len(TASKS), 5)
        self.assertEqual(TASKS["peg_insertion"].environment_id, "Isaac-Factory-PegInsert-Direct-v0")
        self.assertEqual(TASKS["automate_assembly"].environment_id, "Isaac-AutoMate-Assembly-Direct-v0")
        self.assertEqual(TASKS["wiping"].environment_id, "Wipe")

    def test_trace_preserves_physics_rate_and_missing_signal_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with PhysicsTrace(Path(tmp) / "run", task="Wipe", simulator="mujoco", dt_s=0.002,
                              joint_names=["j1", "j2"], signals={"actuator_torque": "simulated"}) as trace:
                for i in range(4):
                    trace.record(sim_time_s=0.002 * (i + 1), joint_position=(0, 0),
                                 joint_velocity=(0, 0), actuator_torque=(0.2, 0.3))
                with self.assertRaisesRegex(ValueError, "skipped a substep"):
                    trace.record(sim_time_s=0.02, joint_position=(0, 0), joint_velocity=(0, 0))
            lines = (Path(tmp) / "run" / "physics.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIsNone(json.loads(lines[0])["measured_effort"])
            self.assertEqual(json.loads(lines[-1])["actuator_torque"], [0.2, 0.3])

    def test_step_proxy_records_both_mujoco_step_paths(self):
        class FakeSim:
            def __init__(self):
                self.ticks = 0
            def step(self):
                self.ticks += 1
            def step2(self):
                self.ticks += 1
        sim = FakeSim()
        seen = []
        proxy = StepProxy(sim, lambda: seen.append(sim.ticks))
        proxy.step()
        proxy.step2()
        self.assertEqual(seen, [1, 2])

    def test_isaac_hook_records_policy_ticks_but_excludes_reset_ticks(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = value
            def __getitem__(self, index):
                value = self.value
                for part in index if isinstance(index, tuple) else (index,):
                    value = value[part]
                return FakeTensor(value)
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return self.value
        class FakeView:
            def get_dof_positions(self):
                return FakeTensor([[0.1, 0.2]])
            def get_dof_velocities(self):
                return FakeTensor([[0.0, 0.0]])
            def get_link_transforms(self):
                return FakeTensor([[[1, 2, 3, 0, 0, 0, 1]]])
            def get_link_velocities(self):
                return FakeTensor([[[0, 0, 0, 0, 0, 0]]])
        class FakeSim:
            current_time = 0.0
            def step(self):
                self.current_time += 0.01
        class FakeRaw:
            sim = FakeSim()
            fingertip_body_idx = 0
            joint_torque = FakeTensor([[0.3, 0.4]])
            _robot = type("Robot", (), {"root_physx_view": FakeView(),
                                         "data": type("Data", (), {"applied_torque": FakeTensor([[0.3, 0.4]])})()})()
            def _apply_action(self):
                pass
        raw = FakeRaw()
        with tempfile.TemporaryDirectory() as tmp:
            with PhysicsTrace(Path(tmp) / "run", task="peg", simulator="isaac", dt_s=0.01,
                              joint_names=["j1", "j2"], signals={}) as trace:
                state, restore = attach_physics_trace(raw, trace)
                state["policy_step"] = 1
                raw._apply_action()
                raw.sim.step()
                raw._apply_action()
                raw.sim.step()
                raw.sim.step()  # upstream reset helper, excluded and marked as gap
                state["policy_step"] = 2
                raw._apply_action()
                raw.sim.step()
                restore()
                self.assertEqual(trace.step, 3)
            rows = [json.loads(x) for x in (Path(tmp) / "run/physics.jsonl").read_text().splitlines()]
            self.assertEqual([row["segment"] for row in rows], [0, 0, 1])
            self.assertEqual(rows[0]["controller_torque"], [0.3, 0.4])

    def test_automate_selection_uses_actual_mesh_descriptors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, width in enumerate((0.01, 0.02, 0.09, 0.11)):
                folder = root / f"{i:05d}"
                folder.mkdir()
                for part in ("plug", "socket"):
                    (folder / f"{part}.obj").write_text(f"v 0 0 0\nv {width} 0 0\nf 1 2 1\n")
            item = inspect_assembly(root / "00002")
            self.assertEqual(item["parts"]["plug"]["bbox_extent"][0], 0.09)
            chosen = select_diverse(root)
            self.assertEqual(len({item["assembly_id"] for item in chosen}), 3)
            self.assertTrue(all(item["parts"]["socket"]["faces"] == 1 for item in chosen))

    def test_local_asset_caches_update_upstream_config_shapes(self):
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "00015").mkdir()
                (root / "plug_grasps.json").write_text("{}")
                (root / "disassembly_dist.json").write_text("{}")
                (root / "00015/disassemble_traj.json").write_text("[]")
                auto_task = SimpleNamespace()
                auto_cfg = SimpleNamespace(task_name="insertion", tasks={"insertion": auto_task})
                configure_automate_data_cache(auto_cfg, "00015", root)
                self.assertEqual(Path.cwd(), (root / "runtime/00015").resolve())
                self.assertTrue(Path(auto_task.disassembly_path_json).is_file())

                factory_root = root / "factory"
                factory_root.mkdir()
                for name in ("factory_bolt_m16.usd", "factory_nut_m16.usd"):
                    (factory_root / name).write_bytes(b"usd")
                factory_task = SimpleNamespace(
                    fixed_asset=SimpleNamespace(spawn=SimpleNamespace(usd_path="remote-bolt")),
                    held_asset=SimpleNamespace(spawn=SimpleNamespace(usd_path="remote-nut")),
                )
                factory_cfg = SimpleNamespace(task=factory_task)
                configure_factory_asset_cache(factory_cfg, "nut_threading", factory_root)
                self.assertEqual(Path(factory_task.fixed_asset.spawn.usd_path).name, "factory_bolt_m16.usd")
                self.assertEqual(Path(factory_task.held_asset.spawn.usd_path).name, "factory_nut_m16.usd")
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
