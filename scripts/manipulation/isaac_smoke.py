#!/usr/bin/env python3
"""Inspect unmodified Isaac Lab Factory/AutoMate tasks, without training.

The per-step hook is attached only to this smoke-test instance. It captures
controller effort after _apply_action and PhysX state after sim.step, once per
policy-loop physics tick. It never labels a command as measured motor torque.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))
from safe_humanoid.manipulation.catalog import TASKS  # noqa: E402
from safe_humanoid.manipulation.trace import PhysicsTrace  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(name for name, task in TASKS.items() if task.simulator == "isaaclab"), required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assembly-id", default="00015", help="AutoMate default is upstream's 00015")
    parser.add_argument("--assembly-metadata", type=Path, help="JSON with verified geometry for a non-default AutoMate ID")
    parser.add_argument("--video", action="store_true", help="record an RGB video at policy rate")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps <= 0 or args.num_envs <= 0 or args.num_envs > 4:
        parser.error("steps must be positive and 1 <= num-envs <= 4")
    if args.task != "automate_assembly" and (args.assembly_id != "00015" or args.assembly_metadata):
        parser.error("AutoMate flags require --task automate_assembly")
    if args.task == "automate_assembly" and args.assembly_id != "00015" and not args.assembly_metadata:
        parser.error("non-default assembly IDs require inspected --assembly-metadata")
    return args


def configure_automate(cfg, assembly_id, metadata_path):
    if not (len(assembly_id) == 5 and assembly_id.isdigit()):
        raise ValueError("assembly ID must be five digits")
    if assembly_id == "00015":
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("assembly_id") != assembly_id:
        raise ValueError("metadata assembly_id mismatch")
    required = ("plug_diameter_m", "plug_height_m", "plug_mass_kg", "socket_diameter_m",
                "socket_height_m", "socket_base_height_m", "socket_mass_kg")
    if any(not isinstance(metadata.get(key), (int, float)) or metadata[key] <= 0 for key in required):
        raise ValueError(f"metadata must contain positive, verified values: {required}")
    from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR

    task = cfg.task
    task.assembly_id = assembly_id
    task.assembly_dir = f"{ASSET_DIR}/{assembly_id}/"
    task.disassembly_path_json = f"{task.assembly_dir}disassemble_traj.json"
    for name, prefix in (("held_asset_cfg", "plug"), ("fixed_asset_cfg", "socket")):
        part = getattr(task, name)
        part.diameter = metadata[f"{prefix}_diameter_m"]
        part.height = metadata[f"{prefix}_height_m"]
        part.mass = metadata[f"{prefix}_mass_kg"]
    task.fixed_asset_cfg.base_height = metadata["socket_base_height_m"]
    task.fixed_asset.spawn.usd_path = f"{task.assembly_dir}socket.usd"
    task.held_asset.spawn.usd_path = f"{task.assembly_dir}plug.usd"
    task.fixed_asset.spawn.mass_props.mass = task.fixed_asset_cfg.mass
    task.held_asset.spawn.mass_props.mass = task.held_asset_cfg.mass
    # Other task-specific geometry/reward parameters may also require adaptation.
    # This is an inspection run only; no training is permitted with this override.


def tensor_row(value, index=0):
    return value[index].detach().cpu().tolist()


def check_observations(observations):
    import torch

    if not isinstance(observations, dict) or not observations:
        raise AssertionError("expected nonempty DirectRLEnv observation dictionary")
    for name, tensor in observations.items():
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all().item():
            raise AssertionError(f"observation {name} is missing or nonfinite")


def collision_counts():
    """Count collision-enabled USD prims under the first simulated scene."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    result = {}
    for name in ("Robot", "FixedAsset", "HeldAsset"):
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        if not root.IsValid():
            result[name] = None
            continue
        result[name] = sum(1 for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI))
    return result


def attach_physics_trace(raw, trace):
    """Pair each DirectRLEnv._apply_action with exactly one sim.step call."""
    original_apply = raw._apply_action
    original_step = raw.sim.step
    state = {"pending": False, "policy_step": 0, "controller": None}
    robot = raw._robot
    view = robot.root_physx_view
    ee_index = raw.fingertip_body_idx

    def apply():
        original_apply()
        if state["pending"]:
            raise RuntimeError("controller action was not followed by a physics tick")
        state["pending"] = True
        torque = getattr(raw, "joint_torque", None)
        state["controller"] = None if torque is None else tensor_row(torque)

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if not state["pending"]:
            trace.mark_reset_gap()
            return result  # Factory/AutoMate reset helper steps are not policy samples.
        pose = view.get_link_transforms()[0, ee_index].detach().cpu().tolist()
        velocity = view.get_link_velocities()[0, ee_index].detach().cpu().tolist()
        trace.record(
            sim_time_s=float(raw.sim.current_time),
            policy_step=state["policy_step"],
            joint_position=tensor_row(view.get_dof_positions()),
            joint_velocity=tensor_row(view.get_dof_velocities()),
            controller_torque=state["controller"],
            actuator_torque=(None if getattr(robot.data, "applied_torque", None) is None
                             else tensor_row(robot.data.applied_torque)),
            ee_position=pose[:3], ee_orientation=pose[3:7],
            ee_linear_velocity=velocity[:3], ee_angular_velocity=velocity[3:6],
        )
        state["pending"] = False
        return result

    raw._apply_action = apply
    raw.sim.step = step

    def restore():
        raw._apply_action = original_apply
        raw.sim.step = original_step

    return state, restore


def main():
    args = parse_args()
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": args.headless, "enable_cameras": args.video, "device": args.device})
    app = launcher.app
    env = None
    restore = None
    try:
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        task = TASKS[args.task]
        spec = gym.spec(task.environment_id)
        cfg = parse_env_cfg(task.environment_id, device=args.device, num_envs=args.num_envs)
        if args.task == "automate_assembly":
            configure_automate(cfg, args.assembly_id, args.assembly_metadata)
        args.output.mkdir(parents=True, exist_ok=False)
        env = gym.make(task.environment_id, cfg=cfg, render_mode="rgb_array" if args.video else None)
        raw = env.unwrapped
        if args.video:
            env = gym.wrappers.RecordVideo(env, video_folder=str(args.output / "videos"),
                                          step_trigger=lambda step: step == 0, video_length=args.steps)
        obs, _ = env.reset(seed=0)
        check_observations(obs)
        scene = raw.scene
        for name in ("robot", "fixed_asset"):
            if name not in scene.articulations or not scene.articulations[name].is_initialized:
                raise AssertionError(f"scene articulation {name} is not initialized")
        if "held_asset" not in scene.articulations and "held_asset" not in scene.rigid_objects:
            raise AssertionError("held asset missing from scene")
        audit = {
            "task": task.environment_id, "assembly_id": args.assembly_id if args.task == "automate_assembly" else None,
            "gym_entry_point": str(spec.entry_point), "scene_articulations": list(scene.articulations),
            "scene_rigid_objects": list(scene.rigid_objects), "robot_joint_names": list(raw._robot.data.joint_names),
            "robot_body_names": list(raw._robot.body_names),
            "collision_prim_counts": collision_counts(),
            "trace_scope": "environment 0 only; events contain all requested environments",
            "physics_dt_s": float(cfg.sim.dt), "physics_hz": 1 / float(cfg.sim.dt),
            "decimation": int(cfg.decimation), "policy_hz": 1 / (float(cfg.sim.dt) * cfg.decimation),
            "action_space": repr(env.action_space), "observation_space": repr(env.observation_space),
            "reset_observation_shapes": {key: list(value.shape) for key, value in obs.items()},
            "controller_cfg": str(cfg.ctrl),
            "action_semantics": "6D end-effector delta pose; controller computes 7 arm torques; gripper position target",
            "torque_semantics": "controller_torque = upstream joint_torque effort target; actuator_torque = Isaac Lab applied_torque buffer if exposed (verify mapping/timing); measured motor torque unavailable",
            "contact_semantics": "not exposed by this original Factory/AutoMate task; controller applied_wrench is NOT contact force",
            "ee_quaternion_order": "raw PhysX tensor order; verify against API before interpreting",
            "status": "reset_succeeded",
        }
        trace_dir = args.output / "trace"
        with PhysicsTrace(trace_dir, task=task.environment_id, simulator="isaaclab", dt_s=float(cfg.sim.dt),
                          joint_names=list(raw._robot.data.joint_names), signals={
                              "controller_torque": "upstream joint_torque effort target, not measured",
                              "actuator_torque": "Isaac Lab applied_torque buffer if exposed; verify against controller command",
                              "measured_effort": "unavailable; null",
                              "contact_force": "unavailable without sensor; null",
                          }) as trace:
            state, restore = attach_physics_trace(raw, trace)
            actions = torch.zeros((args.num_envs, int(cfg.action_space)), device=raw.device)
            events = []
            with torch.inference_mode():
                for index in range(args.steps):
                    state["policy_step"] = index + 1
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    check_observations(obs)
                    if not torch.isfinite(reward).all().item():
                        raise AssertionError("nonfinite reward")
                    events.append({"policy_step": index + 1, "reward": tensor_row(reward),
                                   "terminated": tensor_row(terminated), "truncated": tensor_row(truncated),
                                   "success": tensor_row(raw.ep_succeeded) if hasattr(raw, "ep_succeeded") else None})
            audit["physics_samples"] = trace.step
            audit["expected_physics_samples"] = args.steps * cfg.decimation
            if trace.step != audit["expected_physics_samples"]:
                raise AssertionError("missing or extra physics substeps")
            (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
            audit["status"] = "reset_and_short_step_succeeded"
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit, indent=2))
    finally:
        if restore:
            restore()
        if env:
            env.close()
        app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
