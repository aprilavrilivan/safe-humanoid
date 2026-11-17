#!/usr/bin/env python3
"""Inspect robosuite Wipe and capture raw MuJoCo signals at every physics tick."""

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))
from safe_humanoid.manipulation.trace import PhysicsTrace  # noqa: E402


class StepProxy:
    """Forward all MuJoCo calls, intercepting complete step or step2."""

    def __init__(self, sim, on_step):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_on_step", on_step)

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def __setattr__(self, name, value):
        setattr(self._sim, name, value)

    def step(self, *args, **kwargs):
        result = self._sim.step(*args, **kwargs)
        self._on_step()
        return result

    def step2(self, *args, **kwargs):
        result = self._sim.step2(*args, **kwargs)
        self._on_step()
        return result


def find_sensor(model, suffix):
    for i in range(model.nsensor):
        name = model.sensor_id2name(i)
        if name and name.endswith(suffix):
            return (name, int(model.sensor_adr[i]), int(model.sensor_dim[i]))
    return None


def sensor_value(data, spec):
    return None if spec is None else data.sensordata[spec[1]:spec[1]+spec[2]].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true", help="save a short offscreen frontview video")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    import mujoco
    import numpy as np
    import robosuite

    args.output.mkdir(parents=True, exist_ok=False)
    env = robosuite.make("Wipe", robots="Panda", gripper_types="WipingGripper",
                         has_renderer=not args.headless, has_offscreen_renderer=True,
                         use_camera_obs=False, control_freq=20, lite_physics=True)
    original_sim = None
    try:
        observation = env.reset()
        robot = env.robots[0]
        model, data = env.sim.model, env.sim.data
        arm = robot.arms[0]
        site_id = robot.eef_site_id[arm]
        qpos_idx = robot._ref_arm_joint_pos_indexes
        qvel_idx = robot._ref_arm_joint_vel_indexes
        ctrl_idx = robot._ref_arm_joint_actuator_indexes
        force_spec = find_sensor(model, "force_ee")
        torque_spec = find_sensor(model, "torque_ee")
        audit = {
            "task": "Wipe", "simulator": "robosuite/MuJoCo", "robot": "Panda", "gripper": "WipingGripper",
            "robosuite_version": getattr(robosuite, "__version__", "unknown"),
            "mujoco_version": getattr(mujoco, "__version__", "unknown"),
            "physics_dt_s": float(model.opt.timestep), "physics_hz": 1 / float(model.opt.timestep),
            "control_dt_s": float(env.control_timestep), "policy_hz": 1 / float(env.control_timestep),
            "decimation": int(env.control_timestep / model.opt.timestep),
            "joint_names": list(robot.robot_arm_joints), "action_dim": int(env.action_dim),
            "action_semantics": str(robot.composite_controller_config),
            "observation_shapes": {key: list(np.shape(value)) for key, value in observation.items()},
            "controller_command": "MuJoCo data.ctrl for arm actuators; not position target or measured effort",
            "actuator_torque": "MuJoCo qfrc_actuator projected on arm DoFs, physics rate",
            "measured_effort": "not a real motor measurement",
            "ee_force_sensor": None if force_spec is None else force_spec[0],
            "ee_torque_sensor": None if torque_spec is None else torque_spec[0],
            "contact_force": "wrist force/torque is not isolated tool-surface contact; contact_force remains null",
            "status": "reset_succeeded",
        }
        frames = []
        if args.video:
            frames.append(env.sim.render(width=640, height=480, camera_name="frontview"))
        original_sim = env.sim
        active = {"value": False, "policy_step": 0}
        events = []
        with PhysicsTrace(args.output / "trace", task="Wipe", simulator="mujoco",
                          dt_s=float(model.opt.timestep), joint_names=list(robot.robot_arm_joints),
                          signals={"controller_torque": "MuJoCo actuator input (data.ctrl); units depend on actuator gear",
                                   "actuator_torque": "MuJoCo data.qfrc_actuator; actual simulated generalized actuator force",
                                   "contact_wrench": "wrist force/torque sensors, not isolated contact",
                                   "contact_force": "unavailable without geometry-specific contact aggregation"}) as trace:
            def capture():
                if not active["value"]:
                    return
                speed = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, speed, 0)
                force = sensor_value(data, force_spec)
                torque = sensor_value(data, torque_spec)
                trace.record(sim_time_s=float(data.time), policy_step=active["policy_step"],
                             joint_position=data.qpos[qpos_idx].tolist(),
                             joint_velocity=data.qvel[qvel_idx].tolist(),
                             controller_torque=data.ctrl[ctrl_idx].tolist(),
                             actuator_torque=data.qfrc_actuator[qvel_idx].tolist(),
                             ee_position=data.site_xpos[site_id].tolist(),
                             ee_orientation=data.site_xmat[site_id].reshape(-1).tolist(),
                             ee_linear_velocity=speed[3:6], ee_angular_velocity=speed[:3],
                             contact_wrench=(force + torque) if force is not None and torque is not None else None)

            env.sim = StepProxy(original_sim, capture)
            for index in range(args.steps):
                active["policy_step"] = index + 1
                active["value"] = True
                observation, reward, done, info = env.step(np.zeros(env.action_dim))
                active["value"] = False
                events.append({"policy_step": index + 1, "reward": float(reward), "done": bool(done),
                               "success": len(env.wiped_markers) == env.num_markers,
                               "wiped_markers": len(env.wiped_markers)})
                if args.video:
                    frames.append(env.sim.render(width=640, height=480, camera_name="frontview"))
                if done:
                    break
            audit["physics_samples"] = trace.step
            audit["expected_physics_samples"] = len(events) * audit["decimation"]
            if trace.step != audit["expected_physics_samples"]:
                raise AssertionError("missing or extra MuJoCo physics substeps")
            audit["status"] = "reset_and_short_step_succeeded"
        env.sim = original_sim
        original_sim = None
        (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        if args.video:
            import imageio.v2 as imageio
            with imageio.get_writer(args.output / "scene.mp4", fps=20) as writer:
                for frame in frames:
                    writer.append_data(np.flipud(frame))
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit, indent=2))
    finally:
        if original_sim is not None:
            env.sim = original_sim
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
