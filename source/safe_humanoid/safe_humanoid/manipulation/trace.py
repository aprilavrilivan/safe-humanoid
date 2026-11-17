"""One JSON record per physics substep, with explicit signal provenance.

This intentionally does not import Isaac Sim, MuJoCo, Torch or RSL-RL. A missing
field is null, never a fabricated zero. Video frames remain at policy rate.
"""

import json
import math
from pathlib import Path


class PhysicsTrace:
    def __init__(self, directory: Path, *, task: str, simulator: str, dt_s: float, joint_names: list[str], signals: dict):
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("physics dt must be finite and positive")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("joint names must be nonempty and unique")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.dt_s = dt_s
        self.step = 0
        self.segment = 0
        self.last_time = None
        self.handle = (self.directory / "physics.jsonl").open("x", encoding="utf-8")
        metadata = {
            "schema_version": 1,
            "task": task,
            "simulator": simulator,
            "physics_dt_s": dt_s,
            "physics_frequency_hz": 1 / dt_s,
            "joint_names": joint_names,
            "signals": signals,
            "note": "Every row is a physics step, not a policy transition; null means unavailable.",
        }
        (self.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def record(self, *, sim_time_s: float, joint_position, joint_velocity, controller_torque=None,
               actuator_torque=None, measured_effort=None, ee_position=None, ee_orientation=None,
               ee_linear_velocity=None, ee_angular_velocity=None, ee_acceleration=None,
               contact_force=None, contact_wrench=None, policy_step=None, terminated=None,
               truncated=None, success=None):
        if not math.isfinite(sim_time_s):
            raise ValueError("simulation time must be finite")
        if self.last_time is not None and not math.isclose(sim_time_s - self.last_time, self.dt_s, rel_tol=0.02, abs_tol=1e-8):
            raise ValueError("physics trace skipped a substep or crossed an unmarked reset")
        self.last_time = sim_time_s
        self.step += 1
        vector_names = ("joint_position", "joint_velocity", "controller_torque", "actuator_torque", "measured_effort",
                        "ee_position", "ee_orientation", "ee_linear_velocity", "ee_angular_velocity", "ee_acceleration",
                        "contact_force", "contact_wrench")
        values = locals()
        row = {"physics_step": self.step, "segment": self.segment, "sim_time_s": sim_time_s, "policy_step": policy_step,
               "terminated": terminated, "truncated": truncated, "success": success}
        for name in vector_names:
            value = values[name]
            if value is None:
                row[name] = None
            else:
                sequence = [float(x) for x in value]
                if not all(math.isfinite(x) for x in sequence):
                    raise ValueError(f"{name} contains NaN or Inf")
                row[name] = sequence
        if len(row["joint_position"]) != len(row["joint_velocity"]):
            raise ValueError("joint position and velocity lengths differ")
        for name in ("controller_torque", "actuator_torque", "measured_effort"):
            if row[name] is not None and len(row[name]) != len(row["joint_position"]):
                raise ValueError(f"{name} joint count differs")
        self.handle.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

    def mark_reset_gap(self):
        """Close a policy segment when upstream reset helpers advance simulation."""
        if self.step:
            self.segment += 1
            self.last_time = None

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
