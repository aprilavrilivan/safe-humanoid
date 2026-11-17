# Contact-rich manipulation environment inspection

This phase reuses five upstream simulation tasks and adds bounded inspection scripts. It does **not** register new variants, change existing locomotion experiments, or train a policy. The local host is macOS ARM with Python 3.14.5; it has no `isaaclab`, `isaacsim`, `robosuite`, `mujoco`, `torch`, `rsl_rl`, or `rl_games` import. The Isaac Lab stack manifest pins Isaac Lab v2.3.2 (commit `37ddf626871758333d6ed89cf64ad702aef127d0`), Isaac Sim 5.1.0, Python 3.11, and RSL-RL 3.1.2. No dependency was installed or upgraded in this phase.

## Confirmation table

Here “ready” means **verified from a running physics-rate trace**, not just an available launch command. No simulator was run on this Mac; consequently none of the requested runtime checks can be marked complete.

| Environment | Scene loaded | Physics torque available | Contact force available | Ready for HF |
|---|---|---|---|---|
| Peg Insertion | 待 Linux/GPU 验证 | 控制器力矩指令已定位；实际驱动力待测 | 原任务无已验证接触力信号 | 否，待实测 |
| Gear Meshing | 待 Linux/GPU 验证 | 控制器力矩指令已定位；实际驱动力待测 | 原任务无已验证接触力信号 | 否，待实测 |
| Nut Threading | 待 Linux/GPU 验证 | 控制器力矩指令已定位；实际驱动力待测 | 原任务无已验证接触力信号 | 否，待实测 |
| AutoMate Assembly | 待 Linux/GPU 验证 | 控制器力矩指令已定位；实际驱动力待测 | 原任务无已验证接触力信号 | 否，待实测 |
| Wiping | 待 robosuite/MuJoCo 验证 | `qfrc_actuator` 采样代码已备；待实测 | 腕部力传感器待实测；独立工具接触力未分离 | 否，待实测 |

No screenshots or videos exist yet. The `--video` option records them during simulator smoke tests; a scene load alone is not evidence of task success.

## Reproducible commands

Run from repository root in an appropriate **existing** Python environment:

```sh
python scripts/manipulation/preflight.py
python scripts/manipulation/isaac_smoke.py --task peg_insertion --headless --output outputs/manipulation/peg_001
python scripts/manipulation/isaac_smoke.py --task gear_meshing --headless --output outputs/manipulation/gear_001
python scripts/manipulation/isaac_smoke.py --task nut_threading --headless --output outputs/manipulation/nut_001
python scripts/manipulation/isaac_smoke.py --task automate_assembly --assembly-id 00015 --headless --output outputs/manipulation/automate_00015_001
python scripts/manipulation/wipe_smoke.py --headless --output outputs/manipulation/wipe_001
```

For an Isaac Lab window, omit `--headless`; for a short video add `--video`, including with `--headless` when the offscreen renderer is available. For Wipe, add `--video` for an offscreen frontview video. The five `scripts/manipulation/run_*.sh` wrappers provide individual task entry points. Do not run all five simultaneously in a first inspection. Output directories must be new. `audit.json`, `events.json`, and `trace/physics.jsonl` are written by the scripts.

The Isaac scripts require the pinned Isaac Lab/Isaac Sim runtime, task package, CUDA-capable host, and externally resolved assets. The AutoMate task may download its USD/OBJ and auxiliary JSON assets from the Isaac Lab Nucleus; access and geometry must be verified on the server. `Wipe` requires robosuite, MuJoCo and video encoding support for `--video`; these are optional dependencies and are **not** installed into the humanoid stack. Use a separate environment for them after checking compatibility. No existing CUDA or Isaac Lab version should be changed for this task.

## AutoMate asset selection

The upstream `assembly_tasks_cfg.Insertion` defaults to `assembly_id="00015"`, with 8 mm plug/socket dimensions and asset paths containing that ID. Its `run_w_id.py` edits the upstream config file before a new process. Our script does not alter that upstream file. To select three *geometrically different* IDs after the assets are present locally:

```sh
python scripts/manipulation/select_automate_assets.py /path/to/AutoMate --output outputs/manipulation/asset_candidates.json
```

The selector reads `plug.obj` and `socket.obj` in each five-digit folder, records mesh face counts and bounding-box extents, and chooses three separated profiles. This is an inspection shortlist, **not** a difficulty ranking. Before using a non-default ID, inspect its USD/OBJ geometry, grasp and disassembly metadata, and supply verified dimensions/masses in `--assembly-metadata` JSON; then view the scene. Existing task-specific reward and reset settings may also depend on geometry. Until that verification, only the upstream default `00015` is a supported smoke candidate.

## Physics and control audit

The launchers write actual `sim.dt`, physics Hz, decimation, policy Hz, joint names/count, action/observation spaces, collision prim counts, and controller configuration to `audit.json`. For Factory and AutoMate, the policy action is a six-dimensional end-effector position/rotation increment. `compute_dof_torque` creates arm effort targets, while gripper fingers use position targets. The `joint_torque` variable is a controller-computed **command** supplied to `set_joint_effort_target`; it is not a measured motor torque. The associated `applied_wrench` is a controller output, not contact force. The script intercepts the DirectRLEnv physics loop at `_apply_action` and `sim.step`, recording one sample per physics tick from the PhysX view. It also attempts to capture Isaac Lab's `applied_torque` buffer separately; mapping and timing must be verified on the server. Measured motor effort remains null. To obtain independent tool contact force, the least invasive next change is to enable/contact-filter an Isaac Lab contact sensor for the fingertip and fixed asset, then verify it against controlled collision tests. Do not substitute joint reaction wrench for contact force.

For Wipe, the zero action is a neutral **OSC input**, not zero simulated torque. MuJoCo `data.ctrl` captures actuator input, `qfrc_actuator` captures simulated generalized actuator force, and the optional wrist force/torque sensor is recorded separately. The `step2` hook preserves the `lite_physics` substep path used by robosuite 1.5.0. A wrist wrench can include non-contact effects; tool-surface force needs geometry-specific contact aggregation or calibration. The trace's orientation is a row-major site rotation matrix for MuJoCo, while Isaac records the raw PhysX quaternion for later convention verification. End-effector acceleration is left null until a validated physics-rate derivative or native acceleration source is selected.

Each trace row has the simulator's **original physics timestamp**. The HF filter must later be designed from `physics_dt_s` for each environment; the G1 200 Hz/25 Hz coefficients are not portable. Task success/termination is stored separately at policy rate in `events.json` because it is only known after the physics substeps. No new PPO-Lagrangian integration exists in this phase. The current project uses RSL-RL PPO-Lag; all four Isaac contact-rich tasks use upstream Direct environments and official `rl_games` PPO entries, so a separate adapter is required before training.

## Next server validation order

1. Run `preflight.py`; confirm the four Gym IDs after AppLauncher starts, the installed commit, and optional robosuite version.
2. Smoke Peg Insertion, Gear Meshing, Nut Threading, then AutoMate default 00015; inspect one video and the exact physics sample count for each.
3. Download/inspect AutoMate assets and shortlist three IDs. Confirm their geometry and metadata before attempting the non-default ID path.
4. Run Wipe in its separate environment. Verify the wrist sensor names, MuJoCo torque trace, and scene video.
5. Apply known small contact perturbations and verify torque/contact channels and timestamps before assigning “ready for HF.”

Upstream references: [Isaac Lab v2.3.2 tasks](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/environments.html), [Factory controller](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py), [AutoMate task configuration](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_tasks_cfg.py), [robosuite Wipe](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.0/robosuite/environments/manipulation/wipe.py), [robosuite stepping](https://github.com/ARISE-Initiative/robosuite/blob/v1.5.0/robosuite/environments/base.py).
