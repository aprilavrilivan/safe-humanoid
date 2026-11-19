# Contact-rich manipulation environment audit

Remote validation completed on 2026-11-19 with Ubuntu 22.04.5, RTX 5090 32 GB, driver 595.71.05, Python 3.11, Isaac Sim 5.1.0 and Isaac Lab v2.3.2 (`37ddf626871758333d6ed89cf64ad702aef127d0`). Wipe was isolated in a second environment with robosuite 1.5.0 and MuJoCo 3.3.7. No policy was trained and none of the short neutral/probe runs should be interpreted as solving a task.

## Confirmation table

“Ready for HF” means that a continuous physics-rate actuator-effort trace can be collected for filter and cost development. It does not mean that PPO/PPO-Lagrangian has been connected to the task. The Isaac tasks expose controller-computed effort and Isaac Lab's physics-rate `applied_torque` buffer; neither is a real motor measurement.

| Environment | Scene loaded | Physics torque available | Contact force available | Ready for HF |
|---|---|---|---|---|
| Peg Insertion | ✅ reset + 4 steps | ✅ command + `applied_torque`, 120 Hz | ❌ no contact sensor in original task | ✅ torque HF |
| Gear Meshing | ✅ reset + 4 steps; all three gears loaded | ✅ command + `applied_torque`, 120 Hz | ❌ no contact sensor in original task | ✅ torque HF |
| Nut Threading | ✅ reset + 4 steps | ✅ command + `applied_torque`, 120 Hz | ❌ no contact sensor in original task | ✅ torque HF |
| AutoMate Assembly | ✅ ID `00015`, reset + 4 steps | ✅ command + `applied_torque`, 120 Hz | ❌ no contact sensor in original task | ✅ torque HF |
| Wiping | ✅ reset + 80-step contact probe | ✅ MuJoCo `qfrc_actuator`, 500 Hz | ✅ wrist F/T + isolated tool-table normal load | ✅ torque/contact HF |

## Runtime evidence

| Task | Physics / policy rate | Physics samples | Robot / object collision prims | Result |
|---|---:|---:|---|---|
| Peg Insertion | 120 / 15 Hz | 32 / 32 | robot 12, socket 1, peg 1 | finite reset, observations, reward and four neutral steps |
| Gear Meshing | 120 / 15 Hz | 32 / 32 | robot 12, base 1, held 1, small 1, large 1 | finite reset, observations, reward and four neutral steps |
| Nut Threading | 120 / 15 Hz | 32 / 32 | robot 12, bolt 1, nut 1 | finite reset, observations, reward and four neutral steps |
| AutoMate `00015` | 120 / 15 Hz | 32 / 32 | robot 12, socket 1, plug 1 | finite reset, observations, reward and four neutral steps |
| Wipe | 500 / 20 Hz | 2000 / 2000 | 35 tool geoms + table | contact begins around 3.46 s; 7 markers wiped; no terminal success |

The four Isaac tasks use nine simulated Panda joints (seven arm and two finger joints), a six-dimensional end-effector delta-pose action and an operational-space controller that computes seven arm effort targets; the fingers use position targets. Factory observations are policy 19 / critic 43. AutoMate observations are policy 24 / critic 44. Wipe uses seven Panda arm joints and a 6D `OSC_POSE` delta action.

The actuator traces were non-null at every recorded physics tick. The maximum absolute values observed in these short runs were 30.36 Nm (Peg), 40.0 Nm (Gear), 40.0 Nm (Nut), 38.33 Nm (AutoMate), and 31.71 Nm (Wipe). For Isaac, the large values include the finger actuator channels; the first seven arm entries match the controller effort target in the inspected samples. `measured_effort` remains null because simulation effort is not a real motor measurement.

The original Factory and AutoMate tasks activate contact reporting on several assets but do not create a `ContactSensor` or expose an isolated tool-workpiece wrench. `applied_wrench` is an OSC controller output and is deliberately not labeled as contact force. A future contact-force constraint should add a filtered sensor for the held/fixed asset pair and validate it with a controlled collision. Wipe is already instrumented with `mj_contactForce`; its probe reached 44.97 N peak tool-table normal load and recorded nonzero force at 270 physics samples.

## AutoMate geometry selection

All 100 public AutoMate OBJ pairs were inspected. Three geometry-only representatives were selected without assigning difficulty labels:

| ID | Plug bounding box (m) | Socket bounding box (m) | Geometric character |
|---|---|---|---|
| `00004` | 0.0069 × 0.0069 × 0.0754 | 0.0725 × 0.0725 × 0.0223 | thin, long plug |
| `00190` | 0.0191 × 0.0191 × 0.0127 | 0.0182 × 0.0181 × 0.0772 | compact plug, tall socket |
| `00470` | 0.0705 × 0.0990 × 0.0160 | 0.0771 × 0.0974 × 0.0185 | broad, flat pair |

The selection audit includes face counts, grasp poses and disassembly distances in [`automate-selection-audit.json`](../artifacts/manipulation_remote_20261119/automate-selection-audit.json). The comparison image is [`automate-candidates.png`](../artifacts/manipulation_remote_20261119/automate-candidates.png). Only upstream default `00015` was run in physics; non-default IDs still require verified mass/dimension metadata before simulation.

## Visual and trace artifacts

The Wipe probe has a short [`scene.mp4`](../artifacts/manipulation_remote_20261119/wipe_probe_video_001/scene.mp4), [`first frame`](../artifacts/manipulation_remote_20261119/wipe_probe_video_001/scene_first.png), and [`last frame`](../artifacts/manipulation_remote_20261119/wipe_probe_video_001/scene_last.png). Headless Isaac RGB recording was attempted, but the RTX renderer crashed in `librtx.scenedb.plugin.so`; physics-only Isaac runs remained stable. The AutoMate image is a mesh comparison, not a simulator screenshot.

Each validated run contains `audit.json`, policy-rate `events.json`, physics-rate `trace/metadata.json`, and `trace/physics.jsonl` under [`artifacts/manipulation_remote_20261119`](../artifacts/manipulation_remote_20261119). Null fields mean unavailable signals and are never replaced by zero. Reset helper steps are separated from policy trace segments.

## Reproducible launch commands

Run inside the pinned Isaac environment after accepting the NVIDIA EULA:

```sh
python scripts/manipulation/preflight.py
python scripts/manipulation/isaac_smoke.py --task peg_insertion --headless --output outputs/manipulation/peg_001
python scripts/manipulation/isaac_smoke.py --task gear_meshing --headless --output outputs/manipulation/gear_001
python scripts/manipulation/isaac_smoke.py --task nut_threading --headless --output outputs/manipulation/nut_001
python scripts/manipulation/automate_smoke.py --assembly-id 00015 --headless --output outputs/manipulation/automate_00015_001
```

If Nucleus/S3 caching is unreliable, add `--factory-asset-dir /path/to/factory-cache` for Nut or `--automate-data-dir /path/to/automate-cache` for AutoMate. AutoMate's cache must contain `plug_grasps.json`, `disassembly_dist.json`, and `<assembly-id>/disassemble_traj.json`. The dedicated AutoMate launcher preserves Isaac Lab's required AppLauncher-first import order and works around the v2.3.2 code path that reopens auxiliary JSON by basename.

Run Wipe in its separate robosuite environment:

```sh
python scripts/manipulation/wipe_smoke.py --headless --probe-contact --video \
  --steps 80 --output outputs/manipulation/wipe_probe_001
```

## Recommended integration order

1. Prototype task-specific 120 Hz Butterworth HF costs from the four Isaac torque traces and a separate 500 Hz filter for Wipe.
2. Add and calibrate isolated Isaac held-asset/fixed-asset contact sensors before using contact force as a constraint.
3. Connect one Factory task to the existing PPO-Lagrangian adapter, starting with Peg Insertion, then Nut Threading, Gear Meshing, AutoMate and Wipe.
4. Re-enable Isaac video only after fixing the RTX headless renderer; video is independent of the validated physics and telemetry path.
