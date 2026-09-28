# safe-humanoid

An external Isaac Lab project for studying Unitree G1 velocity tracking and
deployment-relevant signals. The software stack is pinned to Isaac Lab 2.3.2 and
Isaac Sim 5.1.0 in `configs/stack/isaaclab-v2.3.2.yaml`.

## Current status

The **code-first phase** contains a clean-reward G1 task, four velocity
scenarios, RSL-RL PPO training and checkpoint-evaluation entry points, and
policy-rate raw metrics. The second code phase adds two G1 reference-motion
tracking studies. The third code phase adds physics-step velocity and
approximate actuator-torque capture, offline time/frequency analysis, and
provenance-checked (currently uncalibrated) hardware-limit configuration.
Configuration parsing, motion-file validation and metric arithmetic are tested
without Isaac Sim. On an Ubuntu 22.04 RTX 5090 host, the velocity-task smoke
tests, 200 Hz physics recorder, two-iteration RSL-RL integration run and
one-checkpoint evaluation have also passed headlessly. These short checks do
**not** establish locomotion quality or safety. Real retargeted motion clips
are not included, so the tracking environments have not yet been simulator-
tested. G1 continuous hardware limits and safe-RL algorithm adapters remain
future work.

The Gym IDs serve different purposes:

- `Isaac-Velocity-Flat-G1-v0`: official reference task.
- `SafeHumanoid-Velocity-Flat-G1-v0`: intentionally unchanged project-owned
  clone, used for the exact-contract smoke test.
- `SafeHumanoid-Velocity-Clean-Flat-G1-v0`: task-only reward
  `r_vx + r_vy + r_wz + r_upright`, with equal weights. It keeps the official
  G1 scene, policy observations, joint-position action, and fall termination.

The four files in `configs/task/velocity/` are JSON-compatible YAML, allowing
validation with only the Python standard library. Their numeric settings are
**pilot parameters**, not validated G1 safety limits. Training samples commands
from the specified ranges. Abrupt training resamples every 0.4–1.0 seconds;
push-recovery training gives each episode one randomly timed, random-direction
planar velocity kick. Evaluation instead uses a held-out, episode-relative
schedule. In particular, abrupt evaluation has explicit 0.5-second command
changes, and push-recovery applies one world-frame **velocity increment** at 5 seconds.
That kick is an impulse surrogate in m/s, **not** a calibrated force in N.

## GPU-free checks (Mac or Linux)

```bash
python3 -m unittest discover -s tests/unit -v
python3 -m compileall -q source/safe_humanoid scripts tests
```

## Linux GPU sequence (headless verified)

Start with the host and official baseline before training. Use the pinned
stack and a supported NVIDIA host; the package itself is installed by the
bootstrap script.

On a minimal Ubuntu image, run the preflight doctor first. If it reports
missing `libvulkan.so.1`, `libEGL.so.1`, `libXt.so.6`, or `libGLU.so.1`, install the matching
system packages with administrator privileges before launching Isaac Sim:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends libvulkan1 libegl1 libxt6 libglu1-mesa
```

The bootstrap intentionally does not change system packages or accept NVIDIA's
EULA. The first simulator launch asks for explicit acceptance.

```bash
python3 scripts/doctor.py --mode preflight
bash scripts/setup/bootstrap_linux.sh --dry-run
bash scripts/setup/bootstrap_linux.sh
source .venv/bin/activate
python scripts/doctor.py --mode full --probe-task
python scripts/smoke_env.py --headless --num-envs 4 --steps 32
python scripts/smoke_env.py --headless --scenario nominal --num-envs 2 --steps 32
python scripts/smoke_env.py --headless --scenario abrupt --exercise-timing --num-envs 1 --steps 405
python scripts/smoke_env.py --headless --scenario push_recovery --exercise-timing --num-envs 1 --steps 255
python scripts/smoke_safety.py --headless --num-envs 2 --steps 64
python -m unittest discover -s tests/sim -v
```

The smoke script runs zero actions: it checks creation, reset, stepping, tensor
shapes/devices and finite values, **not locomotion competence**. For the clean
task, it permits intentional reward and event changes while checking the
official policy I/O and timing contract.
The timing checks temporarily disable contact termination so an untrained,
zero-action robot can reach the scripted change/push times; they are not
performance evaluations. On the tested cloud image, installing `libegl1`
resolved Vulkan initialization; viewer/rendering operation was not validated.

After smoke succeeds, start a short run before scaling parallelism:

```bash
python scripts/train.py --scenario nominal --num-envs 64 --max-iterations 10
python scripts/train.py --scenario nominal --num-envs 512
python scripts/evaluate_all.py --checkpoint logs/rsl_rl/safe_humanoid_g1_clean_velocity/RUN/model_XXX.pt
```

Training artifacts appear under `logs/rsl_rl/`. Evaluation writes one
`metadata.json`, `trace.csv`, and `summary.json` per scenario, plus
`comparison.csv`. The policy-rate part of the report includes tracking errors,
upright alignment, approximate joint torque/speed and approximate absolute
mechanical power, as well as termination, timeout, and actual push-exposure
counts. These are **raw policy-rate signals**, not physical-limit claims. The
trace's state is sampled before an action; its reward and done flags describe
the following transition. A single checkpoint should be evaluated on all
four scenarios for a fair first comparison.

### Reward-engineering PPO matrix

The core comparison is **4 training scenarios × 3 reward profiles**: clean PPO,
PPO with a time-domain sustained-loading penalty, and PPO with a frequency-
domain high-band penalty. The latter two retain the same policy observations,
actions, PPO configuration and four positive task rewards. A GPU-resident
recorder reads approximate joint torque and joint velocity at the 200 Hz
physics rate. Time cost uses a 0.5-second rolling torque RMS; frequency cost
uses a 128-sample Hann window, 64-sample hop and 25–100 Hz band, normalized by
the explicitly provisional reference scales in
`configs/experiments/velocity_reward_ppo.yaml`. Isaac Lab multiplies reward
weights by the 50 Hz environment step duration. These are soft, uncalibrated
loading/oscillation proxies—not hardware-limit violations. The independent
evaluation still uses the clean task reward and reports raw physics-rate safety
metrics for every policy.

The training-time abrupt command generator, random push, both online reward
buffers, two-iteration PPO updates, and W&B offline writer have passed these
short headless checks on the Ubuntu 22.04 RTX 5090 host:

```bash
python scripts/smoke_training.py --scenario abrupt --reward-profile clean --steps 400
python scripts/smoke_training.py --scenario push_recovery --reward-profile clean --steps 400
python scripts/smoke_training.py --scenario nominal --reward-profile time --steps 150
python scripts/smoke_training.py --scenario nominal --reward-profile frequency --steps 150
python scripts/train.py --scenario abrupt --reward-profile time --num-envs 64 --max-iterations 2
python scripts/train.py --scenario push_recovery --reward-profile frequency --num-envs 64 --max-iterations 2
```

The bootstrap pins and installs `wandb` without configuring an account. W&B
online logging is opt-in and targets the `ivanxu-uc-berkeley/safe-humanoid`
team/project by default. Enter an **API key, not an account password**, in
your own terminal; never paste it into a config, command history or repository:

```bash
read -rs WANDB_API_KEY
export WANDB_API_KEY
export OMNI_KIT_ACCEPT_EULA=YES  # only after you personally accept NVIDIA's EULA
python3 scripts/reward_matrix.py plan --batch-name reward-ppo-v1 --seeds 0 1 2
python3 scripts/reward_matrix.py run --batch-name reward-ppo-v1 --seeds 0 1 2
```

`plan` changes nothing; `run` explicitly starts 36 trainings in waves of at
most four (default 512 environments, 1500 PPO iterations). Choose a smaller
seed list or `--parallel 1` if desired. Failed waves stop the next wave and
leave checkpoints/logs intact. Each run stores a scenario and reward-config
fingerprint plus its W&B run ID. The training writer uploads PPO scalars and
checkpoints to W&B; evaluation is a separate, explicit step after checking
checkpoint quality:

```bash
python scripts/evaluate_all.py \
  --checkpoint logs/rsl_rl/reward_matrix/reward-ppo-v1/abrupt_time_seed0/model_1499.pt \
  --wandb
```

After every expected checkpoint exists, the batch evaluator runs every policy
on all four held-out scripts using seed 10000 by default:

```bash
python3 scripts/reward_matrix.py evaluate --batch-name reward-ppo-v1 --seeds 0 1 2
```

The evaluator appends four scenario summaries and offline-analysis
scalars to that training W&B run; raw traces remain in the local output folder.
Use held-out evaluation seeds and at least three training seeds before making
comparative claims. A W&B offline smoke is possible with `--logger wandb
--wandb-offline`; later upload requires `wandb sync` after authentication.

### Four-scenario pilot: local planning and artifact audit

`configs/experiments/velocity_baseline_pilot.yaml` fixes a **short integration
pilot**, not a converged training budget or a safety threshold. On a Mac, the
following command validates that manifest and all four scenario files and
prints the exact commands for the next Linux GPU session. It does not start
Isaac Sim, train, or rent a machine:

```bash
python3 scripts/velocity_experiment.py plan
```

After following the printed Linux train/evaluate commands, keep the four
scenario directories and `comparison.csv` together under the output root.
The audit can run on Linux or on a Mac after copying that output root back:

```bash
python3 scripts/velocity_experiment.py audit \
  --run-root outputs/evaluation/YOUR_RUN \
  --output reports/generated/velocity_pilot_audit.json
```

The audit checks scenario/config fingerprints, one checkpoint hash and seed
across all four evaluations, completed episode and summary counts, 200 Hz
physics-analysis provenance, and actual scripted-event exposure. For nominal,
aggressive and abrupt it needs at least one completed episode that saw every
scheduled command segment, including the final changes at 14, 12 and 8 seconds;
for push-recovery it needs a completed episode with `push_fired=True` at the 5-second
event. These are minimal **data-completeness** gates. If the pilot policy falls
early, the audit correctly reports `incomplete`; that does not by itself mean
the task implementation failed. An audit pass does not establish good tracking,
statistical significance, or physical safety. Decide and record a proper
training/evaluation budget only after this pilot has exposed the events.

## Physics-rate safety analysis (third code phase)

By default, `scripts/evaluate.py` also attaches a bounded recorder to the
clean G1 task (at most 8 environments, 3 episodes/environment by default).
With the pinned 0.005-second physics step and four-step action decimation,
this captures a 200 Hz trace instead of relying on 50 Hz policy snapshots.
The recorder writes `physics_metadata.json` and `physics_trace.csv`; after
evaluation, the offline analyzer writes `safety_summary.json`, and
`scripts/report.py` includes its scalar aggregates in `comparison.csv`.
For an existing trace, analysis can be rerun on a laptop without Isaac Sim:

```bash
python3 scripts/analyze_safety.py outputs/evaluation/YOUR_RUN
```

The time-domain summary reports peaks, a 0.5-second rolling torque RMS,
finite-difference joint acceleration, and approximate absolute mechanical
power/energy. Welch analysis uses 128 physics samples per Hann window with a
64-sample hop; it reports joint-averaged and worst-joint band power in 0–10,
10–25, and 25–100 Hz bands. Short episodes yield `null` spectral metrics, not
a made-up zero. All windows and derivatives stay within one environment's
episode. These are descriptive metrics, not online safety costs.

The stock Isaac Lab `G1_MINIMAL_CFG` uses **implicit PhysX actuators**: its
`applied_torque` field is an approximate PD-model effort, not measured motor
torque or exact solver output. The recorder labels this estimate and reads joint
velocity directly from the PhysX view after each step; the latter avoids the
cached velocity value that is updated after the recorder callback. Thus torque,
torque-derived power and torque spectra must not be called hardware measurements.
`configs/safety/g1_limits.yaml` remains explicitly **uncalibrated**. Its empty
maps produce `null` limit-exceedance fractions and no physical safety claim.
Even a later source-backed limit map cannot turn the approximate torque signal
into a physical torque-violation measurement; only the speed-threshold
comparison can become available. Isaac Lab actuator caps and URDF or SDK
software guards are not interchangeable with continuous hardware ratings.
Record the exact joint names, units, manufacturer source and asset match before
changing the calibration status. On a Linux GPU host, first validate recorder
construction, 200 Hz sample count, signal variation, reset boundaries, and
evaluation wall-clock cost with a short run. Recorder construction, sample
count and offline analysis passed a short headless run; longer-run signal and
reset-boundary validation is still required before using these metrics in a
study.

## G1 whole-body tracking pilot (second code phase)

The project registers `SafeHumanoid-Tracking-SquatStand-G1-v0` and
`SafeHumanoid-Tracking-FastLegSwing-G1-v0`. Both use the pinned Isaac Lab 2.3.2
**official flat G1 scene and joint-position action**, with an episode-relative
reference command. The policy sees the target position and velocity of **every
simulated joint**, root height and root quaternion; four positive terms reward
matching joint positions, joint velocities, root height and root orientation.
Reset places the robot at reference frame 0. These are deliberately simple
task rewards: this is **not** a verified BeyondMimic reproduction, and the
scenario numbers are pilot thresholds, not safety constraints. In particular,
the pilot does not yet use per-body pose/velocity rewards, adaptive phase
sampling, or custom G1 cylinder assets.

`configs/task/tracking/squat_stand.yaml` requires a clearly varying root
height. `fast_leg_swing.yaml` requires a clearly high leg-joint speed. Both
require a user-supplied **already retargeted** G1 reference clip at least two
seconds long. The repository intentionally does not ship AMASS, Unitree, or
other licensed motion data. Local clips can live under ignored `data/motions/`,
or anywhere outside the repository. A clip can be `.npz` (NumPy available on
the Linux stack) or `.json` (small debugging clips). Its schema-version-1
fields are:

- `schema_version`: scalar integer `1`; `fps`: positive scalar frames/second.
- `joint_names`: one unique string per simulated G1 joint, in source-column
  order; `joint_pos` and `joint_vel`: `[frames, joints]` in rad and rad/s.
- `root_pos_w`: `[frames, 3]` in metres, with x/y relative to the local
  environment origin and z above the ground; `root_quat_w`: `[frames, 4]`
  unit quaternions in **wxyz** order.

All fields must be finite and all arrays must have the same frame count.
At simulator creation the clip's joint names must exactly match the G1 asset's
names; columns are then reordered to simulator order and checked against its
soft joint limits. No matching by array length alone is allowed. The
[BeyondMimic motion format](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md)
contains body-state arrays but **does not include joint names or these root
fields in this exact schema**, so it cannot be passed directly. Its published
stack targets Isaac Lab 2.1, whereas this project pins 2.3.2. Any import must
first establish an asset/joint mapping and produce the schema above. Do not
interpret a rejected motion as a code or GPU failure.

Check each real clip on any machine, then smoke-test and train on Linux GPU:

```bash
python3 scripts/check_motion.py --scenario squat_stand --motion /path/to/g1_squat.npz
python3 scripts/check_motion.py --scenario fast_leg_swing --motion /path/to/g1_leg_swing.npz
python scripts/smoke_tracking.py --headless --scenario squat_stand --motion /path/to/g1_squat.npz --num-envs 2 --steps 16
python scripts/train_tracking.py --scenario squat_stand --motion /path/to/g1_squat.npz --num-envs 64 --max-iterations 10
```

The smoke test checks environment creation, finite policy tensors/rewards, and
reset/step mechanics—not skill. The PPO runner inherits the official G1 flat
starting settings and has not been tuned for tracking. `tests/sim/test_tracking.py`
will exercise a supplied clip when `SAFE_HUMANOID_SQUAT_MOTION` and/or
`SAFE_HUMANOID_LEG_SWING_MOTION` is set on a Linux GPU host. Actual retargeting,
contact feasibility, training convergence, and safety measurements remain
GPU/data-dependent validation tasks.

## Layout

- `source/safe_humanoid`: installable Isaac Lab extension and task code.
- `configs/task/velocity`: versioned velocity scenario specifications.
- `configs/task/tracking`: two reference-motion study specifications.
- `configs/experiments`: reproducible pilot budgets and evaluation gates.
- `scripts`: setup, smoke, training, evaluation and offline summary commands.
- `tests/unit`: checks that require neither GPU nor Isaac Sim.
- `tests/sim`: Linux GPU regression checks; skipped on the Mac.
