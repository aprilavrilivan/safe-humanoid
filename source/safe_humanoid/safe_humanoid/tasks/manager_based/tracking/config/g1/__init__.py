"""Register two project-owned G1 motion-tracking studies.

A motion file is intentionally not bundled. Callers must supply a validated
retargeted G1 clip before creating either environment.
"""

import gymnasium as gym

SQUAT_STAND_TASK_ID = "SafeHumanoid-Tracking-SquatStand-G1-v0"
FAST_LEG_SWING_TASK_ID = "SafeHumanoid-Tracking-FastLegSwing-G1-v0"

for task_id in (SQUAT_STAND_TASK_ID, FAST_LEG_SWING_TASK_ID):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:SafeHumanoidG1TrackingEnvCfg",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.agents.rsl_rl_ppo_cfg:SafeHumanoidG1TrackingPPORunnerCfg"
            ),
        },
    )

__all__ = ["FAST_LEG_SWING_TASK_ID", "SQUAT_STAND_TASK_ID"]
