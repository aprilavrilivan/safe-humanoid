"""Register the project-owned Unitree G1 flat-terrain task."""

import gymnasium as gym


TASK_ID = "SafeHumanoid-Velocity-Flat-G1-v0"
CLEAN_TASK_ID = "SafeHumanoid-Velocity-Clean-Flat-G1-v0"
OFFICIAL_TASK_ID = "Isaac-Velocity-Flat-G1-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SafeHumanoidG1FlatEnvCfg",
    },
)

gym.register(
    id=CLEAN_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:SafeHumanoidG1CleanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:SafeHumanoidG1PPORunnerCfg",
    },
)

__all__ = ["CLEAN_TASK_ID", "OFFICIAL_TASK_ID", "TASK_ID"]
