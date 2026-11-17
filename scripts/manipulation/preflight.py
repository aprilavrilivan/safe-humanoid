#!/usr/bin/env python3
"""Read-only local dependency and upstream-task inventory."""

import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "safe_humanoid"))
from safe_humanoid.manipulation.catalog import TASKS  # noqa: E402


def main():
    modules = {name: importlib.util.find_spec(name) is not None for name in
               ("isaaclab", "isaacsim", "isaaclab_tasks", "gymnasium", "robosuite", "mujoco", "rsl_rl", "rl_games")}
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "stack_manifest": str(ROOT / "configs/stack/isaaclab-v2.3.2.yaml"),
              "modules_installed": modules,
              "task_registration": {name: "not_checked_requires_isaac_sim" if task.simulator == "isaaclab"
                                    else "not_checked_requires_robosuite" for name, task in TASKS.items()}}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
