# safe-humanoid

This repository will host an external Isaac Lab project for studying deployment-safe
Unitree G1 control. The initial scaffold intentionally contains responsibilities and
interfaces rather than working simulator code.

The planned separation is:

1. `tasks`: what the robot must do;
2. `safety`: how raw deployment-safety signals are computed;
3. `telemetry`: how high-rate traces and episode summaries are recorded;
4. `adapters`: how the same task is exposed to later safe-RL libraries;
5. `configs`: which task scenario and safety definition an experiment uses;
6. `scripts`: stable command-line entry points for setup, training, and evaluation.

The first runnable milestone will be a manager-based G1 velocity-tracking task. Whole-body
motion tracking and OmniSafe integration will be added only after that baseline is verified
on a supported Linux NVIDIA GPU host.

