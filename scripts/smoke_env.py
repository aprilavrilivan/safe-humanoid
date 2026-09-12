"""Reset and step a configured environment with deterministic dummy actions.

The smoke test will detect registration failures, invalid tensor shapes, NaNs, reset bugs, and
basic GPU-memory problems before launching an expensive training job.
"""

