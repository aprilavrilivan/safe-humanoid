"""External Isaac Lab extension for safe humanoid-control experiments.

Pure data-contract modules remain importable on a laptop without Isaac Lab.
Gym task registration happens when the simulator environment is installed.
"""

try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    from . import tasks as tasks

__version__ = "0.1.0"

__all__ = ["__version__"]
