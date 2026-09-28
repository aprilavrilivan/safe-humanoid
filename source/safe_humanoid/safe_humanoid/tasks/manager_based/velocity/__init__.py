"""G1 velocity-tracking task family.

This module registers an unchanged upstream clone and a clean-reward G1 task.
Nominal, aggressive, abrupt, and push-recovery are configuration overlays.
"""

from . import config as config

__all__ = ["config"]
