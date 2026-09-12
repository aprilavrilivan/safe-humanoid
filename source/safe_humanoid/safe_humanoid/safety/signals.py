"""Common raw safety-signal interface.

Signals will be normalized, named tensors computed from robot state and commands. They remain
independent of whether a later algorithm treats them as metrics, penalties, costs, or hard
constraints.
"""

