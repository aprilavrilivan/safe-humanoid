"""Existing upstream environments; this package does not register or alter them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    simulator: str
    environment_id: str
    robot: str


TASKS = {
    "peg_insertion": Task("Peg Insertion", "isaaclab", "Isaac-Factory-PegInsert-Direct-v0", "Franka"),
    "gear_meshing": Task("Gear Meshing", "isaaclab", "Isaac-Factory-GearMesh-Direct-v0", "Franka"),
    "nut_threading": Task("Nut Threading", "isaaclab", "Isaac-Factory-NutThread-Direct-v0", "Franka"),
    "automate_assembly": Task("AutoMate Assembly", "isaaclab", "Isaac-AutoMate-Assembly-Direct-v0", "Franka"),
    "wiping": Task("Wiping", "robosuite", "Wipe", "Panda"),
}


def get_task(name: str) -> Task:
    return TASKS[name]
