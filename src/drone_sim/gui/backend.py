from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DroneState:
    drone_id: str
    position: np.ndarray             # [x, y, z]
    velocity: np.ndarray             # [vx, vy, vz]
    radius: float
    safety_zone: float
    adaptive_safety_radius: float | None   # None for non-adaptive drones
    max_adaptive_safety_radius: float | None
    color: str | list[float]
    safety_color: str | list[float]
    trace_color: str | list[float]


@dataclass
class StepResult:
    drones: list[DroneState]
    safety_radii: list[float]         # current effective safety radius per drone
    last_collisions: list[dict]       # raw collision events from Simulator
    infeasible: bool
    infeasible_reason: str | None
    step_count: int
    t: float
    all_reached: bool = False              # True when all drones are at their destination
    admm_iteration_count: int | None = None  # None for non-ADMM coordinators


@dataclass
class SimState:
    drone_count: int
    obstacle_count: int
    obstacles: list[tuple[np.ndarray, np.ndarray]]
    coordinator_type: str             # type(coordinator).__name__ or "none"
    dt: float
    step_count: int
    room_min: np.ndarray
    room_max: np.ndarray
    config_path: str | None = None   # absolute path to loaded JSON; None before first load


class SimulationBackend(ABC):
    """Abstract contract for simulation backends used by GUI widgets."""

    @abstractmethod
    def load_config(self, path: Path) -> SimState:
        """Load a scenario JSON and return the initial simulation state."""
        ...

    @abstractmethod
    def step(self) -> StepResult:
        """Advance one simulation tick and return the resulting state."""
        ...

    @abstractmethod
    def get_state(self) -> SimState:
        """Return current simulation metadata (does not advance the simulation)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation to initial state from the last loaded config."""
        ...
