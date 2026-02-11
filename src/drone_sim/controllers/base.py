from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


@dataclass
class Controller:
   def control(self, drone: Drone, neighbors: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
               obstacles: list[tuple[np.ndarray, float]]) -> np.ndarray:
      """Return acceleration control u=[a_x,a_y,a_z].

      neighbors:
          List of (position, velocity, radius, safety_zone, reference_position).
      obstacles:
          List of (center, radius).
      """
      raise NotImplementedError
