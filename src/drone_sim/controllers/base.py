from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Controller:
   def control(self, x: np.ndarray, p_ref: np.ndarray, neighbors: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
               obstacles: list[tuple[np.ndarray, float]], *, self_radius: float, self_safety_zone: float, ) -> np.ndarray:
      """Return acceleration control u=[a_x,a_y,a_z].

      neighbors:
          List of (position, velocity, radius, safety_zone, reference_position).
      obstacles:
          List of (center, radius).
      """
      raise NotImplementedError
