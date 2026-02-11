from __future__ import annotations

import numpy as np


class PhysicsModel:
   def __init__(self, dt: float, v_max: float = 5.0, u_min=None, u_max=None):
      self.dt = dt
      self._v_max = v_max
      self._u_min = u_min if u_min is not None else [-3.0, -3.0, -3.0]
      self._u_max = u_max if u_max is not None else [3.0, 3.0, 3.0]

   def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
      raise NotImplementedError

   def v_max(self) -> float:
      return self._v_max

   def central_bounds(self) -> tuple[list[float], list[float]]:
      return self._u_min, self._u_max
