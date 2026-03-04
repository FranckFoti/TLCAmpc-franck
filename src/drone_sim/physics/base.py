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

   def predict_trajectory(self, x0: np.ndarray, u_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Predict position and velocity trajectories from initial state and controls.

      :param x0: Initial state (6,).
      :param u_seq: Control sequence (H, 3).
      :return: (positions (H, 3), velocities (H, 3)).
      """
      H = u_seq.shape[0]
      positions = np.zeros((H, 3), dtype=float)
      velocities = np.zeros((H, 3), dtype=float)
      x = np.asarray(x0, dtype=float).reshape(6)
      for k in range(H):
         x = self.step(x, u_seq[k])
         positions[k] = x[:3]
         velocities[k] = x[3:6]
      return positions, velocities

   def central_bounds(self) -> tuple[list[float], list[float]]:
      return self._u_min, self._u_max
