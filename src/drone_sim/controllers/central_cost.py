from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from drone_sim.controllers.base import Controller
from drone_sim.domain.registry import register_controller
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


class CentralCostProvider(Protocol):
   """Optional interface for controllers that can be optimized centrally."""

   dt: float

   def central_bounds(self) -> tuple[np.ndarray, np.ndarray]:
      """Return (u_min, u_max) arrays of shape (3,)."""

   def central_initial_guess(self, x0: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
      """Return an initial guess u_seq of shape (H,3)."""

   def central_cost(self, u_seq: np.ndarray, x0: np.ndarray, p_ref: np.ndarray) -> float:
      """Return scalar cost for this drone."""


def as_diagonal(w: list[float] | np.ndarray) -> np.ndarray:
   w = np.asarray(w, dtype=float).reshape(-1)
   return np.diag(w)


@register_controller("mpc_agent")
@dataclass
class CentralMPCAgent(Controller):
   """Per-drone cost model for centralized MPC coordination.

   This controller is not meant to be used standalone, it exposes `central_cost(...)` which a coordinator can sum over drones,
   as long as the drones cannot run SLSQP in that architecture.
   Has to be changed in a later development, but will not disturb the paper framework.

   Note: `q_vel` is intentionally non-zero by default to discourage the optimizer from accelerating to very high speeds when the horizon is short.
   """

   dt: float
   horizon: int = 12

   q_pos: list[float] = (8.0, 8.0, 8.0)
   q_vel: list[float] = (0.5, 0.5, 0.5)
   r_u: list[float] = (0.2, 0.2, 0.2)

   u_min: list[float] = (-3.0, -3.0, -3.0)
   u_max: list[float] = (3.0, 3.0, 3.0)

   def __post_init__(self) -> None:
      self._phys = LinearKinematicsPhysics(dt=self.dt)
      self._Qp = as_diagonal(self.q_pos)
      self._Qv = as_diagonal(self.q_vel)
      self._R = as_diagonal(self.r_u)
      self._u_min = np.asarray(self.u_min, dtype=float)
      self._u_max = np.asarray(self.u_max, dtype=float)

   def central_bounds(self) -> tuple[np.ndarray, np.ndarray]:
      return self._u_min.copy(), self._u_max.copy()

   def central_initial_guess(self, x0: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
      x0 = np.asarray(x0, dtype=float).reshape(6)
      p_ref = np.asarray(p_ref, dtype=float).reshape(3)

      p = x0[:3]
      v = x0[3:]
      a = (p_ref - p) - 0.5 * v
      a = np.clip(a, self._u_min, self._u_max)
      return np.tile(a.reshape(1, 3), (self.horizon, 1))

   def central_cost(self, u_seq: np.ndarray, x0: np.ndarray, p_ref: np.ndarray) -> float:
      # Allow the coordinator to choose the horizon length.
      u_seq = np.asarray(u_seq, dtype=float).reshape((-1, 3))
      u_seq = np.clip(u_seq, self._u_min, self._u_max)

      x = np.asarray(x0, dtype=float).reshape(6)
      p_ref = np.asarray(p_ref, dtype=float).reshape(3)

      total = 0.0
      for k in range(u_seq.shape[0]):
         u = u_seq[k]
         x = self._phys.A @ x + self._phys.B @ u
         e = x[:3] - p_ref
         v = x[3:]
         total += float(e @ self._Qp @ e + v @ self._Qv @ v + u @ self._R @ u)

      return float(total)

   # Controller interface: when used standalone, we just apply the first step of the initial guess.
   def control(self, x: np.ndarray, p_ref: np.ndarray, neighbors: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
               obstacles: list[tuple[np.ndarray, float]], *, self_radius: float, self_safety_zone: float) -> np.ndarray:
      u0 = self.central_initial_guess(x, p_ref)[0]
      return np.clip(u0, self._u_min, self._u_max)
