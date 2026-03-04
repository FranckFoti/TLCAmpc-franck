from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from drone_sim.controllers.base import Controller
from drone_sim.domain.registry import register_controller

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


class CentralCostProvider(Protocol):
   """Optional interface for controllers that can be optimized centrally."""

   dt: float

   def central_initial_guess(self, drone: Drone) -> np.ndarray:
      """Return an initial guess u_seq of shape (H,3)."""

   def central_cost(self, u_seq: np.ndarray, drone: Drone) -> float:
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

   def __post_init__(self) -> None:
      self._Qp = as_diagonal(self.q_pos)
      self._Qv = as_diagonal(self.q_vel)
      self._R = as_diagonal(self.r_u)

   def central_initial_guess(self, drone: Drone) -> np.ndarray:
      x0 = np.asarray(drone.x, dtype=float).reshape(6)
      p_ref = np.asarray(drone.route.current_ref(), dtype=float).reshape(3)

      p = x0[:3]
      v = x0[3:]
      a = (p_ref - p) - 0.5 * v
      u_min, u_max = drone.bounds()
      a = np.clip(a, u_min, u_max)
      return np.tile(a.reshape(1, 3), (self.horizon, 1))

   def central_cost(self, u_seq: np.ndarray, drone: Drone) -> float:
      # Allow the coordinator to choose the horizon length.
      u_seq = np.asarray(u_seq, dtype=float).reshape((-1, 3))

      p_ref = np.asarray(drone.route.current_ref(), dtype=float).reshape(3)

      positions, velocities = drone.physics.predict_trajectory(drone.x, u_seq)

      errors = positions - p_ref  # (H, 3)
      # Vectorized quadratic forms: sum of e^T Q e for diagonal Q = sum(q * e^2)
      qp = np.diag(self._Qp)
      qv = np.diag(self._Qv)
      r = np.diag(self._R)
      total = float(
         np.sum(errors ** 2 * qp)
         + np.sum(velocities ** 2 * qv)
         + np.sum(u_seq ** 2 * r)
      )
      return total

   # Controller interface: when used standalone, we just apply the first step of the initial guess.
   def control(self, drone: Drone, neighbors: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
               obstacles: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
      return self.central_initial_guess(drone)[0]


@register_controller("mpc_agent_adaptive")
@dataclass
class AdaptiveMPCAgent(CentralMPCAgent):
   """Per-drone cost model with velocity magnitude penalty for adaptive safety zones.

   Adds ``lambda_vel * ||v||^2`` per step on top of the parent cost.
   When drones use adaptive safety zones, higher velocity leads to larger
   safety radii.  The velocity penalty encourages deceleration near
   conflicts, naturally shrinking the safety zone.
   """

   lambda_vel: float = 1.0

   def central_cost(self, u_seq: np.ndarray, drone: Drone) -> float:
      u_seq = np.asarray(u_seq, dtype=float).reshape((-1, 3))

      p_ref = np.asarray(drone.route.current_ref(), dtype=float).reshape(3)

      positions, velocities = drone.physics.predict_trajectory(drone.x, u_seq)

      errors = positions - p_ref
      qp = np.diag(self._Qp)
      qv = np.diag(self._Qv)
      r = np.diag(self._R)
      total = float(
         np.sum(errors ** 2 * qp)
         + np.sum(velocities ** 2 * qv)
         + np.sum(u_seq ** 2 * r)
         + self.lambda_vel * np.sum(velocities ** 2)
      )
      return total
