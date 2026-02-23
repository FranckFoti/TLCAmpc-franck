from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from drone_sim.controllers.base import Controller
from drone_sim.physics.base import PhysicsModel

Color: TypeAlias = str | tuple[float, float, float]


def has_central_cost(ctrl: object) -> bool:
   """Check if controller implements the central_cost interface."""
   return all(hasattr(ctrl, name) for name in ("central_cost", "central_initial_guess", "horizon"))


@dataclass
class Route:
   waypoints: list[np.ndarray]
   target: np.ndarray
   waypoint_radius: float = 0.5
   idx: int = 0

   def current_ref(self) -> np.ndarray:
      if self.idx < len(self.waypoints):
         return self.waypoints[self.idx]
      return self.target

   def advance_if_reached(self, position: np.ndarray) -> None:
      if self.idx < len(self.waypoints):
         if np.linalg.norm(position - self.waypoints[self.idx]) <= self.waypoint_radius:
            self.idx += 1

   def target_reached(self, position: np.ndarray, thresh: float = 1e-3) -> bool:
      return np.linalg.norm(position - self.target) < thresh


@dataclass
class Drone:
   drone_id: str
   radius: float
   safety_zone: float
   cons_stop: float

   color: Color
   safety_color: Color
   trace_color: Color

   controller: Controller
   physics: PhysicsModel
   x: np.ndarray  # [x,y,z,vx,vy,vz]
   route: Route

   # Adaptive safety zone parameter. When set, the drone uses a velocity-dependent
   # safety radius instead of the fixed safety_zone.
   alpha: float | None = None

   @property
   def is_adaptive(self) -> bool:
      """Whether this drone uses velocity-dependent adaptive safety zones."""
      return self.alpha is not None

   @property
   def v_max(self) -> float:
      return self.physics.v_max()

   def compute_adaptive_radius(self, velocity: np.ndarray) -> float:
      """Compute safety radius based on current velocity.

      :param velocity: 3D velocity vector [vx, vy, vz].
      :return: safety radius (fixed safety_zone or adaptive r_min + alpha * s_stop).
      """
      if not self.is_adaptive:
         return self.safety_zone
      u_min, u_max = self.bounds()
      u_max_scalar = float(np.min(np.abs(u_max)))
      v_norm_sq = float(np.dot(velocity, velocity))
      s_stop = v_norm_sq / (2.0 * u_max_scalar)
      return max(self.safety_zone, self.radius + self.alpha * s_stop)

   def compute_max_adaptive_radius(self) -> float:
      """Compute the maximum possible adaptive safety radius (at v_max).

      For non-adaptive drones returns safety_zone unchanged.
      For adaptive drones, computes stopping distance assuming v_max speed.

      :return: maximum safety radius (>= safety_zone).
      """
      if not self.is_adaptive:
         return self.safety_zone
      u_min, u_max = self.bounds()
      u_max_scalar = float(np.min(np.abs(u_max)))
      v_max_sq = self.v_max ** 2
      s_stop_max = v_max_sq / (2.0 * u_max_scalar)
      return max(self.safety_zone, self.radius + self.alpha * s_stop_max)

   def predict(self, u: np.ndarray) -> np.ndarray:
      return self.physics.step(self.x, u)

   def bounds(self) -> tuple[list[float], list[float]]:
      return self.physics.central_bounds()

   def position(self) -> np.ndarray:
      return self.x[:3]

   def velocity(self) -> np.ndarray:
      return self.x[3:]
