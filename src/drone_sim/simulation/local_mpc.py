from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from drone_sim.domain.constraints import (MovingObstacleAvoidanceConstraints, ObstacleAvoidanceConstraints, RoomConstraints, )
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


@dataclass
class LocalMPCSolver:
   """Per-drone MPC solver for distributed optimization.

   Solves a local MPC problem for a single drone, treating neighbor
   predicted trajectories as fixed moving obstacles.

   Used as the primal update step in ADMM-based distributed MPC.
   """

   dt: float
   horizon: int

   # Optimizer settings
   max_iter: int = 100
   f_tol: float = 1e-4

   def solve(self, drone: Drone, neighbor_trajectories: dict[str, tuple[np.ndarray, float]], obstacles: list[tuple[np.ndarray, float]] | None = None,
         room_min: np.ndarray | None = None, room_max: np.ndarray | None = None, u_prev: np.ndarray | None = None, ) -> tuple[np.ndarray, np.ndarray, bool]:
      """Solve local MPC problem for a single drone.

      Args:
          drone: Drone object with state, route, controller, and physics
          neighbor_trajectories: Dict mapping neighbor_id to (trajectory (H,3), safety_zone)
          obstacles: List of (center, radius) static obstacles
          room_min: Room lower bounds (3,) or None
          room_max: Room upper bounds (3,) or None
          u_prev: Previous control sequence (H,3) for warm-start

      Returns:
          u_opt: Optimized control sequence (H, 3)
          traj_opt: Optimized position trajectory (H, 3)
          success: Whether optimization succeeded
      """
      x0 = np.asarray(drone.x, dtype=float).reshape(6)
      controller = drone.controller
      obstacles = obstacles or []

      u_min = drone.bounds()[0]
      u_max = drone.bounds()[1]
      H = self.horizon

      # Initial guess
      if u_prev is not None:
         # Warm-start: shift previous solution
         u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
      else:
         u0 = controller.central_initial_guess(drone)
         # Ensure correct horizon length
         if u0.shape[0] < H:
            pad = np.tile(u0[-1:], (H - u0.shape[0], 1))
            u0 = np.concatenate([u0, pad], axis=0)
         elif u0.shape[0] > H:
            u0 = u0[:H]

      u0 = np.clip(u0, u_min, u_max)

      def cost(u_flat: np.ndarray) -> float:
         u = u_flat.reshape((H, 3))
         u = np.clip(u, u_min, u_max)
         return controller.central_cost(u, drone)

      def constraints(u_flat: np.ndarray) -> np.ndarray:
         u = u_flat.reshape((H, 3))
         u = np.clip(u, u_min, u_max)
         P = self._predict_positions(drone, u)

         vals = np.array([], dtype=float)

         # Neighbor collision avoidance (no cons_stop — not known for neighbors)
         collision_c = MovingObstacleAvoidanceConstraints(horizon=H)
         vals = collision_c.evaluate_single(drone, P, neighbor_trajectories, vals)

         # Static obstacles
         obstacle_c = ObstacleAvoidanceConstraints(horizon=H)
         vals = obstacle_c.evaluate_single(drone, P, obstacles, vals)

         # Room constraints
         if room_min is not None and room_max is not None:
            room_c = RoomConstraints(horizon=H)
            vals = room_c.evaluate_single(drone, P, room_max, room_min, vals)

         return vals

      # Build bounds
      bounds = []
      for _ in range(H):
         for d in range(3):
            bounds.append((float(u_min[d]), float(u_max[d])))

      # Optimize
      result = minimize(cost, u0.flatten(), method="SLSQP", bounds=bounds, constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": self.max_iter, "ftol": self.f_tol, "disp": False}, )

      u_opt = np.clip(result.x.reshape((H, 3)), u_min, u_max)
      traj_opt = self._predict_positions(drone, u_opt)

      # Check constraint satisfaction
      g = constraints(u_opt.flatten())
      feasible = result.success and (len(g) == 0 or g.min() >= -1e-6)

      return u_opt, traj_opt, feasible

   def _predict_positions(self, drone: Drone, u: np.ndarray) -> np.ndarray:
      """Predict position trajectory from state and controls.

      Args:
          x0: Initial state (6,)
          u: Control sequence (H, 3)

      Returns:
          P: Position trajectory (H, 3)
      """
      H = u.shape[0]

      x = np.asarray(drone.x, dtype=float).reshape(6)
      predicted = np.zeros((H, 3), dtype=float)

      for k in range(H):
         x = drone.physics.step(x, u[k])
         predicted[k] = x[:3]

      return predicted
