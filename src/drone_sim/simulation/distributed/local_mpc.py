from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from drone_sim.domain.constraints import (MovingObstacleAvoidanceConstraints, ObstacleAvoidanceConstraints, RoomConstraints)

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


def _pad_or_trim_horizon(u: np.ndarray, horizon: int) -> np.ndarray:
   """Pad or trim a control sequence to match the given horizon."""
   u = np.asarray(u, dtype=float).reshape((-1, 3))
   if u.shape[0] < horizon:
      pad = np.tile(u[-1:], (horizon - u.shape[0], 1))
      u = np.concatenate([u, pad], axis=0)
   elif u.shape[0] > horizon:
      u = u[:horizon]
   return u


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

   def solve(self, drone: Drone, neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]], obstacles: list[tuple[np.ndarray, float]] | None = None,
         room_min: np.ndarray | None = None, room_max: np.ndarray | None = None, u_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
      """ Solve local MPC problem for a single drone.

      :param drone: Drone object with state, route, controller, and physics
      :param neighbor_trajectories: Dict mapping neighbor_id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
      :param obstacles: List of (center, radius) static obstacles
      :param room_min: Room lower bounds (3,) or None
      :param room_max: Room upper bounds (3,) or None
      :param u_prev: Previous control sequence (H,3) for warm-start
      :return:
         u_opt: Optimized control sequence (horizon, 3)
         traj_opt: Optimized position trajectory (horizon, 3)
         success: Whether optimization succeeded
      """
      controller = drone.controller
      obstacles = obstacles or []

      u_min, u_max = drone.bounds()
      horizon = self.horizon

      # Initial guess
      if u_prev is not None:
         # Warm-start: shift previous solution
         u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
      else:
         u0 = _pad_or_trim_horizon(controller.central_initial_guess(drone), horizon)

      u0 = np.clip(u0, u_min, u_max)

      # Pre-build constraint evaluators (stateless, depend only on horizon)
      collision_c = MovingObstacleAvoidanceConstraints(horizon=horizon)
      obstacle_c = ObstacleAvoidanceConstraints(horizon=horizon)
      room_c = RoomConstraints(horizon=horizon) if room_min is not None and room_max is not None else None

      def cost(u_flat: np.ndarray) -> float:
         u = u_flat.reshape((horizon, 3))
         u = np.clip(u, u_min, u_max)
         return controller.central_cost(u, drone)

      def constraints(u_flat: np.ndarray) -> np.ndarray:
         u = u_flat.reshape((horizon, 3))
         u = np.clip(u, u_min, u_max)
         predicted_positions, predicted_velocities = self._predict_states(drone, u)

         vals = np.array([], dtype=float)
         vals = collision_c.evaluate_single(drone, predicted_positions, neighbor_trajectories, vals, pred_vel=predicted_velocities)
         vals = obstacle_c.evaluate_single(drone, predicted_positions, obstacles, vals, pred_vel=predicted_velocities)
         if room_c is not None:
            vals = room_c.evaluate_single(drone, predicted_positions, room_max, room_min, vals, pred_vel=predicted_velocities)

         return vals

      # Build bounds
      axis_bounds = [(float(u_min[a]), float(u_max[a])) for a in range(3)]
      bounds = axis_bounds * horizon

      # Optimize
      result = minimize(cost, u0.flatten(), method="SLSQP", bounds=bounds, constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": self.max_iter, "ftol": self.f_tol, "disp": False})

      u_opt = np.clip(result.x.reshape((horizon, 3)), u_min, u_max)
      traj_opt, _ = self._predict_states(drone, u_opt)

      # Check constraint satisfaction
      g = constraints(u_opt.flatten())
      feasible = result.success and (len(g) == 0 or g.min() >= -1e-6)

      return u_opt, traj_opt, feasible

   def _predict_states(self, drone: Drone, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Predict position and velocity trajectories from state and controls.

      :param drone: Drone with initial state and physics model
      :param u: Control sequence (horizon, 3)
      :return: Tuple of (predicted_positions, predicted_velocities), each (horizon, 3)
      """
      horizon = u.shape[0]

      x = np.asarray(drone.x, dtype=float).reshape(6)
      positions = np.zeros((horizon, 3), dtype=float)
      velocities = np.zeros((horizon, 3), dtype=float)

      for step in range(horizon):
         x = drone.physics.step(x, u[step])
         positions[step] = x[:3]
         velocities[step] = x[3:6]

      return positions, velocities
