from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from drone_sim.domain.constraints import (MovingObstacleAvoidanceConstraints, ObstacleAvoidanceConstraints, RoomConstraints)

_log = logging.getLogger(__name__)

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
   symmetry_break_eps: float = 0.05  # Random noise magnitude for symmetry breaking

   def solve(self, drone: Drone, neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]], obstacles: list[tuple[np.ndarray, np.ndarray]] | None = None,
         room_min: np.ndarray | None = None, room_max: np.ndarray | None = None, u_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
      """ Solve local MPC problem for a single drone.

      :param drone: Drone object with state, route, controller, and physics
      :param neighbor_trajectories: Dict mapping neighbor_id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
      :param obstacles: List of (center, half_extents) static obstacles
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

      # Symmetry-breaking noise to prevent collinear deadlocks
      if self.symmetry_break_eps > 0:
         noise = np.random.uniform(
            -self.symmetry_break_eps, self.symmetry_break_eps, size=u0.shape,
         )
         u0 = np.clip(u0 + noise, u_min, u_max)

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

      # Infeasible fallback: decelerate, then hold position
      if not feasible:
         u_opt, traj_opt, feasible = self._infeasible_fallback(
            drone, u_opt, traj_opt, constraints, u_min, u_max, horizon,
         )

      if _log.isEnabledFor(logging.DEBUG):
         self._debug_log_feasibility_check(drone, neighbor_trajectories, feasible, result, traj_opt, g)

      return u_opt, traj_opt, feasible

   def _infeasible_fallback(
      self,
      drone: Drone,
      u_opt: np.ndarray,
      traj_opt: np.ndarray,
      constraints_fn,
      u_min: np.ndarray,
      u_max: np.ndarray,
      horizon: int,
   ) -> tuple[np.ndarray, np.ndarray, bool]:
      """Try safe fallback controls when the optimizer returns infeasible.

      Fallback order:
      1. Decelerate — brake toward zero velocity
      2. Hold position — zero control (no acceleration, let the other drone resolve)

      :return: (u, trajectory, feasible) for the best fallback found
      """
      v_current = np.asarray(drone.x, dtype=float)[3:6]

      # Fallback 1: decelerate (brake toward zero velocity)
      u_decel_step = np.clip(-v_current / self.dt, u_min, u_max)
      u_decel = np.tile(u_decel_step, (horizon, 1))
      traj_decel, _ = self._predict_states(drone, u_decel)
      g_decel = constraints_fn(u_decel.flatten())
      if len(g_decel) == 0 or g_decel.min() >= -1e-6:
         _log.debug("  fallback: DECELERATE for %s", drone.drone_id)
         return u_decel, traj_decel, True

      # Fallback 2: zero control (hold — don't accelerate)
      u_zero = np.zeros((horizon, 3))
      traj_zero, _ = self._predict_states(drone, u_zero)
      g_zero = constraints_fn(u_zero.flatten())
      if len(g_zero) == 0 or g_zero.min() >= -1e-6:
         _log.debug("  fallback: HOLD for %s", drone.drone_id)
         return u_zero, traj_zero, True

      # All fallbacks failed — pick the least-violating option
      options = [
         (u_opt, traj_opt, constraints_fn(u_opt.flatten())),
         (u_decel, traj_decel, g_decel),
         (u_zero, traj_zero, g_zero),
      ]
      best_u, best_traj, _ = max(options, key=lambda o: o[2].min() if len(o[2]) else 0.0)
      _log.debug("  fallback: ALL FAILED for %s, using least-violating", drone.drone_id)
      return best_u, best_traj, False

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

   def _debug_log_feasibility_check(self, drone: Drone, neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray]], feasible: bool,
                                         result: OptimizeResult, traj_opt: np.ndarray, g: np.ndarray):
      _log.debug("SOLVE %s  feasible=%s  opt_success=%s  g_min=%.4f  cost=%.4f",
                 drone.drone_id, feasible, result.success, g.min() if len(g) else float("nan"), result.fun)
      for nid, (ntraj, nvel) in neighbor_trajectories.items():
         ntraj = np.asarray(ntraj, dtype=float).reshape((-1, 3))
         dists = np.linalg.norm(traj_opt - ntraj, axis=1)
         _log.debug("  neighbor=%s  dists=%s  safety_ego=%.2f  safety_nbr=%.2f  threshold=%.2f",
                    nid, np.round(dists, 3), drone.safety_zone, drone.safety_zone, 2 * drone.safety_zone)
