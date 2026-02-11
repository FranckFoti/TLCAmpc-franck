from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.domain.constraints import (MovingObstacleAvoidanceConstraints, ObstacleAvoidanceConstraints, RoomConstraints, VelocityConstraints)
from drone_sim.domain.drone import Drone
from drone_sim.domain.registry import register_coordinator


def _has_central_cost(ctrl: object) -> bool:
   return all(hasattr(ctrl, name) for name in ("central_cost", "central_initial_guess", "horizon"))


@register_coordinator("mpc_central")
@dataclass
class CentralMPCGlobalCoordinator:
   """Central coordinator for mixed controllers.

   Optimizes only drones whose controller implements the central-cost interface.

   Safety constraints use owner-only rule:
       dist(owner, other) >= owner.safety_zone + other.safety_buffer
   """

   dt: float
   horizon: int = 5
   room_wall_tolerance: float = 0.0

   # Small lateral acceleration used only for warm-start / initial-guess symmetry breaking.
   # This helps SLSQP escape the "head-on, perfectly collinear" deadlock where the distance constraint gradient is zero in lateral directions at the
   # symmetric point.
   symmetry_break_accel: float = 0.05

   max_iter: int = 120
   f_tol: float = 1e-3

   def __post_init__(self) -> None:
      self._u_prev: dict[str, np.ndarray] = {}

   def _pack(self, u: np.ndarray) -> np.ndarray:
      return np.asarray(u, dtype=float).reshape(-1)

   def _unpack(self, u_flat: np.ndarray, num_drones: int) -> np.ndarray:
      return np.asarray(u_flat, dtype=float).reshape((num_drones, self.horizon, 3))

   def _predict_states(self, drones: list[Drone], u: np.ndarray) -> np.ndarray:
      new_states = np.zeros((len(drones), self.horizon, 6), dtype=float)
      for i, drone in enumerate(drones):
         x = np.asarray(drone.x, dtype=float).reshape(6)
         for k in range(self.horizon):
            x = drone.physics.step(x, u[i, k])
            new_states[i, k] = x
      return new_states

   def _predict_positions(self, drones: list[Drone], u: np.ndarray) -> np.ndarray:
      predicted_states = self._predict_states(drones, u)
      return predicted_states[:, :, :3]

   def _apply_symmetry_break(self, u0: np.ndarray) -> np.ndarray:
      """Apply a tiny deterministic perturbation to break perfect symmetry.

      Idea:
          To avoid the situation where all drones sit on z=0 and never try to escape, we nudge all three axes with a very small pattern that alternates per
          drone.
      """

      eps = float(self.symmetry_break_accel)
      if eps <= 0.0:
         return u0

      num_drones = u0.shape[0]
      if num_drones < 2:
         return u0

      u = np.asarray(u0, dtype=float).copy()

      # For each optimized drone i, add a tiny 3D bias vector whose sign alternates with i.
      # This ensures that even if the initial guess sits perfectly on z=0 (and symmetric in x/y), the optimizer sees a non-trivial search direction in all axes.
      # This base direction is randomized per call so that x, y, z components are drawn independently in [0.1, 1.0].
      # We normalize to keep the magnitude controlled and let `symmetry_break_accel` set the scale.
      base_vec = np.random.uniform(0.1, 1.0, size=3)
      base_vec /= np.linalg.norm(base_vec)  # unit-ish direction

      for i in range(num_drones):
         sign = 1.0 if (i % 2) == 0 else -1.0
         delta = sign * eps * base_vec  # shape (3,)
         # Broadcast over the horizon: same tiny bias on each step.
         u[i, :, :] = u[i, :, :] + delta[None, :]

      return u

   def solve_controls(self, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, float]], room_min: np.ndarray | None = None,
         room_max: np.ndarray | None = None) -> dict[str, np.ndarray]:

      from scipy.optimize import minimize

      # Extract values from Drone objects
      drone_ids = [d.drone_id for d in drones]
      controllers = [d.controller for d in drones]

      n = len(drones)
      idx_opt = [i for i in range(n) if _has_central_cost(controllers[i])]

      # If nothing to optimize, return empty.
      if not idx_opt:
         return {}

      opt_ids = [drone_ids[i] for i in idx_opt]
      num_optimized = len(idx_opt)

      # Per-optimized-drone bounds (from physics via Drone)
      bounds_list = [drones[i].bounds() for i in idx_opt]
      u_mins = np.stack([np.asarray(b[0], dtype=float).reshape(3) for b in bounds_list], axis=0)
      u_maxs = np.stack([np.asarray(b[1], dtype=float).reshape(3) for b in bounds_list], axis=0)

      def clip_u(u: np.ndarray) -> np.ndarray:
         return np.clip(u, u_mins[:, None, :], u_maxs[:, None, :])

      # Warm-start: shift previous solution if available
      u0 = np.zeros((num_optimized, self.horizon, 3), dtype=float)
      have_prev = all(did in self._u_prev for did in opt_ids)
      if have_prev:
         prev = np.stack([self._u_prev[did] for did in opt_ids], axis=0)
         u0 = np.concatenate([prev[:, 1:, :], prev[:, -1:, :]], axis=1)
         u0 = self._apply_symmetry_break(u0)
      else:
         # Build per-drone initial guesses and backtrack to feasibility.
         u_guess = []
         for j, i in enumerate(idx_opt):
            ug = controllers[i].central_initial_guess(drones[i])  # type: ignore[attr-defined]
            ug = np.asarray(ug, dtype=float).reshape((-1, 3))

            # Controllers may have their own configured horizon; the coordinator owns the optimization horizon. Trim/pad initial guesses accordingly.
            if ug.shape[0] >= self.horizon:
               ug = ug[: self.horizon]
            else:
               pad = np.repeat(ug[-1:, :], self.horizon - ug.shape[0], axis=0)
               ug = np.concatenate([ug, pad], axis=0)

            u_guess.append(ug)
         u_guess = np.stack(u_guess, axis=0)

         # Apply a tiny symmetry-breaking lateral component to the guess.
         u_guess = self._apply_symmetry_break(u_guess)

         # from alpha = 1 until 0.5^12
         alpha = 1.0
         for _ in range(12):
            u0 = clip_u(alpha * u_guess)
            if (self._constraints(self._pack(u0), drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max).min(initial=0.0) >= 0.0):
               break
            alpha *= 0.5
         else:
            u0 = np.zeros_like(u0)

      bounds = []
      for j in range(num_optimized):
         for _k in range(self.horizon):
            for axis in range(3):
               bounds.append((float(u_mins[j, axis]), float(u_maxs[j, axis])))

      cons = {"type": "ineq", "fun": lambda u_flat: self._constraints(u_flat, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)}

      opt_drones = [drones[i] for i in idx_opt]

      res = minimize(lambda u_flat: self._cost(u_flat, drones=opt_drones, controllers=[controllers[i] for i in idx_opt], clip_u=clip_u), self._pack(u0),
            method="SLSQP", bounds=bounds, constraints=[cons], options={"maxiter": int(self.max_iter), "ftol": float(self.f_tol), "disp": False})

      # Treat optimizer failures or strongly violated constraints as fatal instead of silently continuing with an invalid trajectory.
      # This ensures we do not "find" a route when the constraints (e.g. walls/obstacles) make the problem infeasible.
      if not res.success or not np.isfinite(res.fun):
         raise RuntimeError(f"CentralMPCGlobalCoordinator optimization failed: {res.message} (status={res.status})")

      g = self._constraints(res.x, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)

      min_margin = float(g.min(initial=np.inf)) if g.size else float("inf")
      if not np.isfinite(min_margin):
         raise RuntimeError("CentralMPCGlobalCoordinator produced non-finite constraint margins, treating this as an optimization failure.")

      # Allow a tiny numerical tolerance around zero. Anything clearly below zero means some safety/obstacle constraint is violated (e.g. going through a
      # wall or another drone).
      if min_margin < -1e-6:
         raise RuntimeError(f"CentralMPCGlobalCoordinator produced infeasible controls: min constraint margin {min_margin:.3e} < 0.")

      u_opt = clip_u(self._unpack(res.x, num_optimized))
      for did, u_seq in zip(opt_ids, u_opt, strict=True):
         self._u_prev[did] = u_seq

      return {did: u_opt[k, 0].copy() for k, did in enumerate(opt_ids)}

   def _cost(self, u_flat: np.ndarray, *, drones: list[Drone], controllers: list[object], clip_u) -> float:
      u = clip_u(self._unpack(u_flat, len(drones)))
      total = 0.0

      for i in range(len(drones)):
         total += float(controllers[i].central_cost(u[i], drones[i]))  # type: ignore[attr-defined]
      return float(total)

   def _constraints(self, u_flat: np.ndarray, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, float]], room_min: np.ndarray | None,
                    room_max: np.ndarray | None) -> np.ndarray:
      """Inequality constraints c(u) >= 0.

      Delegates to constraint classes for:
      - Drone-to-drone collision avoidance (with cons_stop)
      - Static obstacle avoidance
      - Room boundary constraints (with wall_tolerance)
      - Velocity magnitude constraints
      """
      num_drones = len(drones)
      u = self._unpack(u_flat, num_drones)
      predicted_states = self._predict_states(drones, u)
      predicted_positions = predicted_states[:, :, :3]
      predicted_velocities = predicted_states[:, :, 3:6]

      pred_pos = {drones[i].drone_id: predicted_positions[i] for i in range(num_drones)}
      vals = np.array([], dtype=float)

      # Drone-to-drone collision avoidance (i<j pairs, includes cons_stop)
      collision_c = MovingObstacleAvoidanceConstraints(horizon=self.horizon)
      vals = collision_c.evaluate_multi(drones, pred_pos, vals)

      # Static obstacle avoidance
      obstacle_c = ObstacleAvoidanceConstraints(horizon=self.horizon)
      vals = obstacle_c.evaluate_multi(drones, pred_pos, obstacles, vals)

      # Room boundary constraints (per-face, with wall_tolerance)
      if room_min is not None and room_max is not None:
         room_c = RoomConstraints(horizon=self.horizon, wall_tolerance=self.room_wall_tolerance)
         vals = room_c.evaluate_multi(drones, pred_pos, room_max, room_min, vals)

      # Velocity magnitude constraints
      velocity_c = VelocityConstraints(horizon=self.horizon)
      vals = velocity_c.evaluate_multi(drones, predicted_velocities, vals)

      return vals
