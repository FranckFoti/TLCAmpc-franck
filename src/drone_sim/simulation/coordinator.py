from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.domain.drone import Drone
from drone_sim.domain.registry import register_coordinator
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _has_central_cost(ctrl: object) -> bool:
   return all(hasattr(ctrl, name) for name in ("central_cost", "central_initial_guess", "horizon",))


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
   # This helps SLSQP escape the "head-on, perfectly collinear" deadlock where the distance constraint gradient is zero in lateral directions at the symmetric point.
   symmetry_break_accel: float = 0.05

   max_iter: int = 120
   f_tol: float = 1e-3

   def __post_init__(self) -> None:
      self._phys = LinearKinematicsPhysics(dt=self.dt)
      self._u_prev: dict[str, np.ndarray] = {}

   def _pack(self, u: np.ndarray) -> np.ndarray:
      return np.asarray(u, dtype=float).reshape(-1)

   def _unpack(self, u_flat: np.ndarray, m: int) -> np.ndarray:
      return np.asarray(u_flat, dtype=float).reshape((m, self.horizon, 3))

   #TODO drone und ohysics
   def _predict_states(self, xs0: np.ndarray, u: np.ndarray) -> np.ndarray:
      # xs0: (M,6), u: (M,H,3) => X: (M,H,6)
      A = self._phys.A
      B = self._phys.B
      M = xs0.shape[0]
      Xk = np.asarray(xs0, dtype=float).copy()
      X = np.zeros((M, self.horizon, 6), dtype=float)
      for k in range(self.horizon):
         for i in range(M):
            Xk[i] = A @ Xk[i] + B @ u[i, k]
            X[i, k] = Xk[i]
      return X

   def _predict_positions(self, xs0: np.ndarray, u: np.ndarray) -> np.ndarray:
      X = self._predict_states(xs0, u)
      return X[:, :, :3]

   def _apply_symmetry_break(self, u0: np.ndarray) -> np.ndarray:
      """Apply a tiny deterministic perturbation to break perfect symmetry.

      Idea:
          To avoid the situation where all drones sit on z=0 and never try to escape, we nudge all three axes with a very small pattern that alternates per drone.
      """

      eps = float(self.symmetry_break_accel)
      if eps <= 0.0:
         return u0

      M = u0.shape[0]
      if M < 2:
         return u0

      u = np.asarray(u0, dtype=float).copy()

      # For each optimized drone i, add a tiny 3D bias vector whose sign alternates with i.
      # This ensures that even if the initial guess sits perfectly on z=0 (and symmetric in x/y), the optimizer sees a non-trivial search direction in all axes.
      # This base direction is randomized per call so that x, y, z components are drawn independently in [0.1, 1.0].
      # We normalize to keep the magnitude controlled and let `symmetry_break_accel` set the scale.
      base_vec = np.random.uniform(0.1, 1.0, size=3)
      base_vec /= np.linalg.norm(base_vec)  # unit-ish direction

      for i in range(M):
         sign = 1.0 if (i % 2) == 0 else -1.0
         delta = sign * eps * base_vec  # shape (3,)
         # Broadcast over the horizon: same tiny bias on each step.
         u[i, :, :] = u[i, :, :] + delta[None, :]

      return u

   def solve_controls(
         self,
         *,
         drones: list[Drone],
         obstacles: list[tuple[np.ndarray, float]],
         room_min: np.ndarray | None = None,
         room_max: np.ndarray | None = None,
   ) -> dict[str, np.ndarray]:

      from scipy.optimize import minimize

      # Extract values from Drone objects
      drone_ids = [d.drone_id for d in drones]
      xs = [d.x for d in drones]
      controllers = [d.controller for d in drones]

      n = len(drones)
      idx_opt = [i for i in range(n) if _has_central_cost(controllers[i])]

      # If nothing to optimize, return empty.
      if not idx_opt:
         return {}

      safety_by_id = {d.drone_id: float(d.safety_zone) for d in drones}
      radii_by_id = {d.drone_id: float(d.radius) for d in drones}
      cons_stops_by_id = {d.drone_id: float(d.cons_stop) for d in drones}
      v_max_by_id = {d.drone_id: float(d.v_max) for d in drones}

      opt_ids = [drone_ids[i] for i in idx_opt]
      M = len(idx_opt)

      xs0 = np.stack([np.asarray(xs[i], dtype=float).reshape(6) for i in idx_opt], axis=0)

      # Per-optimized-drone bounds (from physics via Drone)
      bounds_list = [drones[i].bounds() for i in idx_opt]
      u_mins = np.stack([np.asarray(b[0], dtype=float).reshape(3) for b in bounds_list], axis=0)
      u_maxs = np.stack([np.asarray(b[1], dtype=float).reshape(3) for b in bounds_list], axis=0)

      def clip_u(u: np.ndarray) -> np.ndarray:
         return np.clip(u, u_mins[:, None, :], u_maxs[:, None, :])

      # Warm-start: shift previous solution if available
      u0 = np.zeros((M, self.horizon, 3), dtype=float)
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
            if (self._constraints(self._pack(u0), xs0=xs0, opt_ids=opt_ids, safety_by_id=safety_by_id,
                                  radii_by_id=radii_by_id, cons_stops_by_id=cons_stops_by_id, v_max_by_id=v_max_by_id,
                                  obstacles=obstacles, room_min=room_min, room_max=room_max).min(initial=0.0) >= 0.0):
               break
            alpha *= 0.5
         else:
            u0 = np.zeros_like(u0)

      bounds = []
      for j in range(M):
         for _k in range(self.horizon):
            for d in range(3):
               bounds.append((float(u_mins[j, d]), float(u_maxs[j, d])))

      cons = {"type": "ineq",
            "fun": lambda u_flat: self._constraints(u_flat, xs0=xs0, opt_ids=opt_ids, safety_by_id=safety_by_id,
                                                    radii_by_id=radii_by_id, cons_stops_by_id=cons_stops_by_id,
                                                    v_max_by_id=v_max_by_id, obstacles=obstacles,
                                                    room_min=room_min, room_max=room_max)}

      opt_drones = [drones[i] for i in idx_opt]

      res = minimize(
         lambda u_flat: self._cost(u_flat, xs0=xs0, opt_drones=opt_drones, controllers=[controllers[i] for i in idx_opt],
                                   clip_u=clip_u), self._pack(u0), method="SLSQP", bounds=bounds, constraints=[cons],
         options={"maxiter": int(self.max_iter), "ftol": float(self.f_tol), "disp": False})

      # Treat optimizer failures or strongly violated constraints as fatal instead of silently continuing with an invalid trajectory.
      # This ensures we do not "find" a route when the constraints (e.g. walls/obstacles) make the problem infeasible.
      if not res.success or not np.isfinite(res.fun):
         raise RuntimeError(f"CentralMPCGlobalCoordinator optimization failed: {res.message} (status={res.status})")

      g = self._constraints(res.x, xs0=xs0, opt_ids=opt_ids, safety_by_id=safety_by_id, radii_by_id=radii_by_id,
                            cons_stops_by_id=cons_stops_by_id, v_max_by_id=v_max_by_id,
                            obstacles=obstacles, room_min=room_min, room_max=room_max)

      min_margin = float(g.min(initial=np.inf)) if g.size else float("inf")
      if not np.isfinite(min_margin):
         raise RuntimeError(
            "CentralMPCGlobalCoordinator produced non-finite constraint margins, treating this as an optimization failure.")

      # Allow a tiny numerical tolerance around zero. Anything clearly below zero means some safety/obstacle constraint is violated (e.g. going through a wall or another drone).
      if min_margin < -1e-6:
         raise RuntimeError(
            f"CentralMPCGlobalCoordinator produced infeasible controls: min constraint margin {min_margin:.3e} < 0.")

      u_opt = clip_u(self._unpack(res.x, M))
      for did, u_seq in zip(opt_ids, u_opt, strict=True):
         self._u_prev[did] = u_seq

      return {did: u_opt[k, 0].copy() for k, did in enumerate(opt_ids)}

   def _cost(self, u_flat: np.ndarray, *, xs0: np.ndarray, opt_drones: list[Drone], controllers: list[object],
             clip_u) -> float:
      u = clip_u(self._unpack(u_flat, xs0.shape[0]))
      total = 0.0

      for i in range(xs0.shape[0]):
         total += float(controllers[i].central_cost(u[i], opt_drones[i]))  # type: ignore[attr-defined]
      return float(total)

   def _constraints(self, u_flat: np.ndarray, *, xs0: np.ndarray, opt_ids: list[str], safety_by_id: dict[str, float],
                    radii_by_id: dict[str, float], cons_stops_by_id: dict[str, float], v_max_by_id: dict[str, float],
                    obstacles: list[tuple[np.ndarray, float]],
                    room_min: np.ndarray | None, room_max: np.ndarray | None) -> np.ndarray:
      """Inequality constraints c(u) >= 0 using owner-only safety-zone rule.

          For each optimized drone A and any other object B:
              ||p_A - p_B|| >= A.safety_zone + B.safety_buffer

          Velocity constraint for each optimized drone:
              v_max^2 - ||vel||^2 >= 0
      """

      # Build predicted state/position/velocity for optimized drones
      M = xs0.shape[0]
      u = self._unpack(u_flat, M)
      X_opt = self._predict_states(xs0, u)
      P_opt = X_opt[:, :, :3]
      V_opt = X_opt[:, :, 3:6]  # Velocity components (vx, vy, vz)

      vals: list[float] = []

      # Optimized vs optimized (pairwise): add asymmetric constraints for both owners.
      for kk in range(self.horizon):
         for i in range(M):
            for j in range(i + 1, M):
               pi = P_opt[i, kk]
               pj = P_opt[j, kk]

               d = pi - pj
               dist = float(np.linalg.norm(d))

               id_i = opt_ids[i]
               id_j = opt_ids[j]
               thresh = float(safety_by_id[id_j] + safety_by_id[id_i] + cons_stops_by_id[id_i] + cons_stops_by_id[id_j])

               vals.append(dist - thresh)

      # Optimized vs obstacles
      self.observe_obstacles(M, P_opt, obstacles, opt_ids, safety_by_id, vals)

      # Room (wall) constraints: ensure each drone's physical sphere stays inside the axis-aligned room box if room bounds are provided.
      self.observe_no_flying_zone(M, P_opt, opt_ids, safety_by_id, room_max, room_min, vals)

      # Velocity magnitude constraints: v_max^2 - ||vel||^2 >= 0 for each drone at each horizon step.
      self.observe_velocity_limits(M, V_opt, opt_ids, v_max_by_id, vals)

      return np.asarray(vals, dtype=float)

   def observe_no_flying_zone(self, M, P_opt, opt_ids, safety_by_id, room_max, room_min, vals):
      # We allow a small penetration tolerance `room_wall_tolerance` by shifting the constraint margins: c_room = margin + room_wall_tolerance.
      # This means SLSQP enforces margin >= -room_wall_tolerance, while the simulator still clamps positions exactly in room boundary.
      # TODO, try to only use the self.room_wall_tolerance for the first step
      if room_min is not None and room_max is not None:
         r_min = np.asarray(room_min, dtype=float).reshape(3)
         r_max = np.asarray(room_max, dtype=float).reshape(3)

         for kk in range(self.horizon):
            for i in range(M):
               pi = P_opt[i, kk]
               r_i = float(safety_by_id[opt_ids[i]])
               # Lower bounds: p - r >= room_min  -> margin = p - r - room_min
               for d in range(3):
                  margin_lower = float(pi[d] - r_i - r_min[d])
                  vals.append(margin_lower + self.room_wall_tolerance)
               # Upper bounds: p + r <= room_max -> margin = room_max - (p + r)
               for d in range(3):
                  margin_upper = float(r_max[d] - (pi[d] + r_i))
                  vals.append(margin_upper + self.room_wall_tolerance)

   def observe_obstacles(self, M, P_opt, obstacles, opt_ids, safety_by_id, vals):
      for kk in range(self.horizon):
         for i in range(M):
            pi = P_opt[i, kk]
            for center, r in obstacles:
               c_arr = np.asarray(center, dtype=float).reshape(3)
               dist = float(np.linalg.norm(pi - c_arr))
               thresh = float(safety_by_id[opt_ids[i]] + float(r))
               vals.append(dist - thresh)

   def observe_velocity_limits(self, M, V_opt, opt_ids, v_max_by_id, vals):
      """Velocity magnitude constraints: v_max^2 - ||vel||^2 >= 0.

      Ensures each drone's velocity magnitude does not exceed its configured v_max
      at any point during the prediction horizon.
      """
      for kk in range(self.horizon):
         for i in range(M):
            vel = V_opt[i, kk]  # (vx, vy, vz)
            v_max = float(v_max_by_id[opt_ids[i]])
            # Constraint: v_max^2 - (vx^2 + vy^2 + vz^2) >= 0
            velocity_margin = v_max**2 - float(vel[0]**2 + vel[1]**2 + vel[2]**2)
            vals.append(velocity_margin)
