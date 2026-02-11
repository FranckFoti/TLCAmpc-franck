from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.drone import Drone
from drone_sim.domain.registry import register_coordinator
from drone_sim.physics.base import PhysicsModel
from drone_sim.simulation.admm_state import ADMMState
from drone_sim.simulation.local_mpc import LocalMPCSolver
from drone_sim.simulation.neighbor_graph import NeighborGraph
from drone_sim.simulation.trajectory_exchange import TrajectoryMailbox


def _has_central_cost(ctrl: object) -> bool:
   """Check if controller implements the central_cost interface."""
   return all(hasattr(ctrl, name) for name in ("central_cost", "central_initial_guess", "horizon"))


@register_coordinator("dmpc_admm")
@dataclass
class DistributedMPCCoordinator:
   """Distributed MPC coordinator using ADMM consensus.

   Orchestrates ADMM iterations to produce drone controls by coordinating
   local MPC solvers. Each drone solves its own optimization problem while
   exchanging trajectory information with neighbors to reach consensus on
   collision avoidance constraints.

   This coordinator can replace the central MPC for scalable distributed
   optimization.
   """

   dt: float
   horizon: int = 5
   rho: float = 1.0  # ADMM penalty parameter
   max_admm_iter: int = 50  # Max ADMM iterations per timestep
   primal_tol: float = 1e-3
   dual_tol: float = 1e-3
   comm_radius: float | None = None  # For NeighborGraph
   gauss_seidel: bool = True  # Use Gauss-Seidel updates for symmetry breaking

   # Internal state (initialized in __post_init__)
   _neighbor_graph: NeighborGraph = field(init=False)
   _mailbox: TrajectoryMailbox = field(init=False)
   _admm_state: ADMMState = field(init=False)
   _u_prev: dict[str, np.ndarray] = field(default_factory=dict)
   _last_iteration_count: int = field(default=0, init=False)
   _last_primal_residual: float = field(default=0.0, init=False)
   _last_dual_residual: float = field(default=0.0, init=False)
   _last_converged: bool = field(default=True, init=False)

   def __post_init__(self) -> None:
      self._neighbor_graph = NeighborGraph(comm_radius=self.comm_radius)
      self._mailbox = TrajectoryMailbox()
      self._admm_state = ADMMState(rho=self.rho, primal_tol=self.primal_tol, dual_tol=self.dual_tol, horizon=self.horizon, )

   def solve_controls(self, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, float]], room_min: np.ndarray | None = None,
         room_max: np.ndarray | None = None, ) -> dict[str, np.ndarray]:
      """Solve for drone controls using distributed ADMM optimization.
      Matches the CentralMPCGlobalCoordinator interface.

      Args:
          drones: List of Drone objects to optimize
          obstacles: List of (center, radius) static obstacles
          room_min: Room lower bounds (3,) or None
          room_max: Room upper bounds (3,) or None

      Returns:
          Dict mapping drone_id to control (3,) for first timestep
      """
      # Extract values from Drone objects
      drone_ids = [d.drone_id for d in drones]
      safety_zones = [d.safety_zone for d in drones]

      n = len(drones)

      # Identify which drones to optimize (must have central_cost interface)
      idx_opt = [i for i, drone in enumerate(drones) if _has_central_cost(drone.controller)]

      # If nothing to optimize, return empty
      if not idx_opt:
         return {}

      opt_ids = [drone_ids[i] for i in idx_opt]

      # Build lookup dicts
      idx_by_id = {drone_ids[i]: i for i in idx_opt}
      safety_by_id = {drone_ids[i]: float(safety_zones[i]) for i in range(n)}

      # 1. Update neighbor graph from current positions
      positions = {drone.drone_id: np.asarray(drone.x, dtype=float)[:3] for drone in drones}
      self._neighbor_graph.update(positions)

      # 2. Get neighbor pairs and initialize ADMMState
      neighbor_pairs = self._neighbor_graph.get_neighbor_pairs()
      self._admm_state.initialize(neighbor_pairs)

      trajectories, local_solvers = self.init_trajectories(drones)

      # Add static trajectories for non-optimized drones (needed for neighbor pairs)
      for i in range(n):
         if i not in idx_opt:
            pos = np.asarray(drones[i].x, dtype=float)[:3]
            trajectories[drone_ids[i]] = np.tile(pos, (self.horizon, 1))

      # 4. ADMM iteration loop
      converged = False
      iteration = 0
      timestep = 0  # For mailbox timestamps

      # Track controls across iterations for final output
      controls: dict[str, np.ndarray] = {}

      for iteration in range(self.max_admm_iter):
         # Determine drone solving order for this iteration
         drone_order = list(opt_ids)
         if self.gauss_seidel:
            if iteration < 3:
               # First few iterations: random shuffle for initial symmetry breaking
               random.shuffle(drone_order)
            else:
               # Later iterations: priority-based ordering (most constrained first)
               drone_order.sort(key=lambda d: self._compute_priority(d, trajectories, safety_by_id))

         # 4a. Broadcast current trajectories via mailbox (initial state)
         self._mailbox.clear()
         for drone_id in opt_ids:
            i = idx_by_id[drone_id]
            self._mailbox.broadcast(sender_id=drone_id, trajectory=trajectories[drone_id], safety_zone=float(safety_zones[i]), timestamp=timestep,
                  neighbor_graph=self._neighbor_graph, )

         # 4b-c. For each drone: receive neighbors, solve local MPC
         if self.gauss_seidel:
            # Gauss-Seidel: immediate updates after each drone solves
            for drone_id in drone_order:
               i = idx_by_id[drone_id]
               solver = local_solvers[drone_id]

               # Get neighbor trajectories from mailbox (includes any updates)
               messages = self._mailbox.receive(drone_id)
               neighbor_trajectories: dict[str, tuple[np.ndarray, float]] = {}
               for sender_id, msg in messages.items():
                  neighbor_trajectories[sender_id] = (msg.trajectory, msg.safety_zone)

               # Get warm-start from previous iteration
               u_prev = None
               if drone_id in self._u_prev and iteration == 0:
                  u_prev = np.concatenate([self._u_prev[drone_id][1:], self._u_prev[drone_id][-1:]], axis=0, )

               # Solve local MPC
               u_opt, traj_opt, success = solver.solve(drone=drones[i], neighbor_trajectories=neighbor_trajectories, obstacles=obstacles, room_min=room_min,
                     room_max=room_max, u_prev=u_prev, )

               # Immediate update (Gauss-Seidel style)
               trajectories[drone_id] = traj_opt
               controls[drone_id] = u_opt

               # Broadcast immediately so next drone sees updated trajectory
               self._mailbox.broadcast(sender_id=drone_id, trajectory=traj_opt, safety_zone=float(safety_zones[i]), timestamp=timestep,
                     neighbor_graph=self._neighbor_graph, )
         else:
            # Jacobi: all drones use stale data, update all at once
            trajectories, controls = self._jacobi(drone_order, drones, local_solvers, iteration, obstacles, room_min, room_max)

         # 4d-e. Update z and lambda for all neighbor pairs
         for pair in neighbor_pairs:
            id_i, id_j = pair
            traj_i = trajectories[id_i]
            traj_j = trajectories[id_j]
            min_dist = safety_by_id[id_i] + safety_by_id[id_j]

            self._admm_state.update_z(pair, traj_i, traj_j, min_dist)
            self._admm_state.update_lambda(pair, traj_i, traj_j)

         # 4f. Check convergence
         if self._admm_state.is_converged(trajectories):
            converged = True
            break

      # Store iteration count for debugging/testing
      self._last_iteration_count = iteration + 1

      # Record final residuals for debugging/visualization
      primal_res, dual_res = self._admm_state.compute_residuals(trajectories)
      self._last_primal_residual = primal_res
      self._last_dual_residual = dual_res
      self._last_converged = converged

      # Warn if not converged (but don't fail - use best-effort solution)
      if not converged:
         warnings.warn(f"DistributedMPCCoordinator did not converge after {self.max_admm_iter} "
                       f"iterations (primal/dual residuals may exceed tolerance)", RuntimeWarning, stacklevel=2, )

      # 5. Extract first-step controls from final trajectories
      # and 6. Store trajectories for warm-start
      result: dict[str, np.ndarray] = {}
      for drone_id in opt_ids:
         u_seq = controls[drone_id]
         self._u_prev[drone_id] = u_seq
         result[drone_id] = u_seq[0].copy()

      return result

   def _jacobi(self, drone_order: list[str], drones: list[Drone], local_solvers: dict[str, LocalMPCSolver], iteration: int, obstacles: list[tuple[np.ndarray, float]] | None = None,
                     room_min: np.ndarray | None = None, room_max: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
      # Jacobi: all drones use stale data, update all at once
      new_trajectories: dict[str, np.ndarray] = {}
      new_controls: dict[str, np.ndarray] = {}

      for drone in drones:
         if drone.drone_id not in drone_order:
            continue
         solver = local_solvers[drone.drone_id]

         # Get neighbor trajectories from mailbox
         messages = self._mailbox.receive(drone.drone_id)
         neighbor_trajectories_jac: dict[str, tuple[np.ndarray, float]] = {}
         for sender_id, msg in messages.items():
            neighbor_trajectories_jac[sender_id] = (msg.trajectory, msg.safety_zone,)

         # Get warm-start from previous iteration
         u_prev = None
         if drone.drone_id in self._u_prev and iteration == 0:
            u_prev = np.concatenate([self._u_prev[drone.drone_id][1:], self._u_prev[drone.drone_id][-1:]], axis=0, )

         # Solve local MPC
         u_opt, traj_opt, success = solver.solve(drone=drone, neighbor_trajectories=neighbor_trajectories_jac, obstacles=obstacles, room_min=room_min,
                                                 room_max=room_max, u_prev=u_prev)

         new_trajectories[drone.drone_id] = traj_opt
         new_controls[drone.drone_id] = u_opt

      # # Update all trajectories at once (Jacobi style)
      # trajectories = new_trajectories
      # controls = new_controls
      return new_trajectories, new_controls

   def init_trajectories(self, drones: list[Drone]) -> tuple[dict[str, np.ndarray], dict[str, LocalMPCSolver]]:
      # 3. Initialize trajectories (use warm-start if available)
      trajectories: dict[str, np.ndarray] = {}
      local_solvers: dict[str, LocalMPCSolver] = {}

      for drone_id, drone in enumerate(drones):
         controller = drone.controller
         if not _has_central_cost(controller):
            continue

         # Create local solver for this drone with bounds from physics
         local_solvers[drone.drone_id] = LocalMPCSolver(dt=self.dt, horizon=self.horizon)

         # Initialize trajectory from warm-start or initial guess
         if drone_id in self._u_prev:
            # Warm-start: shift previous solution
            u_prev = self._u_prev[drone.drone_id]
            u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
         else:
            # Get initial guess from controller
            u0 = controller.central_initial_guess(drone)
            u0 = np.asarray(u0, dtype=float).reshape((-1, 3))
            # Pad/trim to horizon
            if u0.shape[0] < self.horizon:
               pad = np.tile(u0[-1:], (self.horizon - u0.shape[0], 1))
               u0 = np.concatenate([u0, pad], axis=0)
            elif u0.shape[0] > self.horizon:
               u0 = u0[: self.horizon]

         # Predict trajectory from controls
         trajectories[drone.drone_id] = local_solvers[drone.drone_id]._predict_positions(drone, u0)
      return trajectories, local_solvers

   def get_last_iteration_count(self) -> int:
      """Get the number of ADMM iterations from the last solve.

      Useful for testing warm-start effectiveness.
      """
      return self._last_iteration_count

   def get_last_residuals(self) -> tuple[float, float]:
      """Get (primal_residual, dual_residual) from last solve."""
      return self._last_primal_residual, self._last_dual_residual

   def get_last_converged(self) -> bool:
      """Check if last solve converged."""
      return self._last_converged

   def get_neighbor_pairs(self) -> list[tuple[str, str]]:
      """Get current neighbor pairs for visualization."""
      return self._neighbor_graph.get_neighbor_pairs()

   def _compute_priority(self, drone_id: str, trajectories: dict[str, np.ndarray], safety_by_id: dict[str, float], ) -> float:
      """Compute priority score - lower = higher priority (solve first).

      Drones with smaller safety margins to neighbors get higher priority.
      This ensures drones in conflict zones solve first and commit to a
      direction, forcing others to adapt.

      Args:
          drone_id: ID of the drone
          trajectories: Current trajectories for all drones
          safety_by_id: Safety zones by drone ID

      Returns:
          Priority score (lower = higher priority = solve first)
      """
      neighbors = self._neighbor_graph.get_neighbors(drone_id)
      if not neighbors:
         return float("inf")  # No neighbors = lowest priority

      min_margin = float("inf")
      traj_i = trajectories.get(drone_id)
      if traj_i is None:
         return float("inf")

      for neighbor_id in neighbors:
         traj_j = trajectories.get(neighbor_id)
         if traj_j is None:
            continue
         min_dist = safety_by_id[drone_id] + safety_by_id[neighbor_id]
         for k in range(traj_i.shape[0]):
            dist = float(np.linalg.norm(traj_i[k] - traj_j[k]))
            margin = dist - min_dist
            min_margin = min(min_margin, margin)

      return min_margin  # Smaller margin = higher priority
