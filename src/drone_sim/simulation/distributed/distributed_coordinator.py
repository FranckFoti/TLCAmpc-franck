from __future__ import annotations

import logging
import random
import warnings
from dataclasses import dataclass, field

import numpy as np

_log = logging.getLogger(__name__)

from drone_sim.domain.drone import Drone, has_central_cost
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.distributed.admm_state import ADMMState
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver, _pad_or_trim_horizon
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMailbox


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
   stagnation_limit: int = 3  # Break early if no drone makes progress for this many iterations

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

      :param drones: List of Drone objects to optimize
      :param obstacles: List of (center, radius) static obstacles
      :param room_min: Room lower bounds (3,) or None
      :param room_max: Room upper bounds (3,) or None
      :return: Dict mapping drone_id to control (3,) for first timestep
      """
      # Identify which drones to optimize (must have central_cost interface)
      opt_drones = [d for d in drones if has_central_cost(d.controller)]

      if not opt_drones:
         return {}

      # TODO: get rid of this all over list handlings. Use the drones list instead
      opt_ids = [d.drone_id for d in opt_drones]
      drone_by_id = {d.drone_id: d for d in opt_drones}
      all_drones_by_id = {d.drone_id: d for d in drones}

      # 1. Update neighbor graph from current positions
      positions = {d.drone_id: np.asarray(d.x, dtype=float)[:3] for d in drones}
      self._neighbor_graph.update(positions)

      # 2. Get neighbor pairs and initialize ADMMState
      neighbor_pairs = self._neighbor_graph.get_neighbor_pairs()
      self._admm_state.initialize(neighbor_pairs)

      trajectories, local_solvers = self.init_trajectories(drones)

      # Add static trajectories for non-optimized drones (needed for neighbor pairs)
      for drone in drones:
         if drone.drone_id not in drone_by_id:
            pos = np.asarray(drone.x, dtype=float)[:3]
            trajectories[drone.drone_id] = np.tile(pos, (self.horizon, 1))

      # 3. ADMM iteration loop
      converged = False
      stagnated = False

      # Track controls across iterations for final output
      controls: dict[str, np.ndarray] = {}

      # Track trajectory changes for stagnation detection
      prev_trajectories: dict[str, np.ndarray] = {did: traj.copy() for did, traj in trajectories.items()}
      stagnation_count = 0

      iteration = 0
      for iteration in range(self.max_admm_iter):
         # Determine drone solving order for this iteration
         drone_order = list(opt_ids)
         if self.gauss_seidel:
            if iteration < 3:
               # First few iterations: random shuffle for initial symmetry breaking
               random.shuffle(drone_order)
            else:
               # Later iterations: priority-based ordering (most constrained first)
               drone_order.sort(key=lambda d: self._compute_priority(d, trajectories, all_drones_by_id))

         # 3a. Broadcast current trajectories via mailbox (initial state)
         self._mailbox.clear()
         for drone_id in opt_ids:
            self._mailbox.broadcast(sender_id=drone_id, trajectory=trajectories[drone_id], predicted_velocities=None, timestamp=0, neighbor_graph=self._neighbor_graph)

         # 3b. For each drone: receive neighbors, solve local MPC
         if self.gauss_seidel:
            # Gauss-Seidel: immediate updates after each drone solves
            for drone_id in drone_order:
               drone = drone_by_id[drone_id]
               solver = local_solvers[drone_id]

               # Get neighbor trajectories from mailbox (includes any updates)
               messages = self._mailbox.receive(drone_id)
               neighbor_trajectories = {
                  sid: (msg.trajectory, msg.predicted_velocities)
                  for sid, msg in messages.items()
               }

               # Get warm-start from previous timestep (only on first ADMM iteration)
               u_prev = None
               if drone_id in self._u_prev and iteration == 0:
                  u_prev = self._u_prev[drone_id]

               u_opt, traj_opt, success = solver.solve(drone=drone, neighbor_trajectories=neighbor_trajectories, obstacles=obstacles, room_min=room_min,
                                                       room_max=room_max, u_prev=u_prev)

               # Compute predicted velocities for broadcast
               _, vel_opt = solver._predict_states(drone, u_opt)

               # Immediate update (Gauss-Seidel style)
               trajectories[drone_id] = traj_opt
               controls[drone_id] = u_opt

               # Broadcast immediately so next drone sees updated trajectory
               self._mailbox.broadcast(sender_id=drone_id, trajectory=traj_opt,
                                       predicted_velocities=vel_opt, timestamp=0,
                                       neighbor_graph=self._neighbor_graph)
         else:
            # Jacobi: all drones use stale data, update all at once
            trajectories, controls = self._jacobi(drone_order, drone_by_id, local_solvers, iteration, obstacles, room_min, room_max)

         # 3c. Stagnation detection — break early if no drone made progress
         stagnated, prev_trajectories = self.is_stagneted(trajectories, prev_trajectories, opt_ids, stagnation_count, iteration)
         if stagnated:
            break

         # 3d. Update z and lambda for all neighbor pairs
         self.update_z(trajectories, all_drones_by_id, neighbor_pairs)

         # 3e. Check convergence
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

      if _log.isEnabledFor(logging.DEBUG):
         self._debug_log_status(iteration, converged, stagnated, primal_res, dual_res, drones, trajectories, controls)

      # Warn if not converged (but don't fail - use best-effort solution)
      if not converged and not stagnated:
         warnings.warn(f"DistributedMPCCoordinator did not converge after {self.max_admm_iter} "
                       f"iterations (primal/dual residuals may exceed tolerance)", RuntimeWarning, stacklevel=2, )

      # 4. Extract first-step controls and store for warm-start
      result: dict[str, np.ndarray] = {}
      for drone_id in opt_ids:
         u_seq = controls[drone_id]
         self._u_prev[drone_id] = u_seq
         result[drone_id] = u_seq[0].copy()

      return result

   def _jacobi(self, drone_order: list[str], drone_by_id: dict[str, Drone], local_solvers: dict[str, LocalMPCSolver], iteration: int,
               obstacles: list[tuple[np.ndarray, float]] | None = None, room_min: np.ndarray | None = None,
               room_max: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
      """Jacobi update: all drones solve using stale neighbor data, then update all at once."""
      new_trajectories: dict[str, np.ndarray] = {}
      new_controls: dict[str, np.ndarray] = {}

      for drone_id in drone_order:
         drone = drone_by_id[drone_id]
         solver = local_solvers[drone_id]

         messages = self._mailbox.receive(drone_id)
         neighbor_trajectories = {
            sid: (msg.trajectory, msg.predicted_velocities)
            for sid, msg in messages.items()
         }

         # Get warm-start from previous timestep (only on first ADMM iteration)
         u_prev = None
         if drone_id in self._u_prev and iteration == 0:
            u_prev = self._u_prev[drone_id]

         u_opt, traj_opt, success = solver.solve(drone=drone, neighbor_trajectories=neighbor_trajectories, obstacles=obstacles, room_min=room_min,
                                                 room_max=room_max, u_prev=u_prev)

         new_trajectories[drone_id] = traj_opt
         new_controls[drone_id] = u_opt

      # Broadcast updated trajectories with velocities
      for drone_id in drone_order:
         drone = drone_by_id[drone_id]
         solver = local_solvers[drone_id]
         _, vel_opt = solver._predict_states(drone, new_controls[drone_id])
         self._mailbox.broadcast(sender_id=drone_id, trajectory=new_trajectories[drone_id],
                                 predicted_velocities=vel_opt, timestamp=0,
                                 neighbor_graph=self._neighbor_graph)

      return new_trajectories, new_controls

   def update_z(self, trajectories: dict[str, np.ndarray], all_drones_by_id: dict[str, Drone], neighbor_pairs: list[tuple[str, str]]) -> None:
      for pair in neighbor_pairs:
         id_i, id_j = pair
         traj_i = trajectories[id_i]
         traj_j = trajectories[id_j]

         # Get velocities from broadcast messages
         vel_i = self._get_velocity_from_messages(id_i, id_j)
         vel_j = self._get_velocity_from_messages(id_j, id_i)

         radii_i = self._compute_safety_radii(all_drones_by_id[id_i], vel_i)
         radii_j = self._compute_safety_radii(all_drones_by_id[id_j], vel_j)
         min_dist = radii_i + radii_j

         self._admm_state.update_z(pair, traj_i, traj_j, min_dist)
         self._admm_state.update_lambda(pair, traj_i, traj_j)

   def is_stagneted(self, trajectories: dict[str, np.ndarray], prev_trajectories: dict[str, np.ndarray], opt_ids: list[str], stagnation_count: int, iteration: int):
      max_change = max(float(np.linalg.norm(trajectories[did] - prev_trajectories[did])) for did in opt_ids)
      if max_change < self.primal_tol:
         stagnation_count += 1
      else:
         stagnation_count = 0
      prev_trajectories = {did: trajectories[did].copy() for did in opt_ids}

      if stagnation_count >= self.stagnation_limit:
         warnings.warn(f"ADMM stagnated after {iteration + 1} iterations — no drone made meaningful trajectory progress for {self.stagnation_limit} "
                       f"consecutive iterations", RuntimeWarning, stacklevel=2)
         return True, prev_trajectories
      return False, prev_trajectories

   def init_trajectories(self, drones: list[Drone]) -> tuple[dict[str, np.ndarray], dict[str, LocalMPCSolver]]:
      # Initialize trajectories (use warm-start if available)
      trajectories: dict[str, np.ndarray] = {}
      local_solvers: dict[str, LocalMPCSolver] = {}

      for drone in drones:
         controller = drone.controller
         if not has_central_cost(controller):
            continue

         # Create local solver for this drone with bounds from physics
         local_solvers[drone.drone_id] = LocalMPCSolver(dt=self.dt, horizon=self.horizon)

         # Initialize trajectory from warm-start or initial guess
         if drone.drone_id in self._u_prev:
            # Warm-start: shift previous solution
            u_prev = self._u_prev[drone.drone_id]
            u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
         else:
            # Get initial guess from controller
            u0 = _pad_or_trim_horizon(controller.central_initial_guess(drone), self.horizon)

         # Predict trajectory from controls
         trajectories[drone.drone_id] = local_solvers[drone.drone_id]._predict_states(drone, u0)[0]
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

   def _compute_priority(self, drone_id: str, trajectories: dict[str, np.ndarray], all_drones_by_id: dict[str, Drone]) -> float:
      """Compute priority score - lower = higher priority (solve first).

      Drones with smaller safety margins to neighbors get higher priority.
      This ensures drones in conflict zones solve first and commit to a
      direction, forcing others to adapt.

      :param drone_id: ID of the drone
      :param trajectories: Current trajectories for all drones
      :param all_drones_by_id: All drones by ID for safety radius computation
      :return: Priority score (lower = higher priority = solve first)
      """
      neighbors = self._neighbor_graph.get_neighbors(drone_id)
      if not neighbors:
         return float("inf")  # No neighbors = lowest priority

      min_margin = float("inf")
      traj_i = trajectories.get(drone_id)
      if traj_i is None:
         return float("inf")

      # Get drone_id's velocity from any neighbor's inbox
      any_neighbor = next(iter(neighbors))
      vel_i = self._get_velocity_from_messages(drone_id, any_neighbor)
      radii_i = self._compute_safety_radii(all_drones_by_id[drone_id], vel_i)

      for neighbor_id in neighbors:
         traj_j = trajectories.get(neighbor_id)
         if traj_j is None:
            continue
         vel_j = self._get_velocity_from_messages(neighbor_id, drone_id)
         radii_j = self._compute_safety_radii(all_drones_by_id[neighbor_id], vel_j)
         min_dist = float(np.mean(radii_i) + np.mean(radii_j))
         dists = np.linalg.norm(traj_i - traj_j, axis=1)
         min_margin = min(min_margin, float(np.min(dists)) - min_dist)

      return min_margin  # Smaller margin = higher priority

   def _compute_safety_radii(self, drone: Drone, velocities: np.ndarray | None) -> np.ndarray:
      """Compute per-step safety radii for a drone.

      :param drone: the drone
      :param velocities: predicted velocities (H, 3), or None
      :return: per-step safety radii (H,)
      """
      if velocities is not None and drone.is_adaptive:
         return np.array([drone.compute_adaptive_radius(velocities[step])
                          for step in range(self.horizon)])
      return np.full(self.horizon, drone.safety_zone)

   def _get_velocity_from_messages(self, sender_id: str, any_receiver_id: str) -> np.ndarray | None:
      """Get a drone's predicted velocities from its broadcast messages.

      :param sender_id: the drone whose velocities we want
      :param any_receiver_id: any neighbor of the sender to look up the message
      :return: predicted velocities (H, 3) or None
      """
      msgs = self._mailbox.receive(any_receiver_id)
      msg = msgs.get(sender_id)
      return msg.predicted_velocities if msg is not None else None

   def _debug_log_status(self, iteration: int, converged: bool, stagnated: bool, primal_res: float, dual_res: float,
                         drones: list[Drone], trajectories: dict[str, np.ndarray], controls: dict[str, np.ndarray]):
      status = "converged" if converged else ("stagnated" if stagnated else "max_iter")
      _log.debug("ADMM done: status=%s  iters=%d  primal=%.4e  dual=%.4e", status, iteration + 1, primal_res, dual_res)
      for i, drone_i in enumerate(drones[:-1]):
         for drone_j in drones[i+1:]:
            dists = np.linalg.norm(trajectories[drone_i.drone_id] - trajectories[drone_j.drone_id], axis=1)
            threshold = drone_i.safety_zone + drone_j.safety_zone
            _log.debug("  pair %s-%s  dists=%s  threshold=%.2f  min_dist=%.3f  violated=%s", drone_i.drone_id, drone_j.drone_id,
                       np.round(dists, 3), threshold, float(dists.min()), dists.min() < threshold)
      for drone in drones:
         u_seq = controls[drone.drone_id]
         _log.debug("  %s  u[0]=%s  traj=%s", drone.drone_id, np.round(u_seq[0], 3),
                    np.round(trajectories[drone.drone_id], 3).tolist())

