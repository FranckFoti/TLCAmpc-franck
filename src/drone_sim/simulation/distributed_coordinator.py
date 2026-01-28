from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.admm_state import ADMMState
from drone_sim.simulation.local_mpc import LocalMPCSolver
from drone_sim.simulation.neighbor_graph import NeighborGraph
from drone_sim.simulation.trajectory_exchange import TrajectoryMailbox


def _has_central_cost(ctrl: object) -> bool:
    """Check if controller implements the central_cost interface."""
    return all(
        hasattr(ctrl, name)
        for name in ("central_cost", "central_bounds", "central_initial_guess", "horizon")
    )


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

    # Internal state (initialized in __post_init__)
    _neighbor_graph: NeighborGraph = field(init=False)
    _mailbox: TrajectoryMailbox = field(init=False)
    _admm_state: ADMMState = field(init=False)
    _u_prev: dict[str, np.ndarray] = field(default_factory=dict)
    _last_iteration_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._neighbor_graph = NeighborGraph(comm_radius=self.comm_radius)
        self._mailbox = TrajectoryMailbox()
        self._admm_state = ADMMState(
            rho=self.rho,
            primal_tol=self.primal_tol,
            dual_tol=self.dual_tol,
            horizon=self.horizon,
        )

    def solve_controls(
        self,
        *,
        drone_ids: list[str],
        xs: list[np.ndarray],
        prefs: list[np.ndarray],
        radii: list[float],
        safety_zones: list[float],
        cons_stops: list[float],
        controllers: list[object],
        obstacles: list[tuple[np.ndarray, float]],
        external_predictions: dict[str, np.ndarray] | None = None,
        all_drone_state: dict[str, tuple[np.ndarray, np.ndarray, float]] | None = None,
        room_min: np.ndarray | None = None,
        room_max: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Solve for drone controls using distributed ADMM optimization.

        Matches the CentralMPCGlobalCoordinator interface.

        Args:
            drone_ids: List of drone IDs to optimize
            xs: List of current states (6,) for each drone [position, velocity]
            prefs: List of reference positions (3,) for each drone
            radii: List of physical radii for each drone
            safety_zones: List of safety zones for each drone
            cons_stops: List of conservative stop distances for each drone
            controllers: List of controller objects for each drone
            obstacles: List of (center, radius) static obstacles
            external_predictions: Optional trajectory overrides for external drones
            all_drone_state: State of all drones including non-optimized ones
            room_min: Room lower bounds (3,) or None
            room_max: Room upper bounds (3,) or None

        Returns:
            Dict mapping drone_id to control (3,) for first timestep
        """
        n = len(drone_ids)

        # Identify which drones to optimize (must have central_cost interface)
        idx_opt = [i for i in range(n) if _has_central_cost(controllers[i])]

        # If nothing to optimize, return empty
        if not idx_opt:
            return {}

        opt_ids = [drone_ids[i] for i in idx_opt]

        # Build lookup dicts
        idx_by_id = {drone_ids[i]: i for i in idx_opt}
        safety_by_id = {drone_ids[i]: float(safety_zones[i]) for i in range(n)}

        # 1. Update neighbor graph from current positions
        positions = {
            drone_ids[i]: np.asarray(xs[i], dtype=float)[:3] for i in idx_opt
        }
        self._neighbor_graph.update(positions)

        # 2. Get neighbor pairs and initialize ADMMState
        neighbor_pairs = self._neighbor_graph.get_neighbor_pairs()
        self._admm_state.initialize(neighbor_pairs)

        # 3. Initialize trajectories (use warm-start if available)
        trajectories: dict[str, np.ndarray] = {}
        local_solvers: dict[str, LocalMPCSolver] = {}

        for drone_id in opt_ids:
            i = idx_by_id[drone_id]
            x0 = np.asarray(xs[i], dtype=float).reshape(6)
            p_ref = np.asarray(prefs[i], dtype=float).reshape(3)
            controller = controllers[i]
            safety_zone = float(safety_zones[i])

            # Create local solver for this drone
            local_solvers[drone_id] = LocalMPCSolver(
                dt=self.dt,
                horizon=self.horizon,
                safety_zone=safety_zone,
            )

            # Initialize trajectory from warm-start or initial guess
            if drone_id in self._u_prev:
                # Warm-start: shift previous solution
                u_prev = self._u_prev[drone_id]
                u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
            else:
                # Get initial guess from controller
                u0 = controller.central_initial_guess(x0, p_ref)
                u0 = np.asarray(u0, dtype=float).reshape((-1, 3))
                # Pad/trim to horizon
                if u0.shape[0] < self.horizon:
                    pad = np.tile(u0[-1:], (self.horizon - u0.shape[0], 1))
                    u0 = np.concatenate([u0, pad], axis=0)
                elif u0.shape[0] > self.horizon:
                    u0 = u0[: self.horizon]

            # Predict trajectory from controls
            trajectories[drone_id] = self._predict_trajectory(x0, u0)

        # 4. ADMM iteration loop
        converged = False
        iteration = 0
        timestep = 0  # For mailbox timestamps

        for iteration in range(self.max_admm_iter):
            # 4a. Broadcast current trajectories via mailbox
            self._mailbox.clear()
            for drone_id in opt_ids:
                i = idx_by_id[drone_id]
                self._mailbox.broadcast(
                    sender_id=drone_id,
                    trajectory=trajectories[drone_id],
                    safety_zone=float(safety_zones[i]),
                    timestamp=timestep,
                    neighbor_graph=self._neighbor_graph,
                )

            # 4b-c. For each drone: receive neighbors, solve local MPC
            new_trajectories: dict[str, np.ndarray] = {}
            new_controls: dict[str, np.ndarray] = {}

            for drone_id in opt_ids:
                i = idx_by_id[drone_id]
                x0 = np.asarray(xs[i], dtype=float).reshape(6)
                p_ref = np.asarray(prefs[i], dtype=float).reshape(3)
                controller = controllers[i]
                solver = local_solvers[drone_id]

                # Get neighbor trajectories from mailbox
                messages = self._mailbox.receive(drone_id)
                neighbor_trajectories: dict[str, tuple[np.ndarray, float]] = {}
                for sender_id, msg in messages.items():
                    neighbor_trajectories[sender_id] = (msg.trajectory, msg.safety_zone)

                # Get warm-start from previous iteration
                u_prev = None
                if drone_id in self._u_prev and iteration == 0:
                    # Use shifted previous timestep solution for first iteration
                    u_prev = np.concatenate(
                        [self._u_prev[drone_id][1:], self._u_prev[drone_id][-1:]],
                        axis=0,
                    )

                # Solve local MPC with neighbor trajectories as moving obstacles
                u_opt, traj_opt, success = solver.solve(
                    x0=x0,
                    p_ref=p_ref,
                    controller=controller,
                    neighbor_trajectories=neighbor_trajectories,
                    obstacles=obstacles,
                    room_min=room_min,
                    room_max=room_max,
                    u_prev=u_prev,
                )

                new_trajectories[drone_id] = traj_opt
                new_controls[drone_id] = u_opt

            # Update trajectories for next iteration
            trajectories = new_trajectories

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

        # Warn if not converged (but don't fail - use best-effort solution)
        if not converged:
            warnings.warn(
                f"DistributedMPCCoordinator did not converge after {self.max_admm_iter} "
                f"iterations (primal/dual residuals may exceed tolerance)",
                RuntimeWarning,
                stacklevel=2,
            )

        # 5. Extract first-step controls from final trajectories
        # and 6. Store trajectories for warm-start
        result: dict[str, np.ndarray] = {}
        for drone_id in opt_ids:
            u_seq = new_controls[drone_id]
            self._u_prev[drone_id] = u_seq
            result[drone_id] = u_seq[0].copy()

        return result

    def _predict_trajectory(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict position trajectory from state and controls.

        Args:
            x0: Initial state (6,)
            u: Control sequence (H, 3)

        Returns:
            P: Position trajectory (H, 3)
        """
        from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics

        phys = LinearKinematicsPhysics(dt=self.dt)
        A = phys.A
        B = phys.B
        H = u.shape[0]

        P = np.zeros((H, 3), dtype=float)
        x = np.asarray(x0, dtype=float).reshape(6).copy()

        for k in range(H):
            x = A @ x + B @ u[k]
            P[k] = x[:3]

        return P

    def get_last_iteration_count(self) -> int:
        """Get the number of ADMM iterations from the last solve.

        Useful for testing warm-start effectiveness.
        """
        return self._last_iteration_count
