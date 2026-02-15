"""Threaded distributed MPC coordinator with asynchronous convergence.

Implements ThreadedMPCCoordinator: spawns AsyncLocalSolver thread per drone,
monitors global convergence, returns first-step controls when all drones'
trajectories stabilize.

Part of v3.0 threaded distributed MPC architecture (Phase 17).
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from drone_sim.domain.drone import Drone, has_central_cost
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.distributed.thread_lifecycle import ThreadLifecycleManager
from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph
from drone_sim.simulation.distributed.async_local_solver import AsyncLocalSolver
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


@register_coordinator("dmpc_threaded")
@dataclass
class ThreadedMPCCoordinator:
    """Asynchronous threaded distributed MPC coordinator.

    Spawns AsyncLocalSolver thread per drone, monitors global convergence,
    returns first-step controls when all drones' trajectories stabilize.

    Unlike DistributedMPCCoordinator (synchronous ADMM rounds), this coordinator
    enables truly asynchronous drone-to-drone negotiation via mailbox. Drones
    iterate until convergence (all trajectories stable), not for a fixed number
    of rounds.

    Coordinator role: lifecycle management + convergence detection, NOT
    orchestration. Drones negotiate independently via ThreadSafeMailbox.

    Stateless design: Creates fresh mailbox, lifecycle manager, and solvers
    per solve_controls() call. No persistent state between calls.
    """

    dt: float
    horizon: int = 5
    comm_radius: float | None = None
    convergence_threshold: float = 1e-3
    max_iterations: int = 100  # Per-drone iteration limit (safety cap)
    convergence_timeout_sec: float = 30.0
    gauss_seidel: bool = False  # Optional staggered spawn

    # Internal state (initialized in __post_init__)
    _neighbor_graph: NeighborGraph = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal neighbor graph."""
        self._neighbor_graph = NeighborGraph(comm_radius=self.comm_radius)

    def solve_controls(
        self,
        *,
        drones: list[Drone],
        obstacles: list[tuple[np.ndarray, float]],
        room_min: np.ndarray | None = None,
        room_max: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Solve for drone controls using asynchronous threaded DMPC.

        Pattern: Spawn threads -> Wait for convergence -> Shutdown -> Extract controls.

        Creates fresh ThreadSafeMailbox, ThreadLifecycleManager, and AsyncLocalSolver
        instances for each call (stateless coordinator).

        :param drones: List of Drone objects to optimize
        :param obstacles: Static obstacles as list of (center, radius) tuples
        :param room_min: Room lower bounds (3,) or None
        :param room_max: Room upper bounds (3,) or None
        :return: Dict mapping drone_id to first-step control (3,)
        """
        # Filter drones to optimize (must have central_cost interface)
        opt_drones = [d for d in drones if has_central_cost(d.controller)]

        if not opt_drones:
            return {}

        # 1. Update neighbor graph from current positions
        positions = {d.drone_id: np.asarray(d.x[:3], dtype=float) for d in drones}
        self._neighbor_graph.update(positions)

        # 2. Create shared mailbox for this solve
        mailbox = ThreadSafeMailbox()

        # 3. Create thread lifecycle manager
        manager = ThreadLifecycleManager()

        # 4. Create AsyncLocalSolver for each drone
        async_solvers: dict[str, AsyncLocalSolver] = {}

        for drone in opt_drones:
            local_solver = LocalMPCSolver(dt=self.dt, horizon=self.horizon)

            async_solver = AsyncLocalSolver(
                drone=drone,
                mailbox=mailbox,
                neighbor_graph=self._neighbor_graph,
                local_solver=local_solver,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
            )
            async_solvers[drone.drone_id] = async_solver

        # 5. Bootstrap mailbox with initial trajectories
        self._bootstrap_mailbox(mailbox, async_solvers)

        # 6. Spawn threads (optionally staggered for Gauss-Seidel)
        if self.gauss_seidel:
            self._spawn_staggered(manager, async_solvers, obstacles, room_min, room_max)
        else:
            self._spawn_parallel(manager, async_solvers, obstacles, room_min, room_max)

        # 7. Wait for convergence or timeout
        converged = self._wait_for_convergence(
            async_solvers,
            timeout_sec=self.convergence_timeout_sec
        )

        # 8. Shutdown threads
        manager.shutdown(timeout=5.0)

        if not converged:
            _log.warning(
                "ThreadedMPCCoordinator: Convergence timeout after %.2fs",
                self.convergence_timeout_sec
            )

        # 9. Extract first-step controls from final solutions
        controls: dict[str, np.ndarray] = {}
        for drone_id, solver in async_solvers.items():
            if solver.u_prev is not None:
                controls[drone_id] = solver.u_prev[0].copy()
            else:
                # Solver failed—return zero control
                _log.error(
                    "ThreadedMPCCoordinator: Solver %s failed, returning zero control",
                    drone_id
                )
                controls[drone_id] = np.zeros(3, dtype=float)

        return controls

    def _bootstrap_mailbox(
        self,
        mailbox: ThreadSafeMailbox,
        async_solvers: dict[str, AsyncLocalSolver],
    ) -> None:
        """Send initial trajectory messages so first iteration doesn't block.

        Pattern: Bootstrap with current position held constant over horizon.
        This prevents deadlock where all drones wait for each other's first message.

        :param mailbox: ThreadSafeMailbox to bootstrap
        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        """
        for drone_id, solver in async_solvers.items():
            drone = solver.drone
            pos = np.asarray(drone.x[:3], dtype=float)
            initial_traj = np.tile(pos, (self.horizon, 1))
            initial_vel = np.zeros((self.horizon, 3), dtype=float)

            message = TrajectoryMessage(
                drone_id=drone_id,
                trajectory=initial_traj,
                predicted_velocities=initial_vel,
                timestamp=time.monotonic(),
                safety_zone_radius=drone.safety_zone,
            )

            neighbor_ids = self._neighbor_graph.get_neighbors(drone_id)
            mailbox.broadcast(
                sender_id=drone_id,
                message=message,
                neighbor_ids=neighbor_ids,
            )

    def _spawn_parallel(
        self,
        manager: ThreadLifecycleManager,
        async_solvers: dict[str, AsyncLocalSolver],
        obstacles: list,
        room_min: np.ndarray | None,
        room_max: np.ndarray | None,
    ) -> None:
        """Spawn all solver threads simultaneously (Jacobi-style).

        :param manager: ThreadLifecycleManager for thread spawning
        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        :param obstacles: Static obstacles
        :param room_min: Room lower bounds
        :param room_max: Room upper bounds
        """
        for drone_id, solver in async_solvers.items():
            manager.spawn(
                thread_id=drone_id,
                target=solver.run,
                args=(obstacles, room_min, room_max),
            )

    def _spawn_staggered(
        self,
        manager: ThreadLifecycleManager,
        async_solvers: dict[str, AsyncLocalSolver],
        obstacles: list,
        room_min: np.ndarray | None,
        room_max: np.ndarray | None,
    ) -> None:
        """Spawn threads with staggered delays (Gauss-Seidel-style ordering).

        Pattern: Priority-based ordering—most constrained drones first.
        Stagger by 50ms to break symmetry and encourage earlier threads to
        publish before later threads start.

        This is a heuristic for symmetry breaking, NOT true Gauss-Seidel.
        Threading doesn't guarantee execution order.

        :param manager: ThreadLifecycleManager for thread spawning
        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        :param obstacles: Static obstacles
        :param room_min: Room lower bounds
        :param room_max: Room upper bounds
        """
        # Simple ordering: lexicographic by drone_id
        # Future: priority-based ordering like DistributedMPCCoordinator._compute_priority
        ordered_ids = sorted(async_solvers.keys())

        for i, drone_id in enumerate(ordered_ids):
            solver = async_solvers[drone_id]
            manager.spawn(
                thread_id=drone_id,
                target=solver.run,
                args=(obstacles, room_min, room_max),
            )
            # Stagger by 50ms to break symmetry
            if i < len(ordered_ids) - 1:
                time.sleep(0.05)

    def _wait_for_convergence(
        self,
        async_solvers: dict[str, AsyncLocalSolver],
        timeout_sec: float,
    ) -> bool:
        """Poll for global convergence (all drones' trajectories stable).

        Pattern: Periodic convergence check with timeout escape.

        Convergence criteria:
        1. All solvers have produced at least one solution (traj_prev not None)
        2. All solvers' trajectory change below threshold OR hit max iterations
        3. No timeout exceeded

        Polls every 50ms to check trajectory snapshots. Compares consecutive
        snapshots to detect when all trajectories stabilize.

        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        :param timeout_sec: Maximum time to wait for convergence
        :return: True if converged, False if timeout
        """
        start_time = time.monotonic()
        prev_trajectories: dict[str, np.ndarray] = {}

        while True:
            # Check timeout first
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                _log.warning(
                    "ThreadedMPCCoordinator: Convergence timeout after %.2fs",
                    timeout_sec
                )
                return False

            # Snapshot current trajectories
            current_trajectories = {
                drone_id: solver.traj_prev.copy()
                for drone_id, solver in async_solvers.items()
                if solver.traj_prev is not None
            }

            # Check if all solvers have produced a solution
            if len(current_trajectories) < len(async_solvers):
                # Some solvers still initializing
                time.sleep(0.05)
                continue

            # Compute trajectory changes
            if prev_trajectories:
                max_change = max(
                    float(np.linalg.norm(current_trajectories[did] - prev_trajectories[did]))
                    for did in current_trajectories.keys()
                )

                if max_change < self.convergence_threshold:
                    # Global convergence achieved
                    _log.info(
                        "ThreadedMPCCoordinator: Converged after %.2fs (max_change=%.6f)",
                        elapsed, max_change
                    )
                    return True

            # Update previous trajectories for next comparison
            prev_trajectories = current_trajectories

            # Sleep before next check
            time.sleep(0.05)
