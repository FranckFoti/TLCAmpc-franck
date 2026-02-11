from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from drone_sim.simulation.neighbor_graph import NeighborGraph


@dataclass
class ADMMState:
    """ADMM consensus state for distributed collision avoidance.

    Manages dual variables (lambdas) and auxiliary variables (z) for ADMM-based
    distributed MPC. The ADMM formulation decomposes collision avoidance constraints:

    Constraint: ||p_i(k) - p_j(k)|| >= d_ij for all timesteps k in horizon
    Reformulated: z_ij = p_i - p_j, ||z_ij|| >= d_ij
    Augmented Lagrangian: lambda_ij^T (p_i - p_j - z_ij) + (rho/2)||p_i - p_j - z_ij||^2
    """

    rho: float = 1.0  # Penalty parameter
    primal_tol: float = 1e-3  # Primal residual tolerance
    dual_tol: float = 1e-3  # Dual residual tolerance
    horizon: int = 5  # MPC horizon length

    _lambdas: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    _z: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    _prev_z: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def _canonicalize_pair(self, pair: tuple[str, str]) -> tuple[str, str]:
        """Canonicalize pair ordering (alphabetical)."""
        return tuple(sorted(pair))

    def initialize(self, neighbor_pairs: list[tuple[str, str]]) -> None:
        """Initialize dual and auxiliary variables for all neighbor pairs.

        :param neighbor_pairs: List of (drone_i, drone_j) pairs to track
        """
        self._lambdas.clear()
        self._z.clear()
        self._prev_z.clear()

        for pair in neighbor_pairs:
            canonical = self._canonicalize_pair(pair)
            self._lambdas[canonical] = np.zeros((self.horizon, 3), dtype=float)
            self._z[canonical] = np.zeros((self.horizon, 3), dtype=float)
            self._prev_z[canonical] = np.zeros((self.horizon, 3), dtype=float)

    def update_z(
        self,
        pair: tuple[str, str],
        traj_i: np.ndarray,
        traj_j: np.ndarray,
        min_dist: float,
    ) -> None:
        """Update auxiliary variable z_ij with projection.

        z-update: z_ij = proj( (p_i - p_j) + lambda_ij / rho )
        Projection enforces ||z_ij[k]|| >= min_dist at each timestep.

        :param pair: (drone_i, drone_j) pair
        :param traj_i: Trajectory of drone i, shape (H, 3)
        :param traj_j: Trajectory of drone j, shape (H, 3)
        :param min_dist: Minimum required distance
        """
        canonical = self._canonicalize_pair(pair)
        if canonical not in self._lambdas:
            return

        traj_i = np.asarray(traj_i, dtype=float).reshape((self.horizon, 3))
        traj_j = np.asarray(traj_j, dtype=float).reshape((self.horizon, 3))

        # Store previous z for dual residual computation
        self._prev_z[canonical] = self._z[canonical].copy()

        # Compute scaled dual variable
        lam = self._lambdas[canonical]

        # z = proj( (p_i - p_j) + lambda / rho )
        diff = traj_i - traj_j
        z_unproj = diff + lam / self.rho

        # Project to satisfy ||z|| >= min_dist at each timestep
        z_new = np.zeros_like(z_unproj)
        for k in range(self.horizon):
            norm = float(np.linalg.norm(z_unproj[k]))
            if norm < min_dist and norm > 1e-10:
                # Scale to min_dist in same direction
                z_new[k] = z_unproj[k] * (min_dist / norm)
            elif norm < 1e-10:
                # Zero difference - arbitrary direction
                z_new[k] = np.array([min_dist, 0.0, 0.0])
            else:
                # Already satisfies constraint
                z_new[k] = z_unproj[k]

        self._z[canonical] = z_new

    def update_lambda(
        self,
        pair: tuple[str, str],
        traj_i: np.ndarray,
        traj_j: np.ndarray,
    ) -> None:
        """Update dual variable lambda_ij.

        Dual update: lambda = lambda + rho * ((p_i - p_j) - z_ij)

        :param pair: (drone_i, drone_j) pair
        :param traj_i: Trajectory of drone i, shape (H, 3)
        :param traj_j: Trajectory of drone j, shape (H, 3)
        """
        canonical = self._canonicalize_pair(pair)
        if canonical not in self._lambdas:
            return

        traj_i = np.asarray(traj_i, dtype=float).reshape((self.horizon, 3))
        traj_j = np.asarray(traj_j, dtype=float).reshape((self.horizon, 3))

        diff = traj_i - traj_j
        z = self._z[canonical]

        # Dual update
        self._lambdas[canonical] = self._lambdas[canonical] + self.rho * (diff - z)

    def get_consensus_term(
        self,
        drone_id: str,
        trajectories: dict[str, np.ndarray],
        neighbor_graph: NeighborGraph,
    ) -> np.ndarray:
        """Get ADMM consensus gradient term for a drone's local cost.

        Returns the gradient contribution from ADMM augmented Lagrangian:
        Sum over neighbors j: lambda_ij + rho * (p_i - p_j - z_ij)
        Sign depends on whether drone_id is the first or second in the pair.

        :param drone_id: ID of the drone
        :param trajectories: Dict mapping drone_id to trajectory (H, 3)
        :param neighbor_graph: NeighborGraph for determining neighbors
        :return: Consensus term (H, 3) to add to local gradient
        """
        result = np.zeros((self.horizon, 3), dtype=float)
        neighbors = neighbor_graph.get_neighbors(drone_id)

        traj_i = np.asarray(trajectories.get(drone_id, np.zeros((self.horizon, 3))),
                           dtype=float).reshape((self.horizon, 3))

        for neighbor_id in neighbors:
            if neighbor_id not in trajectories:
                continue

            traj_j = np.asarray(trajectories[neighbor_id],
                               dtype=float).reshape((self.horizon, 3))

            canonical = self._canonicalize_pair((drone_id, neighbor_id))
            if canonical not in self._lambdas:
                continue

            lam = self._lambdas[canonical]
            z = self._z[canonical]

            # Compute residual
            diff = traj_i - traj_j
            residual = diff - z

            # Sign depends on position in canonical pair
            if drone_id == canonical[0]:
                # drone_id is i: gradient is +lambda + rho * residual
                result = result + lam + self.rho * residual
            else:
                # drone_id is j: gradient is -lambda - rho * residual
                result = result - lam - self.rho * residual

        return result

    def compute_residuals(
        self,
        trajectories: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        """Compute primal and dual residuals for convergence checking.

        Primal residual: max over pairs of ||p_i - p_j - z_ij||
        Dual residual: rho * max over pairs of ||z_ij - z_ij_prev||

        :param trajectories: Dict mapping drone_id to trajectory (H, 3)
        :return: (primal_residual, dual_residual)
        """
        primal_residuals = []
        dual_residuals = []

        for pair, z in self._z.items():
            id_i, id_j = pair

            if id_i not in trajectories or id_j not in trajectories:
                continue

            traj_i = np.asarray(trajectories[id_i],
                               dtype=float).reshape((self.horizon, 3))
            traj_j = np.asarray(trajectories[id_j],
                               dtype=float).reshape((self.horizon, 3))

            # Primal residual: ||p_i - p_j - z||
            diff = traj_i - traj_j
            primal_res = np.linalg.norm(diff - z)
            primal_residuals.append(float(primal_res))

            # Dual residual: rho * ||z - z_prev||
            z_prev = self._prev_z.get(pair, np.zeros_like(z))
            dual_res = self.rho * np.linalg.norm(z - z_prev)
            dual_residuals.append(float(dual_res))

        primal = max(primal_residuals) if primal_residuals else 0.0
        dual = max(dual_residuals) if dual_residuals else 0.0

        return primal, dual

    def is_converged(self, trajectories: dict[str, np.ndarray]) -> bool:
        """Check if ADMM has converged.

        Converged when both primal and dual residuals are below tolerances.

        :param trajectories: Dict mapping drone_id to trajectory (H, 3)
        :return: True if converged
        """
        primal, dual = self.compute_residuals(trajectories)
        return primal < self.primal_tol and dual < self.dual_tol

    def get_lambda(self, pair: tuple[str, str]) -> np.ndarray | None:
        """Get dual variable for a pair (for debugging/visualization).

        :param pair: (drone_i, drone_j) pair
        :return: Lambda array (H, 3) or None if pair not tracked
        """
        canonical = self._canonicalize_pair(pair)
        lam = self._lambdas.get(canonical)
        return lam.copy() if lam is not None else None

    def get_z(self, pair: tuple[str, str]) -> np.ndarray | None:
        """Get auxiliary variable for a pair (for debugging/visualization).

        :param pair: (drone_i, drone_j) pair
        :return: Z array (H, 3) or None if pair not tracked
        """
        canonical = self._canonicalize_pair(pair)
        z = self._z.get(canonical)
        return z.copy() if z is not None else None
