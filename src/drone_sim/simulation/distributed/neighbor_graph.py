from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class NeighborGraph:
    """Tracks which drones are within communication range of each other.

    Used by distributed MPC to determine which drones need to exchange
    trajectory information and coordinate collision constraints.
    """

    comm_radius: float | None = None
    _positions: dict[str, np.ndarray] = field(default_factory=dict)
    _neighbors: dict[str, set[str]] = field(default_factory=dict)

    def update(self, positions: dict[str, np.ndarray]) -> None:
        """Update neighbor graph based on current drone positions.

        :param positions: Dict mapping drone_id to position array (3,)
        """
        # Store positions
        self._positions = {k: np.asarray(v, dtype=float).reshape(3)
                          for k, v in positions.items()}

        # Compute neighbors
        drone_ids = list(self._positions.keys())
        self._neighbors = {did: set() for did in drone_ids}

        for i, id_i in enumerate(drone_ids):
            for j in range(i + 1, len(drone_ids)):
                id_j = drone_ids[j]
                if self._are_neighbors(id_i, id_j):
                    self._neighbors[id_i].add(id_j)
                    self._neighbors[id_j].add(id_i)

    def _are_neighbors(self, id_i: str, id_j: str) -> bool:
        """Check if two drones are within communication range."""
        if self.comm_radius is None:
            return True  # All drones are neighbors when no radius set

        p_i = self._positions[id_i]
        p_j = self._positions[id_j]
        dist = float(np.linalg.norm(p_i - p_j))
        return dist <= self.comm_radius

    def get_neighbors(self, drone_id: str) -> set[str]:
        """Get set of drone IDs within communication range of given drone."""
        return self._neighbors.get(drone_id, set()).copy()

    def get_neighbor_pairs(self) -> list[tuple[str, str]]:
        """Get list of all neighbor pairs (i, j) where i < j alphabetically.

        Useful for generating pairwise collision constraints.
        """
        pairs = [
            (id_i, id_j)
            for id_i, neighbors in self._neighbors.items()
            for id_j in neighbors
            if id_i < id_j
        ]
        return sorted(pairs)

