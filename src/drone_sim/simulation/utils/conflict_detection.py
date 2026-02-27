"""Conflict detection and evasion for adaptive-zone drone pairs.

This module provides the ConflictDetector class, which is the shared foundation
used by both central and distributed coordinator subclasses.  It is standalone
with no coordinator dependencies so either can import it cleanly.

Algorithm overview
------------------
Each coordinator step, ``ConflictDetector.update()`` is called with the current
MPC predicted trajectories.  For every pair of adaptive-zone drones that are
also neighbours in the comm-radius graph the detector:

1. Computes per-step pairwise distances over the MPC horizon (TTC proxy).
2. Gates on *closing velocity* — drones that are moving apart cannot collide.
3. Applies hysteresis: ``HYSTERESIS_STEPS`` consecutive conflict steps are
   required before a drone enters the Evading state.
4. On Evading entry, computes a z-reflection evasion waypoint *once* and holds
   it until the conflict clears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph

from drone_sim.domain.drone import Drone

# ---------------------------------------------------------------------------
# Module-level tuning constants
# ---------------------------------------------------------------------------

TTC_THRESHOLD_FACTOR: float = 2.0
"""Conflict threshold = TTC_THRESHOLD_FACTOR * (safety_zone_i + safety_zone_j).

A predicted step is considered a near-miss when the inter-drone distance
drops below this product.
"""

HYSTERESIS_STEPS: int = 2
"""Number of consecutive steps in conflict before entering Evading state.

Guards against spurious detections from a single noisy trajectory frame.
"""

EVASION_OFFSET_DISTANCE: float = 1.5
"""Distance (metres) along the evasion direction to place the waypoint.

The waypoint is set once on Evading entry and held until the conflict
clears.
"""

VZ_ZERO_THRESHOLD: float = 0.05
"""|vz| below this value is treated as 'purely horizontal flight'.

In the degenerate case the evasion direction gets a synthetic vertical
component to ensure separation.
"""


# ---------------------------------------------------------------------------
# Per-drone conflict state
# ---------------------------------------------------------------------------

@dataclass
class DroneConflictState:
    """Mutable per-drone conflict tracking state."""

    consecutive_conflict_steps: int = 0
    """Number of consecutive steps the drone has been flagged as conflicting."""

    evading: bool = False
    """Whether the drone is currently in the Evading state."""

    evasion_waypoint: np.ndarray | None = None
    """Evasion waypoint (shape (3,)), set once on Evading entry, or None."""

    conflicting_neighbors: set[str] = field(default_factory=set)
    """Set of drone IDs that triggered the conflict on Evading entry."""


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class ConflictDetector:
    """Detects conflict between adaptive-zone drone pairs and computes
    evasion waypoints.

    Operates on per-drone state dicts, updated each coordinator step.
    Designed to be used by both central and distributed coordinator
    subclasses.

    Usage::

        detector = ConflictDetector()
        # each coordinator step:
        detector.update(drones, trajectories, neighbor_graph)
        for drone in drones:
            if detector.is_evading(drone.drone_id):
                wp = detector.get_evasion_waypoint(drone.drone_id)
                # inject wp into this drone's MPC reference
    """

    def __init__(self) -> None:
        self._state: dict[str, DroneConflictState] = {}

    # ------------------------------------------------------------------
    # Public update entry-point
    # ------------------------------------------------------------------

    def update(
        self,
        drones: list[Drone],
        trajectories: dict[str, np.ndarray],
        neighbor_graph: "NeighborGraph",
    ) -> None:
        """Update conflict state for all adaptive drone pairs.

        :param drones: All drones in the scene.  Only pairs where *both*
            drones have ``is_adaptive == True`` are processed.
        :param trajectories: Dict mapping ``drone_id`` to predicted position
            array of shape ``(H, 3)`` (from the MPC horizon).
        :param neighbor_graph: Current NeighborGraph — detection is restricted
            to comm-radius neighbours.
        """
        adaptive_ids: set[str] = {d.drone_id for d in drones if d.is_adaptive}
        drone_by_id: dict[str, Drone] = {d.drone_id: d for d in drones}

        # Determine which drones are "in conflict" this step
        in_conflict: dict[str, set[str]] = {did: set() for did in adaptive_ids}

        # Evaluate all adaptive-pair combinations restricted to neighbours
        for id_i in sorted(adaptive_ids):
            for id_j in sorted(adaptive_ids):
                if id_j <= id_i:
                    continue
                # Only check comm-radius neighbours
                if id_j not in neighbor_graph.get_neighbors(id_i):
                    continue

                drone_i = drone_by_id[id_i]
                drone_j = drone_by_id[id_j]
                traj_i = trajectories.get(id_i)
                traj_j = trajectories.get(id_j)

                if traj_i is None or traj_j is None:
                    continue

                conflict = self._evaluate_pair(drone_i, drone_j, traj_i, traj_j)
                if conflict:
                    in_conflict[id_i].add(id_j)
                    in_conflict[id_j].add(id_i)

        # Update per-drone state machine
        for did in adaptive_ids:
            state = self._get_or_create_state(did)
            drone = drone_by_id[did]
            conflicting = in_conflict.get(did, set())

            if conflicting:
                state.consecutive_conflict_steps += 1
                # Enter Evading state once hysteresis threshold is reached
                if (
                    state.consecutive_conflict_steps >= HYSTERESIS_STEPS
                    and not state.evading
                ):
                    state.evading = True
                    state.conflicting_neighbors = set(conflicting)
                    state.evasion_waypoint = self._compute_evasion_waypoint(
                        drone, conflicting, drone_by_id
                    )
            else:
                # Conflict cleared — exit Evading immediately, no hold time
                state.consecutive_conflict_steps = 0
                if state.evading:
                    state.evading = False
                    state.evasion_waypoint = None
                    state.conflicting_neighbors = set()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_state(self, drone_id: str) -> DroneConflictState:
        """Return the DroneConflictState for *drone_id*, creating a default if absent."""
        return self._get_or_create_state(drone_id)

    def is_evading(self, drone_id: str) -> bool:
        """Return True if the drone is currently in the Evading state."""
        return self._state.get(drone_id, DroneConflictState()).evading

    def get_evasion_waypoint(self, drone_id: str) -> np.ndarray | None:
        """Return the evasion waypoint (shape ``(3,)``) or ``None``."""
        state = self._state.get(drone_id)
        if state is None:
            return None
        return state.evasion_waypoint

    def reset(self, drone_id: str) -> None:
        """Remove the per-drone state entry (for cleanup when a drone is removed)."""
        self._state.pop(drone_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_state(self, drone_id: str) -> DroneConflictState:
        if drone_id not in self._state:
            self._state[drone_id] = DroneConflictState()
        return self._state[drone_id]

    def _evaluate_pair(
        self,
        drone_i: Drone,
        drone_j: Drone,
        traj_i: np.ndarray,
        traj_j: np.ndarray,
    ) -> bool:
        """Return True if the (i, j) pair is in conflict this step.

        Conflict requires BOTH:
        - At least one predicted step is closer than the TTC threshold, AND
        - The drones are currently approaching (closing velocity > 0).
        """
        # Per-step distances over the horizon, shape (H,)
        dists = np.linalg.norm(traj_i - traj_j, axis=1)

        # TTC threshold: factor × sum of safety zones
        threshold = TTC_THRESHOLD_FACTOR * (drone_i.safety_zone + drone_j.safety_zone)

        # TTC: first step index where predicted distance < threshold
        violation_indices = np.where(dists < threshold)[0]
        ttc = int(violation_indices[0]) if len(violation_indices) > 0 else None

        if ttc is None:
            return False

        # Closing velocity gate using current positions and velocities.
        # rel_pos points FROM i TO j; rel_vel is j's velocity relative to i.
        # When drones are approaching, j moves toward i → rel_vel is antiparallel
        # to rel_pos → dot product is NEGATIVE.  We flag conflict when drones
        # are closing (dot < 0) to avoid spurious evasion on diverging pairs.
        rel_pos = drone_j.position() - drone_i.position()  # points i → j
        rel_vel = drone_j.velocity() - drone_i.velocity()
        closing = float(
            np.dot(rel_pos, rel_vel) / (float(np.linalg.norm(rel_pos)) + 1e-9)
        )

        return closing < 0

    def _compute_evasion_waypoint(
        self,
        drone: Drone,
        conflicting_ids: set[str],
        all_drones: dict[str, Drone],
    ) -> np.ndarray:
        """Compute a z-reflection evasion waypoint for *drone*.

        The waypoint is placed ``EVASION_OFFSET_DISTANCE`` metres in the
        evasion direction from the drone's current position.

        z-reflection logic:
        - If |vz| >= VZ_ZERO_THRESHOLD: reflect vz → -vz (direction = [vx, vy, -vz])
        - Else (degenerate horizontal): synthesise a ±1 z component based on the
          sign of (vx + vy).  Positive or zero → +1, negative → -1.
        """
        vel = drone.velocity()  # shape (3,)
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])

        if abs(vz) >= VZ_ZERO_THRESHOLD:
            direction = np.array([vx, vy, -vz], dtype=float)
        else:
            deflect_z = 1.0 if (vx + vy) >= 0.0 else -1.0
            direction = np.array([vx, vy, deflect_z], dtype=float)

        norm = float(np.linalg.norm(direction)) + 1e-9
        waypoint: np.ndarray = (
            drone.position() + EVASION_OFFSET_DISTANCE * direction / norm
        )
        return waypoint
