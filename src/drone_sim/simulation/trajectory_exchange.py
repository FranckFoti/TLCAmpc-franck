from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from drone_sim.simulation.neighbor_graph import NeighborGraph


@dataclass
class TrajectoryMessage:
    """Message containing a drone's predicted trajectory.

    Used for trajectory exchange between neighboring drones during
    ADMM iterations in distributed MPC.
    """

    drone_id: str
    trajectory: np.ndarray  # (H, 3) predicted positions
    safety_zone: float
    timestamp: int  # Simulation timestep when message was created


@dataclass
class TrajectoryMailbox:
    """Mailbox for storing and routing trajectory messages between drones.

    Provides in-memory message passing for trajectory exchange in
    distributed MPC simulation. Each drone has an inbox where neighbors
    can deposit trajectory messages.
    """

    _inbox: dict[str, dict[str, TrajectoryMessage]] = field(default_factory=dict)

    def broadcast(
        self,
        sender_id: str,
        trajectory: np.ndarray,
        safety_zone: float,
        timestamp: int,
        neighbor_graph: NeighborGraph,
    ) -> None:
        """Broadcast trajectory message to all neighbors.

        :param sender_id: ID of the sending drone
        :param trajectory: Predicted positions (H, 3)
        :param safety_zone: Sender's safety zone radius
        :param timestamp: Simulation timestep when message was created
        :param neighbor_graph: NeighborGraph for determining recipients
        """
        trajectory = np.asarray(trajectory, dtype=float)
        message = TrajectoryMessage(
            drone_id=sender_id,
            trajectory=trajectory,
            safety_zone=safety_zone,
            timestamp=timestamp,
        )

        # Send to all neighbors
        neighbors = neighbor_graph.get_neighbors(sender_id)
        for neighbor_id in neighbors:
            if neighbor_id not in self._inbox:
                self._inbox[neighbor_id] = {}
            self._inbox[neighbor_id][sender_id] = message

    def receive(self, receiver_id: str) -> dict[str, TrajectoryMessage]:
        """Receive all messages for a drone.

        :param receiver_id: ID of the receiving drone
        :return: Dict mapping sender_id to TrajectoryMessage. Returns empty dict if no messages.
        """
        return self._inbox.get(receiver_id, {}).copy()

    def clear(self) -> None:
        """Clear all messages from all inboxes.

        Called at start of each timestep to reset for new ADMM iteration.
        """
        self._inbox.clear()

    def clear_drone(self, drone_id: str) -> None:
        """Clear inbox for a specific drone.

        :param drone_id: ID of the drone whose inbox to clear
        """
        if drone_id in self._inbox:
            del self._inbox[drone_id]
