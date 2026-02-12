"""Tests for drone_sim.simulation.trajectory_exchange module.

Tests for TrajectoryMessage and TrajectoryMailbox classes.
"""

from __future__ import annotations

import numpy as np
import pytest
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage, TrajectoryMailbox
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph


class TestTrajectoryMessage:
    """Tests for TrajectoryMessage dataclass."""

    def test_message_creation(self):
        """Test TrajectoryMessage creation with required fields."""
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        msg = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=velocities,
            timestamp=42,
        )

        assert msg.drone_id == "drone-1"
        np.testing.assert_array_equal(msg.trajectory, trajectory)
        np.testing.assert_array_equal(msg.predicted_velocities, velocities)
        assert msg.timestamp == 42

    def test_message_creation_none_velocities(self):
        """Test TrajectoryMessage creation with None velocities."""
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        msg = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=42,
        )

        assert msg.predicted_velocities is None

    def test_message_field_access(self):
        """Test accessing all fields of TrajectoryMessage."""
        trajectory = np.zeros((5, 3))
        msg = TrajectoryMessage(
            drone_id="test-drone",
            trajectory=trajectory,
            predicted_velocities=np.zeros((5, 3)),
            timestamp=100,
        )

        # Verify all fields are accessible
        assert isinstance(msg.drone_id, str)
        assert isinstance(msg.trajectory, np.ndarray)
        assert isinstance(msg.predicted_velocities, np.ndarray)
        assert isinstance(msg.timestamp, int)


class TestTrajectoryMailbox:
    """Tests for TrajectoryMailbox class."""

    @pytest.fixture
    def neighbor_graph(self) -> NeighborGraph:
        """Create a NeighborGraph with known neighbor relationships."""
        graph = NeighborGraph(comm_radius=5.0)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([2.0, 0.0, 0.0]),  # Neighbor of drone-1
            "drone-3": np.array([3.0, 0.0, 0.0]),  # Neighbor of drone-1 and drone-2
            "drone-4": np.array([10.0, 0.0, 0.0]),  # Not a neighbor of drone-1
        }
        graph.update(positions)
        return graph

    @pytest.fixture
    def mailbox(self) -> TrajectoryMailbox:
        """Create an empty TrajectoryMailbox."""
        return TrajectoryMailbox()

    def test_broadcast_sends_to_correct_neighbors(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Test broadcast sends to correct neighbors."""
        trajectory = np.zeros((5, 3))

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        # drone-2 and drone-3 are neighbors of drone-1
        msgs_drone2 = mailbox.receive("drone-2")
        msgs_drone3 = mailbox.receive("drone-3")
        msgs_drone4 = mailbox.receive("drone-4")

        assert "drone-1" in msgs_drone2
        assert "drone-1" in msgs_drone3
        assert "drone-1" not in msgs_drone4  # Not a neighbor

    def test_receive_returns_correct_messages(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Test receive returns correct messages."""
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=velocities,
            timestamp=10,
            neighbor_graph=neighbor_graph,
        )

        msgs = mailbox.receive("drone-2")
        assert len(msgs) == 1
        assert "drone-1" in msgs

        msg = msgs["drone-1"]
        assert msg.drone_id == "drone-1"
        np.testing.assert_array_equal(msg.trajectory, trajectory)
        np.testing.assert_array_equal(msg.predicted_velocities, velocities)
        assert msg.timestamp == 10

    def test_receive_returns_empty_dict_for_unknown_drone(
        self, mailbox: TrajectoryMailbox
    ):
        """Test receive returns empty dict for unknown drone."""
        msgs = mailbox.receive("unknown-drone")
        assert msgs == {}
        assert isinstance(msgs, dict)

    def test_clear_removes_all_messages(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Test clear removes all messages from all inboxes."""
        trajectory = np.zeros((5, 3))

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        mailbox.broadcast(
            sender_id="drone-2",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        # Verify messages exist
        assert len(mailbox.receive("drone-2")) > 0
        assert len(mailbox.receive("drone-1")) > 0

        # Clear all
        mailbox.clear()

        # Verify all inboxes are empty
        assert mailbox.receive("drone-1") == {}
        assert mailbox.receive("drone-2") == {}
        assert mailbox.receive("drone-3") == {}

    def test_multiple_drones_broadcasting_simultaneously(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Test multiple drones can broadcast without interference."""
        trajectory1 = np.ones((5, 3))
        trajectory2 = np.ones((5, 3)) * 2

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory1,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        velocities2 = np.ones((2, 3)) * 0.5
        mailbox.broadcast(
            sender_id="drone-2",
            trajectory=trajectory2,
            predicted_velocities=velocities2,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        # drone-3 should receive from both drone-1 and drone-2
        msgs_drone3 = mailbox.receive("drone-3")
        assert "drone-1" in msgs_drone3
        assert "drone-2" in msgs_drone3

        # Verify trajectories are distinct
        np.testing.assert_array_equal(msgs_drone3["drone-1"].trajectory, trajectory1)
        np.testing.assert_array_equal(msgs_drone3["drone-2"].trajectory, trajectory2)

        # Verify predicted_velocities are distinct
        assert msgs_drone3["drone-1"].predicted_velocities is None
        np.testing.assert_array_equal(msgs_drone3["drone-2"].predicted_velocities, velocities2)

    def test_integration_broadcast_receive_cycle(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Integration test: full broadcast and receive cycle with NeighborGraph."""
        drone_ids = ["drone-1", "drone-2", "drone-3", "drone-4"]

        # All drones broadcast their trajectories
        for drone_id in drone_ids:
            trajectory = np.random.randn(5, 3)
            mailbox.broadcast(
                sender_id=drone_id,
                trajectory=trajectory,
                predicted_velocities=None,
                timestamp=0,
                neighbor_graph=neighbor_graph,
            )

        # Each drone receives messages from its neighbors
        for drone_id in drone_ids:
            neighbors = neighbor_graph.get_neighbors(drone_id)
            msgs = mailbox.receive(drone_id)

            # Should receive from all neighbors
            assert set(msgs.keys()) == neighbors

            # Each message should have correct sender
            for sender_id, msg in msgs.items():
                assert msg.drone_id == sender_id
                assert msg.trajectory.shape == (5, 3)

    def test_receive_returns_copy(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Test receive returns a copy, not the internal dict."""
        trajectory = np.zeros((5, 3))

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        msgs1 = mailbox.receive("drone-2")
        msgs2 = mailbox.receive("drone-2")

        # Should be equal but not the same object
        assert msgs1 == msgs2
        assert msgs1 is not msgs2

        # Modifying returned dict should not affect mailbox
        msgs1.clear()
        msgs3 = mailbox.receive("drone-2")
        assert "drone-1" in msgs3

    def test_broadcast_with_all_neighbors_radius(self):
        """Test broadcast when comm_radius=None (all drones are neighbors)."""
        graph = NeighborGraph(comm_radius=None)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([100.0, 0.0, 0.0]),  # Far away
            "drone-3": np.array([200.0, 0.0, 0.0]),  # Even farther
        }
        graph.update(positions)

        mailbox = TrajectoryMailbox()
        trajectory = np.zeros((5, 3))

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=graph,
        )

        # All drones should be neighbors when comm_radius=None
        assert "drone-1" in mailbox.receive("drone-2")
        assert "drone-1" in mailbox.receive("drone-3")

    def test_overwrite_previous_message_from_same_sender(
        self, mailbox: TrajectoryMailbox, neighbor_graph: NeighborGraph
    ):
        """Test that broadcasting again overwrites previous message."""
        trajectory1 = np.ones((5, 3))
        trajectory2 = np.ones((5, 3)) * 2

        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory1,
            predicted_velocities=None,
            timestamp=0,
            neighbor_graph=neighbor_graph,
        )

        velocities2 = np.ones((5, 3)) * 0.5
        mailbox.broadcast(
            sender_id="drone-1",
            trajectory=trajectory2,
            predicted_velocities=velocities2,
            timestamp=1,
            neighbor_graph=neighbor_graph,
        )

        msgs = mailbox.receive("drone-2")
        assert len(msgs) == 1  # Only one message from drone-1
        np.testing.assert_array_equal(msgs["drone-1"].trajectory, trajectory2)
        np.testing.assert_array_equal(msgs["drone-1"].predicted_velocities, velocities2)
        assert msgs["drone-1"].timestamp == 1
