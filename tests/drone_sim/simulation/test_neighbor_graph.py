import numpy as np
import pytest
from drone_sim.simulation.neighbor_graph import NeighborGraph


class TestNeighborGraph:
    """Tests for NeighborGraph class."""

    def test_all_neighbors_when_no_radius(self):
        """All drones are neighbors when comm_radius is None."""
        graph = NeighborGraph(comm_radius=None)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([100.0, 0.0, 0.0]),  # Far apart
            "drone-3": np.array([50.0, 50.0, 0.0]),
        }
        graph.update(positions)

        assert graph.get_neighbors("drone-1") == {"drone-2", "drone-3"}
        assert graph.get_neighbors("drone-2") == {"drone-1", "drone-3"}
        assert graph.get_neighbors("drone-3") == {"drone-1", "drone-2"}

    def test_limited_radius_excludes_far_drones(self):
        """Only drones within comm_radius are neighbors."""
        graph = NeighborGraph(comm_radius=5.0)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([3.0, 0.0, 0.0]),   # Within range
            "drone-3": np.array([10.0, 0.0, 0.0]),  # Outside range
        }
        graph.update(positions)

        assert graph.get_neighbors("drone-1") == {"drone-2"}
        assert graph.get_neighbors("drone-2") == {"drone-1"}
        assert graph.get_neighbors("drone-3") == set()  # No neighbors

    def test_boundary_distance(self):
        """Drones exactly at comm_radius are neighbors."""
        graph = NeighborGraph(comm_radius=5.0)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),  # Exactly at radius
        }
        graph.update(positions)

        assert "drone-2" in graph.get_neighbors("drone-1")

    def test_get_neighbor_pairs(self):
        """get_neighbor_pairs returns unique sorted pairs."""
        graph = NeighborGraph(comm_radius=None)
        positions = {
            "a": np.array([0.0, 0.0, 0.0]),
            "b": np.array([1.0, 0.0, 0.0]),
            "c": np.array([2.0, 0.0, 0.0]),
        }
        graph.update(positions)

        pairs = graph.get_neighbor_pairs()
        assert ("a", "b") in pairs
        assert ("a", "c") in pairs
        assert ("b", "c") in pairs
        assert len(pairs) == 3

    def test_update_changes_neighbors(self):
        """Neighbors update when positions change."""
        graph = NeighborGraph(comm_radius=5.0)

        # Initially far apart
        graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([10.0, 0.0, 0.0]),
        })
        assert graph.get_neighbors("drone-1") == set()

        # Move closer
        graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([3.0, 0.0, 0.0]),
        })
        assert graph.get_neighbors("drone-1") == {"drone-2"}

    def test_3d_distance(self):
        """Distance is computed correctly in 3D."""
        graph = NeighborGraph(comm_radius=5.0)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([3.0, 4.0, 0.0]),  # Distance = 5.0
        }
        graph.update(positions)

        assert "drone-2" in graph.get_neighbors("drone-1")

    def test_neighbor_count(self):
        """neighbor_count returns correct count."""
        graph = NeighborGraph(comm_radius=None)
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([1.0, 0.0, 0.0]),
            "drone-3": np.array([2.0, 0.0, 0.0]),
        }
        graph.update(positions)

        assert graph.neighbor_count("drone-1") == 2

    def test_get_all_drone_ids(self):
        """get_all_drone_ids returns all tracked drones."""
        graph = NeighborGraph()
        positions = {
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([1.0, 0.0, 0.0]),
        }
        graph.update(positions)

        ids = graph.get_all_drone_ids()
        assert set(ids) == {"drone-1", "drone-2"}

    def test_empty_graph(self):
        """Empty graph returns empty results."""
        graph = NeighborGraph()

        assert graph.get_neighbors("drone-1") == set()
        assert graph.get_neighbor_pairs() == []
        assert graph.get_all_drone_ids() == []
