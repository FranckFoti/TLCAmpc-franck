"""Tests for drone_sim.domain.constraints module.

Tests for:
- VelocityConstraints (evaluate_single, evaluate_multi)
- MovingObstacleAvoidanceConstraints (evaluate_single, evaluate_multi)
- ObstacleAvoidanceConstraints (evaluate_single, evaluate_multi)
- RoomConstraints (evaluate_single, evaluate_multi -- box and sphere)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from drone_sim.domain.constraints import (
    VelocityConstraints,
    MovingObstacleAvoidanceConstraints,
    ObstacleAvoidanceConstraints,
    RoomConstraints,
)
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_drone(
    drone_id: str = "d1",
    x: np.ndarray | None = None,
    target: np.ndarray | None = None,
    radius: float = 0.2,
    safety_zone: float = 1.0,
    cons_stop: float = 0.0,
    v_max: float = 5.0,
) -> Drone:
    """Helper to create a minimal Drone for constraint testing."""

    class _StubController:
        pass

    if x is None:
        x = np.zeros(6, dtype=float)
    if target is None:
        target = np.zeros(3, dtype=float)

    return Drone(
        drone_id=drone_id,
        radius=radius,
        safety_zone=safety_zone,
        cons_stop=cons_stop,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=_StubController(),
        physics=LinearKinematicsPhysics(dt=0.1, v_max=v_max),
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
    )


# ---------------------------------------------------------------
# VelocityConstraints
# ---------------------------------------------------------------

class TestVelocityConstraintsSingle:
    """Tests for VelocityConstraints.evaluate_single."""

    def test_below_vmax_satisfied(self):
        """Velocity below v_max produces positive margins."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=5.0)
        # speed = sqrt(3) ~ 1.73 m/s, well below 5.0
        v_pred = np.ones((horizon, 3))
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    @pytest.mark.skip("seems to be invalid, the clipping is done in step, not in constraints")
    def test_above_vmax_clamped_to_zero(self):
        """Velocity above v_max produces zero (clamped by max(0, ...))."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=2.0)
        # speed = sqrt(27) ~ 5.2 m/s, above 2.0
        v_pred = np.ones((horizon, 3)) * 3.0
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result.shape == (horizon,)
        # max(0, negative) = 0
        assert np.all(result == 0.0)

    def test_zero_velocity_satisfied(self):
        """Zero velocity is always within limits."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=1.0)
        v_pred = np.zeros((horizon, 3))
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_appends_to_existing_values(self):
        """evaluate_single concatenates to existing values array."""
        horizon = 2
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=5.0)
        v_pred = np.zeros((horizon, 3))
        existing = np.array([42.0, 99.0])

        result = velocity_constraints.evaluate_single(drone, v_pred, existing)

        assert result.shape == (2 + horizon,)
        assert result[0] == 42.0
        assert result[1] == 99.0

    def test_exact_margin_value(self):
        """Verify the exact margin: v_max^2 - ||vel||^2."""
        horizon = 1
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=5.0)
        # vel = (3, 0, 0) => ||vel||^2 = 9, margin = 25 - 9 = 16
        v_pred = np.array([[3.0, 0.0, 0.0]])
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result[0] == pytest.approx(16.0)


class TestVelocityConstraintsMulti:
    """Tests for VelocityConstraints.evaluate_multi."""

    def test_multiple_drones_below_vmax(self):
        """All drones below v_max => all positive margins."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drones = [_make_drone("d1", v_max=5.0), _make_drone("d2", v_max=5.0)]
        v_pred = np.ones((2, horizon, 3))  # speed = sqrt(3) ~ 1.73
        values = np.array([])

        result = velocity_constraints.evaluate_multi(drones, v_pred, values)

        assert result.shape == (2 * horizon,)
        assert np.all(result > 0)

    @pytest.mark.skip("seems to be invalid, the clipping is done in step, not in constraints")
    def test_mixed_velocities(self):
        """One drone below, one above v_max."""
        horizon = 2
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drones = [_make_drone("d1", v_max=5.0), _make_drone("d2", v_max=1.0)]
        v_pred = np.ones((2, horizon, 3)) * 2.0  # speed = sqrt(12) ~ 3.46
        values = np.array([])

        result = velocity_constraints.evaluate_multi(drones, v_pred, values)

        assert result.shape == (2 * horizon,)
        # d1 (v_max=5): 25 - 12 = 13 > 0
        assert np.all(result[:horizon] > 0)
        # d2 (v_max=1): max(0, 1 - 12) = 0
        assert np.all(result[horizon:] == 0.0)

    def test_appends_to_existing_values(self):
        """evaluate_multi concatenates to existing values."""
        horizon = 1
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drones = [_make_drone(v_max=5.0)]
        v_pred = np.zeros((1, horizon, 3))
        existing = np.array([1.0, 2.0])

        result = velocity_constraints.evaluate_multi(drones, v_pred, existing)

        assert result.shape == (2 + 1,)
        assert result[0] == 1.0
        assert result[1] == 2.0


# ---------------------------------------------------------------
# MovingObstacleAvoidanceConstraints
# ---------------------------------------------------------------

class TestMovingObstacleAvoidanceSingle:
    """Tests for MovingObstacleAvoidanceConstraints.evaluate_single."""

    def test_far_neighbor_satisfied(self):
        """Distant neighbor produces positive constraint values."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.ones((horizon, 3)) * 10.0
        neighbors = {"n1": (neighbor_traj, 0.5)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_close_neighbor_violated(self):
        """Close neighbor produces negative constraint values."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        # Neighbor at (0.5, 0, 0), dist=0.5, threshold=1.0+1.0=2.0 => -1.5
        neighbor_traj = np.tile(np.array([0.5, 0.0, 0.0]), (horizon, 1))
        neighbors = {"n1": (neighbor_traj, 1.0)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon,)
        assert np.all(result < 0)

    def test_no_neighbors_empty(self):
        """No neighbors produces no constraint values."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone()
        pred_pos = np.zeros((horizon, 3))
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, {}, values)

        assert result.shape == (0,)

    def test_multiple_neighbors(self):
        """Multiple neighbors produce H * num_neighbors constraints."""
        horizon = 2
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.zeros((horizon, 3))
        neighbors = {
            "n1": (np.ones((horizon, 3)) * 10.0, 0.5),
            "n2": (np.ones((horizon, 3)) * 20.0, 0.5),
        }
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon * 2,)
        assert np.all(result > 0)

    def test_exact_margin_value(self):
        """Verify exact margin: dist - (safety_zone_self + safety_zone_neighbor)."""
        horizon = 1
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        # Neighbor at (5, 0, 0), dist=5, threshold=0.5+0.3=0.8
        neighbors = {"n1": (np.array([[5.0, 0.0, 0.0]]), 0.3)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result[0] == pytest.approx(4.2)


class TestMovingObstacleAvoidanceMulti:
    """Tests for MovingObstacleAvoidanceConstraints.evaluate_multi."""

    def test_two_drones_far_apart(self):
        """Two drones far apart produces all positive constraints."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)) * 10.0,
        }
        values = np.array([])

        result = moving_avoidance.evaluate_multi(drones, pred_pos, values)

        # 1 pair (i<j) * horizon constraints
        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_two_drones_overlapping(self):
        """Two overlapping drones produces negative constraints."""
        horizon = 2
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0)
        d2 = _make_drone("d2", safety_zone=1.0)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.tile(np.array([0.5, 0.0, 0.0]), (horizon, 1)),
        }
        values = np.array([])

        result = moving_avoidance.evaluate_multi(drones, pred_pos, values)

        assert np.all(result < 0)

    def test_get_neighbor_trajectories(self):
        """_get_neighbor_trajectories excludes self and includes others."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.8)
        d3 = _make_drone("d3", safety_zone=1.2)
        drones = [d1, d2, d3]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)),
            "d3": np.ones((horizon, 3)) * 2,
        }

        neighbors = moving_avoidance._get_neighbor_trajectories("d1", drones, pred_pos)

        assert "d1" not in neighbors
        assert "d2" in neighbors
        assert "d3" in neighbors
        assert_array_almost_equal(neighbors["d2"][0], pred_pos["d2"])
        assert neighbors["d2"][1] == 0.8
        assert neighbors["d3"][1] == 1.2


# ---------------------------------------------------------------
# ObstacleAvoidanceConstraints
# ---------------------------------------------------------------

class TestObstacleAvoidanceSingle:
    """Tests for ObstacleAvoidanceConstraints.evaluate_single."""

    def test_far_from_obstacle_satisfied(self):
        """Drone far from obstacle produces positive margins."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        obstacles = [(np.array([10.0, 0.0, 0.0]), 0.5)]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_close_to_obstacle_violated(self):
        """Drone close to obstacle produces negative margins."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        # Obstacle at (0.5, 0, 0) with r=0.2, dist=0.5, threshold=1.0+0.2=1.2 => -0.7
        obstacles = [(np.array([0.5, 0.0, 0.0]), 0.2)]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result.shape == (horizon,)
        assert np.all(result < 0)

    def test_no_obstacles_returns_zeros(self):
        """No obstacles produces a zero-length result from _evaluate, but shape=(horizon,)."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone()
        pred_pos = np.zeros((horizon, 3))
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, [], values)

        # _evaluate returns np.zeros(horizon) even with no obstacles
        assert result.shape == (horizon,)

    def test_exact_margin_value(self):
        """Verify exact margin: dist - (safety_zone + obstacle_radius)."""
        horizon = 1
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        # Obstacle at (3, 0, 0) with r=0.3, dist=3.0, threshold=0.5+0.3=0.8
        obstacles = [(np.array([3.0, 0.0, 0.0]), 0.3)]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result[0] == pytest.approx(2.2)

    def test_appends_to_existing_values(self):
        """evaluate_single concatenates to existing values."""
        horizon = 2
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.zeros((horizon, 3))
        obstacles = [(np.array([10.0, 0.0, 0.0]), 0.5)]
        existing = np.array([7.0])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, existing)

        assert result.shape == (1 + horizon,)
        assert result[0] == 7.0


class TestObstacleAvoidanceMulti:
    """Tests for ObstacleAvoidanceConstraints.evaluate_multi."""

    def test_two_drones_far_from_obstacle(self):
        """Both drones far from obstacle produces positive margins."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)) * 20.0,
        }
        obstacles = [(np.array([50.0, 50.0, 50.0]), 0.5)]
        values = np.array([])

        result = obstacle_avoidance.evaluate_multi(drones, pred_pos, obstacles, values)

        assert result.shape == (2 * horizon,)
        assert np.all(result > 0)

    def test_one_drone_close_to_obstacle(self):
        """One drone near obstacle has negative values."""
        horizon = 2
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0)
        d2 = _make_drone("d2", safety_zone=1.0)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),  # at origin
            "d2": np.ones((horizon, 3)) * 100.0,  # far away
        }
        obstacles = [(np.array([0.5, 0.0, 0.0]), 0.5)]  # close to d1
        values = np.array([])

        result = obstacle_avoidance.evaluate_multi(drones, pred_pos, obstacles, values)

        # d1 constraints should be violated
        assert np.any(result[:horizon] < 0)
        # d2 constraints should be satisfied
        assert np.all(result[horizon:] > 0)


# ---------------------------------------------------------------
# RoomConstraints -- Box mode
# ---------------------------------------------------------------

class TestRoomConstraintsSingleBox:
    """Tests for RoomConstraints.evaluate_single with box room."""

    def test_inside_room_satisfied(self):
        """Drone in center of room produces positive margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3)) + 5.0  # center of [0,10]^3
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        # 6 per-face constraints per horizon step
        assert result.shape == (6 * horizon,)
        assert np.all(result > 0)

    def test_outside_room_violated(self):
        """Drone outside room produces negative margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3)) - 5.0  # well outside [0,10]^3
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        assert result.shape == (6 * horizon,)
        assert np.any(result < 0)

    def test_near_wall_accounts_for_safety_zone(self):
        """Drone near wall: safety_zone pushes margin down."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        # Position at (0.5, 5, 5), room [0,10]^3, safety_zone=1.0
        # Lower margin x: 0.5 - 0.0 - 1.0 = -0.5 => violated
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.array([[0.5, 5.0, 5.0]])
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        # Per-face constraints: [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
        # lower_x = 0.5 - 1.0 - 0.0 = -0.5
        assert result.shape == (6,)
        assert result[0] == pytest.approx(-0.5)
        assert np.min(result) == pytest.approx(-0.5)

    def test_exact_margin_center(self):
        """Verify exact margin for drone at center of symmetric room."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        # Center of [-5, 5]^3, each wall is 5 units away, minus safety=1 => 4.0
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        room_min = np.array([-5.0, -5.0, -5.0])
        room_max = np.array([5.0, 5.0, 5.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        # All 6 faces have margin 4.0 at center of symmetric room
        assert result.shape == (6,)
        assert np.all(result == pytest.approx(4.0))

    def test_appends_to_existing_values(self):
        """evaluate_single concatenates to existing values."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[5.0, 5.0, 5.0]])
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        existing = np.array([42.0])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, existing)

        assert result.shape == (1 + 6,)  # existing + 6 per-face constraints
        assert result[0] == 42.0


class TestRoomConstraintsMultiBox:
    """Tests for RoomConstraints.evaluate_multi with box room."""

    def test_two_drones_inside(self):
        """Both drones inside room produces positive margins."""
        horizon = 2
        room_constraints = RoomConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)) + 5.0,
            "d2": np.zeros((horizon, 3)) + 5.0,
        }
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_multi(drones, pred_pos, room_max, room_min, values)

        # 2 drones * 6 per-face * horizon steps
        assert result.shape == (2 * 6 * horizon,)
        assert np.all(result > 0)


# ---------------------------------------------------------------
# RoomConstraints -- Sphere mode
# ---------------------------------------------------------------

class TestRoomConstraintsSingleSphere:
    """Tests for RoomConstraints.evaluate_single with spherical room."""

    def test_inside_sphere_satisfied(self):
        """Drone at origin in large sphere room produces positive margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max=10.0, room_min=0.0, values=values, room_is_sphere=True)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_outside_sphere_violated(self):
        """Drone far from origin in small sphere room produces negative margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3)) + 5.0  # dist from origin = sqrt(75) ~ 8.66
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max=2.0, room_min=0.0, values=values, room_is_sphere=True)

        assert np.all(result < 0)

    def test_exact_margin_sphere(self):
        """Verify exact margin: room_radius - dist_from_origin - safety_zone."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        # Drone at (3, 0, 0), dist=3.0, room_radius=10.0 => 10 - 3 - 0.5 = 6.5
        pred_pos = np.array([[3.0, 0.0, 0.0]])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max=10.0, room_min=0.0, values=values, room_is_sphere=True)

        assert result[0] == pytest.approx(6.5)


class TestRoomConstraintsMultiSphere:
    """Tests for RoomConstraints.evaluate_multi with spherical room."""

    def test_two_drones_inside_sphere(self):
        """Both drones inside sphere produces positive margins."""
        horizon = 2
        room_constraints = RoomConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)),
        }
        values = np.array([])

        result = room_constraints.evaluate_multi(drones, pred_pos, room_max=10.0, room_min=0.0, values=values, room_is_sphere=True)

        assert result.shape == (2 * horizon,)
        assert np.all(result > 0)


# ---------------------------------------------------------------
# MPCConstraints base class
# ---------------------------------------------------------------

class TestMPCConstraintsBase:
    """Tests for base class and label methods."""

    def test_velocity_label(self):
        assert VelocityConstraints(horizon=1).label() == "velocity"

    def test_moving_obstacle_label(self):
        assert MovingObstacleAvoidanceConstraints(horizon=1).label() == "moving_obstacle_avoidance"

    def test_obstacle_label(self):
        assert ObstacleAvoidanceConstraints(horizon=1).label() == "obstacle_avoidance"

    def test_room_label(self):
        assert RoomConstraints(horizon=1).label() == "room"
