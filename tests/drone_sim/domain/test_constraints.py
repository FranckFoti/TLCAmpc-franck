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


class _StubController:
    pass


def _make_drone(
    drone_id: str = "d1",
    x: np.ndarray | None = None,
    target: np.ndarray | None = None,
    radius: float = 0.2,
    safety_zone: float = 1.0,
    cons_stop: float = 0.0,
    v_max: float = 5.0,
    alpha: float | None = None,
) -> Drone:
    """Helper to create a minimal Drone for constraint testing."""
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
        alpha=alpha,
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
        neighbors = {"n1": (neighbor_traj, None)}
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
        neighbors = {"n1": (neighbor_traj, None)}
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
            "n1": (np.ones((horizon, 3)) * 10.0, None),
            "n2": (np.ones((horizon, 3)) * 20.0, None),
        }
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon * 2,)
        assert np.all(result > 0)

    def test_exact_margin_value(self):
        """Verify exact margin: dist - (safety_zone_self + safety_zone_self) with same-type assumption."""
        horizon = 1
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        # Neighbor at (5, 0, 0), dist=5, threshold=0.5+0.5=1.0 (ego safety for both)
        neighbors = {"n1": (np.array([[5.0, 0.0, 0.0]]), None)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result[0] == pytest.approx(4.0)


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
# Adaptive Constraints (velocity-dependent safety radii)
# ---------------------------------------------------------------

class TestAdaptiveConstraints:
    """Tests for adaptive velocity-dependent safety radius in constraint evaluation.

    Hand-computed reference values:
    LinearKinematicsPhysics(dt=0.1) has default u_max=[3,3,3].
    u_max_scalar = min(|u_max|) = 3.0
    For alpha=0.5, radius=0.2:
      velocity=[0,0,0]: s_stop=0, adaptive_radius = 0.2
      velocity=[1,0,0]: ||v||^2=1, s_stop=1/6, adaptive_radius = 0.2 + 0.5*(1/6) = 0.2 + 1/12 ~ 0.2833
    """

    def test_moving_obstacle_adaptive_radius_at_rest(self):
        """Adaptive drone at rest: safety radius = drone.radius (smaller than fixed safety_zone).

        At rest the adaptive radius is just radius (0.2) which is smaller than
        the fixed safety_zone (1.0), so constraint margin should be LARGER.
        Neighbor velocity=None uses ego drone's safety_zone as fallback.
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)

        # Fixed drone: safety_zone=1.0
        drone_fixed = _make_drone(safety_zone=1.0)
        # Adaptive drone: alpha=0.5, radius=0.2, safety_zone=1.0
        drone_adaptive = _make_drone(safety_zone=1.0, alpha=0.5)

        pred_pos = np.array([[0.0, 0.0, 0.0]])
        neighbor_traj = np.array([[5.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, None)}  # neighbor vel=None

        # Fixed: dist=5.0, threshold = 1.0 + 1.0 = 2.0 (both use ego safety_zone), margin = 3.0
        result_fixed = constraints.evaluate_single(drone_fixed, pred_pos, neighbors, np.array([]))
        assert result_fixed[0] == pytest.approx(3.0)

        # Adaptive at rest: ego adaptive_radius = 0.2, neighbor vel=None => safety_zone=1.0
        # threshold = 0.2 + 1.0 = 1.2, margin = 3.8
        pred_vel_rest = np.array([[0.0, 0.0, 0.0]])
        result_adaptive = constraints.evaluate_single(
            drone_adaptive, pred_pos, neighbors, np.array([],), pred_vel=pred_vel_rest,
        )
        assert result_adaptive[0] == pytest.approx(3.8)

        # Adaptive margin should be larger (smaller ego safety radius at rest)
        assert result_adaptive[0] > result_fixed[0]

    def test_moving_obstacle_adaptive_radius_moving(self):
        """Adaptive drone with velocity: verify safety radius increases with speed.

        velocity=[1,0,0]: ||v||^2=1, s_stop=1/(2*3)=1/6
        adaptive_radius = 0.2 + 0.5 * (1/6) = 0.2 + 1/12 ~ 0.28333
        Neighbor vel=None => uses ego safety_zone=1.0
        threshold = adaptive_radius + 1.0 = 1.28333
        margin = 5.0 - 1.28333 = 3.71667
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)

        pred_pos = np.array([[0.0, 0.0, 0.0]])
        neighbor_traj = np.array([[5.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, None)}

        pred_vel = np.array([[1.0, 0.0, 0.0]])
        result = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]), pred_vel=pred_vel)

        expected_adaptive_radius = 0.2 + 0.5 * (1.0 / (2.0 * 3.0))  # ~ 0.28333
        expected_margin = 5.0 - (expected_adaptive_radius + 1.0)  # neighbor uses ego safety_zone
        assert result[0] == pytest.approx(expected_margin)

    def test_moving_obstacle_multi_adaptive_both_drones(self):
        """Both drones adaptive in evaluate_multi with pred_vel dict.

        d1 at origin, velocity=[1,0,0]: adaptive_radius_1 = 0.2 + 0.5*(1/6) ~ 0.28333
        d2 at (5,0,0), velocity=[2,0,0]: ||v||^2=4, s_stop=4/6=2/3
            adaptive_radius_2 = 0.2 + 0.5*(2/3) ~ 0.53333
        threshold = 0.28333 + 0.53333 + 0 + 0 = 0.81667
        dist = 5.0
        margin = 5.0 - 0.81667 = 4.18333
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0, alpha=0.5)
        d2 = _make_drone("d2", safety_zone=1.0, alpha=0.5)
        drones = [d1, d2]

        pred_pos = {
            "d1": np.array([[0.0, 0.0, 0.0]]),
            "d2": np.array([[5.0, 0.0, 0.0]]),
        }
        pred_vel = {
            "d1": np.array([[1.0, 0.0, 0.0]]),
            "d2": np.array([[2.0, 0.0, 0.0]]),
        }

        result = constraints.evaluate_multi(drones, pred_pos, np.array([]), pred_vel=pred_vel)

        ar_d1 = 0.2 + 0.5 * (1.0 / 6.0)
        ar_d2 = 0.2 + 0.5 * (4.0 / 6.0)
        expected_margin = 5.0 - (ar_d1 + ar_d2)
        assert result.shape == (horizon,)
        assert result[0] == pytest.approx(expected_margin)

    def test_moving_obstacle_multi_mixed_fixed_adaptive(self):
        """One fixed drone, one adaptive. Fixed uses safety_zone, adaptive uses compute_adaptive_radius.

        d1 fixed: safety = 1.0
        d2 adaptive, velocity=[1,0,0]: adaptive_radius = 0.2 + 0.5*(1/6) ~ 0.28333
        threshold = 1.0 + 0.28333 = 1.28333
        dist = 5.0
        margin = 5.0 - 1.28333 = 3.71667
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0)          # fixed (alpha=None)
        d2 = _make_drone("d2", safety_zone=1.0, alpha=0.5)  # adaptive
        drones = [d1, d2]

        pred_pos = {
            "d1": np.array([[0.0, 0.0, 0.0]]),
            "d2": np.array([[5.0, 0.0, 0.0]]),
        }
        pred_vel = {
            "d1": np.array([[1.0, 0.0, 0.0]]),  # ignored for fixed drone
            "d2": np.array([[1.0, 0.0, 0.0]]),
        }

        result = constraints.evaluate_multi(drones, pred_pos, np.array([]), pred_vel=pred_vel)

        ar_d2 = 0.2 + 0.5 * (1.0 / 6.0)
        expected_margin = 5.0 - (1.0 + ar_d2)  # d1 uses fixed safety_zone
        assert result[0] == pytest.approx(expected_margin)

    def test_obstacle_avoidance_adaptive(self):
        """Adaptive drone near static obstacle with velocity.

        velocity=[1,0,0]: adaptive_radius = 0.2 + 0.5*(1/6) ~ 0.28333
        obstacle at (3,0,0) with radius=0.3
        threshold = 0.28333 + 0.3 = 0.58333
        dist = 3.0
        margin = 3.0 - 0.58333 = 2.41667
        """
        horizon = 1
        constraints = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        obstacles = [(np.array([3.0, 0.0, 0.0]), 0.3)]
        pred_vel = np.array([[1.0, 0.0, 0.0]])

        result = constraints.evaluate_single(drone, pred_pos, obstacles, np.array([]), pred_vel=pred_vel)

        ar = 0.2 + 0.5 * (1.0 / 6.0)
        expected_margin = 3.0 - (ar + 0.3)
        assert result[0] == pytest.approx(expected_margin)

    def test_room_constraints_adaptive_box(self):
        """Adaptive drone in box room. Verify wall clearance uses adaptive radius.

        Drone at center of [-5,5]^3, velocity=[1,0,0]:
        adaptive_radius = 0.2 + 0.5*(1/6) ~ 0.28333
        Each wall margin = 5.0 - 0.28333 = 4.71667
        (compared to fixed: 5.0 - 1.0 = 4.0)
        """
        horizon = 1
        constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        room_min = np.array([-5.0, -5.0, -5.0])
        room_max = np.array([5.0, 5.0, 5.0])
        pred_vel = np.array([[1.0, 0.0, 0.0]])

        result = constraints.evaluate_single(
            drone, pred_pos, room_max, room_min, np.array([]),
            pred_vel=pred_vel,
        )

        ar = 0.2 + 0.5 * (1.0 / 6.0)
        expected_margin = 5.0 - ar
        assert result.shape == (6,)
        # All 6 faces should have the same margin (symmetric room, drone at center)
        for i in range(6):
            assert result[i] == pytest.approx(expected_margin)

    def test_backward_compat_no_pred_vel(self):
        """Pass pred_vel=None explicitly, verify identical result to omitting pred_vel entirely.

        Both should use fixed safety_zone since pred_vel is None.
        Neighbor vel=None also uses ego safety_zone.
        """
        horizon = 2
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5, alpha=0.5)  # adaptive, but no velocity given
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.ones((horizon, 3)) * 10.0
        neighbors = {"n1": (neighbor_traj, None)}

        # Without pred_vel (default None)
        result_default = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]))
        # With pred_vel=None explicitly
        result_explicit_none = constraints.evaluate_single(
            drone, pred_pos, neighbors, np.array([]), pred_vel=None,
        )

        assert_array_almost_equal(result_default, result_explicit_none)

        # Should use fixed safety_zone (0.5) for both ego and neighbor
        # dist = sqrt(3*100) ~ 17.32, threshold = 0.5 + 0.5 = 1.0, margin ~ 16.32
        expected_margin = np.linalg.norm(np.ones(3) * 10.0) - (0.5 + 0.5)
        for step in range(horizon):
            assert result_default[step] == pytest.approx(expected_margin)


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


# ---------------------------------------------------------------
# Per-step neighbor safety radii (ndarray support)
# ---------------------------------------------------------------

class TestPerStepNeighborVelocity:
    """Tests for per-step neighbor velocity in _evaluate."""

    def test_evaluate_single_per_step_neighbor_velocity(self):
        """Verify _evaluate uses per-step neighbor velocity to compute adaptive radii.

        Set up per-step neighbor velocities that vary across the horizon and verify
        the constraint margin changes accordingly. Ego drone is adaptive with alpha=0.5.

        horizon=3, drone at origin, neighbor at (5,0,0) at all steps.
        ego pred_vel=None => uses safety_zone=1.0.
        Neighbor velocities: [0,0,0], [1,0,0], [3,0,0] per step.
        Neighbor radii computed using ego drone's params (alpha=0.5, radius=0.2, u_max=3.0):
          step 0: vel=[0,0,0] => s_stop=0, r=0.2
          step 1: vel=[1,0,0] => ||v||^2=1, s_stop=1/6, r=0.2+0.5*(1/6) ~ 0.2833
          step 2: vel=[3,0,0] => ||v||^2=9, s_stop=9/6=1.5, r=0.2+0.5*1.5=0.95
        Expected margins:
          5.0 - (1.0 + 0.2) = 3.8,
          5.0 - (1.0 + 0.2833) ~ 3.7167,
          5.0 - (1.0 + 0.95) = 3.05
        """
        horizon = 3
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.tile(np.array([5.0, 0.0, 0.0]), (horizon, 1))
        neighbor_vel = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, neighbor_vel)}

        result = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]))

        assert result.shape == (horizon,)
        r0 = 0.2
        r1 = 0.2 + 0.5 * (1.0 / 6.0)
        r2 = 0.2 + 0.5 * (9.0 / 6.0)
        assert result[0] == pytest.approx(5.0 - (1.0 + r0))
        assert result[1] == pytest.approx(5.0 - (1.0 + r1))
        assert result[2] == pytest.approx(5.0 - (1.0 + r2))

    def test_evaluate_single_none_vs_zero_velocity(self):
        """None neighbor velocity (fixed fallback) vs zero velocity (adaptive at rest).

        For a fixed drone (alpha=None): None vel => safety_zone=0.5
        For an adaptive drone (alpha=0.5): zero vel => radius=0.2
        """
        horizon = 2
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.tile(np.array([5.0, 0.0, 0.0]), (horizon, 1))

        # Fixed drone: neighbor vel=None => neighbor uses ego safety_zone=0.5
        drone_fixed = _make_drone(safety_zone=0.5)
        neighbors_none = {"n1": (neighbor_traj, None)}
        result_none = constraints.evaluate_single(drone_fixed, pred_pos, neighbors_none, np.array([]))

        # Adaptive drone with neighbor zero vel:
        # ego pred_vel=None => ego uses safety_zone=0.5
        # neighbor zero vel => _safety_radius(ego_drone, [0,0,0]) => adaptive radius=0.2
        drone_adaptive = _make_drone(safety_zone=0.5, alpha=0.5)
        neighbors_zero = {"n1": (neighbor_traj, np.zeros((horizon, 3)))}
        result_zero = constraints.evaluate_single(drone_adaptive, pred_pos, neighbors_zero, np.array([]))

        # None => both sides use safety_zone(0.5), margin = 5.0 - 1.0 = 4.0
        assert result_none[0] == pytest.approx(4.0)
        # Zero vel => ego safety_zone(0.5) + neighbor adaptive(0.2) = 0.7, margin = 4.3
        assert result_zero[0] == pytest.approx(4.3)
