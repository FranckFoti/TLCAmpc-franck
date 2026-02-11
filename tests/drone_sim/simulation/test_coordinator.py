"""Tests for drone_sim.simulation.coordinator module.

Tests for:
- CentralMPCGlobalCoordinator._predict_states()
- CentralMPCGlobalCoordinator._constraints()
- CentralMPCGlobalCoordinator.solve_controls()
- CentralMPCGlobalCoordinator observer methods
"""

from __future__ import annotations

import pytest
import numpy as np

from drone_sim.domain.drone import Drone, Route
from drone_sim.simulation.coordinator import CentralMPCGlobalCoordinator
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_drone(
    drone_id: str,
    x: np.ndarray,
    target: np.ndarray,
    controller: object,
    radius: float = 0.2,
    safety_zone: float = 1.0,
    cons_stop: float = 0.0,
    v_max: float = 5.0,
    u_min: list[float] | None = None,
    u_max: list[float] | None = None,
    dt: float = 0.1,
) -> Drone:
    """Helper to create a Drone object for testing."""
    physics = LinearKinematicsPhysics(dt=dt, v_max=v_max, u_min=u_min, u_max=u_max)
    return Drone(
        drone_id=drone_id,
        radius=radius,
        safety_zone=safety_zone,
        cons_stop=cons_stop,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=controller,
        physics=physics,
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
    )


class TestCentralMPCGlobalCoordinatorInit:
   """Tests for CentralMPCGlobalCoordinator initialization."""

   def test_init_default_values(self):
      """Test coordinator initializes with correct default values."""
      coord = CentralMPCGlobalCoordinator(dt=0.1)
      assert coord.dt == 0.1
      assert coord.horizon == 5
      assert coord.room_wall_tolerance == 0.0
      assert coord.max_iter == 120
      assert coord.f_tol == 1e-3

   def test_init_custom_values(self):
      """Test coordinator initializes with custom values."""
      coord = CentralMPCGlobalCoordinator(dt=0.05, horizon=10, room_wall_tolerance=0.1, max_iter=200, f_tol=1e-4)
      assert coord.dt == 0.05
      assert coord.horizon == 10
      assert coord.room_wall_tolerance == 0.1
      assert coord.max_iter == 200
      assert coord.f_tol == 1e-4


class TestCentralMPCGlobalCoordinatorPredictStates:
   """Tests for CentralMPCGlobalCoordinator._predict_states method."""

   def test_predict_states_shape(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states returns correct shape."""
      M = 2  # Number of drones
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      d2 = _make_drone("d2", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u = np.zeros((M, H, 3))

      X = sample_coordinator._predict_states([d1, d2], u)

      assert X.shape == (M, H, 6)

   def test_predict_states_zero_control(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states with zero control maintains velocity trajectory."""
      M = 1
      H = sample_coordinator.horizon
      # Moving in x
      d1 = _make_drone("d1", np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u = np.zeros((M, H, 3))

      X = sample_coordinator._predict_states([d1], u)

      # Position should increase linearly with velocity
      dt = sample_coordinator.dt
      for k in range(H):
         expected_x = (k + 1) * dt * 1.0
         assert X[0, k, 0] == pytest.approx(expected_x, abs=1e-6)

   def test_predict_states_with_acceleration(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states with constant acceleration."""
      M = 1
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u = np.ones((M, H, 3)) * 1.0  # Constant acceleration

      X = sample_coordinator._predict_states([d1], u)

      # Velocity should increase with each step
      assert X[0, -1, 3] > 0  # Final x velocity positive

   def test_predict_states_multiple_drones(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states with multiple drones."""
      M = 3
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      d2 = _make_drone("d2", np.array([5.0, 0.0, 0.0, -1.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      d3 = _make_drone("d2", np.array([2.5, 5.0, 0.0, 0.0, -1.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u = np.zeros((M, H, 3))

      X = sample_coordinator._predict_states([d1, d2, d3], u)

      assert X.shape == (M, H, 6)
      assert X[0, -1, 0] > d1.x[0]
      assert X[1, -1, 0] < d2.x[0]
      assert X[2, -1, 1] < d3.x[1]


class TestCentralMPCGlobalCoordinatorConstraints:
   """Tests for CentralMPCGlobalCoordinator._constraints method."""

   def test_constraints_returns_array(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _constraints returns numpy array."""
      M = 2
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      d2 = _make_drone("d2", np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u_flat = np.zeros(M * H * 3)

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1, d2],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      assert isinstance(g, np.ndarray)
      assert len(g) > 0

   def test_constraints_satisfied_when_far_apart(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints are satisfied (>=0) when drones are far apart."""
      M = 2
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      d2 = _make_drone("d2", np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u_flat = np.zeros(M * H * 3)

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1, d2],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      assert np.all(g >= 0)

   def test_constraints_violated_when_overlapping(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints are violated (<0) when drones overlap."""
      M = 2
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      d2 = _make_drone("d2", np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u_flat = np.zeros(M * H * 3)

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1, d2],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      assert np.any(g < 0)

   def test_constraints_with_obstacles(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints account for obstacles."""
      M = 1
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u_flat = np.zeros(M * H * 3)
      obstacles = [(np.array([0.5, 0.0, 0.0]), 0.2)]

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1],
         obstacles=obstacles,
         room_min=None,
         room_max=None,
      )

      assert np.any(g < 0)

   def test_constraints_with_room_bounds(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints account for room bounds."""
      M = 1
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u_flat = np.zeros(M * H * 3)
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1],
         obstacles=[],
         room_min=room_min,
         room_max=room_max,
      )

      assert np.any(g < 0)

   def test_constraints_velocity_satisfied(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test velocity constraints are satisfied when velocity is below v_max."""
      M = 1
      H = sample_coordinator.horizon
      # Drone with velocity (1, 1, 1) has magnitude sqrt(3) ≈ 1.73 m/s, below v_max=5.0
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      u_flat = np.zeros(M * H * 3)  # Zero acceleration maintains velocity

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      # All constraints should be satisfied (>= 0)
      assert np.all(g >= 0)

   def test_constraints_velocity_clipped_by_physics(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test velocity constraints are satisfied even with high initial velocity (physics clips it)."""
      M = 1
      H = sample_coordinator.horizon
      # Drone with velocity (3, 3, 3) has magnitude sqrt(27) ≈ 5.2 m/s
      # With v_max=2.0, physics.step() clips velocity to 2.0, so constraint stays satisfied.
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0]), np.zeros(3),
                        CentralMPCAgent(dt=sample_coordinator.dt), v_max=2.0)
      u_flat = np.zeros(M * H * 3)

      g = sample_coordinator._constraints(
         u_flat,
         drones=[d1],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      # All constraints should be satisfied because physics.step() clips velocity
      assert np.all(g >= -1e-9)


class TestCentralMPCGlobalCoordinatorSolveControls:
   """Tests for CentralMPCGlobalCoordinator.solve_controls method."""

   def test_solve_controls_multiple_drones(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls with multiple drones."""
      controller1 = CentralMPCAgent(dt=sample_coordinator.dt, horizon=sample_coordinator.horizon)
      controller2 = CentralMPCAgent(dt=sample_coordinator.dt, horizon=sample_coordinator.horizon)

      result = sample_coordinator.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller1),
            _make_drone("d2", np.array([10.0, 10.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller2),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert isinstance(result, dict)
      assert "d1" in result
      assert result["d1"].shape == (3,)
      assert "d2" in result

   def test_solve_controls_raises_on_infeasible(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls raises RuntimeError when optimization is infeasible."""
      controller = CentralMPCAgent(dt=sample_coordinator.dt, horizon=sample_coordinator.horizon)

      with pytest.raises(RuntimeError, match="optimization failed|infeasible"):
         sample_coordinator.solve_controls(
            drones=[
               _make_drone("d1", np.array([-100.0, -100.0, -100.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
            ],
            obstacles=[],
            room_min=np.array([0.0, 0.0, 0.0]),
            room_max=np.array([1.0, 1.0, 1.0])
         )

   def test_solve_controls_respects_bounds(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls produces controls within bounds."""
      controller = CentralMPCAgent(
         dt=sample_coordinator.dt,
         horizon=sample_coordinator.horizon,
      )

      result = sample_coordinator.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0]), controller,
                        u_min=[-2.0, -2.0, -2.0], u_max=[2.0, 2.0, 2.0]),
         ],
         obstacles=[],
         room_min=np.array([-200.0, -200.0, 0.0]),
         room_max=np.array([200.0, 200.0, 200.0])
      )

      u = result["d1"]
      assert np.all(u >= -2.0 - 1e-6)
      assert np.all(u <= 2.0 + 1e-6)


class TestCentralMPCGlobalCoordinatorConstraints:
   """Tests that constraint classes are properly integrated in the coordinator."""

   def test_obstacles_constraint_via_class(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test obstacle constraints produce correct values via constraint classes."""
      from drone_sim.domain.constraints import ObstacleAvoidanceConstraints
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      pred_pos = np.zeros((H, 3))
      obstacles = [(np.array([5.0, 0.0, 0.0]), 0.5)]

      oac = ObstacleAvoidanceConstraints(horizon=H)
      result = oac.evaluate_single(d1, pred_pos, obstacles, np.array([]))

      assert len(result) == H
      assert all(v > 0 for v in result)

   def test_room_constraint_via_class(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test room constraints produce correct per-face values via constraint classes."""
      from drone_sim.domain.constraints import RoomConstraints
      H = sample_coordinator.horizon
      d1 = _make_drone("d1", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=sample_coordinator.dt))
      pred_pos = np.zeros((H, 3)) + 5.0
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])

      rc = RoomConstraints(horizon=H)
      result = rc.evaluate_single(d1, pred_pos, room_max, room_min, np.array([]))

      assert len(result) == H * 6
      assert all(v >= 0 for v in result)


class TestCentralMPCGlobalCoordinatorEdgeCases:
   """Edge case tests for CentralMPCGlobalCoordinator."""

   def test_single_drone_no_collision_constraints(self):
      """Test coordinator with single drone has no drone-drone constraints."""
      coord = CentralMPCGlobalCoordinator(dt=0.1, horizon=3)
      controller = CentralMPCAgent(dt=0.1, horizon=3)

      result = coord.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert "d1" in result

   def test_horizon_one(self):
      """Test coordinator with horizon=1."""
      coord = CentralMPCGlobalCoordinator(dt=0.1, horizon=1)
      controller = CentralMPCAgent(dt=0.1, horizon=1)

      result = coord.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert "d1" in result
      assert result["d1"].shape == (3,)
