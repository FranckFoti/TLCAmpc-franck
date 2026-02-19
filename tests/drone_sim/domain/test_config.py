"""Tests for drone_sim.domain.config module.

Tests Pydantic model validation for:
- PhysicsSpec, ControllerSpec
- DroneConfig
- ObstacleConfig, RoomConfig
- ScenarioConfig
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drone_sim.domain.config import (PhysicsSpec, ControllerSpec, DroneConfig, ObstacleConfig, RoomConfig, ScenarioConfig)


class TestPhysicsSpec:
   """Tests for PhysicsSpec model."""

   def test_physics_spec_valid(self):
      """Test PhysicsSpec creation with valid inputs."""
      spec = PhysicsSpec(type="linear_kinematics", params={"dt": 0.1})
      assert spec.type == "linear_kinematics"
      assert spec.params == {"dt": 0.1}

   def test_physics_spec_default_params(self):
      """Test PhysicsSpec uses empty dict for params by default."""
      spec = PhysicsSpec(type="custom_physics")
      assert spec.type == "custom_physics"
      assert spec.params == {}

   def test_physics_spec_missing_type_raises(self):
      """Test PhysicsSpec raises ValidationError when type is missing."""
      with pytest.raises(ValidationError, match="type"):
         PhysicsSpec()  # type: ignore[call-arg]


class TestControllerSpec:
   """Tests for ControllerSpec model."""

   def test_controller_spec_valid(self):
      """Test ControllerSpec creation with valid inputs."""
      spec = ControllerSpec(type="mpc_agent", params={"horizon": 10})
      assert spec.type == "mpc_agent"
      assert spec.params["horizon"] == 10

   def test_controller_spec_default_params(self):
      """Test ControllerSpec uses empty dict for params by default."""
      spec = ControllerSpec(type="pid")
      assert spec.params == {}

   def test_controller_spec_missing_type_raises(self):
      """Test ControllerSpec raises ValidationError when type is missing."""
      with pytest.raises(ValidationError, match="type"):
         ControllerSpec()  # type: ignore[call-arg]


class TestDroneConfig:
   """Tests for DroneConfig model."""

   def test_drone_config_valid_minimal(self):
      """Test DroneConfig with minimal required fields."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.drone_id == "d1"
      assert cfg.start == [0.0, 0.0, 0.0]
      assert cfg.target == [5.0, 5.0, 5.0]
      assert cfg.waypoints == []
      assert cfg.radius == 0.2  # default
      assert cfg.safety_zone == 1.0  # default

   def test_drone_config_valid_full(self):
      """Test DroneConfig with all fields specified."""
      cfg = DroneConfig(
         drone_id="drone-alpha",
         start=[1.0, 2.0, 3.0],
         waypoints=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
         target=[10.0, 10.0, 10.0],
         controller=ControllerSpec(type="mpc_agent"),
         radius=0.3,
         safety_zone=1.5,
         cons_stop=0.1,
         drone_color="red",
         safety_color="orange",
         trace_color="blue"
      )
      assert cfg.drone_id == "drone-alpha"
      assert len(cfg.waypoints) == 2
      assert cfg.radius == 0.3
      assert cfg.cons_stop == 0.1

   def test_drone_config_start_wrong_length_raises(self):
      """Test DroneConfig raises ValidationError for start with wrong length."""
      with pytest.raises(ValidationError, match="start"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0], target=[5.0, 5.0, 5.0])

   def test_drone_config_target_wrong_length_raises(self):
      """Test DroneConfig raises ValidationError for target with wrong length."""
      with pytest.raises(ValidationError, match="target"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0])

   def test_drone_config_missing_drone_id_raises(self):
      """Test DroneConfig raises ValidationError when drone_id is missing."""
      with pytest.raises(ValidationError, match="drone_id"):
         DroneConfig(start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])  # type: ignore[call-arg]

   def test_drone_config_color_as_rgb_list(self):
      """Test DroneConfig accepts RGB list for colors."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], drone_color=[0.5, 0.2, 0.8])
      assert cfg.drone_color == [0.5, 0.2, 0.8]

   def test_drone_config_edge_case_zero_radius(self):
      """Test DroneConfig allows zero radius (edge case)."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], radius=0.0)
      assert cfg.radius == 0.0

   def test_drone_config_edge_case_empty_waypoints(self):
      """Test DroneConfig with explicit empty waypoints."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], waypoints=[], target=[5.0, 5.0, 5.0])
      assert cfg.waypoints == []

   def test_drone_config_default_alpha_is_none(self):
      """Test default config has alpha=None (fixed mode)."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.alpha is None

   def test_drone_config_with_alpha(self):
      """Test setting alpha makes config adaptive."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], alpha=0.5)
      assert cfg.alpha == 0.5

   def test_drone_config_alpha_must_be_positive(self):
      """Test alpha=0 or alpha=-1 raises validation error."""
      with pytest.raises(ValidationError, match="alpha must be positive"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], alpha=0.0)
      with pytest.raises(ValidationError, match="alpha must be positive"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], alpha=-1.0)

   def test_drone_config_existing_configs_unchanged(self):
      """Test existing config without alpha works exactly as before."""
      cfg = DroneConfig(
         drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
         radius=0.3, safety_zone=1.5, cons_stop=0.1,
      )
      assert cfg.alpha is None
      assert cfg.radius == 0.3
      assert cfg.safety_zone == 1.5
      assert cfg.cons_stop == 0.1


class TestDroneConfigSafetyZoneMode:
   """Tests for DroneConfig.safety_zone_mode field (Phase 23)."""

   def test_default_is_fixed(self):
      """Default safety_zone_mode is 'fixed' — backward compatible."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.safety_zone_mode == "fixed"

   def test_accepts_adaptive_mode(self):
      """safety_zone_mode='adaptive' is valid."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                        safety_zone_mode="adaptive")
      assert cfg.safety_zone_mode == "adaptive"

   def test_accepts_lstm_mode(self):
      """safety_zone_mode='lstm' is valid."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                        safety_zone_mode="lstm")
      assert cfg.safety_zone_mode == "lstm"

   def test_invalid_mode_raises(self):
      """Unknown safety_zone_mode raises ValidationError."""
      with pytest.raises(ValidationError):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                     safety_zone_mode="unknown_mode")

   def test_existing_configs_unaffected(self):
      """Existing DroneConfig without safety_zone_mode still works (backward compat)."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                        radius=0.3, safety_zone=1.5, cons_stop=0.1, alpha=0.5)
      assert cfg.safety_zone_mode == "fixed"  # default
      assert cfg.alpha == 0.5  # existing field unchanged


class TestObstacleConfig:
   """Tests for ObstacleConfig model."""

   def test_obstacle_config_valid(self):
      """Test ObstacleConfig creation with valid inputs."""
      obs = ObstacleConfig(center=[2.0, 3.0, 4.0], half_extents=[0.5, 0.5, 0.5])
      assert obs.center == [2.0, 3.0, 4.0]
      assert obs.half_extents == [0.5, 0.5, 0.5]

   def test_obstacle_config_center_wrong_length_raises(self):
      """Test ObstacleConfig raises ValidationError for center with wrong length."""
      with pytest.raises(ValidationError, match="center"):
         ObstacleConfig(center=[1.0, 2.0], half_extents=[0.5, 0.5, 0.5])

   def test_obstacle_config_missing_half_extents_raises(self):
      """Test ObstacleConfig raises ValidationError when half_extents is missing."""
      with pytest.raises(ValidationError):
         ObstacleConfig(center=[1.0, 2.0, 3.0])  # missing half_extents

   def test_obstacle_config_edge_case_zero_half_extents(self):
      """Test ObstacleConfig allows zero half_extents (point obstacle)."""
      obs = ObstacleConfig(center=[0.0, 0.0, 0.0], half_extents=[0.0, 0.0, 0.0])
      assert obs.half_extents == [0.0, 0.0, 0.0]


class TestRoomConfig:
   """Tests for RoomConfig model."""

   def test_room_config_valid(self):
      """Test RoomConfig creation with valid inputs."""
      room = RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0, 10.0])
      assert room.min == [-10.0, -10.0, 0.0]
      assert room.max == [10.0, 10.0, 10.0]

   def test_room_config_min_wrong_length_raises(self):
      """Test RoomConfig raises ValidationError for min with wrong length."""
      with pytest.raises(ValidationError, match="min"):
         RoomConfig(min=[-10.0, -10.0], max=[10.0, 10.0, 10.0])

   def test_room_config_max_wrong_length_raises(self):
      """Test RoomConfig raises ValidationError for max with wrong length."""
      with pytest.raises(ValidationError, match="max"):
         RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0])


class TestScenarioConfig:
   """Tests for ScenarioConfig model."""

   def test_scenario_config_valid_minimal(self):
      """Test ScenarioConfig with minimal required fields."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.dt == 0.1
      assert cfg.coordinator is None
      assert cfg.obstacles == []
      assert cfg.room is None

   def test_scenario_config_valid_full(self):
      """Test ScenarioConfig with all fields specified."""
      cfg = ScenarioConfig(
         dt=0.05,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent", params={"horizon": 10}),
         coordinator=ControllerSpec(type="mpc_central", params={"horizon": 5}),
         drones=[
            DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0]),
            DroneConfig(drone_id="d2", start=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
         ],
         obstacles=[ObstacleConfig(center=[2.5, 2.5, 2.5], half_extents=[0.3, 0.3, 0.3])],
         room=RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0, 10.0])
      )
      assert cfg.dt == 0.05
      assert len(cfg.drones) == 2
      assert len(cfg.obstacles) == 1
      assert cfg.room is not None

   def test_scenario_config_missing_physics_raises(self):
      """Test ScenarioConfig raises ValidationError when physics is missing."""
      with pytest.raises(ValidationError, match="physics"):
         ScenarioConfig(
            controller=ControllerSpec(type="mpc_agent"),
            drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )  # type: ignore[call-arg]

   def test_scenario_config_missing_controller_raises(self):
      """Test ScenarioConfig raises ValidationError when controller is missing."""
      with pytest.raises(ValidationError, match="controller"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )  # type: ignore[call-arg]

   def test_scenario_config_missing_drones_raises(self):
      """Test ScenarioConfig raises ValidationError when drones is missing."""
      with pytest.raises(ValidationError, match="drones"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
         )  # type: ignore[call-arg]

   def test_scenario_config_edge_case_empty_drones_list(self):
      """Test ScenarioConfig allows empty drones list (edge case)."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[],
      )
      assert cfg.drones == []

   def test_scenario_config_edge_case_zero_dt(self):
      """Test ScenarioConfig allows zero dt (edge case, though impractical)."""
      cfg = ScenarioConfig(
         dt=0.0,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.dt == 0.0

   def test_scenario_config_many_drones(self):
      """Test ScenarioConfig with many drones."""
      drones = [DroneConfig(drone_id=f"d{i}", start=[float(i), 0.0, 0.0], target=[float(i), 10.0, 10.0]) for i in range(100)]
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=drones,
      )
      assert len(cfg.drones) == 100

   def test_scenario_config_default_lstm_model_path_is_none(self):
      """Default lstm_model_path is None — backward compatible."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.lstm_model_path is None

   def test_scenario_config_accepts_lstm_model_path(self):
      """ScenarioConfig accepts a string lstm_model_path."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])],
         lstm_model_path="/path/to/model.pt"
      )
      assert cfg.lstm_model_path == "/path/to/model.pt"
