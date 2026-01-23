"""Tests for drone_sim.domain.registry module.

Tests for:
- register_physics, register_controller, register_coordinator decorators
- create_from_registry helper function
- create_physics, create_controller, create_coordinator factory functions
"""

from __future__ import annotations

import pytest
from typing import Any

from drone_sim.domain.registry import (CONTROLLERS, COORDINATORS, PHYSICS, create_controller, create_coordinator,
                                       create_from_registry, create_physics, register_controller, register_coordinator, register_physics)


# =============================================================================
# Mock-classes
# =============================================================================
class MockPhysics:
   """Mock physics class for testing registry."""

   def __init__(self, dt: float = 0.1, custom_param: str = "default"):
      self.dt = dt
      self.custom_param = custom_param


class MockController:
   """Mock controller class for testing registry."""

   def __init__(self, horizon: int = 10, weight: float = 1.0):
      self.horizon = horizon
      self.weight = weight


class MockCoordinator:
   """Mock coordinator class for testing registry."""

   def __init__(self, max_iter: int = 100):
      self.max_iter = max_iter


# =============================================================================
# register_physics Tests
# =============================================================================
class TestRegisterPhysics:
   """Tests for register_physics decorator."""

   def test_register_physics_adds_to_registry(self):
      """Test register_physics decorator adds factory to PHYSICS registry."""
      test_name = "_test_physics_unique_name_1"

      @register_physics(test_name)
      class TestPhysics:
         def __init__(self, dt: float = 0.1):
            self.dt = dt

      assert test_name in PHYSICS
      assert PHYSICS[test_name] is TestPhysics

      # Cleanup
      del PHYSICS[test_name]

   def test_register_physics_overwrites_existing(self):
      """Test registering with same name overwrites previous entry."""
      test_name = "_test_physics_overwrite"

      @register_physics(test_name)
      class First:
         pass

      @register_physics(test_name)
      class Second:
         pass

      assert PHYSICS[test_name] is Second

      # Cleanup
      del PHYSICS[test_name]


# =============================================================================
# register_controller Tests
# =============================================================================
class TestRegisterController:
   """Tests for register_controller decorator."""

   def test_register_controller_adds_to_registry(self):
      """Test register_controller decorator adds factory to CONTROLLERS registry."""
      test_name = "_test_controller_unique_name_1"

      @register_controller(test_name)
      class TestController:
         def __init__(self, horizon: int = 5):
            self.horizon = horizon

      assert test_name in CONTROLLERS
      assert CONTROLLERS[test_name] is TestController

      # Cleanup
      del CONTROLLERS[test_name]


# =============================================================================
# register_coordinator Tests
# =============================================================================
class TestRegisterCoordinator:
   """Tests for register_coordinator decorator."""

   def test_register_coordinator_adds_to_registry(self):
      """Test register_coordinator decorator adds factory to COORDINATORS registry."""
      test_name = "_test_coordinator_unique_name_1"

      @register_coordinator(test_name)
      class TestCoordinator:
         def __init__(self, max_iter: int = 100):
            self.max_iter = max_iter

      assert test_name in COORDINATORS
      assert COORDINATORS[test_name] is TestCoordinator

      # Cleanup
      del COORDINATORS[test_name]


# =============================================================================
# create_from_registry Tests
# =============================================================================
class TestCreateFromRegistry:
   """Tests for create_from_registry helper function."""

   def setup_method(self):
      """Set up test registry."""
      self.test_registry: dict[str, Any] = {"mock_physics": MockPhysics, "mock_controller": MockController}

   def testcreate_from_registry_valid_type(self):
      """Test create_from_registry creates instance with valid type."""
      spec = {"type": "mock_physics", "params": {"dt": 0.2}}
      result = create_from_registry(spec, self.test_registry, "test")
      assert isinstance(result, MockPhysics)
      assert result.dt == 0.2

   def testcreate_from_registry_default_params(self):
      """Test create_from_registry uses defaults when params not provided."""
      spec = {"type": "mock_physics"}
      result = create_from_registry(spec, self.test_registry, "test")
      assert isinstance(result, MockPhysics)
      assert result.dt == 0.1  # default

   def testcreate_from_registry_empty_params(self):
      """Test create_from_registry handles empty params dict."""
      spec = {"type": "mock_physics", "params": {}}
      result = create_from_registry(spec, self.test_registry, "test")
      assert isinstance(result, MockPhysics)

   def testcreate_from_registry_none_params(self):
      """Test create_from_registry handles None params."""
      spec = {"type": "mock_physics", "params": None}
      result = create_from_registry(spec, self.test_registry, "test")
      assert isinstance(result, MockPhysics)

   def testcreate_from_registry_raises_on_missing_type(self):
      """Test create_from_registry raises ValueError when type is missing."""
      spec = {"params": {"dt": 0.1}}
      with pytest.raises(ValueError, match="spec must include 'type'"):
         create_from_registry(spec, self.test_registry, "test")

   def testcreate_from_registry_raises_on_empty_type(self):
      """Test create_from_registry raises ValueError when type is empty string."""
      spec = {"type": "", "params": {}}
      with pytest.raises(ValueError, match="spec must include 'type'"):
         create_from_registry(spec, self.test_registry, "test")

   def testcreate_from_registry_raises_on_unknown_type(self):
      """Test create_from_registry raises ValueError for unknown type."""
      spec = {"type": "nonexistent_type", "params": {}}
      with pytest.raises(ValueError, match="Unknown test type: nonexistent_type"):
         create_from_registry(spec, self.test_registry, "test")

   def testcreate_from_registry_multiple_params(self):
      """Test create_from_registry passes multiple params correctly."""
      spec = {"type": "mock_physics", "params": {"dt": 0.05, "custom_param": "custom"}}
      result = create_from_registry(spec, self.test_registry, "test")
      assert result.dt == 0.05
      assert result.custom_param == "custom"

   def testcreate_from_registry_kind_in_error_message(self):
      """Test create_from_registry includes kind in error messages."""
      spec = {"type": "unknown"}
      with pytest.raises(ValueError, match="Unknown physics type"):
         create_from_registry(spec, self.test_registry, "physics")


# =============================================================================
# create_physics Tests
# =============================================================================
class TestCreatePhysics:
   """Tests for create_physics factory function."""

   def test_create_physics_linear_kinematics(self):
      """Test create_physics creates LinearKinematicsPhysics."""
      # Ensure the linear_kinematics module is imported to register the physics
      from drone_sim.physics import linear_kinematics  # noqa: F401

      spec = {"type": "linear_kinematics", "params": {"dt": 0.1}}
      physics = create_physics(spec)
      assert physics.dt == 0.1
      assert hasattr(physics, "A")
      assert hasattr(physics, "B")

   def test_create_physics_raises_on_unknown_type(self):
      """Test create_physics raises ValueError for unknown type."""
      spec = {"type": "unknown_physics", "params": {}}
      with pytest.raises(ValueError, match="Unknown physics type"):
         create_physics(spec)

   def test_create_physics_raises_on_missing_type(self):
      """Test create_physics raises ValueError when type is missing."""
      spec = {"params": {"dt": 0.1}}
      with pytest.raises(ValueError, match="physics spec must include 'type'"):
         create_physics(spec)


# =============================================================================
# create_controller Tests
# =============================================================================
class TestCreateController:
   """Tests for create_controller factory function."""

   def test_create_controller_mpc_agent(self):
      """Test create_controller creates CentralMPCAgent."""
      # Ensure the central_cost module is imported to register the controller
      from drone_sim.controllers import central_cost  # noqa: F401

      spec = {"type": "mpc_agent", "params": {"dt": 0.1, "horizon": 8}}
      controller = create_controller(spec)
      assert controller.dt == 0.1
      assert controller.horizon == 8

   def test_create_controller_raises_on_unknown_type(self):
      """Test create_controller raises ValueError for unknown type."""
      spec = {"type": "unknown_controller", "params": {}}
      with pytest.raises(ValueError, match="Unknown controller type"):
         create_controller(spec)

   def test_create_controller_raises_on_missing_type(self):
      """Test create_controller raises ValueError when type is missing."""
      spec = {"params": {"horizon": 10}}
      with pytest.raises(ValueError, match="controller spec must include 'type'"):
         create_controller(spec)


# =============================================================================
# create_coordinator Tests
# =============================================================================
class TestCreateCoordinator:
   """Tests for create_coordinator factory function."""

   def test_create_coordinator_mpc_central(self):
      """Test create_coordinator creates CentralMPCGlobalCoordinator."""
      # Ensure the coordinator module is imported to register the coordinator
      from drone_sim.simulation import coordinator  # noqa: F401

      spec = {"type": "mpc_central", "params": {"dt": 0.1, "horizon": 5}}
      coord = create_coordinator(spec)
      assert coord.dt == 0.1
      assert coord.horizon == 5

   def test_create_coordinator_raises_on_unknown_type(self):
      """Test create_coordinator raises ValueError for unknown type."""
      spec = {"type": "unknown_coordinator", "params": {}}
      with pytest.raises(ValueError, match="Unknown coordinator type"):
         create_coordinator(spec)

   def test_create_coordinator_raises_on_missing_type(self):
      """Test create_coordinator raises ValueError when type is missing."""
      spec = {"params": {"horizon": 10}}
      with pytest.raises(ValueError, match="coordinator spec must include 'type'"):
         create_coordinator(spec)


# =============================================================================
# Integration Tests
# =============================================================================
class TestRegistryIntegration:
   """Integration tests for registry system."""

   def test_built_in_physics_registered(self):
      """Test that built-in physics types are registered after import."""
      from drone_sim.physics import linear_kinematics  # noqa: F401

      assert "linear_kinematics" in PHYSICS

   def test_built_in_controller_registered(self):
      """Test that built-in controller types are registered after import."""
      from drone_sim.controllers import central_cost  # noqa: F401

      assert "mpc_agent" in CONTROLLERS

   def test_built_in_coordinator_registered(self):
      """Test that built-in coordinator types are registered after import."""
      from drone_sim.simulation import coordinator  # noqa: F401

      assert "mpc_central" in COORDINATORS

   def test_full_workflow_physics(self):
      """Test complete workflow: register -> create -> use physics."""
      test_name = "_test_workflow_physics"

      @register_physics(test_name)
      class WorkflowPhysics:
         def __init__(self, dt: float = 0.1):
            self.dt = dt

         def step(self, x, u):
            return x

      spec = {"type": test_name, "params": {"dt": 0.05}}
      physics = create_physics(spec)
      assert physics.dt == 0.05
      result = physics.step([1, 2, 3], [0, 0, 0])
      assert result == [1, 2, 3]

      # Cleanup
      del PHYSICS[test_name]
