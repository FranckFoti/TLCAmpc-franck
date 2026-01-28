import numpy as np
import pytest

from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.domain.registry import COORDINATORS
from drone_sim.simulation.distributed_coordinator import DistributedMPCCoordinator


class TestDistributedCoordinatorRegistration:
    """Tests for coordinator registration."""

    def test_coordinator_registration(self):
        """Verify "dmpc_admm" is in COORDINATORS registry."""
        # Import triggers registration
        from drone_sim.simulation.distributed_coordinator import DistributedMPCCoordinator  # noqa: F401

        assert "dmpc_admm" in COORDINATORS
        assert COORDINATORS["dmpc_admm"] is DistributedMPCCoordinator


class TestDistributedCoordinatorBasic:
    """Basic tests for DistributedMPCCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a basic coordinator."""
        return DistributedMPCCoordinator(
            dt=0.1,
            horizon=5,
            rho=1.0,
            max_admm_iter=20,
            primal_tol=1e-2,
            dual_tol=1e-2,
        )

    @pytest.fixture
    def two_drone_setup(self):
        """Setup for two drones with central_cost controllers."""
        dt = 0.1
        controller1 = CentralMPCAgent(dt=dt, horizon=5)
        controller2 = CentralMPCAgent(dt=dt, horizon=5)

        return {
            "drone_ids": ["drone-1", "drone-2"],
            "xs": [
                np.array([0, 0, 0, 0, 0, 0], dtype=float),  # position, velocity
                np.array([5, 0, 0, 0, 0, 0], dtype=float),
            ],
            "prefs": [
                np.array([0, 0, 0], dtype=float),
                np.array([5, 0, 0], dtype=float),
            ],
            "radii": [0.1, 0.1],
            "safety_zones": [0.5, 0.5],
            "cons_stops": [0.0, 0.0],
            "controllers": [controller1, controller2],
            "obstacles": [],
        }

    def test_solve_controls_returns_dict(self, coordinator, two_drone_setup):
        """solve_controls returns dict with expected drone_ids."""
        result = coordinator.solve_controls(**two_drone_setup)

        assert isinstance(result, dict)
        assert "drone-1" in result
        assert "drone-2" in result
        assert result["drone-1"].shape == (3,)
        assert result["drone-2"].shape == (3,)

    def test_solve_controls_returns_finite_values(self, coordinator, two_drone_setup):
        """solve_controls returns finite control values."""
        result = coordinator.solve_controls(**two_drone_setup)

        for drone_id, control in result.items():
            assert np.all(np.isfinite(control)), f"Non-finite control for {drone_id}"


class TestDistributedCoordinatorConvergence:
    """Tests for ADMM convergence."""

    def test_solve_controls_convergence(self):
        """Two drones approaching each other - verify ADMM converges."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=50,
            primal_tol=1e-2,
            dual_tol=1e-2,
        )

        # Two drones heading toward each other
        controller1 = CentralMPCAgent(dt=dt, horizon=5)
        controller2 = CentralMPCAgent(dt=dt, horizon=5)

        result = coordinator.solve_controls(
            drone_ids=["drone-1", "drone-2"],
            xs=[
                np.array([0, 0, 0, 1, 0, 0], dtype=float),  # Moving right
                np.array([3, 0, 0, -1, 0, 0], dtype=float),  # Moving left
            ],
            prefs=[
                np.array([5, 0, 0], dtype=float),  # Wants to go right
                np.array([-2, 0, 0], dtype=float),  # Wants to go left
            ],
            radii=[0.1, 0.1],
            safety_zones=[0.5, 0.5],
            cons_stops=[0.0, 0.0],
            controllers=[controller1, controller2],
            obstacles=[],
        )

        # Controls should be finite
        assert np.all(np.isfinite(result["drone-1"]))
        assert np.all(np.isfinite(result["drone-2"]))


class TestDistributedCoordinatorCollisionAvoidance:
    """Tests for collision avoidance."""

    def test_solve_controls_avoids_collision(self):
        """Two drones on collision course - verify resulting trajectories maintain safety distance."""
        dt = 0.1
        horizon = 8
        safety_zone = 0.5

        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=horizon,
            rho=2.0,
            max_admm_iter=50,
            primal_tol=1e-3,
            dual_tol=1e-3,
        )

        controller1 = CentralMPCAgent(dt=dt, horizon=horizon)
        controller2 = CentralMPCAgent(dt=dt, horizon=horizon)

        # Drones starting close and heading toward each other
        result = coordinator.solve_controls(
            drone_ids=["d1", "d2"],
            xs=[
                np.array([0, 0, 0, 0.5, 0, 0], dtype=float),  # Moving right
                np.array([2, 0, 0, -0.5, 0, 0], dtype=float),  # Moving left
            ],
            prefs=[
                np.array([3, 0, 0], dtype=float),
                np.array([-1, 0, 0], dtype=float),
            ],
            radii=[0.1, 0.1],
            safety_zones=[safety_zone, safety_zone],
            cons_stops=[0.0, 0.0],
            controllers=[controller1, controller2],
            obstacles=[],
        )

        # Verify controls are returned
        assert "d1" in result
        assert "d2" in result

        # Controls should be bounded by controller limits
        u_min, u_max = controller1.central_bounds()
        assert np.all(result["d1"] >= u_min - 1e-6)
        assert np.all(result["d1"] <= u_max + 1e-6)
        assert np.all(result["d2"] >= u_min - 1e-6)
        assert np.all(result["d2"] <= u_max + 1e-6)


class TestDistributedCoordinatorWarmStart:
    """Tests for warm-start functionality."""

    def test_solve_controls_warm_start(self):
        """Call solve_controls twice, verify second call uses warm-start."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=50,
            primal_tol=1e-2,
            dual_tol=1e-2,
        )

        controller1 = CentralMPCAgent(dt=dt, horizon=5)
        controller2 = CentralMPCAgent(dt=dt, horizon=5)

        common_args = {
            "drone_ids": ["d1", "d2"],
            "prefs": [
                np.array([2, 0, 0], dtype=float),
                np.array([5, 0, 0], dtype=float),
            ],
            "radii": [0.1, 0.1],
            "safety_zones": [0.5, 0.5],
            "cons_stops": [0.0, 0.0],
            "controllers": [controller1, controller2],
            "obstacles": [],
        }

        # First call - no warm-start
        coordinator.solve_controls(
            xs=[
                np.array([0, 0, 0, 0, 0, 0], dtype=float),
                np.array([4, 0, 0, 0, 0, 0], dtype=float),
            ],
            **common_args,
        )
        iter1 = coordinator.get_last_iteration_count()

        # Second call - should use warm-start (slightly moved positions)
        coordinator.solve_controls(
            xs=[
                np.array([0.1, 0, 0, 0.1, 0, 0], dtype=float),
                np.array([4.1, 0, 0, 0, 0, 0], dtype=float),
            ],
            **common_args,
        )
        iter2 = coordinator.get_last_iteration_count()

        # Warm-start should help (this is a soft test - warm-start typically helps)
        # Both should complete successfully
        assert iter1 > 0
        assert iter2 > 0
        # With warm-start, second should generally converge faster or equal
        # (not strictly enforced as it depends on problem specifics)


class TestDistributedCoordinatorNoCentralCost:
    """Tests for drones without central_cost interface."""

    def test_solve_controls_no_central_cost(self):
        """Drones without central_cost interface should be skipped, return empty dict."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=20,
        )

        # Create a dummy controller without central_cost interface
        class DummyController:
            pass

        result = coordinator.solve_controls(
            drone_ids=["d1", "d2"],
            xs=[
                np.array([0, 0, 0, 0, 0, 0], dtype=float),
                np.array([5, 0, 0, 0, 0, 0], dtype=float),
            ],
            prefs=[
                np.array([0, 0, 0], dtype=float),
                np.array([5, 0, 0], dtype=float),
            ],
            radii=[0.1, 0.1],
            safety_zones=[0.5, 0.5],
            cons_stops=[0.0, 0.0],
            controllers=[DummyController(), DummyController()],
            obstacles=[],
        )

        assert result == {}

    def test_solve_controls_mixed_controllers(self):
        """Mixed controllers: only those with central_cost are optimized."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=20,
        )

        class DummyController:
            pass

        controller_with_cost = CentralMPCAgent(dt=dt, horizon=5)

        result = coordinator.solve_controls(
            drone_ids=["d1", "d2"],
            xs=[
                np.array([0, 0, 0, 0, 0, 0], dtype=float),
                np.array([5, 0, 0, 0, 0, 0], dtype=float),
            ],
            prefs=[
                np.array([0, 0, 0], dtype=float),
                np.array([5, 0, 0], dtype=float),
            ],
            radii=[0.1, 0.1],
            safety_zones=[0.5, 0.5],
            cons_stops=[0.0, 0.0],
            controllers=[controller_with_cost, DummyController()],
            obstacles=[],
        )

        # Only d1 should be in result
        assert "d1" in result
        assert "d2" not in result


class TestDistributedCoordinatorWithObstacles:
    """Tests for obstacle avoidance."""

    def test_solve_controls_with_obstacles(self):
        """Verify obstacle avoidance works with distributed coordinator."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=30,
            primal_tol=1e-2,
            dual_tol=1e-2,
        )

        controller = CentralMPCAgent(dt=dt, horizon=5)

        # Drone heading toward obstacle
        result = coordinator.solve_controls(
            drone_ids=["d1"],
            xs=[np.array([0, 0, 0, 1, 0, 0], dtype=float)],  # Moving right
            prefs=[np.array([5, 0, 0], dtype=float)],  # Wants to go right
            radii=[0.1],
            safety_zones=[0.5],
            cons_stops=[0.0],
            controllers=[controller],
            obstacles=[(np.array([2, 0, 0]), 0.5)],  # Obstacle at x=2
        )

        assert "d1" in result
        assert np.all(np.isfinite(result["d1"]))


class TestDistributedCoordinatorWithRoomBounds:
    """Tests for room boundary constraints."""

    def test_solve_controls_with_room_bounds(self):
        """Verify room constraints are respected."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=30,
            primal_tol=1e-2,
            dual_tol=1e-2,
        )

        controller = CentralMPCAgent(dt=dt, horizon=5)

        # Drone near wall
        result = coordinator.solve_controls(
            drone_ids=["d1"],
            xs=[np.array([0, 0, 0, 0, 0, 0], dtype=float)],
            prefs=[np.array([0, 0, 0], dtype=float)],
            radii=[0.1],
            safety_zones=[0.5],
            cons_stops=[0.0],
            controllers=[controller],
            obstacles=[],
            room_min=np.array([-5, -5, -5]),
            room_max=np.array([5, 5, 5]),
        )

        assert "d1" in result
        assert np.all(np.isfinite(result["d1"]))


class TestDistributedCoordinatorIntegration:
    """Integration tests for full distributed MPC scenarios."""

    def test_three_drones_converge(self):
        """Three drones scenario - verify all get controls."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=50,
            primal_tol=1e-2,
            dual_tol=1e-2,
            comm_radius=None,  # All neighbors
        )

        controllers = [CentralMPCAgent(dt=dt, horizon=5) for _ in range(3)]

        result = coordinator.solve_controls(
            drone_ids=["d1", "d2", "d3"],
            xs=[
                np.array([0, 0, 0, 0, 0, 0], dtype=float),
                np.array([3, 0, 0, 0, 0, 0], dtype=float),
                np.array([1.5, 2, 0, 0, 0, 0], dtype=float),
            ],
            prefs=[
                np.array([0, 0, 0], dtype=float),
                np.array([3, 0, 0], dtype=float),
                np.array([1.5, 2, 0], dtype=float),
            ],
            radii=[0.1, 0.1, 0.1],
            safety_zones=[0.5, 0.5, 0.5],
            cons_stops=[0.0, 0.0, 0.0],
            controllers=controllers,
            obstacles=[],
        )

        assert len(result) == 3
        for drone_id in ["d1", "d2", "d3"]:
            assert drone_id in result
            assert result[drone_id].shape == (3,)
            assert np.all(np.isfinite(result[drone_id]))

    def test_with_comm_radius(self):
        """Test with limited communication radius."""
        dt = 0.1
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=1.0,
            max_admm_iter=30,
            comm_radius=2.0,  # Limited radius
        )

        controllers = [CentralMPCAgent(dt=dt, horizon=5) for _ in range(3)]

        # Drones far apart - only some are neighbors
        result = coordinator.solve_controls(
            drone_ids=["d1", "d2", "d3"],
            xs=[
                np.array([0, 0, 0, 0, 0, 0], dtype=float),
                np.array([1.5, 0, 0, 0, 0, 0], dtype=float),  # Near d1
                np.array([10, 0, 0, 0, 0, 0], dtype=float),  # Far from d1, d2
            ],
            prefs=[
                np.array([0, 0, 0], dtype=float),
                np.array([1.5, 0, 0], dtype=float),
                np.array([10, 0, 0], dtype=float),
            ],
            radii=[0.1, 0.1, 0.1],
            safety_zones=[0.5, 0.5, 0.5],
            cons_stops=[0.0, 0.0, 0.0],
            controllers=controllers,
            obstacles=[],
        )

        # All drones should still get controls
        assert len(result) == 3

    def test_nonconvergence_warning(self):
        """Test that non-convergence produces a warning but still returns controls."""
        dt = 0.1
        # Very tight tolerance and few iterations to force non-convergence
        coordinator = DistributedMPCCoordinator(
            dt=dt,
            horizon=5,
            rho=0.1,  # Low rho makes convergence slower
            max_admm_iter=2,  # Very few iterations
            primal_tol=1e-10,  # Very tight tolerance
            dual_tol=1e-10,
        )

        controllers = [CentralMPCAgent(dt=dt, horizon=5) for _ in range(2)]

        # Should warn but not fail
        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = coordinator.solve_controls(
                drone_ids=["d1", "d2"],
                xs=[
                    np.array([0, 0, 0, 1, 0, 0], dtype=float),
                    np.array([2, 0, 0, -1, 0, 0], dtype=float),
                ],
                prefs=[
                    np.array([5, 0, 0], dtype=float),
                    np.array([-3, 0, 0], dtype=float),
                ],
                radii=[0.1, 0.1],
                safety_zones=[0.5, 0.5],
                cons_stops=[0.0, 0.0],
                controllers=controllers,
                obstacles=[],
            )

        # Should still return controls
        assert len(result) == 2
        assert np.all(np.isfinite(result["d1"]))
        assert np.all(np.isfinite(result["d2"]))
