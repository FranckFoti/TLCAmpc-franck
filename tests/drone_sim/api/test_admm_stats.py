"""Tests for ADMM visualization features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drone_sim.api.render import render_png
from drone_sim.api.schemas import ADMMStats, StateResponse
from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator


@pytest.fixture
def dmpc_config() -> ScenarioConfig:
    """Load 4DronesDMPC.json config."""
    config_path = Path(__file__).parents[3] / "configs" / "4DronesDMPC.json"
    with open(config_path) as f:
        data = json.load(f)
    return ScenarioConfig.model_validate(data)


@pytest.fixture
def central_config() -> ScenarioConfig:
    """Create a central MPC config."""
    return ScenarioConfig.model_validate({
        "dt": 0.1,
        "room": {"min": [-2.5, -2.5, -2.5], "max": [2.5, 2.5, 2.5]},
        "physics": {"type": "linear_kinematics", "params": {}},
        "controller": {
            "type": "mpc_agent",
            "params": {
                "horizon": 4,
                "q_pos": [3.0, 3.0, 3.0],
                "r_u": [0.1, 0.1, 0.1],
                "u_min": [-3.0, -3.0, -3.0],
                "u_max": [3.0, 3.0, 3.0],
            },
        },
        "coordinator": {
            "type": "mpc_central",
            "params": {"horizon": 4},
        },
        "drones": [
            {
                "drone_id": "drone-1",
                "start": [-1.0, 0.0, 0.0],
                "target": [1.0, 0.0, 0.0],
                "radius": 0.2,
                "safety_zone": 0.5,
                "cons_stop": 0.0,
                "drone_color": "#00FFFF",
            },
            {
                "drone_id": "drone-2",
                "start": [1.0, 0.0, 0.0],
                "target": [-1.0, 0.0, 0.0],
                "radius": 0.2,
                "safety_zone": 0.5,
                "cons_stop": 0.0,
                "drone_color": "#800080",
            },
        ],
    })


class TestStateResponseIncludesAdmmStats:
    """Test that /state response includes ADMM stats for DMPC."""

    def test_state_response_includes_admm_stats(self, dmpc_config: ScenarioConfig) -> None:
        """DMPC simulator to_dict() includes admm_stats."""
        sim = Simulator.from_config(dmpc_config)

        # Run a few steps to populate ADMM stats
        for _ in range(3):
            sim.step()

        state_dict = sim.to_dict()

        assert "admm_stats" in state_dict
        stats = state_dict["admm_stats"]

        assert "iteration_count" in stats
        assert "primal_residual" in stats
        assert "dual_residual" in stats
        assert "converged" in stats
        assert "neighbor_pairs" in stats

        # Check types
        assert isinstance(stats["iteration_count"], int)
        assert isinstance(stats["primal_residual"], float)
        assert isinstance(stats["dual_residual"], float)
        assert isinstance(stats["converged"], bool)
        assert isinstance(stats["neighbor_pairs"], list)

        # Should have neighbor pairs (4 drones = 6 pairs with comm_radius=None)
        assert len(stats["neighbor_pairs"]) == 6

    def test_state_response_no_admm_stats_for_central(
        self, central_config: ScenarioConfig
    ) -> None:
        """Central MPC simulator to_dict() does NOT include admm_stats."""
        sim = Simulator.from_config(central_config)

        # Run a step
        sim.step()

        state_dict = sim.to_dict()

        assert "admm_stats" not in state_dict


class TestAdmmStatsSchemaValidation:
    """Test ADMMStats schema validation."""

    def test_admm_stats_schema_validation(self) -> None:
        """ADMMStats model validates correctly with sample data."""
        stats = ADMMStats(
            iteration_count=10,
            primal_residual=0.001,
            dual_residual=0.002,
            converged=True,
            neighbor_pairs=[["drone-1", "drone-2"], ["drone-1", "drone-3"]],
        )

        assert stats.iteration_count == 10
        assert stats.primal_residual == 0.001
        assert stats.dual_residual == 0.002
        assert stats.converged is True
        assert len(stats.neighbor_pairs) == 2

    def test_state_response_with_admm_stats(self) -> None:
        """StateResponse accepts optional admm_stats."""
        response = StateResponse(
            t=1.0,
            dt=0.1,
            room={"min": [0, 0, 0], "max": [1, 1, 1]},
            drones=[],
            obstacles=[],
            collisions=[],
            admm_stats=ADMMStats(
                iteration_count=5,
                primal_residual=0.01,
                dual_residual=0.02,
                converged=False,
                neighbor_pairs=[],
            ),
        )

        assert response.admm_stats is not None
        assert response.admm_stats.iteration_count == 5

    def test_state_response_without_admm_stats(self) -> None:
        """StateResponse works without admm_stats."""
        response = StateResponse(
            t=1.0,
            dt=0.1,
            room={"min": [0, 0, 0], "max": [1, 1, 1]},
            drones=[],
            obstacles=[],
            collisions=[],
        )

        assert response.admm_stats is None


class TestRenderWithAdmmData:
    """Test render_png with ADMM visualization parameters."""

    def test_render_with_admm_data(self, dmpc_config: ScenarioConfig) -> None:
        """render_png accepts ADMM params and produces valid PNG."""
        sim = Simulator.from_config(dmpc_config)

        # Run a few steps
        for _ in range(3):
            sim.step()

        # Build neighbor links
        drone_id_to_idx = {d.drone_id: i for i, d in enumerate(sim.drones)}
        id_pairs = sim.coordinator.get_neighbor_pairs()
        neighbor_links = [
            (drone_id_to_idx[id_i], drone_id_to_idx[id_j])
            for id_i, id_j in id_pairs
            if id_i in drone_id_to_idx and id_j in drone_id_to_idx
        ]

        png = render_png(
            room_min=sim.room_min,
            room_max=sim.room_max,
            drone_positions=[d.position() for d in sim.drones],
            drone_radii=[d.radius for d in sim.drones],
            drone_safety_zones=[d.safety_zone for d in sim.drones],
            drone_colors=[d.color for d in sim.drones],
            safety_colors=[d.safety_color for d in sim.drones],
            trace_colors=[d.trace_color for d in sim.drones],
            drone_traces=[sim.traces.get(d.drone_id, []) for d in sim.drones],
            obstacles=sim.obstacles,
            step_count=sim.step_count,
            compute_time_s=sim.compute_time_s,
            neighbor_links=neighbor_links,
            admm_iteration_count=sim.coordinator.get_last_iteration_count(),
            admm_converged=sim.coordinator.get_last_converged(),
        )

        # Should produce valid PNG bytes
        assert isinstance(png, bytes)
        assert len(png) > 0
        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_neighbor_links_in_render(self) -> None:
        """render_png accepts neighbor_links parameter and produces valid output."""
        room_min = np.array([0, 0, 0])
        room_max = np.array([1, 1, 1])

        positions = [np.array([0.2, 0.5, 0.5]), np.array([0.8, 0.5, 0.5])]

        png = render_png(
            room_min=room_min,
            room_max=room_max,
            drone_positions=positions,
            drone_radii=[0.1, 0.1],
            drone_safety_zones=[0.2, 0.2],
            drone_colors=["blue", "red"],
            safety_colors=["lightblue", "pink"],
            trace_colors=["blue", "red"],
            drone_traces=[[], []],
            obstacles=[],
            step_count=0,
            compute_time_s=0.0,
            neighbor_links=[(0, 1)],
            admm_iteration_count=5,
            admm_converged=True,
        )

        # Should produce valid PNG
        assert isinstance(png, bytes)
        assert len(png) > 0
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_without_admm_params(self) -> None:
        """render_png works without ADMM params (backward compatible)."""
        room_min = np.array([0, 0, 0])
        room_max = np.array([1, 1, 1])

        png = render_png(
            room_min=room_min,
            room_max=room_max,
            drone_positions=[np.array([0.5, 0.5, 0.5])],
            drone_radii=[0.1],
            drone_safety_zones=[0.2],
            drone_colors=["blue"],
            safety_colors=["lightblue"],
            trace_colors=["blue"],
            drone_traces=[[]],
            obstacles=[],
            step_count=0,
            compute_time_s=0.0,
            # No neighbor_links, admm_iteration_count, admm_converged
        )

        # Should still work
        assert isinstance(png, bytes)
        assert len(png) > 0
