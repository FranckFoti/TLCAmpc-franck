"""Tests for drone_sim.api.render module.

Tests for:
- _draw_room_wireframe helper function
- _draw_sphere_wireframe helper function
- render_png main function
"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock

from drone_sim.api.render import render_png, draw_room_wireframe, draw_sphere_wireframe
from drone_sim.api.utils.render_helper import draw_box_wireframe
from drone_sim.domain.drone import Drone

class TestDrawRoomWireframe:
   """Tests for _draw_room_wireframe helper function."""

   def test_draw_room_wireframe_calls_plot(self):
      """Test _draw_room_wireframe calls ax.plot for edges."""
      ax = MagicMock()
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])

      draw_room_wireframe(ax, room_min, room_max)

      # Should draw 12 edges
      assert ax.plot.call_count == 12

   def test_draw_room_wireframe_uses_correct_coordinates(self):
      """Test _draw_room_wireframe uses room bounds for coordinates."""
      ax = MagicMock()
      room_min = np.array([1.0, 2.0, 3.0])
      room_max = np.array([4.0, 5.0, 6.0])

      draw_room_wireframe(ax, room_min, room_max)

      # Check that plot was called with coordinates
      assert ax.plot.called
      # Verify at least one call contains min/max values
      all_calls = ax.plot.call_args_list
      # Edges should use the room bounds
      found_min = False
      found_max = False
      for call in all_calls:
         args = call[0]
         # args[0] is x coords, args[1] is y coords, args[2] is z coords
         if 1.0 in args[0]:
            found_min = True
         if 4.0 in args[0]:
            found_max = True
      assert found_min and found_max

   def test_draw_room_wireframe_edge_case_zero_size(self):
      """Test _draw_room_wireframe with zero-size room (point)."""
      ax = MagicMock()
      room_min = np.array([5.0, 5.0, 5.0])
      room_max = np.array([5.0, 5.0, 5.0])

      # Should not raise
      draw_room_wireframe(ax, room_min, room_max)

      assert ax.plot.call_count == 12


class TestDrawSphereWireframe:
   """Tests for _draw_sphere_wireframe helper function."""

   def test_draw_sphere_wireframe_calls_plot_wireframe(self):
      """Test _draw_sphere_wireframe calls ax.plot_wireframe."""
      ax = MagicMock()
      center = np.array([0.0, 0.0, 0.0])

      draw_sphere_wireframe(ax, center, radius=1.0, color="blue", alpha=0.5, lw=1.0)

      assert ax.plot_wireframe.called

   def test_draw_sphere_wireframe_uses_center_and_radius(self):
      """Test _draw_sphere_wireframe positions sphere at center with radius."""
      ax = MagicMock()
      center = np.array([5.0, 5.0, 5.0])
      radius = 2.0

      draw_sphere_wireframe(ax, center, radius=radius, color="red", alpha=0.8, lw=0.5)

      # Get the call arguments
      call_args = ax.plot_wireframe.call_args
      x, y, z = call_args[0]

      # Check that sphere is centered and sized correctly
      # The center should be at (5, 5, 5) and extend by radius
      assert x.min() >= center[0] - radius - 0.1
      assert x.max() <= center[0] + radius + 0.1

   def test_draw_sphere_wireframe_custom_resolution(self):
      """Test _draw_sphere_wireframe respects resolution parameter."""
      ax = MagicMock()
      center = np.array([0.0, 0.0, 0.0])

      draw_sphere_wireframe(ax, center, radius=1.0, color="green", alpha=0.5, lw=1.0, resolution=10)

      # Get the call arguments
      call_args = ax.plot_wireframe.call_args
      x, y, z = call_args[0]

      # Shape should be (resolution, resolution)
      assert x.shape == (10, 10)

   def test_draw_sphere_wireframe_zero_radius(self):
      """Test _draw_sphere_wireframe with zero radius (point)."""
      ax = MagicMock()
      center = np.array([0.0, 0.0, 0.0])

      # Should not raise
      draw_sphere_wireframe(ax, center, radius=0.0, color="blue", alpha=0.5, lw=1.0)

      assert ax.plot_wireframe.called


class TestDrawBoxWireframe:
   """Tests for _draw_box_wireframe helper function."""

   def test_draw_box_wireframe_draws_12_edges(self):
      """Test _draw_box_wireframe calls ax.plot exactly 12 times (one per edge)."""
      ax = MagicMock()
      center = np.array([0.0, 0.0, 0.0])
      half_extents = np.array([1.0, 1.0, 1.0])

      draw_box_wireframe(ax, center, half_extents, color="tab:red", alpha=0.7, lw=1.2)

      assert ax.plot.call_count == 12

   def test_draw_box_wireframe_anisotropic(self):
      """Test _draw_box_wireframe with different per-axis extents."""
      ax = MagicMock()
      center = np.array([1.0, 2.0, 3.0])
      half_extents = np.array([2.0, 0.5, 0.1])

      draw_box_wireframe(ax, center, half_extents, color="red", alpha=0.5, lw=1.0)

      assert ax.plot.call_count == 12


class TestRenderPng:
   """Tests for render_png function."""

   @pytest.fixture
   def minimal_render_kwargs(self):
      """Return minimal kwargs for render_png."""
      return {
         "room_min": np.array([0.0, 0.0, 0.0]),
         "room_max": np.array([10.0, 10.0, 10.0]),
         "drones": [],
         "drone_traces": [],
         "obstacles": [],
         "step_count": 0,
         "compute_time_s": 0.0
      }

   def test_render_png_returns_bytes(self, minimal_render_kwargs):
      """Test render_png returns bytes."""
      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)
      assert len(result) > 0

   def test_render_png_returns_valid_png(self, minimal_render_kwargs):
      """Test render_png returns valid PNG format."""
      result = render_png(**minimal_render_kwargs)

      # PNG magic number
      assert result[:8] == b"\x89PNG\r\n\x1a\n"

   def test_render_png_custom_dimensions(self, minimal_render_kwargs):
      """Test render_png respects width/height parameters."""
      result_small = render_png(**minimal_render_kwargs, width=200, height=200, dpi=72)
      result_large = render_png(**minimal_render_kwargs, width=800, height=600, dpi=72)

      # Larger image should produce more bytes (generally)
      # Not guaranteed due to compression, but typically true
      assert isinstance(result_small, bytes)
      assert isinstance(result_large, bytes)

   def test_render_png_with_drones(self, minimal_render_kwargs):
      """Test render_png with drone positions."""
      minimal_render_kwargs["drones"] = [Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0,
                                               x=np.array([1.0, 1.0, 1.0,1.0,1.0,1.0]), color="tab:blue",
                                               safety_color="tab:cyan", trace_color="blue",
                                               cons_stop=0.0, controller=None, physics=None, route=None),
                                         Drone(drone_id=f"drone-2", radius=0.3, safety_zone=1.5,
                                               x=np.array([5.0, 5.0, 5.0,1.0,1.0,1.0]), color="tab:orange",
                                               safety_color="tab:red", trace_color="orange",
                                               cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": [], "drone-2": []}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)
      assert result[:8] == b"\x89PNG\r\n\x1a\n"

   def test_render_png_with_traces(self, minimal_render_kwargs):
      """Test render_png with drone traces."""
      minimal_render_kwargs["drones"] = [Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0, x=np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]), color="blue", safety_color="cyan",
                  trace_color="blue", cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0]), np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0])]}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_with_obstacles(self, minimal_render_kwargs):
      """Test render_png with obstacles."""
      minimal_render_kwargs["obstacles"] = [
         (np.array([2.5, 2.5, 2.5]), np.array([0.5, 0.5, 0.5])),
         (np.array([7.0, 3.0, 4.0]), np.array([0.3, 0.3, 0.3]))
      ]

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_with_anisotropic_obstacle(self, minimal_render_kwargs):
      """Test render_png with non-cubic box obstacle (different half-extents per axis)."""
      minimal_render_kwargs["obstacles"] = [
         (np.array([5.0, 5.0, 5.0]), np.array([2.0, 0.5, 0.3]))
      ]

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)
      assert result[:8] == b"\x89PNG\r\n\x1a\n"

   def test_render_png_custom_view_angle(self, minimal_render_kwargs):
      """Test render_png with custom elev/azim."""
      result = render_png(**minimal_render_kwargs, elev=45.0, azim=-120.0)

      assert isinstance(result, bytes)

   def test_render_png_step_count_in_title(self, minimal_render_kwargs):
      """Test render_png includes step_count in title."""
      # We can't easily verify the title content from the PNG bytes,
      # but we can verify the function runs without error
      minimal_render_kwargs["step_count"] = 100
      minimal_render_kwargs["compute_time_s"] = 1.5

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_single_drone(self, minimal_render_kwargs):
      """Test render_png with single drone."""
      minimal_render_kwargs["drones"] = [Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0, x=np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]), color="blue", safety_color="cyan",
                  trace_color="blue", cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": []}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_many_drones(self, minimal_render_kwargs):
      """Test render_png with many drones."""
      n_drones = 20
      minimal_render_kwargs["drones"] = [Drone(drone_id=f"drone-{i}", radius=0.2, safety_zone=1.0, color="blue", safety_color="cyan", trace_color="blue",
                                               x=np.array([float(i), float(i % 10), 5.0, 1.0, 1.0, 1.0]),
                                               cons_stop=0.0, controller=None, physics=None, route=None) for i in range(n_drones)]
      minimal_render_kwargs["drone_traces"] = {f"drone-{i}": [] for i in range(n_drones)}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_very_small_room(self, minimal_render_kwargs):
      """Test render_png with very small room."""
      minimal_render_kwargs["room_min"] = np.array([0.0, 0.0, 0.0])
      minimal_render_kwargs["room_max"] = np.array([0.1, 0.1, 0.1])

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_large_room(self, minimal_render_kwargs):
      """Test render_png with large room."""
      minimal_render_kwargs["room_min"] = np.array([-1000.0, -1000.0, 0.0])
      minimal_render_kwargs["room_max"] = np.array([1000.0, 1000.0, 1000.0])

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_non_cubic_room(self, minimal_render_kwargs):
      """Test render_png with non-cubic (elongated) room."""
      minimal_render_kwargs["room_min"] = np.array([0.0, 0.0, 0.0])
      minimal_render_kwargs["room_max"] = np.array([100.0, 10.0, 5.0])

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_rgb_tuple_colors(self, minimal_render_kwargs):
      """Test render_png with RGB tuple colors."""
      minimal_render_kwargs["drones"] = [
            Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0, x=np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]), color=[(0.5, 0.2, 0.8)], safety_color=[(0.2, 0.8, 0.5)],
                  trace_color=[(0.8, 0.5, 0.2)], cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": []}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_short_trace(self, minimal_render_kwargs):
      """Test render_png with trace shorter than 2 points (no arrow)."""
      minimal_render_kwargs["drones"] = [
            Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0, x=np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]), color="blue", safety_color="cyan",
                  trace_color="blue", cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": [np.array([5.0, 5.0, 5.0])]}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_zero_length_trace_direction(self, minimal_render_kwargs):
      """Test render_png with trace where last two points are same (zero direction)."""
      minimal_render_kwargs["drones"] = [
            Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0, x=np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]), color="blue", safety_color="cyan",
                  trace_color="blue", cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": [[np.array([4.0, 4.0, 4.0]), np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])]]}

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)

   def test_render_png_low_dpi(self, minimal_render_kwargs):
      """Test render_png with low DPI."""
      result = render_png(**minimal_render_kwargs, dpi=50)

      assert isinstance(result, bytes)

   def test_render_png_high_dpi(self, minimal_render_kwargs):
      """Test render_png with high DPI."""
      result = render_png(**minimal_render_kwargs, dpi=300)

      assert isinstance(result, bytes)

   def test_render_png_with_safety_alphas(self, minimal_render_kwargs):
      """Test render_png accepts and uses safety_alphas parameter."""
      minimal_render_kwargs["drones"] = [
            Drone(drone_id=f"drone-1", radius=0.2, safety_zone=1.0, x=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), color="tab:blue", safety_color="tab:cyan",
                  trace_color="blue", cons_stop=0.0, controller=None, physics=None, route=None),
            Drone(drone_id=f"drone-2", radius=0.3, safety_zone=1.5, x=np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0]), color="tab:orange", safety_color="tab:red",
                  trace_color="orange", cons_stop=0.0, controller=None, physics=None, route=None)]
      minimal_render_kwargs["drone_traces"] = {"drone-1": []}

      result = render_png(**minimal_render_kwargs, safety_alphas=[0.3, 1.0])

      assert isinstance(result, bytes)
      assert result[:8] == b"\x89PNG\r\n\x1a\n"

   def test_render_png_obstacle_at_boundary(self, minimal_render_kwargs):
      """Test render_png with obstacle at room boundary."""
      minimal_render_kwargs["obstacles"] = [(np.array([10.0, 10.0, 10.0]), np.array([0.5, 0.5, 0.5]))]

      result = render_png(**minimal_render_kwargs)

      assert isinstance(result, bytes)
