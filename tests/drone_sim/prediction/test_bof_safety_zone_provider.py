"""Unit tests for BoFSafetyZoneProvider.

The actual BoF tool is a colleague-supplied stub that raises NotImplementedError.
These tests verify the orchestration around it: the public contract used by
coordinators, the fallback path when payload-builder returns None, the
floor/cap post-processing, and the predicted-trajectories accessor. The tool
call is exercised via subclasses that override `_call_bof_tool` and
`_build_tool_payload` with deterministic stand-ins.
"""
from __future__ import annotations

import numpy as np
import pytest

from drone_sim.prediction.bof_safety_zone_provider import BoFSafetyZoneProvider


# ---------------------------------------------------------------------------
# Helpers — fake BoF tools.
# ---------------------------------------------------------------------------

class _ConstantBoFProvider(BoFSafetyZoneProvider):
   """BoF provider whose tool returns constant trajectory and radii.

   :param trajectory_value: Scalar filling the (H, 3) trajectory.
   :param radius_value: Scalar filling the (H,) radii (raw — pre-postprocess).
   """

   def __init__(self, horizon: int, trajectory_value: float, radius_value: float,
                r_safety_max: float = 5.0) -> None:
      super().__init__(horizon=horizon, r_safety_max=r_safety_max)
      self._traj_v = trajectory_value
      self._rad_v = radius_value

   def _call_bof_tool(self, payload):
      H = payload["horizon"]
      trajectory = np.full((H, 3), self._traj_v, dtype=float)
      radii = np.full(H, self._rad_v, dtype=float)
      return trajectory, radii


class _UnavailableBoFProvider(BoFSafetyZoneProvider):
   """BoF provider whose payload-builder always returns None (fallback path)."""

   def _build_tool_payload(self, neighbor_id):
      return None


# ---------------------------------------------------------------------------
# Public-API tests.
# ---------------------------------------------------------------------------

class TestBoFSafetyZoneProvider:

   def test_compute_returns_dict_with_correct_keys_and_shapes(self):
      provider = _ConstantBoFProvider(horizon=5, trajectory_value=2.0, radius_value=1.5)
      result = provider.compute_neighbor_safety_radii(
         ["d0", "d1"], {"d0": 1.0, "d1": 1.0},
      )
      assert set(result) == {"d0", "d1"}
      assert result["d0"].shape == (5,)
      assert result["d1"].shape == (5,)

   def test_radii_floored_at_r_floor(self):
      """Tool returns 0.5; r_floor=2.0 → output is exactly 2.0 everywhere."""
      provider = _ConstantBoFProvider(horizon=4, trajectory_value=0.0, radius_value=0.5)
      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 2.0})
      np.testing.assert_array_almost_equal(result["d0"], np.full(4, 2.0))

   def test_radii_capped_at_r_safety_max(self):
      """Tool returns 99.0; r_safety_max=5.0 → output capped at 5.0."""
      provider = _ConstantBoFProvider(
         horizon=4, trajectory_value=0.0, radius_value=99.0, r_safety_max=5.0,
      )
      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})
      np.testing.assert_array_almost_equal(result["d0"], np.full(4, 5.0))

   def test_radii_pass_through_when_within_bounds(self):
      """Tool returns 3.0, floor=1.0, cap=5.0 → output is 3.0."""
      provider = _ConstantBoFProvider(horizon=3, trajectory_value=0.0, radius_value=3.0)
      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})
      np.testing.assert_array_almost_equal(result["d0"], np.full(3, 3.0))

   def test_payload_unavailable_falls_back_to_constant_floor(self):
      provider = _UnavailableBoFProvider(horizon=5)
      result = provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.7})
      np.testing.assert_array_almost_equal(result["d0"], np.full(5, 1.7))
      assert result["d0"].shape == (5,)

   def test_unknown_neighbor_id_uses_default_floor_1_0(self):
      provider = _UnavailableBoFProvider(horizon=4)
      result = provider.compute_neighbor_safety_radii(["unknown"], {})
      np.testing.assert_array_almost_equal(result["unknown"], np.full(4, 1.0))

   def test_predicted_trajectories_holds_last_compute_result(self):
      provider = _ConstantBoFProvider(horizon=3, trajectory_value=7.0, radius_value=1.0)
      provider.compute_neighbor_safety_radii(["d0", "d1"], {"d0": 1.0, "d1": 1.0})
      trajs = provider.predicted_trajectories
      assert set(trajs) == {"d0", "d1"}
      assert trajs["d0"].shape == (3, 3)
      np.testing.assert_array_almost_equal(trajs["d0"], np.full((3, 3), 7.0))

   def test_predicted_trajectories_resets_each_call(self):
      """A second compute call replaces the cache — no stale entries from prior call."""
      provider = _ConstantBoFProvider(horizon=3, trajectory_value=1.0, radius_value=1.0)
      provider.compute_neighbor_safety_radii(["d0", "d1"], {"d0": 1.0, "d1": 1.0})
      provider.compute_neighbor_safety_radii(["d2"], {"d2": 1.0})
      trajs = provider.predicted_trajectories
      assert set(trajs) == {"d2"}

   def test_predicted_trajectories_skips_fallback_neighbors(self):
      """Neighbors that fall back (no payload) should not appear in trajectories."""
      provider = _UnavailableBoFProvider(horizon=3)
      provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})
      assert provider.predicted_trajectories == {}

   def test_default_call_bof_tool_is_a_stub(self):
      """The base class's _call_bof_tool raises NotImplementedError with a
      message pointing the colleague at the method to fill in."""
      provider = BoFSafetyZoneProvider(horizon=4)
      with pytest.raises(NotImplementedError) as exc:
         provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})
      assert "_call_bof_tool" in str(exc.value)

   def test_default_payload_builder_returns_neighbor_id_and_horizon(self):
      """Default _build_tool_payload exposes neighbor_id and horizon — the
      contract the colleague extends from."""
      provider = BoFSafetyZoneProvider(horizon=7)
      payload = provider._build_tool_payload("drone-x")
      assert payload == {"neighbor_id": "drone-x", "horizon": 7}

   def test_short_radii_array_raises(self):
      """If the tool returns fewer radii than the horizon, raise ValueError."""

      class _ShortBoF(BoFSafetyZoneProvider):
         def _call_bof_tool(self, payload):
            H = payload["horizon"]
            return np.zeros((H, 3)), np.zeros(H - 1)  # one too short

      provider = _ShortBoF(horizon=5)
      with pytest.raises(ValueError, match="horizon=5"):
         provider.compute_neighbor_safety_radii(["d0"], {"d0": 1.0})
