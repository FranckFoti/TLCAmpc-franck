"""Tests for drone_sim.prediction.data_generation module.

Tests for:
- DataGenerator.run_scenario() — sliding-window extraction, noise application,
  NPZ output format, infeasible-step skipping, short-scenario edge case.

The real-config integration test and the CLI test are marked @pytest.mark.slow
and are skipped by default (use -m slow or -k slow to run them).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from drone_sim.prediction.data_generation import DataGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_M = 20
_T = 80

# Path constants
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DMPC_CFG = _REPO_ROOT / "configs" / "dmpc" / "4DronesDMPC.json"


def _make_mock_sim(num_drones: int, num_steps: int, drone_ids: list[str] | None = None) -> MagicMock:
   """Return a mock Simulator that yields known state sequences.

   Each drone's state at recorded step *i* is ``np.full(6, float(i))``.
   ``sim.infeasible`` is always False (all steps feasible).
   """
   if drone_ids is None:
      drone_ids = [f"drone-{k}" for k in range(num_drones)]

   step_counter = {"n": 0}

   def _step_side_effect():
      step_counter["n"] += 1

   mock_sim = MagicMock()
   mock_sim.infeasible = False
   mock_sim.step.side_effect = _step_side_effect

   # Each drone returns its state based on how many steps have been recorded.
   recorded_count = {"n": 0}

   # We need drones whose .x attribute returns the right state each call.
   # Since infeasible is always False, recorded_count matches step_counter.
   drones = []
   for did in drone_ids:
      d = MagicMock()
      d.drone_id = did
      drones.append(d)

   # Use a closure over recorded_count so each drone's .x is updated.
   # Simpler: attach state to the mock directly and update in step.
   step_states: list[np.ndarray] = [np.full(6, float(i)) for i in range(num_steps)]

   mock_sim.drones = drones

   # Override step to also update each drone's .x
   call_idx = {"n": 0}

   def _step_with_state():
      idx = call_idx["n"]
      for d in drones:
         d.x = step_states[idx].copy() if idx < len(step_states) else np.zeros(6)
      call_idx["n"] += 1

   mock_sim.step.side_effect = _step_with_state
   return mock_sim


# ---------------------------------------------------------------------------
# Test 1: window extraction shape and count
# ---------------------------------------------------------------------------

class TestWindowExtraction:

   def test_window_extraction_basic(self, tmp_path):
      """Windows have correct shapes and the expected count."""
      num_steps = _M + _T + 10   # enough for windows
      num_drones = 2

      mock_sim = _make_mock_sim(num_drones=num_drones, num_steps=num_steps)

      gen = DataGenerator(m=_M, T=_T, sigma_obs=0.0, seed=0)

      with patch(
         "drone_sim.prediction.data_generation.Simulator.from_config",
         return_value=mock_sim,
      ):
         from drone_sim.domain.config import ScenarioConfig
         # We pass a dummy cfg — from_config is mocked so it won't be read.
         cfg = MagicMock(spec=ScenarioConfig)
         n_windows = gen.run_scenario(cfg, num_steps, tmp_path, "test_scene")

      # Expected windows per drone: num_steps - m - T + 1 = 10 + 1 = 11
      expected_per_drone = num_steps - _M - _T + 1
      assert n_windows == expected_per_drone * num_drones

      npz = np.load(tmp_path / "test_scene.npz")
      assert npz["X"].shape == (n_windows, _M, 6)
      assert npz["Y"].shape == (n_windows, _T, 6)


# ---------------------------------------------------------------------------
# Test 2: noise applied to X but NOT Y
# ---------------------------------------------------------------------------

class TestNoise:

   def test_noise_applied_to_X_not_Y(self, tmp_path):
      """X windows contain observation noise; Y windows are clean."""
      num_steps = _M + _T + 5
      sigma = 0.5   # large enough to detect reliably

      mock_sim = _make_mock_sim(num_drones=1, num_steps=num_steps)

      # Build the clean states the mock produces (no infeasible steps).
      clean_states = np.stack(
         [np.full(6, float(i)) for i in range(num_steps)], axis=0
      )   # (num_steps, 6)

      gen = DataGenerator(m=_M, T=_T, sigma_obs=sigma, seed=42)

      with patch(
         "drone_sim.prediction.data_generation.Simulator.from_config",
         return_value=mock_sim,
      ):
         from drone_sim.domain.config import ScenarioConfig
         cfg = MagicMock(spec=ScenarioConfig)
         gen.run_scenario(cfg, num_steps, tmp_path, "noise_test")

      npz = np.load(tmp_path / "noise_test.npz")
      X = npz["X"]
      Y = npz["Y"]

      # Reconstruct the clean history for the first window.
      clean_X0 = clean_states[:_M]   # (m, 6) — first window
      # X[0] should differ from clean_X0 due to noise.
      assert not np.allclose(X[0], clean_X0), "X should have noise but appears clean"

      # Reconstruct clean future for first window.
      clean_Y0 = clean_states[_M : _M + _T]   # (T, 6)
      assert_array_almost_equal(Y[0], clean_Y0, decimal=10,
                                err_msg="Y should be clean (no noise)")


# ---------------------------------------------------------------------------
# Test 3: NPZ keys and shapes
# ---------------------------------------------------------------------------

class TestNPZFormat:

   def test_npz_keys_and_shapes(self, tmp_path):
      """Saved NPZ has exactly keys 'X' and 'Y' with correct shapes."""
      num_steps = _M + _T + 3

      mock_sim = _make_mock_sim(num_drones=1, num_steps=num_steps)

      gen = DataGenerator(m=_M, T=_T, sigma_obs=0.01, seed=0)

      with patch(
         "drone_sim.prediction.data_generation.Simulator.from_config",
         return_value=mock_sim,
      ):
         from drone_sim.domain.config import ScenarioConfig
         cfg = MagicMock(spec=ScenarioConfig)
         gen.run_scenario(cfg, num_steps, tmp_path, "format_check")

      npz = np.load(tmp_path / "format_check.npz")
      assert set(npz.files) == {"X", "Y"}, f"Unexpected keys: {npz.files}"
      assert npz["X"].shape[1] == _M
      assert npz["X"].shape[2] == 6
      assert npz["Y"].shape[1] == _T
      assert npz["Y"].shape[2] == 6


# ---------------------------------------------------------------------------
# Test 4: zero windows when scenario is too short
# ---------------------------------------------------------------------------

class TestShortScenario:

   def test_zero_windows_short_scenario(self, tmp_path):
      """Returns 0 and writes no file when num_steps < m + T."""
      num_steps = _M + _T - 1   # one step too few for any window

      mock_sim = _make_mock_sim(num_drones=1, num_steps=num_steps)

      gen = DataGenerator(m=_M, T=_T, sigma_obs=0.01, seed=0)

      with patch(
         "drone_sim.prediction.data_generation.Simulator.from_config",
         return_value=mock_sim,
      ):
         from drone_sim.domain.config import ScenarioConfig
         cfg = MagicMock(spec=ScenarioConfig)
         n = gen.run_scenario(cfg, num_steps, tmp_path, "too_short")

      assert n == 0
      assert not (tmp_path / "too_short.npz").exists(), (
         "NPZ file must NOT be written when 0 windows are produced"
      )


# ---------------------------------------------------------------------------
# Test 5: infeasible steps are skipped
# ---------------------------------------------------------------------------

class TestInfeasibleSteps:

   def test_infeasible_steps_skipped(self, tmp_path):
      """States from infeasible steps are not recorded."""
      # We run 30 steps: steps 0-9 feasible, steps 10-14 infeasible, 15-29 feasible.
      # Total feasible steps = 25, which is >= m + T only if T is small.
      # Use a small m/T for this test.
      m_small = 5
      T_small = 5
      num_steps = 30
      infeasible_range = range(10, 15)   # 5 infeasible steps

      step_counter = {"n": 0}
      recorded_states: list[np.ndarray] = []

      mock_sim = MagicMock()

      drones = [MagicMock()]
      drones[0].drone_id = "drone-0"
      drones[0].x = np.zeros(6)
      mock_sim.drones = drones

      call_idx = {"n": 0}
      feasible_step_idx = {"n": 0}

      def _step():
         idx = call_idx["n"]
         if idx in infeasible_range:
            mock_sim.infeasible = True
         else:
            mock_sim.infeasible = False
            # State value = feasible step index
            drones[0].x = np.full(6, float(feasible_step_idx["n"]))
            feasible_step_idx["n"] += 1
         call_idx["n"] += 1

      mock_sim.step.side_effect = _step

      gen = DataGenerator(m=m_small, T=T_small, sigma_obs=0.0, seed=0)

      with patch(
         "drone_sim.prediction.data_generation.Simulator.from_config",
         return_value=mock_sim,
      ):
         from drone_sim.domain.config import ScenarioConfig
         cfg = MagicMock(spec=ScenarioConfig)
         n = gen.run_scenario(cfg, num_steps, tmp_path, "infeasible_test")

      # 30 - 5 infeasible = 25 feasible steps.
      # Windows per drone: 25 - 5 - 5 + 1 = 16
      assert n == 16

      npz = np.load(tmp_path / "infeasible_test.npz")
      # All recorded values should be from feasible steps (0..24).
      # Verify there are no duplicate consecutive states that would indicate
      # infeasible-step duplicates were included.
      X = npz["X"]
      # Check that first window starts at feasible step 0.
      assert_array_equal(
         X[0, 0],
         np.full(6, 0.0),
         err_msg="First state in first window should be feasible step 0",
      )


# ---------------------------------------------------------------------------
# Test 6 (slow): Integration test with real 4DronesDMPC config
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIntegrationRealConfig:

   def test_integration_real_config(self, tmp_path):
      """Load a real config, run 200 steps, verify NPZ written with correct shapes."""
      from drone_sim.domain.config import ScenarioConfig

      assert _DMPC_CFG.exists(), f"Config not found: {_DMPC_CFG}"
      cfg = ScenarioConfig.model_validate_json(_DMPC_CFG.read_text())

      gen = DataGenerator(m=_M, T=_T, sigma_obs=0.01, seed=42)
      n = gen.run_scenario(cfg, 200, tmp_path, "4DronesDMPC")

      out = tmp_path / "4DronesDMPC.npz"
      assert out.exists(), "NPZ file was not written"
      npz = np.load(out)
      assert set(npz.files) == {"X", "Y"}
      assert npz["X"].shape[1] == _M
      assert npz["X"].shape[2] == 6
      assert npz["Y"].shape[1] == _T
      assert npz["Y"].shape[2] == 6


# ---------------------------------------------------------------------------
# Test 7 (slow): CLI generates NPZ
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestCLI:

   def test_cli_generates_npz(self, tmp_path):
      """CLI tool creates at least one NPZ file with correct keys and shapes."""
      # Copy a single real config to a temp directory.
      assert _DMPC_CFG.exists(), f"Config not found: {_DMPC_CFG}"
      cfg_dir = tmp_path / "configs"
      cfg_dir.mkdir()
      shutil.copy(_DMPC_CFG, cfg_dir / _DMPC_CFG.name)

      out_dir = tmp_path / "dataset"
      cli = _REPO_ROOT / "tools" / "generate_data.py"

      result = subprocess.run(
         [
            sys.executable,
            str(cli),
            "--config-dir", str(cfg_dir),
            "--output-dir", str(out_dir),
            "--steps", "150",
            "--seed", "42",
         ],
         capture_output=True,
         text=True,
         timeout=120,
      )

      assert result.returncode == 0, (
         f"CLI failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
      )

      npz_files = list(out_dir.glob("*.npz"))
      assert len(npz_files) >= 1, "No NPZ files found in output directory"

      npz = np.load(npz_files[0])
      assert "X" in npz and "Y" in npz
      assert npz["X"].shape[1] == _M
      assert npz["X"].shape[2] == 6
      assert npz["Y"].shape[1] == _T
      assert npz["Y"].shape[2] == 6
