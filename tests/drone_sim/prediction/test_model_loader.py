"""Unit tests for LSTMModelLoader.

Tests: checkpoint loading, eval mode, CPU placement, custom kwargs,
missing file error, and forward pass after load.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from drone_sim.prediction.model import AVLSTMModel
from drone_sim.prediction.model_loader import LSTMModelLoader


def _save_temp_checkpoint(model: AVLSTMModel, tmp_dir: str) -> Path:
   """Save model state_dict to a temporary .pt file and return its path."""
   path = Path(tmp_dir) / "test_model.pt"
   torch.save(model.state_dict(), path)
   return path


class TestLSTMModelLoader:
   """Unit tests for LSTMModelLoader."""

   def test_loads_checkpoint(self, tmp_path):
      """After loading, loader.model is an AVLSTMModel instance."""
      model = AVLSTMModel()
      path = tmp_path / "model.pt"
      torch.save(model.state_dict(), path)

      loader = LSTMModelLoader(checkpoint_path=path)

      assert isinstance(loader.model, AVLSTMModel)

   def test_model_in_eval_mode(self, tmp_path):
      """After loading, model.training must be False (explicit eval mode check)."""
      model = AVLSTMModel()
      path = tmp_path / "model.pt"
      torch.save(model.state_dict(), path)

      loader = LSTMModelLoader(checkpoint_path=path)

      assert loader.model.training is False, (
         "Model must be in eval mode after LSTMModelLoader constructs it"
      )

   def test_model_on_cpu(self, tmp_path):
      """After loading, all parameters must be on CPU."""
      model = AVLSTMModel()
      path = tmp_path / "model.pt"
      torch.save(model.state_dict(), path)

      loader = LSTMModelLoader(checkpoint_path=path)

      for name, param in loader.model.named_parameters():
         assert param.device.type == "cpu", (
            f"Parameter {name} is on {param.device.type}, expected cpu"
         )

   def test_custom_model_kwargs(self, tmp_path):
      """model_kwargs override AVLSTMModel defaults; loader uses matching architecture."""
      kwargs = {"n": 4, "d": 32, "T": 10}
      model = AVLSTMModel(**kwargs)
      path = tmp_path / "custom_model.pt"
      torch.save(model.state_dict(), path)

      loader = LSTMModelLoader(checkpoint_path=path, model_kwargs=kwargs)

      assert loader.model.n == 4
      assert loader.model.T == 10

   def test_missing_file_raises(self):
      """Loading from a non-existent path raises FileNotFoundError (or similar)."""
      bad_path = Path("/tmp/this_file_does_not_exist_xyz.pt")

      with pytest.raises((FileNotFoundError, RuntimeError)):
         LSTMModelLoader(checkpoint_path=bad_path)

   def test_forward_pass_after_load(self, tmp_path):
      """Forward pass with dummy (1, 20, 6) input succeeds; output shapes correct."""
      model = AVLSTMModel()  # defaults: T=80, n=6
      path = tmp_path / "model.pt"
      torch.save(model.state_dict(), path)

      loader = LSTMModelLoader(checkpoint_path=path)

      dummy_input = torch.randn(1, 20, 6)
      with torch.inference_mode():
         mu, sigma = loader.model(dummy_input, Y_gt=None, teacher_forcing_ratio=0.0)

      assert mu.shape == (1, 80, 6), f"Expected mu shape (1, 80, 6), got {mu.shape}"
      assert sigma.shape == (1, 80, 6), f"Expected sigma shape (1, 80, 6), got {sigma.shape}"
