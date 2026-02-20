"""LSTM checkpoint loader for AV-LSTM model inference.

Loads a .pt checkpoint file once at construction time, places the model
in eval mode, and provides CPU-only inference via the `model` property.

Usage:
   loader = LSTMModelLoader(Path("checkpoints/avlstm.pt"))
   with torch.inference_mode():
      mu, sigma = loader.model(X_hat, Y_gt=None, teacher_forcing_ratio=0.0)
"""
from __future__ import annotations

from pathlib import Path

import torch

from drone_sim.prediction.model import AVLSTMModel


class LSTMModelLoader:
   """Load an AV-LSTM checkpoint from disk for CPU inference.

   Loads once at construction time. The model is placed in eval mode.
   CPU-only placement via map_location="cpu" (Pitfall #5 — avoid GPU in MPC loop).

   :param checkpoint_path: Path to .pt file (state_dict saved via torch.save).
   :param model_kwargs: Optional kwargs for AVLSTMModel constructor.
   """

   def __init__(self, checkpoint_path: Path, model_kwargs: dict | None = None, ) -> None:
      checkpoint_path = Path(__file__).parent.parent / "resources" / checkpoint_path
      model_kwargs = model_kwargs or {}
      self._model = AVLSTMModel(**model_kwargs)
      state_dict = torch.load(checkpoint_path, map_location="cpu")
      self._model.load_state_dict(state_dict)
      self._model.eval()

   @property
   def model(self) -> AVLSTMModel:
      """The loaded AVLSTMModel in eval mode on CPU."""
      return self._model
