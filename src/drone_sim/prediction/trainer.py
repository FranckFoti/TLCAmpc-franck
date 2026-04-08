"""Training pipeline for the AV-LSTM trajectory prediction model.

Provides:
  - ``TrajectoryDataset`` — loads Phase 20 NPZ files, returns float32 tensors
  - ``AVLSTMTrainer``     — Adam + GaussianNLLLoss + ReduceLROnPlateau +
                            scheduled sampling (teacher-forcing annealing)

NPZ contract (Phase 20):
  X — noisy history windows,  shape (num_windows, m, n)   dtype float64 on disk
  Y — clean future windows,   shape (num_windows, T, n)   dtype float64 on disk

Both are converted to float32 tensors at load time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from drone_sim.prediction.model import AVLSTMModel


class TrajectoryDataset(Dataset):
   """PyTorch Dataset that loads all *.npz files from a directory.

   Files are expected to contain keys ``"X"`` and ``"Y"`` (uppercase,
   Phase 20 contract).  All arrays are concatenated across files and
   converted to float32 tensors.

   Args:
       data_dir: Directory containing one or more ``*.npz`` files.

   Raises:
       FileNotFoundError: If no NPZ files are found in ``data_dir``.
   """

   def __init__(self, data_dir: Path) -> None:
      npz_files = sorted(data_dir.glob("*.npz"))
      if not npz_files:
         raise FileNotFoundError(f"No NPZ files found in {data_dir}")

      xs: list[np.ndarray] = []
      ys: list[np.ndarray] = []

      for path in npz_files:
         data = np.load(path)
         xs.append(data["X"])
         ys.append(data["Y"])

      X_all = np.concatenate(xs, axis=0)
      Y_all = np.concatenate(ys, axis=0)

      self._X = torch.from_numpy(X_all).float()
      self._Y = torch.from_numpy(Y_all).float()

   def __len__(self) -> int:
      return len(self._X)

   def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
      return self._X[idx], self._Y[idx]


class AVLSTMTrainer:
   """Trainer for the AV-LSTM model.

   Configures:
     - Optimizer:  Adam with ``lr``
     - Loss:       ELBO = ``GaussianNLLLoss`` (reconstruction) + ``beta * KL``
                   (regularisation). GaussianNLLLoss expects *variance*
                   (``sigma ** 2``), not standard deviation.
     - Scheduler:  ``ReduceLROnPlateau(mode="min", factor=0.1, patience=10)``
     - Sampling:   linear teacher-forcing annealing from 1.0 → 0.0 over epochs

   Args:
       model:      Instantiated ``AVLSTMModel``.
       data_dir:   Directory containing NPZ training files.
       n_epochs:   Total number of training epochs (default: 50).
       lr:         Adam learning rate (default: 1e-3).
       batch_size: Mini-batch size for DataLoader (default: 64).
       beta:       KL divergence weight (default: 1.0). Set to 0 to disable
                   the KL term (pure NLL). Values < 1 correspond to beta-VAE.
   """

   def __init__(self, model: AVLSTMModel, data_dir: Path, n_epochs: int = 50, lr: float = 1e-3, batch_size: int = 64, beta: float = 1.0) -> None:
      self.model = model
      self.n_epochs = n_epochs
      self.beta = beta

      self.dataset = TrajectoryDataset(data_dir)
      self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

      self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=10)
      self.criterion = nn.GaussianNLLLoss(reduction="mean")

   def train(self) -> list[float]:
      """Train the model for ``n_epochs`` with scheduled sampling.

      Teacher-forcing ratio is linearly annealed from 1.0 (epoch 0) to
      0.0 (final epoch).  After each epoch the LR scheduler steps on the
      epoch loss.

      Returns:
          List of per-epoch average losses (length == ``n_epochs``).
      """
      epoch_losses: list[float] = []

      for epoch in range(self.n_epochs):
         # Linear annealing: 1.0 at epoch 0 → 0.0 at epoch n_epochs-1
         teacher_forcing_ratio = max(0.0, 1.0 - epoch / max(1, self.n_epochs - 1))
         loss = self._train_epoch(teacher_forcing_ratio)
         self.scheduler.step(loss)
         epoch_losses.append(loss)

      return epoch_losses

   def _train_epoch(self, teacher_forcing_ratio: float) -> float:
      """Run one training epoch.

      Args:
          teacher_forcing_ratio: Per-step probability of using ground-truth
              input instead of the model's own sample.

      Returns:
          Average loss over all batches in the epoch.
      """
      self.model.train()
      total_loss = 0.0
      n_batches = 0

      for X_batch, Y_batch in self.loader:
         self.optimizer.zero_grad()

         Y_target = Y_batch[:, : self.model.T, :]  # slice to model's T if data has more steps
         mu, sigma, mu_z, logvar_z = self.model(X_batch, Y_target, teacher_forcing_ratio)

         # ELBO = NLL (reconstruction) + beta * KL (regularisation)
         var = sigma ** 2  # GaussianNLLLoss expects variance, not std deviation
         nll = self.criterion(mu, Y_target, var)
         kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
         loss = nll + self.beta * kl
         loss.backward()
         self.optimizer.step()

         total_loss += loss.item()
         n_batches += 1

      return total_loss / max(1, n_batches)

   def save(self, path: Path) -> None:
      """Save the model state dict to a .pt checkpoint file.

      Args:
          path: Destination path for the checkpoint (e.g. ``model.pt``).
      """
      torch.save(self.model.state_dict(), path)
