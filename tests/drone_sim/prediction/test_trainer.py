"""Unit and integration tests for TrajectoryDataset and AVLSTMTrainer.

Synthetic NPZ files are created on disk in ``tmp_path`` fixtures using
float64 arrays (the real Phase 20 format) to exercise the float32
conversion path.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_sim.prediction.model import AVLSTMModel
from drone_sim.prediction.trainer import AVLSTMTrainer, TrajectoryDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_SAMPLES = 8
M = 20   # history length (steps)
T = 80   # prediction horizon (steps)
N = 6    # state dimension


def _write_npz(path: Path, num_samples: int = NUM_SAMPLES) -> None:
    """Write a synthetic NPZ file with float64 arrays (Phase 20 format)."""
    X = np.random.randn(num_samples, M, N).astype(np.float64)
    Y = np.random.randn(num_samples, T, N).astype(np.float64)
    np.savez_compressed(path, X=X, Y=Y)


def _make_tiny_model() -> AVLSTMModel:
    """Return a small AVLSTMModel suitable for fast unit tests."""
    return AVLSTMModel(n=N, d=16, L=1, H_heads=1, h_lstm=32, T=T)


# ---------------------------------------------------------------------------
# TestTrajectoryDataset
# ---------------------------------------------------------------------------


class TestTrajectoryDataset:
    def test_loads_npz_files(self, tmp_path: Path) -> None:
        """Dataset length equals total samples across all NPZ files."""
        _write_npz(tmp_path / "a.npz", num_samples=5)
        _write_npz(tmp_path / "b.npz", num_samples=3)
        ds = TrajectoryDataset(tmp_path)
        assert len(ds) == 8

    def test_float32_dtype(self, tmp_path: Path) -> None:
        """X and Y tensors must be float32 even though disk data is float64."""
        _write_npz(tmp_path / "data.npz")
        ds = TrajectoryDataset(tmp_path)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_raises_on_empty_dir(self, tmp_path: Path) -> None:
        """FileNotFoundError when no NPZ files are present."""
        with pytest.raises(FileNotFoundError):
            TrajectoryDataset(tmp_path)

    def test_getitem_returns_tuple(self, tmp_path: Path) -> None:
        """__getitem__ returns (X, Y) tensors with correct shapes."""
        _write_npz(tmp_path / "data.npz")
        ds = TrajectoryDataset(tmp_path)
        x, y = ds[0]
        assert x.shape == (M, N)
        assert y.shape == (T, N)


# ---------------------------------------------------------------------------
# TestAVLSTMTrainer
# ---------------------------------------------------------------------------


class TestAVLSTMTrainer:
    def test_single_epoch(self, tmp_path: Path) -> None:
        """train() returns a list of exactly 1 float when n_epochs=1."""
        _write_npz(tmp_path / "data.npz")
        model = _make_tiny_model()
        trainer = AVLSTMTrainer(
            model, tmp_path, n_epochs=1, lr=1e-3, batch_size=NUM_SAMPLES
        )
        losses = trainer.train()
        assert len(losses) == 1
        assert isinstance(losses[0], float)

    def test_loss_decreases(self, tmp_path: Path) -> None:
        """Loss should decrease over 5 epochs on small, memorizable data."""
        np.random.seed(0)
        _write_npz(tmp_path / "data.npz", num_samples=8)
        model = _make_tiny_model()
        trainer = AVLSTMTrainer(
            model, tmp_path, n_epochs=5, lr=1e-2, batch_size=8
        )
        losses = trainer.train()
        assert losses[-1] < losses[0], (
            f"Expected loss to decrease but got {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_teacher_forcing_annealing(self) -> None:
        """Verify the annealing formula produces 1.0 at epoch 0 and 0.0 at last epoch."""
        n_epochs = 10

        def ratio(epoch: int) -> float:
            return max(0.0, 1.0 - epoch / max(1, n_epochs - 1))

        assert ratio(0) == pytest.approx(1.0)
        assert ratio(n_epochs - 1) == pytest.approx(0.0)

    def test_save_checkpoint(self, tmp_path: Path) -> None:
        """save() writes a .pt file that can be reloaded with torch.load."""
        _write_npz(tmp_path / "data.npz")
        model = _make_tiny_model()
        trainer = AVLSTMTrainer(
            model, tmp_path, n_epochs=1, lr=1e-3, batch_size=NUM_SAMPLES
        )
        trainer.train()

        ckpt_path = tmp_path / "model.pt"
        trainer.save(ckpt_path)

        assert ckpt_path.exists()
        state = torch.load(ckpt_path, weights_only=True)
        assert isinstance(state, dict)
        # Verify at least one key matches the model's parameter names
        assert any(k in model.state_dict() for k in state)


# ---------------------------------------------------------------------------
# Slow integration test — CLI
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_lstm_cli(tmp_path: Path) -> None:
    """End-to-end: CLI trains a model and saves a .pt checkpoint."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_npz(data_dir / "train.npz", num_samples=8)

    output = tmp_path / "checkpoints" / "model.pt"

    cli = Path(__file__).parent.parent.parent.parent / "tools" / "train_lstm.py"

    result = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--data-dir", str(data_dir),
            "--output", str(output),
            "--epochs", "2",
            "--batch-size", "8",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert output.exists(), "Expected .pt checkpoint file to exist after CLI run"
