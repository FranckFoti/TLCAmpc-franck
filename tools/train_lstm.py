"""CLI entry point for offline AV-LSTM model training.

Loads NPZ training files produced by ``tools/generate_data.py`` (Phase 20),
trains the AV-LSTM model with GaussianNLLLoss + scheduled sampling, and
saves a .pt checkpoint to disk.

Usage::

   python tools/train_lstm.py \\
     --data-dir dataset/ \\
     --output checkpoints/model.pt \\
     --epochs 50 \\
     --lr 1e-3 \\
     --batch-size 64

NPZ contract (Phase 20):
  X — noisy history windows,  shape (num_windows, m, n)
  Y — clean future windows,   shape (num_windows, T, n)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
   p = argparse.ArgumentParser(description="Train the AV-LSTM trajectory prediction model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   p.add_argument("--data-dir", type=Path, required=True, metavar="DIR", help="Directory containing NPZ training files (Phase 20 output).")
   p.add_argument("--output", type=Path, required=True, metavar="PATH", help="Destination path for the saved .pt model checkpoint.")
   p.add_argument("--epochs", type=int, default=150, metavar="N", help="Number of training epochs.")
   p.add_argument("--lr", type=float, default=5e-4, metavar="LR", help="Adam learning rate.")
   p.add_argument("--batch-size", type=int, default=32, metavar="BS", help="Mini-batch size for the DataLoader.")
   p.add_argument("--m", type=int, default=20, metavar="M", help="History length in steps (informational; determined by NPZ data shape).")
   p.add_argument("--T", type=int, default=8, metavar="T", help="Prediction horizon in steps (passed to AVLSTMModel).")
   p.add_argument("--sigma_min", type=int, default=0.85, metavar="sigma_min", help="Avoid floor collapse with a weight close to 0.1, more collapse with smaller weights")
   p.add_argument("--h_lstm", type=int, default=128, metavar="h_lstm", help="hidden dimension")
   p.add_argument("--L", type=int, default=2, metavar="L", help="Encoder Layer")
   p.add_argument("--d-z", type=int, default=32, metavar="D_Z", help="Latent dimension for the VAE bottleneck.")
   p.add_argument("--beta", type=float, default=1.0, metavar="BETA", help="KL divergence weight (1.0 = standard ELBO, 0.0 = NLL only).")
   return p


def main() -> None:
   parser = _build_parser()
   args = parser.parse_args()

   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
   log = logging.getLogger(__name__)

   # ------------------------------------------------------------------
   # Deferred imports — keeps --help startup fast
   # ------------------------------------------------------------------
   from drone_sim.prediction.model import AVLSTMModel
   from drone_sim.prediction.trainer import AVLSTMTrainer

   # ------------------------------------------------------------------
   # Build model and trainer
   # ------------------------------------------------------------------
   model = AVLSTMModel(T=args.T, sigma_min=args.sigma_min, L=args.L, h_lstm=args.h_lstm, d_z=args.d_z)

   trainer = AVLSTMTrainer(model, args.data_dir, n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, beta=args.beta)

   log.info("Starting training — dataset_size=%d, epochs=%d, lr=%s, batch_size=%d", len(trainer.dataset), args.epochs, args.lr, args.batch_size)

   # ------------------------------------------------------------------
   # Train
   # ------------------------------------------------------------------
   losses = trainer.train()

   log.info("Training complete — first_loss=%.4f, last_loss=%.4f, min_loss=%.4f", losses[0], losses[-1], min(losses))

   # ------------------------------------------------------------------
   # Save checkpoint
   # ------------------------------------------------------------------
   args.output.parent.mkdir(parents=True, exist_ok=True)
   trainer.save(args.output)
   log.info("Checkpoint saved to %s", args.output)


if __name__ == "__main__":
   main()
