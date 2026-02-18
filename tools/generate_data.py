"""CLI entry point for batch LSTM training-data generation.

Walks a directory tree of scenario JSON configs, runs each scenario through
the simulator, and writes compressed NPZ files (one per scenario) to an
output directory.

Usage::

   python tools/generate_data.py \\
     --config-dir configs/ \\
     --output-dir dataset/ \\
     --steps 500 \\
     --sigma-obs 0.01 \\
     --seed 42

NPZ file format (Phase 21 contract):
  X — noisy history windows,  shape (num_windows, 20, 6)
  Y — clean future windows,   shape (num_windows, 80, 6)

State vector convention: [px, py, pz, vx, vy, vz]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
   p = argparse.ArgumentParser(
      description="Generate LSTM training data from scenario configs.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
   )
   p.add_argument(
      "--config-dir",
      type=Path,
      required=True,
      metavar="DIR",
      help="Root directory to search recursively for *.json scenario configs.",
   )
   p.add_argument(
      "--output-dir",
      type=Path,
      required=True,
      metavar="DIR",
      help="Directory where NPZ files will be written.",
   )
   p.add_argument(
      "--steps",
      type=int,
      default=500,
      metavar="N",
      help="Number of simulation steps to run per scenario.",
   )
   p.add_argument(
      "--sigma-obs",
      type=float,
      default=0.01,
      metavar="SIGMA",
      help="Observation noise standard deviation applied to history windows.",
   )
   p.add_argument(
      "--seed",
      type=int,
      default=42,
      metavar="SEED",
      help="Random seed for reproducible noise generation.",
   )
   p.add_argument(
      "--exclude-dirs",
      nargs="*",
      default=["bof_tests", "fixed_baselin_bug"],
      metavar="DIR_NAME",
      help="Subdirectory names to skip when scanning for configs.",
   )
   return p


def main() -> None:
   # ------------------------------------------------------------------
   # Parse arguments
   # ------------------------------------------------------------------
   parser = _build_parser()
   args = parser.parse_args()

   logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s",
      stream=sys.stdout,
   )
   log = logging.getLogger(__name__)

   # ------------------------------------------------------------------
   # Deferred imports (keep startup fast for argparse --help)
   # ------------------------------------------------------------------
   from drone_sim.domain.config import ScenarioConfig
   from drone_sim.prediction.data_generation import DataGenerator

   # ------------------------------------------------------------------
   # Discover configs
   # ------------------------------------------------------------------
   config_dir: Path = args.config_dir
   if not config_dir.exists():
      log.error("Config directory does not exist: %s", config_dir)
      sys.exit(1)

   exclude_dirs: set[str] = set(args.exclude_dirs or [])
   all_configs = sorted(config_dir.rglob("*.json"))

   # Filter out paths that pass through any excluded directory name.
   configs_to_run = [
      p for p in all_configs
      if not any(part in exclude_dirs for part in p.parts)
   ]

   if not configs_to_run:
      log.warning("No scenario configs found under %s", config_dir)
      return

   log.info(
      "Found %d config(s) to process (excluded %d via --exclude-dirs).",
      len(configs_to_run),
      len(all_configs) - len(configs_to_run),
   )

   # ------------------------------------------------------------------
   # Generate data
   # ------------------------------------------------------------------
   generator = DataGenerator(sigma_obs=args.sigma_obs, seed=args.seed)
   total_windows = 0

   for cfg_path in configs_to_run:
      scenario_name = cfg_path.stem
      log.info("Processing %s …", cfg_path)
      try:
         cfg = ScenarioConfig.model_validate_json(cfg_path.read_text())
         n = generator.run_scenario(cfg, args.steps, args.output_dir, scenario_name)
         total_windows += n
      except Exception as exc:  # noqa: BLE001
         log.warning("Skipping %s — error: %s", cfg_path, exc)
         continue

   log.info(
      "Done. %d windows written across %d scenario(s).",
      total_windows,
      len(configs_to_run),
   )


if __name__ == "__main__":
   main()
