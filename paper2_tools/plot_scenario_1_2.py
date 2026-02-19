"""Scenario 1.2 – Radius-Effekt auf Deadlockrate.

Generates two diagrams from paper2_results/results/metrics.csv:
1. Deadlock rate (bar chart) per number of drones, adaptive vs fixed.
2. Arrival time (box-whisker plot) per number of drones, adaptive vs fixed.
3. Mean step time (box-whisker plot) per number of drones, adaptive vs fixed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.utility.constants import COLOR_BY_DRONE_INDEX

# METRICS_CSV = Path(__file__).resolve().parent / "paper2_results" / "result_8_3" / "metrics.csv"
# OUTPUT_DIR = Path(__file__).resolve().parent / "paper2_results" / "plots"

COORDINATOR_TYPES = ["mpc_central", "dmpc_admm"]
SPHERE_TYPES = ["mpc_agent_adaptive", "mpc_agent"]
LABELS = {"mpc_agent_adaptive": "Adaptive $r(v)$", "mpc_agent": "Static $r_{static}$"}
COLORS = {"mpc_agent_adaptive": COLOR_BY_DRONE_INDEX[1], "mpc_agent": COLOR_BY_DRONE_INDEX[3]}
N_DRONES_FIELD = "num_drones"
SPHERE_TYPE_FIELD = "controller_type"
STATUS_FIELD = "status"
FINISHED_STATUS = "finished"
COORDINATOR_TYPE_FIELD = "coordinator"


def load_data(path: Path) -> pd.DataFrame:
   df = pd.read_csv(path)
   # Strip whitespace from column names (Windows line-endings in CSV)
   df.columns = df.columns.str.strip()
   return df


def plot_deadlock_rate(df: pd.DataFrame, output_dir: Path) -> None:
   """Bar chart: deadlock rate (%) per num_drones, adaptive vs fixed."""

   drone_counts = sorted(df[N_DRONES_FIELD].unique())

   rates: dict[str, list[float]] = {st: [] for st in SPHERE_TYPES}

   for n in drone_counts:
      for st in SPHERE_TYPES:
         subset = df[(df[N_DRONES_FIELD] == n) & (df[SPHERE_TYPE_FIELD] == st)]
         total = len(subset)
         deadlocks = (subset[STATUS_FIELD] != FINISHED_STATUS).sum()
         rate = (deadlocks / total * 100) if total > 0 else 0.0
         rates[st].append(rate)

   x = np.arange(len(drone_counts))
   width = 0.35

   fig, ax = plt.subplots(figsize=(8, 5))
   for i, st in enumerate(SPHERE_TYPES):
      offset = (i - 0.5) * width
      bars = ax.bar(x + offset, rates[st], width, label=LABELS[st], color=COLORS[st], edgecolor="white")
      # Annotate bars with percentage
      for bar, val in zip(bars, rates[st]):
         if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

   ax.set_xlabel("Number of Drones $N$", fontsize=12)
   ax.set_ylabel("Deadlock Rate [%]", fontsize=12)
   ax.set_title("Scenario 1.2 – Deadlock Rate: Adaptive vs. Static Radius", fontsize=13)
   ax.set_xticks(x)
   ax.set_xticklabels(drone_counts)
   ax.set_ylim(0, 110)
   ax.legend(fontsize=11)
   ax.grid(axis="y", alpha=0.3)

   fig.tight_layout()
   out_png = output_dir / "scenario_1_2_deadlock_rate.png"
   out_eps = output_dir / "scenario_1_2_deadlock_rate.eps"
   fig.savefig(out_png, dpi=300)
   fig.savefig(out_eps, format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


def _plot_boxplot(df: pd.DataFrame, output_dir: Path, column: str, ylabel: str, title: str, filename_base: str,
      figsize: tuple[float, float] = (10, 5)) -> None:
   """Generic box-whisker plot for finished runs, adaptive vs fixed."""
   finished = df[df[STATUS_FIELD] == FINISHED_STATUS].copy()
   drone_counts = sorted(finished[N_DRONES_FIELD].unique())

   data: dict[str, list[list[float]]] = {st: [] for st in SPHERE_TYPES}
   for n in drone_counts:
      for st in SPHERE_TYPES:
         subset = finished[(finished[N_DRONES_FIELD] == n) & (finished[SPHERE_TYPE_FIELD] == st)]
         data[st].append(subset[column].tolist())

   fig, ax = plt.subplots(figsize=figsize)
   width = 0.35
   gap = 0.8

   for idx, n in enumerate(drone_counts):
      center = idx * (1 + gap)
      for j, st in enumerate(SPHERE_TYPES):
         pos = center + (j - 0.5) * width
         vals = data[st][idx]
         if not vals:
            continue
         ax.boxplot(vals, positions=[pos], widths=width * 0.85, patch_artist=True, showmeans=True,
               meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5), medianprops=dict(color="black", linewidth=1.5),
               boxprops=dict(facecolor=COLORS[st], alpha=0.75), whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
               flierprops=dict(marker="o", markersize=4, alpha=0.5))

   centers = [idx * (1 + gap) for idx in range(len(drone_counts))]
   ax.set_xticks(centers)
   ax.set_xticklabels(drone_counts)
   ax.set_xlabel("Number of Drones $N$", fontsize=12)
   ax.set_ylabel(ylabel, fontsize=12)
   ax.set_title(title, fontsize=13)

   legend_elements = [Patch(facecolor=COLORS[st], alpha=0.75, label=LABELS[st]) for st in SPHERE_TYPES]
   ax.legend(handles=legend_elements, fontsize=11)
   ax.grid(axis="y", alpha=0.3)

   fig.tight_layout()
   out_png = output_dir / f"{filename_base}.png"
   out_eps = output_dir / f"{filename_base}.eps"
   fig.savefig(out_png, dpi=300)
   fig.savefig(out_eps, format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


def plot_arrival_time_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
   """Box-whisker plot: arrival time (steps) for finished runs, adaptive vs fixed."""
   _plot_boxplot(df, output_dir, column="steps", ylabel="Arrival Time [steps]", title="Scenario 1.2 – Arrival Time: Adaptive vs. Static Radius",
         filename_base="scenario_1_2_arrival_time", figsize=(8, 8))


def plot_mean_step_time_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
   """Box-whisker plot: mean step time (s) for finished runs, adaptive vs fixed."""
   _plot_boxplot(df, output_dir, column="mean_step_time_s", ylabel="Mean Step Time [s]", title="Scenario 1.2 – Mean Step Time: Adaptive vs. Static Radius",
         filename_base="scenario_1_2_mean_step_time", figsize=(10, 5))


def main(metrics_csv: Path|None = None) -> None:
   METRICS_CSV = Path(__file__).resolve().parent / "paper2_results" / "result_8_3" / "metrics.csv" \
      if metrics_csv is None \
      else Path(__file__).resolve().parent / "paper2_results" / metrics_csv / "metrics.csv"
   OUTPUT_DIR = Path(__file__).resolve().parent / "paper2_results" / "result_8_3" / "plots"  \
      if metrics_csv is None \
      else Path(__file__).resolve().parent / "paper2_results" / metrics_csv / "plots"

   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

   df = load_data(METRICS_CSV)
   print(f"Loaded {len(df)} rows from {METRICS_CSV}")
   print(f"  num_drones: {sorted(df['num_drones'].unique())}")
   print(f"  sphere_type: {sorted(df['controller_type'].unique())}")
   print(f"  status counts:\n{df['status'].value_counts().to_string()}")
   print()

   plot_deadlock_rate(df, OUTPUT_DIR)
   plot_arrival_time_boxplot(df, OUTPUT_DIR)
   plot_mean_step_time_boxplot(df, OUTPUT_DIR)

   print("\nDone.")


if __name__ == "__main__":
   main()
