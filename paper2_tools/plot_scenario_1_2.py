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

N_DRONES_FIELD = "num_drones"
SPHERE_TYPE_FIELD = "controller_type"
STATUS_FIELD = "status"
FINISHED_STATUS = "finished"
COORDINATOR_TYPE_FIELD = "coordinator"

CSV_FIELDS = {"mean_step_time": "mean_step_time_s", "arrival": "wall_time_s", "steps": "steps"}

COORDINATOR_TYPES = ["mpc_central", "dmpc_admm"]
SPHERE_TYPES = ["mpc_agent_adaptive", "mpc_agent"]

LABELS = {
      SPHERE_TYPES[0]: "Adaptive $r(v)$", SPHERE_TYPES[1]: "Static $r_{static}$",
      COORDINATOR_TYPES[0]: "Central MPC", COORDINATOR_TYPES[1]: "DMPC-ADMM",
      CSV_FIELDS["mean_step_time"]: "Mean Step Time [s]", "mean_step_time_without": "Mean Step Time",
      CSV_FIELDS["arrival"]: "Mean Arrival Time [s]", "arrival_without": "Mean Arrival Time",
      CSV_FIELDS["steps"]: "Mean Steps"
}


def load_data(path: Path) -> pd.DataFrame:
   df = pd.read_csv(path)
   # Strip whitespace from column names (Windows line-endings in CSV)
   df.columns = df.columns.str.strip()
   return df


SINGLE_COLOR_INDEX = {
   COORDINATOR_TYPES[0]: 10,
   COORDINATOR_TYPES[1]: 20,
   SPHERE_TYPES[0]: 11,
   SPHERE_TYPES[1]: 32,
}
COMBO_COLOR_INDEX = {
   (COORDINATOR_TYPES[0], SPHERE_TYPES[0]): 2,
   (COORDINATOR_TYPES[0], SPHERE_TYPES[1]): 7,
   (COORDINATOR_TYPES[1], SPHERE_TYPES[0]): 8,
   (COORDINATOR_TYPES[1], SPHERE_TYPES[1]): 6,
}


def _build_combos(coordinator_list: list[str], controller_list: list[str]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str], dict[tuple[str, str], str]]:
   """Build coordinator/controller combos with their labels and colors."""
   combos: list[tuple[str, str]] = []
   for coord in coordinator_list:
      for ctrl in controller_list:
         combos.append((coord, ctrl))

   combo_labels = {(co, ct): f"{LABELS.get(co, co)} / {LABELS.get(ct, ct)}" for co, ct in combos}
   
   # Determine color: use combo color if both vary, otherwise use single type color
   combo_colors = {}
   for co, ct in combos:
      if (co, ct) in COMBO_COLOR_INDEX and len(coordinator_list) > 1 and len(controller_list) > 1:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[COMBO_COLOR_INDEX[(co, ct)]]
      elif len(controller_list) > 1:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[SINGLE_COLOR_INDEX.get(ct, 1)]
      else:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[SINGLE_COLOR_INDEX.get(co, 1)]
   
   return combos, combo_labels, combo_colors


def _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df: pd.DataFrame, output_dir: Path, coordinator_list: list[str], controller_list: list[str],
                                                                                          column: str, ylabel: str, title: str, filename_base: str, figsize: tuple[float, float] = (10, 5),
                                                                                          n_drones_range: range | None = None, log_scale: bool = False) -> None:
   """Box-whisker plot for every coordinator+controller combo, filtered by n_drones_range."""
   finished = df[df[STATUS_FIELD] == FINISHED_STATUS].copy()
   n_drones_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   drone_counts = sorted(n for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)

   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   data: dict[tuple[str, str], list[list[float]]] = {combo: [] for combo in combos}
   for n in drone_counts:
      for combo in combos:
         coord, ctrl = combo
         subset = finished[(finished[N_DRONES_FIELD] == n) & (finished[COORDINATOR_TYPE_FIELD] == coord) & (finished[SPHERE_TYPE_FIELD] == ctrl)]
         data[combo].append(subset[column].tolist())

   fig, ax = plt.subplots(figsize=figsize)
   num_combos = len(combos)
   width = 0.8 / max(num_combos, 1)
   gap = 0.8
   has_empty = False

   for idx, n in enumerate(drone_counts):
      center = idx * (1 + gap)
      for j, combo in enumerate(combos):
         pos = center + (j - (num_combos - 1) / 2) * width
         vals = data[combo][idx]
         if not vals:
            ax.text(pos, 0.5, "\u2205", ha="center", va="center", fontsize=16, color=combo_colors[combo], alpha=0.8, transform=ax.get_xaxis_transform(),
                    clip_on=False)
            has_empty = True
            continue
         ax.boxplot(vals, positions=[pos], widths=width * 0.85, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5), medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(facecolor=combo_colors[combo], alpha=0.75), whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

   centers = [idx * (1 + gap) for idx in range(len(drone_counts))]
   ax.set_xticks(centers)
   ax.set_xticklabels(drone_counts)
   ax.set_xlabel("Number of Drones $N$", fontsize=12)
   ax.set_ylabel(ylabel, fontsize=12)
   ax.set_title(title, fontsize=13)

   if log_scale:
      ax.set_yscale("log")

   legend_elements = [Patch(facecolor=combo_colors[c], alpha=0.75, label=combo_labels[c]) for c in combos]
   ax.legend(handles=legend_elements, fontsize=11)
   ax.grid(axis="y", alpha=0.3)

   if has_empty:
      x_min, x_max = ax.get_xlim()
      ax.set_xlim(x_min, x_max + 0.5)

   fig.tight_layout()
   out_png = output_dir / f"{filename_base}.png"
   out_eps = output_dir / f"{filename_base}.eps"
   fig.savefig(out_png, dpi=300)
   fig.savefig(out_eps, format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


def _deadlock_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df: pd.DataFrame, output_dir: Path,  coordinator_list: list[str], controller_list: list[str],
                                                                                           n_drones_range: list[int], title: str, filename_base: str, figsize: tuple[float, float] = (8, 5), ) -> None:
   """Bar chart: deadlock rate (%) for every coordinator+controller combo, filtered by n_drones_range."""
   drone_counts = sorted(n for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)

   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   rates: dict[tuple[str, str], list[float]] = {combo: [] for combo in combos}
   for n in drone_counts:
      for combo in combos:
         coord, ctrl = combo
         subset = df[(df[N_DRONES_FIELD] == n) & (df[COORDINATOR_TYPE_FIELD] == coord) & (df[SPHERE_TYPE_FIELD] == ctrl)]
         total = len(subset)
         deadlocks = (subset[STATUS_FIELD] != FINISHED_STATUS).sum()
         rate = (deadlocks / total * 100) if total > 0 else 0.0
         rates[combo].append(rate)

   x = np.arange(len(drone_counts))
   num_combos = len(combos)
   width = 0.8 / max(num_combos, 1)

   fig, ax = plt.subplots(figsize=figsize)
   for i, combo in enumerate(combos):
      offset = (i - (num_combos - 1) / 2) * width
      bars = ax.bar(x + offset, rates[combo], width, label=combo_labels[combo], color=combo_colors[combo], edgecolor="white")
      for bar, val in zip(bars, rates[combo]):
         if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

   ax.set_xlabel("Number of Drones $N$", fontsize=12)
   ax.set_ylabel("Deadlock Rate [%]", fontsize=12)
   ax.set_title(title, fontsize=13)
   ax.set_xticks(x)
   ax.set_xticklabels(drone_counts)
   ax.set_ylim(0, 110)
   ax.legend(fontsize=11)
   ax.grid(axis="y", alpha=0.3)

   fig.tight_layout()
   out_png = output_dir / f"{filename_base}.png"
   out_eps = output_dir / f"{filename_base}.eps"
   fig.savefig(out_png, dpi=300)
   fig.savefig(out_eps, format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


def plot_coord_and_sphere_combination(df: pd.DataFrame, output_dir: Path, field: str, n_drones_range: range | list[int] | None = None):
   # Mean step time:
   for coordinator in COORDINATOR_TYPES:
      for log_scale in [True, False]:
         _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir, filename_base=f"{CSV_FIELDS[field]}_{coordinator}_{('log' if log_scale else '')}_{n_drones_range}",
                                                                                               title=f"{LABELS[f"{field}_without"]}: {LABELS[SPHERE_TYPES[0]]} vs. {LABELS[SPHERE_TYPES[1]]} Spheres ({LABELS[coordinator]})",
                                                                                               ylabel=LABELS[CSV_FIELDS[field]], column=CSV_FIELDS[field], coordinator_list=[coordinator], controller_list=SPHERE_TYPES,
                                                                                               n_drones_range=n_drones_range, figsize=(10, 5), log_scale=log_scale)
   for sphere in SPHERE_TYPES:
      for log_scale in [True, False]:
         _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir, filename_base=f"{CSV_FIELDS[field]}_{sphere}_{('log' if log_scale else '')}_{n_drones_range}",
                                                                                               title=f"{LABELS[f"{field}_without"]}: {LABELS[COORDINATOR_TYPES[0]]} vs. {LABELS[COORDINATOR_TYPES[1]]} ({LABELS[sphere]})",
                                                                                               ylabel=LABELS[CSV_FIELDS[field]], column=CSV_FIELDS[field], coordinator_list=COORDINATOR_TYPES, controller_list=[sphere],
                                                                                               n_drones_range=n_drones_range, figsize=(10, 5), log_scale=log_scale)


def plot_all_four_comparison(df: pd.DataFrame, output_dir: Path, column: str, ylabel: str, title_prefix: str, filename_base: str, n_drones_range: range | list[int] | None = None, figsize: tuple[float, float] = (12, 5)) -> None:
   """Boxplot with all 4 combos (2 coordinators x 2 controllers) side by side, for a given column."""
   for log_scale in [False, True]:
      _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir, coordinator_list=COORDINATOR_TYPES,
            controller_list=SPHERE_TYPES, n_drones_range=n_drones_range, column=column, ylabel=ylabel, title=f"{title_prefix}: All Coordinators & Controllers",
            filename_base=f"{filename_base}_all4{('_log' if log_scale else '')}", figsize=figsize, log_scale=log_scale)


def plot_all_four_deadlock(df: pd.DataFrame, output_dir: Path, n_drones_range: range | list[int] | None = None, figsize: tuple[float, float] = (12, 5)) -> None:
   """Deadlock bar chart with all 4 combos (2 coordinators x 2 controllers) side by side."""
   n_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   _deadlock_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir,
         coordinator_list=COORDINATOR_TYPES, controller_list=SPHERE_TYPES, n_drones_range=n_range, title="Deadlock Rate: All Coordinators & Controllers",
         filename_base="deadlock_rate_all4", figsize=figsize, )


def main(metrics_csv: Path | None = None) -> None:
   METRICS_CSV = Path(__file__).resolve().parent / "paper2_results" / "results_hetzner" / "metrics.csv" if metrics_csv is None else Path(
         __file__).resolve().parent / "paper2_results" / metrics_csv / "metrics.csv"
   OUTPUT_DIR = Path(__file__).resolve().parent / "paper2_results" / "results_hetzner" / "plots" if metrics_csv is None else Path(
         __file__).resolve().parent / "paper2_results" / metrics_csv / "plots"

   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

   df = load_data(METRICS_CSV)
   print(f"Loaded {len(df)} rows from {METRICS_CSV}")
   print(f"  num_drones: {sorted(df['num_drones'].unique())}")
   print(f"  sphere_type: {sorted(df['controller_type'].unique())}")
   print(f"  status counts:\n{df['status'].value_counts().to_string()}")
   print()

   n_range = range(2, 12)

   plot_coord_and_sphere_combination(df, OUTPUT_DIR, n_drones_range=n_range, field="mean_step_time")
   plot_coord_and_sphere_combination(df, OUTPUT_DIR, n_drones_range=n_range, field="arrival")
   plot_coord_and_sphere_combination(df, OUTPUT_DIR, field="mean_step_time")
   plot_coord_and_sphere_combination(df, OUTPUT_DIR, field="arrival")

   # All 4 combos (2 coordinators x 2 controllers) direct comparison
   plot_all_four_comparison(df, OUTPUT_DIR, column="mean_step_time_s", ylabel="Mean Step Time [s]", title_prefix="Mean Step Time", filename_base=f"mean_step_time{n_range}", n_drones_range=n_range)
   plot_all_four_comparison(df, OUTPUT_DIR, column="wall_time_s", ylabel="Wall Time [s]", title_prefix="Wall Time", filename_base=f"wall_time{n_range}", n_drones_range=n_range)
   plot_all_four_comparison(df, OUTPUT_DIR, column="steps", ylabel="Arrival Time [steps]", title_prefix="Arrival Time", filename_base=f"arrival_time{n_range}", n_drones_range=n_range)
   plot_all_four_comparison(df, OUTPUT_DIR, column="mean_step_time_s", ylabel="Mean Step Time [s]", title_prefix="Mean Step Time", filename_base=f"mean_step_time")
   plot_all_four_comparison(df, OUTPUT_DIR, column="wall_time_s", ylabel="Wall Time [s]", title_prefix="Wall Time", filename_base=f"wall_time")
   plot_all_four_comparison(df, OUTPUT_DIR, column="steps", ylabel="Arrival Time [steps]", title_prefix="Arrival Time", filename_base=f"arrival_time")

   plot_all_four_deadlock(df, OUTPUT_DIR, n_drones_range=n_range)

   print("\nDone.")


if __name__ == "__main__":
   main()
