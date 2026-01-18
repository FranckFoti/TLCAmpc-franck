from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Use a slightly smaller default font size to fit multi-row figures better.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, })

# Default location of the live_view_horizon_grid folder, resolved relative to this file so it does not depend on the current
# working directory when the script is executed.
_DEFAULT_ROOT = (Path(__file__).resolve().parents[3] / "live_view_horizon_grid").resolve()


def find_result_dirs(root: Path) -> list[Path]:
   """
   Find all result folders under `root` that look like:
     - results
     - results_<tolerance>
   """
   result_dirs: list[Path] = []
   for entry in root.iterdir():
      if not entry.is_dir():
         continue
      if not entry.name.startswith("results"):
         continue
      if not (entry / "metrics.csv").exists():
         continue
      result_dirs.append(entry)
   return sorted(result_dirs)


def tolerance_from_dirname(dirname: str) -> tuple[str, float | None]:
   """
   Map directory name to a human-readable label and a numeric tolerance (if possible).

   examples:
     "results"       -> ("baseline", None)
     "results_0.001" -> ("0.001", 0.001)
     "results_0.5"   -> ("0.5", 0.5)
   """
   if dirname == "results":
      return "baseline", None
   suffix = dirname.split("_", 1)[1] if "_" in dirname else dirname
   try:
      tol_value = float(suffix)
   except ValueError:
      tol_value = None
   return suffix, tol_value


def load_all_metrics(root: Path) -> pd.DataFrame:
   """
   Load and concatenate all metrics.csv files, adding tolerance info.
   """
   frames: list[pd.DataFrame] = []
   for d in find_result_dirs(root):
      label, tol = tolerance_from_dirname(d.name)
      csv_path = d / "metrics.csv"
      df = pd.read_csv(csv_path)
      df["tolerance_label"] = label
      df["tolerance"] = tol
      frames.append(df)

   if not frames:
      raise RuntimeError(f"No metrics.csv files found under {root}")

   df_all = pd.concat(frames, ignore_index=True)
   # Keep all rows; some infeasible runs have steps == 0. We handle those explicitly when computing jerk_total and sim_time_s.
   return df_all.copy()


def add_jerk_total(df: pd.DataFrame) -> pd.DataFrame:
   """Add a `jerk_total` column based on the raw metrics.

   One jerk sample is computed per drone per step after the first two
   velocity samples, so we approximate the total jerk as:

       jerk_total ≈ jerk_mean * max(0, (steps - 2) * num_drones)
   """
   df = df.copy()
   steps_clipped = df["steps"].fillna(0).clip(lower=0)
   samples_per_drone = (steps_clipped - 2).clip(lower=0)
   df["jerk_total"] = df["jerk_mean"] * samples_per_drone * df["num_drones"]
   return df


def prepare_sim_time_per_tolerance(df_tol: pd.DataFrame) -> pd.DataFrame:
   """Compute `sim_time_s` for one tolerance slice.

   Base value: frames * mean_step_time_s.
   For runs with steps == 0 we assign a time that is 10% slower than the
   slowest successful run with the same number of drones.
   """
   df_tol = df_tol.sort_values(["num_drones", "horizon"]).copy()

   # Overall simulation time from start to end (approx.):
   df_tol["sim_time_s"] = df_tol["frames"] * df_tol["mean_step_time_s"]

   # Penalise zero-step runs per N.
   for N, df_N_all in df_tol.groupby("num_drones"):
      base = df_N_all[df_N_all["steps"].fillna(0) > 0]
      if base.empty:
         continue
      slowest = float(base["sim_time_s"].max())
      zero_mask = (df_tol["num_drones"] == N) & (df_tol["steps"].fillna(0) == 0)
      df_tol.loc[zero_mask, "sim_time_s"] = 1.1 * slowest

   return df_tol


def score_best_horizon(df_N: pd.DataFrame) -> float | None:
   """Return the best horizon H using 70% time / 30% jerk weighting.

   Only finished runs are considered. Returns None if no finished
   simulations are available for this N.
   """
   df_finished = df_N[df_N["status"].astype(str).str.lower() == "finished"].copy()
   if df_finished.empty:
      return None

   j = df_finished["jerk_total"].astype(float)
   t = df_finished["sim_time_s"].astype(float)

   j_min, j_max = float(j.min()), float(j.max())
   t_min, t_max = float(t.min()), float(t.max())

   if j_max > j_min:
      j_norm = (j - j_min) / (j_max - j_min)
   else:
      j_norm = j * 0.0

   if t_max > t_min:
      t_norm = (t - t_min) / (t_max - t_min)
   else:
      t_norm = t * 0.0

   score = 0.3 * j_norm + 0.7 * t_norm
   best_idx = score.idxmin()
   return float(df_finished.loc[best_idx, "horizon"])


def plot_subplot_for_n(ax: plt.Axes, df_N: pd.DataFrame) -> None:
   """Render one row (fixed num_drones N) of the multi-diagram figure."""

   if df_N.empty:
      ax.set_visible(False)
      return

   df_N = df_N.sort_values("horizon").copy()

   # Normalise jerk values for plotting (per N & tolerance) so that the vertical scale is comparable across subplots.
   j_all = df_N["jerk_total"].astype(float)
   j_min, j_max = float(j_all.min()), float(j_all.max())
   if j_max > j_min:
      j_plot = (j_all - j_min) / (j_max - j_min)
   else:
      j_plot = j_all * 0.0

   # Left axis: normalised jerk (paper-style cyan line, thicker stroke).
   ax.plot(df_N["horizon"], j_plot, marker="o", color="tab:cyan", linewidth=1, markersize=4)
   N_val = int(df_N["num_drones"].iloc[0])
   ax.set_ylabel(f"{N_val} drones\nnormalised jerk", color="black")
   ax.tick_params(axis="y", labelcolor="black")
   ax.grid(True, which="both", linestyle="--", alpha=0.3)

   # Right axis: overall simulation time (paper-style purple color)
   ax2 = ax.twinx()
   ax2.plot(df_N["horizon"], df_N["sim_time_s"], marker="o", color="tab:purple", linewidth=1, markersize=4)
   ax2.set_ylabel("time [s]", color="black")
   ax2.tick_params(axis="y", labelcolor="black")

   # Mark infeasible simulations in red on the jerk axis.
   infeasible_sim = df_N["status"].astype(str).str.lower() == "infeasible"
   if infeasible_sim.any():
      ax.scatter(df_N.loc[infeasible_sim, "horizon"], j_plot[infeasible_sim], color="red", marker="x", zorder=4)
      ax2.scatter(df_N.loc[infeasible_sim, "horizon"], df_N.loc[infeasible_sim, "sim_time_s"], color="red", marker="x", zorder=4)

   # Draw a vertical dashed line at the best horizon, similar to the "best H" indicator in the paper figure.
   best_H = score_best_horizon(df_N)
   if best_H is not None:
      ax.axvline(best_H, color="red", linestyle=":", alpha=0.6)
      ax.text(best_H, ax.get_ylim()[1], f"best H={int(best_H)}", color="red", rotation=90, va="top", ha="right", fontsize=6)


def plot_multidiagrams_per_tolerance(df: pd.DataFrame, out_dir: Path) -> None:
   """Create multi-diagram figures per tolerance for jerk and overall time."""
   out_dir.mkdir(parents=True, exist_ok=True)

   # Ensure jerk_total is available.
   df = add_jerk_total(df)

   # Sort N so that 2 is at the top and 7 at the bottom (assuming 2..7 exist).
   all_N = sorted(df["num_drones"].unique())

   for tol_label, df_tol_raw in df.groupby("tolerance_label"):
      df_tol = prepare_sim_time_per_tolerance(df_tol_raw)

      # --- Summary figure: jerk_total + sim_time_s ---
      fig, axes = plt.subplots(nrows=len(all_N), ncols=1, sharex=True, figsize=(8, 2.0 * len(all_N)))
      if len(all_N) == 1:
         axes = [axes]

      for ax, N in zip(axes, all_N):
         df_N = df_tol[df_tol["num_drones"] == N]
         plot_subplot_for_n(ax, df_N)

      # Integer ticks on the shared MPC horizon axis.
      horizons = sorted(df_tol["horizon"].unique())
      if horizons:
        axes[-1].set_xticks(horizons)
        axes[-1].set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)

        axes[-1].set_xlabel("MPC horizon")
        n_min, n_max = int(min(all_N)), int(max(all_N))
        fig.suptitle(f"Paper configs: horizon sweep (H=1..20) for {n_min}..{n_max} drones  -  tol={tol_label}")

      legend_elements = [
            Line2D([0], [0], color="tab:cyan", marker="o", linewidth=1, markersize=4, label="normalised jerk"),
            Line2D([0], [0], color="tab:purple", marker="o", linewidth=1, markersize=4, label="overall time [s]"),
            Line2D([0], [0], color="red", marker="x", linewidth=0.0, markersize=5, label="infeasible (no solution)")
      ]
      fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), bbox_to_anchor=(0.5, 0.01), frameon=False)
      fig.tight_layout(rect=[0, 0.03, 1, 0.95])

      summary_path = out_dir / f"multidiag_jerk_time_tol-{tol_label}.png"
      fig.savefig(summary_path, dpi=200)
      plt.close(fig)


def main() -> None:
   parser = argparse.ArgumentParser(description=("Plot live_view_horizon_grid results as curves over horizon H.\n"
                                                 "For each drone number N, creates:\n"
                                                 "  - N*_wall_time_s.png : overall timing vs H\n"
                                                 "  - N*_jerk_mean.png   : jerk_mean vs H\n"
                                                 "One curve per tolerance (result folder)."))
   parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT,
         help="Root folder that contains result folders (default: live_view_horizon_grid next to repo root)")
   parser.add_argument("--out-dir", type=Path, default=None,
         help="Output directory for plots (default: <root>/plots)")
   args = parser.parse_args()

   root: Path = args.root
   print(f"Plotting {root}")
   out_dir: Path = args.out_dir or (root / "plots")

   df = load_all_metrics(root)

   # Create one multi-diagram figure per tolerance with
   #   - total jerk (sum over direction changes)
   #   - overall simulation time (approx. frames * mean_step_time_s)
   plot_multidiagrams_per_tolerance(df, out_dir)

   print(f"Multi-diagram jerk/time plots written to: {out_dir}")


if __name__ == "__main__":
   main()
