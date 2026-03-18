"""Re-render a screenshot from a saved JSON snapshot.

Usage:
    python tools/render_snapshot.py screenshots/paper/MySnapshot.json
    python tools/render_snapshot.py screenshots/paper/MySnapshot.json --dpi 300
    python tools/render_snapshot.py screenshots/paper/MySnapshot.json --out rerendered.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 23,
    "axes.labelsize": 23,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
})

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drone_sim.api.utils.render_helper import (
    draw_room_wireframe,
    draw_sphere_wireframe,
    draw_trace,
    draw_obstacles,
    draw_obj_mesh,
)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "src" / "drone_sim" / "resources" / "assets"


def render_from_snapshot(snapshot: dict, figsize: tuple[float, float] = (10, 8),
                         dpi: int = 150, show_title: bool = True,
                         obj_path: Path | str | None = None,
                         ghost_sphere: bool = False) -> Figure:
    """Render a matplotlib Figure from a JSON snapshot dict."""
    room_min = np.array(snapshot["room_min"])
    room_max = np.array(snapshot["room_max"])
    obstacles = [(np.array(c), np.array(e)) for c, e in snapshot["obstacles"]]
    view = snapshot["view"]

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=view["elev"], azim=view["azim"])

    draw_room_wireframe(ax, room_min, room_max)
    draw_obstacles(ax, obstacles)

    for d in snapshot["drones"]:
        pos = np.array(d["position"])
        color = d["color"]
        safety_r = d["adaptive_safety_radius"] if d["adaptive_safety_radius"] is not None else d["safety_zone"]
        safety_color = d["safety_color"]

        if obj_path is not None:
            draw_obj_mesh(ax, pos, obj_path, scale=d["radius"],
                          color=color if isinstance(color, str) else "steelblue", alpha=0.8)
            # Invisible scatter for legend entry
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=40,
                       c=[color] if isinstance(color, str) else [color],
                       alpha=0.0, label=d["drone_id"])
        else:
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=80,
                       c=[color] if isinstance(color, str) else [color],
                       depthshade=True, label=d["drone_id"])
            draw_sphere_wireframe(ax, pos, safety_r, color=safety_color, alpha=0.6, lw=0.6)

        # Ghost sphere: max adaptive radius or static safety zone
        if ghost_sphere:
            ghost_r = d["max_adaptive_safety_radius"] if d["adaptive_safety_radius"] is not None else d["safety_zone"]
            if ghost_r is not None and ghost_r > 1e-4:
                draw_sphere_wireframe(ax, pos, ghost_r, color=safety_color, alpha=0.15, lw=0.4)
        elif not obj_path:
            # Default behavior without mesh: ghost max sphere for adaptive drones
            if d["adaptive_safety_radius"] is not None and d["max_adaptive_safety_radius"] is not None:
                max_r = d["max_adaptive_safety_radius"]
                if max_r > safety_r + 1e-4:
                    draw_sphere_wireframe(ax, pos, max_r, color=safety_color, alpha=0.12, lw=0.4)

        trace = d.get("trace", [])
        if trace:
            draw_trace(ax, trace, d["trace_color"])

    # Axis limits
    zoom = view.get("zoom", 1.0)
    cx = (room_min[0] + room_max[0]) / 2
    cy = (room_min[1] + room_max[1]) / 2
    cz = (room_min[2] + room_max[2]) / 2
    hx = (room_max[0] - room_min[0]) / 2 * zoom
    hy = (room_max[1] - room_min[1]) / 2 * zoom
    hz = (room_max[2] - room_min[2]) / 2 * zoom
    ax.set_xlim(cx - hx, cx + hx)
    ax.set_ylim(cy - hy, cy + hy)
    ax.set_zlim(cz - hz, cz + hz)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # # Hide grey panes but keep axis labels and ticks
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    # ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    # ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    # ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

    box = [room_max[i] - room_min[i] for i in range(3)]
    if all(b > 0 for b in box):
        ax.set_box_aspect(box)

    if show_title:
        scenario_name = snapshot.get("scenario_name", "unknown")
        fig.text(0.02, 0.02, scenario_name, fontsize=9, ha="left", va="bottom")
        fig.text(0.98, 0.02, f"t = {snapshot['t']:.2f} s (step {snapshot['step_count']})",
                 fontsize=9, ha="right", va="bottom")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-render a screenshot from a JSON snapshot.")
    parser.add_argument("json_path", type=Path, help="Path to the snapshot JSON file")
    parser.add_argument("--out", type=Path, default=None, help="Output image path (default: same name with _rerender.png)")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 8], metavar=("W", "H"), help="Figure size in inches (default: 10 8)")
    parser.add_argument("--no-title", action="store_true", help="Omit scenario name and timestamp annotations")
    parser.add_argument("--obj", type=str, default=None,
                        help="OBJ mesh file to render drones (name in assets/ or full path)")
    parser.add_argument("--ghost-sphere", action="store_true",
                        help="Draw ghost sphere (max adaptive radius or static safety zone)")
    args = parser.parse_args()

    # Resolve OBJ path
    obj_path = None
    if args.obj is not None:
        p = Path(args.obj)
        if not p.exists():
            p = ASSETS_DIR / args.obj
        if not p.exists():
            parser.error(f"OBJ file not found: {args.obj} (also checked {ASSETS_DIR})")
        obj_path = p

    snapshot = json.loads(args.json_path.read_text())
    fig = render_from_snapshot(snapshot, figsize=tuple(args.figsize), dpi=args.dpi,
                               show_title=not args.no_title, obj_path=obj_path,
                               ghost_sphere=args.ghost_sphere)

    out_path = args.out if args.out else args.json_path.with_name(args.json_path.stem + "_rerender.png")
    fig.savefig(str(out_path), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
