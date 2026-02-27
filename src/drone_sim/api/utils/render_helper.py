from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from drone_sim.domain.drone import Drone


# ------------------------------------------------------------------ #
# OBJ mesh loader + renderer                                          #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=8)
def _load_obj(path: str) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Load vertices and face indices from a Wavefront .obj file.

    Returns
    -------
    vertices : ndarray, shape (V, 3)
    faces    : list of tuples of 0-based vertex indices (triangles or quads)
    """
    verts: list[list[float]] = []
    faces: list[tuple[int, ...]] = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # face indices can be v, v/vt, v/vt/vn, or v//vn
                idx = tuple(int(p.split("/")[0]) - 1 for p in parts[1:])
                faces.append(idx)
    return np.array(verts, dtype=float), faces


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (degrees), applied X -> Y -> Z."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def draw_obj_mesh(
    ax: object,
    center: np.ndarray,
    obj_path: str | Path,
    *,
    scale: float = 1.0,
    color: str = "steelblue",
    alpha: float = 0.8,
    edgecolor: str = "k",
    linewidth: float = 0.3,
    rotation_deg: tuple[float, float, float] = (-90.0, 0.0, 0.0),
) -> None:
    """Draw a 3D OBJ mesh on *ax* centred at *center*.

    The mesh is normalised so that its longest axis spans *scale* units,
    then rotated by *rotation_deg* (rx, ry, rz in degrees) and translated
    to *center*.

    The default rotation ``(-90, 0, 0)`` converts Blender's Y-up convention
    to the simulation's Z-up convention.
    """
    verts, faces = _load_obj(str(obj_path))
    if len(verts) == 0 or len(faces) == 0:
        return

    # Normalise: centre at origin, fit inside [-0.5, 0.5] * scale
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_range = bbox_max - bbox_min
    max_extent = bbox_range.max()
    if max_extent < 1e-12:
        return
    normed = (verts - (bbox_min + bbox_max) / 2) / max_extent * scale

    # Apply rotation (around the model's own centre, i.e. the origin)
    R = _rotation_matrix(*rotation_deg)
    normed = (R @ normed.T).T

    # Translate to desired centre
    center = np.asarray(center, dtype=float).reshape(3)
    translated = normed + center

    # Build polygon list for Poly3DCollection
    polys = []
    for face in faces:
        polys.append([translated[i] for i in face])

    collection = Poly3DCollection(
        polys, alpha=alpha, facecolor=color, edgecolor=edgecolor, linewidth=linewidth
    )
    ax.add_collection3d(collection)

def _get_edges(x0:float, y0:float, z0:float, x1:float, y1:float, z1:float):
   # 12 edges of the axis-aligned box
   return [  # bottom rectangle
         ((x0, y0, z0), (x1, y0, z0)), ((x1, y0, z0), (x1, y1, z0)),
         ((x1, y1, z0), (x0, y1, z0)), ((x0, y1, z0), (x0, y0, z0)),  # top rectangle
         ((x0, y0, z1), (x1, y0, z1)), ((x1, y0, z1), (x1, y1, z1)),
         ((x1, y1, z1), (x0, y1, z1)), ((x0, y1, z1), (x0, y0, z1)),  # vertical edges
         ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
         ((x1, y1, z0), (x1, y1, z1)), ((x0, y1, z0), (x0, y1, z1))
   ]

def draw_room_wireframe(ax: object, room_min: np.ndarray, room_max: np.ndarray) -> None:
   x0, y0, z0 = room_min.tolist()
   x1, y1, z1 = room_max.tolist()
   edges = _get_edges(x0, y0, z0, x1, y1, z1)

   for (xa, ya, za), (xb, yb, zb) in edges:
      ax.plot([xa, xb], [ya, yb], [za, zb], color="black", linewidth=1.0, alpha=0.4)


def draw_sphere_wireframe(ax: object, center: np.ndarray, radius: float, *, color: str, alpha: float, lw: float, resolution: int = 24) -> None:
   """Draw a simple sphere wireframe."""

   u = np.linspace(0.0, 2.0 * np.pi, resolution)
   v = np.linspace(0.0, np.pi, resolution)
   x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
   y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
   z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

   ax.plot_wireframe(x, y, z, color=color, linewidth=lw, alpha=alpha, rstride=2, cstride=2)


def draw_box_wireframe(ax: object, center: np.ndarray, half_extents: np.ndarray, *, color: str, alpha: float, lw: float) -> None:
   """Draw an axis-aligned box wireframe (12 edges) given center and half-extents."""
   cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
   hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
   x0, x1 = cx - hx, cx + hx
   y0, y1 = cy - hy, cy + hy
   z0, z1 = cz - hz, cz + hz
   edges = _get_edges(x0, y0, z0, x1, y1, z1)

   for (xa, ya, za), (xb, yb, zb) in edges:
      ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=lw, alpha=alpha)


def has_drones_safety_zones(drones: list[Drone]) -> bool:
   return any(d.safety_zone is not None for d in drones)


def draw_ghost_max_sphere(ax: object, drone: Drone, print_always: bool = False):
   if drone.is_adaptive:
      pos = np.asarray(drone.position(), dtype=float).reshape(3)
      max_r = drone.compute_max_adaptive_radius()
      # only draw if meaningfully larger
      if print_always or max_r > float(drone.safety_zone) + 1e-4:
         draw_sphere_wireframe(ax, pos, max_r, color=drone.safety_color, alpha=0.12, lw=0.4)


def draw_trace(ax: object, trace: list[np.ndarray]|list[list[float]], trace_color) -> None:
   # Trace as a dotted/pointed line + a small arrow.
   if trace and len(trace) >= 2:
      T = np.stack([np.asarray(tp, dtype=float).reshape(3) for tp in trace], axis=0)
      ax.plot(T[:, 0], T[:, 1], T[:, 2], color=trace_color, linewidth=1.2, alpha=0.9, linestyle=":", marker=".", markersize=2.5)

      p1 = T[-1]
      d = p1 - T[-2]
      dn = float(np.linalg.norm(d))
      if dn > 1e-9:
         d = d / dn
         arrow_len = 0.35
         ax.quiver(p1[0] - arrow_len * d[0], p1[1] - arrow_len * d[1], p1[2] - arrow_len * d[2], arrow_len * d[0], arrow_len * d[1], arrow_len * d[2],
                   color=trace_color, linewidth=1.0, arrow_length_ratio=0.35)


def draw_neighbor_links(ax: object, neighbor_links: list[tuple[int, int]], drone_positions: list[np.ndarray]) -> None:
   if neighbor_links:
      for i, j in neighbor_links:
         if i < len(drone_positions) and j < len(drone_positions):
            p1 = np.asarray(drone_positions[i], dtype=float)
            p2 = np.asarray(drone_positions[j], dtype=float)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", linewidth=0.8, alpha=0.5, linestyle="--")


def _draw_box_faces(ax: object, center: np.ndarray, half_extents: np.ndarray, *, color: str, alpha: float) -> None:
   """Draw the 6 faces of an axis-aligned box as semi-transparent surfaces."""
   cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
   hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
   x0, x1 = cx - hx, cx + hx
   y0, y1 = cy - hy, cy + hy
   z0, z1 = cz - hz, cz + hz

   # Each face is a 2x2 grid for plot_surface
   # Bottom face (z = z0)
   X = np.array([[x0, x1], [x0, x1]])
   Y = np.array([[y0, y0], [y1, y1]])
   Z = np.array([[z0, z0], [z0, z0]])
   ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False)

   # Top face (z = z1)
   Z = np.array([[z1, z1], [z1, z1]])
   ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False)

   # Front face (y = y0)
   X = np.array([[x0, x1], [x0, x1]])
   Y = np.array([[y0, y0], [y0, y0]])
   Z = np.array([[z0, z0], [z1, z1]])
   ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False)

   # Back face (y = y1)
   Y = np.array([[y1, y1], [y1, y1]])
   ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False)

   # Left face (x = x0)
   X = np.array([[x0, x0], [x0, x0]])
   Y = np.array([[y0, y1], [y0, y1]])
   Z = np.array([[z0, z0], [z1, z1]])
   ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False)

   # Right face (x = x1)
   X = np.array([[x1, x1], [x1, x1]])
   ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=False)


def draw_obstacles(ax: object, obstacles: list[tuple[np.ndarray, np.ndarray]]):
   # Obstacles — draw as 3D filled cuboids with wireframe edges
   if obstacles:
      for i, (c, he) in enumerate(obstacles):
         center_arr = np.asarray(c, dtype=float).reshape(3)
         he_arr = np.asarray(he, dtype=float).reshape(3)
         label = "obstacles" if i == 0 else None
         # Draw semi-transparent filled faces
         _draw_box_faces(ax, center_arr, he_arr, color="tab:red", alpha=0.1)
         # Draw wireframe edges on top
         draw_box_wireframe(ax, center_arr, he_arr, color="tab:red", alpha=0.6, lw=1.5)
         if label:
            # Invisible scatter point to register the legend entry
            ax.scatter([float(center_arr[0])], [float(center_arr[1])], [float(center_arr[2])], s=1, c="tab:red", alpha=0.0, label=label)
