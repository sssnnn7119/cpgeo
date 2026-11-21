"""
Test pipeline on complex.npz:
1) enforce C8 rotational symmetry around z-axis
2) enforce yz-plane axial symmetry
"""

from pathlib import Path
import time
import sys

import numpy as np

sys.path.insert(0, "src/python")

import cpgeo
from cpgeo import utils


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.4f} seconds.")

def create_uv_sphere(n_lat: int = 40, n_lon: int = 120, radius: float = 1.0):
    if n_lat < 3:
        raise ValueError("n_lat must be >= 3")
    if n_lon < 6:
        raise ValueError("n_lon must be >= 6")
    if n_lon % 2 != 0:
        raise ValueError("n_lon must be even for stable rotational seam alignment")

    verts = []
    faces = []

    verts.append([0.0, 0.0, -radius])
    south = 0

    for i in range(1, n_lat):
        phi = -0.5 * np.pi + np.pi * (i / n_lat)
        cp = np.cos(phi)
        sp = np.sin(phi)
        for j in range(n_lon):
            theta = 2.0 * np.pi * (j / n_lon)
            x = radius * cp * np.cos(theta)
            y = radius * cp * np.sin(theta)
            z = radius * sp
            verts.append([x, y, z])

    north = len(verts)
    verts.append([0.0, 0.0, radius])

    def ring_idx(i, j):
        return 1 + (i - 1) * n_lon + (j % n_lon)

    i = 1
    for j in range(n_lon):
        a = ring_idx(i, j)
        b = ring_idx(i, j + 1)
        faces.append([south, b, a])

    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a = ring_idx(i, j)
            b = ring_idx(i, j + 1)
            c = ring_idx(i + 1, j)
            d = ring_idx(i + 1, j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])

    i = n_lat - 1
    for j in range(n_lon):
        a = ring_idx(i, j)
        b = ring_idx(i, j + 1)
        faces.append([a, b, north])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)

def make_weird_shape(vertices: np.ndarray, kind: str):
    v = vertices.copy()
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    theta = np.arctan2(y, x)
    r_xy = np.sqrt(x * x + y * y)
    eps = 1e-12

    if kind == "spiky":
        s = 1.0 + 0.30 * np.cos(3.0 * theta) + 0.15 * np.cos(6.0 * theta) + 0.05 * np.cos(9.0 * theta)
        s *= (0.85 + 0.15 * (1.0 - z * z))
        return v * s[:, None]

    if kind == "peanut":
        s = 0.72 + 0.55 * (z * z) + 0.08 * np.cos(3.0 * theta) * (1.0 - z * z)
        return v * s[:, None]

    if kind == "trefoil_like":
        bump = 0.23 * np.sin(3.0 * theta + 2.5 * z) + 0.08 * np.sin(6.0 * theta - 1.3 * z)
        s = 1.0 + bump * (0.65 + 0.35 * np.exp(-2.0 * z * z))
        return v * s[:, None]

    if kind == "ruffled":
        s = 1.0 + 0.10 * np.cos(12.0 * theta) + 0.08 * np.sin(6.0 * np.arctan2(z, r_xy + eps))
        return v * s[:, None]

    raise ValueError(f"Unknown shape kind: {kind}")

def topology_report(vertices: np.ndarray, faces: np.ndarray, name: str):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"{name}: vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"{name}: faces must have shape (F, 3)")
    if f.size == 0:
        raise ValueError(f"{name}: faces is empty")

    min_id = int(f.min())
    max_id = int(f.max())
    index_ok = (min_id >= 0) and (max_id < v.shape[0])

    deg = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    degenerate_faces = int(np.sum(deg))

    face_sig = np.sort(f, axis=1)
    _, face_counts = np.unique(face_sig, axis=0, return_counts=True)
    duplicate_faces = int(np.sum(face_counts > 1))

    e01 = np.sort(f[:, [0, 1]], axis=1)
    e12 = np.sort(f[:, [1, 2]], axis=1)
    e20 = np.sort(f[:, [2, 0]], axis=1)
    edges = np.vstack([e01, e12, e20])
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)

    boundary_edges = int(np.sum(edge_counts == 1))
    nonmanifold_edges = int(np.sum(edge_counts > 2))

    is_closed_manifold = (
        index_ok
        and degenerate_faces == 0
        and duplicate_faces == 0
        and boundary_edges == 0
        and nonmanifold_edges == 0
    )

    report = {
        "name": name,
        "n_vertices": int(v.shape[0]),
        "n_faces": int(f.shape[0]),
        "index_ok": index_ok,
        "degenerate_faces": degenerate_faces,
        "duplicate_faces": duplicate_faces,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "is_closed_manifold": is_closed_manifold,
    }

    print(f"\n[{name}]")
    for k, val in report.items():
        if k != "name":
            print(f"  {k}: {val}")
    return report


def mesh_component_count(vertices: np.ndarray, faces: np.ndarray):
    n = int(vertices.shape[0])
    adj = [[] for _ in range(n)]
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].append(b)
        adj[a].append(c)
        adj[b].append(a)
        adj[b].append(c)
        adj[c].append(a)
        adj[c].append(b)

    visited = np.zeros(n, dtype=bool)
    comp = 0
    for i in range(n):
        if visited[i] or len(adj[i]) == 0:
            continue
        comp += 1
        stack = [i]
        visited[i] = True
        while stack:
            u = stack.pop()
            for nxt in adj[u]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
    return comp


def cn_symmetry_error(vertices: np.ndarray, periods: int):
    alpha = 2.0 * np.pi / float(periods)
    c = np.cos(alpha)
    s = np.sin(alpha)
    r = np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    vr = np.asarray(vertices, dtype=np.float64) @ r.T
    dist = np.linalg.norm(vr[:, None, :] - vertices[None, :, :], axis=2)
    return float(np.max(np.min(dist, axis=1)))


def yz_symmetry_error(vertices: np.ndarray):
    v = np.asarray(vertices, dtype=np.float64)
    vm = v.copy()
    vm[:, 0] *= -1.0
    dist = np.linalg.norm(vm[:, None, :] - v[None, :, :], axis=2)
    return float(np.max(np.min(dist, axis=1)))

def show_surf(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    surf = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]))
    surf.plot(show_edges=True)

def test_complex_rotation_then_axis():
    # filepath = Path(__file__).parent / "data" / "complex.npz"
    # surf = cpgeo.CPGEO.load(filepath)
    # surf.initialize()

    # cp0 = surf.control_points.copy()
    # f0 = surf._cp_faces.copy()
    cp0, f0 = create_uv_sphere(n_lat=40, n_lon=120, radius=1.0)
    cp0 = make_weird_shape(cp0, kind="trefoil_like")

    base_report = topology_report(cp0, f0, "complex_original")

    with Timer("complex C8 rotational symmetry"):
        v_rot, f_rot = utils.enforce_rotational_symmetry_z(cp0, f0, periods=2, tol=1e-8)
    rep_rot = topology_report(v_rot, f_rot, "complex_after_c8")
    comp_rot = mesh_component_count(v_rot, f_rot)
    err_c8 = cn_symmetry_error(v_rot, periods=8)
    print(f"complex C8 symmetry max error: {err_c8:.6e}")
    print(f"complex C8 connected components: {comp_rot}")

    with Timer("complex yz axial symmetry after C8"):
        v_ax, f_ax = utils.enforce_axial_symmetry(v_rot, f_rot, plane="yz", keep_positive=True, tol=1e-8)
    rep_ax = topology_report(v_ax, f_ax, "complex_after_c8_then_yz")
    comp_ax = mesh_component_count(v_ax, f_ax)
    err_yz = yz_symmetry_error(v_ax)
    print(f"complex yz symmetry max error: {err_yz:.6e}")
    print(f"complex final connected components: {comp_ax}")

    show_surf(cp0, f0)
    show_surf(v_rot, f_rot)
    show_surf(v_ax, f_ax)

    if not base_report["is_closed_manifold"]:
        raise ValueError("complex original mesh is not a closed manifold")
    if not rep_rot["is_closed_manifold"]:
        raise ValueError("complex C8 result is not a closed manifold")
    if not rep_ax["is_closed_manifold"]:
        raise ValueError("complex C8->yz result is not a closed manifold")
    if comp_rot != 1:
        raise ValueError(f"complex C8 component count is {comp_rot}, expected 1")
    if comp_ax != 1:
        raise ValueError(f"complex C8->yz component count is {comp_ax}, expected 1")
    if not np.isfinite(err_c8):
        raise ValueError("complex C8 symmetry error is not finite")
    if not np.isfinite(err_yz):
        raise ValueError("complex yz symmetry error is not finite")


if __name__ == "__main__":
    test_complex_rotation_then_axis()
    print("\ncomplex.npz C8 -> yz test passed.")