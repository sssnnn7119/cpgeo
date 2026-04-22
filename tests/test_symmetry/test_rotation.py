"""
Test mesh partition and boundary extraction functionality.
"""

from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, 'src/python')

import cpgeo
from cpgeo import utils
import os
import time
class Timer:
    def __init__(self, name="Operation"):
        self.name = name

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


def topology_report(vertices: np.ndarray, faces: np.ndarray, name: str):
    v = np.asarray(vertices)
    f = np.asarray(faces, dtype=np.int64)

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
    manifold_edges = int(np.sum(edge_counts == 2))
    total_unique_edges = int(edge_counts.shape[0])

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
        "total_unique_edges": total_unique_edges,
        "manifold_edges": manifold_edges,
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
    for tri in faces:
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
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
    return comp


def mesh_volume(vertices: np.ndarray, faces: np.ndarray):
    tri = vertices[faces]
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    vol = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0
    return float(abs(vol))


def cn_symmetry_error(vertices: np.ndarray, periods: int):
    alpha = 2.0 * np.pi / float(periods)
    c = np.cos(alpha)
    s = np.sin(alpha)
    r = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    vr = vertices @ r.T
    dist = np.linalg.norm(vr[:, None, :] - vertices[None, :, :], axis=2)
    return float(np.max(np.min(dist, axis=1)))


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


def show_surf(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    surf = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]))
    surf.plot(show_edges=True)




if __name__ == "__main__":
    
    # data = np.load('tests/testdata.npz')

    # cps = data['control_points'].T
    # faces = data['mesh_elements']
    # surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)
    # surf.initialize()
    filepath = Path(__file__).parent / 'data' / 'rotation_4.npz'
    surf = cpgeo.CPGEO.load(filepath)
    print(os.getpid())
    

    # np.savetxt('src/cpp/tests/control_points.txt', surf.control_points, delimiter=',')
    # np.savetxt('src/cpp/tests/knots.txt', surf._knots, delimiter=',')
    # np.savetxt('src/cpp/tests/cp_faces.txt', surf._cp_faces, fmt='%d', delimiter=',')

    surf.initialize()

    # surf.show_control_points()

    cp0 = surf.control_points.copy()
    faces = surf._cp_faces.copy()
    name = "input"
    periods = 3

    base_report = topology_report(cp0, faces, f"{name}_original")
    v0 = mesh_volume(cp0, faces)
    print(f"\n{name} original volume: {v0:.10f}")
    print(f"{name} input C{periods}")

    # show_surf(cp0, faces)

    # surf.refine_surface()

    # show_surf(surf.control_points, surf._cp_faces)

    cp0 = surf.control_points.copy()
    faces = surf._cp_faces.copy()
    with Timer(f"{name} rotational symmetry C{periods}"):
        v_cn, f_cn = utils.enforce_rotational_symmetry_z(cp0, faces, periods=periods, tol=1e-8)
        cn_report = topology_report(v_cn, f_cn, f"{name}_rotational_c{periods}")
        v_after = mesh_volume(v_cn, f_cn)
        sym_err = cn_symmetry_error(v_cn, periods=periods)
        n_comp = mesh_component_count(v_cn, f_cn)
        rel_err = abs(v_after - v0) / max(abs(v0), 1e-12)
        print(f"\n{name} C{periods} volume:      {v_after:.10f}")
        print(f"{name} volume rel err: {rel_err:.3e}")
        print(f"{name} C{periods} sym max err: {sym_err:.3e}")
        print(f"{name} connected components: {n_comp}")
        surf.control_points = v_cn
        surf._cp_faces = f_cn
        surf.initialize()
        surf.show()
        show_surf(v_cn, f_cn)
    if not base_report["is_closed_manifold"]:
        raise ValueError(f"{name}: original is not closed manifold")
    if not cn_report["is_closed_manifold"]:
        raise ValueError(f"{name}: C{periods} output is not closed manifold")
    if n_comp != 1:
        raise ValueError(f"{name}: output component count is {n_comp}, expected 1")

    

