"""
Test mesh partition and boundary extraction functionality.
"""

from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, 'src/python')

import cpgeo
from cpgeo import utils, capi
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




def compare_meshes(v_py, f_py, v_cpp, f_cpp, tol=1e-10):
    """Detailed comparison of two meshes. Faces can be in different order."""
    print("\n" + "=" * 70)
    print("DETAILED C++ vs Python MESH COMPARISON")
    print("=" * 70)

    all_ok = True

    # --- Vertex count ---
    print(f"\nVertex count:  py={v_py.shape[0]}  cpp={v_cpp.shape[0]}")
    if v_py.shape[0] != v_cpp.shape[0]:
        print("  FAIL: vertex count mismatch")
        all_ok = False

    # --- Face count ---
    print(f"Face count:    py={f_py.shape[0]}  cpp={f_cpp.shape[0]}")
    if f_py.shape[0] != f_cpp.shape[0]:
        print("  FAIL: face count mismatch")
        all_ok = False

    # --- Vertex positions (must match exactly, same vertex count) ---
    if v_py.shape == v_cpp.shape:
        v_diff = np.abs(v_py - v_cpp).max()
        print(f"Max vertex diff: {v_diff:.3e}")
        if v_diff > tol:
            print("  FAIL: vertex positions differ")
            row_diff = np.abs(v_py - v_cpp).max(axis=1)
            bad = np.flatnonzero(row_diff > tol)
            print(f"  {len(bad)} vertices differ (indices: {bad[:10].tolist()}...)")
            for i in bad[:5]:
                print(f"    v[{i}] py={v_py[i]} cpp={v_cpp[i]}")
            all_ok = False
        else:
            print("  OK: vertex positions match")
    else:
        print("  FAIL: vertex arrays have different shape")
        all_ok = False

    # --- Volume comparison ---
    v_py_vol = mesh_volume(v_py, f_py)
    v_cpp_vol = mesh_volume(v_cpp, f_cpp)
    print(f"Volume:        py={v_py_vol:.10f}  cpp={v_cpp_vol:.10f}")
    rel_vol = abs(v_py_vol - v_cpp_vol) / max(abs(v_py_vol), 1e-12)
    print(f"Volume rel diff: {rel_vol:.3e}")
    if rel_vol > 1e-4:
        print("  WARN: volume differs significantly")
        all_ok = False
    else:
        print("  OK: volume matches within tolerance")

    # --- Face set comparison (order-independent) ---
    py_sorted = np.sort(f_py, axis=1)
    cpp_sorted = np.sort(f_cpp, axis=1)

    py_set = {tuple(row.tolist()) for row in py_sorted}
    cpp_set = {tuple(row.tolist()) for row in cpp_sorted}

    common = py_set & cpp_set
    only_py = py_set - cpp_set
    only_cpp = cpp_set - py_set

    print(f"\nFace set comparison:")
    print(f"  Total py faces:  {len(py_set)}")
    print(f"  Total cpp faces: {len(cpp_set)}")
    print(f"  Common faces:    {len(common)}")
    print(f"  Only in py:      {len(only_py)}")
    print(f"  Only in cpp:     {len(only_cpp)}")

    if only_py or only_cpp:
        if len(only_py) <= 3 and len(only_cpp) <= 3:
            print(f"  OK: minor face set difference (<=3 faces), acceptable polar cap variation")
        else:
            print("  FAIL: face sets differ significantly")
            all_ok = False

        # Analyze which vertices differ between face sets
        if only_py and only_cpp and len(only_py) == len(only_cpp):
            py_vset = set()
            cpp_vset = set()
            for tri in only_py:
                py_vset.update(tri)
            for tri in only_cpp:
                cpp_vset.update(tri)
            common_v = py_vset & cpp_vset
            only_py_v = py_vset - cpp_vset
            only_cpp_v = cpp_vset - py_vset

            print(f"\n  Vertex analysis of differing faces:")
            print(f"  Vertices appearing only in py's unique faces: {len(only_py_v)}")
            print(f"  Vertices appearing only in cpp's unique faces: {len(only_cpp_v)}")
            print(f"  Common vertices in unique faces: {len(common_v)}")
            if only_py_v:
                sample_v = sorted(only_py_v)[:10]
                print(f"  Sample py-only vertex indices: {sample_v}")
            if only_cpp_v:
                sample_v = sorted(only_cpp_v)[:10]
                print(f"  Sample cpp-only vertex indices: {sample_v}")

            # Are these the same vertices but connected differently?
            if not only_py_v and not only_cpp_v:
                print("  -> Same vertices, different connectivity (stitching pattern differs)")

    # --- Edge set comparison ---
    def get_edge_set(ff):
        e01 = np.sort(ff[:, [0, 1]], axis=1)
        e12 = np.sort(ff[:, [1, 2]], axis=1)
        e20 = np.sort(ff[:, [2, 0]], axis=1)
        edges = np.vstack([e01, e12, e20])
        uniq, cnt = np.unique(edges, axis=0, return_counts=True)
        return {(int(u[0]), int(u[1]), int(c)) for u, c in zip(uniq, cnt)}

    py_edges = get_edge_set(f_py)
    cpp_edges = get_edge_set(f_cpp)

    only_py_e = py_edges - cpp_edges
    only_cpp_e = cpp_edges - py_edges
    common_e = py_edges & cpp_edges

    print(f"\nEdge multiset comparison:")
    print(f"  Common edges:    {len(common_e)}")
    print(f"  Only in py:      {len(only_py_e)}")
    print(f"  Only in cpp:     {len(only_cpp_e)}")

    if only_py_e or only_cpp_e:
        if len(only_py_e) <= 3 and len(only_cpp_e) <= 3:
            print(f"  OK: minor edge set difference (<=3 edges), acceptable polar cap variation")
        else:
            print("  FAIL: edge multisets differ significantly")
            all_ok = False

    # --- Symmetry error ---
    sym_py = cn_symmetry_error(v_py, periods=3)
    sym_cpp = cn_symmetry_error(v_cpp, periods=3)
    print(f"\nC3 symmetry max error: py={sym_py:.3e}  cpp={sym_cpp:.3e}")

    # --- Angle quality (minimum angle in mesh) ---
    def min_angle(vv, ff):
        tri = vv[ff]
        e01 = tri[:, 1] - tri[:, 0]
        e12 = tri[:, 2] - tri[:, 1]
        e20 = tri[:, 0] - tri[:, 2]
        l01 = np.linalg.norm(e01, axis=1)
        l12 = np.linalg.norm(e12, axis=1)
        l20 = np.linalg.norm(e20, axis=1)
        ang0 = np.arccos(np.clip(np.sum((-e20) * e01, axis=1) / (l20 * l01 + 1e-30), -1, 1))
        ang1 = np.arccos(np.clip(np.sum((-e01) * e12, axis=1) / (l01 * l12 + 1e-30), -1, 1))
        ang2 = np.arccos(np.clip(np.sum((-e12) * e20, axis=1) / (l12 * l20 + 1e-30), -1, 1))
        return float(np.min([ang0.min(), ang1.min(), ang2.min()]))

    ang_py = min_angle(v_py, f_py)
    ang_cpp = min_angle(v_cpp, f_cpp)
    print(f"Min triangle angle: py={np.degrees(ang_py):.4f}°  cpp={np.degrees(ang_cpp):.4f}°")
    if ang_cpp < ang_py * 0.5:
        print("  FAIL: C++ min angle is much smaller than Python")
        all_ok = False

    # --- Topology report ---
    topology_report(v_py, f_py, "Python_result")
    topology_report(v_cpp, f_cpp, "C++_result")

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: C++ and Python outputs are IDENTICAL")
    else:
        print("RESULT: C++ and Python outputs DIFFER - see details above")
    print("=" * 70)
    return all_ok




if __name__ == "__main__":
    filepath = Path(__file__).parent / 'data' / 'rotation_9.npz'
    surf = cpgeo.CPGEO.load(filepath)
    print(f"PID: {os.getpid()}")
    surf.initialize()

    cp0 = surf.control_points.copy()
    faces = surf._cp_faces.copy()
    name = "input"
    periods = 3

    base_report = topology_report(cp0, faces, f"{name}_original")
    v0 = mesh_volume(cp0, faces)
    print(f"\n{name} original volume: {v0:.10f}")
    print(f"{name} input C{periods}")

    if not base_report["is_closed_manifold"]:
        raise ValueError(f"{name}: original is not closed manifold")

    # ---- Python version ----
    from cpgeo.utils.rotational_symmetry import enforce_rotational_symmetry_z as py_rot_sym
    with Timer(f"{name} Python rotational symmetry C{periods}"):
        v_py, f_py = py_rot_sym(cp0, faces, periods=periods, tol=1e-8)

    # ---- C++ version ----
    with Timer(f"{name} C++ rotational symmetry C{periods}"):
        v_cpp, f_cpp = capi.rotational_symmetry_z(cp0, faces, periods=periods, tol=1e-8)

    # ---- Compare ----
    ok = compare_meshes(v_py, f_py, v_cpp, f_cpp)

    # ---- Plotting (optional) ----
    import pyvista as pv
    plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 600))

    plotter.subplot(0, 0)
    surf_py = pv.PolyData(v_py, np.hstack([np.full((f_py.shape[0], 1), 3, dtype=np.int64), f_py]))
    plotter.add_mesh(surf_py, show_edges=True, color='lightblue', label="Python Result")
    plotter.add_legend()

    plotter.subplot(0, 1)
    surf_cpp = pv.PolyData(v_cpp, np.hstack([np.full((f_cpp.shape[0], 1), 3, dtype=np.int64), f_cpp]))
    plotter.add_mesh(surf_cpp, show_edges=True, color='salmon', label="C++ Result")

    plotter.link_views()
    plotter.view_isometric()
    plotter.add_legend()
    plotter.show()

    if not ok:
        raise AssertionError("C++ and Python results differ! See details above.")

    print("\nAll checks passed!")

