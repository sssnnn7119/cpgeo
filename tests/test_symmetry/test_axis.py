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
        self.start_time = time.time()  # user + system time
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        elapsed = end_time - self.start_time
        print(f"{self.name} took {elapsed:.4f} seconds.")


def topology_report(vertices: np.ndarray, faces: np.ndarray, name: str):
    v = np.asarray(vertices)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"{name}: vertices must be shape (N, 3).")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"{name}: faces must be shape (F, 3).")
    if f.size == 0:
        raise ValueError(f"{name}: faces is empty.")

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
        if k == "name":
            continue
        print(f"  {k}: {val}")

    return report

if __name__ == "__main__":
    
    # data = np.load('tests/testdata.npz')

    # cps = data['control_points'].T
    # faces = data['mesh_elements']
    # surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)
    # surf.initialize()
    filepath = Path(__file__).parent / 'data' / 'axis.npz'
    surf = cpgeo.CPGEO.load(filepath)
    print(os.getpid())
    

    # np.savetxt('src/cpp/tests/control_points.txt', surf.control_points, delimiter=',')
    # np.savetxt('src/cpp/tests/knots.txt', surf._knots, delimiter=',')
    # np.savetxt('src/cpp/tests/cp_faces.txt', surf._cp_faces, fmt='%d', delimiter=',')

    surf.initialize()

    # surf.show_control_points()

    cp0 = surf.control_points.copy()
    faces = surf._cp_faces.copy()

    base_report = topology_report(cp0, faces, "original")

    with Timer("planar symmetry (yz-plane) + topology check"):
        v_ax, f_ax = utils.enforce_axial_symmetry(cp0, faces, plane="yz", keep_positive=True, tol=1e-8)
        ax_report = topology_report(v_ax, f_ax, "axial_symmetry")
        surf1 = cpgeo.CPGEO(control_points=v_ax, cp_faces=f_ax)
        surf1.initialize()
        surf1.show_control_points()
        # surf1.refine_surface()
        # surf1.show_control_points()


    if not base_report["is_closed_manifold"]:
        raise RuntimeError("Original mesh is not a closed manifold; topology baseline failed.")
    if not ax_report["is_closed_manifold"]:
        raise RuntimeError("Axial symmetry output is not a closed manifold.")

    

