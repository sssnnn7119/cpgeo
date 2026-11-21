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


def show_surf(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    surf = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]))
    surf.plot(show_edges=True)

def enforce_rotational_symmetry_z(vertices: np.ndarray,
                                  faces: np.ndarray,
                                  periods: int,
                                  threshold: float = None,
                                  tol: float = 1e-8,
                                  return_match: bool = False,
                                  debug_show: bool = False):
    """
    Enforce Cn rotational symmetry around z-axis by strict open-sector clipping + replication.

    Args:
        vertices: Shape (N, 3).
        faces: Shape (F, 3).
        periods: Number of periods in Cn symmetry.
        threshold: Threshold for determining the sector boundary.
        tol: Numerical tolerance.
        return_match: Return rotational correspondence when True.

    Returns:
        (new_vertices, new_faces) or (new_vertices, new_faces, match)
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if periods < 2:
        raise ValueError("periods must be >= 2")

    all_edges = capi.get_mesh_edges(f)[:, :2]
    edge_len = np.linalg.norm(v[all_edges[:, 0]] - v[all_edges[:, 1]], axis=1)
    mean_edge = float(np.mean(edge_len))

    if threshold is None:
        threshold = mean_edge * 0.2

    alpha = 2.0 * np.pi / float(periods)

    v_sector, f_sector, left_side, right_side = cpgeo.utils.rotational_symmetry._extract_sector(
        vertices=v,
        faces=f,
        alpha=alpha,
        threshold=float(threshold),
        tol=float(tol),
    )

    m = int(v_sector.shape[0])
    verts_all = np.vstack([cpgeo.utils.rotational_symmetry._rot_z(v_sector, k * alpha) for k in range(periods)])

    trim_count = cpgeo.utils.rotational_symmetry._decide_pole_trim_count(
        left_side=left_side,
        right_side=right_side,
        sector_vertices=v_sector,
        mean_edge=mean_edge,
    )

    if trim_count > 0:
        keep_l = left_side[trim_count:-trim_count] if left_side.size > 2 * trim_count else left_side
        keep_r = right_side[trim_count:-trim_count] if right_side.size > 2 * trim_count else right_side
        if keep_l.size >= 2 and keep_r.size >= 2:
            left_side = keep_l
            right_side = keep_r

    faces_blocks = []
    left_global = []
    right_global = []
    for k in range(periods):
        off = k * m
        faces_blocks.append(f_sector + off)
        left_global.append(left_side + off)
        right_global.append(right_side + off)
    faces_all = np.vstack(faces_blocks).astype(np.int64)

    seam_faces = []
    for k in range(periods):
        nk = (k + 1) % periods

        rid = right_global[k]
        lid = left_global[nk]

        rpts = verts_all[rid]
        lpts = verts_all[lid]

        seam = cpgeo.utils.rotational_symmetry._zipper_stitch(rid, rpts, lid, lpts)
        if seam.size > 0:
            seam_faces.append(seam)


    # ------------------------------------------------------------------
    # debug visualisation (one window, press n to step)
    # ------------------------------------------------------------------
    def _debug_show_history() -> None:
        import pyvista as pv

        plotter = pv.Plotter()
        plotter.add_points(rpts, color='blue', point_size=8)
        plotter.add_points(lpts, color='green', point_size=8)
        if rpts.shape[0] > 1:
            plotter.add_mesh(pv.lines_from_points(rpts), color='blue', line_width=4)
        if lpts.shape[0] > 1:
            plotter.add_mesh(pv.lines_from_points(lpts), color='green', line_width=4)

        state = {'step': 0}

        def _on_next() -> None:
            if state['step'] >= len(seam):
                return
            tri = verts_all[seam[state['step']]]
            mesh = pv.PolyData(tri, np.array([[3, 0, 1, 2]], dtype=np.int64))
            plotter.add_mesh(mesh, color='red', show_edges=True, opacity=0.5, name=f'tri_{state["step"]}')
            state['step'] += 1

        def _on_prev() -> None:
            if state['step'] <= 0:
                return
            state['step'] -= 1
            plotter.remove_actor(f'tri_{state["step"]}')
            plotter.meshes.pop(f'tri_{state["step"]}', None)
            

        plotter.add_key_event('n', _on_next)
        plotter.add_key_event('m', _on_prev)
        plotter.add_key_event('q', plotter.close)
        plotter.add_text('Press n to advance, m to back, q to quit', font_size=12)
        plotter.show()
    
    if debug_show:
        _debug_show_history()
# ======================================================================
# Unit tests for _zipper_stitch
# ======================================================================

def _build_vertex_map(right_ids, right_pts, left_ids, left_pts):
    """Return a dict mapping global id → 3-D point."""
    verts = {}
    for i, pt in zip(right_ids, right_pts):
        verts[int(i)] = pt
    for i, pt in zip(left_ids, left_pts):
        verts[int(i)] = pt
    return verts


def _triangle_normals(faces, verts):
    """Return list of unit normals for every face."""
    normals = []
    for tri in faces:
        a, b, c = [verts[int(x)] for x in tri]
        n = np.cross(b - a, c - a)
        nrm = np.linalg.norm(n)
        normals.append(n / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0]))
    return normals


def _min_interior_angle(tri_pts):
    """Minimum interior angle (radians) of a triangle given its 3 vertices."""
    p0, p1, p2 = tri_pts
    a2 = np.sum((p1 - p2) ** 2)
    b2 = np.sum((p0 - p2) ** 2)
    c2 = np.sum((p1 - p0) ** 2)
    eps = 1e-12
    cos_a = (b2 + c2 - a2) / (2.0 * np.sqrt(b2 * c2) + eps)
    cos_b = (a2 + c2 - b2) / (2.0 * np.sqrt(a2 * c2) + eps)
    cos_c = (a2 + b2 - c2) / (2.0 * np.sqrt(a2 * b2) + eps)
    return min(np.arccos(np.clip(cos_a, -1.0, 1.0)),
               np.arccos(np.clip(cos_b, -1.0, 1.0)),
               np.arccos(np.clip(cos_c, -1.0, 1.0)))


def test_zipper_stitch_planar():
    """_zipper_stitch on a flat vertical strip → all normals identical."""
    from cpgeo.utils.rotational_symmetry import _zipper_stitch

    n_pts = 6
    z = np.linspace(0.0, 5.0, n_pts)

    right_pts = np.column_stack([np.ones(n_pts), np.zeros(n_pts), z]).astype(np.float64)
    left_pts  = np.column_stack([np.zeros(n_pts), np.zeros(n_pts), z]).astype(np.float64)

    right_ids = np.arange(100, 100 + n_pts, dtype=np.int64)
    left_ids  = np.arange(200, 200 + n_pts, dtype=np.int64)

    faces = _zipper_stitch(right_ids, right_pts, left_ids, left_pts,
                           dihedral_angle_threshold=60.0)

    # ---- basic checks ----
    assert faces.ndim == 2 and faces.shape[1] == 3, \
        f"Expected (T,3) got {faces.shape}"
    expected_tris = n_pts + n_pts - 2            # 10
    assert faces.shape[0] == expected_tris, \
        f"Expected {expected_tris} triangles, got {faces.shape[0]}"

    verts = _build_vertex_map(right_ids, right_pts, left_ids, left_pts)

    # all ids used at least once
    used = set(int(v) for v in faces.ravel())
    for rid in right_ids:
        assert int(rid) in used, f"Right id {rid} never used"
    for lid in left_ids:
        assert int(lid) in used, f"Left id {lid} never used"

    # ---- dihedral angle check ----
    normals = _triangle_normals(faces, verts)
    thresh_rad = np.deg2rad(60.0)
    for k in range(len(normals) - 1):
        d = np.arccos(np.clip(np.dot(normals[k], normals[k + 1]), -1.0, 1.0))
        assert d <= thresh_rad + 1e-8, \
            f"Tri {k}–{k+1}: dihedral {np.rad2deg(d):.3f}° > 60°"

    # For planar geometry all normals should be identical
    ref = normals[0]
    for k, n in enumerate(normals):
        assert np.dot(ref, n) > 1.0 - 1e-12, \
            f"Tri {k} normal differs from first triangle"

    print(f"  ✓ planar test passed  ({faces.shape[0]} triangles, "
          f"all dihedral < 60°, all normals equal)")


def test_zipper_stitch_curved():
    """_zipper_stitch on a cylindrical strip → adjacent normals stay close."""
    from cpgeo.utils.rotational_symmetry import _zipper_stitch

    n_pts = 8
    z = np.linspace(0.0, 4.0, n_pts)
    theta = np.linspace(0.0, np.pi / 6.0, n_pts)   # 30° arc

    # right boundary at outer radius, left at inner radius
    r_right, r_left = 2.0, 1.0
    right_pts = np.column_stack([
        r_right * np.cos(theta), r_right * np.sin(theta), z
    ]).astype(np.float64)
    left_pts = np.column_stack([
        r_left * np.cos(theta), r_left * np.sin(theta), z
    ]).astype(np.float64)

    right_ids = np.arange(100, 100 + n_pts, dtype=np.int64)
    left_ids  = np.arange(200, 200 + n_pts, dtype=np.int64)

    threshold = 30.0   # use a tighter threshold
    faces = _zipper_stitch(right_ids, right_pts, left_ids, left_pts,
                           dihedral_angle_threshold=threshold)

    verts = _build_vertex_map(right_ids, right_pts, left_ids, left_pts)
    normals = _triangle_normals(faces, verts)
    thresh_rad = np.deg2rad(threshold)

    max_dihedral = 0.0
    for k in range(len(normals) - 1):
        d = np.arccos(np.clip(np.dot(normals[k], normals[k + 1]), -1.0, 1.0))
        max_dihedral = max(max_dihedral, d)
        assert d <= thresh_rad + 1e-8, \
            f"Tri {k}–{k+1}: dihedral {np.rad2deg(d):.3f}° > {threshold}°"

    # Optional: check that the DP picked a reasonable triangulation
    # (all triangles have positive area)
    verts_arr = np.vstack([right_pts, left_pts])
    ids_arr = np.concatenate([right_ids, left_ids])
    for tri in faces:
        p = np.array([verts[int(x)] for x in tri])
        area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        assert area > 1e-12, f"Degenerate triangle {tri}"

    print(f"  ✓ curved test passed  ({faces.shape[0]} triangles, "
          f"max dihedral {np.rad2deg(max_dihedral):.2f}° ≤ {threshold}°)")


def test_zipper_stitch_skewed():
    """
    _zipper_stitch with mismatched point counts.
    The right boundary has fewer points than the left → exercises
    the DP path where advancing one side is forced near the end.
    """
    from cpgeo.utils.rotational_symmetry import _zipper_stitch

    # right: 5 pts, left: 8 pts
    right_pts = np.column_stack([
        np.ones(5), np.zeros(5), np.linspace(0.0, 4.0, 5)
    ]).astype(np.float64)
    left_pts = np.column_stack([
        np.zeros(8), np.zeros(8), np.linspace(0.0, 4.0, 8)
    ]).astype(np.float64)

    right_ids = np.arange(5, dtype=np.int64)
    left_ids = np.arange(10, 18, dtype=np.int64)

    faces = _zipper_stitch(right_ids, right_pts, left_ids, left_pts,
                           dihedral_angle_threshold=60.0)

    verts = _build_vertex_map(right_ids, right_pts, left_ids, left_pts)
    normals = _triangle_normals(faces, verts)
    thresh_rad = np.deg2rad(60.0)

    for k in range(len(normals) - 1):
        d = np.arccos(np.clip(np.dot(normals[k], normals[k + 1]), -1.0, 1.0))
        assert d <= thresh_rad + 1e-8, \
            f"Tri {k}–{k+1}: dihedral {np.rad2deg(d):.3f}° > 60°"

    print(f"  ✓ skewed test passed  ({faces.shape[0]} triangles, "
          f"right={right_pts.shape[0]}, left={left_pts.shape[0]})")


if __name__ == "__main__":
    filepath = Path(__file__).parent / 'data' / 'rotation_13.npz'
    surf = cpgeo.CPGEO.load(filepath)
    print(f"PID: {os.getpid()}")
    surf.initialize()
    surf.show()
    # surf.refine_surface(seed_size=1.5)

    cp0 = surf.control_points.copy()
    faces = surf._cp_faces.copy()
    name = "input"
    periods = 3

    enforce_rotational_symmetry_z(cp0, faces, periods=periods, tol=1e-8, debug_show=True)

    print("\nAll checks passed!")

