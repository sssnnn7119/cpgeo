import numpy as np

def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n <= 0:
        raise ValueError("Axis vector must be non-zero.")
    return vec / n

def _parse_plane(plane: str):
    key = plane.lower().strip()
    if key == "yz":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0
    if key == "xz":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64), 1
    if key == "xy":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 2
    raise ValueError("plane must be one of {'xy', 'xz', 'yz'}.")


def _reflection_matrix_from_normal(n: np.ndarray) -> np.ndarray:
    n = _normalize(np.asarray(n, dtype=np.float64))
    return np.eye(3, dtype=np.float64) - 2.0 * np.outer(n, n)


def _compute_axial_match_indices(vertices: np.ndarray,
                                 plane: str,
                                 plane_offset: float) -> np.ndarray:
    n, axis_id = _parse_plane(plane)
    d = float(plane_offset)

    shift = np.zeros(3, dtype=np.float64)
    shift[axis_id] = d
    r = _reflection_matrix_from_normal(n)

    v = np.asarray(vertices, dtype=np.float64)
    v_ref = ((v - shift[None, :]) @ r.T) + shift[None, :]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(v)
        _, idx = tree.query(v_ref, k=1)
    except Exception:
        diff = v_ref[:, None, :] - v[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(dist2, axis=1)

    ids = np.arange(v.shape[0], dtype=np.int64)
    return np.stack([ids, idx.astype(np.int64)], axis=1)


def _clip_polygon_halfspace(poly: np.ndarray,
                            n: np.ndarray,
                            d: float,
                            tol: float) -> np.ndarray:
    if poly.shape[0] == 0:
        return poly

    out = []
    m = poly.shape[0]
    val = poly @ n - d

    for i in range(m):
        s = poly[i]
        e = poly[(i + 1) % m]
        vs = val[i]
        ve = val[(i + 1) % m]
        s_in = vs >= -tol
        e_in = ve >= -tol

        if s_in and e_in:
            out.append(e)
            continue

        if s_in and (not e_in):
            denom = np.dot(n, e - s)
            if abs(denom) > 1e-15:
                t = (d - np.dot(n, s)) / denom
                t = min(max(t, 0.0), 1.0)
                out.append(s + t * (e - s))
            continue

        if (not s_in) and e_in:
            denom = np.dot(n, e - s)
            if abs(denom) > 1e-15:
                t = (d - np.dot(n, s)) / denom
                t = min(max(t, 0.0), 1.0)
                out.append(s + t * (e - s))
            out.append(e)

    if len(out) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _clip_mesh_by_halfspaces(vertices: np.ndarray,
                             faces: np.ndarray,
                             planes,
                             tol: float):
    verts_out = []
    faces_out = []

    for tri in faces:
        poly = vertices[tri].astype(np.float64)
        for n, d in planes:
            poly = _clip_polygon_halfspace(poly, n, d, tol)
            if poly.shape[0] < 3:
                break

        if poly.shape[0] < 3:
            continue

        base = len(verts_out)
        verts_out.extend(poly.tolist())
        for i in range(1, poly.shape[0] - 1):
            faces_out.append([base, base + i, base + i + 1])

    if len(faces_out) == 0:
        raise ValueError("No faces left after clipping. Try another axis/offset or tolerance.")

    return np.asarray(verts_out, dtype=np.float64), np.asarray(faces_out, dtype=np.int64)


def _weld_vertices(vertices: np.ndarray, faces: np.ndarray, weld_tol: float):
    if vertices.shape[0] == 0:
        return vertices, faces

    scale = 1.0 / max(weld_tol, 1e-15)
    keys = np.round(vertices * scale).astype(np.int64)
    _, unique_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    new_vertices = vertices[unique_idx]
    new_faces = inverse[faces]

    # Remove degenerate triangles.
    deg = (new_faces[:, 0] == new_faces[:, 1]) | (new_faces[:, 1] == new_faces[:, 2]) | (new_faces[:, 0] == new_faces[:, 2])
    new_faces = new_faces[~deg]

    if new_faces.shape[0] == 0:
        raise ValueError("All faces became degenerate after welding.")

    # Remove duplicate faces ignoring orientation.
    sig = np.sort(new_faces, axis=1)
    _, keep_idx = np.unique(sig, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    new_faces = new_faces[keep_idx]

    return new_vertices, new_faces






def enforce_axial_symmetry(vertices: np.ndarray,
                           faces: np.ndarray,
                           plane: str = "yz",
                           plane_offset: float = 0.0,
                           keep_positive: bool = True,
                           tol: float = 1e-8,
                           return_match: bool = False):
    """
    Enforce strict planar symmetry by: keep one half-space -> mirror -> rebuild faces.

    Args:
        vertices: Shape (N, 3).
        faces: Shape (F, 3).
        plane: Symmetry plane in {'xy', 'xz', 'yz'}.
        plane_offset: Plane offset along its normal. For 'yz', the plane is x=plane_offset.
        keep_positive: Keep normal-positive side if True, else keep negative side.
        tol: Cut and welding tolerance.

    Returns:
        (new_vertices, new_faces) when return_match is False.
        (new_vertices, new_faces, match_indices) when return_match is True,
        where match_indices has shape (V, 2).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3).")

    n, axis_id = _parse_plane(plane)
    d = float(plane_offset)
    n_cut = n if keep_positive else -n
    d_cut = d if keep_positive else -d

    v_half, f_half = _clip_mesh_by_halfspaces(v, f, planes=[(n_cut, d_cut)], tol=tol)

    r = _reflection_matrix_from_normal(n)
    shift = np.zeros(3, dtype=np.float64)
    shift[axis_id] = d
    v_centered = v_half - shift[None, :]
    v_mirror = (v_centered @ r.T) + shift[None, :]

    n_half = v_half.shape[0]
    verts_all = np.vstack([v_half, v_mirror])
    # Mirror changes orientation, so flip winding of mirrored faces.
    f_mirror = (f_half + n_half)[:, [0, 2, 1]]
    faces_all = np.vstack([f_half, f_mirror])

    verts_all, faces_all = _weld_vertices(verts_all, faces_all, weld_tol=tol)

    if not return_match:
        return verts_all, faces_all

    match = _compute_axial_match_indices(verts_all, plane=plane, plane_offset=plane_offset)
    return verts_all, faces_all, match



