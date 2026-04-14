import numpy as np

from .axissymmetry import _clip_mesh_by_halfspaces, _weld_vertices


def _rotation_matrix_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _mesh_topology_stats(faces: np.ndarray):
    if faces.shape[0] == 0:
        return 0, 0

    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    edges = np.vstack([e01, e12, e20])
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)

    boundary_edges = int(np.sum(edge_counts == 1))
    nonmanifold_edges = int(np.sum(edge_counts > 2))
    return boundary_edges, nonmanifold_edges


def _vertex_component_count(vertices: np.ndarray, faces: np.ndarray):
    n = int(vertices.shape[0])
    if n == 0 or faces.shape[0] == 0:
        return 0

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
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
    return comp


def _enforce_rotational_symmetry_z_vertex_projection(vertices: np.ndarray,
                                                     faces: np.ndarray,
                                                     periods: int):
    """
    Fallback path: preserve topology, only move vertices by periodic nearest-neighbor averaging
    in a canonical wedge frame.
    """
    alpha = 2.0 * np.pi / float(periods)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    theta = np.arctan2(v[:, 1], v[:, 0])
    theta_mod = np.mod(theta + 2.0 * np.pi, 2.0 * np.pi)
    sector = np.floor(theta_mod / alpha).astype(np.int64)
    sector = np.clip(sector, 0, periods - 1)

    r_pos = [_rotation_matrix_z(alpha * float(k)) for k in range(periods)]
    r_neg = [_rotation_matrix_z(-alpha * float(k)) for k in range(periods)]

    canonical = np.zeros_like(v)
    ids = []
    for k in range(periods):
        idx = np.where(sector == k)[0]
        ids.append(idx)
        if idx.shape[0] > 0:
            canonical[idx] = v[idx] @ r_neg[k].T

    trees = []
    try:
        from scipy.spatial import cKDTree
        for k in range(periods):
            pts = canonical[ids[k]]
            trees.append(cKDTree(pts) if pts.shape[0] > 0 else None)
    except Exception:
        trees = [None] * periods

    out = v.copy()
    for k in range(periods):
        idx = ids[k]
        if idx.shape[0] == 0:
            continue
        query = canonical[idx]
        avg = np.zeros_like(query)
        used = 0

        for j in range(periods):
            j_idx = ids[j]
            if j_idx.shape[0] == 0:
                continue
            pts = canonical[j_idx]

            if trees[j] is not None:
                _, nn = trees[j].query(query, k=1)
                avg += pts[nn]
            else:
                diff = query[:, None, :] - pts[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                nn = np.argmin(dist2, axis=1)
                avg += pts[nn]
            used += 1

        if used == 0:
            continue

        avg /= float(used)
        out[idx] = avg @ r_pos[k].T

    return out, f


def enforce_rotational_symmetry_z(vertices: np.ndarray,
                                  faces: np.ndarray,
                                  periods: int,
                                  tol: float = 1e-8):
    """
    Enforce Cn rotational symmetry around z-axis by sector clipping + replication.

    Args:
        vertices: Shape (N, 3).
        faces: Shape (F, 3).
        periods: Cn symmetry order, n >= 2.
        tol: Clipping and welding tolerance.

    Returns:
        (new_vertices, new_faces)
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3).")
    if int(periods) != periods or periods < 2:
        raise ValueError("periods must be an integer >= 2.")

    periods = int(periods)
    half = np.pi / float(periods)
    c = float(np.cos(half))
    s = float(np.sin(half))

    # Keep a centered wedge: -half <= theta <= +half.
    n1 = np.array([s, c, 0.0], dtype=np.float64)
    n2 = np.array([s, -c, 0.0], dtype=np.float64)
    v_sector, f_sector = _clip_mesh_by_halfspaces(v, f, planes=[(n1, 0.0), (n2, 0.0)], tol=tol)

    alpha = 2.0 * np.pi / float(periods)
    v_blocks = []
    f_blocks = []
    for k in range(periods):
        r = _rotation_matrix_z(alpha * float(k))
        vk = v_sector @ r.T
        base = sum(x.shape[0] for x in v_blocks)
        v_blocks.append(vk)
        f_blocks.append(f_sector + base)

    verts_all = np.vstack(v_blocks)
    faces_all = np.vstack(f_blocks)
    verts_all, faces_all = _weld_vertices(verts_all, faces_all, weld_tol=tol)

    boundary_edges, nonmanifold_edges = _mesh_topology_stats(faces_all)
    n_comp = _vertex_component_count(verts_all, faces_all)
    if boundary_edges == 0 and nonmanifold_edges == 0 and n_comp == 1:
        return verts_all, faces_all

    # Fallback keeps original topology and enforces periodic geometry in vertex positions.
    v_fb, f_fb = _enforce_rotational_symmetry_z_vertex_projection(v, f, periods)
    boundary_edges_fb, nonmanifold_edges_fb = _mesh_topology_stats(f_fb)
    n_comp_fb = _vertex_component_count(v_fb, f_fb)
    if boundary_edges_fb > 0 or nonmanifold_edges_fb > 0 or n_comp_fb != 1:
        raise ValueError(
            "Output mesh is not a closed manifold single component after enforcing rotational symmetry. "
            "Try adjusting tolerance or input quality."
        )

    return v_fb, f_fb
