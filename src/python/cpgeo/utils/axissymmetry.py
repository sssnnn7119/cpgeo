import numpy as np
from .. import capi


def show_surf(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv
    mesh = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]))
    mesh.plot(show_edges=True)


def _enforce_axial_symmetry_yz_core(vertices: np.ndarray, 
                                    faces: np.ndarray,
                                    threshold: float = None):
    """
    Enforce yz axial symmetry by: keep one half-space -> mirror -> rebuild faces.

    Args:
        vertices: Shape (N, 3).
        faces: Shape (F, 3).
        threshold: Threshold for determining the symmetry plane.

    Returns:
        (new_vertices, new_faces)
    """

    if threshold is None:
        all_edges = capi.get_mesh_edges(faces)
        edge_length = np.linalg.norm(vertices[all_edges[:, 0]] - vertices[all_edges[:, 1]], axis=1)
        threshold = float(np.mean(edge_length)) * 0.1

    while True:
        # get the points where x > 0 and their faces fully inside the positive half-space
        pos_mask = np.flatnonzero(vertices[:, 0] > threshold)

        faces_pos = faces[np.all(np.isin(faces, pos_mask), axis=1)]
        pos_mask = np.unique(faces_pos)
        v_pos = vertices[pos_mask]

        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[pos_mask] = np.arange(pos_mask.size, dtype=np.int64)
        faces_pos_local = remap[faces_pos]

        # mirror the positive points to get the negative points
        v_neg = v_pos.copy()
        v_neg[:, 0] *= -1.0

        # combine the positive and negative points and faces
        verts_all = np.vstack([v_pos, v_neg])
        faces_all = np.vstack([faces_pos_local, faces_pos_local + v_pos.shape[0]])

        # seal the cut by adding faces between the positive and negative points on the boundary
        boundary_pos = capi.extract_boundary_loops(faces_pos_local)

        # if there are boundaries, we need to reduce the threshold and try again, until we get a clean cut
        if len(boundary_pos) > 1:
            threshold *= 0.5
            continue

        boundary_pos = boundary_pos[0]
        break

    boundary = np.asarray(boundary_pos, dtype=np.int64)
    next_boundary = np.roll(boundary, -1)
    offset = np.int64(v_pos.shape[0])

    # collect all directed edges from faces_all for fast existence tests
    directed_edges = np.vstack([
        faces_all[:, [0, 1]],
        faces_all[:, [1, 2]],
        faces_all[:, [2, 0]],
    ])

    packed_edges = (directed_edges[:, 0].astype(np.int64) << 32) | directed_edges[:, 1].astype(np.int64)
    packed_boundary = (boundary << 32) | next_boundary
    edge_exists = np.isin(packed_boundary, packed_edges)

    front_faces = np.column_stack([boundary, next_boundary, boundary + offset])
    back_faces = np.column_stack([next_boundary, next_boundary + offset, boundary + offset])
    front_faces_flipped = np.column_stack([next_boundary, boundary, boundary + offset])
    back_faces_flipped = np.column_stack([next_boundary + offset, next_boundary, boundary + offset])

    faces_cut = np.empty((boundary.shape[0] * 2, 3), dtype=np.int64)
    faces_cut[0::2] = np.where(edge_exists[:, None], front_faces_flipped, front_faces)
    faces_cut[1::2] = np.where(edge_exists[:, None], back_faces_flipped, back_faces)

    faces_all = np.vstack([faces_all, faces_cut])

    # refine faces
    faces_all = capi.optimize_mesh_by_edge_flipping(vertices=verts_all,
                                                    faces=faces_all,)

    return verts_all, faces_all


def enforce_axial_symmetry(vertices: np.ndarray,
                           faces: np.ndarray,
                           plane: str = "yz",
                           threshold: float = None):
    """
    Enforce strict planar symmetry by: keep one half-space -> mirror -> rebuild faces.

    Args:
        vertices: Shape (N, 3).
        faces: Shape (F, 3).
        plane: Symmetry plane in {'xy', 'xz', 'yz'}.
        threshold: Threshold for determining the symmetry plane.

    Returns:
        (new_vertices, new_faces)
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    
    
    if plane == "yz":
        verts_all, faces_all = _enforce_axial_symmetry_yz_core(v, f, threshold=threshold)

    elif plane == "xz":
        v = v[:, [1, 0, 2]]
        faces_all, verts_all = _enforce_axial_symmetry_yz_core(v, f, threshold=threshold)
        verts_all = verts_all[:, [1, 0, 2]]

    elif plane == "xy":
        v = v[:, [2, 1, 0]]
        faces_all, verts_all = _enforce_axial_symmetry_yz_core(v, f, threshold=threshold)
        verts_all = verts_all[:, [2, 1, 0]]

    return verts_all, faces_all



