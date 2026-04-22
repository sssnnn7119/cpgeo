

import numpy as np

from .. import capi


def show_surf(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv
    mesh = pv.PolyData(vertices, np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]))

    plotter = pv.Plotter()
    def callback(picked_point, picker):
        point_id = picker.GetPointId()
        if point_id < 0: return
        point = mesh.points[point_id]
        print(f"Node Index: {point_id}, Coordinates: {point}")
        plotter.add_point_labels([point], [f"ID: {point_id}"], point_size=20, font_size=18, name="picked_label", always_visible=True)

    plotter.enable_point_picking(callback=callback, show_message=True, use_picker=True, show_point=True, color='red', picker='point')
    plotter.add_mesh(mesh, show_edges=True)
    plotter.show()


def _rot_z(points: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return points @ r.T


def _zipper_stitch(right_ids: np.ndarray,
                   right_pts: np.ndarray,
                   left_ids: np.ndarray,
                   left_pts: np.ndarray) -> np.ndarray:
    if right_ids.size < 2 or left_ids.size < 2:
        return np.zeros((0, 3), dtype=np.int64)

    rid, rpt, lid, lpt = _order_seam_chains(right_ids, right_pts, left_ids, left_pts)

    tris = []
    i = 0
    j = 0
    nr = rid.size
    nl = lid.size

    while i < nr - 1 or j < nl - 1:
        can_i = i < nr - 1
        can_j = j < nl - 1

        if can_i and can_j:
            ci = np.linalg.norm(rpt[i + 1] - lpt[j])
            cj = np.linalg.norm(rpt[i] - lpt[j + 1])
            if ci <= cj:
                tris.append([rid[i], rid[i + 1], lid[j]])
                i += 1
            else:
                tris.append([rid[i], lid[j + 1], lid[j]])
                j += 1
        elif can_i:
            tris.append([rid[i], rid[i + 1], lid[j]])
            i += 1
        else:
            tris.append([rid[i], lid[j + 1], lid[j]])
            j += 1

    return np.asarray(tris, dtype=np.int64)


def _decide_pole_trim_count(left_side: np.ndarray,
                            right_side: np.ndarray,
                            sector_vertices: np.ndarray,
                            mean_edge: float) -> int:
    left = np.asarray(left_side, dtype=np.int64)
    right = np.asarray(right_side, dtype=np.int64)
    if left.size < 6 or right.size < 6:
        return 0

    ds = float(np.linalg.norm(sector_vertices[left[0]] - sector_vertices[right[0]]))
    dn = float(np.linalg.norm(sector_vertices[left[-1]] - sector_vertices[right[-1]]))
    span = max(ds, dn) / max(float(mean_edge), 1e-12)

    if span > 3.0 and left.size >= 12 and right.size >= 12:
        return 2
    if span > 1.8:
        return 1
    return 0


def _order_seam_chains(right_ids: np.ndarray,
                       right_pts: np.ndarray,
                       left_ids: np.ndarray,
                       left_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep the boundary-chain order from _extract_sector to avoid seam crossing.
    rid = right_ids.astype(np.int64)
    rpt = right_pts
    lid = left_ids.astype(np.int64)
    lpt = left_pts

    d_same = np.linalg.norm(rpt[0] - lpt[0]) + np.linalg.norm(rpt[-1] - lpt[-1])
    d_flip = np.linalg.norm(rpt[0] - lpt[-1]) + np.linalg.norm(rpt[-1] - lpt[0])
    if d_flip < d_same:
        lid = lid[::-1]
        lpt = lpt[::-1]

    # Use the nearest endpoint pair as seam start anchor.
    if np.linalg.norm(rpt[0] - lpt[0]) > np.linalg.norm(rpt[0] - lpt[-1]):
        lid = lid[::-1]
        lpt = lpt[::-1]

    return rid, rpt, lid, lpt


def _face_components_by_edges(faces: np.ndarray) -> list[np.ndarray]:
    n_faces = int(faces.shape[0])
    if n_faces == 0:
        return []

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for i, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(i)

    adj = [[] for _ in range(n_faces)]
    for lst in edge_to_faces.values():
        if len(lst) < 2:
            continue
        for i in range(len(lst)):
            fi = lst[i]
            for j in range(i + 1, len(lst)):
                fj = lst[j]
                adj[fi].append(fj)
                adj[fj].append(fi)

    visited = np.zeros(n_faces, dtype=bool)
    comps: list[np.ndarray] = []
    for i in range(n_faces):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        cur = []
        while stack:
            u = stack.pop()
            cur.append(u)
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    stack.append(w)
        comps.append(np.asarray(cur, dtype=np.int64))

    return comps


def _tri_area_sum(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))


def _mean_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    ff = np.asarray(faces, dtype=np.int64)
    if ff.size == 0:
        return 1.0
    edges = np.vstack([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]])
    lens = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if lens.size == 0:
        return 1.0
    return float(np.mean(lens))


def _axis_triangle_intersection(vertices: np.ndarray,
                                tri: np.ndarray,
                                eps: float = 1e-12) -> tuple[bool, np.ndarray]:
    t = np.asarray(tri, dtype=np.int64)
    p = vertices[t]
    m = np.array([
        [p[0, 0], p[1, 0], p[2, 0]],
        [p[0, 1], p[1, 1], p[2, 1]],
        [1.0, 1.0, 1.0],
    ], dtype=np.float64)
    rhs = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    det = float(np.linalg.det(m))
    if abs(det) <= eps:
        return False, np.zeros((3,), dtype=np.float64)

    w = np.linalg.solve(m, rhs)
    if np.any(w < -1e-8) or np.any(w > 1.0 + 1e-8):
        return False, np.zeros((3,), dtype=np.float64)

    q = w[0] * p[0] + w[1] * p[1] + w[2] * p[2]
    return True, q.astype(np.float64)


def _find_axis_pole_triangles(vertices: np.ndarray,
                              faces: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray] | None,
                                                          tuple[np.ndarray, np.ndarray] | None]:
    hits = []
    for tri in np.asarray(faces, dtype=np.int64):
        ok, q = _axis_triangle_intersection(vertices, tri)
        if not ok:
            continue
        hits.append((float(q[2]), np.asarray(tri, dtype=np.int64), q))

    if not hits:
        return None, None

    south = min(hits, key=lambda x: x[0])
    north = max(hits, key=lambda x: x[0])
    return (south[1], south[2]), (north[1], north[2])


def _virtual_anchor_from_triangle(tri_points: np.ndarray,
                                  axis_hit: np.ndarray,
                                  phase: float,
                                  alpha: float,
                                  target_r: float) -> np.ndarray:
    # Shrink triangle toward axis, then rotate anchor into current sector interior.
    r_tri = np.linalg.norm(tri_points[:, :2], axis=1)
    r_src = max(1e-9, float(np.min(r_tri)) * 0.55)
    r = min(max(r_src, 0.60 * target_r), 1.30 * target_r)
    theta = float(phase + 0.5 * alpha)
    return np.array([r * np.cos(theta), r * np.sin(theta), float(axis_hit[2])], dtype=np.float64)


def _recover_removed_faces_for_interior_holes(f_sector: np.ndarray,
                                              f_comp: np.ndarray,
                                              interior_boundary_vertices: set[int]) -> np.ndarray:
    if not interior_boundary_vertices:
        return np.zeros((0, 3), dtype=np.int64)

    comp_set = {tuple(row.tolist()) for row in np.sort(np.asarray(f_comp, dtype=np.int64), axis=1)}
    removed = []
    for tri in np.asarray(f_sector, dtype=np.int64):
        key = tuple(np.sort(tri).tolist())
        if key not in comp_set:
            removed.append(np.asarray(tri, dtype=np.int64))
    if not removed:
        return np.zeros((0, 3), dtype=np.int64)

    removed = np.asarray(removed, dtype=np.int64)
    n = int(removed.shape[0])
    used = np.zeros(n, dtype=bool)
    frontier = set(int(x) for x in interior_boundary_vertices)

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if used[i]:
                continue
            tri = removed[i]
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            if (a in frontier) or (b in frontier) or (c in frontier):
                used[i] = True
                frontier.add(a)
                frontier.add(b)
                frontier.add(c)
                changed = True

    if not np.any(used):
        return np.zeros((0, 3), dtype=np.int64)
    return removed[used]


def _seal_loops_constrained(faces: np.ndarray,
                            loops: list[np.ndarray] | None = None,
                            max_loop_size: int = 256) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if loops is None:
        loops = [np.asarray(lp, dtype=np.int64) for lp in capi.extract_boundary_loops(faces)]
    if not loops:
        return faces

    edges = capi.get_mesh_edges(faces)
    edge_count: dict[tuple[int, int], int] = {}
    for a, b, c in edges:
        u = int(a)
        v = int(b)
        edge_count[(min(u, v), max(u, v))] = int(c)

    def can_add_triangle(a: int, b: int, c: int) -> bool:
        for x, y in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y))
            if edge_count.get(key, 0) >= 2:
                return False
        return True

    def add_triangle(a: int, b: int, c: int):
        for x, y in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y))
            edge_count[key] = edge_count.get(key, 0) + 1

    add = []
    for loop in loops:
        ring = np.asarray(loop, dtype=np.int64)
        if ring.size < 3 or ring.size > max_loop_size:
            continue

        poly = [int(x) for x in ring.tolist()]
        while len(poly) > 2:
            n = len(poly)
            ear_found = False
            for i in range(n):
                a = poly[(i - 1) % n]
                b = poly[i]
                c = poly[(i + 1) % n]
                if a == b or b == c or a == c:
                    continue
                if not can_add_triangle(a, b, c):
                    continue
                add.append([a, b, c])
                add_triangle(a, b, c)
                poly.pop(i)
                ear_found = True
                break
            if not ear_found:
                break

    if not add:
        return faces
    return np.vstack([faces, np.asarray(add, dtype=np.int64)])


def _split_seam_paths(loop: np.ndarray,
                      vertices: np.ndarray,
                      preferred_south_idx: int | None = None,
                      preferred_north_idx: int | None = None,
                      preferred_south_point: np.ndarray | None = None,
                      preferred_north_point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    boundary = np.asarray(loop, dtype=np.int64)
    z_b = vertices[boundary, 2]


    def _pick_from_preferred(preferred_idx: int | None,
                             preferred_point: np.ndarray | None) -> int | None:
        if preferred_idx is not None:
            pos = np.flatnonzero(boundary == int(preferred_idx))
            if pos.size > 0:
                return int(pos[0])
            if preferred_point is None and 0 <= int(preferred_idx) < vertices.shape[0]:
                preferred_point = vertices[int(preferred_idx)]
        if preferred_point is not None:
            d = np.linalg.norm(vertices[boundary] - np.asarray(preferred_point, dtype=np.float64)[None, :], axis=1)
            return int(np.argmin(d))
        return None

    i_s = _pick_from_preferred(preferred_south_idx, preferred_south_point)
    i_n = _pick_from_preferred(preferred_north_idx, preferred_north_point)

    if i_s <= i_n:
        path1 = boundary[i_s:i_n + 1]
        path2 = np.concatenate([boundary[i_s::-1], boundary[:i_n - 1:-1]])
    else:
        path1 = np.concatenate([boundary[i_s:], boundary[:i_n + 1]])
        path2 = boundary[i_s:i_n - 1:-1]

    if path1.size < 2 or path2.size < 2:
        return None, None
    return path1, path2


def _loop_perimeter(vertices: np.ndarray, loop: np.ndarray) -> float:
    ring = np.asarray(loop, dtype=np.int64)
    pts = vertices[ring]
    nxt = np.roll(pts, -1, axis=0)
    return float(np.sum(np.linalg.norm(nxt - pts, axis=1)))


def _rotate_cycle(ids: np.ndarray, start_idx: int) -> np.ndarray:
    if ids.size == 0:
        return ids
    k = int(start_idx) % int(ids.size)
    return np.concatenate([ids[k:], ids[:k]])


def _zipper_stitch_closed(loop_a: np.ndarray,
                          loop_b: np.ndarray,
                          vertices: np.ndarray) -> np.ndarray:
    a = np.asarray(loop_a, dtype=np.int64)
    b = np.asarray(loop_b, dtype=np.int64)
    if a.size < 3 or b.size < 3:
        return np.zeros((0, 3), dtype=np.int64)

    pa = vertices[a]
    pb = vertices[b]

    start_b = int(np.argmin(np.linalg.norm(pb - pa[0], axis=1)))
    b = _rotate_cycle(b, start_b)
    pb = vertices[b]

    d_same = np.linalg.norm(pa[0] - pb[0]) + np.linalg.norm(pa[1 % a.size] - pb[1 % b.size])
    d_flip = np.linalg.norm(pa[0] - pb[0]) + np.linalg.norm(pa[1 % a.size] - pb[-1])
    if d_flip < d_same:
        b = b[::-1]
        b = _rotate_cycle(b, int(np.argmin(np.linalg.norm(vertices[b] - pa[0], axis=1))))
        pb = vertices[b]

    tris = []
    i = 0
    j = 0
    na = int(a.size)
    nb = int(b.size)
    step_i = 0
    step_j = 0

    while step_i < na or step_j < nb:
        can_i = step_i < na
        can_j = step_j < nb
        ai = int(a[i % na])
        bj = int(b[j % nb])

        if can_i and can_j:
            ai1 = int(a[(i + 1) % na])
            bj1 = int(b[(j + 1) % nb])
            ci = np.linalg.norm(vertices[ai1] - vertices[bj])
            cj = np.linalg.norm(vertices[ai] - vertices[bj1])
            if ci <= cj:
                tris.append([ai, ai1, bj])
                i += 1
                step_i += 1
            else:
                tris.append([ai, bj1, bj])
                j += 1
                step_j += 1
        elif can_i:
            ai1 = int(a[(i + 1) % na])
            tris.append([ai, ai1, bj])
            i += 1
            step_i += 1
        else:
            bj1 = int(b[(j + 1) % nb])
            tris.append([ai, bj1, bj])
            j += 1
            step_j += 1

    return np.asarray(tris, dtype=np.int64)


def _sample_loop_ids(loop: np.ndarray, periods: int) -> np.ndarray:
    ring = np.asarray(loop, dtype=np.int64)
    p = int(periods)
    n = int(ring.size)
    if p <= 0 or n == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == p:
        return ring.copy()
    idx = np.floor(np.arange(p, dtype=np.float64) * float(n) / float(p)).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    return ring[idx]


def _build_polar_candidate_ring(vertices: np.ndarray,
                                loop: np.ndarray,
                                periods: int,
                                mean_edge: float,
                                far_factor: float = 2.0) -> tuple[np.ndarray, bool]:
    verts = np.asarray(vertices, dtype=np.float64)
    ids = _sample_loop_ids(loop, periods)
    if ids.size != int(periods):
        return np.zeros((0, 3), dtype=np.float64), False

    cand = verts[ids].copy()
    r = np.linalg.norm(cand[:, :2], axis=1)
    target_r = max(float(mean_edge), 1e-9)
    far = r > float(far_factor) * target_r
    if np.any(far):
        theta = np.arctan2(cand[far, 1], cand[far, 0])
        cand[far, 0] = target_r * np.cos(theta)
        cand[far, 1] = target_r * np.sin(theta)

    return cand, bool(np.any(far))


def _insert_polar_quality_ring(vertices: np.ndarray,
                               faces: np.ndarray,
                               loop: np.ndarray,
                               periods: int,
                               mean_edge: float,
                               upward: bool) -> tuple[np.ndarray, np.ndarray]:
    ring = np.asarray(loop, dtype=np.int64)
    if ring.size < 3:
        return vertices, faces

    verts = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    pts = verts[ring]

    z_loop = float(np.mean(pts[:, 2]))
    r_loop = float(np.mean(np.linalg.norm(pts[:, :2], axis=1)))
    if r_loop <= 1e-12:
        return verts, f

    cand_ring, pulled = _build_polar_candidate_ring(
        vertices=verts,
        loop=ring,
        periods=periods,
        mean_edge=float(mean_edge),
        far_factor=2.0,
    )

    if cand_ring.shape[0] == int(periods) and pulled:
        # If selected polar candidates are too far from z-axis, pull them first.
        new_ring = cand_ring
        z_new = float(np.mean(new_ring[:, 2]))
        dz = max(0.35 * mean_edge, 0.08 * max(r_loop, 1e-9))
    else:
        r_new = max(0.35 * r_loop, 0.8 * mean_edge)
        dz = max(0.35 * mean_edge, 0.08 * r_loop)
        z_new = z_loop + (dz if upward else -dz)
        ang0 = float(np.arctan2(pts[0, 1], pts[0, 0]))
        alphas = ang0 + np.arange(periods, dtype=np.float64) * (2.0 * np.pi / float(periods))
        new_ring = np.column_stack([r_new * np.cos(alphas), r_new * np.sin(alphas), np.full(periods, z_new)])

    center = np.array([[0.0, 0.0, z_new + (0.5 * dz if upward else -0.5 * dz)]], dtype=np.float64)

    base = int(verts.shape[0])
    ring_ids = np.arange(base, base + periods, dtype=np.int64)
    center_id = int(base + periods)
    verts_new = np.vstack([verts, new_ring, center])

    bridge = _zipper_stitch_closed(ring, ring_ids, verts_new)

    cap = []
    for i in range(periods):
        a = int(ring_ids[i])
        b = int(ring_ids[(i + 1) % periods])
        if upward:
            cap.append([a, b, center_id])
        else:
            cap.append([a, center_id, b])
    cap = np.asarray(cap, dtype=np.int64)

    f_new = np.vstack([f, bridge, cap])
    return verts_new, f_new


def _mesh_polar_holes(vertices: np.ndarray,
                              faces: np.ndarray,
                              periods: int,
                              mean_edge: float) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    loops = capi.extract_boundary_loops(f)
    if not loops:
        return verts, f

    if len(loops) == 1:
        loops_use = [np.asarray(loops[0], dtype=np.int64)]
    else:
        z_mean = [float(np.mean(verts[np.asarray(lp, dtype=np.int64), 2])) for lp in loops]
        low_i = int(np.argmin(z_mean))
        high_i = int(np.argmax(z_mean))
        loops_use = [np.asarray(loops[low_i], dtype=np.int64), np.asarray(loops[high_i], dtype=np.int64)]

    for lp in loops_use:
        perim = _loop_perimeter(verts, lp)
        # "Large" hole heuristic: enough boundary span that direct fan cap tends to make bad triangles.
        is_large = (lp.size >= periods + 1) and (perim > 2.0 * float(periods) * float(mean_edge))
        is_small = (lp.size <= periods) and (perim < 0.3 * float(periods) * float(mean_edge))
        if is_large:
            zc = float(np.mean(verts[lp, 2]))
            up = zc >= float(np.median(verts[:, 2]))
            verts, f = _insert_polar_quality_ring(verts, f, lp, periods=periods, mean_edge=float(mean_edge), upward=up)

        elif is_small:
            zc = float(np.mean(verts[lp, 2]))
            up = zc >= float(np.median(verts[:, 2]))

            # Remove this tiny ring layer: delete all faces touching the ring vertices.
            touch = np.any(np.isin(f, lp), axis=1)
            f_cut = f[~touch]
            if f_cut.shape[0] == 0:
                continue
            f = f_cut

            # Recompute boundary loops and pick the new polar ring on the same side.
            loops_new = capi.extract_boundary_loops(f)
            if not loops_new:
                continue
            if len(loops_new) == 1:
                new_lp = np.asarray(loops_new[0], dtype=np.int64)
            else:
                z_new = [float(np.mean(verts[np.asarray(x, dtype=np.int64), 2])) for x in loops_new]
                idx = int(np.argmax(z_new) if up else np.argmin(z_new))
                new_lp = np.asarray(loops_new[idx], dtype=np.int64)

            # Seal the new ring explicitly.
            f = _seal_loops_constrained(f, loops=[new_lp], max_loop_size=512)

        else:
            # Neither large nor small: seal directly with generic small-loop filler.
            f = _seal_loops_constrained(f, loops=None, max_loop_size=16)


    if f.shape[0] == 0:
        return verts, f

    # Compact vertex indexing once at the end so removed tiny-ring vertices are truly deleted.
    used = np.unique(f.reshape(-1))
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    verts = verts[used]
    f = remap[f]

    return verts, f


def _extract_sector(vertices: np.ndarray,
                    faces: np.ndarray,
                    alpha: float,
                    threshold: float,
                    tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)

    r_xy = np.linalg.norm(v[:, :2], axis=1)
    theta = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2.0 * np.pi)

    radial_ref = float(np.mean(r_xy[r_xy > tol])) if np.any(r_xy > tol) else 1.0
    ang_eps = max(float(tol), float(threshold) / max(radial_ref, 1e-12))
    ang_eps = min(0.24 * alpha, ang_eps)
    ang_eps = max(1e-6, ang_eps)

    axis_eps = max(float(tol), 0.5 * float(threshold))

    south_pole_tri, north_pole_tri = _find_axis_pole_triangles(v, f)

    phase_samples = max(8, int(np.ceil(2.0 * np.pi / alpha)))
    best = None
    best_key = None

    for phase in np.linspace(0.0, alpha, num=phase_samples, endpoint=False):
        eps_try = float(ang_eps)
        for _ in range(18):
            theta_shift = np.mod(theta - phase, 2.0 * np.pi)
            keep = ((theta_shift > eps_try) & (theta_shift < (alpha - eps_try))) | (r_xy <= axis_eps)
            ids = np.flatnonzero(keep)
            if ids.size == 0:
                eps_try *= 0.5
                continue

            keep_mask = np.zeros(v.shape[0], dtype=bool)
            keep_mask[ids] = True
            f_sector = f[np.all(keep_mask[f], axis=1)]
            if f_sector.shape[0] == 0:
                eps_try *= 0.5
                continue

            comps = _face_components_by_edges(f_sector)
            if not comps:
                eps_try *= 0.5
                continue

            side_band = max(3.0 * eps_try, 5.0 * float(tol))
            best_comp_faces = None
            best_comp_key = None
            for comp in comps:
                f_comp = f_sector[comp]
                comp_ids = np.unique(f_comp)
                t_comp = theta_shift[comp_ids]
                touch_left = int(np.sum(t_comp <= side_band))
                touch_right = int(np.sum((alpha - t_comp) <= side_band))
                touch_both = int(touch_left >= 2 and touch_right >= 2)
                area = _tri_area_sum(v, f_comp)
                key = (touch_both, area)
                if best_comp_key is None or key > best_comp_key:
                    best_comp_key = key
                    best_comp_faces = f_comp

            if best_comp_faces is None:
                eps_try *= 0.5
                continue

            ids_comp = np.unique(best_comp_faces)
            remap = np.full(v.shape[0], -1, dtype=np.int64)
            remap[ids_comp] = np.arange(ids_comp.size, dtype=np.int64)
            f_local = remap[best_comp_faces]
            v_local = v[ids_comp]

            loops = capi.extract_boundary_loops(f_local)
            if len(loops) == 0:
                eps_try *= 0.5
                continue

            theta_local = np.mod(np.arctan2(v_local[:, 1], v_local[:, 0]) - phase, 2.0 * np.pi)
            seam_idx = -1
            seam_score = None
            for i, loop in enumerate(loops):
                ring = np.asarray(loop, dtype=np.int64)
                lcnt = int(np.sum(theta_local[ring] <= side_band))
                rcnt = int(np.sum((alpha - theta_local[ring]) <= side_band))
                score = (min(lcnt, rcnt), lcnt + rcnt, ring.size)
                if seam_score is None or score > seam_score:
                    seam_score = score
                    seam_idx = i

            if seam_idx < 0:
                eps_try *= 0.5
                continue

            seam_loop = np.asarray(loops[seam_idx], dtype=np.int64)
            mean_edge_local = _mean_edge_length(v_local, f_local)

            pref_s_idx = None
            pref_n_idx = None
            pref_s_point = None
            pref_n_point = None

            if south_pole_tri is not None:
                tri_s, q_s = south_pole_tri
                local_s = [int(remap[int(g)]) for g in tri_s.tolist() if remap[int(g)] >= 0]
                if local_s:
                    rr = np.linalg.norm(v_local[local_s, :2], axis=1)
                    pref_s_idx = int(local_s[int(np.argmin(rr))])
                else:
                    pref_s_point = _virtual_anchor_from_triangle(
                        tri_points=v[tri_s],
                        axis_hit=q_s,
                        phase=float(phase),
                        alpha=float(alpha),
                        target_r=float(mean_edge_local),
                    )

            if north_pole_tri is not None:
                tri_n, q_n = north_pole_tri
                local_n = [int(remap[int(g)]) for g in tri_n.tolist() if remap[int(g)] >= 0]
                if local_n:
                    rr = np.linalg.norm(v_local[local_n, :2], axis=1)
                    pref_n_idx = int(local_n[int(np.argmin(rr))])
                else:
                    pref_n_point = _virtual_anchor_from_triangle(
                        tri_points=v[tri_n],
                        axis_hit=q_n,
                        phase=float(phase),
                        alpha=float(alpha),
                        target_r=float(mean_edge_local),
                    )

            path1, path2 = _split_seam_paths(
                seam_loop,
                v_local,
                preferred_south_idx=pref_s_idx,
                preferred_north_idx=pref_n_idx,
                preferred_south_point=pref_s_point,
                preferred_north_point=pref_n_point,
            )
            if path1 is None or path2 is None:
                eps_try *= 0.5
                continue

            mean1 = float(np.mean(theta_local[path1]))
            mean2 = float(np.mean(theta_local[path2]))
            if mean1 <= mean2:
                left = path1.astype(np.int64)
                right = path2.astype(np.int64)
            else:
                left = path2.astype(np.int64)
                right = path1.astype(np.int64)

            if left.size < 2 or right.size < 2:
                eps_try *= 0.5
                continue

            interior_loops = [np.asarray(loops[i], dtype=np.int64) for i in range(len(loops)) if i != seam_idx]
            if interior_loops:
                # Do not triangulate-fill interior holes. Restore removed original faces instead.
                interior_global = set()
                for ring in interior_loops:
                    for x in ring.tolist():
                        interior_global.add(int(ids_comp[int(x)]))

                f_patch = _recover_removed_faces_for_interior_holes(
                    f_sector=f_sector,
                    f_comp=best_comp_faces,
                    interior_boundary_vertices=interior_global,
                )

                if f_patch.size > 0:
                    old_ids = ids_comp
                    old_set = set(int(x) for x in old_ids.tolist())
                    extra = sorted({int(vv) for vv in np.unique(f_patch) if int(vv) not in old_set})
                    if extra:
                        ids_comp = np.concatenate([old_ids, np.asarray(extra, dtype=np.int64)])

                    remap = np.full(v.shape[0], -1, dtype=np.int64)
                    remap[ids_comp] = np.arange(ids_comp.size, dtype=np.int64)

                    best_comp_faces = np.vstack([best_comp_faces, f_patch])
                    f_local = remap[best_comp_faces]
                    v_local = v[ids_comp]
                    # Keep the original seam_loop/left/right; only fill interior holes.

            area_local = _tri_area_sum(v_local, f_local)
            candidate_key = (
                len(interior_loops),
                abs(int(left.size) - int(right.size)),
                -area_local,
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best = (v_local, f_local.astype(np.int64), left.astype(np.int64), right.astype(np.int64))

            break

    if best is not None:
        return best

    raise RuntimeError("Failed to extract a clean rotational sector. Consider decreasing threshold/tol.")


def enforce_rotational_symmetry_z(vertices: np.ndarray,
                                  faces: np.ndarray,
                                  periods: int,
                                  threshold: float = None,
                                  tol: float = 1e-8,
                                  return_match: bool = False):
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

    v_sector, f_sector, left_side, right_side = _extract_sector(
        vertices=v,
        faces=f,
        alpha=alpha,
        threshold=float(threshold),
        tol=float(tol),
    )

    m = int(v_sector.shape[0])
    verts_all = np.vstack([_rot_z(v_sector, k * alpha) for k in range(periods)])

    trim_count = _decide_pole_trim_count(
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

        seam = _zipper_stitch(rid, rpts, lid, lpts)
        if seam.size > 0:
            seam_faces.append(seam)

    if seam_faces:
        faces_all = np.vstack([faces_all] + seam_faces)

    verts_all, faces_all = _mesh_polar_holes(
        vertices=verts_all,
        faces=faces_all,
        periods=int(periods),
        mean_edge=mean_edge,
    )

    faces_all = capi.optimize_mesh_by_edge_flipping(vertices=verts_all, faces=faces_all)

    if return_match:
        # match[k, i] is the index of base-sector vertex i in sector k
        match = (np.arange(periods, dtype=np.int64)[:, None] * m) + np.arange(m, dtype=np.int64)[None, :]
        return verts_all, faces_all.astype(np.int64), match

    return verts_all, faces_all.astype(np.int64)

