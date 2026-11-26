import numpy as np

from cpgeo import CPGEO, capi


def test_fit_agrees_with_dense_least_squares():
    # Include both poles and a constant coordinate (zero initial residual).
    rng = np.random.default_rng(2026)
    knots = rng.normal(size=(48, 3))
    knots /= np.linalg.norm(knots, axis=1, keepdims=True)
    knots[:2] = [[0, 0, 1], [0, 0, -1]]
    targets = rng.normal(size=knots.shape)
    targets[:, 2] = 7.0
    thresholds = capi.compute_thresholds(knots, 8)
    tree = capi.space_tree_create(knots, thresholds)
    try:
        # Assemble W independently from the compactly supported basis.
        dist = np.linalg.norm(knots[:, None] - knots[None, :], axis=2)
        t = np.minimum(dist / thresholds[None, :], 1.0)
        weights = (1 - t)**4 * (1 + 4*t + 10*t**2 + 20*t**3)
        weights /= weights.sum(axis=1, keepdims=True)
        for alpha in (0.0, 0.1, 1.0):
            fitted = capi.fit_reconstructed_control_points(tree, targets, alpha)
            a = np.vstack([weights, np.sqrt(alpha) * np.eye(len(knots))])
            b = np.vstack([targets, np.sqrt(alpha) * targets])
            expected = np.linalg.lstsq(a, b, rcond=None)[0]
            np.testing.assert_allclose(fitted, expected, atol=2e-7, rtol=1e-7)
            np.testing.assert_allclose(capi.map_points(knots, tree, fitted), weights @ fitted, atol=1e-10)
            np.testing.assert_array_equal(fitted[:, 2], targets[:, 2])

            # Remove the prescribed columns explicitly in the dense reference.
            fixed = targets[:7] + [0.5, -1.0, 2.0]
            pinned_fit = capi.fit_reconstructed_control_points(tree, targets, alpha, fixed)
            np.testing.assert_array_equal(pinned_fit[:7], fixed)
            expected_free = np.linalg.lstsq(a[:, 7:], b - a[:, :7] @ fixed, rcond=None)[0]
            np.testing.assert_allclose(pinned_fit[7:], expected_free, atol=2e-7, rtol=1e-7)
    finally:
        capi.space_tree_destroy(tree)


def test_refine_surface_matches_old_surface_samples():
    # Keep the two seeding runs identical while exercising the real reference
    # mesh rebuild and the default post-processing through the public API.
    i = np.arange(120)
    z = 1 - 2 * (i + 0.5) / len(i)
    radius = np.sqrt(1 - z*z)
    angle = i * np.pi * (3 - np.sqrt(5))
    sphere = np.column_stack([radius*np.cos(angle), radius*np.sin(angle), z])
    faces = capi.get_sphere_triangulation(sphere)
    controls = sphere * [1.0, 1.4, 2.0]
    baseline = CPGEO(controls, faces.copy())
    fitted = CPGEO(controls, faces.copy())
    baseline.refine_surface(0.4, 0, matching_alpha=None)
    fitted.refine_surface(0.4, 0)
    np.testing.assert_array_equal(baseline._cp_faces, fitted._cp_faces)
    np.testing.assert_allclose(baseline._knots, fitted._knots)
    targets = baseline.control_points
    before = np.linalg.norm(baseline.map3(baseline._knots) - targets)
    after = np.linalg.norm(fitted.map3(fitted._knots) - targets)
    assert after**2 + 0.1 * np.linalg.norm(fitted.control_points - targets)**2 < before**2

    fixed_ids = np.array([117, 6, 42])
    original = fitted.control_points[fixed_ids].copy()
    free = np.ones(len(fitted._knots), dtype=bool)
    free[fixed_ids] = False
    seeds = np.concatenate((fitted._knots[fixed_ids], fitted._knots[free]))
    remeshed, _ = fitted.uniformly_mesh(seeds, 0.4, 1, fixed_count=len(fixed_ids))
    np.testing.assert_array_equal(remeshed[:len(fixed_ids)], seeds[:len(fixed_ids)])
    for alpha in (0.1, None):
        fixed_ids = fitted.refine_surface(0.4, 1, matching_alpha=alpha, fixed_control_indices=fixed_ids)
        assert fitted.control_points[fixed_ids].tobytes() == original.tobytes()


def test_coarse_spherical_knots_and_face_writeback():
    # An octahedron has too few anchors for the second harmonic chart.
    points = np.vstack((np.eye(3), -np.eye(3)))
    triangles = np.array([[x, y, z] for x in (0, 3) for y in (1, 4) for z in (2, 5)])
    storage = np.zeros((len(triangles), 6), dtype=np.int64)
    faces = storage[:, ::2]
    faces[:] = triangles
    faces[::2] = faces[::2, ::-1]
    model = CPGEO(points, faces, knot_influence_num=3)
    model.initialize()
    knots = model._knots
    np.testing.assert_allclose(np.linalg.norm(knots, axis=1), 1.0)
    a, b, c = knots[faces].transpose(1, 0, 2)
    det = np.einsum('ij,ij->i', a, np.cross(b, c))
    assert np.all(det > 0)
    angles = 2 * np.arctan2(det, 1 + (a*b + b*c + c*a).sum(axis=1))
    np.testing.assert_allclose(angles.sum(), 4 * np.pi)
    np.testing.assert_array_equal(storage[:, 1::2], 0)
