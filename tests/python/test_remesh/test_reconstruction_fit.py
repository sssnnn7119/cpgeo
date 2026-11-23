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
