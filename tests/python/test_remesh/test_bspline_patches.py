import numpy as np
import pytest
from scipy.interpolate import BSpline

from cpgeo import CPGEO, capi


@pytest.fixture(scope="module")
def model():
    i = np.arange(80)
    z = 1 - 2 * (i + 0.5) / len(i)
    angle = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z*z)
    sphere = np.column_stack((r*np.cos(angle), r*np.sin(angle), z))
    return CPGEO(sphere * [1.2, 1.0, 0.9] + [10, 20, -30],
                 capi.get_sphere_triangulation(sphere), knot_influence_num=16)


def test_patch_fit_shares_edges_and_matches_map3(model):
    fit = model.fit_bspline_patches(0.02)
    n = fit.poles.shape[1]
    knots = np.r_[np.zeros(3), np.linspace(0, 1, n-2), np.ones(3)]
    uv = np.random.default_rng(1127).uniform(size=(500, 2))
    basis = [BSpline.design_matrix(t, knots, 3).toarray() for t in uv.T]
    edges = {}
    for face, patch in enumerate(fit.poles):
        xyz = np.empty((len(uv), 3))
        axis, side = divmod(face, 2)
        sign = 1 - 2*side
        xyz[:, axis] = sign
        xyz[:, (axis+1) % 3] = sign * (2*uv[:, 0]-1)
        xyz[:, (axis+2) % 3] = 2*uv[:, 1]-1
        expected = model.map3(xyz / np.linalg.norm(xyz, axis=1)[:, None])
        fitted = np.einsum("ki,kj,ijd->kd", *basis, patch)
        assert np.linalg.norm(fitted - expected, axis=1).max() < 0.02
        for edge in (patch[0], patch[-1], patch[:, 0], patch[:, -1]):
            canonical = min(edge.tobytes(), edge[::-1].tobytes())
            edges[canonical] = edges.get(canonical, 0) + 1
    assert len(edges) == 12 and set(edges.values()) == {2}
    assert 0 < fit.rms_error <= fit.max_error <= 0.02
    assert 0 <= fit.max_seam_angle_deg < 90


def test_failed_tolerance_is_reported(model):
    with pytest.raises(ValueError, match="exceeds tolerance"):
        model.fit_bspline_patches(1e-12, max_spans=2)


def test_step_export(model, tmp_path):
    pytest.importorskip("OCP")
    target = tmp_path / "surface.stp"
    model.fit_bspline_patches(0.02).write_step(target)
    assert "MANIFOLD_SOLID_BREP" in target.read_text()


def test_banded_atlas_shares_seams_and_hits_both_poles(model):
    from cpgeo.bspline import axial_chart

    # Non-antipodal clusters exercise the fractional stereographic transform.
    rng = np.random.default_rng(42)
    north = np.array([0., 0., 1.])
    south = np.array([0.2, 0., -np.sqrt(0.96)])
    directions = np.repeat([north, south], 100, axis=0) + rng.normal(0, 0.015, (200, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    scales = np.tile(np.geomspace(.002, .15, 100), 2)
    frame, bands = axial_chart(directions, scales)
    assert bands > 1
    assert axial_chart(np.vstack((np.eye(3), -np.eye(3))), np.ones(6)) is None

    model.initialize()
    poles, metrics = capi.fit_bspline_patches(model._space_tree, model.control_points, 8,
                                             axial_chart=(frame, bands))
    assert len(poles) == 4*bands + 2
    edges = {}
    for patch in poles:
        for edge in (patch[0], patch[-1], patch[:, 0], patch[:, -1]):
            key = min(edge.tobytes(), edge[::-1].tobytes())
            edges[key] = edges.get(key, 0) + 1
    assert len(edges) == 2*len(poles) and set(edges.values()) == {2}

    # The cap centers map to the recovered poles.
    axis, tangent = frame[:3], frame[3:6]
    inverse, r_n, r_s = frame[9:]
    mapped_south = ((inverse**2-1)*axis + 2*inverse*tangent) / (inverse**2+1)
    knots = np.r_[np.zeros(3), np.linspace(0, 1, 9), np.ones(3)]
    mid = BSpline.design_matrix([.5], knots, 3).toarray()[0]
    for patch, direction in zip(poles[-2:], (axis, mapped_south)):
        point = np.einsum('i,j,ijd->d', mid, mid, patch)
        assert np.linalg.norm(point-model.map3(direction[None])[0]) <= metrics[0] + 1e-10
    # Check the +X side chart at the middle of a band as well.
    local = .5
    patch = 4 * (bands // 2)
    axial_fraction = (bands // 2 + local) / bands
    radius = np.exp(axial_fraction*np.log(r_n) - (1-axial_fraction)*np.log(r_s))
    den = 1 + inverse*radius
    expected = ((den**2-radius**2)*axis + 2*radius*den*tangent) / (den**2+radius**2)
    point = np.einsum('i,j,ijd->d', mid, mid, poles[patch])
    assert np.linalg.norm(point-model.map3(expected[None])[0]) <= metrics[0] + 1e-10


def test_adaptive_patches_match_source_and_share_partial_edges(model, tmp_path):
    from cpgeo.adaptive_patches import _axes, _locate

    fit = model.fit_bspline_patches(0.01, target_nodes=5)
    assert len(fit.domains) > 6
    assert sum(count for _, _, count in fit.domains) == len(model._knots)
    rng = np.random.default_rng(123)
    uv = rng.uniform(size=(120, 2))
    edges = {}
    for index, ((face, box, _), breaks, poles) in enumerate(zip(fit.domains, fit.breaks, fit.poles)):
        world_uv = box[:, 0] + uv * np.diff(box).ravel()
        q = capi.patch_reference(face, world_uv, fit.frame)
        sides, local = _locate(q, fit.frame)
        assert np.all(sides == face)
        np.testing.assert_allclose(local, world_uv, atol=1e-12)
        basis = [BSpline.design_matrix(uv[:, k], np.pad(t, 3, mode="edge"), 3).toarray()
                 for k, t in enumerate(breaks)]
        xyz = np.einsum("ki,kj,ijd->kd", *basis, poles)
        assert np.linalg.norm(model.map3(q) - xyz, axis=1).max() < 0.01
        for k, axis in enumerate(_axes(face)):
            for end in (0, 1):
                corner = np.zeros(3)
                corner[face//2] = 1 - face % 2
                corner[_axes(face)[1-k]] = box[1-k, end]
                key = (axis, *np.delete(corner, axis))
                row = 0 if end == 0 else -1
                edge_poles = poles[row] if k == 1 else poles[:, row]
                spline = BSpline(np.pad(breaks[k], 3, mode="edge"), edge_poles, 3)
                edges.setdefault(key, []).append((*box[k], spline))
    partial = False
    for group in edges.values():
        for i, (a, b, s) in enumerate(group):
            coverage = 0.
            for j, (c, d, other) in enumerate(group):
                lo, hi = max(a, c), min(b, d)
                if i == j or lo >= hi:
                    continue
                t = np.linspace(lo, hi, 37)
                np.testing.assert_allclose(s((t-a)/(b-a)), other((t-c)/(d-c)), atol=1e-10, rtol=0)
                coverage += hi-lo
                partial |= (hi-lo < b-a)
            assert coverage == pytest.approx(b-a)
    assert partial
    pytest.importorskip("OCP")
    fit.write_step(tmp_path / "adaptive.stp")
