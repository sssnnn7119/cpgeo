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
