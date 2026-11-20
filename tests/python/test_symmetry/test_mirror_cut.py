import numpy as np
import pytest

from cpgeo import capi, utils


def tetrahedron():
    vertices = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [-1, 0, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    return vertices, faces


def test_mirror_retries_empty_cut_and_preserves_winding():
    vertices, faces = tetrahedron()
    mirrored, triangles = utils.enforce_axial_symmetry(vertices, faces, threshold=2.0)
    assert len(mirrored) == 6
    edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    assert len(np.unique(edges, axis=0)) == len(edges)
    assert np.all(capi.get_mesh_edges(triangles)[:, 2] == 2)


def test_mirror_terminates_when_lower_threshold_cannot_join_two_loops(monkeypatch):
    vertices, faces = tetrahedron()
    vertices = np.vstack((vertices, vertices + [0, 3, 0]))
    faces = np.vstack((faces, faces + 4))
    extract = capi.extract_boundary_loops
    calls = 0

    def count_cuts(triangles):
        nonlocal calls
        calls += 1
        assert calls <= 2, "The mirror search retried an unchanged cut"
        return extract(triangles)

    monkeypatch.setattr(capi, "extract_boundary_loops", count_cuts)
    with pytest.raises(ValueError, match="one simple boundary loop"):
        utils.enforce_axial_symmetry(vertices, faces)
