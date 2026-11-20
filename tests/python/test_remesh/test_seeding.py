from pathlib import Path

import numpy as np
import pytest

from cpgeo import capi


@pytest.fixture(scope="module")
def surface():
    data = Path(__file__).resolve().parents[3] / "src/cpp/tests"
    knots = np.loadtxt(data / "knots.txt", delimiter=",")
    controls = np.loadtxt(data / "control_points.txt", delimiter=",")
    tree = capi.space_tree_create(knots, capi.compute_thresholds(knots, 20))
    try:
        yield knots, controls, tree
    finally:
        capi.space_tree_destroy(tree)


@pytest.mark.parametrize("seed_size", [0.25, 1.0, 20.0])
def test_local_seeding_preserves_closed_sphere(surface, seed_size):
    knots, controls, tree = surface
    vertices, faces = capi.uniformly_mesh(knots, controls, tree, seed_size, max_iterations=0)
    assert len(vertices) >= 4
    assert len(faces) == 2 * len(vertices) - 4
    assert faces.min() >= 0 and faces.max() < len(vertices)
    assert len(np.unique(faces)) == len(vertices)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0)

    a, b, c = vertices[faces].transpose(1, 0, 2)
    orientation = np.einsum("ij,ij->i", a, np.cross(b, c))
    assert np.all(orientation > 0) or np.all(orientation < 0)

    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    assert len(np.unique(directed, axis=0)) == len(directed)
    edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)

    mapped = capi.map_points(vertices, tree, controls)
    lengths = np.linalg.norm(mapped[edges[:, 0]] - mapped[edges[:, 1]], axis=1)
    assert 0.5 * seed_size <= np.median(lengths) <= 1.5 * seed_size


@pytest.mark.parametrize("seed_size", [0.0, -1.0, np.nan, np.inf])
def test_invalid_seed_size(surface, seed_size):
    with pytest.raises(ValueError, match="seed_size"):
        capi.uniformly_mesh(*surface, seed_size=seed_size, max_iterations=0)


def test_spherical_triangulation_retains_close_vertices():
    # A small cluster near the pole produces projected triangles with area below 1e-10.
    i = np.arange(48)
    z = 1 - 2 * (i + 0.5) / len(i)
    angle = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z*z)
    sphere = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    cluster = np.array([[0, 0, 1], [2e-6, 0, 1], [0, 2e-6, 1], [5e-7, 5e-7, 1]])
    vertices = np.vstack((sphere, cluster / np.linalg.norm(cluster, axis=1, keepdims=True)))

    faces = capi.get_sphere_triangulation(vertices)
    assert len(np.unique(faces)) == len(vertices)
    assert len(faces) == 2 * len(vertices) - 4
    assert np.all(capi.get_mesh_edges(faces)[:, 2] == 2)
