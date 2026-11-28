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


def test_tight_cluster_away_from_pole_preserves_spherical_connectivity():
    # A distant insertion must not create overlapping Delaunay cavities inside
    # a previously inserted cluster. This scale occurs on mirrored cylinders.
    i = np.arange(80)
    z = 1 - 2 * (i + 0.5) / len(i)
    angle = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z*z)
    sphere = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    cluster = [0.2, -0.6, -0.77] + 1e-9 * np.random.default_rng(334).normal(size=(60, 3))
    vertices = np.vstack((cluster / np.linalg.norm(cluster, axis=1, keepdims=True), sphere))

    faces = capi.get_sphere_triangulation(vertices)
    assert len(faces) == 2 * len(vertices) - 4
    np.testing.assert_array_equal(np.unique(faces), np.arange(len(vertices)))
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    assert len(np.unique(directed, axis=0)) == len(directed)
    assert np.all(capi.get_mesh_edges(faces)[:, 2] == 2)
    a, b, c = vertices[faces].transpose(1, 0, 2)
    assert np.all(np.einsum("ij,ij->i", a, np.cross(b - a, c - a)) < 0)


def test_smoothing_decreases_edge_loss_within_volume_tolerance(surface):
    knots, controls, tree = surface
    initial, faces = capi.uniformly_mesh(knots, controls, tree, seed_size=4, max_iterations=0)
    smoothed, _ = capi.uniformly_mesh(knots, controls, tree, seed_size=4, max_iterations=1)
    # The outer remesher may flip edges after smoothing. Compare the objective
    # on the same connectivity the Newton iteration used.
    edges = capi.get_mesh_edges(faces)[:, :2]

    def edge_loss(points):
        mapped = capi.map_points(points, tree, controls)
        lengths = np.linalg.norm(mapped[edges[:, 0]] - mapped[edges[:, 1]], axis=1)
        return np.sum((lengths + 1e-8) ** 4)

    def volume(points):
        a, b, c = points[faces].transpose(1, 0, 2)
        return abs(np.einsum("ij,ij->", a, np.cross(b, c))) / 6

    np.testing.assert_allclose(np.linalg.norm(smoothed, axis=1), 1.0)
    assert edge_loss(smoothed) < edge_loss(initial)
    assert abs(volume(smoothed) / volume(initial) - 1.0) < 0.05


def test_patch_sweep_moves_seeds_only_once_and_preserves_anchors(surface):
    knots, controls, tree = surface
    fixed_count = 5

    def remesh(iterations):
        return capi.uniformly_mesh(knots, controls, tree, seed_size=1.0,
                                   max_iterations=iterations, fixed_count=fixed_count)

    initial, faces = remesh(0)
    once, _ = remesh(1)
    larger_budget, _ = remesh(4)
    assert len(once) - fixed_count > 64
    np.testing.assert_array_equal(once[:fixed_count], knots[:fixed_count])
    assert np.linalg.norm(once - initial) > 1e-6
    np.testing.assert_allclose(larger_budget, once, rtol=0, atol=1e-12)
    a, b, c = once[faces].transpose(1, 0, 2)
    determinants = np.einsum('ij,ij->i', a, np.cross(b, c))
    assert np.all(determinants > 0) or np.all(determinants < 0)


def test_smoothing_preserves_mapped_face_orientation_across_overlapping_patches():
    i = np.arange(120)
    z = 1 - 2 * (i + 0.5) / len(i)
    angle = i * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z*z)
    knots = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    radial_scale = 1 + 0.6 * np.random.default_rng(23).normal(size=(len(i), 1))
    controls = knots * radial_scale * [2, 2, 4]
    tree = capi.space_tree_create(knots, capi.compute_thresholds(knots, 20))
    try:
        seeds, faces = capi.uniformly_mesh(knots, controls, tree, 0.55, 0)
        smoothed, _ = capi.uniformly_mesh(knots, controls, tree, 0.55, 1)

        def normals(points):
            a, b, c = capi.map_points(points, tree, controls)[faces].transpose(1, 0, 2)
            return np.cross(b-a, c-a)

        # Compare identical faces before the optional final edge flips.
        assert np.all(np.einsum("ij,ij->i", normals(seeds), normals(smoothed)) > 0)
    finally:
        capi.space_tree_destroy(tree)
