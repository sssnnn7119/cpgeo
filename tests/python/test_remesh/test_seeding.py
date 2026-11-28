import os
from pathlib import Path
import subprocess
import sys

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


def test_patch_rounds_repeat_smoothing_and_preserve_anchors(surface, capfd):
    knots, controls, tree = surface
    fixed_count = 5

    def remesh(iterations):
        return capi.uniformly_mesh(knots, controls, tree, seed_size=1.0,
                                   max_iterations=iterations, fixed_count=fixed_count)

    initial, faces = remesh(0)
    once, _ = remesh(1)
    capfd.readouterr()
    larger_budget, final_faces = remesh(4)
    assert capfd.readouterr().out.count("Smoothing sweep:") == 4
    assert len(once) - fixed_count > 64
    np.testing.assert_array_equal(once[:fixed_count], knots[:fixed_count])
    assert np.linalg.norm(once - initial) > 1e-6
    np.testing.assert_array_equal(larger_budget[:fixed_count], knots[:fixed_count])
    assert np.linalg.norm(larger_budget - once) > 1e-6
    assert np.all(capi.get_mesh_edges(final_faces)[:, 2] == 2)
    a, b, c = once[faces].transpose(1, 0, 2)
    determinants = np.einsum('ij,ij->i', a, np.cross(b, c))
    assert np.all(determinants > 0) or np.all(determinants < 0)


def test_patch_workers_match_serial_sweep(tmp_path):
    # Separate processes initialize OpenMP with the requested worker budget.
    data = Path(__file__).resolve().parents[3] / "src/cpp/tests"
    script = """
from pathlib import Path
import sys
import numpy as np
from cpgeo import capi

data = Path(sys.argv[1])
knots = np.loadtxt(data / "knots.txt", delimiter=",")
controls = np.loadtxt(data / "control_points.txt", delimiter=",")
tree = capi.space_tree_create(knots, capi.compute_thresholds(knots, 20))
try:
    vertices, faces = capi.uniformly_mesh(knots, controls, tree, 1.0, 2, fixed_count=5)
    np.savez(sys.argv[2], vertices=vertices, faces=faces)
finally:
    capi.space_tree_destroy(tree)
"""
    for workers in (1, 4):
        subprocess.run([sys.executable, "-c", script, str(data), str(tmp_path / f"{workers}.npz")],
                       env={**os.environ, "OMP_NUM_THREADS": str(workers)}, check=True)
    with np.load(tmp_path / "1.npz") as serial, np.load(tmp_path / "4.npz") as parallel:
        np.testing.assert_array_equal(parallel["vertices"], serial["vertices"])
        np.testing.assert_array_equal(parallel["faces"], serial["faces"])


@pytest.fixture
def icosahedron_knots():
    phi = (1 + np.sqrt(5)) / 2
    knots = np.array([np.roll([0., a, b], k) for a in (-1., 1.)
                      for b in (-phi, phi) for k in range(3)])
    knots /= np.linalg.norm(knots, axis=1, keepdims=True)
    return knots @ np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.], [-0.48, 0.64, 0.6]])


def test_local_volume_limit_uses_all_faces_incident_on_the_free_vertex(icosahedron_knots):
    knots = icosahedron_knots
    seeds = knots.copy()
    seeds[-1] += 0.6 * np.array([1., 0.65, -0.5])
    seeds[-1] /= np.linalg.norm(seeds[-1])
    tree = capi.space_tree_create(knots, capi.compute_thresholds(knots, 8))
    try:
        initial, faces = capi.uniformly_mesh(seeds, knots, tree, 5., 0, fixed_count=11)
        smoothed, _ = capi.uniformly_mesh(seeds, knots, tree, 5., 1, fixed_count=11)
        assert len(initial) == len(smoothed) == 12
        np.testing.assert_array_equal(smoothed[:11], initial[:11])
        patch_faces = faces[(faces == 11).any(axis=1)]

        def volume6(points, triangles):
            a, b, c = points[triangles].transpose(1, 0, 2)
            return np.einsum("ij,ij->", a, np.cross(b-a, c-a))

        # The former whole-mesh bound allowed this star to change by 7.6%.
        change = abs(volume6(smoothed, patch_faces) / volume6(initial, patch_faces) - 1)
        assert 0.04 < change <= 0.05 + 1e-12
        assert abs(volume6(smoothed, faces) / volume6(initial, faces) - 1) < 0.02
    finally:
        capi.space_tree_destroy(tree)


def test_patch_converges_to_symmetric_spring_equilibrium(icosahedron_knots):
    # With eleven fixed vertices of a regular icosahedron, symmetry makes the
    # unperturbed twelfth vertex stationary for the five identical springs.
    knots = icosahedron_knots
    seeds = knots.copy()
    seeds[-1] += [0.24, 0.16, -0.12]
    seeds[-1] /= np.linalg.norm(seeds[-1])
    tree = capi.space_tree_create(knots, capi.compute_thresholds(knots, 8))
    try:
        mapped = capi.map_points(knots, tree, knots)
        spacing = np.linalg.norm(mapped[1:] - mapped[0], axis=1).min()
        result, faces = capi.uniformly_mesh(seeds, knots, tree, spacing, 1, fixed_count=11)
        assert len(faces) == 20
        np.testing.assert_array_equal(result[:-1], knots[:-1])
        # Independently check the legacy stopping rule by finite differences
        # in the chart opposite the initial free vertex, using its five springs.
        neighbors = np.unique(faces[(faces == 11).any(axis=1)])
        anchors = mapped[neighbors[neighbors != 11]]
        pole = -seeds[-1]
        tangent = np.cross([0., 0., 1.], pole)
        tangent /= np.linalg.norm(tangent)
        frame = np.array([tangent, np.cross(pole, tangent), pole])

        def step_norm(point):
            x, y, z = frame @ point
            uv = 2 * np.array([x, y]) / (1 - z)
            h = 1e-4
            def energy(i, j):
                u, v = uv + h * np.array([i, j])
                t = 1 / (4 + u*u + v*v)
                q = np.array([4*u*t, 4*v*t, 1-8*t]) @ frame
                xyz = capi.map_points(q[None], tree, knots)[0]
                return np.sum(np.linalg.norm(anchors - xyz, axis=1)**4)
            center = energy(0, 0)
            dx, dy = energy(1, 0), energy(0, 1)
            mx, my = energy(-1, 0), energy(0, -1)
            mixed = (energy(1, 1)-energy(-1, 1)-energy(1, -1)+energy(-1, -1))/4
            H = np.array([[dx+mx-2*center, mixed], [mixed, dy+my-2*center]]) / h**2
            gradient = np.array([dx-mx, dy-my]) / (2*h)
            return np.linalg.norm(np.linalg.solve(H, gradient))

        assert step_norm(seeds[-1]) > 0.1
        assert step_norm(result[-1]) < 0.1
        assert np.linalg.norm(result[-1]-knots[-1]) < np.linalg.norm(seeds[-1]-knots[-1])
    finally:
        capi.space_tree_destroy(tree)


def test_overlapping_patches_preserve_reference_winding_on_rough_surface():
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

        def orientation(points):
            a, b, c = points[faces].transpose(1, 0, 2)
            return np.einsum("ij,ij->i", a, np.cross(b-a, c-a))

        # Backtracking preserves the winding of each reference triangle on
        # the connectivity used for smoothing, before any final edge flips.
        assert np.all(orientation(seeds) * orientation(smoothed) > 0)
        assert np.linalg.norm(smoothed - seeds) > 1e-6
    finally:
        capi.space_tree_destroy(tree)
