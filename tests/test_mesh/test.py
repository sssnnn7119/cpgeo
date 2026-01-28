"""
Test mesh partition and boundary extraction functionality.
"""

import numpy as np
import sys
from collections import defaultdict

sys.path.insert(0, 'src/python')

import cpgeo
import os


def triangles_to_edges(tris):
    edges = []
    for a, b, c in tris:
        edges.append((a, b))
        edges.append((b, c))
        edges.append((c, a))
    return edges


def check_edge_topology(tris, verbose=True):
    edge_count = defaultdict(int)
    for a, b in triangles_to_edges(tris):
        if a > b:
            a, b = b, a
        edge_count[(a, b)] += 1

    boundary = [e for e, c in edge_count.items() if c == 1]
    invalid = [e for e, c in edge_count.items() if c > 2]

    if verbose:
        print("Edge topology check:")
        print("  Total edges:", len(edge_count))
        print("  Boundary edges:", len(boundary))
        print("  Invalid edges:", len(invalid))

    return len(boundary) == 0 and len(invalid) == 0


def check_directed_edge_usage(tris, verbose=True):
    edge_count = defaultdict(int)
    for a, b in triangles_to_edges(tris):
        edge_count[(a, b)] += 1

    bad = [e for e, c in edge_count.items() if c != 1]
    if verbose:
        print("Directed edge usage check:")
        print("  Total directed edges:", len(edge_count))
        print("  Bad directed edges:", len(bad))
        if bad:
            print("  Example bad edge:", bad[0], "count=", edge_count[bad[0]])

    return len(bad) == 0


def check_duplicate_triangles(tris, verbose=True):
    tri_count = defaultdict(int)
    for a, b, c in tris:
        key = tuple(sorted((a, b, c)))
        tri_count[key] += 1
    dup = [k for k, v in tri_count.items() if v > 1]
    if verbose:
        print("Duplicate triangle check:")
        print("  Duplicate triangles:", len(dup))
        if dup:
            print("  Example duplicate:", dup[0], "count=", tri_count[dup[0]])
    return len(dup) == 0


def check_euler_characteristic(tris, num_vertices, verbose=True):
    edge_count = defaultdict(int)
    for a, b in triangles_to_edges(tris):
        if a > b:
            a, b = b, a
        edge_count[(a, b)] += 1
    v = num_vertices
    e = len(edge_count)
    f = len(tris)
    chi = v - e + f
    if verbose:
        print("Euler characteristic check:")
        print("  V=", v, "E=", e, "F=", f, "=> chi=", chi)
    return chi == 2


def check_mesh_quality(tris, verts, min_angle_deg=0.1, verbose=True):
    def angle_at(a, b, c):
        ab = b - a
        ac = c - a
        lab = np.linalg.norm(ab)
        lac = np.linalg.norm(ac)
        if lab < 1e-12 or lac < 1e-12:
            return 0.0
        cosv = np.dot(ab, ac) / (lab * lac)
        cosv = np.clip(cosv, -1.0, 1.0)
        return np.degrees(np.arccos(cosv))

    min_angle = 1e9
    for a, b, c in tris:
        v0 = verts[a]
        v1 = verts[b]
        v2 = verts[c]
        a0 = angle_at(v0, v1, v2)
        a1 = angle_at(v1, v2, v0)
        a2 = angle_at(v2, v0, v1)
        min_angle = min(min_angle, a0, a1, a2)

    if verbose:
        print("Triangle quality check:")
        print("  Min angle:", min_angle, "deg")

    return min_angle >= min_angle_deg


if __name__ == "__main__":
    
    # data = np.load('tests/testdata.npz')

    # cps = data['control_points'].T
    # faces = data['mesh_elements']
    # surf = cpgeo.CPGEO(control_points=cps, cp_faces=faces)
    # surf.initialize()

    grids = np.loadtxt('tests/test_mesh/data/data1.txt').reshape((-1, 3))
    cps = np.loadtxt('tests/test_mesh/data/data1_cp.txt')
    mesh = cpgeo.capi.get_sphere_triangulation(grids)

    mesh2 = cpgeo.capi.optimize_mesh_by_edge_flipping(cps, mesh)

    # import pyvista as pv
    # mesh_pv = pv.PolyData(grids, np.hstack([np.full((mesh.shape[0], 1), 3), mesh]))
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh_pv, color='lightblue', show_edges=True, opacity=1.0)
    # plotter.show()

    # mesh_pv2 = pv.PolyData(cps, np.hstack([np.full((mesh2.shape[0], 1), 3), mesh2]))
    # plotter2 = pv.Plotter()
    # plotter2.add_mesh(mesh_pv2, color='lightgreen', show_edges=True, opacity=1.0)
    # plotter2.show()

    # Quality checks for optimized mesh on cps
    mesh2 = np.asarray(mesh2, dtype=int)
    cps = np.asarray(cps, dtype=float)

    topology_ok = check_edge_topology(mesh2, verbose=True)
    directed_ok = check_directed_edge_usage(mesh2, verbose=True)
    duplicate_ok = check_duplicate_triangles(mesh2, verbose=True)
    euler_ok = check_euler_characteristic(mesh2, cps.shape[0], verbose=True)
    quality_ok = check_mesh_quality(mesh2, cps, min_angle_deg=0.1, verbose=True)

    assert topology_ok, "Topology check failed"
    assert directed_ok, "Directed edge usage check failed"
    assert duplicate_ok, "Duplicate triangle check failed"
    assert euler_ok, "Euler characteristic check failed"
    assert quality_ok, "Triangle quality check failed"
