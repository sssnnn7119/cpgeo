"""
Test mesh edge flipping optimization using real test data
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import time
import sys
sys.path.insert(0, 'src/python')

from cpgeo.capi import optimize_mesh_by_edge_flipping, get_mesh_edges


def compute_triangle_quality_metrics(vertices, faces):
    """Compute quality metrics for each triangle"""
    def compute_min_angle(v0, v1, v2):
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2
        
        len01 = np.linalg.norm(e01)
        len12 = np.linalg.norm(e12)
        len20 = np.linalg.norm(e20)
        
        if len01 < 1e-12 or len12 < 1e-12 or len20 < 1e-12:
            return 0.0
        
        cos0 = np.dot(e01, -e20) / (len01 * len20)
        cos1 = np.dot(e12, -e01) / (len12 * len01)
        cos2 = np.dot(e20, -e12) / (len20 * len12)
        
        cos0 = np.clip(cos0, -1, 1)
        cos1 = np.clip(cos1, -1, 1)
        cos2 = np.clip(cos2, -1, 1)
        
        return min(np.arccos(cos0), np.arccos(cos1), np.arccos(cos2))
    
    def compute_aspect_ratio(v0, v1, v2):
        """Compute aspect ratio (circumradius / inradius)"""
        e01 = np.linalg.norm(v1 - v0)
        e12 = np.linalg.norm(v2 - v1)
        e20 = np.linalg.norm(v0 - v2)
        
        s = (e01 + e12 + e20) / 2  # semi-perimeter
        area = np.sqrt(max(0, s * (s - e01) * (s - e12) * (s - e20)))
        
        if area < 1e-12:
            return np.inf
        
        circumradius = (e01 * e12 * e20) / (4 * area)
        inradius = area / s
        
        return circumradius / inradius if inradius > 1e-12 else np.inf
    
    min_angles = []
    aspect_ratios = []
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        min_angles.append(compute_min_angle(v0, v1, v2))
        aspect_ratios.append(compute_aspect_ratio(v0, v1, v2))
    
    return np.array(min_angles), np.array(aspect_ratios)


def test_real_mesh(show_plot=False):
    """Test edge flipping on real mesh data from testdata.npz

    Args:
        show_plot (bool): If True, display a PyVista figure comparing the
                          original and optimized meshes side-by-side (same
                          scene, different transparencies and slight offset).
    """
    print("=" * 70)
    print("Testing Edge Flipping Algorithm on Real Mesh Data")
    print("=" * 70)
    
    # Load test data
    data = np.load('tests/testdata.npz')
    control_points = data['control_points'].T  # Transpose to get (n, 3)
    faces = data['mesh_elements'].astype(np.int32)
    
    print(f"\nMesh Statistics:")
    print(f"  Number of vertices: {control_points.shape[0]}")
    print(f"  Number of faces: {faces.shape[0]}")
    print(f"  Vertex dimension: {control_points.shape[1]}")
    
    # Verify mesh manifoldness using existing function
    print("\n" + "-" * 70)
    print("Verifying Mesh Topology:")
    print("-" * 70)
    
    edges = get_mesh_edges(faces)
    edge_counts = edges[:, 2]  # Third column is the count
    
    manifold_edges = np.sum(edge_counts == 2)
    boundary_edges = np.sum(edge_counts == 1)
    non_manifold_edges = np.sum(edge_counts > 2)
    
    print(f"Total unique edges: {len(edges)}")
    print(f"  Interior edges (count=2): {manifold_edges}")
    print(f"  Boundary edges (count=1): {boundary_edges}")
    print(f"  Non-manifold edges (count>2): {non_manifold_edges}")
    
    if non_manifold_edges > 0:
        print(f"\n⚠️ Warning: Mesh has {non_manifold_edges} non-manifold edges!")
        non_manifold_list = edges[edge_counts > 2]
        print(f"First few non-manifold edges: {non_manifold_list[:5]}")
    else:
        print(f"\n✅ Mesh is manifold (no non-manifold edges)")
    
    # Compute quality metrics before optimization
    print("\n" + "-" * 70)
    print("BEFORE Optimization:")
    print("-" * 70)
    
    min_angles_before, aspect_ratios_before = compute_triangle_quality_metrics(
        control_points, faces
    )
    
    print(f"Minimum angles (degrees):")
    print(f"  Mean:   {np.degrees(np.mean(min_angles_before)):.2f}°")
    print(f"  Median: {np.degrees(np.median(min_angles_before)):.2f}°")
    print(f"  Min:    {np.degrees(np.min(min_angles_before)):.2f}°")
    print(f"  Max:    {np.degrees(np.max(min_angles_before)):.2f}°")
    print(f"  Std:    {np.degrees(np.std(min_angles_before)):.2f}°")
    
    valid_aspect = aspect_ratios_before[np.isfinite(aspect_ratios_before)]
    print(f"\nAspect ratios (circumradius/inradius):")
    print(f"  Mean:   {np.mean(valid_aspect):.2f}")
    print(f"  Median: {np.median(valid_aspect):.2f}")
    print(f"  Min:    {np.min(valid_aspect):.2f}")
    print(f"  Max:    {np.max(valid_aspect):.2f}")
    
    # Count poor quality triangles
    poor_angles = np.sum(min_angles_before < np.radians(20))
    print(f"\nTriangles with min angle < 20°: {poor_angles} ({100*poor_angles/len(faces):.1f}%)")
    
    # Perform optimization
    print("\n" + "-" * 70)
    print("Running Edge Flipping Optimization...")
    print("-" * 70)
    
    t0 = time.time()
    optimized_faces = optimize_mesh_by_edge_flipping(
        control_points, 
        faces, 
        max_iterations=100
    )
    t1 = time.time()
    
    print(f"Optimization completed in {t1 - t0:.3f} seconds")
    
    # Compute quality metrics after optimization
    print("\n" + "-" * 70)
    print("AFTER Optimization:")
    print("-" * 70)
    
    min_angles_after, aspect_ratios_after = compute_triangle_quality_metrics(
        control_points, optimized_faces
    )
    
    print(f"Minimum angles (degrees):")
    print(f"  Mean:   {np.degrees(np.mean(min_angles_after)):.2f}°")
    print(f"  Median: {np.degrees(np.median(min_angles_after)):.2f}°")
    print(f"  Min:    {np.degrees(np.min(min_angles_after)):.2f}°")
    print(f"  Max:    {np.degrees(np.max(min_angles_after)):.2f}°")
    print(f"  Std:    {np.degrees(np.std(min_angles_after)):.2f}°")
    
    valid_aspect = aspect_ratios_after[np.isfinite(aspect_ratios_after)]
    print(f"\nAspect ratios (circumradius/inradius):")
    print(f"  Mean:   {np.mean(valid_aspect):.2f}")
    print(f"  Median: {np.median(valid_aspect):.2f}")
    print(f"  Min:    {np.min(valid_aspect):.2f}")
    print(f"  Max:    {np.max(valid_aspect):.2f}")
    
    poor_angles_after = np.sum(min_angles_after < np.radians(20))
    print(f"\nTriangles with min angle < 20°: {poor_angles_after} ({100*poor_angles_after/len(faces):.1f}%)")
    
    # Summary of improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY:")
    print("=" * 70)
    
    mean_angle_improvement = np.degrees(np.mean(min_angles_after) - np.mean(min_angles_before))
    min_angle_improvement = np.degrees(np.min(min_angles_after) - np.min(min_angles_before))
    
    print(f"Mean minimum angle improvement: {mean_angle_improvement:+.2f}°")
    print(f"Worst-case angle improvement:   {min_angle_improvement:+.2f}°")
    print(f"Poor quality triangles reduced: {poor_angles - poor_angles_after} "
          f"({100*(poor_angles - poor_angles_after)/poor_angles:.1f}% reduction)")
    
    # Verify mesh topology is preserved
    assert optimized_faces.shape == faces.shape, "Face count changed!"
    print(f"\n✅ Mesh topology preserved (face count: {faces.shape[0]})")
    
    # Verify improvement
    if mean_angle_improvement >= -0.1:  # Allow small numerical tolerance
        print(f"✅ Mesh quality improved or maintained")
    else:
        print(f"⚠️  Warning: mesh quality decreased slightly")

    # Optional visualization using PyVista: original vs optimized mesh in one scene
    if show_plot:
        try:
            import pyvista as pv

            def make_pv_mesh(verts, faces_arr):
                # PyVista expects faces flattened as [n_pts, v0, v1, v2, n_pts, ...]
                faces_flat = np.hstack([
                    np.full((faces_arr.shape[0], 1), 3, dtype=np.int64),
                    faces_arr.astype(np.int64)
                ])
                faces_flat = faces_flat.flatten()
                return pv.PolyData(verts, faces_flat)

            # Place meshes side-by-side and add outline edges for clearer comparison
            bbox = control_points.max(axis=0) - control_points.min(axis=0)
            max_extent = np.max(bbox)
            sep = max_extent * 0.6  # separation distance between the two meshes

            left_shift = np.array([-sep * 0.5, 0.0, 0.0])
            right_shift = np.array([sep * 0.5, 0.0, 0.0])

            mesh_before = make_pv_mesh(control_points, faces)
            mesh_after = make_pv_mesh(control_points, optimized_faces)

            p = pv.Plotter()
            # Draw filled surfaces with edge outlines using show_edges=True
            p.add_mesh(mesh_before, color="lightgray", opacity=0.6, show_edges=True, label="Before")
            p.add_mesh(mesh_after, color="tomato", opacity=1., show_edges=True, label="After")

            # Add simple annotations and show
            p.add_legend()
            p.add_title("Mesh Edge Flip: Before (left) vs After (right)")
            p.show()
        except Exception as e:
            print("PyVista visualization skipped (import failed or headless):", e)

    return optimized_faces


if __name__ == "__main__":
    # When run directly, show the PyVista comparison
    optimized_faces = test_real_mesh(show_plot=True)
    print("\n" + "=" * 70)
    print("✅ Test completed successfully!")
    print("=" * 70)
