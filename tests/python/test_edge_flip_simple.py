"""
Test mesh edge flipping with a simple synthetic mesh
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import sys
sys.path.insert(0, 'src/python')

from cpgeo.capi import optimize_mesh_by_edge_flipping


def create_test_mesh():
    """Create a simple mesh with suboptimal triangulation"""
    # Create a stretched quad where one diagonal creates worse triangles
    #
    #        1
    #       /|\
    #      / | \
    #     /  |  \
    #    0   |   2
    #     \  |  /
    #      \ | /
    #       \|/
    #        3
    #
    # Initial diagonal: 0-2 (creates two very obtuse triangles)
    # Better diagonal: 1-3 (creates two acute triangles)
    
    vertices = np.array([
        [-2.0, 0.0, 0.0],  # 0: left
        [0.0, 3.0, 0.0],   # 1: top (tall)
        [2.0, 0.0, 0.0],   # 2: right
        [0.0, -1.0, 0.0],  # 3: bottom (short)
    ], dtype=np.float64)
    
    # Poor triangulation: horizontal diagonal 0-2 creates very obtuse angles
    faces = np.array([
        [0, 1, 2],  # Triangle 1: top, contains angle at 1
        [0, 2, 3],  # Triangle 2: bottom, contains angle at 3
    ], dtype=np.int32)
    
    return vertices, faces


def compute_angles(vertices, faces):
    """Compute all angles in all triangles"""
    angles = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        e01 = v1 - v0
        e12 = v2 - v1
        e20 = v0 - v2
        
        len01 = np.linalg.norm(e01)
        len12 = np.linalg.norm(e12)
        len20 = np.linalg.norm(e20)
        
        if len01 < 1e-12 or len12 < 1e-12 or len20 < 1e-12:
            continue
        
        cos0 = np.dot(e01, -e20) / (len01 * len20)
        cos1 = np.dot(e12, -e01) / (len12 * len01)
        cos2 = np.dot(e20, -e12) / (len20 * len12)
        
        cos0 = np.clip(cos0, -1, 1)
        cos1 = np.clip(cos1, -1, 1)
        cos2 = np.clip(cos2, -1, 1)
        
        angles.extend([np.arccos(cos0), np.arccos(cos1), np.arccos(cos2)])
    
    return np.array(angles)


def test_simple_mesh():
    """Test edge flipping on a simple 2-triangle mesh"""
    print("=" * 70)
    print("Testing Edge Flipping on Simple Synthetic Mesh")
    print("=" * 70)
    
    vertices, faces = create_test_mesh()
    
    print(f"\nMesh Statistics:")
    print(f"  Number of vertices: {vertices.shape[0]}")
    print(f"  Number of faces: {faces.shape[0]}")
    
    # Compute angles before
    angles_before = compute_angles(vertices, faces)
    max_angle_before = np.max(angles_before)
    min_angle_before = np.min(angles_before)
    
    print("\n" + "-" * 70)
    print("BEFORE Optimization:")
    print("-" * 70)
    print(f"All angles (degrees): {np.degrees(sorted(angles_before))}")
    print(f"Min angle: {np.degrees(min_angle_before):.2f}°")
    print(f"Max angle: {np.degrees(max_angle_before):.2f}°")
    print(f"Faces: {faces.tolist()}")
    
    # Perform optimization
    print("\n" + "-" * 70)
    print("Running Edge Flipping Optimization...")
    print("-" * 70)
    
    optimized_faces = optimize_mesh_by_edge_flipping(
        vertices, 
        faces, 
        max_iterations=10
    )
    
    # Compute angles after
    angles_after = compute_angles(vertices, optimized_faces)
    max_angle_after = np.max(angles_after)
    min_angle_after = np.min(angles_after)
    
    print("\n" + "-" * 70)
    print("AFTER Optimization:")
    print("-" * 70)
    print(f"All angles (degrees): {np.degrees(sorted(angles_after))}")
    print(f"Min angle: {np.degrees(min_angle_after):.2f}°")
    print(f"Max angle: {np.degrees(max_angle_after):.2f}°")
    print(f"Faces: {optimized_faces.tolist()}")
    
    # Summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY:")
    print("=" * 70)
    print(f"Min angle change: {np.degrees(min_angle_after - min_angle_before):+.2f}°")
    print(f"Max angle change: {np.degrees(max_angle_after - max_angle_before):+.2f}°")
    
    if max_angle_after < max_angle_before:
        print(f"✅ Max angle reduced by {np.degrees(max_angle_before - max_angle_after):.2f}°")
    elif np.allclose(max_angle_after, max_angle_before):
        print(f"➖ No change in max angle (mesh may already be optimal)")
    else:
        print(f"⚠️  Max angle increased by {np.degrees(max_angle_after - max_angle_before):.2f}°")


if __name__ == "__main__":
    test_simple_mesh()
    print("\n" + "=" * 70)
    print("✅ Test completed!")
    print("=" * 70)
