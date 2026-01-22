#pragma once
#include <vector>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <cmath>
#include <algorithm>

namespace cpgeo {

/**
 * @brief Optimize a triangle mesh by edge flipping to make triangles more equilateral
 * 
 * This function performs iterative edge flipping on a manifold triangle mesh to improve
 * triangle quality. The goal is to make each triangle as close to equilateral as possible
 * while maintaining mesh topology (each edge appears in exactly 2 triangles, no holes).
 * 
 * The algorithm uses the following criteria for edge flipping:
 * - An edge is flipped if it improves the minimum angle of the involved triangles
 * - Boundary edges (appearing in only 1 triangle) are never flipped
 * - The process iterates until no beneficial flips are found or max iterations reached
 * 
 * @param vertices Vertex coordinates (flattened: [x0, y0, z0, x1, y1, z1, ...])
 * @param vertices_dim Dimension of vertices (2 for 2D, 3 for 3D)
 * @param faces_in Input triangle faces (flattened: [t0v0, t0v1, t0v2, t1v0, t1v1, t1v2, ...])
 * @param max_iterations Maximum number of flip iterations (default: 100)
 * @return Optimized triangle faces with the same format as input
 */
std::vector<int> optimize_mesh_by_edge_flipping(
    std::span<const double> vertices,
    int vertices_dim,
    std::span<const int> faces_in,
    int max_iterations = 100
);

} // namespace cpgeo
