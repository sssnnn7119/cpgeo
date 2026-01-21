#pragma once
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <span>
#include <cmath>
#include <array>
#include <omp.h>

// edges method
namespace cpgeo {

// Hash function for an edge represented as a pair of vertex indices
// 使用int64拼接两个int32，更鲁棒且无冲突（假设顶点索引<2^31）
struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const noexcept {
        // 将两个int32拼接成int64作为hash值
        uint64_t combined = (static_cast<uint64_t>(static_cast<uint32_t>(edge.first)) << 32) | 
                           static_cast<uint64_t>(static_cast<uint32_t>(edge.second));
        return std::hash<uint64_t>{}(combined);
    }
};

/**
 * @brief Extract edges from a triangle mesh and count their occurrences
 * 
 * @param triangles Input triangle mesh (flattened: [t0v0, t0v1, t0v2, t1v0, ...])
 * @return A map from edges (as pairs of vertex indices) to their occurrence counts
 */
std::unordered_map<std::pair<int, int>, int, EdgeHash> extractEdgesWithNumber(const std::span<const int>& triangles);

/**
 * @brief Extract all boundary loops from a triangular mesh
 * 
 * A boundary edge is an edge that belongs to only one triangle.
 * This function finds all closed boundary loops and returns them as ordered vertex sequences.
 * 
 * @param triangles Input triangle mesh (flattened: [t0v0, t0v1, t0v2, t1v0, ...])
 * @return Vector of boundary loops, where each loop is a vector of vertex indices forming a closed path
 */
std::vector<std::vector<int>> extractBoundaryLoops(const std::span<const int>& triangles);


}

// mesh fineturing method
namespace cpgeo {

// 四个整数数组的hash - 使用类似boost::hash_combine的混合方法
struct Array4IntHash {
    std::size_t operator()(const std::array<int, 4>& arr) const noexcept {
        std::size_t h0 = std::hash<int>{}(std::get<0>(arr));
        std::size_t h1 = std::hash<int>{}(std::get<1>(arr));
        std::size_t h2 = std::hash<int>{}(std::get<2>(arr));
        std::size_t h3 = std::hash<int>{}(std::get<3>(arr));
        
        std::size_t seed = 0;
        seed ^= h0 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};


/**
 * @brief Compute the closure edge length loss
 * 
 * @param vertices Vertex positions (flattened: [v0x, v0y, v1x, v1y, ...])
 * @param vertices_dim Dimension of each vertex (e.g., 2 for 2D, 3 for 3D)
 * @param edges Edges to consider (flattened: [e0v0, e0v1, e1v0, e1v1, ...])
 * @param order Order of the length penalty (e.g., 2 for squared length)
 * @return The computed loss value
 */
double closure_edge_length_derivative0(std::span<const double> vertices, int vertices_dim, std::span<const int> edges, int order);

/**
 * @brief Compute the closure edge length loss and its first and second derivatives
 * 
 * @param vertices Vertex positions (flattened: [v0x, v0y, v1x, v1y, ...])
 * @param vertices_dim Dimension of each vertex (e.g., 2 for 2D, 3 for 3D)
 * @param edges Edges to consider (flattened: [e0v0, e0v1, e1v0, e1v1, ...])
 * @param order Order of the length penalty (e.g., 2 for squared length)
 * @return A tuple containing:
 *         - loss value (double)
 *         - first derivative vector (flattened, same size as vertices)
 *         - second derivative in COO format: indices (flattened: [v0_idx, v0_dim, v1_idx, v1_dim, ...])
 *         - second derivative in COO format: values (corresponding values)
 */
std::tuple<double, std::vector<double>, std::vector<int>, std::vector<double>> closure_edge_length_derivative2(std::span<const double> vertices, int vertices_dim, std::span<const int> edges, int order);
}
