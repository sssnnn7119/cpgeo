#pragma once

#include <unordered_map>
#include <vector>
#include <span>

namespace cpgeo {

// Hash function for an edge represented as a pair of vertex indices
struct EdgeHash {
    std::size_t operator()(const std::pair<int, int>& edge) const noexcept {
        return std::hash<int>{}(edge.first) ^ (std::hash<int>{}(edge.second) << 1);
    }
};

// Result structure for mesh partitioning
struct MeshPartition {
    std::vector<int> hemisphere1_faces;  // Triangle indices for first hemisphere (flattened: [t0v0, t0v1, t0v2, ...])
    std::vector<int> hemisphere2_faces;  // Triangle indices for second hemisphere
    std::vector<int> cut_vertices;       // Vertex indices on the cutting boundary
};

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