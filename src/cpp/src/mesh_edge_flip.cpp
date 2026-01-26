#include "mesh_edge_flip.h"
#include "mesh_utils.h"
#include <limits>
#include <cmath>
#include <iostream>

namespace cpgeo {

// Helper structure to represent an edge with its adjacent triangles
struct EdgeInfo {
    int tri0 = -1;  // First adjacent triangle index (-1 if boundary)
    int tri1 = -1;  // Second adjacent triangle index (-1 if boundary)
    int opp0 = -1;  // Index (0-2) of the vertex opposite to this edge in tri0
    int opp1 = -1;  // Index (0-2) of the vertex opposite to this edge in tri1
};

// Compute the minimum angle in a triangle (in radians)
inline double compute_min_angle(
    const double* v0, const double* v1, const double* v2, int dim)
{
    // Compute edge vectors
    std::array<double, 3> e01{}, e12{}, e20{};
    for (int d = 0; d < dim; ++d) {
        e01[d] = v1[d] - v0[d];
        e12[d] = v2[d] - v1[d];
        e20[d] = v0[d] - v2[d];
    }
    
    // Compute edge lengths
    double len01 = 0.0, len12 = 0.0, len20 = 0.0;
    for (int d = 0; d < dim; ++d) {
        len01 += e01[d] * e01[d];
        len12 += e12[d] * e12[d];
        len20 += e20[d] * e20[d];
    }
    len01 = std::sqrt(len01);
    len12 = std::sqrt(len12);
    len20 = std::sqrt(len20);
    
    if (len01 < 1e-12 || len12 < 1e-12 || len20 < 1e-12) {
        return 0.0;  // Degenerate triangle
    }
    
    // Compute angles using dot products
    double dot0 = 0.0, dot1 = 0.0, dot2 = 0.0;
    for (int d = 0; d < dim; ++d) {
        dot0 += e01[d] * (-e20[d]);
        dot1 += e12[d] * (-e01[d]);
        dot2 += e20[d] * (-e12[d]);
    }
    
    double cos0 = dot0 / (len01 * len20);
    double cos1 = dot1 / (len12 * len01);
    double cos2 = dot2 / (len20 * len12);
    
    // Clamp to [-1, 1] to avoid numerical errors
    cos0 = std::clamp(cos0, -1.0, 1.0);
    cos1 = std::clamp(cos1, -1.0, 1.0);
    cos2 = std::clamp(cos2, -1.0, 1.0);
    
    double angle0 = std::acos(cos0);
    double angle1 = std::acos(cos1);
    double angle2 = std::acos(cos2);
    
    return std::min({angle0, angle1, angle2});
}

// Compute normal vector of a triangle
inline std::array<double, 3> compute_normal(
    const double* v0, const double* v1, const double* v2, int dim)
{
    std::array<double, 3> e01{}, e02{};
    for (int d = 0; d < std::min(dim, 3); ++d) {
        e01[d] = v1[d] - v0[d];
        e02[d] = v2[d] - v0[d];
    }
    
    // Cross product: e01 x e02
    std::array<double, 3> normal{
        e01[1] * e02[2] - e01[2] * e02[1],
        e01[2] * e02[0] - e01[0] * e02[2],
        e01[0] * e02[1] - e01[1] * e02[0]
    };
    
    return normal;
}

// Check if flipping an edge would improve mesh quality
// Returns true if flip is beneficial, false otherwise
inline bool should_flip_edge(
    const std::vector<int>& faces,
    const EdgeInfo& edge_info,
    std::span<const double> vertices,
    int vertices_dim)
{
    if (edge_info.tri0 == -1 || edge_info.tri1 == -1) {
        return false;  // Boundary edge, don't flip
    }
    
    const int* tri0 = &faces[edge_info.tri0 * 3];
    const int* tri1 = &faces[edge_info.tri1 * 3];
    
    // Get the four vertices involved
    int v_opp0 = tri0[edge_info.opp0];
    int v_opp1 = tri1[edge_info.opp1];
    int v_shared0 = tri0[(edge_info.opp0 + 1) % 3];
    int v_shared1 = tri0[(edge_info.opp0 + 2) % 3];
    
    const double* p_opp0 = &vertices[v_opp0 * vertices_dim];
    const double* p_opp1 = &vertices[v_opp1 * vertices_dim];
    const double* p_shared0 = &vertices[v_shared0 * vertices_dim];
    const double* p_shared1 = &vertices[v_shared1 * vertices_dim];
    
    // Compute normals before flip
    auto normal_before_0 = compute_normal(p_opp0, p_shared0, p_shared1, vertices_dim);
    auto normal_before_1 = compute_normal(p_opp1, p_shared1, p_shared0, vertices_dim);
    
    // Compute normals after flip (new edge: v_opp0 <-> v_opp1)
    auto normal_after_0 = compute_normal(p_opp0, p_opp1, p_shared0, vertices_dim);
    auto normal_after_1 = compute_normal(p_opp1, p_opp0, p_shared1, vertices_dim);
    
    // Check if normals would flip direction (dot product becomes negative)
    double dot0 = 0.0, dot1 = 0.0;
    for (int d = 0; d < 3; ++d) {
        dot0 += normal_before_0[d] * normal_after_0[d];
        dot1 += normal_before_1[d] * normal_after_1[d];
    }
    
    // Reject flip if it would reverse normal direction
    if (dot0 < 0.0 || dot1 < 0.0) {
        return false;
    }
    
    // Compute minimum angles before flip
    double min_angle_before_0 = compute_min_angle(p_opp0, p_shared0, p_shared1, vertices_dim);
    double min_angle_before_1 = compute_min_angle(p_opp1, p_shared1, p_shared0, vertices_dim);
    double min_angle_before = std::min(min_angle_before_0, min_angle_before_1);
    
    // Compute minimum angles after flip (new edge: v_opp0 <-> v_opp1)
    double min_angle_after_0 = compute_min_angle(p_opp0, p_opp1, p_shared0, vertices_dim);
    double min_angle_after_1 = compute_min_angle(p_opp1, p_opp0, p_shared1, vertices_dim);
    double min_angle_after = std::min(min_angle_after_0, min_angle_after_1);
    
    // Flip if it improves the minimum angle (with small threshold to avoid oscillation)
    const double threshold = 1e-6;
    return (min_angle_after > min_angle_before + threshold);
}

// Perform the edge flip operation
inline void flip_edge(
    std::vector<int>& faces,
    std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash>& edge_map,
    const std::pair<int, int>& edge,
    int vertices_dim)
{
    auto it = edge_map.find(edge);
    if (it == edge_map.end()) {
        return;  // Edge not found
    }
    
    EdgeInfo& info = it->second;
    if (info.tri0 == -1 || info.tri1 == -1) {
        return;  // Boundary edge
    }
    
    int* tri0 = &faces[info.tri0 * 3];
    int* tri1 = &faces[info.tri1 * 3];
    
    // Get vertices
    int v_opp0 = tri0[info.opp0];
    int v_opp1 = tri1[info.opp1];
    int v_shared0 = tri0[(info.opp0 + 1) % 3];
    int v_shared1 = tri0[(info.opp0 + 2) % 3];
    
    // Store old edges to remove from map
    std::vector<std::pair<int, int>> old_edges;
    for (int i = 0; i < 3; ++i) {
        int va = tri0[i];
        int vb = tri0[(i + 1) % 3];
        old_edges.push_back({std::min(va, vb), std::max(va, vb)});
    }
    for (int i = 0; i < 3; ++i) {
        int va = tri1[i];
        int vb = tri1[(i + 1) % 3];
        auto e = std::make_pair(std::min(va, vb), std::max(va, vb));
        if (std::find(old_edges.begin(), old_edges.end(), e) == old_edges.end()) {
            old_edges.push_back(e);
        }
    }
    
    // Update triangles (flip the edge)
    // New tri0: v_opp0, v_opp1, v_shared0
    // New tri1: v_opp1, v_opp0, v_shared1
    tri0[0] = v_opp0;
    tri0[1] = v_opp1;
    tri0[2] = v_shared0;
    
    tri1[0] = v_opp1;
    tri1[1] = v_opp0;
    tri1[2] = v_shared1;
    
    // Remove old edges from map
    for (const auto& e : old_edges) {
        edge_map.erase(e);
    }
    
    // Rebuild edge map for affected triangles
    // We need to be careful: some edges might be shared with other triangles
    for (int tri_idx : {info.tri0, info.tri1}) {
        const int* tri = &faces[tri_idx * 3];
        for (int i = 0; i < 3; ++i) {
            int va = tri[i];
            int vb = tri[(i + 1) % 3];
            auto e = std::make_pair(std::min(va, vb), std::max(va, vb));
            
            // Check if this edge already exists in the map (from another triangle)
            auto it_edge = edge_map.find(e);
            if (it_edge == edge_map.end()) {
                // New edge, create entry
                EdgeInfo new_info;
                new_info.tri0 = tri_idx;
                new_info.opp0 = (i + 2) % 3;
                edge_map[e] = new_info;
            } else {
                // Edge exists, add this triangle as the second adjacent
                if (it_edge->second.tri0 == -1) {
                    it_edge->second.tri0 = tri_idx;
                    it_edge->second.opp0 = (i + 2) % 3;
                } else if (it_edge->second.tri1 == -1) {
                    it_edge->second.tri1 = tri_idx;
                    it_edge->second.opp1 = (i + 2) % 3;
                }
                // If both tri0 and tri1 are already set, this is a non-manifold situation
                // which shouldn't happen in a valid manifold mesh
            }
        }
    }
}

// Build edge-to-triangle adjacency map
inline std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash> build_edge_map(
    const std::vector<int>& faces)
{
    std::unordered_map<std::pair<int, int>, EdgeInfo, EdgeHash> edge_map;
    int num_triangles = static_cast<int>(faces.size() / 3);
    
    for (int tri_idx = 0; tri_idx < num_triangles; ++tri_idx) {
        const int* tri = &faces[tri_idx * 3];
        for (int i = 0; i < 3; ++i) {
            int va = tri[i];
            int vb = tri[(i + 1) % 3];
            auto edge = std::make_pair(std::min(va, vb), std::max(va, vb));
            
            auto& info = edge_map[edge];
            if (info.tri0 == -1) {
                info.tri0 = tri_idx;
                info.opp0 = (i + 2) % 3;  // Opposite vertex index
            } else if (info.tri1 == -1) {
                info.tri1 = tri_idx;
                info.opp1 = (i + 2) % 3;
            } else {
                // Non-manifold edge (appears in more than 2 triangles)
                // Skip this edge - don't print warnings as they may be false positives
                // during edge flipping operations
            }
        }
    }
    
    return edge_map;
}

std::vector<int> mesh_optimize_by_edge_flipping(
    std::span<const double> vertices,
    int vertices_dim,
    std::span<const int> faces_in,
    int max_iterations)
{
    // Copy input faces to working buffer
    std::vector<int> faces(faces_in.begin(), faces_in.end());
    
    // Build initial edge map
    auto edge_map = build_edge_map(faces);
    
    int iteration = 0;
    int total_flips = 0;
    
    // Main optimization loop: repeat until no more flips are possible
    while (iteration < max_iterations) {
        // Rebuild candidate edges from current edge map
        std::vector<std::pair<int, int>> candidate_edges;
        candidate_edges.reserve(edge_map.size());
        for (const auto& [edge, info] : edge_map) {
            if (info.tri0 != -1 && info.tri1 != -1) {
                candidate_edges.push_back(edge);
            }
        }
        
        int flips_this_iteration = 0;
        
        // Try to flip each candidate edge
        for (const auto& edge : candidate_edges) {
            if (should_flip_edge(faces, edge_map[edge], vertices, vertices_dim)) {
                flip_edge(faces, edge_map, edge, vertices_dim);
                ++flips_this_iteration;
            }
        }
        
        total_flips += flips_this_iteration;
        ++iteration;
        
        // If no flips happened, we've converged - stop iteration
        if (flips_this_iteration == 0) {
            break;
        }
    }
    
    return faces;
}

} // namespace cpgeo
